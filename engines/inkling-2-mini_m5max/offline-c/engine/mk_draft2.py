"""The 2DSD DSpark drafter for the variable-loop Inkling 2 Mini (minseo25/2dsd-inkling-2-mini-varloop-dspark-drafter),
in place of the checkpoint's own draft head (mk_draft.Draft): five decoder layers over a block of 8 rows (the anchor and
7 rows of a learned mask vector) attending, with full bidirectional attention, to a context made from five feature rows
of every committed token, and a low-rank Markov bias that conditions each draft on the one before it.

The feature rows are the inputs of layers 2, 7, 12, 17 and 22 in the target's last loop (Target.taps, set from the
drafter's config when MK_DRAFT2 names it): visits [2, 31, 36, 41, 46] at 3 passes, [2, 55, 60, 65, 70] at 5. A token's
five rows are concatenated, projected by fc to the hidden width and normalised; each layer projects that row to its keys
and values with the same k_proj / v_proj it uses for the block (k_norm, RoPE at the token's position).

This is the 1-D use of the drafter (drafts from verified last-loop features only). The cycle, the depth policy, copy
proposals and the fused verify-and-draft pass are mk_draft.Draft's.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import mlx.core as mx

import mk_kernels as kn
import mk_draft as md

F32 = mx.float32
NB = md.NB
rms = md.rms

# the anchor row: the target's embedding through its embedding norm ("norm") or as stored ("raw")
ANCHOR = os.environ.get("MK_D2_ANCHOR", "norm")
# the head's softcap before the Markov bias is added ("before") or after ("after")
CAP = os.environ.get("MK_D2_CAP", "after")
# greedy drafts: the Markov bias is computed for the base logits' top TOPC tokens of a row only (0: the whole vocabulary)
TOPC = int(os.environ.get("MK_D2_TOPC", "64"))


def last_loop_taps(tc: dict, layers) -> tuple:
    """Layer visits (0-based) of the inputs of the given weight layers in the last loop pass."""
    lc = tc["loop_config"]
    cs, ce, passes = lc["core_start"], lc["core_end"], lc["num_passes"]
    n = ce - cs
    out = []
    for i in layers:
        if i < cs:
            out.append(i)
        elif i < ce:
            out.append(cs + (passes - 1) * n + (i - cs))
        else:
            out.append(cs + passes * n + (i - ce))
    return tuple(out)


class Draft2(md.Draft):
    NL = 5

    def __init__(self, draft_dir: str, tg):
        d = Path(draft_dir)
        cfg = json.loads((d / "config.json").read_text())
        lt = cfg["loopedtf_drafter"]
        assert cfg["num_hidden_layers"] == self.NL and cfg["block_size"] == NB and cfg["hidden_size"] == 2304
        assert cfg["num_attention_heads"] == 24 and cfg["num_key_value_heads"] == 8 and cfg["head_dim"] == 128
        assert lt["dspark"]["markov_head_type"] == "vanilla" and lt["dspark"]["markov_rank"] == 256
        assert tuple(tg.taps) == last_loop_taps(tg.tc, lt["feature_layers_per_loop"]), (tg.taps, lt)
        self.tg = tg
        self.TD = min(tg.T, int(os.environ.get("MK_DRAFT_CTX", "4104")))
        self.ring = self.TD < tg.T
        self.T = self.TD
        self.low = 0
        self.vocab, self.cap = tg.vocab, float(lt["target_head"]["softcap"])
        self.theta = float(os.environ.get("MK_CUM_TH", "0.3"))
        self.kmin = int(os.environ.get("MK_KMIN", "1"))
        self.ng_min = int(os.environ.get("MK_NGRAM", "5"))
        self.ng_wide = os.environ.get("MK_WIDE", "1") == "1"
        self.ng_agree = os.environ.get("MK_NG_AGREE", "0") == "1"
        self.policy = os.environ.get("MK_POLICY", "cala")
        self.cal_thr = float(os.environ.get("MK_CAL_THR", "0.3"))
        self.thr_lo, self.thr_hi = float(os.environ.get("MK_THR_LO", "0.2")), float(os.environ.get("MK_THR_HI", "0.6"))
        self.row_ms = float(os.environ.get("MK_ROW_MS", "2.5"))
        self.ema = float(os.environ.get("MK_EMA", "0.9"))
        self.rate_dspark = False
        self.lazy_state = False
        self.chain = os.environ.get("MK_CHAIN", "1") == "1"
        self.n1 = int(os.environ.get("MK_DRAFT_N1", "3"))
        self.local, self.window = set(), 1 << 30          # full attention in every layer
        w = mx.load(str(d / "model.safetensors"))
        u16 = lambda a: mx.contiguous(a).view(mx.uint16)  # noqa: E731
        f32 = lambda k: w[k].astype(F32)  # noqa: E731
        self.q8 = os.environ.get("MK_DRAFT8", "1") == "1"
        self.q5 = False
        q = (lambda a: kn.quant8(a)) if self.q8 else (lambda a: u16(a))  # noqa: E731
        self.L = []
        for l in range(self.NL):
            p = f"layers.{l}."
            g = lambda k: w[p + k]  # noqa: E731
            self.L.append({
                "qkv": q(mx.concatenate([g("self_attn.q_proj.weight"), g("self_attn.k_proj.weight"), g("self_attn.v_proj.weight")])),
                "kv_ctx": u16(mx.concatenate([g("self_attn.k_proj.weight"), g("self_attn.v_proj.weight")])),
                "o": q(g("self_attn.o_proj.weight")),
                "gate_up": q(mx.concatenate([g("mlp.gate_proj.weight"), g("mlp.up_proj.weight")])),
                "down": q(g("mlp.down_proj.weight")),
                "in_ln": f32(p + "input_layernorm.weight"), "post_ln": f32(p + "post_attention_layernorm.weight"),
                "q_norm": f32(p + "self_attn.q_norm.weight"), "k_norm": f32(p + "self_attn.k_norm.weight"),
            })
        self.fc = w["fc.weight"]                                                                     # [2304, 5 * 2304] bf16
        self.hidden_norm = f32("hidden_norm.weight")
        self.norm = f32("norm.weight")
        if (d / "mask_embedding.npy").exists():        # the vector of mask_embedding.pt, saved without PyTorch's format
            import numpy as np
            self.mask_embed = mx.array(np.load(str(d / "mask_embedding.npy")).astype("float32"))
        else:
            import torch
            m = torch.load(str(d / "mask_embedding.pt"), map_location="cpu", weights_only=True)
            assert int(m["mask_token_id"]) == cfg["dflash_config"]["mask_token_id"]
            self.mask_embed = mx.array(m["embedding"].float().numpy())
        self.w1 = w["markov_head.markov_w1.weight"]                                                  # [V, 256] bf16
        w2 = w["markov_head.markov_w2.weight"]                                                       # [V, 256] bf16
        self.w2b = w2                                                                                # rows gathered by candidate
        self.w2 = kn.quant8(w2) if self.q8 else u16(w2)
        un = tg.unembed.view(mx.bfloat16)
        self.un8 = None
        if self.q8:
            self.un8 = tg.un8 if getattr(tg, "un8", None) is not None else kn.quant8(un[:self.vocab])
        self.inv_freq = 1.0 / (float(cfg["rope_theta"]) ** (mx.arange(0, 128, 2).astype(F32) / 128))
        keep = [self.fc, self.hidden_norm, self.norm, self.mask_embed, self.w1, self.w2, self.w2b, self.un8, self.inv_freq] + \
            [v for dd in self.L for v in dd.values()]
        mx.eval(keep)
        self.gm = kn.gemv8 if self.q8 else kn.gemv
        self.gmu = self.gm
        self.tpos = mx.arange(self.T).astype(F32)
        self.tposi = mx.arange(self.T).astype(mx.int32)
        self._cyc, self._sfn, self._fn = {}, {}, None
        del w
        self.reset()

    def reset(self):
        self.kc = [mx.zeros((self.T, 8, 128), dtype=F32) for _ in range(self.NL)]
        self.vc = [mx.zeros((self.T, 8, 128), dtype=F32) for _ in range(self.NL)]
        self.frow = None
        self.hist = []
        self.low = 0

    # ---------------------------------------------------------------------------------------------- context
    def _ctx(self, feats):
        """The first five feature arrays [n, H] -> the normalised context rows [n, H]."""
        x = mx.concatenate(list(feats[:5]), axis=-1).astype(mx.bfloat16)
        return rms((x @ self.fc.T).astype(F32), self.hidden_norm)

    def _ctx_kv(self, ctx, ps, l):
        w = self.L[l]
        n = ctx.shape[0]
        kv = kn.gemv(w["kv_ctx"], kn.bits(ctx))                                                     # [n, 2048]
        k = self._rope(rms(kv[:, :1024].reshape(n, 8, 128), w["k_norm"]), ps)
        return k, kv[:, 1024:].reshape(n, 8, 128)

    def append(self, feats, pos: int, n: int):
        ctx = self._ctx([x[:n] for x in feats])
        ps = mx.arange(pos, pos + n).astype(F32)
        for l in range(self.NL):
            k, v = self._ctx_kv(ctx, ps, l)
            if self.ring:
                idx = (pos + mx.arange(n)) % self.TD
                self.kc[l][idx] = k
                self.vc[l][idx] = v
            else:
                self.kc[l][pos:pos + n] = k
                self.vc[l][pos:pos + n] = v
        self.frow = [x[n - 1:n] for x in feats]

    def _append_pure(self, feats, cposf, kc, vc):
        K = feats[0].shape[0]
        if self.ring:
            cposf = cposf[:1]
            fwd = self.tpos - cposf
            rel = (fwd - self.TD * mx.floor(fwd * (1.0 / self.TD))).astype(mx.int32)
        else:
            rel = self.tposi - cposf.astype(mx.int32)
        ps = cposf + mx.arange(K).astype(F32)
        mask = ((rel >= 0) & (rel < K))[:, None, None]
        rows = mx.clip(rel, 0, K - 1)
        ctx = self._ctx(feats)
        kc2, vc2 = [], []
        for l in range(self.NL):
            k, v = self._ctx_kv(ctx, ps, l)
            kc2.append(mx.where(mask, k[rows], kc[l]))
            vc2.append(mx.where(mask, v[rows], vc[l]))
        return kc2, vc2

    # ---------------------------------------------------------------------------------------------- the block
    def _pure(self, anchor, cpos, kc, vc, frow, samp=None):
        tg = self.tg
        nd = NB - 1
        e = tg.embed[anchor].astype(F32)
        if ANCHOR == "norm":
            e = rms(e, tg.embed_norm)
        x = mx.concatenate([e, mx.broadcast_to(self.mask_embed[None], (nd, 2304))])
        if self.ring:
            cpos, low = cpos[:1], cpos[1:]
            back = cpos - 1.0 - self.tpos
            pj = cpos - 1.0 - (back - self.TD * mx.floor(back * (1.0 / self.TD)))
            vis = (pj >= mx.maximum(low, cpos - (self.TD - 16))) & (pj >= 0)
        else:
            vis = self.tpos < cpos
        pos = cpos + mx.arange(NB).astype(F32)
        mask = mx.concatenate([mx.where(vis, 0.0, -mx.inf), mx.zeros((NB,), dtype=F32)])
        for l in range(self.NL):
            w = self.L[l]
            h = rms(x, w["in_ln"])
            qkv = self.gm(w["qkv"], kn.bits(h))                                                     # [8, 5120]
            q = self._rope(rms(qkv[:, :3072].reshape(NB, 24, 128), w["q_norm"]), pos)
            k = self._rope(rms(qkv[:, 3072:4096].reshape(NB, 8, 128), w["k_norm"]), pos)
            v = qkv[:, 4096:].reshape(NB, 8, 128)
            keys = mx.concatenate([kc[l], k]).transpose(1, 2, 0)                                    # [8, 128, T + 8]
            vals = mx.concatenate([vc[l], v]).transpose(1, 0, 2)                                    # [8, T + 8, 128]
            qq = q.reshape(NB, 8, 3, 128).transpose(1, 0, 2, 3).reshape(8, NB * 3, 128)
            s = mx.softmax((qq @ keys) * (128 ** -0.5) + mask, axis=-1)
            a = (s @ vals).reshape(8, NB, 3, 128).transpose(1, 0, 2, 3).reshape(NB, 3072)
            x = x + self.gm(w["o"], kn.bits(a))
            h = rms(x, w["post_ln"])
            gu = self.gm(w["gate_up"], kn.bits(h))                                                  # [8, 18432]
            gate, up = gu[:, :9216], gu[:, 9216:]
            x = x + self.gm(w["down"], kn.bits(mx.clip(gate * mx.sigmoid(gate) * up, -60000.0, 60000.0)))
        hid = rms(x, self.norm)[1:1 + nd]                                                           # [nd, H]
        base = self.gmu(self.un8, kn.bits(hid)) if self.q8 else kn.gemv(tg.unembed, kn.bits(hid), self.vocab)
        if CAP == "before":
            base = self.cap * mx.tanh(base * (1.0 / self.cap))
        state = mx.zeros((1, 256), dtype=F32)                   # unused: the Markov head here has no recurrent state
        if samp is not None:
            samp = (samp[0], samp[1], cpos[:1].astype(mx.int32) + 1, samp[2])
            ctx = mx.zeros((nd, 1), dtype=mx.int32)             # the whole vocabulary (sampled tokens need every logit)
        elif TOPC:
            ctx = mx.argpartition(-base, TOPC - 1, axis=-1)[:, :TOPC].astype(mx.int32)              # [nd, TOPC] candidates
            base = mx.take_along_axis(base, ctx, axis=-1)
        else:
            ctx = mx.zeros((nd, 1), dtype=mx.int32)
        d, p, state, prev = self._markov(ctx[:self.n1], base[:self.n1], state, anchor, samp)
        return d, p, state, prev, ctx[self.n1:], base[self.n1:]

    def _markov(self, context, base, state, prev, samp=None):
        """logits_i = base_i + markov_w2 . markov_w1[previous draft]; drafts in order."""
        import mk_target as mt
        drafts, probs = [], []
        cand = context.shape[-1] > 1 and samp is None
        for i in range(base.shape[0]):
            e = self.w1[prev].astype(F32)                                                          # [1, 256]
            if cand:            # the bias of the row's candidates only: [TOPC, 256] rows of markov_w2
                bias = (self.w2b[context[i]].astype(F32) @ e[0])[None]
            else:
                bias = (kn.gemv8 if self.q8 else kn.gemv)(self.w2, kn.bits(e), None)[:, :self.vocab]
            z = base[i:i + 1] + bias
            lg = self.cap * mx.tanh(z * (1.0 / self.cap)) if CAP == "after" else z
            if samp is not None:
                prev, pr = mt.sample_rows(lg, samp[0], samp[1], samp[2] + i, samp[3], want_p=True)
                probs.append(pr)
            else:
                sel = mx.argmax(lg, axis=-1).astype(mx.int32)
                prev = mx.take(context[i], sel).astype(mx.int32) if cand else sel
                probs.append(mx.exp(lg.max(axis=-1) - mx.logsumexp(lg, axis=-1)))
            drafts.append(prev)
        return mx.concatenate(drafts), mx.concatenate(probs), state, prev

    def warm(self):
        z = mx.zeros((16, 2304), dtype=F32)
        zs = [z] * (len(self.tg.taps) + 1)
        for n in range(1, 17):
            self.append(zs, 0, n)
        for _ in range(2):
            d, p = self.propose(mx.zeros((1,), dtype=mx.int32), 8)
            mx.eval(d, p)
            if self.n1 < NB - 1:
                mx.eval(*self.more())
        mx.eval(self.kc, self.vc)
        tg = self.tg
        self.frow = [z[:1]] * len(zs)
        for K in list(range(1, 9)) + [16]:
            for chain in ((True, False) if K <= 8 else (False,)):
                tg.reset()
                out = self.run(mx.zeros((K,), dtype=mx.int32), chain)
                mx.eval(out[0], *out[5:])
                tg.commit(K)
        tg.reset()

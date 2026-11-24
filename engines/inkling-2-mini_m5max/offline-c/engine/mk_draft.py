"""The checkpoint's DSpark draft head (mtp.safetensors, mtp_config): three decoder layers over a block of 8 rows (the
anchor = last committed token, and 7 mask rows) attending to a context built from the target's features, and a low-rank
Markov recurrence that turns the block into 7 drafts. The draft only proposes; the verify pass of the target decides
(Draft.cycle below: draft -> K-row verify pass -> accept the matching prefix).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

import mk_kernels as kn
import mk_target as mt

F32 = mx.float32
NB = 8
NARROW = os.environ.get("MK_NARROW", "1") == "1"


def rms(x, w=None, eps=1e-6):
    return mx.fast.rms_norm(x, w, eps)


def ngram(hist, min_match: int, n: int = NB - 1, max_match: int = 24):
    """Copy proposal from the sequence's own history: the longest earlier occurrence (>= min_match tokens, the most
    recent among the longest) of the suffix that ends at the anchor hist[-1]; its continuation, read periodically when
    it runs into the end of the history, is the proposal. Returns a list of n tokens or None."""
    L = len(hist)
    if L < min_match + 1:
        return None
    h = np.asarray(hist, dtype=np.int64)
    cand = np.nonzero(h[:L - 1] == h[L - 1])[0]
    if cand.size == 0:
        return None
    best, bj = 0, -1
    for j in cand[::-1]:
        m = 1
        while m < max_match and j - m >= 0 and h[j - m] == h[L - 1 - m]:
            m += 1
        if m > best:
            best, bj = m, int(j)
            if m >= max_match:
                break
    if best < min_match:
        return None
    period = L - 1 - bj
    return [int(h[bj + 1 + (i % period)]) for i in range(n)]


_BUCK = np.array([0.5, 0.8, 0.95, 0.99])


class Cal:
    """Online calibration of the draft for one request: P(draft j is accepted | the drafts before it were), by depth
    (1, 2, 3+) and by bucket of the draft's own probability, counted with a decay from a prior. The draft's
    probabilities mean different things on different text (0.97+ is accepted 96-100% of the time on prose and code and
    45% on some degenerate continuations: tests/trace.py), so the rows worth a verify are chosen from what this request
    has shown, not from a fixed threshold on the raw probability."""
    PRIOR = np.array([[0.2, 0.5, 0.7, 0.85, 0.97]] * 3)

    def __init__(self, w0: float = 4.0, decay: float = 0.97):
        self.n = np.full((3, 5), w0)
        self.a = self.PRIOR * w0
        self.decay = decay

    def q(self, j: int, p: float) -> float:
        b = int(np.searchsorted(_BUCK, p, side="right"))
        d = min(j, 2)
        return float(self.a[d, b] / self.n[d, b])

    def upd(self, j: int, p: float, ok: bool) -> None:
        b = int(np.searchsorted(_BUCK, p, side="right"))
        d = min(j, 2)
        self.n[d, b] = self.n[d, b] * self.decay + 1.0
        self.a[d, b] = self.a[d, b] * self.decay + (1.0 if ok else 0.0)

    def depth(self, p, thr: float, j0: int = 0, pr: float = 1.0):
        """Rows worth verifying among drafts j0 .. j0 + len(p) - 1: while the calibrated probability that the whole
        prefix is accepted stays >= thr (a row costs ~2.5 ms, a token is worth ~8-10). -> (count, prefix probability)"""
        k = 0
        for i in range(len(p)):
            pr *= self.q(j0 + i, float(p[i]))
            if pr < thr:
                break
            k = i + 1
        return k, pr


class Draft:
    def __init__(self, model_dir: str, tg):
        cfg = json.loads((Path(model_dir) / "config.json").read_text())
        mc = cfg["mtp_config"]
        assert mc["markov_head_type"] == "rnn_hidden" and mc["block_size"] == NB and mc["num_hidden_layers"] == 3
        assert mc["hidden_size"] == 2304 and mc["markov_rank"] == 256 and list(mc["target_layer_ids"]) == list(tg.taps) + [tg.napp]
        assert mc["num_attention_heads"] == 24 and mc["num_key_value_heads"] == 8 and mc["head_dim"] == 128
        self.tg = tg
        # The draft's own K/V. Up to DRAFT_CTX positions it is by position, as the target's. For a longer context it is a
        # ring over the last TD positions (position p in slot p mod TD): the draft's passes rewrite the whole array, so
        # their cost would grow with the context, and what the draft sees only changes what it proposes.
        self.TD = min(tg.T, int(os.environ.get("MK_DRAFT_CTX", "4104")))
        self.ring = self.TD < tg.T
        self.T = self.TD
        self.low = 0                # ring: positions below this are no longer in it (after a rollback)
        self.window = mc["sliding_window_size"]
        self.local = set(mc["local_layer_ids"])
        self.vocab, self.cap = tg.vocab, tg.cap
        self.theta = float(os.environ.get("MK_CUM_TH", "0.3"))
        self.kmin = int(os.environ.get("MK_KMIN", "1"))
        self.skip_low = float(os.environ.get("MK_SKIP_LOW", "0.5"))
        self.skip_n = int(os.environ.get("MK_SKIP_N", "0"))
        self.ng_min = int(os.environ.get("MK_NGRAM", "5"))
        self.ng_wide = os.environ.get("MK_WIDE", "1") == "1"
        self.ng_agree = os.environ.get("MK_NG_AGREE", "0") == "1"      # a DSpark veto on copies: stalls loops the draft cannot predict (journal)
        self.policy = os.environ.get("MK_POLICY", "cala")         # cal: online-calibrated depth; static: theta / kmin
        self.cal_thr = float(os.environ.get("MK_CAL_THR", "0.3"))
        self.thr_lo, self.thr_hi = float(os.environ.get("MK_THR_LO", "0.2")), float(os.environ.get("MK_THR_HI", "0.6"))
        self.row_ms = float(os.environ.get("MK_ROW_MS", "2.5"))
        self.ema = float(os.environ.get("MK_EMA", "0.9"))
        # the rate that sets the threshold counts DSpark cycles only: after a stretch of copy cycles (3 ms per token)
        # the request's rate would put the threshold at its cap for the next cycles of ordinary text
        self.rate_dspark = os.environ.get("MK_RATE_DSPARK", "0") == "1"      # measured: no difference (journal)
        w = mx.load(str(Path(model_dir) / "mtp.safetensors"))
        u16 = lambda a: mx.contiguous(a).view(mx.uint16)  # noqa: E731
        f32 = lambda k: w[k].astype(F32)  # noqa: E731
        self.L = []
        for l in range(3):
            p = f"layers.{l}."
            g = lambda k: w[p + k]  # noqa: E731
            self.L.append({
                "qkv": u16(mx.concatenate([g("self_attn.q_proj.weight"), g("self_attn.k_proj.weight"), g("self_attn.v_proj.weight")])),
                "kv_ctx": u16(mx.concatenate([g("self_attn.k_proj_ctx.weight"), g("self_attn.v_proj_ctx.weight")])),
                "o": u16(g("self_attn.o_proj.weight")),
                "gate_up": u16(mx.concatenate([g("mlp.gate_proj.weight"), g("mlp.up_proj.weight")])),
                "down": u16(g("mlp.down_proj.weight")),
                "in_ln": f32(p + "input_layernorm.weight"), "post_ln": f32(p + "post_attention_layernorm.weight"),
                "dflare": f32(p + "dflare_norm.weight"), "q_norm": f32(p + "self_attn.q_norm.weight"),
                "k_norm": f32(p + "self_attn.k_norm.weight"), "k_norm_ctx": f32(p + "self_attn.k_norm_ctx.weight"),
            })
        self.q8 = os.environ.get("MK_DRAFT8", "1") == "1"
        # 5-bit packed copies of the draft's layer matrices and of its unembedding (MK_DRAFT5=0: the 8-bit copies)
        self.q5 = self.q8 and os.environ.get("MK_DRAFT5", "0") == "1"
        if self.q8:
            for d in self.L:
                for k in ("qkv", "o", "gate_up", "down"):
                    d[k] = kn.quant5(d[k].view(mx.bfloat16)) if self.q5 else kn.quant8(d[k].view(mx.bfloat16), int(os.environ.get("MK_DBITS_L", "8")))
        self.norm = f32("norm.weight")
        self.mask_embed = f32("mask_embed")
        self.alpha = mx.softmax(f32("fuse_weight"), axis=-1).astype(mx.bfloat16).astype(F32)       # [4, 4]
        mh = "markov_head."
        self.m_norm = f32(mh + "dflare_norm.weight")
        self.w1 = w[mh + "markov_w1.weight"]                                                       # [rows, 256] bf16
        rnn = w[mh + "rnn_proj.weight"]                                                            # [768, 5120]
        self.w_rec = u16(rnn[:, :512])
        self.w_ctx = u16(rnn[:, 512:])
        self.rnn_b = f32(mh + "rnn_proj.bias")
        ho_w, ho_b = f32(mh + "hidden_out.weight"), f32(mh + "hidden_out.bias")                   # [2304, 256], [2304]
        # logits = (hid + hidden_out(latent)) . unembed: the hidden_out x unembed product folded once (stored BF16)
        fw, fb = [], []
        un = tg.unembed.view(mx.bfloat16)
        for s in range(0, self.vocab, 16384):
            head = un[s:min(s + 16384, self.vocab)].astype(F32)
            fw.append((head @ ho_w).astype(mx.bfloat16))
            fb.append(head @ ho_b)
            mx.eval(fw[-1], fb[-1])
        self.fold_w = u16(mx.concatenate(fw))                                                      # [V, 256]
        self.un8 = None
        if self.q8:
            ub = int(os.environ.get("MK_DBITS_U", "8"))
            self.fold_w = kn.quant8(self.fold_w.view(mx.bfloat16), ub)
            self.un8 = tg.un8 if (getattr(tg, "un8", None) is not None and ub == 8) else kn.quant8(un[:self.vocab], ub)
            if self.q5:
                self.un8 = kn.quant5(un[:self.vocab])
        self.fold_b = mx.concatenate(fb)                                                           # [V]
        self.inv_freq = 1.0 / (mc["rope_theta"] ** (mx.arange(0, 128, 2).astype(F32) / 128))
        keep = [self.norm, self.mask_embed, self.alpha, self.m_norm, self.w1, self.w_rec, self.w_ctx, self.rnn_b,
                self.fold_w, self.fold_b, self.inv_freq, self.un8] + [v for d in self.L for v in d.values()]
        mx.eval(keep)
        self.gm = kn.gemv5 if self.q5 else (kn.gemv8 if self.q8 else kn.gemv)
        self.gmu = self.gm
        self.tpos = mx.arange(self.T).astype(F32)
        self.tposi = mx.arange(self.T).astype(mx.int32)
        self.chain = os.environ.get("MK_CHAIN", "1") == "1"
        # the committed state (K/V rows, conv windows) is left lazy after a cycle and evaluated as an input of the next
        # pass: a separate async_eval for it holds the host ~0.45 ms before the next pass can be launched
        self.lazy_state = os.environ.get("MK_LAZY_STATE", "0") == "1"      # measured: no difference (journal)
        self._cyc = {}
        self._sfn = {}
        self._fn = None
        self.n1 = int(os.environ.get("MK_DRAFT_N1", "3"))     # Markov steps before the first look at the probabilities
        del w
        self.reset()

    def reset(self):
        self.kc = [mx.zeros((self.T, 8, 128), dtype=F32) for _ in range(3)]
        self.vc = [mx.zeros((self.T, 8, 128), dtype=F32) for _ in range(3)]
        self.frow = None            # the 4 features of the last committed position, [1, H] each
        self.hist = []              # recent acceptance (tokens per cycle)
        self.low = 0

    def _cp(self, pos: int):
        """The position argument of the traced passes: [position] or, for the ring, [position, lowest position kept]."""
        return mx.array([float(pos), float(self.low)] if self.ring else [float(pos)], dtype=F32)

    def rollback(self, old_pos: int):
        """The target was cut back from old_pos: ring slots that later positions (or rows of a pending pass) have
        overwritten no longer hold the positions below old_pos + 16 - TD."""
        if self.ring:
            self.low = max(self.low, old_pos + 16 - self.TD)
        self.frow, self.hist = None, []

    # ---------------------------------------------------------------------------------------------- pieces
    def _rope(self, x, pos):
        """x [n, heads, 128], pos float32 [n]."""
        f = pos[:, None] * self.inv_freq[None]
        cos, sin = mx.cos(f)[:, None], mx.sin(f)[:, None]
        x1, x2 = x[..., :64], x[..., 64:]
        return mx.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)

    def _mix(self, feats, row, weight):
        m = feats[0] * self.alpha[row, 0] + feats[1] * self.alpha[row, 1] + feats[2] * self.alpha[row, 2] + feats[3] * self.alpha[row, 3]
        return rms(m, weight)

    def append(self, feats, pos: int, n: int):
        """feats: 4 arrays [K, H] of a target pass at positions pos .. pos + K - 1; the first n rows become context."""
        f = [x[:n] for x in feats]
        ps = mx.arange(pos, pos + n).astype(F32)
        for l in range(3):
            w = self.L[l]
            kv = kn.gemv(w["kv_ctx"], kn.bits(self._mix(f, l, w["dflare"])))               # [n, 2048]
            k = self._rope(rms(kv[:, :1024].reshape(n, 8, 128), w["k_norm_ctx"]), ps)
            if self.ring:
                idx = (pos + mx.arange(n)) % self.TD
                self.kc[l][idx] = k
                self.vc[l][idx] = kv[:, 1024:].reshape(n, 8, 128)
            else:
                self.kc[l][pos:pos + n] = k
                self.vc[l][pos:pos + n] = kv[:, 1024:].reshape(n, 8, 128)
        self.frow = [x[n - 1:n] for x in feats]

    def _pure(self, anchor, cpos, kc, vc, frow, samp=None):
        """The draft as a pure function with static shapes (traced once by mx.compile): the context is the whole K/V
        array with a mask on the positions (>= c, or outside the window of the local layers)."""
        tg = self.tg
        nd = NB - 1
        e = rms(tg.embed[anchor].astype(F32), tg.embed_norm)
        x = mx.concatenate([e, mx.broadcast_to(self.mask_embed[None], (NB - 1, 2304))])
        if self.ring:               # slot j holds position pj: the largest one below cpos that is j mod TD
            cpos, low = cpos[:1], cpos[1:]
            back = cpos - 1.0 - self.tpos
            pj = cpos - 1.0 - (back - self.TD * mx.floor(back * (1.0 / self.TD)))
            vis_g = (pj >= mx.maximum(low, cpos - (self.TD - 16))) & (pj >= 0)
            vis_l = vis_g & (pj >= cpos - self.window)
        else:
            vis_g = self.tpos < cpos
            vis_l = vis_g & (self.tpos >= cpos - self.window)
        pos = cpos + mx.arange(NB).astype(F32)
        zero8 = mx.zeros((NB,), dtype=F32)
        mask_g = mx.concatenate([mx.where(vis_g, 0.0, -mx.inf), zero8])
        mask_l = mx.concatenate([mx.where(vis_l, 0.0, -mx.inf), zero8])
        residual = None
        for l in range(3):
            w = self.L[l]
            if residual is None:
                residual, h = x, rms(x, w["in_ln"])
            else:
                residual = x + residual
                h = rms(residual, w["in_ln"])
            qkv = self.gm(w["qkv"], kn.bits(h))                                              # [8, 5120]
            q = self._rope(rms(qkv[:, :3072].reshape(NB, 24, 128), w["q_norm"]), pos)
            k = self._rope(rms(qkv[:, 3072:4096].reshape(NB, 8, 128), w["k_norm"]), pos)
            v = qkv[:, 4096:].reshape(NB, 8, 128)
            keys = mx.concatenate([kc[l], k]).transpose(1, 2, 0)                             # [8, 128, T + 8]
            vals = mx.concatenate([vc[l], v]).transpose(1, 0, 2)                             # [8, T + 8, 128]
            qq = q.reshape(NB, 8, 3, 128).transpose(1, 0, 2, 3).reshape(8, NB * 3, 128)
            s = mx.softmax((qq @ keys) * (128 ** -0.5) + (mask_l if l in self.local else mask_g), axis=-1)
            a = (s @ vals).reshape(8, NB, 3, 128).transpose(1, 0, 2, 3).reshape(NB, 3072)
            o = self.gm(w["o"], kn.bits(a))
            residual = o + residual
            h = rms(residual, w["post_ln"])
            gu = self.gm(w["gate_up"], kn.bits(h))                                           # [8, 18432]
            gate, up = gu[:, :9216], gu[:, 9216:]
            x = self.gm(w["down"], kn.bits(mx.clip(gate * mx.sigmoid(gate) * up, -60000.0, 60000.0)))
        hid = rms(x + residual, self.norm)[1:1 + nd]                                         # [nd, H]
        f = self._mix(frow, 3, self.m_norm)                                                  # [1, H]
        inp = mx.concatenate([hid, mx.broadcast_to(f, (nd, 2304))], axis=-1)
        context = kn.gemv(self.w_ctx, kn.bits(inp)) + self.rnn_b                             # [nd, 768]
        base = (self.gmu(self.un8, kn.bits(hid)) if self.q8 else kn.gemv(tg.unembed, kn.bits(hid), self.vocab)) + self.fold_b
        state = mx.zeros((1, 256), dtype=F32)
        if samp is not None:        # (sp, seed, filt): the drafts are drawn with the target's noise of their positions
            samp = (samp[0], samp[1], cpos[:1].astype(mx.int32) + 1, samp[2])
        d, p, state, prev = self._markov(context[:self.n1], base[:self.n1], state, anchor, samp)
        return d, p, state, prev, context[self.n1:], base[self.n1:]

    def _append_pure(self, feats, cposf, kc, vc):
        """The context append inside a traced pass: every row of the pass (positions cpos ..) is written; rows past the
        accepted ones sit at positions the draft masks (>= its anchor) until a later pass rewrites them."""
        K = feats[0].shape[0]
        if self.ring:
            cposf = cposf[:1]
            fwd = self.tpos - cposf
            rel = (fwd - self.TD * mx.floor(fwd * (1.0 / self.TD))).astype(mx.int32)       # (slot - position) mod TD
        else:
            rel = self.tposi - cposf.astype(mx.int32)
        ps = cposf + mx.arange(K).astype(F32)
        mask = ((rel >= 0) & (rel < K))[:, None, None]
        rows = mx.clip(rel, 0, K - 1)
        kc2, vc2 = [], []
        for l in range(3):
            w = self.L[l]
            kv = kn.gemv(w["kv_ctx"], kn.bits(self._mix(feats, l, w["dflare"])))               # [K, 2048]
            k = self._rope(rms(kv[:, :1024].reshape(K, 8, 128), w["k_norm_ctx"]), ps)
            kc2.append(mx.where(mask, k[rows], kc[l]))
            vc2.append(mx.where(mask, kv[:, 1024:].reshape(K, 8, 128)[rows], vc[l]))
        return kc2, vc2

    def _cyc_pure(self, tokens, m_full, m_loc, kc, vc, kr, vr, cw, dkc, dvc, cposf):
        """One verify pass and, behind it on the GPU, the draft of the next cycle: the accepted prefix is counted on
        the GPU, the anchor of the next block is the target's token at the last accepted row. One host sync per cycle."""
        tg = self.tg
        K = tokens.shape[0]
        tgt, feats, knew, vnew, wins = tg._pure(tokens, m_full, m_loc, kc, vc, kr, vr, cw, tok_only=True)
        dkc2, dvc2 = self._append_pure(feats, cposf, dkc, dvc)
        if K > 1:
            a = mx.cumprod((tgt[:K - 1] == tokens[1:]).astype(mx.int32)).sum().reshape(1)
        else:
            a = mx.zeros((1,), dtype=mx.int32)
        frow = [mx.take(f, a, axis=0) for f in feats]
        anchor = mx.take(tgt, a, axis=0).astype(mx.int32)
        nxt = cposf[:1] + (a + 1).astype(F32)
        d, p, state, prev, ctx, base = self._pure(anchor, mx.concatenate([nxt, cposf[1:]]) if self.ring else nxt, dkc2, dvc2, frow)
        return tgt, feats, knew, vnew, wins, dkc2, dvc2, d, p, state, prev, ctx, base

    def _nod_pure(self, tokens, m_full, m_loc, kc, vc, kr, vr, cw, dkc, dvc, cposf):
        """The verify pass with the context append, no draft behind it (copy cycles)."""
        tgt, feats, knew, vnew, wins = self.tg._pure(tokens, m_full, m_loc, kc, vc, kr, vr, cw, tok_only=True)
        dkc2, dvc2 = self._append_pure(feats, cposf, dkc, dvc)
        return tgt, feats, knew, vnew, wins, dkc2, dvc2

    def _cyc_samp(self, chain: bool, filt: bool):
        """_cyc_pure / _nod_pure for a sampling request: the rows' tokens are drawn from their logits (row r chooses
        the token of position pos + r + 1), and so are the drafts behind the pass."""
        def f(tokens, m_full, m_loc, kc, vc, kr, vr, cw, dkc, dvc, cposf, sp, seed):
            tg = self.tg
            K = tokens.shape[0]
            lg, feats, knew, vnew, wins = tg._pure(tokens, m_full, m_loc, kc, vc, kr, vr, cw, tok_only=False)
            tgt = mt.sample_rows(lg, sp, seed, m_full[0:1] + 1, filt)[0]
            dkc2, dvc2 = self._append_pure(feats, cposf, dkc, dvc)
            if not chain:
                return tgt, feats, knew, vnew, wins, dkc2, dvc2
            if K > 1:
                a = mx.cumprod((tgt[:K - 1] == tokens[1:]).astype(mx.int32)).sum().reshape(1)
            else:
                a = mx.zeros((1,), dtype=mx.int32)
            frow = [mx.take(x, a, axis=0) for x in feats]
            anchor = mx.take(tgt, a, axis=0).astype(mx.int32)
            nxt = cposf[:1] + (a + 1).astype(F32)
            d, p, state, prev, ctx, base = self._pure(anchor, mx.concatenate([nxt, cposf[1:]]) if self.ring else nxt, dkc2, dvc2, frow,
                                                      samp=(sp, seed, filt))
            return tgt, feats, knew, vnew, wins, dkc2, dvc2, d, p, state, prev, ctx, base, nxt.astype(mx.int32) + 1 + self.n1
        return f

    def run(self, toks: mx.array, chain: bool, samp=None):
        """The pass over toks at the target's position (pending until tg.commit), traced once per (K, chain) (and per
        kind of sampling: none, plain, with top_p / top_k)."""
        tg = self.tg
        K = toks.shape[0]
        assert tg.pos + K <= tg.T and tg.pos - tg.base + K <= tg.RECENT
        key = (K, chain) if samp is None else (K, chain, samp.filt)
        fn = self._cyc.get(key)
        if fn is None:
            fn = self._cyc[key] = mx.compile((self._cyc_pure if chain else self._nod_pure) if samp is None else self._cyc_samp(chain, samp.filt))
        out = fn(toks, mx.array([tg.pos, 0, tg.base], dtype=mx.int32), mx.array([tg.pos, 1024, tg.base], dtype=mx.int32),
                 tg.kc, tg.vc, tg.kr, tg.vr, tg.cw, self.kc, self.vc, self._cp(tg.pos), *(() if samp is None else (samp.sp, samp.seed)))
        tg.pending = (out[2], out[3], out[4], K)
        return out

    def _markov(self, context, base, state, prev, samp=None):
        """Markov steps over the rows of context / base: each draft conditions the next through the low-rank state.
        samp = (sp, seed, position of the first draft, filt): each draft is drawn as the target draws at its position
        (mk_target.sample_rows) instead of the argmax."""
        drafts, probs = [], []
        for i in range(context.shape[0]):
            rin = mx.concatenate([state, self.w1[prev].astype(F32)], axis=-1)
            raw = context[i:i + 1] + kn.gemv(self.w_rec, kn.bits(rin))
            gt = mx.sigmoid(raw[:, :256])
            state = gt * state + (1.0 - gt) * mx.tanh(raw[:, 256:512])
            latent = mx.tanh(raw[:, 512:])
            z = base[i:i + 1] + (kn.gemv8_narrow(self.fold_w, kn.bits(latent)) if (self.q8 and NARROW) else (kn.gemv8 if self.q8 else kn.gemv)(self.fold_w, kn.bits(latent)))
            lg = self.cap * mx.tanh(z * (1.0 / self.cap))
            if samp is not None:
                prev, pr = mt.sample_rows(lg, samp[0], samp[1], samp[2] + i, samp[3], want_p=True)
                probs.append(pr)
            else:
                prev = mx.argmax(lg, axis=-1).astype(mx.int32)
                probs.append(mx.exp(lg.max(axis=-1) - mx.logsumexp(lg, axis=-1)))
            drafts.append(prev)
        return mx.concatenate(drafts), mx.concatenate(probs), state, prev

    def _samp_fns(self, filt: bool):
        """(the block, the rest of the block) for a sampling request, traced once per filt."""
        fns = self._sfn.get(filt)
        if fns is None:
            fns = self._sfn[filt] = (
                mx.compile(lambda anchor, cpos, kc, vc, frow, sp, seed: self._pure(anchor, cpos, kc, vc, frow, samp=(sp, seed, filt))),
                mx.compile(lambda context, base, state, prev, sp, seed, q: self._markov(context, base, state, prev, (sp, seed, q, filt))))
        return fns

    def propose(self, anchor, c: int, nd: int = NB - 1, samp=None):
        """anchor: int32 [1], the token at position c -> the first n1 drafts and their probabilities (lazy) and the
        rest of the block for more()."""
        if self._fn is None:
            comp = os.environ.get("MK_COMPILE", "1") == "1"
            self._fn = mx.compile(self._pure) if comp else self._pure
            self._fn2 = mx.compile(self._markov) if comp else self._markov
        if samp is not None:
            fn, self._rest_fn = self._samp_fns(samp.filt)
            d, p, state, prev, ctx, base = fn(anchor, self._cp(c), self.kc, self.vc, self.frow, samp.sp, samp.seed)
            self._rest = (ctx, base, state, prev, samp.sp, samp.seed, mx.array([c + 1 + self.n1], dtype=mx.int32))
            return d, p
        d, p, state, prev, ctx, base = self._fn(anchor, self._cp(c), self.kc, self.vc, self.frow)
        self._rest, self._rest_fn = (ctx, base, state, prev), self._fn2
        return d, p

    def more(self):
        """The remaining drafts of the block proposed last (run only while the draft is still confident)."""
        d, p, _, _ = self._rest_fn(*self._rest)
        return d, p

    def warm(self):
        z = mx.zeros((8, 2304), dtype=F32)
        z = mx.zeros((16, 2304), dtype=F32)
        for n in range(1, 17):
            self.append([z, z, z, z], 0, n)
        for _ in range(2):
            d, p = self.propose(mx.zeros((1,), dtype=mx.int32), 8)
            mx.eval(d, p)
            if self.n1 < NB - 1:
                mx.eval(*self.more())
        mx.eval(self.kc, self.vc)
        tg = self.tg
        self.frow = [z[:1]] * 4
        for K in list(range(1, 9)) + [16]:
            for chain in ((True, False) if K <= 8 else (False,)):
                tg.reset()
                out = self.run(mx.zeros((K,), dtype=mx.int32), chain)
                mx.eval(out[0], *out[5:])
                tg.commit(K)
        tg.reset()

    # ---------------------------------------------------------------------------------------------- one cycle
    def cycle(self, eng, act, left: int):
        """One draft-and-verify cycle; returns the committed tokens (1 + the accepted drafts). The draft of a DSpark
        cycle was computed behind the previous verify pass (run(chain=True)) and read with its tokens; only the
        first cycle of a request and the cycle after a copy cycle draft on their own."""
        tg = self.tg
        c = tg.pos
        nd = NB - 1
        cnt = eng.counters
        samp = act.get("samp")      # None: greedy; else the request's sampling (the same cycle, drawn tokens for argmax ones)
        t0 = time.perf_counter()
        syncs = 1
        # copy proposals: 7 drafts, or 15 (a 16-row pass) while the previous copy cycle was accepted in full
        wide = self.ng_wide and act.get("ng_full", False) and left - 1 >= 15
        cont = ngram(act["hist"], self.ng_min, 15 if wide else nd) if self.ng_min > 0 else None
        dp = act.pop("draft", None)
        if cont is not None and self.ng_agree and dp is not None and int(dp[0][0]) != int(cont[0]):
            # the DSpark draft for this anchor is in hand and its first token contradicts the copy: a copy proposal is
            # then right about its first token about one time in three (tests/replay3.py); draft with DSpark
            cont = None
            cnt["copy_vetoed"] = cnt.get("copy_vetoed", 0) + 1
        if cont is not None:
            d = np.asarray(cont, dtype=np.int64)
            kd = len(cont)
            cnt["ngram_cycles"] = cnt.get("ngram_cycles", 0) + 1
            if dp is not None:
                cnt["draft_unused"] = cnt.get("draft_unused", 0) + 1
        else:
            if dp is None:
                dl, pl = self.propose(mx.array([act["last"]], dtype=mx.int32), c, nd, samp)
                mx.eval(dl, pl, *self._rest)
                dp = (np.array(dl), np.array(pl))
                syncs += 1
                cnt["draft_alone"] = cnt.get("draft_alone", 0) + 1
            d, p = dp
            cal = act.get("cal") if self.policy in ("cal", "cala") else None
            thr = self.cal_thr
            if self.policy == "cala" and act.get("ema_t", 0.0) > 0.0:
                # a row pays when its chance of yielding a token, per ms of row, beats the request's recent
                # tokens per ms: threshold = row ms x tokens per ms (hard prose ~0.23, easy text ~0.5)
                thr = min(self.thr_hi, max(self.thr_lo, self.row_ms * 1e-3 * act["ema_n"] / act["ema_t"]))
            if cal is not None:
                kd, pr = cal.depth(p, thr)
                want_more = kd == len(p)
            else:
                want_more = np.prod(p) >= self.theta
            if len(p) < nd and want_more:
                d2, p2 = self.more()
                mx.eval(d2, p2)
                p2 = np.array(p2)
                if cal is not None:
                    kd += cal.depth(p2, thr, len(p), pr)[0]
                d, p = np.concatenate([d, np.array(d2)]), np.concatenate([p, p2])
                syncs += 1
                cnt["draft_more"] = cnt.get("draft_more", 0) + 1
            if cal is None:
                cum = np.cumprod(p)
                # rows worth verifying: the draft's probabilities are overconfident (cumulative 0.5 is accepted
                # ~25-40%), a row costs ~2.3 ms and a token is worth ~10 ms
                kd = max(self.kmin, int((cum >= self.theta).cumprod().sum()))
        t1 = time.perf_counter()
        kd = max(0, min(kd, left - 1))
        K = 1 + kd
        toks = mx.array(np.concatenate([[act["last"]], d[:kd]]).astype(np.int32))
        chain = self.chain and cont is None and left - K > 0
        out = self.run(toks, chain, samp)
        tgt, feats = out[0], out[1]
        t2 = time.perf_counter()
        if "t_sync" in act:
            cnt["gap_s"] = cnt.get("gap_s", 0.0) + (t2 - act["t_sync"])        # host time between a sync and the next launch
            cnt["build_s"] = cnt.get("build_s", 0.0) + (t2 - t1)
        if chain:
            mx.eval(tgt, *out[7:])
        else:
            mx.eval(tgt)
        act["t_sync"] = time.perf_counter()
        t = np.array(tgt)
        a = 0
        while a < kd and int(d[a]) == int(t[a]):
            a += 1
        n_acc = a + 1
        dt = time.perf_counter() - t0
        if cont is None or not self.rate_dspark:
            act["ema_t"] = act.get("ema_t", 0.0) * self.ema + dt
            act["ema_n"] = act.get("ema_n", 0.0) * self.ema + n_acc
        if cont is None and act.get("cal") is not None and self.policy in ("cal", "cala"):
            for j in range(min(a + 1, kd)):
                act["cal"].upd(j, float(p[j]), j < a)
        self.kc, self.vc = out[5], out[6]
        self.frow = [f[a:a + 1] for f in feats]
        tg.commit(n_acc)
        if not self.lazy_state:
            mx.async_eval(self.kc, self.vc, *tg.state())
        if chain:
            act["draft"] = (np.array(out[7]), np.array(out[8]))
            if samp is None:
                self._rest, self._rest_fn = (out[11], out[12], out[9], out[10]), self._fn2
            else:
                self._rest, self._rest_fn = (out[11], out[12], out[9], out[10], samp.sp, samp.seed, out[13]), self._samp_fns(samp.filt)[1]
        cnt["rows"] += K
        cnt["draft_s"] = cnt.get("draft_s", 0.0) + (t1 - t0)
        cnt["verify_s"] = cnt.get("verify_s", 0.0) + (time.perf_counter() - t1)
        cnt["host_syncs"] += syncs
        act["ng_full"] = cont is not None and a == kd and kd >= nd
        if cont is not None:
            cnt["ngram_tokens"] = cnt.get("ngram_tokens", 0) + n_acc
            if K == 16:
                cnt["wide_cycles"] = cnt.get("wide_cycles", 0) + 1
        res = [int(v) for v in t[:n_acc]]
        act["hist"].extend(res)
        return res

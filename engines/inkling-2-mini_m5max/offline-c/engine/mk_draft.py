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
# greedy drafts: the Markov steps score only the base logits' MTOP best tokens of a row (0: the whole vocabulary)
MTOP = int(os.environ.get("MK_MTOP", "64"))      # 64: 1-3% faster decode, the more() steps 0.76 -> 0.19 ms
TRACE = bool(os.environ.get("MK_HIST_DUMP"))     # debug: a per-cycle trace (position, kind, rows, accepted, seconds)


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


def ngram_src(hist, min_match: int, max_match: int = 24):
    """ngram()'s match as the position where the copied continuation starts in hist (None: no match; or the match's
    continuation reaches the end of the history)."""
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
    return bj + 1, best


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
    tiers = False       # Draft2: the context K/V in two tiers (the position argument then carries dbase too)

    def __init__(self, model_dir: str, tg):
        cfg = json.loads((Path(model_dir) / "config.json").read_text())
        mc = cfg["mtp_config"]
        assert mc["markov_head_type"] == "rnn_hidden" and mc["block_size"] == NB and mc["num_hidden_layers"] == 3
        assert mc["hidden_size"] == 2304 and mc["markov_rank"] == 256 and getattr(tg, "mtp_ids", list(mc["target_layer_ids"])) == list(tg.taps) + [tg.napp]
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
        self.jc_ts = None
        if JSON_COPY:
            try:
                from tokenizers import Tokenizer
                tk = Tokenizer.from_file(str(Path(model_dir) / "tokenizer.json"))
                self.jc_ts = tk.decode_batch([[i] for i in range(tk.get_vocab_size())], skip_special_tokens=False)
                _tok_ids(self.jc_ts)
            except Exception as e:      # no tokenizer: plain copy proposals only
                print("MK_JSON_COPY off:", e, flush=True)
                self.jc_ts = None
        self.review_r = int(os.environ.get("MK_REVIEW", "0"))
        self.review_thr = float(os.environ.get("MK_REVIEW_THR", "0.2"))
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
        elif MTOP and self.q8:
            # greedy drafts: the Markov steps score the base logits' MTOP best tokens of each row only (their values and
            # indices side by side in `base`, which the rest of the block carries to more())
            import mk_draft2
            cands, vals = mk_draft2.top_cands(base, MTOP)
            base = mx.concatenate([vals, cands.astype(F32)], axis=-1)
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
        d, p, state, prev, ctx, base = self._pure(anchor, mx.concatenate([nxt, cposf[1:]]) if (self.ring or self.tiers) else nxt, dkc2, dvc2, frow)
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
            d, p, state, prev, ctx, base = self._pure(anchor, mx.concatenate([nxt, cposf[1:]]) if (self.ring or self.tiers) else nxt, dkc2, dvc2, frow,
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
            if samp is None and MTOP and self.q8:
                ci = base[i, MTOP:].astype(mx.int32)                                         # [MTOP] candidates
                fq = self.fold_w[0][ci].astype(F32) * self.fold_w[1][ci][:, None]            # [MTOP, 256]
                lat = latent.astype(mx.float16).astype(F32)                                  # the half bits gemv8 reads
                z = base[i:i + 1, :MTOP] + (lat @ fq.T)
                lg = self.cap * mx.tanh(z * (1.0 / self.cap))
                j = mx.argmax(lg, axis=-1)
                prev = ci[j].astype(mx.int32)
                probs.append(mx.exp(lg.max(axis=-1) - mx.logsumexp(lg, axis=-1)))
                drafts.append(prev)
                continue
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
        for K in list(range(1, 17)) if (self.jc_ts is not None and JC_MID) else list(range(1, 9)) + [16]:
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
        if self.tiers:
            self.maybe_flush()
        # copy proposals: 7 drafts, or 15 (a 16-row pass) while the previous copy cycle was accepted in full
        wide = self.ng_wide and act.get("ng_full", False) and left - 1 >= 15
        cont, jc = None, None
        if self.jc_ts is not None and left > 1:
            jc = act.get("jc")
            if jc is None:
                jc = act["jc"] = JsonCopy(self.jc_ts, act.get("plen", len(act["hist"]) - 1))
            try:
                jc.update(act["hist"])
                if jc.active():
                    hist = act["hist"]
                    h = np.asarray(hist, dtype=np.int64)
                    cont = jc.propose(hist, h, min(JC_MAX, left - 1),
                                      (lambda: ngram_src(h, self.ng_min)) if self.ng_min > 0 else None, act.get("ng_full", False))
                    if cont is not None and not JC_MID and nd < len(cont) < 15:
                        cont = cont[:nd]        # passes of 1-8 and 16 rows only
                    if cont is not None:
                        cnt["json_cycles"] = cnt.get("json_cycles", 0) + 1
            except Exception as e:      # a proposal is never worth a failed request: plain copies for the rest of it
                print("MK_JSON_COPY error:", repr(e), flush=True)
                jc.lx.dead, cont = True, None
            if cont is None:
                jc = None
        if cont is None:
            cont = ngram(act["hist"], self.ng_min, 15 if wide else nd) if self.ng_min > 0 else None
        dp = act.pop("draft", None)
        nod = act.pop("nodraft", False)
        m_ok = getattr(self, "rounds", 0) > 0 and left >= 3 and act.get("cal") is not None and (samp is None or getattr(self, "rounds_samp", False))
        if nod and cont is None and m_ok:
            return self._cycle_m(eng, act, left, None, t0, 0, samp=samp)      # no draft was made for this anchor: readout rounds
        if (cont is not None and not wide and getattr(self, "copy_review", False) and samp is None and getattr(self, "rounds", 0) > 0
                and left >= 3 and act.get("cal") is not None):
            # a copy proposal not yet proven (the first of a loop): its tokens as the drafts of a readout round, which
            # drops a wrong copy at a third of a verified row's cost and goes on with the readout's token
            cnt["ngram_cycles"] = cnt.get("ngram_cycles", 0) + 1
            cnt["copy_rounds"] = cnt.get("copy_rounds", 0) + 1
            return self._cycle_m(eng, act, left, (np.asarray(cont, dtype=np.int64), None), t0, 0, copy=True)
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
            if self.review_r > 0 and samp is None:
                thr = self.review_thr       # a reviewed row costs a third of a verified one
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
        if cont is None and m_ok and cal is not None and kd < self.m_kd:
            # the draft is not worth many rows here: rounds of early readouts instead (mk_draft2)
            return self._cycle_m(eng, act, left, (d, p), t0, syncs, samp=samp)
        reviewed = False
        if cont is None and samp is None and self.review_r > 0 and left > 1:
            # review: an early readout (review_r loops, exit through the suffix) of the anchor and the first kr drafts.
            # Row j's token is the early model's choice after draft j: the drafts it confirms stay, the first one it
            # contradicts is replaced by its choice and the rest dropped. A proposal only: the verify pass decides.
            kr = max(0, min(kd, left - 2, NB - 2))
            e = np.array(tg.early(mx.array(np.concatenate([[act["last"]], d[:kr]]).astype(np.int32)), self.review_r)[0])
            syncs += 1
            m = 0
            while m < kr and int(e[m]) == int(d[m]):
                m += 1
            if cal is not None:
                for j in range(min(m + 1, kr)):
                    cal.upd(j, float(p[j]), j < m)
            d = np.concatenate([d[:m], e[m:m + 1]]).astype(np.int64)
            kd = m + 1
            reviewed = True
            cnt["reviews"] = cnt.get("reviews", 0) + 1
            cnt["review_rows"] = cnt.get("review_rows", 0) + kr + 1
            cnt["review_kept"] = cnt.get("review_kept", 0) + m
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
        if reviewed:
            cnt["review_acc"] = cnt.get("review_acc", 0) + a
            cnt["review_sent"] = cnt.get("review_sent", 0) + kd
        elif cont is None and act.get("cal") is not None and self.policy in ("cal", "cala"):
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
        if jc is not None:
            jc.note(a)
        if cont is not None:
            cnt["ngram_rows"] = cnt.get("ngram_rows", 0) + K
            cnt["ngram_full"] = cnt.get("ngram_full", 0) + int(a == kd)
            cnt["ngram_tokens"] = cnt.get("ngram_tokens", 0) + n_acc
            if K == 16:
                cnt["wide_cycles"] = cnt.get("wide_cycles", 0) + 1
        res = [int(v) for v in t[:n_acc]]
        if TRACE:
            act.setdefault("trace", []).append((len(act["hist"]), "c" if cont is not None else "d", K, a, round(time.perf_counter() - t0, 5)))
        act["hist"].extend(res)
        return res


# ------------------------------------------------------------------------------------------------- JSON-aware copies
# An answer that is a JSON object (a tool call's arguments, a structured output) is mostly copied: string values quote
# the prompt (a field's text, a field id), a value repeats an earlier one (a new_string that starts as its old_string),
# and the punctuation and keys between values repeat. JsonCopy follows the answer's JSON structure on its tokens and
# builds copy proposals that know where a copied string ends (its closing quote), what comes after the value (the same
# keys and punctuation as the last time a value of that key closed), and what the next value likely starts with. Only
# proposals: the verify pass decides every token. An answer that does not start with { or [ gets none.
JSON_COPY = os.environ.get("MK_JSON_COPY", "1") == "1"      # on: +17% on tool-call answers; rare wording differences from strict greedy output (verify-pass grouping), MK_JSON_COPY=0 turns it off
JC_STRUCT = os.environ.get("MK_JC_STRUCT", "1") == "1"       # the keys and punctuation after a value closes
JC_VALUE = os.environ.get("MK_JC_VALUE", "1") == "1"         # a value from its key's context as soon as it opens
JC_PFX = int(os.environ.get("MK_JC_PFX", "1"))               # complete a string from its first JC_PFX+ tokens (0: off)
JC_PFX_AGREE = int(os.environ.get("MK_JC_PFX_AGREE", "1"))  # tokens the candidate completions must agree on
JC_SCHEMA_FIRST = os.environ.get("MK_JC_SCHEMA_FIRST", "1") == "1"
JC_NG_SURE = int(os.environ.get("MK_JC_NG_SURE", "8"))      # a copy match this long proposes up to JC_MAX drafts
JC_WIDE_M = int(os.environ.get("MK_JC_WIDE_M", "99"))        # after a full copy cycle: 15 drafts once this far into a string
JC_TRY = int(os.environ.get("MK_JC_TRY", "3"))              # drafts past a value guess not yet right in this answer
JC_MID = os.environ.get("MK_JC_MID", "1") == "1"            # verify passes of 9-15 rows too
JC_CUT = os.environ.get("MK_JC_CUT", "1") == "1"             # a copy stops where its source string ends
JC_MAX = int(os.environ.get("MK_JC_MAX", "15"))              # drafts of a JSON proposal (rows - 1)
JC_STRUCT_CAP = 16


class _Lex:
    """Incremental JSON lexer over decoded tokens. A frame: [kind '{' | '[' | '?', expect 'k' key | 'c' colon | 'v' value
    | ',' comma, the object's last key (token tuple), the last string value's (start, end) in the sequence]."""
    COMMA = 11                  # the token id of ",", set by JsonCopy

    __slots__ = ("st", "ins", "esc", "key", "o", "al", "dead", "done", "started", "last", "lc", "lk", "kq", "sep")

    def __init__(self, st=None):
        self.st = st if st is not None else []
        self.ins = self.esc = self.key = False
        self.o, self.al = -1, True
        self.dead = self.done = False
        self.started = st is not None
        self.last = None            # ('v', key) right after a value string closed, ('k', key) after a key closed
        self.lc = None              # (('v', key), the position after the close) until the next value starts
        self.lk = None              # the same for the last key that closed
        self.kq = 1                 # the token that opened the last key (a quote, with or without a space)
        self.sep = None             # the comma and quote tokens before a key that follows a value

    @staticmethod
    def ts_comma(t) -> bool:
        return int(t) == _Lex.COMMA

    def copy(self):
        c = _Lex.__new__(_Lex)
        c.st = [f[:] for f in self.st]
        c.ins, c.esc, c.key, c.o, c.al = self.ins, self.esc, self.key, self.o, self.al
        c.dead, c.done, c.started, c.last, c.lc, c.lk, c.kq, c.sep = self.dead, self.done, self.started, self.last, self.lc, self.lk, self.kq, self.sep
        return c

    def feed(self, s: str, idx: int, seq):
        """Feed the token at seq[idx] (decoded s). -> an event or None: ('vopen',) a value string opened, ('vclose', key)
        a value string closed, ('lit',) a non-string value started."""
        if self.dead or self.done:
            return None
        if not self.started:
            if s.startswith("<|") and s.endswith("|>"):
                return None
            if not s.strip():
                return None
        ev = None
        n = len(s)
        for i, ch in enumerate(s):
            if self.ins:
                if self.esc:
                    self.esc = False
                elif ch == "\\":
                    self.esc = True
                elif ch == '"':
                    self.ins = False
                    st = self.st[-1]
                    a, b = self.o + 1, idx
                    ok = self.al and i == 0
                    if self.key:
                        st[2] = tuple(seq[a:b]) if ok else None
                        st[1] = "c"
                        self.last = ("k", st[2])
                        ev = ("kclose", st[2])
                        self.lk = (self.last, idx + 1)
                    else:
                        st[3] = (a, b) if ok else None
                        st[1] = ","
                        self.last = ("v", st[2] if st[0] != "[" else None)
                        ev = ("vclose", self.last[1])
                        self.lc = (self.last, idx + 1)
                elif ch == "\n":
                    self.dead = True
                    return None
                continue
            if ch in " \t\r\n":
                continue
            if not self.started:
                if ch not in "{[":
                    self.dead = True
                    return None
                self.started = True
            self.last = None
            st = self.st[-1] if self.st else None
            if ch == '"':
                if st is None:
                    self.dead = True
                    return None
                self.ins, self.esc = True, False
                self.key = st[0] != "[" and st[1] == "k"
                self.o, self.al = idx, i == n - 1
                if not self.key:
                    st[1] = ","
                    ev = ("vopen",)
                    self.lc = self.lk = None
                else:
                    self.kq = int(seq[idx])
                    if idx > 0 and self.ts_comma(seq[idx - 1]):
                        self.sep = [int(seq[idx - 1]), self.kq]
            elif ch in "{[":
                if st is not None:
                    if st[1] == "v" or st[0] == "[":
                        ev = ("copen",)
                        self.lk = None
                    st[1] = ","
                self.st.append([ch, "k" if ch == "{" else "v", None, None])
            elif ch in "}]":
                if not self.st:
                    self.dead = True
                    return None
                self.st.pop()
                if not self.st:
                    self.done = True
                    return ev
            elif ch == ":":
                if st is not None:
                    st[1] = "v"
            elif ch == ",":
                if st is not None:
                    st[1] = "v" if st[0] == "[" else "k"
            else:
                if st is None:
                    self.dead = True
                    return None
                if st[1] != ",":
                    ev = ("lit",)
                    self.lc = self.lk = None
                st[1] = ","
        return ev


_IDS = {}


def _tok_ids(ts):
    """The ids of the one-character tokens the proposals write: '"', ',', ':' and the quote tokens."""
    r = _IDS.get(id(ts))
    if r is None:
        first = {}
        for i, s in enumerate(ts):
            if s in ('"', ",", ":", ' "') and s not in first:
                first[s] = i
        r = _IDS[id(ts)] = {'"': first['"'], ",": first[","], ":": first[":"],
                            "q1": {first['"']} | ({first[' "']} if ' "' in first else set())}
    return r


class JsonCopy:
    """Per-request state of the JSON-aware proposals (see above)."""

    def __init__(self, ts, plen: int):
        self.ts = ts                # token id -> decoded text
        ids = _tok_ids(ts)
        self.q1 = ids["q1"]         # token ids that are one quote (with or without a leading space)
        close_tok, self.colon = ids['"'], ids[":"]
        _Lex.COMMA = ids[","]
        self.plen = plen            # the answer starts here
        self.lx = _Lex()
        self.n = plen               # tokens fed
        self.close = close_tok
        self.M = {}                 # ('v', key) / ('k', key) -> the tokens that followed the last time
        self.coll = []              # (ctx, start): collecting what follows a close
        self.cache = {}
        self.stats = {}             # (key, kind) -> [ok, tried]
        self.segs = []

    def s(self, t: int) -> str:
        return self.ts[t] if 0 <= t < len(self.ts) else "\x00"

    def update(self, hist):
        """Feed the committed tokens hist[self.n:]; learn what follows each close."""
        for i in range(self.n, len(hist)):
            ev = self.lx.feed(self.s(hist[i]), i, hist)
            if self.lx.dead:
                break
            keep = []
            for ctx, st in self.coll:
                if (ev is not None and (ev[0] in ("vopen", "lit") or (ev[0] == "copen" and ctx[0] == "k"))) or i + 1 - st >= JC_STRUCT_CAP or self.lx.done:
                    self.M[ctx] = list(hist[st:i + 1])
                    if ctx[0] == "k":
                        self.M[("k@", len(self.lx.st) - (1 if ev is not None and ev[0] == "copen" else 0))] = self.M[ctx]
                        if ev is not None and ev[0] == "vopen":
                            self.M[("k", None)] = self.M[ctx]
                else:
                    keep.append((ctx, st))
            self.coll = keep
            if ev is not None and ev[0] in ("vclose", "kclose"):
                self.coll.append((("v" if ev[0] == "vclose" else "k", ev[1]), i + 1))
        self.n = len(hist)

    def active(self) -> bool:
        return self.lx.started and not self.lx.dead and not self.lx.done

    # -------------------------------------------------------------------------------- searches over the sequence
    def _read_str(self, h, j: int, end: int, cap: int):
        """The string content from h[j] up to (excluding) its closing quote, stopping at a raw newline too.
        -> (tokens, closed) or (tokens, False) when the cap or the end came first."""
        out = []
        esc = False
        while j < end and len(out) < cap:
            t = int(h[j])
            s = self.s(t)
            if t in self.q1 and not esc:
                return out, True
            if "\n" in s or '"' in s:
                return out, len(out) > 0
            out.append(t)
            k = len(s) - len(s.rstrip("\\"))
            esc = (k % 2 == 1)
            j += 1
        return out, False

    def _find(self, h, seq, lo: int, hi: int):
        """Start positions p in [lo, hi - len(seq)] of seq in h, most recent first."""
        m = len(seq)
        if m == 0 or hi - lo < m:
            return np.zeros(0, dtype=np.int64)
        c = np.nonzero(h[lo:hi - m + 1] == seq[0])[0] + lo
        for k in range(1, m):
            if c.size == 0:
                break
            c = c[h[c + k] == seq[k]]
        return c[::-1]

    def _quoted_at(self, h, p: int) -> bool:
        return p > 0 and int(h[p - 1]) in self.q1

    def kv_value(self, h, v, hi: int):
        """The value of key v where v last appears as a key before hi: "v": "<value>" (or v: <text> to the line end)."""
        ck = ("kv", tuple(v))
        if ck in self.cache:
            return self.cache[ck]
        res = None
        for p in self._find(h, list(v), 0, hi)[:64]:
            p = int(p)
            e = p + len(v)
            if e + 2 >= hi:
                continue
            if self._quoted_at(h, p) and int(h[e]) == self.close and self.s(int(h[e + 1])).strip() == ":" and int(h[e + 2]) in self.q1:
                toks, closed = self._read_str(h, e + 3, hi, 96)
                if closed:
                    res = toks
                    break
        self.cache[ck] = res
        return res

    def struct_after(self, h, ctx, hi: int, depth: int = 0):
        """What follows a close of kind ctx: learned in this answer, else from the last "key": "value" of that key in
        the sequence (its continuation up to the next string value's opening quote)."""
        r = self.M.get(ctx)
        if r is None and ctx[0] == "k":
            r = self.M.get(("k@", depth)) or self.M.get(("k", None))
        if r is not None:
            return r
        ck = ("st", ctx)
        if ck in self.cache:
            return self.cache[ck]
        res = None
        kind, key = ctx
        if JC_SCHEMA_FIRST and kind == "v" and key is not None:
            res = self._next_key(h, key, hi)
        if res is None and key is not None:
            for p in self._find(h, list(key), 0, hi)[:64]:
                p = int(p)
                e = p + len(key)
                if not (self._quoted_at(h, p) and e + 1 < hi and int(h[e]) == self.close):
                    continue
                if kind == "k":
                    if self.s(int(h[e + 1])).strip() != ":":
                        continue
                    j = e + 1
                else:
                    if not (e + 2 < hi and self.s(int(h[e + 1])).strip() == ":" and int(h[e + 2]) in self.q1):
                        continue
                    toks, closed = self._read_str(h, e + 3, hi, 256)
                    if not closed:
                        continue
                    j = e + 3 + len(toks) + 1
                lx = _Lex([["?", ",", None, None], ["?", ",", None, None], ["{", "c" if kind == "k" else ",", key, None]])
                out = []
                while j < hi and len(out) < JC_STRUCT_CAP:
                    t = int(h[j])
                    ev = lx.feed(self.s(t), j, h)
                    if lx.dead:
                        break
                    out.append(t)
                    if lx.done or (ev is not None and (ev[0] in ("vopen", "lit") or (ev[0] == "copen" and kind == "k"))):
                        break
                    j += 1
                if out and not lx.dead and (lx.ins or lx.done or len(out) == JC_STRUCT_CAP):
                    res = out
                    break
        if res is None and kind == "v" and key is not None:
            res = self._next_key(h, key, hi)
        self.cache[ck] = res
        return res

    def _next_key(self, h, key, hi: int):
        """A key list in the sequence (a JSON array of strings such as a schema's "required") that holds key: the next
        element is the next key -> , "next": " (the punctuation as this answer writes it once it has)."""
        m = len(key)
        for p in self._find(h, list(key), 0, hi)[:64]:
            p = int(p)
            e = p + m
            if not (self._quoted_at(h, p) and e + 2 < hi and int(h[e]) == self.close and self.s(int(h[e + 1])).strip() == ","
                    and int(h[e + 2]) in self.q1):
                continue
            nxt, closed = self._read_str(h, e + 3, hi, 16)
            if not closed or not nxt:
                continue
            kc = self.M.get(("k", None)) or [self.colon, self.close]
            return list(self.lx.sep or [int(h[e + 1]), self.lx.kq]) + nxt + [self.close] + list(kc)
        return None

    def _complete(self, h, cur, here: int, hi: int, n: int):
        """A string that starts with cur: the continuation every earlier quoted string starting so agrees on (to the
        closing quote when they all end alike)."""
        r = None
        for p in self._find(h, cur, 0, hi)[:64]:
            p = int(p)
            if p == here or not self._quoted_at(h, p):
                continue
            toks, closed = self._read_str(h, p + len(cur), hi, n)
            c = toks + ([self.close] if closed else [])
            if r is None:
                r = c
                continue
            k = 0
            while k < len(r) and k < len(c) and r[k] == c[k]:
                k += 1
            if k < len(r):
                r = r[:k]
                if k < JC_PFX_AGREE:
                    return None
        return (r, "pfx", None) if r else None

    def _ok(self, key, kind) -> bool:
        ok, n = self.stats.get((key, kind), (0, 0))
        return n < 2 or ok * 2 >= n

    def note(self, a: int):
        """After the verify pass accepted a drafts: which value guesses were right (their first token)."""
        for start, key, kind in self.segs:
            if a < start:
                break
            st = self.stats.setdefault((key, kind), [0, 0])
            st[1] += 1
            st[0] += int(a > start)
        self.segs = []

    # -------------------------------------------------------------------------------- the proposal
    def propose(self, hist, h, nmax: int, ng, wide: bool = False):
        """-> drafts (list, up to nmax) or None (no JSON proposal here: the plain copy rule decides). ng() -> (where the
        plain copy match's continuation starts in hist, the match length) or None."""
        if not self.active():
            return None
        lx = self.lx.copy()
        L0 = len(hist)
        prop = []
        segs = []
        cap = [nmax]

        def seq_at(i):
            return hist[i] if i < L0 else prop[i - L0]

        lab = []

        def push(toks, why):
            for t in toks:
                if len(prop) >= cap[0]:
                    return False
                i = L0 + len(prop)
                prop.append(int(t))
                lab.append(why)
                lx.feed(self.s(int(t)), i, _Seq(hist, prop))
                if lx.dead or lx.done:
                    return False
            return True

        for _ in range(12):
            if len(prop) >= cap[0] or lx.dead or lx.done:
                break
            pos = L0 + len(prop)
            if lx.ins:
                m = pos - (lx.o + 1) if lx.al else -1
                src = None
                if m == 0 and not lx.key and JC_VALUE:
                    st = lx.st[-1]
                    key = st[2] if st[0] == "{" else None
                    prev = None
                    if st[3] is not None:
                        prev = [seq_at(i) for i in range(st[3][0], st[3][1])]
                    elif len(lx.st) >= 2 and st[0] == "{" and lx.st[-2][0] == "{" and lx.st[-2][2] is not None:
                        prev = list(lx.st[-2][2])          # the first value of an object whose key names it
                    if prev:
                        if self._ok(key, "kv"):
                            v = self.kv_value(h, prev, L0)
                            if v is not None:
                                src = (v + [self.close], "kv", key)
                        if src is None and st[3] is not None and self._ok(key, "rep"):
                            src = (prev, "rep", key)
                if src is None and m >= 1 and JC_PFX and m >= JC_PFX:
                    src = self._complete(h, [seq_at(i) for i in range(lx.o + 1, pos)], lx.o + 1, L0, cap[0] - len(prop))
                if src is None and not prop and ng is not None:
                    r = ng()
                    if r is not None:
                        j, ml = r
                        toks, closed = self._read_str(h, j, L0, nmax)
                        if not closed and len(toks) < nmax:
                            return None             # the source runs into the end of the sequence: the plain proposal
                        if not ((wide and m >= JC_WIDE_M) or min(ml, m) >= JC_NG_SURE):     # the match inside this string decides
                            cap[0] = min(cap[0], NB - 1)
                        src = ((toks + [self.close]) if (closed and JC_CUT) else toks, None, None)
                        if not JC_CUT:
                            return None
                if src is None or not src[0]:
                    break
                if src[1] is not None:
                    segs.append((len(prop), src[2], src[1]))
                    if self.stats.get((src[2], src[1]), (0, 0))[0] == 0:
                        cap[0] = min(cap[0], len(prop) + JC_TRY)      # an unproven guess: few rows
                if not push(src[0], src[1] or "ng") or src[1] == "rep":
                    break
                continue
            r = None
            for c in ((lx.lc, lx.lk) if JC_STRUCT else ()):
                if c is None:
                    continue
                st = self.struct_after(h, c[0], L0, len(lx.st))
                done = [seq_at(i) for i in range(c[1], pos)]
                if st and len(done) < len(st) and list(st[:len(done)]) == done:
                    r = st[len(done):]
                    break
            if not r or not push(r, "st"):
                break
        if not prop:
            return None
        self.segs = segs
        self.lab = lab
        return prop


class _Seq:
    """hist + prop as one indexable sequence (slices for _Lex key tuples)."""
    __slots__ = ("a", "b", "n")

    def __init__(self, a, b):
        self.a, self.b, self.n = a, b, len(a)

    def __getitem__(self, k):
        if isinstance(k, slice):
            return [self[i] for i in range(*k.indices(self.n + len(self.b)))]
        return self.a[k] if k < self.n else self.b[k - self.n]

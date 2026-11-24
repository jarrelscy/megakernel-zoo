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
import numpy as np

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


# The draft's attention as one kernel per layer: a (row, head) threadgroup scores the n committed context positions and
# the block's own 8 rows, then sums the values. It reads only the positions the block can see (MLX's matmuls over the
# whole context array with a mask cost the same at any position). Draft only: no sum here has to match anything.
FAST_ATT = os.environ.get("MK_D2_ATT", "1") == "1"
FAST_TOP = os.environ.get("MK_D2_TOP", "1") == "1"
_DATT = """
    uint tid = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    uint grp = threadgroup_position_in_grid.x;
    int head = grp % 24, kvh = head / 3;
    int cpos = meta[0], dbase = meta[2];
    int nb = RING ? TT : min(dbase, TT);        // slots of the big tier that are looked at (ring: all of them)
    int nr = cpos - dbase;                      // rows of the recent tier in sight: positions dbase .. cpos - 1
    int n = nb + nr;
    int tot = n + NBK;
    int lowest = max(meta[1], cpos - (TT - 16));
    threadgroup float sc[SMAX];
    threadgroup float red[8];
    threadgroup float4 part[256];
    const device float* qp = q + grp * 128;
    float lmax = -INFINITY;
    for (int s = int(tid); s < tot; s += 256) {
        if (RING && s < nb) {                   // slot s holds the largest position below dbase that is s mod TT
            int c1 = dbase - 1;
            int pj = c1 - (((c1 - s) % TT + TT) % TT);
            if (pj < lowest || pj < 0) { sc[s] = -INFINITY; continue; }
        }
        const device float* kp = (s < nb ? kc + (size_t(s) * 8 + kvh) * 128 : (s < n ? kr + (size_t(s - nb) * 8 + kvh) * 128
                                                                                  : kb + (size_t(s - n) * 8 + kvh) * 128));
        float4 a = float4(0.0f);
        for (int j = 0; j < 32; j++) a = fma(*((const device float4*)(kp + 4 * j)), *((const device float4*)(qp + 4 * j)), a);
        float v = ((a.x + a.y) + (a.z + a.w)) * 0.08838834764831845f;
        sc[s] = v;
        lmax = max(lmax, v);
    }
    float m = simd_max(lmax);
    if (lane == 0) red[sg] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    m = red[0];
    for (int i = 1; i < 8; i++) m = max(m, red[i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float lsum = 0.0f;
    for (int s = int(tid); s < tot; s += 256) {
        float pr = exp(sc[s] - m);
        sc[s] = pr;
        lsum += pr;
    }
    float ss = simd_sum(lsum);
    if (lane == 0) red[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (int i = 0; i < 8; i++) total += red[i];
    int c = tid / 32;
    int d4 = (tid % 32) * 4;
    float4 acc = float4(0.0f);
    for (int s = c; s < tot; s += 8) {
        const device float* vp = (s < nb ? vc + (size_t(s) * 8 + kvh) * 128 : (s < n ? vr + (size_t(s - nb) * 8 + kvh) * 128
                                                                                  : vb + (size_t(s - n) * 8 + kvh) * 128)) + d4;
        acc = fma(*((const device float4*)vp), float4(sc[s]), acc);
    }
    part[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 32) {
        float4 o = part[tid];
        for (int i = 1; i < 8; i++) o += part[i * 32 + tid];
        *((device float4*)(out + grp * 128 + d4)) = o / total;
    }
"""
_datt = mx.fast.metal_kernel(name="mk_draft_att", input_names=["q", "kc", "vc", "kr", "vr", "kb", "vb", "meta"], output_names=["out"], source=_DATT)
DREC = 128      # rows of the draft context's recent tier


def draft_att(q, kc, vc, kr, vr, kb, vb, meta, ring: bool):
    """q float32 [NB, 24, 128]; kc / vc float32 [T, 8, 128]: the context below position dbase (by position, or the
    ring's slots); kr / vr float32 [DREC, 8, 128]: positions dbase ..; kb / vb float32 [NB, 8, 128]: the block's own
    rows; meta int32 [3] = (cpos, lowest position kept, dbase) -> float32 [NB, 3072]."""
    nb, T = q.shape[0], kc.shape[0]
    return _datt(inputs=[q, kc, vc, kr, vr, kb, vb, meta], template=[("SMAX", T + kr.shape[0] + nb), ("TT", T), ("NBK", nb), ("RING", int(ring))],
                 grid=(nb * 24 * 256, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(nb, 3072)], output_dtypes=[mx.float32])[0]


def top_cands(base, k: int):
    """base float32 [n, V] -> (indices int32 [n, k], values [n, k]) of each row's k largest (in no order): the k
    chunks of 256 with the largest maxima hold them, so only those are partitioned."""
    n, V = base.shape
    C = 256
    nc = -(-V // C)
    b = mx.concatenate([base, mx.full((n, nc * C - V), -mx.inf, dtype=base.dtype)], axis=1).reshape(n, nc, C)
    ci = mx.argpartition(-b.max(axis=-1), k - 1, axis=-1)[:, :k]                               # [n, k] chunks
    flat = mx.take_along_axis(b, ci[:, :, None], axis=1).reshape(n, k * C)
    i2 = mx.argpartition(-flat, k - 1, axis=-1)[:, :k]
    idx = mx.take_along_axis(ci, i2 // C, axis=1) * C + i2 % C
    return idx.astype(mx.int32), mx.take_along_axis(flat, i2, axis=-1)


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


CALIB_LOG = None


class CalR:
    """Online calibration of the early readout for one request: P(the readout's token is the target's | the readout's
    own probability of it, in buckets), counted with a decay from a prior. fine: more buckets near 1, where most
    tokens are and where a chain's chance is the product of many factors."""
    EDGES = np.array([0.4, 0.6, 0.8, 0.9, 0.97, 0.995])
    PRIOR = np.array([0.35, 0.55, 0.75, 0.87, 0.94, 0.98, 0.995])
    EDGES_F = np.array([0.3, 0.5, 0.65, 0.8, 0.9, 0.95, 0.98, 0.99, 0.997, 0.999, 0.9999])
    PRIOR_F = np.array([0.3, 0.45, 0.6, 0.75, 0.86, 0.91, 0.95, 0.97, 0.985, 0.993, 0.997, 0.9995])

    def __init__(self, w0: float = 6.0, decay: float = 0.98, fine: bool = False):
        self.edges, prior = (self.EDGES_F, self.PRIOR_F) if fine else (self.EDGES, self.PRIOR)
        self.n = np.full(len(prior), w0)
        self.a = prior * w0
        self.decay = decay

    def q(self, p: float) -> float:
        b = int(np.searchsorted(self.edges, p, side="right"))
        return float(self.a[b] / self.n[b])

    def upd(self, p: float, ok: bool) -> None:
        b = int(np.searchsorted(self.edges, p, side="right"))
        self.n[b] = self.n[b] * self.decay + 1.0
        self.a[b] = self.a[b] * self.decay + (1.0 if ok else 0.0)


class Draft2(md.Draft):
    NL = 5

    def __init__(self, draft_dir: str, tg):
        d = Path(draft_dir)
        cfg = json.loads((d / "config.json").read_text())
        lt = cfg["loopedtf_drafter"]
        assert cfg["num_hidden_layers"] == self.NL and cfg["block_size"] == NB and cfg["hidden_size"] == 2304
        assert cfg["num_attention_heads"] == 24 and cfg["num_key_value_heads"] == 8 and cfg["head_dim"] == 128
        assert lt["dspark"]["markov_head_type"] == "vanilla" and lt["dspark"]["markov_rank"] == 256
        assert tuple(tg.taps) == last_loop_taps(tg.tc, lt["feature_layers_per_loop"]) or os.environ.get("MK_TAPS"), (tg.taps, lt)
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
        self.review_r = int(os.environ.get("MK_REVIEW", "0"))
        self.review_thr = float(os.environ.get("MK_REVIEW_THR", "0.3"))
        # rounds: stage A (prefix + loop 1, with an early readout that reviews the drafts and corrects the first one it
        # rejects) on a few rows at a time, the 2-D draft after each, then one stage B (loops 2 .. 5, suffix) over the rows kept
        self.rounds = int(os.environ.get("MK_ROUNDS", "12"))
        self.max_rows = int(os.environ.get("MK_MAXROWS", "12"))
        self.round_policy = os.environ.get("MK_ROUND_POLICY", "cost")
        # ms: a round's draft + stage A of one row + syncs; a further stage-A row; a kept row in stage B
        self.c_a1 = float(os.environ.get("MK_C_A1", "7.0"))
        self.c_draft = float(os.environ.get("MK_C_DRAFT", "3.0"))
        self.m_kd = int(os.environ.get("MK_M_KD", "7"))       # rounds when the draft is worth fewer rows than this
        self.c_arow = float(os.environ.get("MK_C_AROW", "1.0"))
        self.c_brow = float(os.environ.get("MK_C_BROW", "2.6"))
        self.round_margin = float(os.environ.get("MK_ROUND_MARGIN", "1.0"))
        self.row_margin = float(os.environ.get("MK_ROW_MARGIN", "1.0"))
        self.draft_policy = os.environ.get("MK_DRAFT_POLICY", "cost")
        self.draft_probe = int(os.environ.get("MK_DRAFT_PROBE", "8"))
        self.copy_review = os.environ.get("MK_COPY_REVIEW", "1") == "1"
        self.rounds_samp = os.environ.get("MK_ROUNDS_SAMP", "0") == "1"       # rounds for sampling requests: exact, 1-3% slower (journal)
        self.calr_fine = os.environ.get("MK_CALR_FINE", "0") == "1"
        self.calr_decay = float(os.environ.get("MK_CALR_DECAY", "0.98"))
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
        self.rposi = mx.arange(DREC).astype(mx.int32)
        self._cyc, self._sfn, self._fn = {}, {}, None
        del w
        self.reset()

    tiers = True

    def reset(self):
        # the context K/V in two tiers, as the target's: positions < dbase in the big arrays (by position, or the
        # ring's slots), positions dbase .. in a recent tier of DREC rows. A pass appends its rows to the recent tier
        # (the traced passes rewrite the arrays they append to: 0.5 MB each instead of 17 MB); the list is
        # [big x NL] + [recent x NL] wherever the context is passed around.
        self.kc = [mx.zeros((self.T, 8, 128), dtype=F32) for _ in range(self.NL)] + [mx.zeros((DREC, 8, 128), dtype=F32) for _ in range(self.NL)]
        self.vc = [mx.zeros((self.T, 8, 128), dtype=F32) for _ in range(self.NL)] + [mx.zeros((DREC, 8, 128), dtype=F32) for _ in range(self.NL)]
        self.dbase = 0
        self.frow = None
        self.hist = []
        self.low = 0

    def _cp(self, pos: int):
        return mx.array([float(pos), float(self.low), float(self.dbase)], dtype=F32)

    def flush(self, pos: int):
        """Move the recent tier's rows of positions dbase .. pos - 1 (committed) into the big tier."""
        n = pos - self.dbase
        # (more than DREC positions behind: they were never appended, e.g. tokens fed through step(), which keeps no
        # draft context; the draft then simply lacks them, as it always did)
        if 0 < n <= DREC:
            NL = self.NL
            idx = (self.dbase + mx.arange(n)) % self.TD if self.ring else None
            for l in range(NL):
                if self.ring:
                    self.kc[l][idx] = self.kc[NL + l][:n]
                    self.vc[l][idx] = self.vc[NL + l][:n]
                else:
                    self.kc[l][self.dbase:pos] = self.kc[NL + l][:n]
                    self.vc[l][self.dbase:pos] = self.vc[NL + l][:n]
        self.dbase = max(self.dbase, pos)

    def maybe_flush(self):
        if self.tg.pos - self.dbase >= DREC // 2:
            self.flush(self.tg.pos)

    def rollback(self, old_pos: int):
        super().rollback(old_pos)
        self.dbase = min(self.dbase, self.tg.pos)       # the big tier is valid below the cut; the recent one restarts there

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
        """Host-side append of a pass's first n rows at positions pos .. (prefill): straight into the big tier."""
        self.flush(pos)
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
        self.dbase = pos + n
        self.frow = [x[n - 1:n] for x in feats]

    def _append_pure(self, feats, cposf, kc, vc):
        """The rows of a pass at positions cposf[0] .. into the recent tier (cposf = (position, low, dbase))."""
        K = feats[0].shape[0]
        NL = self.NL
        c = cposf[:1]
        rel = self.rposi - (c - cposf[2:3]).astype(mx.int32)
        ps = c + mx.arange(K).astype(F32)
        mask = ((rel >= 0) & (rel < K))[:, None, None]
        rows = mx.clip(rel, 0, K - 1)
        ctx = self._ctx(feats)
        kc2, vc2 = list(kc[:NL]), list(vc[:NL])
        for l in range(NL):
            k, v = self._ctx_kv(ctx, ps, l)
            kc2.append(mx.where(mask, k[rows], kc[NL + l]))
            vc2.append(mx.where(mask, v[rows], vc[NL + l]))
        return kc2, vc2

    # ---------------------------------------------------------------------------------------------- the block
    def _pure(self, anchor, cpos, kc, vc, frow, samp=None):
        tg = self.tg
        nd = NB - 1
        e = tg.embed[anchor].astype(F32)
        if ANCHOR == "norm":
            e = rms(e, tg.embed_norm)
        x = mx.concatenate([e, mx.broadcast_to(self.mask_embed[None], (nd, 2304))])
        ameta = cpos.astype(mx.int32)            # (position of the anchor, lowest position kept, dbase)
        cpos = cpos[:1]
        pos = cpos + mx.arange(NB).astype(F32)
        NL = self.NL
        for l in range(self.NL):
            w = self.L[l]
            h = rms(x, w["in_ln"])
            qkv = self.gm(w["qkv"], kn.bits(h))                                                     # [8, 5120]
            q = self._rope(rms(qkv[:, :3072].reshape(NB, 24, 128), w["q_norm"]), pos)
            k = self._rope(rms(qkv[:, 3072:4096].reshape(NB, 8, 128), w["k_norm"]), pos)
            v = qkv[:, 4096:].reshape(NB, 8, 128)
            a = draft_att(q, kc[l], vc[l], kc[NL + l], vc[NL + l], k, v, ameta, self.ring)
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
            if FAST_TOP:
                ctx, base = top_cands(base, TOPC)
            else:
                ctx = mx.argpartition(-base, TOPC - 1, axis=-1)[:, :TOPC].astype(mx.int32)          # [nd, TOPC] candidates
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

    # ---------------------------------------------------------------------------------------------- rounds
    def _next_draft(self, tok, tokens, cposf, dkc, dvc, samp=None):
        """Behind a pass on the GPU: the number of leading rows whose token is the next row's input, and the draft block
        anchored at the token of the last of them."""
        K = tokens.shape[0]
        if K > 1:
            a = mx.cumprod((tok[:K - 1] == tokens[1:]).astype(mx.int32)).sum().reshape(1)
        else:
            a = mx.zeros((1,), dtype=mx.int32)
        anchor = mx.take(tok, a, axis=0).astype(mx.int32)
        nxt = cposf[:1] + (a + 1).astype(F32)
        out = self._pure(anchor, mx.concatenate([nxt, cposf[1:]]), dkc, dvc, None, samp=samp)
        if samp is None:
            return out
        return out + (nxt.astype(mx.int32) + 1 + self.n1,)       # and the position of the next draft to draw (more())

    def _roundA_pure(self, tokens, m_full, m_loc, kc, vc, kr, vr, cwa, dkc, dvc, cposf):
        e, pe, feats, carry, ka, va, wa = self.tg._stage_a(tokens, m_full, m_loc, kc, vc, kr, vr, cwa)
        dkc2, dvc2 = self._append_pure(feats, cposf, dkc, dvc)          # in-flight rows: loop-1 features (the 2-D form)
        d, p, state, prev, ctx, base = self._next_draft(e, tokens, cposf, dkc2, dvc2)
        return e, pe, carry, ka, va, wa, dkc2, dvc2, d, p, state, prev, ctx, base

    def _stageB_pure(self, tokens, carry, m_full, m_loc, kc, vc, kr, vr, cw, dkc, dvc, cposf):
        tgt, feats, kb, vb, wb = self.tg._stage_b(carry, m_full, m_loc, kc, vc, kr, vr, cw)
        dkc2, dvc2 = self._append_pure(feats, cposf, dkc, dvc)          # committed rows: last-loop features
        d, p, state, prev, ctx, base = self._next_draft(tgt, tokens, cposf, dkc2, dvc2)
        return tgt, feats, kb, vb, wb, dkc2, dvc2, d, p, state, prev, ctx, base

    def _stageBn_pure(self, tokens, carry, m_full, m_loc, kc, vc, kr, vr, cw, dkc, dvc, cposf):
        """Stage B with no draft behind it (the next cycle starts with a readout round of the anchor alone)."""
        tgt, feats, kb, vb, wb = self.tg._stage_b(carry, m_full, m_loc, kc, vc, kr, vr, cw)
        dkc2, dvc2 = self._append_pure(feats, cposf, dkc, dvc)
        return tgt, feats, kb, vb, wb, dkc2, dvc2

    def _samp_round_fns(self, filt: bool):
        """The traced passes of a sampling request (round, stage B with and without a draft): the readout, the rows'
        tokens and the drafts are drawn with the request's noise; the last output of the first two is the position of
        the next draft more() draws."""
        def ra(tokens, m_full, m_loc, kc, vc, kr, vr, cwa, dkc, dvc, cposf, sp, seed):
            e, pe, feats, carry, ka, va, wa = self.tg._stage_a(tokens, m_full, m_loc, kc, vc, kr, vr, cwa, samp=(sp, seed))
            dkc2, dvc2 = self._append_pure(feats, cposf, dkc, dvc)
            d, p, state, prev, ctx, base, q = self._next_draft(e, tokens, cposf, dkc2, dvc2, samp=(sp, seed, filt))
            return e, pe, carry, ka, va, wa, dkc2, dvc2, d, p, state, prev, ctx, base, q

        def rb(tokens, carry, m_full, m_loc, kc, vc, kr, vr, cw, dkc, dvc, cposf, sp, seed):
            tgt, feats, kb, vb, wb = self.tg._stage_b(carry, m_full, m_loc, kc, vc, kr, vr, cw, samp=(sp, seed, filt))
            dkc2, dvc2 = self._append_pure(feats, cposf, dkc, dvc)
            d, p, state, prev, ctx, base, q = self._next_draft(tgt, tokens, cposf, dkc2, dvc2, samp=(sp, seed, filt))
            return tgt, feats, kb, vb, wb, dkc2, dvc2, d, p, state, prev, ctx, base, q

        def rbn(tokens, carry, m_full, m_loc, kc, vc, kr, vr, cw, dkc, dvc, cposf, sp, seed):
            tgt, feats, kb, vb, wb = self.tg._stage_b(carry, m_full, m_loc, kc, vc, kr, vr, cw, samp=(sp, seed, filt))
            dkc2, dvc2 = self._append_pure(feats, cposf, dkc, dvc)
            return tgt, feats, kb, vb, wb, dkc2, dvc2
        return ra, rb, rbn

    def _run_a(self, toks, pr: int, cwa, dkc, dvc, samp=None):
        tg = self.tg
        K = toks.shape[0]
        assert pr + K <= tg.T and pr - tg.base + K <= tg.RECENT
        key = ("A", K) if samp is None else ("A", K, samp.filt)
        fn = self._cyc.get(key)
        if fn is None:
            fn = self._cyc[key] = mx.compile(self._roundA_pure if samp is None else self._samp_round_fns(samp.filt)[0])
        return fn(toks, mx.array([pr, 0, tg.base], dtype=mx.int32), mx.array([pr, 1024, tg.base], dtype=mx.int32),
                  tg.kc, tg.vc, tg.kr, tg.vr, cwa, dkc, dvc, self._cp(pr), *(() if samp is None else (samp.sp, samp.seed)))

    def _run_b(self, toks, carry, dkc, dvc, draft: bool = True, samp=None):
        tg = self.tg
        K = toks.shape[0]
        assert tg.pos + K <= tg.T and tg.pos - tg.base + K <= tg.RECENT
        key = ("B", K, draft) if samp is None else ("B", K, draft, samp.filt)
        fn = self._cyc.get(key)
        if fn is None:
            if samp is None:
                fn = mx.compile(self._stageB_pure if draft else self._stageBn_pure)
            else:
                fn = mx.compile(self._samp_round_fns(samp.filt)[1 if draft else 2])
            self._cyc[key] = fn
        return fn(toks, carry, mx.array([tg.pos, 0, tg.base], dtype=mx.int32), mx.array([tg.pos, 1024, tg.base], dtype=mx.int32),
                  tg.kc, tg.vc, tg.kr, tg.vr, tg.cw, dkc, dvc, self._cp(tg.pos), *(() if samp is None else (samp.sp, samp.seed)))

    def _more_rounds(self, act, L: int, r: int, maxrows: int, pl: float) -> int:
        """Another round after r rounds and L kept rows? 0: no; 1: the early model's token alone; 2: with the drafts
        the 2-D draft makes after it. pl: the calibrated probability that the round's first row is accepted (every
        token up to it right). A round pays when its tokens per ms beat the request's recent rate."""
        if r >= self.rounds or L >= maxrows:
            return 0
        if self.round_policy == "fixed":
            return 2
        lam = act["ema_n"] / (1e3 * act["ema_t"]) if act.get("ema_t", 0.0) > 0.0 else 0.07
        rate1 = pl / (self.c_a1 + self.c_brow)
        al = min(0.995, max(0.5, act.get("al_a", 9.0) / act.get("al_n", 10.0)))
        n = min(1.0 + act.get("ema_kept", 1.5), float(maxrows - L))
        rate2 = pl * (1.0 - al ** n) / (1.0 - al) / (self.c_draft + self.c_a1 + self.c_arow * act.get("ema_kd", 1.5) + self.c_brow * n)
        if self.round_policy == "nodraft":
            rate2 = 0.0
        if max(rate1, rate2) < self.round_margin * lam:
            return 0
        return 2 if rate2 > rate1 else 1

    def _cycle_m(self, eng, act, left: int, dp, t0: float, syncs: int = 0, copy: bool = False, samp=None):
        """One cycle in rounds. Round: stage A of [a token, drafts after it]; its readout confirms a prefix of the
        drafts (kept) and gives the early model's token after it, which starts the next round (alone, or with the
        drafts the 2-D draft makes behind it). Then stage B over every kept row: the rows' exact tokens decide, as in
        cycle(). Every token after the first is the readout's choice after the row before it; the readout's
        probability of it, calibrated online against the verify's verdicts (CalR), says how far the chain is worth
        extending and verifying."""
        import time
        tg = self.tg
        c = tg.pos
        nd = NB - 1
        cnt = eng.counters
        cal = act["cal"]
        calr = act.get("calr")
        if calr is None:
            calr = act["calr"] = CalR(fine=self.calr_fine, decay=self.calr_decay)
        d, p = dp if dp is not None else (None, None)
        n0, nsuf = tg.core.start + len(tg.core), len(tg.layers) - tg.core.stop
        s0 = tg.napp - nsuf
        cwa = tg.cwa()
        dkc, dvc = self.kc, self.vc
        chain, cl, kl, vl, wl = [], [], [], [], []
        pq = []                     # pq[i]: the readout's probability of the token after chain row i
        first = int(act["last"])
        pr = c
        maxrows = min(self.max_rows, left)
        lam = act["ema_n"] / (1e3 * act["ema_t"]) if act.get("ema_t", 0.0) > 0.0 else 0.07
        r = 0
        copy_all = False
        mode = 2 if dp is not None else 1
        pl = 1.0                    # calibrated probability that the next row to add is accepted
        while True:
            room = maxrows - len(chain) - 1
            kd = 0
            if copy and r == 0:
                kd = max(0, min(len(d), room, nd))          # copy tokens: all of them are reviewed
            elif mode == 2:
                kd, prb = cal.depth(p, self.review_thr)
                if kd == len(p) and len(p) < nd and room > kd:
                    d2, p2 = self.more()
                    mx.eval(d2, p2)
                    syncs += 1
                    p2 = np.array(p2)
                    kd += cal.depth(p2, self.review_thr, len(p), prb)[0]
                    d, p = np.concatenate([d, np.array(d2)]), np.concatenate([p, p2])
                    cnt["draft_more"] = cnt.get("draft_more", 0) + 1
                kd = max(0, min(kd, room, nd))
            rows = [first] + ([int(v) for v in d[:kd]] if kd else [])
            dkc0, dvc0 = dkc, dvc
            out = self._run_a(mx.array(np.asarray(rows, dtype=np.int32)), pr, cwa, dkc, dvc, samp)
            e_l, pe_l, carry, ka, va, wa, dkc, dvc, dn, pn, state, prev, ctx, bse = out[:14]
            ta = time.perf_counter()
            mx.eval(e_l, pe_l, carry, ka, va, *wa)
            cnt["t_a"] = cnt.get("t_a", 0.0) + time.perf_counter() - ta
            syncs += 1
            e, pe = np.array(e_l), np.array(pe_l)
            m = 0
            while m < kd and int(e[m]) == rows[m + 1]:
                m += 1
            if not (copy and r == 0):
                for j in range(min(m + 1, kd)):
                    cal.upd(j, float(p[j]), j < m)
            copy_kept = copy and r == 0 and kd == nd and m == kd       # the readout confirms the whole copy
            n = m + 1
            # rows worth a place in stage B: while the chance that the row is accepted pays for it
            keep = 0
            for j in range(n):
                if len(chain) + j > 0 and pl < self.c_brow * lam * self.row_margin:
                    break
                keep = j + 1
                pl *= calr.q(float(pe[j]))
            stop = keep < n
            n = keep
            if n > 0:
                chain += rows[:n]
                pq += [float(v) for v in pe[:n]]
                cl.append(carry[:n])
                kl.append(ka[:n0, :n])
                vl.append(va[:n0, :n])
                wl.append([w[:n0, :n + 3] if not wl else w[:n0, 3:n + 3] for w in wa])
            r += 1
            if mode == 2 and not (copy and r == 1):
                act["ema_kept"] = act.get("ema_kept", 1.5) * 0.9 + 0.1 * m
                act["ema_kd"] = act.get("ema_kd", 1.5) * 0.9 + 0.1 * kd
            cnt["a_rows"] = cnt.get("a_rows", 0) + len(rows)
            cnt["a_rounds"] = cnt.get("a_rounds", 0) + 1
            mode = 0 if (stop or (copy and r == 1 and copy_kept)) else self._more_rounds(act, len(chain), r, maxrows, pl)
            if copy and r == 1:
                copy_all = copy_kept and not stop
            if mode != 2:
                dkc, dvc = dkc0, dvc0       # this round's draft and its context rows are not computed
            if mode == 0:
                break
            # the kept rows' stage-A K/V and windows, where the next round reads them (the suffix ones are the readout's:
            # provisional, rewritten by stage B's at the commit; rows past tg.pos are read by rounds only)
            s = pr - tg.base
            tg.kr[0:n0, s:s + n] = ka[:n0, :n]
            tg.kr[s0:, s:s + n] = ka[n0:, :n]
            tg.vr[0:n0, s:s + n] = va[:n0, :n]
            tg.vr[s0:, s:s + n] = va[n0:, :n]
            cwa = [w[:, n:n + 3] for w in wa]
            pr += n
            first = int(e[m])
            if mode == 2:
                ta = time.perf_counter()
                mx.eval(dn, pn, state, prev, ctx, bse, dkc, dvc, *out[14:])
                cnt["t_d"] = cnt.get("t_d", 0.0) + time.perf_counter() - ta
                syncs += 1
                d, p = np.array(dn), np.array(pn)
                if samp is None:
                    self._rest, self._rest_fn = (ctx, bse, state, prev), self._fn2
                else:
                    self._rest, self._rest_fn = (ctx, bse, state, prev, samp.sp, samp.seed, out[14]), self._samp_fns(samp.filt)[1]
                cnt["a_drafts"] = cnt.get("a_drafts", 0) + 1
        t1 = time.perf_counter()
        K = len(chain)
        toks = mx.array(np.asarray(chain, dtype=np.int32))
        # a draft behind stage B (for the next cycle's first round) while a draft call keeps enough rows to pay for
        # itself (a kept draft saves its token a round); when off, tried again every few cycles
        nc = act["m_n"] = act.get("m_n", 0) + 1
        want = (self.draft_policy == "always" or left - K <= 0 or nc % self.draft_probe == 0
                or act.get("ema_kept", 1.5) * self.c_a1 > self.c_draft + self.c_arow * act.get("ema_kd", 1.5))
        if copy and copy_all:
            want = False            # the next cycle is most likely a copy pass again
        out = self._run_b(toks, cl[0] if len(cl) == 1 else mx.concatenate(cl), dkc, dvc, want, samp)
        if want:
            tgt, feats, kb, vb, wb, dkc2, dvc2, dn, pn, state, prev, ctx, bse = out[:13]
            ev = [dn, pn, state, prev, ctx, bse] + list(out[13:])
        else:
            tgt, feats, kb, vb, wb, dkc2, dvc2 = out
            ev = []
            cnt["b_nodraft"] = cnt.get("b_nodraft", 0) + 1
        cat1 = lambda xs: xs[0] if len(xs) == 1 else mx.concatenate(xs, axis=1)  # noqa: E731
        tg.pending = (mx.concatenate([cat1(kl), kb]), mx.concatenate([cat1(vl), vb]),
                      [mx.concatenate([cat1([w[j] for w in wl]), wb[j]]) for j in range(4)], K)
        ta = time.perf_counter()
        mx.eval(tgt, *ev)
        cnt["t_b"] = cnt.get("t_b", 0.0) + time.perf_counter() - ta
        cnt["t_m"] = cnt.get("t_m", 0.0) + time.perf_counter() - t0
        syncs += 1
        act["t_sync"] = time.perf_counter()
        t = np.array(tgt)
        a = 0
        while a < K - 1 and chain[a + 1] == int(t[a]):
            a += 1
        n_acc = a + 1
        if CALIB_LOG is not None:                   # (tests: predicted chance of each row against the verdict)
            pp = 1.0
            for i in range(K - 1):
                pp *= calr.q(pq[i])
                CALIB_LOG.append((i + 1, pp, i < a))
        for i in range(min(a + 1, K - 1)):          # the tokens the verify judged: right up to a, wrong at a + 1
            calr.upd(pq[i], i < a)
        dt = time.perf_counter() - t0
        act["ema_t"] = act.get("ema_t", 0.0) * self.ema + dt
        act["ema_n"] = act.get("ema_n", 0.0) * self.ema + n_acc
        act["al_n"] = act.get("al_n", 10.0) * 0.97 + (K - 1 if a == K - 1 else a + 1)    # successor rows judged
        act["al_a"] = act.get("al_a", 9.0) * 0.97 + a                                    # and accepted
        self.kc, self.vc = dkc2, dvc2
        self.frow = [f[a:a + 1] for f in feats]
        tg.commit(n_acc)
        mx.async_eval(self.kc, self.vc, *tg.state())
        if want:
            act["draft"] = (np.array(dn), np.array(pn))
            if samp is None:
                self._rest, self._rest_fn = (ctx, bse, state, prev), self._fn2
            else:
                self._rest, self._rest_fn = (ctx, bse, state, prev, samp.sp, samp.seed, out[13]), self._samp_fns(samp.filt)[1]
        else:
            act["nodraft"] = True
        cnt["rows"] += K
        cnt["draft_s"] = cnt.get("draft_s", 0.0) + (t1 - t0)
        cnt["verify_s"] = cnt.get("verify_s", 0.0) + (time.perf_counter() - t1)
        cnt["host_syncs"] += syncs
        cnt["m_cycles"] = cnt.get("m_cycles", 0) + 1
        if K == 1:
            cnt["m_k1"] = cnt.get("m_k1", 0) + 1
            if act.get("prev_k1"):
                cnt["m_k1k1"] = cnt.get("m_k1k1", 0) + 1
        act["prev_k1"] = K == 1
        cnt["m_sent"] = cnt.get("m_sent", 0) + K - 1
        cnt["m_acc"] = cnt.get("m_acc", 0) + a
        # a copy the readout confirmed in full and the verify accepted: the next copy cycle is a 16-row pass
        act["ng_full"] = bool(copy and copy_all and a == K - 1 and K >= NB)
        if copy:
            cnt["ngram_rows"] = cnt.get("ngram_rows", 0) + K
            cnt["ngram_tokens"] = cnt.get("ngram_tokens", 0) + n_acc
            cnt["ngram_full"] = cnt.get("ngram_full", 0) + int(act["ng_full"])
        res = [int(v) for v in t[:n_acc]]
        act["hist"].extend(res)
        return res

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
        if self.review_r > 0:
            for K in range(1, NB):
                mx.eval(tg.early(mx.zeros((K,), dtype=mx.int32), self.review_r)[0])
        if self.rounds > 0:
            import mk_target as mt
            if mt.SL_C > 0:
                try:
                    tg.fit_shortlist()
                except Exception:           # the readout then scans the 8-bit unembedding, as before
                    tg.sl_w = None
            tg.reset()
            carry = None
            for K in range(1, NB + 1):
                out = self._run_a(mx.zeros((K,), dtype=mx.int32), 0, tg.cwa(), self.kc, self.vc)
                mx.eval(*[o for o in out])
                carry = out[2]
            for K in range(1, self.max_rows + 1):
                cr = mx.concatenate([carry] * 2)[:K]
                out = self._run_b(mx.zeros((K,), dtype=mx.int32), cr, self.kc, self.vc)
                mx.eval(out[0], *out[2:])
                out = self._run_b(mx.zeros((K,), dtype=mx.int32), cr, self.kc, self.vc, False)
                mx.eval(out[0], *out[2:])
        tg.reset()

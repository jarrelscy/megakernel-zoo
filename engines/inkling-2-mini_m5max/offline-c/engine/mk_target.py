"""Inkling-2-Mini's text model as one K-row pass (K consecutive positions of one sequence), written on MLX arrays and the
kernels of kernels.py. K = 1 is the decode step; K = 2..8 is the verify pass of track C. The weights stay in the stored
formats (NVFP4 / MXFP8 expert codes and scales as stored, BF16 everything else).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import mlx.core as mx

import mk_kernels as kn

F32 = mx.float32
EPS = 1e-6
import os
ROUND = os.environ.get("MK_ROUND", "0") == "1"
COMPILE = os.environ.get("MK_COMPILE", "1") == "1"
SHORT = os.environ.get("MK_SHORT", "1") == "1"
EXP2 = os.environ.get("MK_EXP2", "1") == "1"
GATE3 = os.environ.get("MK_GATE3", "0") == "1"      # gate stages folded into their neighbours: same bits, no gain or slower (journal)
GATE2 = os.environ.get("MK_GATE2", "0") == "1"      # one fused gate kernel: correct but slower (20.7 vs 19.6 ms at K=2)
MSLOT = [int(v) for v in os.environ.get("MK_M", "1,2,2,2,2,2,2,2").split(",")]     # pairs per expert slot, by rows K
RUP, RDN = int(os.environ.get("MK_RUP", "2")), int(os.environ.get("MK_RDN", "2"))
# weight rows per simdgroup of the down kernel in wide passes (same lane layout per row, so the same bits): 4 rows are
# 0.8 ms faster at 16 rows and 0.2 ms slower at 3
RDN_WIDE, RDN_WIDE_K = int(os.environ.get("MK_RDN_WIDE", "4")), int(os.environ.get("MK_RDN_WIDE_K", "8"))
# the large BF16 matrices of every layer (q | gate | rel | sink, k, v, o) read from a lossless 12-bit re-encoding that
# the kernel decodes to the stored bits (kernels: pack12 / gemv12); 0 reads them as stored
W12 = os.environ.get("MK_W12", "1") == "1"
PROJT = os.environ.get("MK_PROJT", "0") == "1"       # transposed relative-bias table: same bits, no gain (journal)
# passes of 2 .. BR_MAX rows: an expert slot that holds one pair runs the one-pair loop (most slots there; same bits)
BR_MAX = int(os.environ.get("MK_BR_MAX", "3"))
LAYOUT4 = os.environ.get("MK_LAYOUT4", "1") == "1"
PAD = int(os.environ.get("MK_WHATIF_PAD", "0"))   # timing experiment only (0 in the engine)
# independent dispatches of a layer visit in one (kernels: merged dispatches): the q / k / v projections; the three
# message gates' projections. Same bits (tests/merge.py); no measurable gain (a dispatch costs 1.4 us), so off
DENSE3 = os.environ.get("MK_DENSE3", "0") == "1"
GATEM = os.environ.get("MK_GATEM", "0") == "1"
SL_C = int(os.environ.get("MK_SL_C", "4096"))       # tokens whose exact logits a readout computes (0: the 8-bit scan)


WHATIF_NO = os.environ.get("MK_WHATIF_NO", "")     # timing experiment only: "dense" / "experts" replaced by a stand-in
# prompt prefill in wide passes (WIDE_MIN rows and more; decode and verify passes have at most 16):
#   exact: the kernels of the step, GEMVs in row blocks and experts in slots of PF_M pairs (the bits of one-row steps)
#   fast:  MLX GEMMs (half) for the dense matrices and the experts (not the step's bits: relative error about 5e-4)
#   off:   passes of 8 rows, as before
PREFILL = os.environ.get("MK_PREFILL", "exact")
PF_M = int(os.environ.get("MK_PF_M", "8"))
WIDE_MIN = 17
# prefix cache: the K/V arrays are by position, so a later request that starts with the same tokens can keep them; the
# conv windows (the last 3 inputs of every conv) are the only state that is not, so they are kept for every SNAP-th
# position (3.8 MB each) and a sequence can be cut back to any of those (rollback()). 0: no snapshots
SNAP = int(os.environ.get("MK_SNAP", "-1"))        # -1: 64 positions, 256 for a context beyond 8,192 (0.5 GB at 32,768)


def dense(L, nm, x):
    """The BF16 GEMV of layer matrix nm, from its 12-bit code when it has one (the same bits either way); in a fast wide
    pass a half GEMM on a half copy of the matrix."""
    p = L.get(nm + "12")
    if "dense" in WHATIF_NO:
        return kn.fill(x, (p[0].shape[0] if p is not None else L[nm].shape[0]))
    if x.shape[0] >= WIDE_MIN and (nm + "h") in L:
        return mx.matmul(x.view(mx.float16), L[nm + "h"].T).astype(F32)
    return kn.gemv12(p, x) if p is not None else kn.gemv(L[nm], x)


def expert_layout(K: int):
    """(pairs per slot, gate/up row pairs per simdgroup, down rows per simdgroup, pair-count branch) by rows of the pass.
    Every layout gives each (row, expert) pair the same sums (tests/e4.py, tests/loop16.py: equal hashes).
    From 2 rows up: slots of up to 4 pairs, so each distinct expert is read about once (57 slots against 79 of 2 pairs
    in a 16-row loop pass); a branch on the slot's pair count, so a slot computes only the pairs it holds; and one
    gate/up row pair per simdgroup, which keeps the 8 accumulators in registers (with two row pairs, slots of 4 were
    slower than slots of 2). Experts of a pass: 3 rows 8.83 -> 8.35 ms, 8 rows 17.9 -> 16.4, 16 rows 31.3 -> 28.5."""
    if not LAYOUT4 or K < 2:
        return MSLOT[min(K, len(MSLOT)) - 1], RUP, (RDN if K < RDN_WIDE_K else RDN_WIDE), 2 <= K <= BR_MAX
    return 4, 1, (2 if K < 9 else 4), True


# ------------------------------------------------------------------------------------------------- sampling
# A sampled token is a function of its row's logits and of (seed, position) alone: the argmax of logits / T plus Gumbel
# noise that is a hash of (seed, position, token id). So a row of a verify pass draws the token a one-row step draws
# (the logits have the same bits and so has the noise), and exact draft-and-verify works as it does for the greedy
# argmax; the draft head draws with the same noise, which keeps its tokens close to the target's (a shared-noise
# coupling). argmax(z + Gumbel) is a draw from softmax(z); top_p / top_k restrict it to the tokens at or above a
# threshold logit first.
U32 = mx.uint32


def _u32(v):
    return mx.array(v, dtype=U32)


def gumbel(seed, q, n: int):
    """seed uint32 [1], q int32 [K] (positions) -> Gumbel noise float32 [K, n]; 24 uniform bits per value, never 0 or 1."""
    key = ((seed ^ (q.astype(U32) * _u32(0x85EBCA6B))) * _u32(0xC2B2AE35) + _u32(0x27D4EB2F))[:, None]
    x = mx.arange(n, dtype=U32)[None] * _u32(0x9E3779B1) + key
    x = x ^ (x >> 16)
    x = x * _u32(0x85EBCA6B)
    x = x ^ (x >> 13)
    x = x * _u32(0xC2B2AE35)
    x = x ^ (x >> 16)
    u = ((x >> 8).astype(F32) + 0.5) * (1.0 / 16777216.0)
    return -mx.log(-mx.log(u))


def _threshold(z1, sp):
    """z1 [1, n] tempered logits -> [1, 1]: the lowest one still allowed by top_p (sp[1]) and top_k (sp[2], 0 = off).
    Always on one row, so that its sums have one shape (and one order) in a step and in a verify pass."""
    zs = -mx.sort(-z1, axis=-1)
    ps = mx.softmax(zs, axis=-1)
    n_p = ((mx.cumsum(ps, axis=-1) - ps) < sp[1]).sum(axis=-1)
    k = sp[2].astype(mx.int32)
    n = mx.where(k > 0, mx.minimum(k, n_p), n_p)
    return mx.take_along_axis(zs, (n - 1).reshape(1, 1), axis=-1)


def sample_rows(lg, sp, seed, q0, filt: bool, want_p: bool = False):
    """lg float32 [K, n] logits; sp float32 [3] = (1 / temperature, top_p, top_k); seed uint32 [1]; q0 int32 [1] = the
    position of the token row 0 chooses (row r: q0 + r) -> tokens int32 [K] (and their probabilities under
    softmax(lg / T), for the draft's depth model)."""
    K, n = lg.shape
    z = lg * sp[0]
    score = kn.gumbel_add(z, seed, q0)          # the noise of gumbel(), in one kernel
    if filt:
        thr = mx.concatenate([_threshold(z[r:r + 1], sp) for r in range(K)])
        score = mx.where(z >= thr, score, -mx.inf)
    tok = mx.argmax(score, axis=-1).astype(mx.int32)
    if not want_p:
        return tok, None
    return tok, mx.exp(mx.take_along_axis(z, tok[:, None], axis=-1)[:, 0] - mx.logsumexp(z, axis=-1))


MSG_MIX2 = os.environ.get("MK_MSG_MIX2", "0") == "1"      # bit-equal, and no gain in a pass (not on its critical path)


def msg_mix(sources, queries):
    """The cross-pass messages of a looped layer: the depth mix of the previous loop's blocks under 4 queries. Through
    the mix2 kernel (all its dot products behind one barrier: 7 us at one row) instead of the first mix kernel (a
    barrier per dot product: 23 us); the same partial sums in the same order."""
    return kn.mix2(sources, queries) if MSG_MIX2 else kn.mix(sources, queries)


def rms(x, w=None):
    return mx.fast.rms_norm(x, w, EPS)


def depth_mix(blocks: list, partial, query):
    """Attention of learned queries over the depth sources of each row. blocks: list of [K, D]; partial [K, D];
    query [D] -> [K, D], or [Q, D] -> list of Q arrays [K, D]."""
    if query.ndim == 1:
        return kn.mix(blocks + [partial], query[None])[:, 0]
    o = kn.mix(blocks + [partial], query)
    return [o[:, j] for j in range(query.shape[0])]


def l2clip(x, scale, thr):
    """x [K, H, d] float32."""
    s = x * scale if scale is not None else x
    n = mx.sqrt((s * s).sum(axis=-1, keepdims=True))
    return s * mx.minimum(thr / (n + 1e-6), 1.0)


class Target:
    def __init__(self, model_dir: str, max_context: int):
        cfg = json.loads((Path(model_dir) / "config.json").read_text())
        self.tc = tc = cfg["text_config"]
        assert tc["hidden_size"] == 2304 and tc["num_attention_heads"] == 24 and tc["num_key_value_heads"] == 8
        assert tc["n_routed_experts"] == 128 and tc["num_experts_per_tok"] == 8 and tc["sconv_kernel_size"] == 4
        assert tc["d_rel"] == 16 and tc["rel_extent"] == 1024 and tc["sliding_window_size"] == 1024
        assert tc["mlp_gate_softcap"] == 4.0 and tc["mlp_up_softcap"] == 25.0 and tc["route_scale"] == 8.0
        self.T = int(max_context) + 8
        # K/V: 4 KB per position per layer visit (48 visits at 3 passes: 12.8 GB at 65,536; 72 at 5 passes: 38.7 GB at 131,072)
        assert self.T <= 131072 + 8, "context too long (at most 131,072 positions)"
        self.smax = self.T + 8
        # global-attention layers scale q by tau = 1 + alpha log(max((position + 1) / n_floor, 1)): 1 below n_floor
        self.tau = (float(tc["log_scaling_alpha"]), float(tc["log_scaling_n_floor"])) if self.T + 1 > tc["log_scaling_n_floor"] else None
        self.snap = SNAP if SNAP >= 0 else (64 if self.T <= 8200 else 256)
        self.vocab = tc["unpadded_vocab_size"]
        self.cap = tc["final_logit_softcapping"]
        self.local = set(tc["local_layer_ids"])
        lc = tc["loop_config"]
        self.core = range(lc["core_start"], lc["core_end"])
        # MK_PASSES: run a variable-loop checkpoint at another number of loop passes than its config's (3 instead of 5
        # is about 1.5x faster to read and write, from the same weights)
        trained = lc["num_passes"]
        self.passes = int(os.environ.get("MK_PASSES") or trained)
        if self.passes != trained:
            lc = tc["loop_config"] = dict(lc, num_passes=self.passes)
        # the layer visits whose input the draft head reads, and the final norm (the last id: the number of visits).
        # [5, 23, 41, 48] for 3 passes; a varloop checkpoint run at 5 passes gives [5, 23, 65, 72] in its config.
        ids = cfg.get("mtp_config", {}).get("target_layer_ids", [5, 23, 41, 48])
        if self.passes != trained:
            # a visit before the loop stays; one in a loop pass keeps its place counted from the last pass (pass 5 of 5
            # becomes pass 3 of 3, pass 2 of 5 stays pass 2 when there are enough); one after the loop moves with it
            cs, n = lc["core_start"], lc["core_end"] - lc["core_start"]
            def moved(t):
                if t < cs:
                    return t
                if t >= cs + trained * n:
                    return t - (trained - self.passes) * n
                k, off = divmod(t - cs, n)
                return cs + max(0, k - (trained - self.passes)) * n + off if k >= self.passes else t
            ids = [moved(t) for t in ids]
        self.mtp_ids = list(ids)       # what the checkpoint's draft head reads at this number of passes
        self.taps = tuple(ids[:-1])
        # MK_DRAFT2: the 2DSD drafter (mk_draft2.py) reads the inputs of five weight layers in the last loop instead
        # (default for a variable-loop checkpoint when the drafter is on this Mac; MK_DRAFT2=0 keeps the checkpoint's head)
        d2_default = str(Path.home() / "Models/Inkling/2dsd-varloop-dspark")
        # (the drafter was trained on the checkpoint's own number of passes: at another one the checkpoint's head drafts)
        self.draft2 = os.environ.get("MK_DRAFT2", d2_default if tc["loop_config"].get("is_varloop") and self.passes == trained else "")
        if self.draft2 == "0" or not (Path(self.draft2) / "config.json").exists():
            self.draft2 = ""
        if self.draft2:
            import mk_draft2
            d2 = json.loads((Path(self.draft2) / "config.json").read_text())["loopedtf_drafter"]
            self.taps = mk_draft2.last_loop_taps(tc, d2["feature_layers_per_loop"])
            self.tap_layers = tuple(d2["feature_layers_per_loop"])
            if os.environ.get("MK_TAPS"):       # experiment: other visits for the committed tokens' features
                self.taps = tuple(int(v) for v in os.environ["MK_TAPS"].split(","))
        self.qthr, self.rthr = tc["qk_l2_clip_threshold"], tc["rel_l2_clip_threshold"]
        w = mx.load(str(Path(model_dir) / "model-00001-of-00001.safetensors"))
        g = lambda k: w["model.llm." + k]  # noqa: E731
        u16 = lambda a: a.view(mx.uint16)  # noqa: E731
        self.embed = g("embed.weight")
        self.embed_norm = g("embed_norm.weight").astype(F32)
        self.unembed = u16(g("unembed.weight"))
        self.norm = g("norm.weight").astype(F32)
        self.out_query = g("attn_res.output_query")
        cq = g("middle_cycle_cross_pass_queries").reshape(-1, 2, 2304)
        ckv = g("middle_cycle_cross_pass_kv_queries").reshape(-1, 2, 2304)
        self.msg_query = mx.concatenate([cq[:, :1], ckv, cq[:, 1:]], axis=1)     # [12, 4, D]: q, k, v, mlp
        self.layers = []
        for i in range(tc["num_hidden_layers"]):
            p = f"layers.{i}."
            L = {"full": i not in self.local, "boundary": i % tc["n_layers_per_attn_res_block"] == 0, "i": i}
            L["qgrs"] = u16(mx.concatenate([g(p + "attn.wq_du.weight"), g(p + "attn.gate_attn.weight"),
                                            g(p + "attn.wr_du.weight"), g(p + "attn.attn_sink_proj.weight")]))
            if L["full"]:
                L["wk"] = u16(mx.concatenate([g(p + "attn.wk_dv.weight"), g(p + "attn.k_sconv.weight_proj")]))
                L["wv"] = u16(mx.concatenate([g(p + "attn.wv_dv.weight"), g(p + "attn.v_sconv.weight_proj")]))
                # static [4 lags, 1024] -> [8 heads, 128, 4 lags] as the reference; kept as [4, 1024]
                L["k_static"] = g(p + "attn.k_sconv.static_w").astype(F32)
                L["v_static"] = g(p + "attn.v_sconv.static_w").astype(F32)
                L["qn"], L["kn"], L["vn"] = (g(p + f"{c}_input_norm.weight").astype(F32) for c in "qkv")
            else:
                L["wk"] = u16(g(p + "attn.wk_dv.weight"))
                L["wv"] = u16(g(p + "attn.wv_dv.weight"))
                L["k_conv"] = g(p + "attn.k_sconv.weight").astype(F32)[:, 0, :].T    # [4, 1024], tap j <-> window[t + j]
                L["v_conv"] = g(p + "attn.v_sconv.weight").astype(F32)[:, 0, :].T
                L["an"] = g(p + "attn_norm.weight").astype(F32)
            L["wo"] = u16(g(p + "attn.wo_ud.weight"))
            L["q_scale"] = g(p + "attn.q_l2_clip.scale").astype(F32)
            L["r_scale"] = g(p + "attn.rel_l2_clip.scale").astype(F32)
            L["proj"] = g(p + "attn.rel_logits_proj.proj").astype(F32)
            if PROJT:           # [16, 1024] -> [1024, 16]: the 16 values of a distance in one cache line
                L["proj"] = mx.contiguous(L["proj"].T)
            L["a_conv"] = g(p + "attn_sconv.weight").astype(F32)[:, 0, :].T
            L["m_conv"] = g(p + "mlp_sconv.weight").astype(F32)[:, 0, :].T
            L["mn"] = g(p + "mlp_norm.weight").astype(F32)
            L["router"] = u16(g(p + "mlp.gate.weight"))
            L["rbias"] = g(p + "mlp.gate.bias")
            e = p + "mlp.experts."
            if ("model.llm." + e + "w13_weight_amax") in w:
                L["fmt"] = "nv"
                L["g13"] = g(e + "w13_weight_amax") / g(e + "w13_weight_outer_scale_denominator")
                L["g2"] = g(e + "w2_weight_amax") / g(e + "w2_weight_outer_scale_denominator")
            else:
                L["fmt"] = "mx"
                L["g13"] = L["g2"] = mx.ones((128,), dtype=F32)
            L["w13"], L["s13"], L["w2"], L["s2"] = (g(e + n) for n in ("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale"))
            if i > 0:
                L["aq"] = g(f"attn_res.attn_queries.{i - 1}")
            L["mq"] = g(f"attn_res.mlp_queries.{i}")
            if i in self.core:
                for c, nm in (("q", "attn"), ("k", "k"), ("v", "v"), ("m", "mlp")):
                    cp = p + f"controllers.{nm}."
                    L["c" + c + "_down"] = u16(g(cp + "down_weight"))
                    L["c" + c + "_fg"] = mx.concatenate([g(cp + "forget_weight"), g(cp + "gain_weight")]).astype(F32)
            L["k_taps"], L["v_taps"] = (L["k_static"], L["v_static"]) if L["full"] else (L["k_conv"], L["v_conv"])
            if i > 0:
                L["aq2"] = L["aq"] if L["aq"].ndim == 2 else L["aq"][None]
            L["anw3"] = mx.stack([L["qn"], L["kn"], L["vn"]]) if L["full"] else mx.stack([L["an"]] * 3)
            L["anw"] = L["anw3"] if L["full"] else L["an"][None]
            L["mq2"], L["mnw"] = L["mq"][None], L["mn"][None]
            if i in self.core:
                L["c_down3"] = [L["cq_down"], L["ck_down"], L["cv_down"]]
                L["c_fg3"] = [L["cq_fg"], L["ck_fg"], L["cv_fg"]]
                L["c_down1"], L["c_fg1"] = [L["cm_down"]], [L["cm_fg"]]
                L["c_ft3"] = [mx.contiguous(f.T) for f in L["c_fg3"]]
                L["c_down3s"] = mx.concatenate(L["c_down3"])
                L["c_fg3s"] = mx.concatenate(L["c_fg3"])
                L["c_ft1"] = [mx.contiguous(L["cm_fg"].T)]
            self.layers.append(L)
        keep = [self.embed, self.embed_norm, self.unembed, self.norm, self.out_query, self.msg_query]
        for L in self.layers:
            keep += [v for v in L.values() if isinstance(v, (mx.array, list))]
        mx.eval(keep)
        del w
        if PREFILL == "fast":       # half copies of the dense matrices for the GEMMs of wide passes (BF16 -> half is exact
            for L in self.layers:   # for every weight of magnitude 2^-14 .. 65504)
                for nm in ("qgrs", "wk", "wv", "wo"):
                    L[nm + "h"] = L[nm].view(mx.bfloat16).astype(mx.float16)
                mx.eval([L[nm + "h"] for nm in ("qgrs", "wk", "wv", "wo")])
        if W12:
            for L in self.layers:
                for nm in ("qgrs", "wk", "wv", "wo"):
                    L[nm + "12"] = kn.pack12(L[nm])
                    del L[nm]
            mx.clear_cache()
        # 8-bit per-row-scaled copy of the unembedding: the short list of the exact argmax (and the draft's logits)
        self.un8 = kn.quant8(self.unembed.view(mx.bfloat16)[:self.vocab]) if SHORT else None
        # the layer applications: 48 at 3 passes, 72 at 5
        cs, ce = self.core.start, self.core.stop
        self.apps = list(range(cs)) + list(self.core) * self.passes + list(range(ce, len(self.layers)))
        self.napp = len(self.apps)
        assert (self.draft2 or (len(ids) == 4 and ids[-1] == self.napp)) and all(0 <= t < self.napp for t in self.taps), \
            f"mtp_config.target_layer_ids {ids} do not fit {self.napp} layer visits"
        self._fns = {}
        self.aidx = [mx.array([0, a], dtype=mx.int32) for a in range(self.napp)]     # index in the big tier's array, in the recent tier
        self.reset()

    # ---------------------------------------------------------------------------------------------- state
    RECENT = 128        # rows of the recent tier; flushed into the big cache when 96 are in use

    def reset(self):
        T = self.T
        # K/V of every application (half bits) in two tiers: positions < base in a big array per application, positions
        # base .. pos - 1 of all applications in a small one (a commit rewrites only the small one; MLX slice updates
        # copy the array, so the big tier is one array per application: a flush copies 2 T KB at a time, not 96 T KB).
        # conv windows (the last 3 inputs) per kind, stacked over the applications: k, v (1024), post-attention,
        # post-MLP (2304)
        self.kc = [mx.zeros((1, T, 1024), dtype=mx.uint16) for _ in range(self.napp)]
        self.vc = [mx.zeros((1, T, 1024), dtype=mx.uint16) for _ in range(self.napp)]
        self.kr = mx.zeros((self.napp, self.RECENT, 1024), dtype=mx.uint16)
        self.vr = mx.zeros((self.napp, self.RECENT, 1024), dtype=mx.uint16)
        self.cw = [mx.zeros((self.napp, 3, d), dtype=F32) for d in (1024, 1024, 2304, 2304)]
        self.pos = 0
        self.base = 0
        self.pending = None
        self.snaps = {}             # position q -> the conv windows after q tokens
        self._fresh = []            # snapshots of the last commit, not yet evaluated (state() hands them out)

    def _snap(self, p: int, n: int, wins):
        """Keep the conv windows at every SNAP-th position in (p, p + n]: rows q - p .. q - p + 2 of the pass's windows."""
        self._fresh = []
        if self.snap <= 0:
            return
        q = (p // self.snap + 1) * self.snap
        while q <= p + n:
            s = [mx.contiguous(w[:, q - p:q - p + 3]) for w in wins]
            self.snaps[q] = s
            self._fresh += s
            q += self.snap

    def rollback(self, q: int):
        """Cut the sequence back to its first q tokens (q a snapshot position <= pos): the K/V of positions < q stay."""
        self.flush()
        self.cw = list(self.snaps[q])
        self.pos = self.base = q
        self.pending = None
        self._fresh = []
        for k in [k for k in self.snaps if k > q]:
            del self.snaps[k]

    def commit(self, n_acc: int):
        """Keep the first n_acc rows of the last pass (K/V rows, conv windows) and advance."""
        knew, vnew, wins, K = self.pending
        p, b = self.pos, self.base
        self._snap(p, n_acc, wins)
        self.kr[:, p - b:p - b + n_acc] = knew[:, :n_acc] if n_acc < K else knew
        self.vr[:, p - b:p - b + n_acc] = vnew[:, :n_acc] if n_acc < K else vnew
        self.cw = [w[:, n_acc:n_acc + 3] for w in wins]
        self.pos = p = p + n_acc
        if p - b >= self.RECENT - 32:
            self._to_big(b, p)
            self.base = p
        self.pending = None

    def state(self):
        return [self.kc, self.vc, self.kr, self.vr, self.cw] + self._fresh

    def row_logits(self, row: int):
        """The logits [1, vocab] of row `row` of the last pass (forward() or prefill()): the bits step() returns for
        that position (the row's final state is kept as the half values the unembedding reads)."""
        z = kn.gemv(self.unembed, kn.bits(self.last_feats[-1][row:row + 1]), self.vocab)
        return mx.tanh(z * (1.0 / self.cap)) * self.cap

    # ---------------------------------------------------------------------------------------------- prefill
    def flush(self):
        """Move the recent tier into the big cache (base = pos)."""
        p, b = self.pos, self.base
        if p > b:
            self._to_big(b, p)
        self.base = p

    def _to_big(self, b: int, p: int):
        for a in range(self.napp):
            self.kc[a][0, b:p] = self.kr[a, :p - b]
            self.vc[a][0, b:p] = self.vr[a, :p - b]

    def prefill(self, toks, K: int, want_feats: bool = False, emb=None):
        """One wide pass over the n = len(toks) prompt tokens at positions pos .. pos + n - 1, padded to K rows (rows
        after the n-th see the real ones, never the other way round, and are dropped); commits the n rows. -> (greedy
        token after the last one, int32 [1], lazy; the draft features of the K rows, or None).
        emb = (rows float32 [K, D], use bool [K, 1]): rows given as input embeddings in place of their tokens' (the
        embedding norm already applied, as Inkling's own embedding layer outputs them), e.g. projected audio."""
        n = len(toks)
        assert WIDE_MIN <= K and n <= K and self.pos + K <= self.T + 8
        self.flush()
        tokens = mx.array(list(toks) + [0] * (K - n), dtype=mx.int32)
        m_full = mx.array([self.pos, 0, self.pos], dtype=mx.int32)
        m_loc = mx.array([self.pos, 1024, self.pos], dtype=mx.int32)
        fo16, feats, knew, vnew, wins = self._pure(tokens, m_full, m_loc, self.kc, self.vc, self.kr, self.vr, self.cw, emb=emb)
        self.last_feats = feats
        if self.un8 is not None:
            tok = kn.argmax_short(self.unembed, self.un8, fo16[n - 1:n], self.vocab, self.cap)[0].astype(mx.int32)
        else:
            z = kn.gemv(self.unembed, fo16[n - 1:n], self.vocab)
            tok = mx.argmax(mx.tanh(z * (1.0 / self.cap)) * self.cap, axis=-1).astype(mx.int32)
        self.commit_wide(n, knew, vnew, wins)
        return tok, (feats if want_feats else None)

    def commit_wide(self, n: int, knew, vnew, wins):
        """Keep the first n rows of a wide pass run at base = pos: straight into the big tier."""
        p = self.pos
        self._snap(p, n, wins)
        for a in range(self.napp):
            self.kc[a][0, p:p + n] = knew[a, :n]
            self.vc[a][0, p:p + n] = vnew[a, :n]
        self.cw = [w[:, n:n + 3] for w in wins]
        self.pos = self.base = p + n
        self.pending = None

    # ---------------------------------------------------------------------------------------------- pieces
    @staticmethod
    def _sconv(x, state, taps):
        """Static depthwise conv over the last 4 inputs + residual. x [K, C] float32 (bf16 values), state [3, C],
        taps [4, C] (tap j multiplies window[t + j]; j = 3 is the current input)."""
        K = x.shape[0]
        win = mx.concatenate([state, x])
        out = win[0:K] * taps[0] + win[1:K + 1] * taps[1] + win[2:K + 2] * taps[2] + win[3:K + 3] * taps[3]
        return (out + x) * (1.0 / math.sqrt(5.0)), win

    @staticmethod
    def _dconv(x, dyn, state, static):
        """Dynamic conv of the full-attention layers. x [K, 1024]; dyn [K, 32] (8 heads x 4 lags); static [4 lags, 1024];
        lag 0 is the current input."""
        K = x.shape[0]
        win = mx.concatenate([state, x])
        d = dyn.reshape(K, 8, 1, 4)
        out = None
        for lag in range(4):
            wl = static[lag].reshape(1, 8, 128) + d[..., lag]                 # [K, 8, 128]
            t = win[3 - lag:3 - lag + K].reshape(K, 8, 128) * wl
            out = t if out is None else out + t
        return (out.reshape(K, 1024) + x) * (1.0 / 3.0), win

    @staticmethod
    def _msg_gate(L, c, h, m):
        st = mx.concatenate([rms(h), rms(m)], axis=-1)
        dn = kn.gemv(L["c" + c + "_down"], kn.bits(st))                      # [K, 64]
        fg = mx.tanh(kn.gemv_small(L["c" + c + "_fg"], dn))                  # [K, 4608]
        return (1 - fg[:, :2304]) * h + fg[:, 2304:] * m

    def _layer(self, st, L, hidden, blocks, msgs, meta, K, knew_l, vnew_l, wins_l):
        """hidden float32 [K, D]; blocks: list of unit-RMS float32 [K, D]; msgs float32 [K, 4, D] or None."""
        app, kc_a, vc_a, kr_a, vr_a, cw, a_int = st
        normed_res = None
        if L["i"] == 0:
            normed_res = rms(hidden)
            hq = hk = hv = kn.bits(rms(hidden, L["an"]))
        else:
            srcs = blocks + [hidden]
            if msgs is None:
                if L["boundary"]:
                    y, normed_res = kn.mix2(srcs, L["aq2"], norm_in=True, pout=True, wt=L["anw"])
                else:
                    y = kn.mix2(srcs, L["aq2"], norm_in=True, wt=L["anw"])
                if L["full"]:
                    hq, hk, hv = y[:, 0], y[:, 1], y[:, 2]
                else:
                    hq = hk = hv = y[:, 0]
            else:
                if GATE3:
                    hm, pn_, st_ = kn.mix2g(srcs, L["aq2"], msgs, 3, 0, pout=L["boundary"])
                    if L["boundary"]:
                        normed_res = pn_
                    hq, hk, hv = kn.gates3(st_, hm, msgs, L["c_down3"], L["c_fg3"], L["anw3"], 0)
                elif L["boundary"]:
                    hm, normed_res = kn.mix2(srcs, L["aq2"], norm_in=True, pout=True)
                else:
                    hm = kn.mix2(srcs, L["aq2"], norm_in=True)
                if GATE3:
                    pass
                elif GATE2:
                    hq, hk, hv = kn.gates2(hm, msgs, L["c_down3"], L["c_ft3"], L["anw3"], 0)
                elif GATEM and K <= kn.GEMV_KMAX:
                    hq, hk, hv = kn.gates_m(hm, msgs, L["c_down3s"], L["c_fg3s"], L["anw3"], 0)
                else:
                    hq, hk, hv = kn.gates(hm, msgs, L["c_down3"], L["c_fg3"], L["anw3"], 0)
        if DENSE3 and K <= kn.GEMV_KMAX and "qgrs12" in L and not WHATIF_NO:
            o, kk, vv = kn.gemv12_3(L["qgrs12"], L["wk12"], L["wv12"], hq, hk, hv)
        else:
            o = dense(L, "qgrs", hq)                                        # [K, 3072 + 3072 + 384 + 24]
            kk = dense(L, "wk", hk)
            vv = dense(L, "wv", hv)
        gate, sinks = o[:, 3072:6144], o[:, 6528:]
        q, rel, knew, vnew, w0, w1 = kn.qkv_post(L["full"], o, kk, vv, cw[0], cw[1], L["k_taps"], L["v_taps"], L["q_scale"],
                                                 L["r_scale"], self.qthr, self.rthr)
        if L["full"] and self.tau is not None:
            row_pos = (meta[0][0] + mx.arange(K) + 1).astype(F32)
            q = q * (1.0 + self.tau[0] * mx.log(mx.maximum(row_pos * (1.0 / self.tau[1]), 1.0)))[:, None]
        if K >= WIDE_MIN and PREFILL == "fast":
            att16 = self._attn_wide(L, a_int, q, knew, vnew, rel, sinks, gate)
        else:
            att16 = kn.attention_gate(q, kc_a, vc_a, kr_a, vr_a, knew, vnew, rel, L["proj"], sinks, meta[0 if L["full"] else 1],
                                      app, gate, self.smax, local=not L["full"])
        ao = dense(L, "wo", att16)
        if L["boundary"]:
            h, w2 = kn.sconv2(ao, cw[2], L["a_conv"])
            blocks = blocks + [normed_res]
        else:
            h, w2 = kn.sconv2(ao, cw[2], L["a_conv"], add=hidden)
        srcs = blocks + [h]
        # the router GEMV, top-8 and the experts' slot plan in one dispatch (kernels: route); its counter is zeroed
        # by the kernel that produces the router's input
        r1 = kn.ROUTE1 and K < WIDE_MIN and not WHATIF_NO and (msgs is None or not (GATE2 or GATE3))
        zc = plan = None
        if msgs is None:
            if r1:
                y_, zc = kn.mix2(srcs, L["mq2"], norm_in=True, wt=L["mnw"], zc=True)
                x16 = y_[:, 0]
            else:
                x16 = kn.mix2(srcs, L["mq2"], norm_in=True, wt=L["mnw"])[:, 0]
        else:
            if GATE3:
                xm, _, st_ = kn.mix2g(srcs, L["mq2"], msgs, 1, 3)
                x16 = kn.gates3(st_, xm, msgs, L["c_down1"], L["c_fg1"], L["mnw"], 3)[0]
            else:
              xm = kn.mix2(srcs, L["mq2"], norm_in=True)
              if GATE2:
                  x16 = kn.gates2(xm, msgs, L["c_down1"], L["c_ft1"], L["mnw"], 3)[0]
              elif r1:
                  x16, zc = kn.gates(xm, msgs, L["c_down1"], L["c_fg1"], L["mnw"], 3, zc=True)
              else:
                  x16 = kn.gates(xm, msgs, L["c_down1"], L["c_fg1"], L["mnw"], 3)[0]
        if r1:
            lay = expert_layout(K)
            idx, wts, se, sp = kn.route(L["router"], x16, L["rbias"], zc, lay[0])
            plan = (se, sp)
        else:
            idx, wts = kn.router(kn.gemv(L["router"], x16), L["rbias"])
        if "experts" in WHATIF_NO:
            y = kn.fill(x16, 2304) * (1.0 + 0.0 * wts[:, :1]) + 0.0 * idx[:, :1].astype(F32)
        elif K >= WIDE_MIN and PREFILL == "fast":
            y = kn.experts_mm(L["fmt"], L["w13"], L["s13"], L["g13"], L["w2"], L["s2"], L["g2"], x16, idx, wts)
        elif K >= WIDE_MIN:
            y = kn.experts2(L["fmt"], L["w13"], L["s13"], L["g13"], L["w2"], L["s2"], L["g2"], x16, idx, wts, PF_M, 1, 4, True)
        else:
            y = kn.experts2(L["fmt"], L["w13"], L["s13"], L["g13"], L["w2"], L["s2"], L["g2"], x16, idx, wts, *expert_layout(K), plan=plan)
        out, w3 = kn.sconv2(y, cw[3], L["m_conv"], add=h)
        for _ in range(PAD):                     # timing experiment only: dependent trivial dispatches
            out = kn.ident(out)
        knew_l.append(knew); vnew_l.append(vnew); wins_l.append([w0, w1, w2, w3])
        return out, blocks

    def _attn_wide(self, L, a: int, q, knew, vnew, rel, sinks, gate):
        """The attention of a fast wide pass (prefill(), base = pos) through MLX's fused attention: the relative bias and
        the causal / window limits as an additive mask, the per-row sink as one more key with a zero key and value.
        -> uint16 (half bits) [K, 3072] = att * sigmoid(gate)."""
        K, p0 = q.shape[0], self.pos
        lo = 0 if L["full"] else max(0, p0 - 1023)
        S = p0 - lo + K
        z = mx.zeros((1, 1024), dtype=mx.uint16)
        keys = mx.concatenate([self.kc[a][0, lo:p0], knew, z]).view(mx.float16).astype(F32)
        vals = mx.concatenate([self.vc[a][0, lo:p0], vnew, z]).view(mx.float16).astype(F32)
        keys = keys.reshape(S + 1, 8, 128).transpose(1, 0, 2)[None]
        vals = vals.reshape(S + 1, 8, 128).transpose(1, 0, 2)[None]
        proj = L["proj"] if L["proj"].shape[0] == 16 else L["proj"].T
        relb = mx.matmul(rel.reshape(K, 24, 16).transpose(1, 0, 2), proj)                 # [24, K, 1024] by distance
        far = max(0, S - (1023 + K))        # keys further than 1,023 from every row: no bias, visible to all (global layers)
        d = (p0 + mx.arange(K))[:, None] - (lo + far + mx.arange(S - far))[None, :]         # [K, S - far]
        bias = mx.take_along_axis(relb, mx.broadcast_to(mx.clip(d, 0, 1023)[None], (24, K, S - far)), axis=2)
        bias = mx.where((d < 1024)[None], bias, 0.0)
        ok = (d >= 0) if L["full"] else ((d >= 0) & (d < 1024))
        parts = [mx.where(ok[None], bias, -mx.inf), sinks.T[:, :, None]]
        if far:
            parts.insert(0, mx.zeros((24, K, far), dtype=F32))
        mask = mx.concatenate(parts, axis=2)[None]
        o = mx.fast.scaled_dot_product_attention(q.reshape(K, 24, 128).transpose(1, 0, 2)[None], keys, vals,
                                                 scale=0.08838834764831845, mask=mask)
        o = o[0].transpose(1, 0, 2).reshape(K, 3072)
        return (o * mx.sigmoid(gate)).astype(mx.float16).view(mx.uint16)

    # ---------------------------------------------------------------------------------------------- the pass
    def _pure(self, tokens, m_full, m_loc, kc, vc, kr, vr, cw, tok_only=False, early=0, emb=None):
        """The pass as a pure function of (tokens, position, state): what mx.compile traces, once per K.
        early = r > 0: a draft-only readout after r loop passes that exits through the suffix layers (reading the
        committed tokens' K/V and conv windows of the visits it makes: the prefix, loops 1 .. r and the final suffix);
        its rows are never committed. The features are the inputs of the draft's tap layers in loop r."""
        K = tokens.shape[0]
        meta = (m_full, m_loc)
        passes = early if early else self.passes
        taps = self.taps
        if early:
            n = len(self.core)
            taps = tuple(t if t < self.core.start else t - (self.passes - early) * n for t in self.taps)
        h = rms(self.embed[tokens].astype(F32), self.embed_norm)
        if emb is not None:
            h = mx.where(emb[1], emb[0], h)
        knew, vnew, wins, feats = [], [], [], []
        blocks = []
        cs, ce = self.core.start, self.core.stop
        a = 0           # the visit whose state is read; v counts the visits this pass makes (taps)
        v = 0
        for i in range(cs):
            if v in taps: feats.append(h)
            h, blocks = self._layer((self.aidx[a], kc[a], vc[a], kr, vr, [c[a] for c in cw], a), self.layers[i], h, blocks, None, meta, K, knew, vnew, wins); a += 1; v += 1
        pre_h, pre_blocks = h, blocks
        for lp in range(passes):
            if lp > 0:
                bank_blocks, bank_part = blocks[len(pre_blocks) + 1:], rms(h)
                h, blocks = pre_h, pre_blocks
            allm = kn.mix_all(bank_blocks + [bank_part], self.msg_query) if lp > 0 and kn.MSGM else None
            for j, i in enumerate(self.core):
                msgs = (allm[j] if allm is not None else msg_mix(bank_blocks + [bank_part], self.msg_query[j])) if lp > 0 else None
                if v in taps: feats.append(h)
                h, blocks = self._layer((self.aidx[a], kc[a], vc[a], kr, vr, [c[a] for c in cw], a), self.layers[i], h, blocks, msgs, meta, K, knew, vnew, wins); a += 1; v += 1
        a = cs + self.passes * len(self.core)
        for i in range(ce, len(self.layers)):
            if v in taps: feats.append(h)
            h, blocks = self._layer((self.aidx[a], kc[a], vc[a], kr, vr, [c[a] for c in cw], a), self.layers[i], h, blocks, None, meta, K, knew, vnew, wins); a += 1; v += 1
        fo16 = kn.mix2(blocks + [h], self.out_query[None], norm_in=True, wt=self.norm[None])[:, 0]
        if K >= WIDE_MIN:           # wide pass: the caller takes the token of the one row it needs (prefill())
            logits = fo16
        elif tok_only:
            # the rows' greedy tokens only: exact argmax through the 8-bit short list (kernels: argmax_short)
            logits = kn.argmax_short(self.unembed, self.un8, fo16, self.vocab, self.cap)[0].astype(mx.int32)
        else:
            z = kn.gemv(self.unembed, fo16, self.vocab)
            logits = mx.tanh(z * (1.0 / self.cap)) * self.cap
        feats.append(fo16.view(mx.float16).astype(F32))
        if early:
            return logits, feats
        return logits, feats, mx.stack(knew), mx.stack(vnew), [mx.stack([w[j] for w in wins]) for j in range(4)]

    # ---------------------------------------------------------------------------------------------- staged pass
    # The pass in two stages, for draft-and-verify in rounds (mk_draft2): stage A is the prefix and the first loop
    # (visits 0 .. 17) of some rows, with a draft-only readout through the suffix; stage B is loops 2 .. 5 and the
    # suffix of rows whose stage A is in hand. Every row goes through the statements of _pure with the same inputs
    # (stage A's results are carried, not recomputed), so A then B gives each row the bits of the one pass.
    def _lay(self, a, ci, i, h, blocks, msgs, meta, K, kc, vc, kr, vr, cw, knew, vnew, wins):
        return self._layer((self.aidx[a], kc[a], vc[a], kr, vr, [c[ci] for c in cw], a), self.layers[i], h, blocks, msgs, meta, K, knew, vnew, wins)

    def _stage_a(self, tokens, m_full, m_loc, kc, vc, kr, vr, cwa, samp=None):
        """tokens int32 [K] at positions m_full[0] ..; cwa: the conv windows [24, 3, d] of visits 0 .. 17 and of the 6
        suffix visits (for the readout: provisional for rows not yet committed). -> (readout tokens [K], their probabilities, readout
        features, carry [K, NC, D], K/V rows [24, K, 1024] x 2, windows [24, K + 3, d] x 4); the first 18 of the 24
        are the pass's own visits 0 .. 17."""
        K = tokens.shape[0]
        meta = (m_full, m_loc)
        h = rms(self.embed[tokens].astype(F32), self.embed_norm)
        knew, vnew, wins, feats = [], [], [], []
        blocks = []
        cs, ce, n = self.core.start, self.core.stop, len(self.core)
        tl = self.tap_layers
        for i in range(cs):
            if i in tl: feats.append(h)
            h, blocks = self._lay(i, i, i, h, blocks, None, meta, K, kc, vc, kr, vr, cwa, knew, vnew, wins)
        pre_h, pre_blocks = h, blocks
        for j, i in enumerate(self.core):
            if i in tl: feats.append(h)
            h, blocks = self._lay(cs + j, cs + j, i, h, blocks, None, meta, K, kc, vc, kr, vr, cwa, knew, vnew, wins)
        tail = blocks[len(pre_blocks) + 1:]
        self._nc = (len(pre_blocks), len(tail))
        carry = mx.stack([pre_h, h, feats[0]] + pre_blocks + tail, axis=1)
        a0 = cs + self.passes * n
        for i in range(ce, len(self.layers)):
            if i in tl: feats.append(h)
            h, blocks = self._lay(a0 + i - ce, cs + n + i - ce, i, h, blocks, None, meta, K, kc, vc, kr, vr, cwa, knew, vnew, wins)
        fo16 = kn.mix2(blocks + [h], self.out_query[None], norm_in=True, wt=self.norm[None])[:, 0]
        if samp is None:
            tok, pr = self._readout(fo16)
        else:
            # a sampling request: the readout draws with the target's noise of each position (a shared-noise coupling,
            # as the draft's), from its 8-bit logits and without the top_p / top_k cut (a draft: the verify decides)
            lg = mx.tanh(kn.gemv8(self.un8, fo16) * (1.0 / self.cap)) * self.cap
            tok, pr = sample_rows(lg, samp[0], samp[1], m_full[0:1] + 1, False, want_p=True)
        tok = tok.astype(mx.int32)
        feats.append(fo16.view(mx.float16).astype(F32))
        return tok, pr, feats, carry, mx.stack(knew), mx.stack(vnew), [mx.stack([w[j] for w in wins]) for j in range(4)]

    def _stage_b(self, carry, m_full, m_loc, kc, vc, kr, vr, cw, samp=None):
        """carry [K, NC, D] of rows at positions m_full[0] ..; cw: the committed conv windows [72, 3, d] -> (greedy tokens
        [K], the draft features, K/V rows [54, K, 1024] x 2, windows [54, K + 3, d] x 4) of visits 18 .. 71."""
        K = carry.shape[0]
        meta = (m_full, m_loc)
        nb0, nt = self._nc
        pre_h, h = carry[:, 0], carry[:, 1]
        feats = [carry[:, 2]]
        pre_blocks = [carry[:, 3 + i] for i in range(nb0)]
        bank_blocks = [carry[:, 3 + nb0 + i] for i in range(nt)]
        knew, vnew, wins = [], [], []
        cs, ce, n = self.core.start, self.core.stop, len(self.core)
        a = cs + n
        blocks = None
        for lp in range(1, self.passes):
            if blocks is not None:
                bank_blocks = blocks[len(pre_blocks) + 1:]
            bank_part = rms(h)
            h, blocks = pre_h, pre_blocks
            allm = kn.mix_all(bank_blocks + [bank_part], self.msg_query) if kn.MSGM else None
            for j, i in enumerate(self.core):
                msgs = allm[j] if allm is not None else msg_mix(bank_blocks + [bank_part], self.msg_query[j])
                if a in self.taps: feats.append(h)
                h, blocks = self._lay(a, a, i, h, blocks, msgs, meta, K, kc, vc, kr, vr, cw, knew, vnew, wins); a += 1
        for i in range(ce, len(self.layers)):
            if a in self.taps: feats.append(h)
            h, blocks = self._lay(a, a, i, h, blocks, None, meta, K, kc, vc, kr, vr, cw, knew, vnew, wins); a += 1
        fo16 = kn.mix2(blocks + [h], self.out_query[None], norm_in=True, wt=self.norm[None])[:, 0]
        if samp is None:
            tok = kn.argmax_short(self.unembed, self.un8, fo16, self.vocab, self.cap)[0].astype(mx.int32)
        else:           # the rows' tokens drawn from their exact logits: the statements of a sampling step
            z = kn.gemv(self.unembed, fo16, self.vocab)
            tok = sample_rows(mx.tanh(z * (1.0 / self.cap)) * self.cap, samp[0], samp[1], m_full[0:1] + 1, samp[2])[0]
        feats.append(fo16.view(mx.float16).astype(F32))
        return tok, feats, mx.stack(knew), mx.stack(vnew), [mx.stack([w[j] for w in wins]) for j in range(4)]

    # ---------------------------------------------------------------------------------------------- readout argmax
    # The readout's token (a draft) from a short list instead of a scan of the whole 8-bit unembedding (461 MB): the
    # readout's final states lie almost in a subspace (768 principal directions of states the model produces on random
    # tokens at load hold the exact argmax in the top 4096 scores for 99.7% or more of the states of prose, code and
    # chat: tests/pca.py), so the scores come from the unembedding projected on it (154 MB, 8 bits) and the exact
    # logits are computed for about SL_C tokens.
    def fit_shortlist(self, r: int = 768, n_seq: int = 96):
        import numpy as np
        rng = np.random.default_rng(20261124)
        self.reset()
        F = []
        for _ in range(n_seq):
            f = self.early(mx.array(rng.integers(0, self.vocab, size=16).astype(np.int32)), 1)[1][-1]
            mx.eval(f)
            F.append(f)
        F = mx.concatenate(F)
        g = np.array(F.T @ F).astype(np.float64)
        _, V = np.linalg.eigh(g)
        P = mx.array(np.ascontiguousarray(V[:, -r:]).astype(np.float32))
        un = self.unembed.view(mx.bfloat16)
        parts = []
        for s0 in range(0, self.vocab, 16384):
            parts.append(un[s0:min(s0 + 16384, self.vocab)].astype(F32) @ P)
            mx.eval(parts[-1])
        self.sl_w = kn.quant8(mx.concatenate(parts))
        self.sl_p = P
        mx.eval(self.sl_p, *self.sl_w)
        self.reset()

    def _readout(self, fo16):
        """fo16 uint16 (half bits) [K, D] -> (tokens [K], their probabilities [K]) of a draft-only readout."""
        if getattr(self, "sl_w", None) is None:
            return kn.argmax_short_p(self.unembed, self.un8, fo16, self.vocab, self.cap)
        K = fo16.shape[0]
        V = self.vocab
        z = fo16.view(mx.float16).astype(F32) @ self.sl_p
        sc = kn.gemv8(self.sl_w, kn.bits(z))                                                       # [K, V]
        samp = sc[:, ::16]
        kth = samp.shape[1] - SL_C // 16
        thr = mx.partition(samp, kth, axis=-1)[:, kth:kth + 1]                                     # ~ the SL_C-th largest
        flag = (sc >= thr).any(axis=0).astype(mx.uint8)
        zx = kn._gemv_flag(inputs=[self.unembed, fo16, flag], template=[("D", 2304), ("N", V), ("K", K)], grid=((V + 31) // 32 * 32, 1, 1),
                           threadgroup=(32, 1, 1), output_shapes=[(K, V)], output_dtypes=[F32])[0]
        lg = mx.where(zx == -mx.inf, -mx.inf, mx.tanh(zx * (1.0 / self.cap)) * self.cap)
        return mx.argmax(lg, axis=-1), mx.exp(lg.max(axis=-1) - mx.logsumexp(lg, axis=-1))

    def cwa(self):
        """The committed conv windows of the visits stage A makes: 0 .. 17 and the suffix."""
        n0 = self.core.start + len(self.core)
        return [mx.concatenate([c[:n0], c[self.napp - (len(self.layers) - self.core.stop):]]) for c in self.cw]

    def early(self, tokens: mx.array, r: int = 1):
        """Draft-only: the greedy tokens int32 [K] (and the draft features) of an r-loop readout of tokens at positions
        pos .. pos + K - 1. Nothing is pending or committed."""
        K = tokens.shape[0]
        assert self.pos + K <= self.T and self.pos - self.base + K <= self.RECENT and self.un8 is not None
        fn = self._fns.get(("early", K, r))
        if fn is None:
            pure = lambda tokens, m_full, m_loc, kc, vc, kr, vr, cw: self._pure(tokens, m_full, m_loc, kc, vc, kr, vr, cw, tok_only=True, early=r)  # noqa: E731
            fn = self._fns[("early", K, r)] = mx.compile(pure) if COMPILE else pure
        return fn(tokens, mx.array([self.pos, 0, self.base], dtype=mx.int32), mx.array([self.pos, 1024, self.base], dtype=mx.int32),
                  self.kc, self.vc, self.kr, self.vr, self.cw)

    def _pure_tok(self, tokens, m_full, m_loc, kc, vc, kr, vr, cw):
        return self._pure(tokens, m_full, m_loc, kc, vc, kr, vr, cw, tok_only=True)

    def forward(self, tokens: mx.array, want_feats: bool = False, tok_only: bool = False):
        """tokens int32 [K] at positions pos .. pos + K - 1 -> logits float32 [K, vocab] (lazy), or with tok_only the
        rows' greedy tokens int32 [K] (the first index of each row's maximum logit, without the full logits). The pass
        is pending until commit(n)."""
        K = tokens.shape[0]
        assert self.pos + K <= self.T
        assert self.pos - self.base + K <= self.RECENT
        m_full = mx.array([self.pos, 0, self.base], dtype=mx.int32)
        m_loc = mx.array([self.pos, 1024, self.base], dtype=mx.int32)
        tok_only = tok_only and self.un8 is not None
        pure = self._pure_tok if tok_only else self._pure
        fn = self._fns.get((K, tok_only)) if COMPILE else pure
        if fn is None:
            fn = self._fns[(K, tok_only)] = mx.compile(pure)
        logits, feats, knew, vnew, wins = fn(tokens, m_full, m_loc, self.kc, self.vc, self.kr, self.vr, self.cw)
        self.pending = (knew, vnew, wins, K)
        self.last_feats = feats
        if want_feats:
            return logits, feats
        return logits

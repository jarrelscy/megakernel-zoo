"""Metal kernels of the engine (compiled at run time through mx.fast.metal_kernel).

Every kernel that sums over a weight row does it per (row of the pass, output) in an order that does not depend on how
many rows the pass has: explicit fma, default ("safe") math mode, simd_sum at the end. A K-row verify pass therefore
gives each row the bits the one-row step gives it (tests/rowexact.py).
"""
from __future__ import annotations

import mlx.core as mx

_BF = "as_type<float4>(uint4(*((const device vec<ushort,4>*)(%s))) << 16)"


def bf4(expr: str) -> str:
    """4 BF16 weights at expr as float4."""
    return _BF % expr


def hf4(expr: str) -> str:
    """4 activations at expr (IEEE half bits in a uint16 buffer) as float4."""
    if _WHATIF_X and "xr" in expr:
        return "float4(1.0f)"
    return "float4(*((const device half4*)(%s)))" % expr


import os as _osw
_WHATIF_X = _osw.environ.get("MK_WHATIF_X", "0") == "1"       # timing experiment only: expert kernels without x loads
# timing experiment only: every weight loop stops after one iteration (wrong results; the same dispatches, buffers and
# graph, a tenth of the GPU work): what a pass costs apart from reading its weights
_WHATIF_SHORT = _osw.environ.get("MK_WHATIF_SHORT", "0") == "1"


def _wi(src: str) -> str:
    if not _WHATIF_SHORT:
        return src
    return src.replace("i < D / 128;", "i < 1;").replace("i < D / 256;", "i < 1;")


# ------------------------------------------------------------------------------------------------- dense BF16 GEMV
# out[k, n] = sum_d W[n, d] x[k, d]; W bf16 bits [N, D], x bf16 bits [K, D], D % 128 == 0. One simdgroup per output.
_GEMV = f"""
    uint n = thread_position_in_grid.x / 32;
    uint lane = thread_index_in_simdgroup;
    float4 acc[K];
    for (int k = 0; k < K; k++) acc[k] = float4(0.0f);
    const device ushort* wr = w + size_t(n) * D;
    for (int i = 0; i < D / 128; i++) {{
        int idx = (i * 32 + lane) * 4;
        float4 wf = {bf4('wr + idx')};
        for (int k = 0; k < K; k++) {{
            float4 xv = {hf4('x + k * D + idx')};
            acc[k] = fma(wf, xv, acc[k]);
        }}
    }}
    for (int k = 0; k < K; k++) {{
        float4 a = acc[k];
        float s = simd_sum((a.x + a.y) + (a.z + a.w));
        if (lane == 0) out[k * N + n] = s;
    }}
"""
_gemv = mx.fast.metal_kernel(name="mk_gemv", input_names=["w", "x"], output_names=["out"], source=_wi(_GEMV))


def gemv(w16: mx.array, x16: mx.array, n_out: int | None = None) -> mx.array:
    """w16 uint16 [N, D] (bf16 bits), x16 uint16 [K, D] (bf16 bits) -> float32 [K, N]."""
    N, D = w16.shape
    if n_out is not None:
        N = n_out
    K = x16.shape[0]
    if K > GEMV_KMAX:
        return _blocked_run(_gemv_b, [w16], x16, N, D)
    return _gemv(inputs=[w16, x16], template=[("D", D), ("N", N), ("K", K)], grid=(N * 32, 1, 1),
                 threadgroup=(256 if N % 8 == 0 else 32, 1, 1), output_shapes=[(K, N)], output_dtypes=[mx.float32])[0]


def bits(x: mx.array) -> mx.array:
    """float -> IEEE half bits: the activation format the kernels read (11 significant bits; BF16 has 8)."""
    return x.astype(mx.float16).view(mx.uint16)


# small float GEMV: out[k, n] = sum_d W[n, d] x[k, d], any D, one thread per output
_GEMV_S = """
    uint g = thread_position_in_grid.x;
    uint k = g / N, n = g % N;
    float acc = 0.0f;
    for (int d = 0; d < D / 4 * 4; d += 4) {            // vector loads; the sum keeps its element order
        float4 wv = *((const device float4*)(w + n * D + d));
        float4 xv = *((const device float4*)(x + k * D + d));
        acc = fma(wv.x, xv.x, acc);
        acc = fma(wv.y, xv.y, acc);
        acc = fma(wv.z, xv.z, acc);
        acc = fma(wv.w, xv.w, acc);
    }
    for (int d = D / 4 * 4; d < D; d++) acc = fma(w[n * D + d], x[k * D + d], acc);
    out[g] = acc;
"""
_gemv_s = mx.fast.metal_kernel(name="mk_gemv_small", input_names=["w", "x"], output_names=["out"], source=_GEMV_S)


def gemv_small(w: mx.array, x: mx.array) -> mx.array:
    N, D = w.shape
    K = x.shape[0]
    return _gemv_s(inputs=[w, x], template=[("D", D), ("N", N)], grid=(K * N, 1, 1), threadgroup=(32, 1, 1),
                   output_shapes=[(K, N)], output_dtypes=[mx.float32])[0]


# ------------------------------------------------------------------------------------------------- router
_ROUTER = """
    uint row = thread_position_in_grid.x / 32;
    uint lane = thread_index_in_simdgroup;
    float sc[4];
    float bi[4];
    for (int i = 0; i < 4; i++) {
        uint e = lane + 32 * i;
        float x = logits[row * 128 + e];
        float y = 1.0f / (1.0f + metal::precise::exp(metal::abs(x)));
        sc[i] = x < 0 ? y : 1.0f - y;
        bi[i] = sc[i] + bias[e];
    }
    float wsel[8];
    uint esel[8];
    for (int k = 0; k < 8; k++) {
        float best = -INFINITY;
        uint be = 128;
        for (int i = 0; i < 4; i++) {
            if (bi[i] > best) { best = bi[i]; be = lane + 32 * i; }
        }
        float top = simd_max(best);
        uint win = simd_min(best == top ? be : 128u);
        float s = 0.0f;
        for (int i = 0; i < 4; i++) {
            if (lane + 32 * i == win) { s = sc[i]; bi[i] = -INFINITY; }
        }
        wsel[k] = simd_sum(s);
        esel[k] = win;
    }
    float total = 0.0f;
    for (int k = 0; k < 8; k++) total += wsel[k];
    if (lane == 0) {
        for (int k = 0; k < 8; k++) {
            idx[row * 8 + k] = esel[k];
            wts[row * 8 + k] = wsel[k] / (total + 1e-20f) * 8.0f;
        }
    }
"""
_router = mx.fast.metal_kernel(name="mk_router", input_names=["logits", "bias"], output_names=["idx", "wts"], source=_ROUTER)


def router(logits: mx.array, bias: mx.array):
    K = logits.shape[0]
    return _router(inputs=[logits, bias], grid=(K * 32, 1, 1), threadgroup=(32, 1, 1), output_shapes=[(K, 8), (K, 8)],
                   output_dtypes=[mx.uint32, mx.float32])


# ------------------------------------------------------------------------------------------------- experts
def _e4m3(b: int) -> float:
    s = -1.0 if b & 128 else 1.0
    e, m = (b >> 3) & 15, b & 7
    return s * (m / 8 * 2.0 ** -6 if e == 0 else (1 + m / 8) * 2.0 ** (e - 7))


_E2 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
import os as _os0
NVDEC = int(_os0.environ.get("MK_NVDEC", "2"))      # 2: -7% on the experts of a 16-row pass, same bits (tests/e4.py)
# decode of 4 E2M1 codes (the low 16 bits of c) to float4: the same values by three routes
_NV4 = {
    0: "inline float4 nv4(uint c) { return float4(E2M1[c & 15u], E2M1[(c >> 4) & 15u], E2M1[(c >> 8) & 15u], E2M1[(c >> 12) & 15u]); }",
    # arithmetic: magnitude code n3 >= 2 is 0x3F000000 + (n3 << 22) as float bits; 0 and 1 are 0 and 0.5
    1: """inline float4 nv4(uint c) {
    uint4 c4 = uint4(c) >> uint4(0u, 4u, 8u, 12u);
    uint4 n3 = c4 & 7u;
    uint4 b = select(0x3F000000u + (n3 << 22), n3 * 0x3F000000u, n3 < uint4(2u));
    return as_type<float4>(b | ((c4 & 8u) << 28));
}""",
    # a table per code byte (two weights)
    2: "constant float2 E2M1B[256] = {" + ", ".join(
        "float2(%sf, %sf)" % (repr(_E2[b & 15]), repr(_E2[b >> 4])) for b in range(256)) + """};
inline float4 nv4(uint c) { return float4(E2M1B[c & 255u], E2M1B[(c >> 8) & 255u]); }""",
}
_HDR = ("constant float E2M1[16] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f, -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};\n"
        "constant float E4M3[256] = {" + ", ".join(repr(float(_e4m3(b) if (b & 127) != 127 else 0.0)) + "f" for b in range(256)) + "};\n"
        """
NV4_IMPL
inline float4 mx4(uint c) { return float4(E4M3[c & 255u], E4M3[(c >> 8) & 255u], E4M3[(c >> 16) & 255u], E4M3[c >> 24]); }
inline float hsum(float4 a) { return (a.x + a.y) + (a.z + a.w); }
// one NVFP4 group: 16 elements (8 code bytes), x16 = 16 bf16 inputs
inline float nvgroup(const device uchar* cp, float4 x0, float4 x1, float4 x2, float4 x3) {
    uint c0 = *((const device uint*)cp);
    uint c1 = *((const device uint*)(cp + 4));
    float4 a = nv4(c0) * x0;
    a = fma(nv4(c0 >> 16), x1, a);
    a = fma(nv4(c1), x2, a);
    a = fma(nv4(c1 >> 16), x3, a);
    return hsum(a);
}
// one MXFP8 group: 32 elements (32 code bytes)
inline float mxgroup(const device uchar* cp, const device ushort* xp) {
    float4 a = float4(0.0f);
    for (int j = 0; j < 8; j++) {
        uint c = *((const device uint*)(cp + 4 * j));
        float4 xv = float4(*((const device half4*)(xp + 4 * j)));
        a = fma(mx4(c), xv, a);
    }
    return hsum(a);
}
inline ushort to_bf16(float f) {       // (name kept) the activation format between the two expert matrices: IEEE half bits
    return as_type<ushort>(half(f));
}
""")
_HDR = _HDR.replace("NV4_IMPL", _NV4[NVDEC])


def _dot_src(fmt: str, D: int, acc: str, cp: str, sp: str, xp: str) -> str:
    """Statements that add one lane's share of the dot of a quantized row (codes at cp, scales at sp) with the bf16 row
    at xp to `acc` (already scaled per group)."""
    if fmt == "nv":
        groups = D // 16
        it = (groups + 31) // 32
        return f"""
        for (int i = 0; i < {it}; i++) {{
            int g = i * 32 + int(lane);
            if (g < {groups}) {{
                const device ushort* xq = {xp} + g * 16;
                float4 x0 = {hf4('xq')};
                float4 x1 = {hf4('xq + 4')};
                float4 x2 = {hf4('xq + 8')};
                float4 x3 = {hf4('xq + 12')};
                {acc} = fma(nvgroup({cp} + g * 8, x0, x1, x2, x3), E4M3[({sp})[g]], {acc});
            }}
        }}"""
    groups = D // 32
    it = (groups + 31) // 32
    return f"""
        for (int i = 0; i < {it}; i++) {{
            int g = i * 32 + int(lane);
            if (g < {groups}) {{
                {acc} = fma(mxgroup({cp} + g * 32, {xp} + g * 32), as_type<float>(uint(({sp})[g]) << 23), {acc});
            }}
        }}"""


def _make_up(fmt: str):
    D, cb, sb = 2304, (1152 if fmt == "nv" else 2304), (144 if fmt == "nv" else 72)
    src = f"""
    uint g0 = thread_position_in_grid.x / 32;
    uint lane = thread_index_in_simdgroup;
    uint p = g0 / 1152, m = g0 % 1152;
    uint e = idx[p];
    const device ushort* xr = x + (p / 8) * {D};
    const device uchar* c0 = codes + (size_t(e) * 2304 + 2 * m) * {cb};
    const device uchar* s0 = scales + (size_t(e) * 2304 + 2 * m) * {sb};
    float a0 = 0.0f, a1 = 0.0f;
    {_dot_src(fmt, D, 'a0', 'c0', 's0', 'xr')}
    {_dot_src(fmt, D, 'a1', f'(c0 + {cb})', f'(s0 + {sb})', 'xr')}
    float gs = gsc[e];
    float gt = simd_sum(a0) * gs;
    float up = simd_sum(a1) * gs;
    if (lane == 0) {{
        float gc = metal::precise::tanh(gt * 0.25f) * 4.0f;
        float uc = metal::precise::tanh(up * 0.04f) * 25.0f;
        float y = 1.0f / (1.0f + metal::precise::exp(metal::abs(gt)));
        float sg = gt < 0 ? y : 1.0f - y;
        out[g0] = to_bf16(gc * sg * uc);
    }}
"""
    return mx.fast.metal_kernel(name=f"mk_up_{fmt}", input_names=["codes", "scales", "gsc", "x", "idx"], output_names=["out"],
                                source=src, header=_HDR)


def _make_down(fmt: str):
    D, cb, sb = 1152, (576 if fmt == "nv" else 1152), (72 if fmt == "nv" else 36)
    src = f"""
    uint g0 = thread_position_in_grid.x / 32;
    uint lane = thread_index_in_simdgroup;
    uint row = g0 / 2304, n = g0 % 2304;
    float y = 0.0f;
    for (int j = 0; j < 8; j++) {{
        uint p = row * 8 + j;
        uint e = idx[p];
        const device ushort* xr = h + p * {D};
        const device uchar* c0 = codes + (size_t(e) * 2304 + n) * {cb};
        const device uchar* s0 = scales + (size_t(e) * 2304 + n) * {sb};
        float a0 = 0.0f;
        {_dot_src(fmt, D, 'a0', 'c0', 's0', 'xr')}
        y = fma(simd_sum(a0) * gsc[e], wts[p], y);
    }}
    if (lane == 0) out[g0] = y;
"""
    return mx.fast.metal_kernel(name=f"mk_down_{fmt}", input_names=["codes", "scales", "gsc", "h", "idx", "wts"],
                                output_names=["out"], source=src, header=_HDR)


_up = {f: _make_up(f) for f in ("nv", "mx")}
_down = {f: _make_down(f) for f in ("nv", "mx")}


def experts(fmt: str, w13, s13, g13, w2, s2, g2, x16: mx.array, idx: mx.array, wts: mx.array) -> mx.array:
    """x16 uint16 [K, 2304] (bf16 bits), idx uint32 [K, 8], wts float32 [K, 8] -> float32 [K, 2304]: the weighted sum of
    the 8 chosen experts (gate/up rows interleaved, soft-capped SwiGLU, BF16 between the two matrices)."""
    K = x16.shape[0]
    h = _up[fmt](inputs=[w13, s13, g13, x16, idx], grid=(K * 8 * 1152 * 32, 1, 1), threadgroup=(256, 1, 1),
                 output_shapes=[(K * 8, 1152)], output_dtypes=[mx.uint16])[0]
    return _down[fmt](inputs=[w2, s2, g2, h, idx, wts], grid=(K * 2304 * 32, 1, 1), threadgroup=(256, 1, 1),
                      output_shapes=[(K, 2304)], output_dtypes=[mx.float32])[0]


# ------------------------------------------------------------------------------------------------- attention
# One threadgroup per (row, head). Keys at positions < p0 come from the cache (bf16 bits [T, 1024]), positions p0.. from
# the rows of this pass (knew / vnew [K, 1024]). The relative bias is rel . proj[:, distance] for distance < 1024; the
# sink is one extra logit with a zero value.
def _value_loop(nv: int) -> str:
    """The value sums of one chain, nv keys per iteration: the loads first (they miss the cache; one after the other
    they are the latency of this kernel), then the fmas in key order, so the sum is the sequential one."""
    vp = lambda e: (f"float4(*((const device half4*)(({e} < base ? vc + (size_t(app[0]) * TT + {e}) * 1024 : ({e} < p0 ? vrec + "
                    f"(size_t(app[1]) * RR + ({e} - base)) * 1024 : vnew + ({e} - p0) * 1024)) + kvh * 128 + d4)))")
    out = f"        for (; s + {8 * (nv - 1)} <= pos; s += {8 * nv}) {{\n"
    for i in range(nv):
        out += f"            float4 v{i} = {vp(f'(s + {8 * i})')};\n"
    for i in range(nv):
        out += f"            float w{i} = sc[s + {8 * i} - first];\n"
    for i in range(nv):
        out += f"            acc = fma(v{i}, float4(w{i}), acc);\n"
    out += "        }\n"
    return out


import os as _os3
ATT_NV = int(_os3.environ.get("MK_ATT_NV", "4"))          # keys per iteration of the value sums (same bits for any)
_ATT = f"""
    constexpr int TG = 256;            // the layout that defines the bits: 8 simdgroups, 8 value chains
    constexpr int RT = RTG;            // real threads: the extra ones only compute per-key scores
    uint tid = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    uint grp = threadgroup_position_in_grid.x;
    int row = grp / 24, head = grp % 24, kvh = head / 3;
    int p0 = meta[0];
    int base = meta[2];
    int pos = p0 + row;
    int first = meta[1] > 0 ? max(0, pos - meta[1] + 1) : 0;
    threadgroup float sc[SMAX];
    threadgroup float red[8];
    threadgroup float redm[RT / 32];
    threadgroup float4 part[TG];
    const device float* qp = q + grp * 128;
    const device float* rp = rel + grp * 16;
    int wi = ((WI & 16) && head % 3 != 0) ? 3 : WI;          // timing experiments only (WI = 0 in the engine)
    float lmax = -INFINITY;
    for (int s = first + int(tid); s <= pos; s += RT) {{
        if (wi & 1) {{ sc[s - first] = 0.0f; lmax = 0.0f; continue; }}        // timing experiment: no scores
        const device ushort* kp = (s < base ? kc + (size_t(app[0]) * TT + s) * 1024 : (s < p0 ? krec + (size_t(app[1]) * RR + (s - base)) * 1024 : knew + (s - p0) * 1024)) + kvh * 128;
        float4 a = float4(0.0f);
        for (int j = 0; j < 32; j += 4) {{                 // loads first, the sum in dim order
            float4 k0 = {hf4('kp + 4 * j')}, k1 = {hf4('kp + 4 * j + 4')}, k2 = {hf4('kp + 4 * j + 8')}, k3 = {hf4('kp + 4 * j + 12')};
            float4 q0 = *((const device float4*)(qp + 4 * j)), q1 = *((const device float4*)(qp + 4 * j + 4));
            float4 q2 = *((const device float4*)(qp + 4 * j + 8)), q3 = *((const device float4*)(qp + 4 * j + 12));
            a = fma(k0, q0, a); a = fma(k1, q1, a); a = fma(k2, q2, a); a = fma(k3, q3, a);
        }}
        float v = ((a.x + a.y) + (a.z + a.w)) * 0.08838834764831845f;
        int d = pos - s;
        if (d < 1024) {{
            float b = 0.0f;
            for (int r = 0; r < 16; r += 4) {{
                float4 r4 = *((const device packed_float4*)(rp + r));
                float pa, pb, pc, pd;
                if (PROJT) {{                                       // table stored [distance][16]: one line per key
                    float4 p4 = *((const device float4*)(proj + d * 16 + r));
                    pa = p4.x; pb = p4.y; pc = p4.z; pd = p4.w;
                }} else {{
                    pa = proj[r * 1024 + d]; pb = proj[(r + 1) * 1024 + d]; pc = proj[(r + 2) * 1024 + d]; pd = proj[(r + 3) * 1024 + d];
                }}
                b = fma(r4.x, pa, b); b = fma(r4.y, pb, b); b = fma(r4.z, pc, b); b = fma(r4.w, pd, b);
            }}
            v += b;
        }}
        sc[s - first] = v;
        lmax = max(lmax, v);
    }}
    float m = simd_max(lmax);
    if (lane == 0) redm[sg] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sink = sinks[grp];
    m = sink;
    for (int i = 0; i < RT / 32; i++) m = max(m, redm[i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float lsum = 0.0f;
    if (tid < TG) for (int s = first + int(tid); s <= pos; s += TG) {{
        if (WI & 4) {{ sc[s - first] = 1.0f; lsum += 1.0f; continue; }}       // timing experiment: no exp
        float pr = metal::precise::exp(sc[s - first] - m);
        sc[s - first] = pr;
        lsum += pr;
    }}
    float ss = simd_sum(lsum);
    if (lane == 0 && tid < TG) red[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = metal::precise::exp(sink - m);
    for (int i = 0; i < 8; i++) tot += red[i];
    int c = tid / 32;
    int d4 = (tid % 32) * 4;
    float4 acc = float4(0.0f);
    if (tid < TG && !(wi & 2)) {{                              // (WI & 2: timing experiment, no value sums)
        int s = first + c;
VALUE_LOOP
        for (; s <= pos; s += 8) {{
            float4 vf = float4(*((const device half4*)((s < base ? vc + (size_t(app[0]) * TT + s) * 1024 : (s < p0 ? vrec + (size_t(app[1]) * RR + (s - base)) * 1024 : vnew + (s - p0) * 1024)) + kvh * 128 + d4)));
            acc = fma(vf, float4(sc[s - first]), acc);
        }}
    }}
    if (tid < TG) part[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 32) {{
        float4 o = part[tid];
        for (int i = 1; i < 8; i++) o += part[i * 32 + tid];
        o = o / tot;
        GATE_WRITE
    }}
"""
_ATT = _ATT.replace("VALUE_LOOP", _value_loop(ATT_NV))
_att = mx.fast.metal_kernel(name="mk_attention", input_names=["q", "kc", "vc", "krec", "vrec", "knew", "vnew", "rel", "proj", "sinks", "meta", "app"],
                            output_names=["out"], source=_ATT.replace("GATE_WRITE", "*((device float4*)(out + grp * 128 + d4)) = o;"))
# the same with the output gate folded in: out = half(att * sigmoid(gate)), the input of o_proj
_attg = mx.fast.metal_kernel(name="mk_attention_gate", input_names=["q", "kc", "vc", "krec", "vrec", "knew", "vnew", "rel", "proj", "sinks", "meta", "app", "gate"],
                             output_names=["out"], source=_ATT.replace("GATE_WRITE", """float4 gv = *((const device float4*)(gate + grp * 128 + d4));
        float4 sgm = 1.0f / (1.0f + metal::precise::exp(-gv));
        *((device half4*)(out + grp * 128 + d4)) = half4(o * sgm);"""))


def attention_gate(q, kc, vc, krec, vrec, knew, vnew, rel, proj, sinks, meta, app, gate, smax: int, local: bool = False) -> mx.array:
    """As attention(); kc / vc uint16 [NAPP, T, 1024] and app int32 [1] = this application; gate float32 [K, 3072] ->
    uint16 (half bits) [K, 3072] = att * sigmoid(gate). app int32 [2]: this application's index in kc / vc and in krec / vrec
    (the target keeps the big tier as one array per application, [1, T, 1024], and the recent tier as one array). A context beyond ATT_SMAX keys: a local layer needs scores for
    its window only; a global one keeps them in a device buffer (the same statements otherwise: the same bits)."""
    K = q.shape[0]
    if ATT_SHARED and (not local or ATT_SHARED_LOCAL):
        return attention_shared(q, kc, vc, krec, vrec, knew, vnew, rel, proj, sinks, meta, app, gate, smax, local)
    rtg = ATT_RTG if K < ATT_WIDE else 256
    projt = proj.shape[0] == 1024           # [1024, 16] (transposed at load) or the stored [16, 1024]
    if smax > ATT_SMAX and local:
        smax = 1024 + 32
    elif smax > ATT_SMAX:
        return _attg_big(inputs=[q, kc, vc, krec, vrec, knew, vnew, rel, proj, sinks, meta, app, gate],
                         template=[("SMAX", smax), ("TT", kc.shape[1]), ("RR", krec.shape[1]), ("RTG", rtg), ("PROJT", projt), ("WI", ATT_WI)],
                         grid=(K * 24 * rtg, 1, 1), threadgroup=(rtg, 1, 1), output_shapes=[(K, 3072), (K * 24, smax)],
                         output_dtypes=[mx.uint16, mx.float32])[0]
    return _attg(inputs=[q, kc, vc, krec, vrec, knew, vnew, rel, proj, sinks, meta, app, gate],
                 template=[("SMAX", smax), ("TT", kc.shape[1]), ("RR", krec.shape[1]), ("RTG", rtg), ("PROJT", projt), ("WI", ATT_WI)], grid=(K * 24 * rtg, 1, 1),
                 threadgroup=(rtg, 1, 1), output_shapes=[(K, 3072)], output_dtypes=[mx.uint16])[0]


# a threadgroup has 32 KB of its own memory: scores for at most this many keys (the rest of it holds the partial sums)
ATT_SMAX = 7168
_ATT_BIG = (_ATT.replace("    threadgroup float sc[SMAX];\n", "    device float* sc = scr + size_t(grp) * SMAX;\n")
            .replace("threadgroup_barrier(mem_flags::mem_threadgroup);",
                     "threadgroup_barrier(mem_flags::mem_threadgroup); threadgroup_barrier(mem_flags::mem_device);"))
assert "device float* sc" in _ATT_BIG
_attg_big = mx.fast.metal_kernel(name="mk_attention_gate_big", input_names=["q", "kc", "vc", "krec", "vrec", "knew", "vnew", "rel", "proj", "sinks", "meta", "app", "gate"],
                                 output_names=["out", "scr"], source=_ATT_BIG.replace("GATE_WRITE", """float4 gv = *((const device float4*)(gate + grp * 128 + d4));
        float4 sgm = 1.0f / (1.0f + metal::precise::exp(-gv));
        *((device half4*)(out + grp * 128 + d4)) = half4(o * sgm);"""))

# ------------------------------------------------------------------------------------------------- shared attention
# A global layer at a long context through the one-threadgroup-per-(row, head) kernel is 24 threadgroups per row each
# walking every key, and each row and each of the 3 query heads of a KV head reads the keys again: ~360 us a visit at
# 14K keys in a one-row pass, ~770 us at 4 rows.
# Shared reads: one threadgroup per (KV head, chunk of ATT_CH keys at fixed positions) does every row of the pass and
# the 3 query heads of the KV head, so each key and value is read once per pass (the split kernel above reads them once
# per (row, head)). A row's sums cover the same keys in the same order at any K: the chunks are fixed positions, the
# per-row statements those of a one-row pass.
ATT_CH = int(_os3.environ.get("MK_ATT_CH", "256"))
ATT_SHARED = _os3.environ.get("MK_ATT_SHARED", "1") == "1"
ATT_SHARED_LOCAL = _os3.environ.get("MK_ATT_SHARED_LOCAL", "0") == "1"      # local windows: no gain (26 vs 27 us at one row), slower at 8 rows
_ATT_SH1 = """
#define hf4_(e) float4(*((const device half4*)(e)))
    constexpr int TG = 256;
    constexpr int RB = 4;                    // rows per round: 3 * RB score rows in threadgroup memory
    uint tid = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    uint grp = threadgroup_position_in_grid.x;
    int p0 = meta[0];
    int base = meta[2];
    int win = meta[1];
    int first0 = win > 0 ? max(0, p0 - win + 1) : 0;    // the first row's first key; later rows start later
    int cbase = first0 / CH;                            // partials are kept from this chunk on
    int chunk = cbase + int(grp / 8), kvh = grp % 8;
    int lo = chunk * CH;
    if (lo > p0 + KR - 1) return;
    threadgroup float sc[3 * RB][CH];
    threadgroup float tq[3 * RB][128];
    threadgroup float red[8][3 * RB];
    threadgroup float4 part[TG];
    int c = tid / 32;
    int d4 = (tid % 32) * 4;
    for (int rb = 0; rb < KR; rb += RB) {
        int nr = min(RB, KR - rb);
        if (p0 + rb + nr - 1 < lo) break;               // rows in order: the later rounds' rows are later still
        int hiall = min(lo + CH - 1, p0 + rb + nr - 1);
        if (win > 0 && lo + CH - 1 < p0 + rb - win + 1) continue;     // the chunk is before every window of the round
        for (int i = tid; i < 3 * nr * 128; i += TG) {
            int rr = i / 384, hh = (i / 128) % 3, dd = i % 128;
            tq[rr * 3 + hh][dd] = q[((rb + rr) * 24 + kvh * 3 + hh) * 128 + dd];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float lmax[3 * RB];
        for (int j = 0; j < 3 * RB; j++) lmax[j] = -INFINITY;
        for (int s = lo + int(tid); s <= hiall; s += TG) {
            const device ushort* kp = (s < base ? kc + (size_t(app[0]) * TT + s) * 1024 : (s < p0 ? krec + (size_t(app[1]) * RR + (s - base)) * 1024 : knew + (s - p0) * 1024)) + kvh * 128;
            float4 a[3 * RB];
            for (int j = 0; j < 3 * RB; j++) a[j] = float4(0.0f);
            for (int j = 0; j < 32; j += 4) {                 // loads first, each sum in dim order
                float4 k0 = hf4_(kp + 4 * j), k1 = hf4_(kp + 4 * j + 4), k2 = hf4_(kp + 4 * j + 8), k3 = hf4_(kp + 4 * j + 12);
                for (int t = 0; t < 3 * nr; t++) {
                    threadgroup const float* qp = tq[t];
                    float4 q0 = *((threadgroup const float4*)(qp + 4 * j)), q1 = *((threadgroup const float4*)(qp + 4 * j + 4));
                    float4 q2 = *((threadgroup const float4*)(qp + 4 * j + 8)), q3 = *((threadgroup const float4*)(qp + 4 * j + 12));
                    a[t] = fma(k0, q0, a[t]); a[t] = fma(k1, q1, a[t]); a[t] = fma(k2, q2, a[t]); a[t] = fma(k3, q3, a[t]);
                }
            }
            for (int t = 0; t < 3 * nr; t++) {
                int pos = p0 + rb + t / 3;
                if (s > pos || (win > 0 && s < pos - win + 1)) continue;
                int d = pos - s;
                int pr_ = (rb + t / 3) * 24 + kvh * 3 + t % 3;
                float v = ((a[t].x + a[t].y) + (a[t].z + a[t].w)) * 0.08838834764831845f;
                if (d < 1024) {
                    const device float* rp = rel + pr_ * 16;
                    float b = 0.0f;
                    for (int r = 0; r < 16; r += 4) {
                        float4 r4 = *((const device packed_float4*)(rp + r));
                        float pa, pb, pc, pd;
                        if (PROJT) {
                            float4 p4 = *((const device float4*)(proj + d * 16 + r));
                            pa = p4.x; pb = p4.y; pc = p4.z; pd = p4.w;
                        } else {
                            pa = proj[r * 1024 + d]; pb = proj[(r + 1) * 1024 + d]; pc = proj[(r + 2) * 1024 + d]; pd = proj[(r + 3) * 1024 + d];
                        }
                        b = fma(r4.x, pa, b); b = fma(r4.y, pb, b); b = fma(r4.z, pc, b); b = fma(r4.w, pd, b);
                    }
                    v += b;
                }
                sc[t][s - lo] = v;
                lmax[t] = max(lmax[t], v);
            }
        }
        for (int j = 0; j < 3 * nr; j++) {
            float m = simd_max(lmax[j]);
            if (lane == 0) red[sg][j] = m;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float mx_[3 * RB];
        for (int j = 0; j < 3 * nr; j++) {
            float m = red[0][j];
            for (int i = 1; i < 8; i++) m = max(m, red[i][j]);
            mx_[j] = m;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float ls[3 * RB];
        for (int j = 0; j < 3 * nr; j++) {
            int hi = min(lo + CH - 1, p0 + rb + j / 3);
            int fj = win > 0 ? max(0, p0 + rb + j / 3 - win + 1) : 0;
            float lsum = 0.0f;
            for (int s = lo + int(tid); s <= hi; s += TG) {
                if (s < fj) continue;
                float pr = metal::precise::exp(sc[j][s - lo] - mx_[j]);
                sc[j][s - lo] = pr;
                lsum += pr;
            }
            ls[j] = simd_sum(lsum);
        }
        for (int j = 0; j < 3 * nr; j++) if (lane == 0) red[sg][j] = ls[j];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float4 acc[3 * RB];
        for (int j = 0; j < 3 * RB; j++) acc[j] = float4(0.0f);
        for (int s = lo + c; s <= hiall; s += 8) {
            float4 vf = float4(*((const device half4*)((s < base ? vc + (size_t(app[0]) * TT + s) * 1024 : (s < p0 ? vrec + (size_t(app[1]) * RR + (s - base)) * 1024 : vnew + (s - p0) * 1024)) + kvh * 128 + d4)));
            for (int j = 0; j < 3 * nr; j++)
                if (s <= p0 + rb + j / 3 && (win == 0 || s >= p0 + rb + j / 3 - win + 1)) acc[j] = fma(vf, float4(sc[j][s - lo]), acc[j]);
        }
        for (int j = 0; j < 3 * nr; j++) {
            float tot = 0.0f;
            for (int i = 0; i < 8; i++) tot += red[i][j];
            part[tid] = acc[j];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid < 32) {
                int pr_ = (rb + j / 3) * 24 + kvh * 3 + j % 3;
                float4 o = part[tid];
                for (int i = 1; i < 8; i++) o += part[i * 32 + tid];
                size_t at = size_t(pr_) * NCH + (chunk - cbase);
                *((device float4*)(pacc + at * 128 + d4)) = o;
                if (tid == 0) { pm[at] = mx_[j]; pl[at] = tot; }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
"""
_att_sh1 = mx.fast.metal_kernel(name="mk_attention_shared1", input_names=["q", "kc", "vc", "krec", "vrec", "knew", "vnew", "rel", "proj", "meta", "app"],
                                output_names=["pm", "pl", "pacc"], source=_ATT_SH1)
_ATT_SH2 = """
    uint grp = threadgroup_position_in_grid.x;
    uint d4 = thread_position_in_threadgroup.x * 4;
    int pos = meta[0] + int(grp) / 24;
    int win = meta[1];
    int cbase = (win > 0 ? max(0, meta[0] - win + 1) : 0) / CH;
    int c0 = (win > 0 ? max(0, pos - win + 1) : 0) / CH - cbase;
    int last = pos / CH - cbase;
    float sink = sinks[grp];
    float M = sink;
    for (int c = c0; c <= last; c++) M = max(M, pm[grp * NCH + c]);
    float tot = metal::precise::exp(sink - M);
    float4 o = float4(0.0f);
    for (int c = c0; c <= last; c++) {
        float w = metal::precise::exp(pm[grp * NCH + c] - M);
        tot += pl[grp * NCH + c] * w;
        o = fma(*((const device float4*)(pacc + (size_t(grp) * NCH + c) * 128 + d4)), float4(w), o);
    }
    o = o / tot;
    float4 gv = *((const device float4*)(gate + grp * 128 + d4));
    float4 sgm = 1.0f / (1.0f + metal::precise::exp(-gv));
    *((device half4*)(out + grp * 128 + d4)) = half4(o * sgm);
"""
_att_sh2 = mx.fast.metal_kernel(name="mk_attention_shared2", input_names=["pm", "pl", "pacc", "sinks", "gate", "meta"],
                                output_names=["out"], source=_ATT_SH2)


def attention_shared(q, kc, vc, krec, vrec, knew, vnew, rel, proj, sinks, meta, app, gate, smax: int, local: bool = False) -> mx.array:
    """attention_gate() of a global layer, keys in chunks of ATT_CH at fixed positions, each read once per pass."""
    K = q.shape[0]
    nch = (1024 + K + 2 * ATT_CH - 2) // ATT_CH if local else (smax + ATT_CH - 1) // ATT_CH
    projt = proj.shape[0] == 1024
    pm, pl, pacc = _att_sh1(inputs=[q, kc, vc, krec, vrec, knew, vnew, rel, proj, meta, app],
                            template=[("CH", ATT_CH), ("NCH", nch), ("KR", K), ("TT", kc.shape[1]), ("RR", krec.shape[1]), ("PROJT", projt)],
                            grid=(nch * 8 * 256, 1, 1), threadgroup=(256, 1, 1),
                            output_shapes=[(K * 24 * nch,), (K * 24 * nch,), (K * 24 * nch, 128)],
                            output_dtypes=[mx.float32, mx.float32, mx.float32])
    return _att_sh2(inputs=[pm, pl, pacc, sinks, gate, meta], template=[("CH", ATT_CH), ("NCH", nch)], grid=(K * 24 * 32, 1, 1),
                    threadgroup=(32, 1, 1), output_shapes=[(K, 3072)], output_dtypes=[mx.uint16])[0]


import os as _os2
# attention: real threads per (row, head) group in passes of < ATT_WIDE rows. The sums keep the 256-thread layout (same
# bits); the extra threads only spread the per-key score products.
ATT_RTG = int(_os2.environ.get("MK_ATT_RTG", "512"))
ATT_WIDE = int(_os2.environ.get("MK_ATT_WIDE", "4"))
ATT_WI = int(_os2.environ.get("MK_WHATIF_ATT", "0"))      # timing experiment only (0 in the engine)


def attention(q, kc, vc, knew, vnew, rel, proj, sinks, meta, smax: int) -> mx.array:
    """q float32 [K, 24, 128]; kc / vc uint16 [T, 1024]; knew / vnew uint16 [K, 1024]; rel float32 [K, 24, 16]; proj
    float32 [16, 1024]; sinks float32 [K, 24]; meta int32 [2] = (position of row 0, window or 0) -> float32 [K, 3072]."""
    K = q.shape[0]
    rtg = ATT_RTG if K < ATT_WIDE else 256
    kc3, vc3 = (kc[None] if kc.ndim == 2 else kc), (vc[None] if vc.ndim == 2 else vc)
    meta3 = mx.concatenate([meta, meta[:1]])                     # base = p0: every earlier key from the big cache
    return _att(inputs=[q, kc3, vc3, kc3, vc3, knew, vnew, rel, proj, sinks, meta3,
                        mx.zeros((2,), dtype=mx.int32)], template=[("SMAX", smax), ("TT", kc3.shape[1]), ("RR", kc3.shape[1]), ("RTG", rtg), ("PROJT", proj.shape[0] == 1024), ("WI", ATT_WI)], grid=(K * 24 * rtg, 1, 1),
                threadgroup=(rtg, 1, 1), output_shapes=[(K, 3072)], output_dtypes=[mx.float32])[0]


# ------------------------------------------------------------------------------------------------- experts, shared reads
# A verify pass of K rows chooses 8K (row, expert) pairs but far fewer distinct experts. The plan kernel groups the pairs
# by expert into slots of up to M pairs; the gate/up and down kernels then read each expert's codes once per slot and
# accumulate every pair of the slot (each pair in its own accumulator, the same fma order as a one-row step: the bits
# per row do not depend on M, on the slot or on the other rows). Unused slots exit at once.
_PLAN = """
    uint e = thread_position_in_threadgroup.x;          // one thread per expert, one threadgroup of 128
    threadgroup uint cnt[128];
    uint pairs[KM];
    uint m = 0;
    for (uint p = 0; p < NP; p += 4) {                  // vector loads: a load per element made this kernel latency-bound
        vec<uint, 4> v = *((const device packed_uint4*)(idx + p));
        if (v.x == e && m < KM) { pairs[m] = p; m++; }
        if (v.y == e && m < KM) { pairs[m] = p + 1; m++; }
        if (v.z == e && m < KM) { pairs[m] = p + 2; m++; }
        if (v.w == e && m < KM) { pairs[m] = p + 3; m++; }
    }
    uint ns = (m + M - 1) / M;
    cnt[e] = ns;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    vec<uint, 4> offv = vec<uint, 4>(0u), totv = vec<uint, 4>(0u);
    for (uint i = 0; i < 128; i += 4) {
        vec<uint, 4> c4 = *((threadgroup packed_uint4*)(cnt + i));
        totv += c4;
        offv += select(vec<uint, 4>(0u), c4, (vec<uint, 4>(i) + vec<uint, 4>(0u, 1u, 2u, 3u)) < vec<uint, 4>(e));
    }
    uint off = offv.x + offv.y + offv.z + offv.w, total = totv.x + totv.y + totv.z + totv.w;
    for (uint s = 0; s < ns; s++) {
        slot_e[off + s] = e;
        for (uint i = 0; i < M; i++) slot_p[(off + s) * M + i] = (s * M + i < m) ? pairs[s * M + i] : 0xFFFFFFFFu;
    }
    for (uint s = total + e; s < S; s += 128) {
        slot_e[s] = 0xFFFFFFFFu;
        for (uint i = 0; i < M; i++) slot_p[s * M + i] = 0xFFFFFFFFu;
    }
"""
_plan = mx.fast.metal_kernel(name="mk_plan", input_names=["idx"], output_names=["slot_e", "slot_p"], source=_PLAN)


def _decode_src(fmt: str, cp: str, g: str) -> str:
    if fmt == "nv":
        return f"""uint q0 = *((const device uint*)({cp} + {g} * 8));
                uint q1 = *((const device uint*)({cp} + {g} * 8 + 4));
                float4 w0 = nv4(q0), w1 = nv4(q0 >> 16), w2 = nv4(q1), w3 = nv4(q1 >> 16);"""
    return "\n".join(f"float4 w{j} = mx4(*((const device uint*)({cp} + {g} * 32 + {4 * j})));" for j in range(8))


def _rows_src(fmt: str, D: int, M: int, R: int, cps, sps, xps, accs) -> str:
    """One lane's share of R weight rows x M input rows. accs[r][i] accumulators; cps[r] / sps[r] code and scale pointers;
    xps[i] input row pointers."""
    per, nf = (16, 4) if fmt == "nv" else (32, 8)
    groups = D // per
    it = (groups + 31) // 32
    body = ""
    for r in range(R):
        body += "{ " + _decode_src(fmt, cps[r], "g")
        sc = f"E4M3[({sps[r]})[g]]" if fmt == "nv" else f"as_type<float>(uint(({sps[r]})[g]) << 23)"
        body += f"\n                float sc = {sc};"
        for i in range(M):
            xq = f"({xps[i]} + g * {per})"
            if fmt == "nv":
                body += f"""
                {{ float4 a = w0 * {hf4(xq)};
                  a = fma(w1, {hf4(xq + ' + 4')}, a);
                  a = fma(w2, {hf4(xq + ' + 8')}, a);
                  a = fma(w3, {hf4(xq + ' + 12')}, a);
                  {accs[r][i]} = fma(hsum(a), sc, {accs[r][i]}); }}"""
            else:
                body += f"\n                {{ float4 a = float4(0.0f);"
                for j in range(8):
                    body += f"\n                  a = fma(w{j}, {hf4(xq + f' + {4 * j}')}, a);"
                body += f"\n                  {accs[r][i]} = fma(hsum(a), sc, {accs[r][i]}); }}"
        body += " }\n"
    if _WHATIF_SHORT:
        it = 1
    return f"""
        for (int it = 0; it < {it}; it++) {{
            int g = it * 32 + int(lane);
            if (g < {groups}) {{
                {body}
            }}
        }}"""


def _rows_br(fmt, D, M, R, cps, sps, xps, accs, BR):
    """_rows_src, or with BR: a slot that holds n < M pairs runs the n-pair loop (the plan fills a slot's pairs in
    order; the branch is uniform over the simdgroup; each pair's sums are the same statements either way)."""
    if not BR or M < 2:
        return _rows_src(fmt, D, M, R, cps, sps, xps, accs)
    out = ""
    for n in range(1, M):
        body = _rows_src(fmt, D, n, R, cps, sps, xps[:n], [a[:n] for a in accs])
        out += f"{'else ' if n > 1 else ''}if (p{n} == 0xFFFFFFFFu) {{ {body} }}\n    "
    out += f"else {{ {_rows_src(fmt, D, M, R, cps, sps, xps, accs)} }}"
    return "\n    " + out


def _make_up2(fmt: str, M: int, R: int, BR: bool = False):
    D, cb, sb = 2304, (1152 if fmt == "nv" else 2304), (144 if fmt == "nv" else 72)
    accs = [[f"a{r}_{i}" for i in range(M)] for r in range(2 * R)]
    cps = [f"(c0 + {r * cb})" for r in range(2 * R)]
    sps = [f"(s0 + {r * sb})" for r in range(2 * R)]
    xps = [f"xr{i}" for i in range(M)]
    decl = "float " + ", ".join(f"{a} = 0.0f" for row in accs for a in row) + ";"
    xdecl = "\n".join(f"    uint p{i} = slot_p[s * {M} + {i}]; const device ushort* xr{i} = x + ((p{i} == 0xFFFFFFFFu ? p0 : p{i}) / 8) * {D};"
                      for i in range(M))
    fin = ""
    for r in range(R):
        for i in range(M):
            fin += f"""
    {{ float gt = simd_sum({accs[2 * r][i]}) * gs; float up = simd_sum({accs[2 * r + 1][i]}) * gs;
      if (lane == 0 && p{i} != 0xFFFFFFFFu) {{
        float gc = metal::precise::tanh(gt * 0.25f) * 4.0f;
        float uc = metal::precise::tanh(up * 0.04f) * 25.0f;
        float y = 1.0f / (1.0f + metal::precise::exp(metal::abs(gt)));
        float sg = gt < 0 ? y : 1.0f - y;
        out[p{i} * 1152 + m + {r}] = to_bf16(gc * sg * uc);
      }} }}"""
    src = f"""
    uint g0 = thread_position_in_grid.x / 32;
    uint lane = thread_index_in_simdgroup;
    uint s = g0 / {1152 // R}, m = (g0 % {1152 // R}) * {R};
    uint e = slot_e[s];
    if (e == 0xFFFFFFFFu) return;
{xdecl}
    const device uchar* c0 = codes + (size_t(e) * 2304 + 2 * m) * {cb};
    const device uchar* s0 = scales + (size_t(e) * 2304 + 2 * m) * {sb};
    {decl}
    {_rows_br(fmt, D, M, 2 * R, cps, sps, xps, accs, BR)}
    float gs = gsc[e];
    {fin}
"""
    return mx.fast.metal_kernel(name=f"mk_up2_{fmt}_{M}_{R}_{int(BR)}", input_names=["codes", "scales", "gsc", "x", "slot_e", "slot_p"],
                                output_names=["out"], source=src, header=_HDR)


def _make_down2(fmt: str, M: int, R: int, BR: bool = False):
    D, cb, sb = 1152, (576 if fmt == "nv" else 1152), (72 if fmt == "nv" else 36)
    accs = [[f"a{r}_{i}" for i in range(M)] for r in range(R)]
    cps = [f"(c0 + {r * cb})" for r in range(R)]
    sps = [f"(s0 + {r * sb})" for r in range(R)]
    xps = [f"xr{i}" for i in range(M)]
    decl = "float " + ", ".join(f"{a} = 0.0f" for row in accs for a in row) + ";"
    xdecl = "\n".join(f"    uint p{i} = slot_p[s * {M} + {i}]; const device ushort* xr{i} = h + (p{i} == 0xFFFFFFFFu ? p0 : p{i}) * {D};"
                      for i in range(M))
    fin = ""
    for r in range(R):
        for i in range(M):
            fin += f"""
    {{ float o = simd_sum({accs[r][i]}) * gs;
      if (lane == 0 && p{i} != 0xFFFFFFFFu) out[p{i} * 2304 + n + {r}] = o; }}"""
    src = f"""
    uint g0 = thread_position_in_grid.x / 32;
    uint lane = thread_index_in_simdgroup;
    uint s = g0 / {2304 // R}, n = (g0 % {2304 // R}) * {R};
    uint e = slot_e[s];
    if (e == 0xFFFFFFFFu) return;
{xdecl}
    const device uchar* c0 = codes + (size_t(e) * 2304 + n) * {cb};
    const device uchar* s0 = scales + (size_t(e) * 2304 + n) * {sb};
    {decl}
    {_rows_br(fmt, D, M, R, cps, sps, xps, accs, BR)}
    float gs = gsc[e];
    {fin}
"""
    return mx.fast.metal_kernel(name=f"mk_down2_{fmt}_{M}_{R}_{int(BR)}", input_names=["codes", "scales", "gsc", "h", "slot_e", "slot_p"],
                                output_names=["out"], source=src, header=_HDR)


_COMB = """
    uint g = thread_position_in_grid.x;
    uint row = g / 2304, n = g % 2304;
    float4 w0 = *((const device packed_float4*)(wts + row * 8)), w1 = *((const device packed_float4*)(wts + row * 8 + 4));
    const device float* op = o + size_t(row * 8) * 2304 + n;
    float o0 = op[0], o1 = op[2304], o2 = op[2 * 2304], o3 = op[3 * 2304], o4 = op[4 * 2304], o5 = op[5 * 2304], o6 = op[6 * 2304], o7 = op[7 * 2304];
    float y = 0.0f;                                     // loads first, the sum in expert order
    y = fma(o0, w0.x, y); y = fma(o1, w0.y, y); y = fma(o2, w0.z, y); y = fma(o3, w0.w, y);
    y = fma(o4, w1.x, y); y = fma(o5, w1.y, y); y = fma(o6, w1.z, y); y = fma(o7, w1.w, y);
    out[g] = y;
"""
_comb = mx.fast.metal_kernel(name="mk_comb", input_names=["o", "wts"], output_names=["out"], source=_COMB)
_up2, _down2 = {}, {}


# the router's top-k folded into the plan kernel (one threadgroup of 128 threads: simdgroup j runs the statements of
# _ROUTER for rows j, j + 4, ..; then the plan reads the idx this threadgroup wrote): one dependent dispatch fewer
_ROUTE_PLAN = ("""
    uint lane = thread_index_in_simdgroup;
    for (uint row = simdgroup_index_in_threadgroup; row < KM; row += 4) {
""" + _ROUTER.split("uint lane = thread_index_in_simdgroup;\n", 1)[1] + """
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
""" + _PLAN)
assert "uint row = thread_position_in_grid.x / 32;" in _ROUTER.split("uint lane", 1)[0]
_route_plan = mx.fast.metal_kernel(name="mk_route_plan", input_names=["logits", "bias"], output_names=["slot_e", "slot_p", "idx", "wts"],
                                   source=_ROUTE_PLAN)


def route_plan(logits, bias, M: int, S: int):
    """router(logits, bias) and the expert plan of its idx in one kernel -> (slot_e, slot_p, idx, wts)."""
    K = logits.shape[0]
    return _route_plan(inputs=[logits, bias], template=[("KM", K), ("NP", 8 * K), ("M", M), ("S", S)], grid=(128, 1, 1),
                       threadgroup=(128, 1, 1), output_shapes=[(S,), (S, M), (K, 8), (K, 8)],
                       output_dtypes=[mx.uint32, mx.uint32, mx.uint32, mx.float32])


def experts2(fmt: str, w13, s13, g13, w2, s2, g2, x16, idx, wts, M: int, RU: int = 2, RD: int = 2, BR: bool = False,
             plan=None) -> mx.array:
    """As experts(), each distinct expert read once per slot of up to M (row, expert) pairs. plan: (slot_e, slot_p) from
    route() (the same values mk_plan computes). idx = None: wts is (router logits, bias), and the router's top-k runs in
    the plan kernel (route_plan)."""
    K = x16.shape[0]
    S = min(8 * K, 128) if K <= 16 else 128 + 8 * K // M         # wide passes: sum over experts of ceil(pairs / M)
    if plan is not None:
        slot_e, slot_p = plan
    elif idx is None:
        slot_e, slot_p, idx, wts = route_plan(wts[0], wts[1], M, S)
    else:
        slot_e, slot_p = _plan(inputs=[idx], template=[("KM", K), ("NP", 8 * K), ("M", M), ("S", S)], grid=(128, 1, 1),
                               threadgroup=(128, 1, 1), output_shapes=[(S,), (S, M)], output_dtypes=[mx.uint32, mx.uint32])
    ku = _up2.get((fmt, M, RU, BR)) or _up2.setdefault((fmt, M, RU, BR), _make_up2(fmt, M, RU, BR))
    kd = _down2.get((fmt, M, RD, BR)) or _down2.setdefault((fmt, M, RD, BR), _make_down2(fmt, M, RD, BR))
    h = ku(inputs=[w13, s13, g13, x16, slot_e, slot_p], grid=(S * (1152 // RU) * 32, 1, 1), threadgroup=(256, 1, 1),
           output_shapes=[(K * 8, 1152)], output_dtypes=[mx.uint16])[0]
    o = kd(inputs=[w2, s2, g2, h, slot_e, slot_p], grid=(S * (2304 // RD) * 32, 1, 1), threadgroup=(256, 1, 1),
           output_shapes=[(K * 8, 2304)], output_dtypes=[mx.float32])[0]
    return _comb(inputs=[o, wts], grid=(K * 2304, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(K, 2304)],
                 output_dtypes=[mx.float32])[0]


# ------------------------------------------------------------------------------------------------- depth mix
# Attention of Q learned queries over N depth sources of each row (N - 1 block states and the current partial), in one
# kernel: one threadgroup of 576 threads per row, 4 dims per thread; scores by fixed-order sums, softmax over the
# sources, weighted sum. out [K, Q, D].
def _make_mix(N: int, Q: int):
    names = [f"b{i}" for i in range(N)]
    load = "\n".join(f"    float4 v{i} = *((const device float4*)({n} + row * 2304 + d4));" for i, n in enumerate(names))
    dots = "\n".join(f"    float s{i}_{j} = dot4(v{i}, *((const device float4*)(qs + {j} * 2304 + d4)));" for i in range(N) for j in range(Q))
    red = "\n".join(f"    s{i}_{j} = tgsum(s{i}_{j}, buf + {(i * Q + j) * 18}, lane, sg);" for i in range(N) for j in range(Q))
    outs = ""
    for j in range(Q):
        outs += f"    {{ float m = s0_{j};\n"
        for i in range(1, N):
            outs += f"      m = max(m, s{i}_{j});\n"
        for i in range(N):
            outs += f"      float e{i} = metal::precise::exp((s{i}_{j} - m) * SCALE);\n"
        outs += "      float z = e0" + "".join(f" + e{i}" for i in range(1, N)) + ";\n"
        outs += "      float4 o = v0 * (e0 / z);\n"
        for i in range(1, N):
            outs += f"      o = fma(v{i}, float4(e{i} / z), o);\n"
        outs += f"      *((device float4*)(out + (row * {Q} + {j}) * 2304 + d4)) = o; }}\n"
    src = f"""
    constexpr float SCALE = 0.020833333333333332f;      // 2304 ^ -0.5
    uint row = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    uint d4 = tid * 4;
    threadgroup float buf[{N * Q * 18}];
{load}
{dots}
{red}
{outs}
"""
    hdr = """
inline float dot4(float4 a, float4 b) { float4 p = a * b; return (p.x + p.y) + (p.z + p.w); }
// the sum of v over the 576 threads of the group: simdgroup sums, then the 18 partials in order
inline float tgsum(float v, threadgroup float* buf, uint lane, uint sg) {
    float s = simd_sum(v);
    if (lane == 0) buf[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float t = 0.0f;
    for (int i = 0; i < 16; i += 4) {                   // vector loads, the partials still added in order
        float4 b4 = *((threadgroup packed_float4*)(buf + i));
        t += b4.x; t += b4.y; t += b4.z; t += b4.w;
    }
    t += buf[16]; t += buf[17];
    return t;
}
"""
    return mx.fast.metal_kernel(name=f"mk_mix_{N}_{Q}", input_names=names + ["qs"], output_names=["out"], source=src, header=hdr)


_mix = {}


def mix(sources: list, queries: mx.array) -> mx.array:
    """sources: N arrays float32 [K, 2304]; queries float32 [Q, 2304] -> float32 [K, Q, 2304]."""
    N, Q, K = len(sources), queries.shape[0], sources[0].shape[0]
    k = _mix.get((N, Q)) or _mix.setdefault((N, Q), _make_mix(N, Q))
    return k(inputs=list(sources) + [queries], grid=(K * 576, 1, 1), threadgroup=(576, 1, 1), output_shapes=[(K, Q, 2304)],
             output_dtypes=[mx.float32])[0]


# ------------------------------------------------------------------------------------------------- after q/k/v GEMVs
# One kernel for what sits between the q/k/v projections and the attention: the short convolutions of k and v over the
# last 4 inputs (static taps, or static + per-token dynamic taps in the full-attention layers), the L2 clips of q, k and
# the relative states, the RMS norm of v. One thread per (row, head unit): 24 q heads, 8 k heads, 8 v heads, 24 relative.
def _make_qkv(full: bool, qthr: float, rthr: float):
    """One threadgroup of 32 threads per (row, unit): 24 q heads, 8 k heads, 8 v heads, 24 relative. The lanes compute
    4 dims each (scale, or conv + residual); the norm is still summed over the dims in order by one lane, so the bits
    are those of a sequential loop per head, without its 128-step chain of convs. Rows KR .. KR + 2 of the grid only
    write the conv windows."""
    if full:
        conv = """
        float4 acc = float4(0.0f);
        for (int lag = 0; lag < 4; lag++) {
            int i = int(row) - lag;
            float4 wv4 = i < 0 ? *((const device float4*)(st + (3 + i) * 1024 + c)) : *((const device float4*)(src + i * W + c));
            acc = fma(wv4, *((const device float4*)(taps + lag * 1024 + c)) + float4(src[row * W + 1024 + hh * 4 + lag]), acc);
        }
        float4 y = (acc + *((const device float4*)(src + row * W + c))) * 0.3333333333333333f;"""
    else:
        conv = """
        float4 acc = float4(0.0f);
        for (int j = 0; j < 4; j++) {
            int i = int(row) + j - 3;
            float4 wv4 = i < 0 ? *((const device float4*)(st + (3 + i) * 1024 + c)) : *((const device float4*)(src + i * W + c));
            acc = fma(wv4, *((const device float4*)(taps + j * 1024 + c)), acc);
        }
        float4 y = (acc + *((const device float4*)(src + row * W + c))) * 0.4472135954999579f;"""
    seqsum = """
        yb[d4] = y.x; yb[d4 + 1] = y.y; yb[d4 + 2] = y.z; yb[d4 + 3] = y.w;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0) {
            float ss = 0.0f;
            for (int d = 0; d < ND; d += 4) {
                float4 v4 = *((threadgroup packed_float4*)(yb + d));
                ss = fma(v4.x, v4.x, ss); ss = fma(v4.y, v4.y, ss); ss = fma(v4.z, v4.z, ss); ss = fma(v4.w, v4.w, ss);
            }
            fb[0] = ss;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float ss = fb[0];"""
    src = f"""
    constexpr float QTHR = {float(qthr)!r}f;
    constexpr float RTHR = {float(rthr)!r}f;
    uint g = threadgroup_position_in_grid.x;
    uint lane = thread_position_in_threadgroup.x;
    uint d4 = lane * 4;
    uint row = g / 64, u = g % 64;
    threadgroup float yb[128];
    threadgroup float fb[1];
    if (u >= 24 && u < 40) {{
        bool isk0 = u < 32;
        uint c0 = (isk0 ? u - 24 : u - 32) * 128 + d4;
        if (row < 3) {{
            float4 sv0 = *((const device float4*)((isk0 ? kst : vst) + row * 1024 + c0));
            if (isk0) *((device float4*)(wk + row * 1024 + c0)) = sv0; else *((device float4*)(wv + row * 1024 + c0)) = sv0;
        }}
        if (row < KR) {{
            float4 xv0 = *((const device float4*)((isk0 ? kk : vv) + row * W + c0));
            if (isk0) *((device float4*)(wk + (row + 3) * 1024 + c0)) = xv0; else *((device float4*)(wv + (row + 3) * 1024 + c0)) = xv0;
        }}
    }}
    if (row >= KR) return;
    if (u < 24) {{
        constexpr int ND = 128;
        float4 y = *((const device float4*)(o + row * 6552 + u * 128 + d4)) * *((const device float4*)(q_scale + u * 128 + d4));
        {seqsum}
        float f = min(QTHR / (metal::precise::sqrt(ss) + 1e-6f), 1.0f);
        *((device float4*)(q + (row * 24 + u) * 128 + d4)) = y * f;
    }} else if (u < 40) {{
        constexpr int ND = 128;
        bool isk = u < 32;
        uint hh = isk ? u - 24 : u - 32;
        const device float* src = isk ? kk : vv;
        const device float* st = isk ? kst : vst;
        const device float* taps = isk ? ktaps : vtaps;
        uint c = hh * 128 + d4;
        {conv}
        {seqsum}
        float f = isk ? min(QTHR / (metal::precise::sqrt(ss) + 1e-6f), 1.0f) : metal::precise::rsqrt(ss * 0.0078125f + 1e-6f);
        half4 hv = half4(y * f);
        if (isk) *((device half4*)(knew + row * 1024 + c)) = hv; else *((device half4*)(vnew + row * 1024 + c)) = hv;
    }} else {{
        constexpr int ND = 16;
        uint hh = u - 40;
        float4 y = float4(0.0f);
        if (lane < 4) y = *((const device float4*)(o + row * 6552 + 6144 + hh * 16 + d4)) * *((const device float4*)(r_scale + hh * 16 + d4));
        if (lane < 4) {{ yb[d4] = y.x; yb[d4 + 1] = y.y; yb[d4 + 2] = y.z; yb[d4 + 3] = y.w; }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0) {{
            float s2 = 0.0f;
            for (int d = 0; d < ND; d += 4) {{
                float4 v4 = *((threadgroup packed_float4*)(yb + d));
                s2 = fma(v4.x, v4.x, s2); s2 = fma(v4.y, v4.y, s2); s2 = fma(v4.z, v4.z, s2); s2 = fma(v4.w, v4.w, s2);
            }}
            fb[0] = s2;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float f = min(RTHR / (metal::precise::sqrt(fb[0]) + 1e-6f), 1.0f);
        if (lane < 4) *((device float4*)(rel + (row * 24 + hh) * 16 + d4)) = y * f;
    }}
"""
    return mx.fast.metal_kernel(name=f"mk_qkv_{int(full)}", input_names=["o", "kk", "vv", "kst", "vst", "ktaps", "vtaps", "q_scale", "r_scale"],
                                output_names=["q", "rel", "knew", "vnew", "wk", "wv"], source=src)


_qkv = {}


def qkv_post(full: bool, o, kk, vv, kst, vst, ktaps, vtaps, q_scale, r_scale, qthr: float, rthr: float):
    """o float32 [K, 6552] (q | gate | r | sinks), kk / vv float32 [K, 1024 (+32 dynamic taps)], conv states [3, 1024],
    taps [4, 1024] -> q float32 [K, 3072], rel float32 [K, 384], knew / vnew uint16 (half bits) [K, 1024]."""
    K = o.shape[0]
    k = _qkv.get(full) or _qkv.setdefault(full, _make_qkv(full, qthr, rthr))
    return k(inputs=[o, kk, vv, kst, vst, ktaps, vtaps, q_scale, r_scale],
             template=[("W", kk.shape[1]), ("KR", K)], grid=((K + 3) * 64 * 32, 1, 1),
             threadgroup=(32, 1, 1), output_shapes=[(K, 3072), (K, 384), (K, 1024), (K, 1024), (K + 3, 1024), (K + 3, 1024)],
             output_dtypes=[mx.float32, mx.float32, mx.uint16, mx.uint16, mx.float32, mx.float32])


# static short convolution + residual over [K, C] (post-attention and post-MLP): out = (sum_j win[t + j] taps[j] + x) / sqrt(5)
# (+ add[t] when ADD: the block residual)
_SCONV = """
    uint g = thread_position_in_grid.x;
    uint row = g / (C / 4), c = (g % (C / 4)) * 4;
    float4 acc = float4(0.0f);
    for (int j = 0; j < 4; j++) {
        int i = int(row) + j - 3;
        float4 wv = i < 0 ? *((const device float4*)(st + (3 + i) * C + c)) : *((const device float4*)(x + i * C + c));
        acc = fma(wv, *((const device float4*)(taps + j * C + c)), acc);
    }
    float4 y = (acc + *((const device float4*)(x + row * C + c))) * 0.4472135954999579f;
    *((device float4*)(out + row * C + c)) = y;
"""
_sconv = mx.fast.metal_kernel(name="mk_sconv", input_names=["x", "st", "taps"], output_names=["out"], source=_SCONV)


def sconv(x, st, taps):
    K, C = x.shape
    return _sconv(inputs=[x, st, taps], template=[("C", C)], grid=(K * C // 4, 1, 1), threadgroup=(64, 1, 1),
                  output_shapes=[(K, C)], output_dtypes=[mx.float32])[0]


# ------------------------------------------------------------------------------------------------- draft-only 8-bit GEMV
# The draft head only proposes, so its weights may be read from an 8-bit copy (per-row scale; what oMLX and the B200
# engine do for the draft). The target never uses this kernel.
_GEMV8 = f"""
    uint n = thread_position_in_grid.x / 32;
    uint lane = thread_index_in_simdgroup;
    float4 acc[K];
    for (int k = 0; k < K; k++) acc[k] = float4(0.0f);
    const device int8_t* wr = w + size_t(n) * D;
    for (int i = 0; i < D / 256; i++) {{
        int idx = (i * 32 + lane) * 8;
        float4 w0 = float4(*((const device packed_char4*)(wr + idx)));
        float4 w1 = float4(*((const device packed_char4*)(wr + idx + 4)));
        for (int k = 0; k < K; k++) {{
            acc[k] = fma(w0, {hf4('x + k * D + idx')}, acc[k]);
            acc[k] = fma(w1, {hf4('x + k * D + idx + 4')}, acc[k]);
        }}
    }}
    float sc = scale[n];
    for (int k = 0; k < K; k++) {{
        float4 a = acc[k];
        float s = simd_sum((a.x + a.y) + (a.z + a.w));
        if (lane == 0) out[k * N + n] = s * sc;
    }}
"""
_gemv8 = mx.fast.metal_kernel(name="mk_gemv8", input_names=["w", "scale", "x"], output_names=["out"], source=_wi(_GEMV8))


def quant8(w: mx.array, bits: int = 8):
    """float [N, D] -> (int8 [N, D], float32 [N]): symmetric per-row 8-bit copy (draft only)."""
    out_q, out_s = [], []
    for s0 in range(0, w.shape[0], 16384):
        wf = w[s0:s0 + 16384].astype(mx.float32)
        sc = mx.maximum(mx.abs(wf).max(axis=1), 1e-30) / float(2 ** (bits - 1) - 1)
        q = mx.round(wf / sc[:, None]).astype(mx.int8)
        mx.eval(q, sc)
        out_q.append(q); out_s.append(sc)
    q, s = mx.concatenate(out_q), mx.concatenate(out_s)
    mx.eval(q, s)
    return q, s


def gemv8(wq, x16: mx.array, n_out: int | None = None) -> mx.array:
    """wq = (int8 [N, D], scale [N]); x16 uint16 [K, D] (half bits), D % 256 == 0 -> float32 [K, N]."""
    w, sc = wq
    N, D = w.shape
    if n_out is not None:
        N = n_out
    K = x16.shape[0]
    return _gemv8(inputs=[w, sc, x16], template=[("D", D), ("N", N), ("K", K)], grid=(N * 32, 1, 1),
                  threadgroup=(256 if N % 8 == 0 else 32, 1, 1), output_shapes=[(K, N)], output_dtypes=[mx.float32])[0]


# ================================================================================================= fused glue (v2)
_TGHDR = """
inline float dot4(float4 a, float4 b) { float4 p = a * b; return (p.x + p.y) + (p.z + p.w); }
// the sum of v over the 576 threads of the group: simdgroup sums, then the 18 partials in order
inline float tgsum(float v, threadgroup float* buf, uint lane, uint sg) {
    float s = simd_sum(v);
    if (lane == 0) buf[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float t = 0.0f;
    for (int i = 0; i < 16; i += 4) {                   // vector loads, the partials still added in order
        float4 b4 = *((threadgroup packed_float4*)(buf + i));
        t += b4.x; t += b4.y; t += b4.z; t += b4.w;
    }
    t += buf[16]; t += buf[17];
    return t;
}
"""


_TGHDR2 = _TGHDR + """
// the two halves of tgsum, so that several sums can share one barrier
inline void tgput(float v, threadgroup float* buf, uint lane, uint sg) {
    float s = simd_sum(v);
    if (lane == 0) buf[sg] = s;
}
inline float tgget(threadgroup float* buf) {
    float t = 0.0f;
    for (int i = 0; i < 16; i += 4) {
        float4 b4 = *((threadgroup packed_float4*)(buf + i));
        t += b4.x; t += b4.y; t += b4.z; t += b4.w;
    }
    t += buf[16]; t += buf[17];
    return t;
}
"""


def _make_mix2(N: int, Q: int, norm_in: bool, pout: bool, wout: bool, conv: tuple = (), zc: bool = False):
    """Depth mix with the neighbours folded in. Sources b0..b{N-1}; the last one is the partial, RMS-normalized here
    when norm_in. Outputs: out [K, Q, D] float32 (unless wout), pn [K, D] (the normalized partial, when pout),
    y [K, Q, D] half bits = rms_norm(out_j) * wt[j] (when wout). conv = (add,): the partial is computed here from the
    static conv (_make_mix2c)."""
    names = [f"b{i}" for i in range(N)]
    L = N - 1
    load = "\n".join(f"    float4 v{i} = *((const device float4*)({n} + row * 2304 + d4));" for i, n in enumerate(names[:L] if conv else names))
    if conv:
        # the statements of _SCONV2 (C = 2304) for this thread's 4 dims of the row
        load += f"""
    if (row == 0) for (int r = 0; r < 3; r++) *((device float4*)(win + r * 2304 + d4)) = *((const device float4*)(st + r * 2304 + d4));
    float4 xv = *((const device float4*)(x + row * 2304 + d4));
    *((device float4*)(win + (row + 3) * 2304 + d4)) = xv;
    float4 acc = float4(0.0f);
    for (int j = 0; j < 4; j++) {{
        int i = int(row) + j - 3;
        float4 wv = i < 0 ? *((const device float4*)(st + (3 + i) * 2304 + d4)) : *((const device float4*)(x + i * 2304 + d4));
        acc = fma(wv, *((const device float4*)(taps + j * 2304 + d4)), acc);
    }}
    float4 hc = (acc + xv) * 0.4472135954999579f;
{"    hc += *((const device float4*)(add + row * 2304 + d4));" if conv[0] else ""}
    *((device float4*)(hout + row * 2304 + d4)) = hc;
    float4 v{L} = hc;"""
    nred = N * Q + (1 if norm_in else 0) + (Q if wout else 0)
    pre = ""
    if norm_in:
        pre = f"""
    float hh = tgsum(dot4(v{L}, v{L}), buf + {N * Q * 18}, lane, sg);
    v{L} = v{L} * metal::precise::rsqrt(hh * (1.0f / 2304.0f) + 1e-6f);"""
    dots = "\n".join(f"    tgput(dot4(v{i}, *((const device float4*)(qs + {j} * 2304 + d4))), buf + {(i * Q + j) * 18}, lane, sg);"
                     for i in range(N) for j in range(Q))
    dots += "\n    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    dots += "\n".join(f"    float s{i}_{j} = tgget(buf + {(i * Q + j) * 18});" for i in range(N) for j in range(Q))
    outs = ""
    for j in range(Q):
        outs += f"    {{ float m = s0_{j};\n"
        for i in range(1, N):
            outs += f"      m = max(m, s{i}_{j});\n"
        for i in range(N):
            outs += f"      float e{i} = metal::precise::exp((s{i}_{j} - m) * SCALE);\n"
        outs += "      float z = e0" + "".join(f" + e{i}" for i in range(1, N)) + ";\n"
        outs += "      float4 o = v0 * (e0 / z);\n"
        for i in range(1, N):
            outs += f"      o = fma(v{i}, float4(e{i} / z), o);\n"
        if wout:
            outs += f"""      float oo = tgsum(dot4(o, o), buf + {(N * Q + (1 if norm_in else 0) + j) * 18}, lane, sg);
      o = o * metal::precise::rsqrt(oo * (1.0f / 2304.0f) + 1e-6f) * *((const device float4*)(wt + {j} * 2304 + d4));
      *((device half4*)(y + (row * {Q} + {j}) * 2304 + d4)) = half4(o); }}\n"""
        else:
            outs += f"      *((device float4*)(out + (row * {Q} + {j}) * 2304 + d4)) = o; }}\n"
    if pout:
        outs += f"    *((device float4*)(pn + row * 2304 + d4)) = v{L};\n"
    src = f"""
    constexpr float SCALE = 0.020833333333333332f;      // 2304 ^ -0.5
    uint row = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    uint d4 = tid * 4;
    threadgroup float buf[{nred * 18}];
{load}
{pre}
{dots}
{outs}
"""
    onames = ([] if wout else ["out"]) + (["pn"] if pout else []) + (["y"] if wout else [])
    if zc:
        src += "    if (row == 0 && tid == 0) zc[0] = 0u;\n"
    if conv:
        cin = ["x", "st", "taps"] + (["add"] if conv[0] else [])
        return mx.fast.metal_kernel(name=f"mk_mix2c_{N}_{Q}_{int(norm_in)}{int(pout)}{int(wout)}{int(conv[0])}{'z' if zc else ''}",
                                    input_names=names[:L] + cin + ["qs"] + (["wt"] if wout else []),
                                    output_names=onames + ["hout", "win"] + (["zc"] if zc else []), source=src, header=_TGHDR2)
    onames += ["zc"] if zc else []
    return mx.fast.metal_kernel(name=f"mk_mix2_{N}_{Q}_{int(norm_in)}{int(pout)}{int(wout)}{'z' if zc else ''}", input_names=names + ["qs"] + (["wt"] if wout else []),
                                output_names=onames, source=src, header=_TGHDR2)


def _make_mix2f(N: int, Q: int, norm_in: bool, pout: bool, wout: bool):
    """_make_mix2 with the softmax weights of a row computed once: in _make_mix2 each of the 576 threads adds up the
    18 partials of all N * Q dot products and takes N * Q exponentials, all to the same values; here thread j does it
    for query j and the others read the N * Q weights after one more barrier. The same statements on the same
    operands, so the same bits (tests/mixeq.py); the kernel's own latency, which is what a pass waits for, is lower."""
    names = [f"b{i}" for i in range(N)]
    L = N - 1
    load = "\n".join(f"    float4 v{i} = *((const device float4*)({n} + row * 2304 + d4));" for i, n in enumerate(names))
    nred = N * Q + (1 if norm_in else 0) + (Q if wout else 0)
    pre = ""
    if norm_in:
        pre = f"""
    float hh = tgsum(dot4(v{L}, v{L}), buf + {N * Q * 18}, lane, sg);
    v{L} = v{L} * metal::precise::rsqrt(hh * (1.0f / 2304.0f) + 1e-6f);"""
    dots = "\n".join(f"    tgput(dot4(v{i}, *((const device float4*)(qs + {j} * 2304 + d4))), buf + {(i * Q + j) * 18}, lane, sg);"
                     for i in range(N) for j in range(Q))
    dots += "\n    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    dots += f"    if (tid < {Q}u) {{\n"
    dots += "\n".join(f"      float s{i} = tgget(buf + ({i * Q} + tid) * 18);" for i in range(N)) + "\n"
    dots += "      float m = s0;\n" + "".join(f"      m = max(m, s{i});\n" for i in range(1, N))
    dots += "".join(f"      float e{i} = metal::precise::exp((s{i} - m) * SCALE);\n" for i in range(N))
    dots += "      float z = e0" + "".join(f" + e{i}" for i in range(1, N)) + ";\n"
    dots += "".join(f"      wts[tid * {N} + {i}] = e{i} / z;\n" for i in range(N))
    dots += "    }\n    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    outs = ""
    for j in range(Q):
        outs += f"    {{ float4 o = v0 * wts[{j * N}];\n"
        for i in range(1, N):
            outs += f"      o = fma(v{i}, float4(wts[{j * N + i}]), o);\n"
        if wout:
            outs += f"""      float oo = tgsum(dot4(o, o), buf + {(N * Q + (1 if norm_in else 0) + j) * 18}, lane, sg);
      o = o * metal::precise::rsqrt(oo * (1.0f / 2304.0f) + 1e-6f) * *((const device float4*)(wt + {j} * 2304 + d4));
      *((device half4*)(y + (row * {Q} + {j}) * 2304 + d4)) = half4(o); }}\n"""
        else:
            outs += f"      *((device float4*)(out + (row * {Q} + {j}) * 2304 + d4)) = o; }}\n"
    if pout:
        outs += f"    *((device float4*)(pn + row * 2304 + d4)) = v{L};\n"
    src = f"""
    constexpr float SCALE = 0.020833333333333332f;      // 2304 ^ -0.5
    uint row = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    uint d4 = tid * 4;
    threadgroup float buf[{nred * 18}];
    threadgroup float wts[{N * Q}];
{load}
{pre}
{dots}
{outs}
"""
    onames = ([] if wout else ["out"]) + (["pn"] if pout else []) + (["y"] if wout else [])
    return mx.fast.metal_kernel(name=f"mk_mix2f_{N}_{Q}_{int(norm_in)}{int(pout)}{int(wout)}", input_names=names + ["qs"] + (["wt"] if wout else []),
                                output_names=onames, source=src, header=_TGHDR2)


def _make_mix2c(N: int, Q: int, norm_in: bool, pout: bool, wout: bool, add: bool, zc: bool = False):
    """_make_mix2 with the static conv that makes its last source folded in (the sconv2 kernel of the visit, which ran
    just before as its own dispatch): each thread computes its 4 dims of the partial h = (conv(win) + x) / sqrt(5)
    (+ add) from x, the window st and the taps with the statements of _SCONV2, writes h and its rows of the window
    [K + 3, D], and goes on with the depth mix of _make_mix2 on h. Same statements on the same operands: same bits."""
    return _make_mix2(N, Q, norm_in, pout, wout, conv=(add,), zc=zc)


_mix2 = {}
import os as _os5
MIXF = _os5.environ.get("MK_MIXF", "0") == "1"       # bit-equal and 2x faster alone, no gain in a pass (journal): off


def mix2(sources: list, queries: mx.array, norm_in: bool = False, pout: bool = False, wt: mx.array | None = None, fast: bool | None = None,
         zc: bool = False):
    """Returns the kernel's outputs in the order (out or y, [pn]): out float32 [K, Q, D] / y uint16 [K, Q, D], pn [K, D].
    zc: also a zeroed uint32 [1] (route()'s counter), last."""
    N, Q, K = len(sources), queries.shape[0], sources[0].shape[0]
    fast = MIXF if fast is None else fast
    if zc:
        fast = False
    key = (N, Q, norm_in, pout, wt is not None, fast, zc)
    k = _mix2.get(key) or _mix2.setdefault(key, _make_mix2(*key[:5], zc=True) if zc else (_make_mix2f if fast else _make_mix2)(*key[:5]))
    shapes, dts = [], []
    if wt is None:
        shapes.append((K, Q, 2304)); dts.append(mx.float32)
    if pout:
        shapes.append((K, 2304)); dts.append(mx.float32)
    if wt is not None:
        shapes.append((K, Q, 2304)); dts.append(mx.uint16)
    if zc:
        shapes.append((16,)); dts.append(mx.uint32)
    o = k(inputs=list(sources) + [queries] + ([wt] if wt is not None else []), grid=(K * 576, 1, 1), threadgroup=(576, 1, 1),
          output_shapes=shapes, output_dtypes=dts)
    if wt is not None and pout:
        return (o[1], o[0]) + ((o[2],) if zc else ())
    return o if len(o) > 1 else o[0]


_mix2c = {}


def mix2c(blocks: list, x, st, taps, add, queries: mx.array, norm_in: bool = False, pout: bool = False, wt: mx.array | None = None,
          zc: bool = False):
    """sconv2(x, st, taps, add) -> (h, win) and mix2(blocks + [h], queries, ...) in one kernel.
    -> (the outputs of mix2 as a list in mix2's return order (with zc: route()'s zeroed counter last), h [K, D],
    win [K + 3, D])."""
    N, Q, K = len(blocks) + 1, queries.shape[0], x.shape[0]
    assert x.shape[1] == 2304
    key = (N, Q, norm_in, pout, wt is not None, add is not None, zc)
    k = _mix2c.get(key) or _mix2c.setdefault(key, _make_mix2c(*key))
    shapes, dts = [], []
    if wt is None:
        shapes.append((K, Q, 2304)); dts.append(mx.float32)
    if pout:
        shapes.append((K, 2304)); dts.append(mx.float32)
    if wt is not None:
        shapes.append((K, Q, 2304)); dts.append(mx.uint16)
    shapes += [(K, 2304), (K + 3, 2304)]; dts += [mx.float32, mx.float32]
    if zc:
        shapes.append((16,)); dts.append(mx.uint32)
    ins = list(blocks) + [x, st, taps] + ([add] if add is not None else []) + [queries] + ([wt] if wt is not None else [])
    o = k(inputs=ins, grid=(K * 576, 1, 1), threadgroup=(576, 1, 1), output_shapes=shapes, output_dtypes=dts)
    z = [o[-1]] if zc else []
    o = o[:-1] if zc else o
    main, h, win = list(o[:-2]), o[-2], o[-1]
    if wt is not None and pout:
        main = [main[1], main[0]]
    return main + z, h, win


# message gates of the looped layers: G gates at once (q, k, v: G = 3, messages 0..2; mlp: G = 1, message 3)
def _make_gate_pre(G: int, QH: int, M0: int):
    """st_g = [unit-RMS(h_g), unit-RMS(m_g)] as half bits [K, 4608], one output per gate (the input of down_proj)."""
    src = f"""
    uint grp = threadgroup_position_in_grid.x;
    uint row = grp / {G}, g = grp % {G};
    uint tid = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    uint d4 = tid * 4;
    threadgroup float buf[36];
    float4 hv = *((const device float4*)(h + (row * {QH} + min(g, {QH - 1}u)) * 2304 + d4));
    float4 mv = *((const device float4*)(msgs + (row * 4 + {M0} + g) * 2304 + d4));
    float hh = tgsum(dot4(hv, hv), buf, lane, sg);
    float mm = tgsum(dot4(mv, mv), buf + 18, lane, sg);
    half4 ho = half4(hv * metal::precise::rsqrt(hh * (1.0f / 2304.0f) + 1e-6f));
    half4 mo = half4(mv * metal::precise::rsqrt(mm * (1.0f / 2304.0f) + 1e-6f));
""" + "".join(f"""
    if (g == {i}) {{ *((device half4*)(st{i} + row * 4608 + d4)) = ho; *((device half4*)(st{i} + row * 4608 + 2304 + d4)) = mo; }}""" for i in range(G))
    return mx.fast.metal_kernel(name=f"mk_gate_pre_{G}_{QH}_{M0}", input_names=["h", "msgs"], output_names=[f"st{i}" for i in range(G)],
                                source=src, header=_TGHDR)


def _make_gate_post(G: int, QH: int, M0: int, zc: bool = False):
    """y_g = half(rms_norm((1 - tanh(f_g)) h_g + tanh(gain_g) m_g) * wt_g); fg_g float32 [K, 4608] = forget | gain."""
    sel = "".join(f"    if (g == {i}) fg = fg{i};\n" for i in range(1, G))
    src = f"""
    uint grp = threadgroup_position_in_grid.x;
    uint row = grp / {G}, g = grp % {G};
    uint tid = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    uint d4 = tid * 4;
    threadgroup float buf[18];
    const device float* fg = fg0;
{sel}
    float4 hv = *((const device float4*)(h + (row * {QH} + min(g, {QH - 1}u)) * 2304 + d4));
    float4 mv = *((const device float4*)(msgs + (row * 4 + {M0} + g) * 2304 + d4));
    float4 f = metal::precise::tanh(*((const device float4*)(fg + row * 4608 + d4)));
    float4 gn = metal::precise::tanh(*((const device float4*)(fg + row * 4608 + 2304 + d4)));
    float4 v = (1.0f - f) * hv + gn * mv;
    float vv = tgsum(dot4(v, v), buf, lane, sg);
    half4 o = half4(v * metal::precise::rsqrt(vv * (1.0f / 2304.0f) + 1e-6f) * *((const device float4*)(wt + g * 2304 + d4)));
""" + "".join(f"""
    if (g == {i}) *((device half4*)(y{i} + row * 2304 + d4)) = o;""" for i in range(G))
    if zc:
        src += "\n    if (grp == 0 && tid == 0) zc[0] = 0u;\n"
    return mx.fast.metal_kernel(name=f"mk_gate_post_{G}_{QH}_{M0}{'z' if zc else ''}", input_names=["h", "msgs", "wt"] + [f"fg{i}" for i in range(G)],
                                output_names=[f"y{i}" for i in range(G)] + (["zc"] if zc else []), source=src, header=_TGHDR)


_gpre, _gpost = {}, {}


def gates(h, msgs, downs: list, fgs: list, wt, m0: int, zc: bool = False) -> list:
    """h float32 [K, QH, D] (QH = 1: the same state for every gate); msgs float32 [K, 4, D]; downs: G down_proj matrices
    (bf16 bits [64, 4608]); fgs: G float32 [4608, 64] (forget | gain); wt float32 [G, D] norm weights -> G arrays uint16
    [K, D]: the gated, normalized states as half bits."""
    K, QH = h.shape[0], h.shape[1]
    G = len(downs)
    key = (G, QH, m0)
    kp = _gpre.get(key) or _gpre.setdefault(key, _make_gate_pre(*key))
    kq = _gpost.get(key + (zc,)) or _gpost.setdefault(key + (zc,), _make_gate_post(*key, zc=zc))
    st = kp(inputs=[h, msgs], grid=(K * G * 576, 1, 1), threadgroup=(576, 1, 1), output_shapes=[(K, 4608)] * G,
            output_dtypes=[mx.uint16] * G)
    fg = [gemv_small(fgs[i], gemv(downs[i], st[i])) for i in range(G)]
    return kq(inputs=[h, msgs, wt] + fg, grid=(K * G * 576, 1, 1), threadgroup=(576, 1, 1), output_shapes=[(K, 2304)] * G + [(16,)] * zc,
              output_dtypes=[mx.uint16] * G + [mx.uint32] * zc)


# static short convolution + residual (+ block residual) and the conv window for the next pass, one kernel
_SCONV2 = """
    uint g = thread_position_in_grid.x;
    uint row = g / (C / 4), c = (g % (C / 4)) * 4;
    if (row < 3) *((device float4*)(win + row * C + c)) = *((const device float4*)(st + row * C + c));
    if (row >= KR) return;
    float4 xv = *((const device float4*)(x + row * C + c));
    *((device float4*)(win + (row + 3) * C + c)) = xv;
    float4 acc = float4(0.0f);
    for (int j = 0; j < 4; j++) {
        int i = int(row) + j - 3;
        float4 wv = i < 0 ? *((const device float4*)(st + (3 + i) * C + c)) : *((const device float4*)(x + i * C + c));
        acc = fma(wv, *((const device float4*)(taps + j * C + c)), acc);
    }
    float4 y = (acc + xv) * 0.4472135954999579f;
    if (ADD) y += *((const device float4*)(add + row * C + c));
    *((device float4*)(out + row * C + c)) = y;
"""
_sconv2 = mx.fast.metal_kernel(name="mk_sconv2", input_names=["x", "st", "taps", "add"], output_names=["out", "win"], source=_SCONV2)


def sconv2(x, st, taps, add=None):
    """-> (out [K, C], win [K + 3, C]); out = (conv + x) / sqrt(5) (+ add)."""
    K, C = x.shape
    return _sconv2(inputs=[x, st, taps, add if add is not None else x], template=[("C", C), ("KR", K), ("ADD", add is not None)],
                   grid=((K + 3) * C // 4, 1, 1), threadgroup=(64, 1, 1), output_shapes=[(K, C), (K + 3, C)],
                   output_dtypes=[mx.float32, mx.float32])


# ------------------------------------------------------------------------------------------------- message gates, one kernel
def _make_gate_all(G: int, QH: int, M0: int):
    """G message gates of one looped layer in one kernel, one threadgroup of 576 threads per (row, gate):
    st = [unit-RMS(h), unit-RMS(m)]; dn = down_proj st (64); f, gain = tanh(forget / gain proj dn);
    y = half(rms_norm((1 - f) h + gain m) * wt)."""
    selw = "".join(f"    if (g == {i}) {{ dw = dw{i}; ft = ft{i}; }}\n" for i in range(1, G))
    wr = "".join(f"\n    if (g == {i}) *((device half4*)(y{i} + row * 2304 + d4)) = o;" for i in range(G))
    src = f"""
    uint grp = threadgroup_position_in_grid.x;
    uint row = grp / {G}, g = grp % {G};
    uint tid = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    uint d4 = tid * 4;
    threadgroup float buf[36];
    threadgroup float part[64 * 18];
    threadgroup float dnv[64];
    const device ushort* dw = dw0;
    const device float* ft = ft0;
{selw}
    float4 hv = *((const device float4*)(h + (row * {QH} + min(g, {QH - 1}u)) * 2304 + d4));
    float4 mv = *((const device float4*)(msgs + (row * 4 + {M0} + g) * 2304 + d4));
    float hh = tgsum(dot4(hv, hv), buf, lane, sg);
    float mm = tgsum(dot4(mv, mv), buf + 18, lane, sg);
    float4 hn = hv * metal::precise::rsqrt(hh * (1.0f / 2304.0f) + 1e-6f);
    float4 mn = mv * metal::precise::rsqrt(mm * (1.0f / 2304.0f) + 1e-6f);
    for (int n = 0; n < 64; n++) {{
        float4 wa = {bf4('dw + n * 4608 + d4')};
        float4 wb = {bf4('dw + n * 4608 + 2304 + d4')};
        float p = simd_sum(dot4(wa, hn) + dot4(wb, mn));
        if (lane == 0) part[n * 18 + sg] = p;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 64) {{
        float t = 0.0f;
        for (int i = 0; i < 18; i++) t += part[tid * 18 + i];
        dnv[tid] = t;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float4 fa = float4(0.0f), ga = float4(0.0f);
    for (int n = 0; n < 64; n++) {{
        float dv = dnv[n];
        fa = fma(*((const device float4*)(ft + n * 4608 + d4)), float4(dv), fa);
        ga = fma(*((const device float4*)(ft + n * 4608 + 2304 + d4)), float4(dv), ga);
    }}
    float4 v = (1.0f - metal::precise::tanh(fa)) * hv + metal::precise::tanh(ga) * mv;
    float vv = tgsum(dot4(v, v), buf, lane, sg);
    half4 o = half4(v * metal::precise::rsqrt(vv * (1.0f / 2304.0f) + 1e-6f) * *((const device float4*)(wt + g * 2304 + d4)));
{wr}
"""
    return mx.fast.metal_kernel(name=f"mk_gate_all_{G}_{QH}_{M0}",
                                input_names=["h", "msgs", "wt"] + [f"dw{i}" for i in range(G)] + [f"ft{i}" for i in range(G)],
                                output_names=[f"y{i}" for i in range(G)], source=src, header=_TGHDR)


_gall = {}


def gates2(h, msgs, downs: list, fts: list, wt, m0: int) -> list:
    """As gates(), one dispatch. fts: G float32 [64, 4608] (forget | gain projections, transposed)."""
    K, QH = h.shape[0], h.shape[1]
    G = len(downs)
    key = (G, QH, m0)
    k = _gall.get(key) or _gall.setdefault(key, _make_gate_all(*key))
    return k(inputs=[h, msgs, wt] + list(downs) + list(fts), grid=(K * G * 576, 1, 1), threadgroup=(576, 1, 1),
             output_shapes=[(K, 2304)] * G, output_dtypes=[mx.uint16] * G)


# ------------------------------------------------------------------------------------------------- exact argmax, short list
# A verify pass needs each row's argmax, not its logits. approx = the 8-bit copy's product (gemv8); for every output n
# |approx - exact| <= scale[n] / 2 * ||x||_1 (rounding of the copy) + float slop, so only outputs whose upper bound
# reaches the best lower bound of some row can be the argmax. Those are recomputed by exactly the GEMV arithmetic of
# mk_gemv (so the value is the bits step() computes); all others are written as -inf.
# One simdgroup per block of 32 outputs (a simdgroup per output would launch 6.4 M threads to recompute a handful): each
# lane writes -inf for its own output unless it is flagged; flagged outputs of the block are then computed one after the
# other by the whole simdgroup with the lane layout and fma order of mk_gemv (same bits).
_GEMV_FLAG = f"""
    uint blk = thread_position_in_grid.x / 32;
    uint lane = thread_index_in_simdgroup;
    uint n0 = blk * 32 + lane;
    uint f0 = n0 < N ? uint(flag[n0]) : 0u;
    if (n0 < N && f0 == 0u) {{ for (int k = 0; k < K; k++) out[k * N + n0] = -INFINITY; }}
    if (simd_or(f0) == 0u) return;
    for (uint j = 0; j < 32; j++) {{
        uint n = blk * 32 + j;
        if (n >= N || flag[n] == 0) continue;
        float4 acc[K];
        for (int k = 0; k < K; k++) acc[k] = float4(0.0f);
        const device ushort* wr = w + size_t(n) * D;
        for (int i = 0; i < D / 128; i++) {{
            int idx = (i * 32 + lane) * 4;
            float4 wf = {bf4('wr + idx')};
            for (int k = 0; k < K; k++) {{
                float4 xv = {hf4('x + k * D + idx')};
                acc[k] = fma(wf, xv, acc[k]);
            }}
        }}
        for (int k = 0; k < K; k++) {{
            float4 a = acc[k];
            float s = simd_sum((a.x + a.y) + (a.z + a.w));
            if (lane == 0) out[k * N + n] = s;
        }}
    }}
"""
_gemv_flag = mx.fast.metal_kernel(name="mk_gemv_flag", input_names=["w", "x", "flag"], output_names=["out"], source=_wi(_GEMV_FLAG))


def argmax_short(w16, w8, x16: mx.array, n_out: int, cap: float):
    """w16: BF16 bits [rows, D]; w8 = (int8 [n_out, D], scale [n_out]) its 8-bit copy; x16 uint16 [K, D] (half bits).
    -> (tokens uint32 [K] = first index of the maximum of cap * tanh(z / cap) over the exact z, candidates int32 [])."""
    K, D = x16.shape
    approx = gemv8(w8, x16)                                                    # [K, n_out]
    l1 = mx.abs(x16.view(mx.float16).astype(mx.float32)).sum(axis=-1, keepdims=True)
    eps = (0.51 * l1) * w8[1][None, :] + 0.02                                  # rigorous bound + slop (see above)
    lo = (approx - eps).max(axis=-1, keepdims=True)
    flag = ((approx + eps) >= lo).any(axis=0).astype(mx.uint8)                 # [n_out]
    z = _gemv_flag(inputs=[w16, x16, flag], template=[("D", D), ("N", n_out), ("K", K)], grid=((n_out + 31) // 32 * 32, 1, 1),
                   threadgroup=(32, 1, 1), output_shapes=[(K, n_out)], output_dtypes=[mx.float32])[0]
    logits = mx.where(z == -mx.inf, -mx.inf, mx.tanh(z * (1.0 / cap)) * cap)
    return mx.argmax(logits, axis=-1), flag.astype(mx.int32).sum()


def argmax_short_p(w16, w8, x16: mx.array, n_out: int, cap: float):
    """As argmax_short(), for draft-only readouts: also the probability of each row's largest logit under the 8-bit
    copy's logits (softcapped), float32 [K]: a confidence for the depth policy, not part of any token."""
    K, D = x16.shape
    approx = gemv8(w8, x16)                                                    # [K, n_out]
    l1 = mx.abs(x16.view(mx.float16).astype(mx.float32)).sum(axis=-1, keepdims=True)
    eps = (0.51 * l1) * w8[1][None, :] + 0.02
    lo = (approx - eps).max(axis=-1, keepdims=True)
    flag = ((approx + eps) >= lo).any(axis=0).astype(mx.uint8)                 # [n_out]
    z = _gemv_flag(inputs=[w16, x16, flag], template=[("D", D), ("N", n_out), ("K", K)], grid=((n_out + 31) // 32 * 32, 1, 1),
                   threadgroup=(32, 1, 1), output_shapes=[(K, n_out)], output_dtypes=[mx.float32])[0]
    logits = mx.where(z == -mx.inf, -mx.inf, mx.tanh(z * (1.0 / cap)) * cap)
    la = mx.tanh(approx * (1.0 / cap)) * cap
    pr = mx.exp(la.max(axis=-1) - mx.logsumexp(la, axis=-1))
    return mx.argmax(logits, axis=-1), pr


# draft-only 8-bit GEMV for short rows (the Markov fold, D = 256, one input row): one thread per output instead of
# one simdgroup (200058 x 32 threads of one iteration each were launch-bound: 0.33 ms for 51 MB)
_GEMV8N = f"""
    uint n = thread_position_in_grid.x;
    if (n >= N) return;
    const device int8_t* wr = w + size_t(n) * D;
    float4 a0 = float4(0.0f), a1 = float4(0.0f), a2 = float4(0.0f), a3 = float4(0.0f);
    for (int i = 0; i < D; i += 16) {{
        a0 = fma(float4(*((const device packed_char4*)(wr + i))), {hf4('x + i')}, a0);
        a1 = fma(float4(*((const device packed_char4*)(wr + i + 4))), {hf4('x + i + 4')}, a1);
        a2 = fma(float4(*((const device packed_char4*)(wr + i + 8))), {hf4('x + i + 8')}, a2);
        a3 = fma(float4(*((const device packed_char4*)(wr + i + 12))), {hf4('x + i + 12')}, a3);
    }}
    float4 a = (a0 + a1) + (a2 + a3);
    out[n] = ((a.x + a.y) + (a.z + a.w)) * scale[n];
"""
_gemv8n = mx.fast.metal_kernel(name="mk_gemv8_narrow", input_names=["w", "scale", "x"], output_names=["out"], source=_GEMV8N)


def gemv8_narrow(wq, x16: mx.array) -> mx.array:
    """wq = (int8 [N, D], scale [N]); x16 uint16 [1, D] (half bits), D % 16 == 0 -> float32 [1, N]."""
    w, sc = wq
    N, D = w.shape
    return _gemv8n(inputs=[w, sc, x16], template=[("D", D), ("N", N)], grid=((N + 1023) // 1024 * 1024, 1, 1),
                   threadgroup=(1024, 1, 1), output_shapes=[(1, N)], output_dtypes=[mx.float32])[0]


# ------------------------------------------------------------------------------------------------- lossless 12-bit BF16
# A BF16 weight is sign | 8-bit exponent | 7-bit mantissa. Within one output row the exponents of almost every weight
# fall inside 16 consecutive values, so a row is stored as 6 bytes per 4 weights: 4 bytes sign | mantissa and 4 nibbles
# exponent - lo[row]. The kernel rebuilds the identical 16 bits of every weight (pack12 checks the round trip), so
# every product and sum is the one of mk_gemv. A row with a weight outside its window (about one row in six) is kept as
# plain BF16 in a side array and read by the plain loop.
_GEMV12 = f"""
    uint n = thread_position_in_grid.x / 32;
    uint lane = thread_index_in_simdgroup;
    float4 acc[K];
    for (int k = 0; k < K; k++) acc[k] = float4(0.0f);
    int ri = rawidx[n];
    if (ri < 0) {{
        const device uchar* wr = w + size_t(n) * (D / 4 * 6);
        uint base = lo[n];
        for (int i = 0; i < D / 128; i++) {{
            int g = i * 32 + lane;
            uint4 mv = uint4(*((const device packed_uchar4*)(wr + g * 6)));
            uint ev = *((const device ushort*)(wr + g * 6 + 4));
            uint4 e4 = (uint4(ev) >> uint4(0u, 4u, 8u, 12u)) & 15u;
            float4 wf = as_type<float4>(((mv & 0x80u) << 24) | ((mv & 0x7Fu) << 16) | ((e4 << 23) + base));
            for (int k = 0; k < K; k++) {{
                float4 xv = {hf4('x + k * D + g * 4')};
                acc[k] = fma(wf, xv, acc[k]);
            }}
        }}
    }} else {{
        const device ushort* wr = raw + size_t(ri) * D;
        for (int i = 0; i < D / 128; i++) {{
            int idx = (i * 32 + lane) * 4;
            float4 wf = {bf4('wr + idx')};
            for (int k = 0; k < K; k++) {{
                float4 xv = {hf4('x + k * D + idx')};
                acc[k] = fma(wf, xv, acc[k]);
            }}
        }}
    }}
    for (int k = 0; k < K; k++) {{
        float4 a = acc[k];
        float s = simd_sum((a.x + a.y) + (a.z + a.w));
        if (lane == 0) out[k * N + n] = s;
    }}
"""
_gemv12 = mx.fast.metal_kernel(name="mk_gemv12", input_names=["w", "lo", "rawidx", "raw", "x"], output_names=["out"], source=_wi(_GEMV12))


def pack12(w16):
    """uint16 [N, D] (bf16 bits; numpy or mx) -> (w uint8 [N, D / 4 * 6], lo uint32 [N] (shifted), rawidx int32 [N],
    raw uint16 [R, D]); checked to decode to the identical bits."""
    import numpy as np
    b = np.array(w16, copy=False) if not isinstance(w16, np.ndarray) else w16
    N, D = b.shape
    e = ((b >> 7) & 0xFF).astype(np.int32)
    lo = np.maximum(e.max(axis=1) - 15, 0)
    esc = (e < lo[:, None]).any(axis=1)
    nib = np.clip(e - lo[:, None], 0, 15).astype(np.uint16)
    out = np.empty((N, D // 4, 6), dtype=np.uint8)
    out[:, :, :4] = (((b >> 8) & 0x80) | (b & 0x7F)).astype(np.uint8).reshape(N, D // 4, 4)
    n4 = nib.reshape(N, D // 4, 4)
    ev = (n4[:, :, 0] | (n4[:, :, 1] << 4) | (n4[:, :, 2] << 8) | (n4[:, :, 3] << 12)).astype(np.uint16)
    out[:, :, 4] = (ev & 0xFF).astype(np.uint8)
    out[:, :, 5] = (ev >> 8).astype(np.uint8)
    # round trip (rows kept packed)
    m = out[:, :, :4].reshape(N, D).astype(np.uint16)
    dec = ((m & 0x80) << 8) | (m & 0x7F) | ((nib + lo[:, None].astype(np.uint16)) << 7)
    assert (dec[~esc] == b[~esc]).all(), "12-bit code does not round-trip"
    rawidx = np.full((N,), -1, dtype=np.int32)
    rawidx[esc] = np.arange(int(esc.sum()), dtype=np.int32)
    raw = b[esc] if esc.any() else np.zeros((1, D), dtype=np.uint16)
    out[esc] = 0
    res = (mx.array(out.reshape(N, D // 4 * 6)), mx.array((lo.astype(np.uint32) << 23)), mx.array(rawidx), mx.array(np.ascontiguousarray(raw)))
    mx.eval(*res)
    return res


def gemv12(wp, x16: mx.array) -> mx.array:
    """wp = pack12(w16); x16 uint16 [K, D] (half bits) -> float32 [K, N], the bits of gemv(w16, x16)."""
    w, lo, rawidx, raw = wp
    N, D = w.shape[0], w.shape[1] // 6 * 4
    K = x16.shape[0]
    if K > GEMV_KMAX:
        return _blocked_run(_gemv12_b, [w, lo, rawidx, raw], x16, N, D)
    return _gemv12(inputs=[w, lo, rawidx, raw, x16], template=[("D", D), ("N", N), ("K", K)], grid=(N * 32, 1, 1),
                   threadgroup=(256 if N % 8 == 0 else 32, 1, 1), output_shapes=[(K, N)], output_dtypes=[mx.float32])[0]


# ------------------------------------------------------------------------------------------------- draft-only 5-bit GEMV
# The draft's matrices from a 5-bit per-row-scaled copy, 8 weights in 5 bytes (the draft only proposes; with 5 bits it
# proposes the same tokens as with 8 on every text tried, with 4 it does not: tests/nat.py). The target never uses this.
_GEMV5 = f"""
    uint n = thread_position_in_grid.x / 32;
    uint lane = thread_index_in_simdgroup;
    float4 acc[K];
    for (int k = 0; k < K; k++) acc[k] = float4(0.0f);
    const device uchar* wr = w + size_t(n) * (D / 8 * 5);
    for (int i = 0; i < D / 256; i++) {{
        int g = i * 32 + lane;
        uint4 b = uint4(*((const device packed_uchar4*)(wr + g * 5)));
        uint c = b.x | (b.y << 8) | (b.z << 16) | (b.w << 24);
        uint hi = wr[g * 5 + 4];
        uint4 u0 = (uint4(c) >> uint4(0u, 5u, 10u, 15u)) & 31u;
        uint4 u1 = uint4(c >> 20, c >> 25, (c >> 30) | (hi << 2), hi >> 3) & 31u;
        float4 w0 = float4(int4(u0) - 16), w1 = float4(int4(u1) - 16);
        for (int k = 0; k < K; k++) {{
            acc[k] = fma(w0, {hf4('x + k * D + g * 8')}, acc[k]);
            acc[k] = fma(w1, {hf4('x + k * D + g * 8 + 4')}, acc[k]);
        }}
    }}
    float sc = scale[n];
    for (int k = 0; k < K; k++) {{
        float4 a = acc[k];
        float s = simd_sum((a.x + a.y) + (a.z + a.w));
        if (lane == 0) out[k * N + n] = s * sc;
    }}
"""
_gemv5 = mx.fast.metal_kernel(name="mk_gemv5", input_names=["w", "scale", "x"], output_names=["out"], source=_wi(_GEMV5))


def quant5(w: mx.array):
    """float [N, D] (D % 8 == 0) -> (uint8 [N, D / 8 * 5], float32 [N]): symmetric per-row 5-bit copy, codes + 16
    packed 8 to 5 bytes (draft only)."""
    import numpy as np
    N, D = w.shape
    out_q, out_s = [], []
    sh = (np.arange(8, dtype=np.uint64) * np.uint64(5))[None, None, :]
    for s0 in range(0, N, 16384):
        wf = w[s0:s0 + 16384].astype(mx.float32)
        sc = mx.maximum(mx.abs(wf).max(axis=1), 1e-30) / 15.0
        q = (mx.round(wf / sc[:, None]) + 16).astype(mx.uint8)
        mx.eval(q, sc)
        u = np.array(q).reshape(-1, D // 8, 8).astype(np.uint64)
        v = (u << sh).sum(axis=-1, dtype=np.uint64)                       # 40 bits per 8 weights
        b = v.view(np.uint8).reshape(-1, D // 8, 8)[:, :, :5]
        out_q.append(np.ascontiguousarray(b).reshape(-1, D // 8 * 5)); out_s.append(sc)
    q, s = mx.array(np.concatenate(out_q)), mx.concatenate(out_s)
    mx.eval(q, s)
    return q, s


def gemv5(wq, x16: mx.array) -> mx.array:
    """wq = quant5(w); x16 uint16 [K, D] (half bits), D % 256 == 0 -> float32 [K, N]."""
    w, sc = wq
    N, D = w.shape[0], w.shape[1] // 5 * 8
    K = x16.shape[0]
    return _gemv5(inputs=[w, sc, x16], template=[("D", D), ("N", N), ("K", K)], grid=(N * 32, 1, 1),
                  threadgroup=(256 if N % 8 == 0 else 32, 1, 1), output_shapes=[(K, N)], output_dtypes=[mx.float32])[0]


# timing experiment only (MK_WHATIF_PAD): a dependent trivial dispatch, out = x
_ident = mx.fast.metal_kernel(name="mk_ident", input_names=["x"], output_names=["out"], source="""
    uint g = thread_position_in_grid.x;
    *((device float4*)(out + g * 4)) = *((const device float4*)(x + g * 4));
""")


def ident(x):
    return _ident(inputs=[x], grid=(x.size // 4, 1, 1), threadgroup=(64, 1, 1), output_shapes=[x.shape], output_dtypes=[mx.float32])[0]


# timing experiment only (MK_WHATIF_NO): a cheap dependent stand-in for a big kernel, out [K, N] from x [K, D] (wrong values)
_fill = mx.fast.metal_kernel(name="mk_fill", input_names=["x"], output_names=["out"], source="""
    uint g = thread_position_in_grid.x;
    uint k = g / N, n = g % N;
    out[g] = float(*((const device half*)(x + k * D + n % D))) * 0.01f;
""")


def fill(x16, N):
    K, D = x16.shape
    return _fill(inputs=[x16], template=[("N", N), ("D", D)], grid=(K * N, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(K, N)], output_dtypes=[mx.float32])[0]


# ------------------------------------------------------------------------------------------------- message gates, two stages fewer
# The gates of a looped layer were four dependent stages behind the depth mix (pre, down_proj, forget / gain
# projections, post), each a few microseconds of latency: 1.8 ms of a one-row pass (tests/whatif.py). Here the first is
# folded into the depth mix (mix2g: the mixed states and, per gate, the unit-RMS [h, m] as half bits) and the third into
# the last (gate_fgpost: each thread runs the forget / gain sums of its 4 dims, then the gate). Every sum is the
# statement sequence of the kernels it replaces (tgsum order, the scalar fma chain of gemv_small), so the bits are theirs.
def _make_mix2g(N: int, Q: int, pout: bool, G: int, M0: int):
    names = [f"b{i}" for i in range(N)]
    L = N - 1
    load = "\n".join(f"    float4 v{i} = *((const device float4*)({n} + row * 2304 + d4));" for i, n in enumerate(names))
    nred = N * Q + 1 + Q + G
    pre = f"""
    float hh = tgsum(dot4(v{L}, v{L}), buf + {N * Q * 18}, lane, sg);
    v{L} = v{L} * metal::precise::rsqrt(hh * (1.0f / 2304.0f) + 1e-6f);"""
    dots = "\n".join(f"    tgput(dot4(v{i}, *((const device float4*)(qs + {j} * 2304 + d4))), buf + {(i * Q + j) * 18}, lane, sg);"
                     for i in range(N) for j in range(Q))
    dots += "\n    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    dots += "\n".join(f"    float s{i}_{j} = tgget(buf + {(i * Q + j) * 18});" for i in range(N) for j in range(Q))
    outs = ""
    for j in range(Q):
        outs += f"    float4 o{j};\n    {{ float m = s0_{j};\n"
        for i in range(1, N):
            outs += f"      m = max(m, s{i}_{j});\n"
        for i in range(N):
            outs += f"      float e{i} = metal::precise::exp((s{i}_{j} - m) * SCALE);\n"
        outs += "      float z = e0" + "".join(f" + e{i}" for i in range(1, N)) + ";\n"
        outs += "      float4 o = v0 * (e0 / z);\n"
        for i in range(1, N):
            outs += f"      o = fma(v{i}, float4(e{i} / z), o);\n"
        outs += f"      *((device float4*)(out + (row * {Q} + {j}) * 2304 + d4)) = o; o{j} = o; }}\n"
    if pout:
        outs += f"    *((device float4*)(pn + row * 2304 + d4)) = v{L};\n"
    b0 = (N * Q + 1) * 18
    for g in range(G):
        outs += f"    float4 m{g} = *((const device float4*)(msgs + (row * 4 + {M0 + g}) * 2304 + d4));\n"
    for j in range(Q):
        outs += f"    tgput(dot4(o{j}, o{j}), buf + {b0 + j * 18}, lane, sg);\n"
    for g in range(G):
        outs += f"    tgput(dot4(m{g}, m{g}), buf + {b0 + (Q + g) * 18}, lane, sg);\n"
    outs += "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    for j in range(Q):
        outs += f"    float hh{j} = tgget(buf + {b0 + j * 18});\n"
    for g in range(G):
        j = min(g, Q - 1)
        outs += f"""    {{ float mm = tgget(buf + {b0 + (Q + g) * 18});
      half4 ho = half4(o{j} * metal::precise::rsqrt(hh{j} * (1.0f / 2304.0f) + 1e-6f));
      half4 mo = half4(m{g} * metal::precise::rsqrt(mm * (1.0f / 2304.0f) + 1e-6f));
      *((device half4*)(st{g} + row * 4608 + d4)) = ho; *((device half4*)(st{g} + row * 4608 + 2304 + d4)) = mo; }}\n"""
    src = f"""
    constexpr float SCALE = 0.020833333333333332f;      // 2304 ^ -0.5
    uint row = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    uint d4 = tid * 4;
    threadgroup float buf[{nred * 18}];
{load}
{pre}
{dots}
{outs}
"""
    return mx.fast.metal_kernel(name=f"mk_mix2g_{N}_{Q}_{int(pout)}_{G}_{M0}", input_names=names + ["qs", "msgs"],
                                output_names=["out"] + (["pn"] if pout else []) + [f"st{g}" for g in range(G)], source=src, header=_TGHDR2)


_mix2g = {}


def mix2g(sources: list, queries: mx.array, msgs: mx.array, G: int, m0: int, pout: bool = False):
    """Depth mix (normalized partial) + the gates' first stage. -> (out float32 [K, Q, D], pn [K, D] or None,
    [st_g uint16 [K, 4608] for the G gates])."""
    N, Q, K = len(sources), queries.shape[0], sources[0].shape[0]
    key = (N, Q, pout, G, m0)
    k = _mix2g.get(key) or _mix2g.setdefault(key, _make_mix2g(*key))
    shapes = [(K, Q, 2304)] + ([(K, 2304)] if pout else []) + [(K, 4608)] * G
    dts = [mx.float32] + ([mx.float32] if pout else []) + [mx.uint16] * G
    o = k(inputs=list(sources) + [queries, msgs], grid=(K * 576, 1, 1), threadgroup=(576, 1, 1), output_shapes=shapes, output_dtypes=dts)
    return o[0], (o[1] if pout else None), list(o[(2 if pout else 1):])


def _make_gate_fgpost(G: int, QH: int, M0: int):
    """y_g = half(rms_norm((1 - tanh(f_g)) h_g + tanh(gain_g) m_g) * wt_g) with f_g | gain_g = fw_g dn_g computed here:
    fw_g float32 [4608, 64], dn_g float32 [K, 64]; per output the scalar fma chain of gemv_small."""
    sel = "".join(f"    if (g == {i}) {{ dn = dn{i}; fw = fw{i}; }}\n" for i in range(1, G))
    chains = ""
    for half_, nm in ((0, "f"), (2304, "q")):
        for r in range(4):
            chains += f"""
            {{ float4 w = *((const device float4*)(fw + ({half_} + d4 + {r}) * 64 + i));
              {nm}{r} = fma(w.x, xv.x, {nm}{r}); {nm}{r} = fma(w.y, xv.y, {nm}{r}); {nm}{r} = fma(w.z, xv.z, {nm}{r}); {nm}{r} = fma(w.w, xv.w, {nm}{r}); }}"""
    src = f"""
    uint grp = threadgroup_position_in_grid.x;
    uint row = grp / {G}, g = grp % {G};
    uint tid = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    uint d4 = tid * 4;
    threadgroup float buf[18];
    const device float* dn = dn0;
    const device float* fw = fw0;
{sel}
    float f0 = 0.0f, f1 = 0.0f, f2 = 0.0f, f3 = 0.0f, q0 = 0.0f, q1 = 0.0f, q2 = 0.0f, q3 = 0.0f;
    for (int i = 0; i < 64; i += 4) {{
        float4 xv = *((const device float4*)(dn + row * 64 + i));
        {chains}
    }}
    float4 hv = *((const device float4*)(h + (row * {QH} + min(g, {QH - 1}u)) * 2304 + d4));
    float4 mv = *((const device float4*)(msgs + (row * 4 + {M0} + g) * 2304 + d4));
    float4 f = metal::precise::tanh(float4(f0, f1, f2, f3));
    float4 gn = metal::precise::tanh(float4(q0, q1, q2, q3));
    float4 v = (1.0f - f) * hv + gn * mv;
    float vv = tgsum(dot4(v, v), buf, lane, sg);
    half4 o = half4(v * metal::precise::rsqrt(vv * (1.0f / 2304.0f) + 1e-6f) * *((const device float4*)(wt + g * 2304 + d4)));
""" + "".join(f"""
    if (g == {i}) *((device half4*)(y{i} + row * 2304 + d4)) = o;""" for i in range(G))
    return mx.fast.metal_kernel(name=f"mk_gate_fgpost_{G}_{QH}_{M0}", input_names=["h", "msgs", "wt"] + [f"dn{i}" for i in range(G)] + [f"fw{i}" for i in range(G)],
                                output_names=[f"y{i}" for i in range(G)], source=src, header=_TGHDR)


_gfp = {}
import os as _os4
GATE_FGPOST = _os4.environ.get("MK_GATE_FGPOST", "0") == "1"


def gates3(st: list, h, msgs, downs: list, fgs: list, wt, m0: int) -> list:
    """The gates after mix2g: st from mix2g; h float32 [K, QH, D]; -> G arrays uint16 [K, D] (the bits of gates())."""
    K, QH = h.shape[0], h.shape[1]
    G = len(downs)
    key = (G, QH, m0)
    kq = _gfp.get(key) or _gfp.setdefault(key, _make_gate_fgpost(*key))
    if not GATE_FGPOST:
        # the forget / gain projections stay their own dispatch (one thread per output over many small threadgroups):
        # inside the 576-thread group of the last stage they run on as many GPU cores as there are (row, gate) groups,
        # and the pass is 1.2 ms slower
        kp = _gpost.get(key) or _gpost.setdefault(key, _make_gate_post(*key))
        fg = [gemv_small(fgs[i], gemv(downs[i], st[i])) for i in range(G)]
        return kp(inputs=[h, msgs, wt] + fg, grid=(K * G * 576, 1, 1), threadgroup=(576, 1, 1), output_shapes=[(K, 2304)] * G,
                  output_dtypes=[mx.uint16] * G)
    dn = [gemv(downs[i], st[i]) for i in range(G)]
    return kq(inputs=[h, msgs, wt] + dn + list(fgs), grid=(K * G * 576, 1, 1), threadgroup=(576, 1, 1), output_shapes=[(K, 2304)] * G,
              output_dtypes=[mx.uint16] * G)


# ------------------------------------------------------------------------------------------------- draft-only 5-bit GEMV
# The draft's matrices from a 5-bit per-row-scaled copy, 8 weights in 5 bytes (the draft only proposes; with 5 bits it
# proposes the same tokens as with 8 on every text tried, with 4 it does not: tests/nat.py). The target never uses this.
_GEMV5 = f"""
    uint n = thread_position_in_grid.x / 32;
    uint lane = thread_index_in_simdgroup;
    float4 acc[K];
    for (int k = 0; k < K; k++) acc[k] = float4(0.0f);
    const device uchar* wr = w + size_t(n) * (D / 8 * 5);
    for (int i = 0; i < D / 256; i++) {{
        int g = i * 32 + lane;
        uint4 b = uint4(*((const device packed_uchar4*)(wr + g * 5)));
        uint c = b.x | (b.y << 8) | (b.z << 16) | (b.w << 24);
        uint hi = wr[g * 5 + 4];
        uint4 u0 = (uint4(c) >> uint4(0u, 5u, 10u, 15u)) & 31u;
        uint4 u1 = uint4(c >> 20, c >> 25, (c >> 30) | (hi << 2), hi >> 3) & 31u;
        float4 w0 = float4(int4(u0) - 16), w1 = float4(int4(u1) - 16);
        for (int k = 0; k < K; k++) {{
            acc[k] = fma(w0, {hf4('x + k * D + g * 8')}, acc[k]);
            acc[k] = fma(w1, {hf4('x + k * D + g * 8 + 4')}, acc[k]);
        }}
    }}
    float sc = scale[n];
    for (int k = 0; k < K; k++) {{
        float4 a = acc[k];
        float s = simd_sum((a.x + a.y) + (a.z + a.w));
        if (lane == 0) out[k * N + n] = s * sc;
    }}
"""
_gemv5 = mx.fast.metal_kernel(name="mk_gemv5", input_names=["w", "scale", "x"], output_names=["out"], source=_wi(_GEMV5))


def quant5(w: mx.array):
    """float [N, D] (D % 8 == 0) -> (uint8 [N, D / 8 * 5], float32 [N]): symmetric per-row 5-bit copy, codes + 16
    packed 8 to 5 bytes (draft only)."""
    import numpy as np
    N, D = w.shape
    out_q, out_s = [], []
    sh = (np.arange(8, dtype=np.uint64) * np.uint64(5))[None, None, :]
    for s0 in range(0, N, 16384):
        wf = w[s0:s0 + 16384].astype(mx.float32)
        sc = mx.maximum(mx.abs(wf).max(axis=1), 1e-30) / 15.0
        q = (mx.round(wf / sc[:, None]) + 16).astype(mx.uint8)
        mx.eval(q, sc)
        u = np.array(q).reshape(-1, D // 8, 8).astype(np.uint64)
        v = (u << sh).sum(axis=-1, dtype=np.uint64)                       # 40 bits per 8 weights
        b = v.view(np.uint8).reshape(-1, D // 8, 8)[:, :, :5]
        out_q.append(np.ascontiguousarray(b).reshape(-1, D // 8 * 5)); out_s.append(sc)
    q, s = mx.array(np.concatenate(out_q)), mx.concatenate(out_s)
    mx.eval(q, s)
    return q, s


def gemv5(wq, x16: mx.array) -> mx.array:
    """wq = quant5(w); x16 uint16 [K, D] (half bits), D % 256 == 0 -> float32 [K, N]."""
    w, sc = wq
    N, D = w.shape[0], w.shape[1] // 5 * 8
    K = x16.shape[0]
    return _gemv5(inputs=[w, sc, x16], template=[("D", D), ("N", N), ("K", K)], grid=(N * 32, 1, 1),
                  threadgroup=(256 if N % 8 == 0 else 32, 1, 1), output_shapes=[(K, N)], output_dtypes=[mx.float32])[0]


# timing experiment only (MK_WHATIF_PAD): a dependent trivial dispatch, out = x
_ident = mx.fast.metal_kernel(name="mk_ident", input_names=["x"], output_names=["out"], source="""
    uint g = thread_position_in_grid.x;
    *((device float4*)(out + g * 4)) = *((const device float4*)(x + g * 4));
""")


def ident(x):
    return _ident(inputs=[x], grid=(x.size // 4, 1, 1), threadgroup=(64, 1, 1), output_shapes=[x.shape], output_dtypes=[mx.float32])[0]


# timing experiment only (MK_WHATIF_NO): a cheap dependent stand-in for a big kernel, out [K, N] from x [K, D] (wrong values)
_fill = mx.fast.metal_kernel(name="mk_fill", input_names=["x"], output_names=["out"], source="""
    uint g = thread_position_in_grid.x;
    uint k = g / N, n = g % N;
    out[g] = float(*((const device half*)(x + k * D + n % D))) * 0.01f;
""")


def fill(x16, N):
    K, D = x16.shape
    return _fill(inputs=[x16], template=[("N", N), ("D", D)], grid=(K * N, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(K, N)], output_dtypes=[mx.float32])[0]


# ------------------------------------------------------------------------------------------------- message gates, two stages fewer
# The gates of a looped layer were four dependent stages behind the depth mix (pre, down_proj, forget / gain
# projections, post), each a few microseconds of latency: 1.8 ms of a one-row pass (tests/whatif.py). Here the first is
# folded into the depth mix (mix2g: the mixed states and, per gate, the unit-RMS [h, m] as half bits) and the third into
# the last (gate_fgpost: each thread runs the forget / gain sums of its 4 dims, then the gate). Every sum is the
# statement sequence of the kernels it replaces (tgsum order, the scalar fma chain of gemv_small), so the bits are theirs.
def _make_mix2g(N: int, Q: int, pout: bool, G: int, M0: int):
    names = [f"b{i}" for i in range(N)]
    L = N - 1
    load = "\n".join(f"    float4 v{i} = *((const device float4*)({n} + row * 2304 + d4));" for i, n in enumerate(names))
    nred = N * Q + 1 + Q + G
    pre = f"""
    float hh = tgsum(dot4(v{L}, v{L}), buf + {N * Q * 18}, lane, sg);
    v{L} = v{L} * metal::precise::rsqrt(hh * (1.0f / 2304.0f) + 1e-6f);"""
    dots = "\n".join(f"    tgput(dot4(v{i}, *((const device float4*)(qs + {j} * 2304 + d4))), buf + {(i * Q + j) * 18}, lane, sg);"
                     for i in range(N) for j in range(Q))
    dots += "\n    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    dots += "\n".join(f"    float s{i}_{j} = tgget(buf + {(i * Q + j) * 18});" for i in range(N) for j in range(Q))
    outs = ""
    for j in range(Q):
        outs += f"    float4 o{j};\n    {{ float m = s0_{j};\n"
        for i in range(1, N):
            outs += f"      m = max(m, s{i}_{j});\n"
        for i in range(N):
            outs += f"      float e{i} = metal::precise::exp((s{i}_{j} - m) * SCALE);\n"
        outs += "      float z = e0" + "".join(f" + e{i}" for i in range(1, N)) + ";\n"
        outs += "      float4 o = v0 * (e0 / z);\n"
        for i in range(1, N):
            outs += f"      o = fma(v{i}, float4(e{i} / z), o);\n"
        outs += f"      *((device float4*)(out + (row * {Q} + {j}) * 2304 + d4)) = o; o{j} = o; }}\n"
    if pout:
        outs += f"    *((device float4*)(pn + row * 2304 + d4)) = v{L};\n"
    b0 = (N * Q + 1) * 18
    for g in range(G):
        outs += f"    float4 m{g} = *((const device float4*)(msgs + (row * 4 + {M0 + g}) * 2304 + d4));\n"
    for j in range(Q):
        outs += f"    tgput(dot4(o{j}, o{j}), buf + {b0 + j * 18}, lane, sg);\n"
    for g in range(G):
        outs += f"    tgput(dot4(m{g}, m{g}), buf + {b0 + (Q + g) * 18}, lane, sg);\n"
    outs += "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    for j in range(Q):
        outs += f"    float hh{j} = tgget(buf + {b0 + j * 18});\n"
    for g in range(G):
        j = min(g, Q - 1)
        outs += f"""    {{ float mm = tgget(buf + {b0 + (Q + g) * 18});
      half4 ho = half4(o{j} * metal::precise::rsqrt(hh{j} * (1.0f / 2304.0f) + 1e-6f));
      half4 mo = half4(m{g} * metal::precise::rsqrt(mm * (1.0f / 2304.0f) + 1e-6f));
      *((device half4*)(st{g} + row * 4608 + d4)) = ho; *((device half4*)(st{g} + row * 4608 + 2304 + d4)) = mo; }}\n"""
    src = f"""
    constexpr float SCALE = 0.020833333333333332f;      // 2304 ^ -0.5
    uint row = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    uint d4 = tid * 4;
    threadgroup float buf[{nred * 18}];
{load}
{pre}
{dots}
{outs}
"""
    return mx.fast.metal_kernel(name=f"mk_mix2g_{N}_{Q}_{int(pout)}_{G}_{M0}", input_names=names + ["qs", "msgs"],
                                output_names=["out"] + (["pn"] if pout else []) + [f"st{g}" for g in range(G)], source=src, header=_TGHDR2)


_mix2g = {}


def mix2g(sources: list, queries: mx.array, msgs: mx.array, G: int, m0: int, pout: bool = False):
    """Depth mix (normalized partial) + the gates' first stage. -> (out float32 [K, Q, D], pn [K, D] or None,
    [st_g uint16 [K, 4608] for the G gates])."""
    N, Q, K = len(sources), queries.shape[0], sources[0].shape[0]
    key = (N, Q, pout, G, m0)
    k = _mix2g.get(key) or _mix2g.setdefault(key, _make_mix2g(*key))
    shapes = [(K, Q, 2304)] + ([(K, 2304)] if pout else []) + [(K, 4608)] * G
    dts = [mx.float32] + ([mx.float32] if pout else []) + [mx.uint16] * G
    o = k(inputs=list(sources) + [queries, msgs], grid=(K * 576, 1, 1), threadgroup=(576, 1, 1), output_shapes=shapes, output_dtypes=dts)
    return o[0], (o[1] if pout else None), list(o[(2 if pout else 1):])


def _make_gate_fgpost(G: int, QH: int, M0: int):
    """y_g = half(rms_norm((1 - tanh(f_g)) h_g + tanh(gain_g) m_g) * wt_g) with f_g | gain_g = fw_g dn_g computed here:
    fw_g float32 [4608, 64], dn_g float32 [K, 64]; per output the scalar fma chain of gemv_small."""
    sel = "".join(f"    if (g == {i}) {{ dn = dn{i}; fw = fw{i}; }}\n" for i in range(1, G))
    chains = ""
    for half_, nm in ((0, "f"), (2304, "q")):
        for r in range(4):
            chains += f"""
            {{ float4 w = *((const device float4*)(fw + ({half_} + d4 + {r}) * 64 + i));
              {nm}{r} = fma(w.x, xv.x, {nm}{r}); {nm}{r} = fma(w.y, xv.y, {nm}{r}); {nm}{r} = fma(w.z, xv.z, {nm}{r}); {nm}{r} = fma(w.w, xv.w, {nm}{r}); }}"""
    src = f"""
    uint grp = threadgroup_position_in_grid.x;
    uint row = grp / {G}, g = grp % {G};
    uint tid = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    uint d4 = tid * 4;
    threadgroup float buf[18];
    const device float* dn = dn0;
    const device float* fw = fw0;
{sel}
    float f0 = 0.0f, f1 = 0.0f, f2 = 0.0f, f3 = 0.0f, q0 = 0.0f, q1 = 0.0f, q2 = 0.0f, q3 = 0.0f;
    for (int i = 0; i < 64; i += 4) {{
        float4 xv = *((const device float4*)(dn + row * 64 + i));
        {chains}
    }}
    float4 hv = *((const device float4*)(h + (row * {QH} + min(g, {QH - 1}u)) * 2304 + d4));
    float4 mv = *((const device float4*)(msgs + (row * 4 + {M0} + g) * 2304 + d4));
    float4 f = metal::precise::tanh(float4(f0, f1, f2, f3));
    float4 gn = metal::precise::tanh(float4(q0, q1, q2, q3));
    float4 v = (1.0f - f) * hv + gn * mv;
    float vv = tgsum(dot4(v, v), buf, lane, sg);
    half4 o = half4(v * metal::precise::rsqrt(vv * (1.0f / 2304.0f) + 1e-6f) * *((const device float4*)(wt + g * 2304 + d4)));
""" + "".join(f"""
    if (g == {i}) *((device half4*)(y{i} + row * 2304 + d4)) = o;""" for i in range(G))
    return mx.fast.metal_kernel(name=f"mk_gate_fgpost_{G}_{QH}_{M0}", input_names=["h", "msgs", "wt"] + [f"dn{i}" for i in range(G)] + [f"fw{i}" for i in range(G)],
                                output_names=[f"y{i}" for i in range(G)], source=src, header=_TGHDR)


_gfp = {}
import os as _os4
GATE_FGPOST = _os4.environ.get("MK_GATE_FGPOST", "0") == "1"


def gates3(st: list, h, msgs, downs: list, fgs: list, wt, m0: int) -> list:
    """The gates after mix2g: st from mix2g; h float32 [K, QH, D]; -> G arrays uint16 [K, D] (the bits of gates())."""
    K, QH = h.shape[0], h.shape[1]
    G = len(downs)
    key = (G, QH, m0)
    kq = _gfp.get(key) or _gfp.setdefault(key, _make_gate_fgpost(*key))
    if not GATE_FGPOST:
        # the forget / gain projections stay their own dispatch (one thread per output over many small threadgroups):
        # inside the 576-thread group of the last stage they run on as many GPU cores as there are (row, gate) groups,
        # and the pass is 1.2 ms slower
        kp = _gpost.get(key) or _gpost.setdefault(key, _make_gate_post(*key))
        fg = [gemv_small(fgs[i], gemv(downs[i], st[i])) for i in range(G)]
        return kp(inputs=[h, msgs, wt] + fg, grid=(K * G * 576, 1, 1), threadgroup=(576, 1, 1), output_shapes=[(K, 2304)] * G,
                  output_dtypes=[mx.uint16] * G)
    dn = [gemv(downs[i], st[i]) for i in range(G)]
    return kq(inputs=[h, msgs, wt] + dn + list(fgs), grid=(K * G * 576, 1, 1), threadgroup=(576, 1, 1), output_shapes=[(K, 2304)] * G,
              output_dtypes=[mx.uint16] * G)


# ------------------------------------------------------------------------------------------------- experts, one weight row per simdgroup
# The gate/up kernel above keeps a gate row and an up row in one simdgroup (2 x M accumulators). Here a simdgroup takes
# one weight row and writes its scaled sum; a second, elementwise kernel applies the soft-capped SwiGLU to the pair
# of sums. Half the accumulators, so slots can hold more pairs and each expert is read closer to once.
def _make_up1(fmt: str, M: int, BR: bool):
    D, cb, sb = 2304, (1152 if fmt == "nv" else 2304), (144 if fmt == "nv" else 72)
    accs = [[f"a0_{i}" for i in range(M)]]
    xps = [f"xr{i}" for i in range(M)]
    decl = "float " + ", ".join(f"a0_{i} = 0.0f" for i in range(M)) + ";"
    xdecl = "\n".join(f"    uint p{i} = slot_p[s * {M} + {i}]; const device ushort* xr{i} = x + ((p{i} == 0xFFFFFFFFu ? p0 : p{i}) / 8) * {D};"
                      for i in range(M))
    fin = "".join(f"""
    {{ float v = simd_sum(a0_{i}) * gs; if (lane == 0 && p{i} != 0xFFFFFFFFu) out[p{i} * 2304 + r] = v; }}""" for i in range(M))
    src = f"""
    uint g0 = thread_position_in_grid.x / 32;
    uint lane = thread_index_in_simdgroup;
    uint s = g0 / 2304, r = g0 % 2304;
    uint e = slot_e[s];
    if (e == 0xFFFFFFFFu) return;
{xdecl}
    const device uchar* c0 = codes + (size_t(e) * 2304 + r) * {cb};
    const device uchar* s0 = scales + (size_t(e) * 2304 + r) * {sb};
    {decl}
    {_rows_br(fmt, D, M, 1, ["c0"], ["s0"], xps, accs, BR)}
    float gs = gsc[e];
    {fin}
"""
    return mx.fast.metal_kernel(name=f"mk_up1_{fmt}_{M}_{int(BR)}", input_names=["codes", "scales", "gsc", "x", "slot_e", "slot_p"],
                                output_names=["out"], source=src, header=_HDR)


_ACT = """
    uint g = thread_position_in_grid.x;
    float2 v = *((const device float2*)(raw + size_t(g) * 2));
    float gt = v.x, up = v.y;
    float gc = metal::precise::tanh(gt * 0.25f) * 4.0f;
    float uc = metal::precise::tanh(up * 0.04f) * 25.0f;
    float y = 1.0f / (1.0f + metal::precise::exp(metal::abs(gt)));
    float sg = gt < 0 ? y : 1.0f - y;
    out[g] = to_bf16(gc * sg * uc);
"""
_act = mx.fast.metal_kernel(name="mk_up_act", input_names=["raw"], output_names=["out"], source=_ACT, header=_HDR)
_up1 = {}


def experts3(fmt: str, w13, s13, g13, w2, s2, g2, x16, idx, wts, M: int, RD: int = 1, BR: bool = True) -> mx.array:
    """As experts2(), the gate/up product with one weight row per simdgroup (the same bits)."""
    K = x16.shape[0]
    S = min(8 * K, 128)
    slot_e, slot_p = _plan(inputs=[idx], template=[("KM", K), ("NP", 8 * K), ("M", M), ("S", S)], grid=(128, 1, 1),
                           threadgroup=(128, 1, 1), output_shapes=[(S,), (S, M)], output_dtypes=[mx.uint32, mx.uint32])
    ku = _up1.get((fmt, M, BR)) or _up1.setdefault((fmt, M, BR), _make_up1(fmt, M, BR))
    kd = _down2.get((fmt, M, RD, BR)) or _down2.setdefault((fmt, M, RD, BR), _make_down2(fmt, M, RD, BR))
    raw = ku(inputs=[w13, s13, g13, x16, slot_e, slot_p], grid=(S * 2304 * 32, 1, 1), threadgroup=(256, 1, 1),
             output_shapes=[(K * 8, 2304)], output_dtypes=[mx.float32])[0]
    h = _act(inputs=[raw], grid=(K * 8 * 1152, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(K * 8, 1152)], output_dtypes=[mx.uint16])[0]
    o = kd(inputs=[w2, s2, g2, h, slot_e, slot_p], grid=(S * (2304 // RD) * 32, 1, 1), threadgroup=(256, 1, 1),
           output_shapes=[(K * 8, 2304)], output_dtypes=[mx.float32])[0]
    return _comb(inputs=[o, wts], grid=(K * 2304, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(K, 2304)],
                 output_dtypes=[mx.float32])[0]


# ------------------------------------------------------------------------------------------------- wide passes (prefill)
# A prompt is prefilled in passes of tens to hundreds of rows. The GEMVs above keep one accumulator per row of the pass
# in registers (they spill from 32 rows on), so a wide pass runs them in blocks of GB rows: grid y = row block, each
# simdgroup the lane layout and fma order of the one-row kernel (the same bits per row).
import os as _osw2
GB = int(_osw2.environ.get("MK_GEMV_BLOCK", "8"))
GEMV_KMAX = 16


def _blocked(src: str) -> str:
    s = src.replace("uint lane = thread_index_in_simdgroup;",
                    "uint lane = thread_index_in_simdgroup;\n    uint k0 = thread_position_in_grid.y * K;", 1)
    s = s.replace("x + k * D", "x + (k0 + k) * D").replace("out[k * N + n]", "out[(k0 + k) * N + n]")
    assert "k0 + k" in s
    return s


_gemv_b = mx.fast.metal_kernel(name="mk_gemv_b", input_names=["w", "x"], output_names=["out"], source=_wi(_blocked(_GEMV)))
_gemv12_b = mx.fast.metal_kernel(name="mk_gemv12_b", input_names=["w", "lo", "rawidx", "raw", "x"], output_names=["out"],
                                 source=_wi(_blocked(_GEMV12)))


def _blocked_run(kern, ws: list, x16: mx.array, N: int, D: int) -> mx.array:
    K = x16.shape[0]
    Kp = -(-K // GB) * GB
    if Kp != K:
        x16 = mx.concatenate([x16, mx.zeros((Kp - K, D), dtype=x16.dtype)])
    o = kern(inputs=ws + [x16], template=[("D", D), ("N", N), ("K", GB)], grid=(N * 32, Kp // GB, 1),
             threadgroup=(256 if N % 8 == 0 else 32, 1, 1), output_shapes=[(Kp, N)], output_dtypes=[mx.float32])[0]
    return o[:K] if Kp != K else o


def experts_mm(fmt: str, w13, s13, g13, w2, s2, g2, x16, idx, wts) -> mx.array:
    """As experts(), through MLX's quantized gather matmul on the stored NVFP4 / MXFP8 codes (pairs sorted by expert;
    half inputs and outputs): not the bits of the one-row step (relative difference about 5e-4). The products are
    computed before the per-expert outer scale of NVFP4 (about 1e-4), so the inputs are scaled by 2^-XS first (exact
    for magnitudes above 2^(XS-14)) to keep the raw products inside the half range."""
    K = x16.shape[0]
    mode, gs, nb = ("nvfp4", 16, 4) if fmt == "nv" else ("mxfp8", 32, 8)
    xs_ = float(2.0 ** -EXP_XS) if fmt == "nv" else 1.0
    flat = idx.reshape(-1).astype(mx.int32)
    order = mx.argsort(flat)
    es = flat[order]
    xs = x16.view(mx.float16)[order // 8]
    if EXP_F32:
        xs, xs_ = xs.astype(mx.float32), 1.0
    elif xs_ != 1.0:
        xs = xs * mx.array(xs_, dtype=mx.float16)
    gu = mx.gather_qmm(xs[:, None, :], w13.view(mx.uint32), s13, None, rhs_indices=es, transpose=True, group_size=gs,
                       bits=nb, mode=mode, sorted_indices=True)[:, 0].astype(mx.float32)
    if EXP_STATS is not None:
        EXP_STATS.append(("up", mx.abs(gu).max()))
    gu = gu * (g13[es] / xs_)[:, None]
    gt, up = gu[:, 0::2], gu[:, 1::2]
    h = (mx.tanh(gt * 0.25) * 4.0) * mx.sigmoid(gt) * (mx.tanh(up * 0.04) * 25.0)
    h = h.astype(mx.float32) if EXP_F32 else (h * xs_).astype(mx.float16)
    dn = mx.gather_qmm(h[:, None, :], w2.view(mx.uint32), s2, None, rhs_indices=es, transpose=True, group_size=gs,
                       bits=nb, mode=mode, sorted_indices=True)[:, 0].astype(mx.float32)
    if EXP_STATS is not None:
        EXP_STATS.append(("down", mx.abs(dn).max()))
    dn = dn * (g2[es] / xs_ * wts.reshape(-1)[order])[:, None]
    return dn[mx.argsort(order)].reshape(K, 8, 2304).sum(axis=1)


EXP_XS = int(_osw2.environ.get("MK_EXP_XS", "8"))
EXP_F32 = _osw2.environ.get("MK_EXP_F32", "0") == "1"
EXP_STATS = None


# ------------------------------------------------------------------------------------------------- sampling
# score[k, n] = z[k, n] + Gumbel(seed, q0 + k, n): the noise of position q0 + k is a hash of (seed, position, token id),
# 24 uniform bits per value (never 0 or 1). One kernel instead of about fifteen array operations per row.
_GUMBEL = """
    uint g = thread_position_in_grid.x;
    uint row = g / N, n = g % N;
    uint key = ((seed[0] ^ (uint(q0[0] + int(row)) * 0x85EBCA6Bu)) * 0xC2B2AE35u) + 0x27D4EB2Fu;
    uint x = n * 0x9E3779B1u + key;
    x ^= x >> 16; x *= 0x85EBCA6Bu; x ^= x >> 13; x *= 0xC2B2AE35u; x ^= x >> 16;
    float u = (float(x >> 8) + 0.5f) * 5.9604644775390625e-08f;
    out[g] = z[g] - metal::precise::log(-metal::precise::log(u));
"""
_gumbel = mx.fast.metal_kernel(name="mk_gumbel", input_names=["z", "seed", "q0"], output_names=["out"], source=_GUMBEL)


def gumbel_add(z: mx.array, seed: mx.array, q0: mx.array) -> mx.array:
    """z float32 [K, N]; seed uint32 [1]; q0 int32 [1] (the position row 0 draws for) -> z + noise, float32 [K, N]."""
    K, N = z.shape
    return _gumbel(inputs=[z, seed, q0], template=[("N", N)], grid=(K * N, 1, 1), threadgroup=(256, 1, 1),
                   output_shapes=[(K, N)], output_dtypes=[mx.float32])[0]


# ------------------------------------------------------------------------------------------------- merged dispatches
# Dispatches that do not depend on each other, in one: the q | gate | rel | sink, k and v projections of a layer (three
# 12-bit GEMVs on three inputs) and the three message gates' two projections. Each output row runs the statements of
# its own kernel (mk_gemv12, mk_gemv, mk_gemv_small) on the same operands, so the bits are those of the separate
# dispatches; what goes is two (and four) dispatch latencies per layer visit.
_GEMV12_3 = f"""
    uint n = thread_position_in_grid.x / 32;
    uint lane = thread_index_in_simdgroup;
    const device uchar* w = w0;
    const device uint* lo = lo0;
    const device int* rawidx = ri0;
    const device ushort* raw = raw0;
    const device ushort* x = x0;
    uint nn = n;
    if (n >= N0 + N1) {{ w = w2; lo = lo2; rawidx = ri2; raw = raw2; x = x2; nn = n - (N0 + N1); }}
    else if (n >= N0) {{ w = w1; lo = lo1; rawidx = ri1; raw = raw1; x = x1; nn = n - N0; }}
    float4 acc[K];
    for (int k = 0; k < K; k++) acc[k] = float4(0.0f);
    int ri = rawidx[nn];
    if (ri < 0) {{
        const device uchar* wr = w + size_t(nn) * (D / 4 * 6);
        uint base = lo[nn];
        for (int i = 0; i < D / 128; i++) {{
            int g = i * 32 + lane;
            uint4 mv = uint4(*((const device packed_uchar4*)(wr + g * 6)));
            uint ev = *((const device ushort*)(wr + g * 6 + 4));
            uint4 e4 = (uint4(ev) >> uint4(0u, 4u, 8u, 12u)) & 15u;
            float4 wf = as_type<float4>(((mv & 0x80u) << 24) | ((mv & 0x7Fu) << 16) | ((e4 << 23) + base));
            for (int k = 0; k < K; k++) {{
                float4 xv = {hf4('x + k * D + g * 4')};
                acc[k] = fma(wf, xv, acc[k]);
            }}
        }}
    }} else {{
        const device ushort* wr = raw + size_t(ri) * D;
        for (int i = 0; i < D / 128; i++) {{
            int idx = (i * 32 + lane) * 4;
            float4 wf = {bf4('wr + idx')};
            for (int k = 0; k < K; k++) {{
                float4 xv = {hf4('x + k * D + idx')};
                acc[k] = fma(wf, xv, acc[k]);
            }}
        }}
    }}
    for (int k = 0; k < K; k++) {{
        float4 a = acc[k];
        float s = simd_sum((a.x + a.y) + (a.z + a.w));
        if (lane == 0) out[k * N + n] = s;
    }}
"""
_gemv12_3 = mx.fast.metal_kernel(name="mk_gemv12_3", input_names=["w0", "lo0", "ri0", "raw0", "w1", "lo1", "ri1", "raw1", "w2", "lo2", "ri2", "raw2",
                                                                    "x0", "x1", "x2"], output_names=["out"], source=_wi(_GEMV12_3))


def gemv12_3(p0, p1, p2, x0, x1, x2):
    """Three pack12 matrices on three inputs uint16 [K, D] (K <= GEMV_KMAX) -> the three float32 results, the bits of
    gemv12() on each."""
    N0, N1, N2 = p0[0].shape[0], p1[0].shape[0], p2[0].shape[0]
    D = p0[0].shape[1] // 6 * 4
    K = x0.shape[0]
    N = N0 + N1 + N2
    o = _gemv12_3(inputs=list(p0) + list(p1) + list(p2) + [x0, x1, x2], template=[("D", D), ("N", N), ("N0", N0), ("N1", N1), ("K", K)],
                  grid=(N * 32, 1, 1), threadgroup=(256 if N % 8 == 0 else 32, 1, 1), output_shapes=[(K, N)], output_dtypes=[mx.float32])[0]
    return o[:, :N0], o[:, N0:N0 + N1], o[:, N0 + N1:]


_GEMV_G = f"""
    uint n = thread_position_in_grid.x / 32;
    uint lane = thread_index_in_simdgroup;
    uint g = n / NG;
    const device ushort* x = g == 0 ? x0 : (g == 1 ? x1 : x2);
    float4 acc[K];
    for (int k = 0; k < K; k++) acc[k] = float4(0.0f);
    const device ushort* wr = w + size_t(n) * D;
    for (int i = 0; i < D / 128; i++) {{
        int idx = (i * 32 + lane) * 4;
        float4 wf = {bf4('wr + idx')};
        for (int k = 0; k < K; k++) {{
            float4 xv = {hf4('x + k * D + idx')};
            acc[k] = fma(wf, xv, acc[k]);
        }}
    }}
    for (int k = 0; k < K; k++) {{
        float4 a = acc[k];
        float s = simd_sum((a.x + a.y) + (a.z + a.w));
        if (lane == 0) out[k * N + n] = s;
    }}
"""
_gemv_g = mx.fast.metal_kernel(name="mk_gemv_g3", input_names=["w", "x0", "x1", "x2"], output_names=["out"], source=_wi(_GEMV_G))

_GEMV_SG = """
    uint gi = thread_position_in_grid.x;
    uint k = gi / N, n = gi % N;
    uint g = n / NF;
    const device float* xr = x + (k * G + g) * D;
    float acc = 0.0f;
    for (int d = 0; d < D / 4 * 4; d += 4) {
        float4 wv = *((const device float4*)(w + n * D + d));
        float4 xv = *((const device float4*)(xr + d));
        acc = fma(wv.x, xv.x, acc);
        acc = fma(wv.y, xv.y, acc);
        acc = fma(wv.z, xv.z, acc);
        acc = fma(wv.w, xv.w, acc);
    }
    for (int d = D / 4 * 4; d < D; d++) acc = fma(w[n * D + d], xr[d], acc);
    out[gi] = acc;
"""
_gemv_sg = mx.fast.metal_kernel(name="mk_gemv_small_g", input_names=["w", "x"], output_names=["out"], source=_GEMV_SG)


def _make_gate_post_m(G: int, QH: int, M0: int):
    """_make_gate_post with the G forget | gain arrays as one [K, G * 4608]."""
    src = f"""
    uint grp = threadgroup_position_in_grid.x;
    uint row = grp / {G}, g = grp % {G};
    uint tid = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    uint d4 = tid * 4;
    threadgroup float buf[18];
    const device float* fg = fga + (row * {G} + g) * 4608;
    float4 hv = *((const device float4*)(h + (row * {QH} + min(g, {QH - 1}u)) * 2304 + d4));
    float4 mv = *((const device float4*)(msgs + (row * 4 + {M0} + g) * 2304 + d4));
    float4 f = metal::precise::tanh(*((const device float4*)(fg + d4)));
    float4 gn = metal::precise::tanh(*((const device float4*)(fg + 2304 + d4)));
    float4 v = (1.0f - f) * hv + gn * mv;
    float vv = tgsum(dot4(v, v), buf, lane, sg);
    half4 o = half4(v * metal::precise::rsqrt(vv * (1.0f / 2304.0f) + 1e-6f) * *((const device float4*)(wt + g * 2304 + d4)));
""" + "".join(f"""
    if (g == {i}) *((device half4*)(y{i} + row * 2304 + d4)) = o;""" for i in range(G))
    return mx.fast.metal_kernel(name=f"mk_gate_post_m_{G}_{QH}_{M0}", input_names=["h", "msgs", "wt", "fga"],
                                output_names=[f"y{i}" for i in range(G)], source=src, header=_TGHDR)


_gpost_m = {}


def gates_m(h, msgs, down_s, fg_s, wt, m0: int) -> list:
    """gates() for G = 3 with the three down projections in one dispatch and the three forget | gain projections in
    one: down_s uint16 [3 * 64, 4608] and fg_s float32 [3 * 4608, 64] are the three matrices stacked."""
    K, QH = h.shape[0], h.shape[1]
    G = 3
    key = (G, QH, m0)
    kp = _gpre.get(key) or _gpre.setdefault(key, _make_gate_pre(*key))
    kq = _gpost_m.get(key) or _gpost_m.setdefault(key, _make_gate_post_m(*key))
    st = kp(inputs=[h, msgs], grid=(K * G * 576, 1, 1), threadgroup=(576, 1, 1), output_shapes=[(K, 4608)] * G,
            output_dtypes=[mx.uint16] * G)
    dn = _gemv_g(inputs=[down_s] + list(st), template=[("D", 4608), ("N", G * 64), ("NG", 64), ("K", K)], grid=(G * 64 * 32, 1, 1),
                 threadgroup=(256, 1, 1), output_shapes=[(K, G * 64)], output_dtypes=[mx.float32])[0]
    fg = _gemv_sg(inputs=[fg_s, dn], template=[("D", 64), ("N", G * 4608), ("NF", 4608), ("G", G)], grid=(K * G * 4608, 1, 1),
                  threadgroup=(32, 1, 1), output_shapes=[(K, G * 4608)], output_dtypes=[mx.float32])[0]
    return kq(inputs=[h, msgs, wt, fg], grid=(K * G * 576, 1, 1), threadgroup=(576, 1, 1), output_shapes=[(K, 2304)] * G,
              output_dtypes=[mx.uint16] * G)


# ------------------------------------------------------------------------------------------------- router, one dispatch
# The router GEMV, the top-8 choice and the experts' slot plan in one dispatch (three dependent ones before): the GEMV's
# threadgroups as mk_gemv (one simdgroup per expert logit, the same lane layout and fma order: the same bits); each
# threadgroup then counts itself done on a device counter, and the last one to finish reads the 128 logits of every row
# and runs mk_router's statements (one simdgroup per row) and mk_plan's (one thread per expert) on them. The counter is
# a uint32 [16] (element 0) that the kernel producing the router's input zeroes (an extra output of it: no fill dispatch).
# Same bits (k2/bench.py: K = 1..16, 2 and 5 passes); in dependent chains of mix2 + router + experts 1.5 us faster per
# visit at one row (kmicro.py). Off until a pass measures faster on a quiet GPU (the full-pass timings of this change
# were taken on a GPU saturated by other processes and were within noise). The counter's device-scope fences are cheap
# with 16 threadgroups; folding mk_comb into the down kernel the same way (a fence in each of ~1000 threadgroups) was
# 18 us slower per visit and is not kept.
ROUTE1 = _osw2.environ.get("MK_ROUTE1", "1") == "1"
_ROUTE = f"""
    uint n = thread_position_in_grid.x / 32;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    uint tid = thread_position_in_threadgroup.x;
    {{
        float4 acc[K];
        for (int k = 0; k < K; k++) acc[k] = float4(0.0f);
        const device ushort* wr = w + size_t(n) * D;
        for (int i = 0; i < D / 128; i++) {{
            int idx = (i * 32 + lane) * 4;
            float4 wf = {bf4('wr + idx')};
            for (int k = 0; k < K; k++) {{
                float4 xv = {hf4('x + k * D + idx')};
                acc[k] = fma(wf, xv, acc[k]);
            }}
        }}
        for (int k = 0; k < K; k++) {{
            float4 a = acc[k];
            float s = simd_sum((a.x + a.y) + (a.z + a.w));
            if (lane == 0) lg[k * 128 + n] = s;
        }}
    }}
    threadgroup uint last;
    threadgroup uint tidx[K * 8];
    threadgroup_barrier(mem_flags::mem_device);
    if (tid == 0) {{
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
        last = atomic_fetch_add_explicit((device atomic_uint*)zc, 1u, memory_order_relaxed) == NTG - 1;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (last == 0) return;
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    for (uint row = sg; row < K; row += NSG) {{
        const volatile device float* logits = lg;
        float sc[4];
        float bi[4];
        for (int i = 0; i < 4; i++) {{
            uint e = lane + 32 * i;
            float x = logits[row * 128 + e];
            float y = 1.0f / (1.0f + metal::precise::exp(metal::abs(x)));
            sc[i] = x < 0 ? y : 1.0f - y;
            bi[i] = sc[i] + bias[e];
        }}
        float wsel[8];
        uint esel[8];
        for (int k = 0; k < 8; k++) {{
            float best = -INFINITY;
            uint be = 128;
            for (int i = 0; i < 4; i++) {{
                if (bi[i] > best) {{ best = bi[i]; be = lane + 32 * i; }}
            }}
            float top = simd_max(best);
            uint win = simd_min(best == top ? be : 128u);
            float s = 0.0f;
            for (int i = 0; i < 4; i++) {{
                if (lane + 32 * i == win) {{ s = sc[i]; bi[i] = -INFINITY; }}
            }}
            wsel[k] = simd_sum(s);
            esel[k] = win;
        }}
        float total = 0.0f;
        for (int k = 0; k < 8; k++) total += wsel[k];
        if (lane == 0) {{
            for (int k = 0; k < 8; k++) {{
                idx[row * 8 + k] = esel[k];
                tidx[row * 8 + k] = esel[k];
                wts[row * 8 + k] = wsel[k] / (total + 1e-20f) * 8.0f;
            }}
        }}
    }}
    if (PLAN == 0) return;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup uint cnt[128];
    uint e = tid;
    uint pairs[K];
    uint m = 0;
    uint ns = 0;
    if (e < 128) {{
        for (uint p = 0; p < K * 8; p += 4) {{
            vec<uint, 4> v = *((threadgroup packed_uint4*)(tidx + p));
            if (v.x == e && m < K) {{ pairs[m] = p; m++; }}
            if (v.y == e && m < K) {{ pairs[m] = p + 1; m++; }}
            if (v.z == e && m < K) {{ pairs[m] = p + 2; m++; }}
            if (v.w == e && m < K) {{ pairs[m] = p + 3; m++; }}
        }}
        ns = (m + M - 1) / M;
        cnt[e] = ns;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (e >= 128) return;
    vec<uint, 4> offv = vec<uint, 4>(0u), totv = vec<uint, 4>(0u);
    for (uint i = 0; i < 128; i += 4) {{
        vec<uint, 4> c4 = *((threadgroup packed_uint4*)(cnt + i));
        totv += c4;
        offv += select(vec<uint, 4>(0u), c4, (vec<uint, 4>(i) + vec<uint, 4>(0u, 1u, 2u, 3u)) < vec<uint, 4>(e));
    }}
    uint off = offv.x + offv.y + offv.z + offv.w, total = totv.x + totv.y + totv.z + totv.w;
    for (uint s = 0; s < ns; s++) {{
        slot_e[off + s] = e;
        for (uint i = 0; i < M; i++) slot_p[(off + s) * M + i] = (s * M + i < m) ? pairs[s * M + i] : 0xFFFFFFFFu;
    }}
    for (uint s = total + e; s < S; s += 128) {{
        slot_e[s] = 0xFFFFFFFFu;
        for (uint i = 0; i < M; i++) slot_p[s * M + i] = 0xFFFFFFFFu;
    }}
"""
_route = mx.fast.metal_kernel(name="mk_route", input_names=["w", "x", "bias", "zc"], output_names=["lg", "idx", "wts", "slot_e", "slot_p"],
                              source=_wi(_ROUTE))


def route(w16, x16, bias, zc, M: int = 0):
    """router(gemv(w16, x16), bias) in one dispatch; zc uint32 [16] (zc[0] zero; 16 elements so that MLX binds it as a
    device buffer). M > 0: also experts2's slot plan for slots of M pairs (experts2(plan=...)). -> (idx, wts, slot_e,
    slot_p) (the last two None without M)."""
    K, D = x16.shape
    S = min(8 * K, 128)
    TG = 256
    o = _route(inputs=[w16, x16, bias, zc], template=[("D", D), ("K", K), ("NTG", 128 * 32 // TG), ("NSG", TG // 32), ("PLAN", int(M > 0)),
                                                       ("M", max(M, 1)), ("S", S)],
               grid=(128 * 32, 1, 1), threadgroup=(TG, 1, 1), output_shapes=[(K, 128), (K, 8), (K, 8), (S,), (S, max(M, 1))],
               output_dtypes=[mx.float32, mx.uint32, mx.float32, mx.uint32, mx.uint32])
    return o[1], o[2], (o[3] if M else None), (o[4] if M else None)


# ------------------------------------------------------------------------------------------------- messages of a pass, one dispatch
# The cross-pass messages of a loop pass: J looped layers, each the depth mix of the same sources under its own Q = 4
# queries. mix2's statements (one threadgroup per (row, layer), the same per-row sums), all J layers in one dispatch.
# Same bits; 12 message mixes in one dispatch: ~50 us less per loop pass at one row in a dependent microbenchmark
# (kmicro.py: 115 -> 62 us); off until a pass measures faster on a quiet GPU (see ROUTE1).
MSGM = _osw2.environ.get("MK_MSGM", "1") == "1"
_mixj = {}


def mix_all(sources: list, queries: mx.array) -> mx.array:
    """sources: N arrays float32 [K, D]; queries float32 [J, Q, D] -> float32 [J, K, Q, D]; [j] = mix2(sources, queries[j])."""
    N, K = len(sources), sources[0].shape[0]
    J, Q = queries.shape[0], queries.shape[1]
    key = (N, Q)
    k = _mixj.get(key)
    if k is None:           # _make_mix2(N, Q, False, False, False) with the layer from the threadgroup index
        names = [f"b{i}" for i in range(N)]
        load = "\n".join(f"    float4 v{i} = *((const device float4*)({n} + row * 2304 + d4));" for i, n in enumerate(names))
        dots = "\n".join(f"    tgput(dot4(v{i}, *((const device float4*)(qs + {j} * 2304 + d4))), buf + {(i * Q + j) * 18}, lane, sg);"
                         for i in range(N) for j in range(Q))
        dots += "\n    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        dots += "\n".join(f"    float s{i}_{j} = tgget(buf + {(i * Q + j) * 18});" for i in range(N) for j in range(Q))
        outs = ""
        for j in range(Q):
            outs += f"    {{ float m = s0_{j};\n"
            for i in range(1, N):
                outs += f"      m = max(m, s{i}_{j});\n"
            for i in range(N):
                outs += f"      float e{i} = metal::precise::exp((s{i}_{j} - m) * SCALE);\n"
            outs += "      float z = e0" + "".join(f" + e{i}" for i in range(1, N)) + ";\n"
            outs += "      float4 o = v0 * (e0 / z);\n"
            for i in range(1, N):
                outs += f"      o = fma(v{i}, float4(e{i} / z), o);\n"
            outs += f"      *((device float4*)(out + ((jq * KR + row) * {Q} + {j}) * 2304 + d4)) = o; }}\n"
        src = f"""
    constexpr float SCALE = 0.020833333333333332f;      // 2304 ^ -0.5
    uint row = threadgroup_position_in_grid.x % KR;
    uint jq = threadgroup_position_in_grid.x / KR;
    const device float* qs = qa + jq * {Q * 2304};
    uint tid = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    uint d4 = tid * 4;
    threadgroup float buf[{N * Q * 18}];
{load}
{dots}
{outs}
"""
        k = _mixj[key] = mx.fast.metal_kernel(name=f"mk_mixj_{N}_{Q}", input_names=names + ["qa"], output_names=["out"], source=src,
                                              header=_TGHDR2)
    return k(inputs=list(sources) + [queries], template=[("KR", K)], grid=(J * K * 576, 1, 1), threadgroup=(576, 1, 1),
             output_shapes=[(J, K, Q, 2304)], output_dtypes=[mx.float32])[0]

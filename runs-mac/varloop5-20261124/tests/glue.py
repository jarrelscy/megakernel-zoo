"""Latency of the small ("glue") kernels at one row: N dependent calls in one graph. python tests/glue.py"""
import os, sys, time
from pathlib import Path
import numpy as np
WS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WS / "engine"))
os.environ["MK_SPEC"] = "0"
MD = str(Path.home() / "Models/Inkling/TML-d24-varloop-5pass")
import mlx.core as mx
import mk_kernels as kn
from api import Engine
eng = Engine(MD, 1, 2048)
tg = eng.tg
L = tg.layers[8]            # a looped, full-attention? layer
N = 200
def bench(name, f, x0):
    ts = []
    for _ in range(6):
        x = x0
        for _ in range(N):
            x = f(x)
        mx.synchronize(); t0 = time.perf_counter(); mx.eval(x); ts.append(time.perf_counter() - t0)
    print(f"{name:34s} {1e6 * min(ts[1:]) / N:7.1f} us per call", flush=True)
for K in (1, 4):
    print("K =", K)
    h3 = mx.random.normal((K, 3, 2304)); h1 = mx.random.normal((K, 1, 2304)); msgs = mx.random.normal((K, 4, 2304)); mx.eval(h3, h1, msgs)
    x = mx.random.normal((K, 2304)); mx.eval(x)
    bench("ident (trivial dispatch)", lambda v: kn.ident(v), x)
    bench("astype f32->f16->f32 (MLX ops)", lambda v: v.astype(mx.float16).astype(mx.float32), x)
    G = 3 if L["full"] else 1
    downs, fgs, wt = (L["c_down3"], L["c_fg3"], L["anw3"]) if L["full"] else (L["c_down3"][:1], L["c_fg3"][:1], L["anw3"][:1])
    hh = h3 if L["full"] else h1
    def g_attn(v):
        ys = kn.gates(hh + v[:, None, :] * 0.0, msgs, L["c_down3"], L["c_fg3"], L["anw3"], 0)
        return ys[0].view(mx.float16).astype(mx.float32)
    bench("gates G=3 (attention input)", g_attn, x)
    def g_mlp(v):
        ys = kn.gates(h1 + v[:, None, :] * 0.0, msgs, L["c_down1"], L["c_fg1"], L["mnw"], 3)
        return ys[0].view(mx.float16).astype(mx.float32)
    bench("gates G=1 (mlp input)", g_mlp, x)
    key = (3, hh.shape[1], 0)
    kp = kn._gpre.get(key) or kn._gpre.setdefault(key, kn._make_gate_pre(*key))
    def g_pre(v):
        st = kp(inputs=[hh + v[:, None, :] * 0.0, msgs], grid=(K * 3 * 576, 1, 1), threadgroup=(576, 1, 1), output_shapes=[(K, 4608)] * 3, output_dtypes=[mx.uint16] * 3)
        return st[0][:, :2304].view(mx.float16).astype(mx.float32)
    bench("  gate_pre G=3 alone", g_pre, x)
    kq = kn._gpost.get(key) or kn._gpost.setdefault(key, kn._make_gate_post(*key))
    fg = [mx.random.normal((K, 4608)) for _ in range(3)]; mx.eval(fg)
    def g_post(v):
        ys = kq(inputs=[hh + v[:, None, :] * 0.0, msgs, L["anw3"]] + fg, grid=(K * 3 * 576, 1, 1), threadgroup=(576, 1, 1), output_shapes=[(K, 2304)] * 3, output_dtypes=[mx.uint16] * 3)
        return ys[0].view(mx.float16).astype(mx.float32)
    bench("  gate_post G=3 alone", g_post, x)
    st0 = kn.bits(mx.random.normal((K, 4608))); mx.eval(st0)
    def g_proj(v):
        d = kn.gemv(L["c_down3"][0], kn.bits(mx.concatenate([v, v], axis=1)))
        f = kn.gemv_small(L["c_fg3"][0], d)
        return f[:, :2304]
    bench("  gemv down + gemv_small fg (1 gate)", g_proj, x)
    srcs = [mx.random.normal((K, 2304)) for _ in range(4)]; mx.eval(srcs)
    aq = L["aq2"]
    def m2(v):
        y = kn.mix2(srcs + [v], aq, norm_in=True, wt=L["anw"])
        return y[:, 0].view(mx.float16).astype(mx.float32)
    bench(f"mix2 N=5 Q={aq.shape[0]} (wt)", m2, x)
    def m2b(v):
        y = kn.mix2(srcs + [v], aq, norm_in=True)
        return y[:, 0]
    bench(f"mix2 N=5 Q={aq.shape[0]} (no wt)", m2b, x)
    def mm(v):
        return kn.mix(srcs + [v], tg.msg_query[3])[:, 0]
    bench("mix (msgs) N=5 Q=4", mm, x)
    st3 = mx.zeros((3, 2304))
    bench("sconv2", lambda v: kn.sconv2(v, st3, L["m_conv"], add=v)[0], x)
    bench("router gemv + router", lambda v: v + 0.0 * kn.router(kn.gemv(L["router"], kn.bits(v)), L["rbias"])[1][:, :1], x)

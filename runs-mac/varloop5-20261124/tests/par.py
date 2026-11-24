"""Do independent chains of dispatches overlap on the GPU? One chain of N dependent calls against two independent
chains of N in one eval. python tests/par.py"""
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
L = tg.layers[9]
N = 150
K = 1
msgs = mx.random.normal((K, 4, 2304)); mx.eval(msgs)
def gate_step(v):
    h = mx.stack([v, v, v], axis=1) if L["full"] else v[:, None, :]
    ys = kn.gates(h, msgs, L["c_down3"], L["c_fg3"], L["anw3"], 0)
    return ys[0].view(mx.float16).astype(mx.float32)
def dense_step(v):
    o = kn.gemv12(L["qgrs12"], kn.bits(v))
    return o[:, :2304]
def exp_step(v):
    x16 = kn.bits(v)
    idx, wts = kn.router(kn.gemv(L["router"], x16), L["rbias"])
    import mk_target as mt
    return kn.experts2(L["fmt"], L["w13"], L["s13"], L["g13"], L["w2"], L["s2"], L["g2"], x16, idx, wts, *mt.expert_layout(K)) * 0.01 + v
def chain(f, v, n):
    for _ in range(n): v = f(v)
    return v
xs = [mx.random.normal((K, 2304)) for _ in range(4)]; mx.eval(xs)
for name, f in (("gates G=3 (small kernels)", gate_step), ("dense q|gate|rel|sink GEMV (30 MB)", dense_step), ("router + experts (63 MB)", exp_step)):
    res = []
    for m in (1, 2, 4):
        ts = []
        for _ in range(5):
            outs = [chain(f, xs[i], N) for i in range(m)]
            mx.synchronize(); t0 = time.perf_counter(); mx.eval(*outs); ts.append(time.perf_counter() - t0)
        res.append(1e6 * min(ts[1:]) / N)
    print(f"{name}: per step of one chain {res[0]:.1f} us; two independent chains {res[1]:.1f} us; four {res[2]:.1f} us", flush=True)

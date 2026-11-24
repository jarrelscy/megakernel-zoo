"""Merged dispatches against the separate ones: bit equality. python tests/merge.py"""
import os, sys
from pathlib import Path
import numpy as np
WS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WS / "engine"))
os.environ["MK_SPEC"] = "0"
MD = str(Path.home() / "Models/Inkling/TML-d24-varloop-5pass")
import mlx.core as mx
import mk_kernels as kn
import mk_target as mt
from api import Engine
eng = Engine(MD, 1, 2048)
tg = eng.tg
eq = lambda a, b: bool(mx.array_equal(a.view(mx.uint32) if a.dtype == mx.float32 else a, b.view(mx.uint32) if b.dtype == mx.float32 else b).item())
ok = True
for li in (0, 3, 7, 11, 20, 23):
    L = tg.layers[li]
    for K in (1, 2, 5, 16):
        xs = [kn.bits(mx.random.normal((K, 2304)) * (0.3 + i)) for i in range(3)]
        a = [kn.gemv12(L[nm + "12"], x) for nm, x in zip(("qgrs", "wk", "wv"), xs)]
        b = kn.gemv12_3(L["qgrs12"], L["wk12"], L["wv12"], *xs)
        r = [eq(u, v) for u, v in zip(a, b)]
        ok = ok and all(r)
        if "c_down3" in L:
            h = mx.random.normal((K, 3, 2304)); msgs = mx.random.normal((K, 4, 2304))
            g0 = kn.gates(h, msgs, L["c_down3"], L["c_fg3"], L["anw3"], 0)
            g1 = kn.gates_m(h, msgs, L["c_down3s"], L["c_fg3s"], L["anw3"], 0)
            h1 = mx.random.normal((K, 1, 2304))
            g2 = kn.gates(h1, msgs, L["c_down3"], L["c_fg3"], L["anw3"], 0)
            g3 = kn.gates_m(h1, msgs, L["c_down3s"], L["c_fg3s"], L["anw3"], 0)
            r += [eq(u, v) for u, v in zip(g0, g1)] + [eq(u, v) for u, v in zip(g2, g3)]
            ok = ok and all(r)
        print(li, K, r, flush=True)
# whole passes: merged against separate, same state
rng = np.random.default_rng(1)
eng._hard_reset(); eng.prefill(rng.integers(0, 200000, size=90).tolist())
mf = lambda: (mx.array([tg.pos, 0, tg.base], dtype=mx.int32), mx.array([tg.pos, 1024, tg.base], dtype=mx.int32))
for K in (1, 3, 8, 16):
    toks = mx.array(rng.integers(0, 200000, size=K).astype(np.int32))
    outs = []
    for d3, gm in ((False, False), (True, True)):
        mt.DENSE3, mt.GATEM = d3, gm
        o = tg._pure(toks, *mf(), tg.kc, tg.vc, tg.kr, tg.vr, tg.cw)
        mx.eval(o); outs.append(o)
    r = [eq(outs[0][0], outs[1][0])] + [eq(u, v) for u, v in zip(outs[0][1], outs[1][1])] + [eq(outs[0][2], outs[1][2]), eq(outs[0][3], outs[1][3])] + [eq(u, v) for u, v in zip(outs[0][4], outs[1][4])]
    print("pass K", K, all(r), flush=True)
    ok = ok and all(r)
print("ALL OK" if ok else "FAILED")

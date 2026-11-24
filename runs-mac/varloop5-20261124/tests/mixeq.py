"""mix == mix2 == mix2f, bit for bit, over the shapes the pass uses; and their latency. python tests/mixeq.py"""
import os, sys, time
from pathlib import Path
WS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WS / "engine"))
import mlx.core as mx
import mk_kernels as kn
eq = lambda a, b: bool(mx.array_equal(a.view(mx.uint32) if a.dtype == mx.float32 else a, b.view(mx.uint32) if b.dtype == mx.float32 else b).item())
ok = True
for K in (1, 3, 8, 16, 40):
    for N in (1, 2, 3, 4, 5, 6, 7, 8):
        for Q in (1, 3, 4):
            srcs = [mx.random.normal((K, 2304)) * (1 + 0.3 * i) for i in range(N)]
            q = mx.random.normal((Q, 2304)) * 3.0
            wt = mx.random.normal((Q, 2304))
            r = []
            if N >= 2:
                r.append(eq(kn.mix(srcs, q), kn.mix2(srcs, q, fast=True)))
            r.append(eq(kn.mix2(srcs, q, fast=False), kn.mix2(srcs, q, fast=True)))
            r.append(eq(kn.mix2(srcs, q, norm_in=True, fast=False), kn.mix2(srcs, q, norm_in=True, fast=True)))
            a = kn.mix2(srcs, q, norm_in=True, wt=wt, fast=False); b = kn.mix2(srcs, q, norm_in=True, wt=wt, fast=True)
            r.append(eq(a, b))
            a = kn.mix2(srcs, q, norm_in=True, pout=True, wt=wt, fast=False); b = kn.mix2(srcs, q, norm_in=True, pout=True, wt=wt, fast=True)
            r += [eq(a[0], b[0]), eq(a[1], b[1])]
            a = kn.mix2(srcs, q, norm_in=True, pout=True, fast=False); b = kn.mix2(srcs, q, norm_in=True, pout=True, fast=True)
            r += [eq(a[0], b[0]), eq(a[1], b[1])]
            if not all(r): print("DIFF", K, N, Q, r)
            ok = ok and all(r)
print("all equal:", ok, flush=True)
x = mx.random.normal((1, 2304)); mx.eval(x)
def lat(f, n=200):
    ts = []
    for _ in range(6):
        v = x
        for _ in range(n): v = f(v)
        mx.synchronize(); t0 = time.perf_counter(); mx.eval(v); ts.append(time.perf_counter() - t0)
    return 1e6 * min(ts[1:]) / n
for N, Q in ((5, 4), (5, 3), (5, 1), (7, 3), (7, 1)):
    srcs = [mx.random.normal((1, 2304)) for _ in range(N - 1)]; q = mx.random.normal((Q, 2304)); wt = mx.random.normal((Q, 2304)); mx.eval(srcs, q, wt)
    for nm, kw in (("plain", {}), ("norm_in + wt", {"norm_in": True, "wt": wt})):
        f0 = lambda v: (kn.mix2(srcs + [v], q, fast=False, **kw)[:, 0]).view(mx.float16).astype(mx.float32) if kw else kn.mix2(srcs + [v], q, fast=False)[:, 0]
        f1 = lambda v: (kn.mix2(srcs + [v], q, fast=True, **kw)[:, 0]).view(mx.float16).astype(mx.float32) if kw else kn.mix2(srcs + [v], q, fast=True)[:, 0]
        print(f"N={N} Q={Q} {nm}: {lat(f0):.1f} -> {lat(f1):.1f} us per call")

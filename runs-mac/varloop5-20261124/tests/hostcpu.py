"""How much of a pass's wall time is the calling thread busy? python tests/hostcpu.py"""
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
eng = Engine(MD, 1, 4096)
tg = eng.tg
rng = np.random.default_rng(0)
eng._hard_reset(); eng.prefill(rng.integers(0, 200000, size=300).tolist())
for K in (1, 8):
    w = c = b = p = 0.0
    for rep in range(12):
        toks = mx.array(rng.integers(0, 200000, size=K).astype(np.int32))
        mx.synchronize()
        t0 = time.perf_counter(); out = tg.forward(toks, tok_only=True); t1 = time.perf_counter()
        c0 = time.thread_time(); p0 = time.process_time()
        mx.eval(out)
        c1 = time.thread_time(); p1 = time.process_time(); t2 = time.perf_counter()
        tg.pending = None
        if rep >= 2:
            b += t1 - t0; w += t2 - t1; c += c1 - c0; p += p1 - p0
    print(f"K={K}: build {1e2 * b:.2f} ms, eval wall {1e2 * w:.2f} ms, calling thread busy {1e2 * c:.2f} ms, process CPU {1e2 * p:.2f} ms", flush=True)
# two independent chains of trivial dispatches against one
x = mx.random.normal((1, 2304)); y = mx.random.normal((1, 2304)); mx.eval(x, y)
def chain(v, n):
    for _ in range(n): v = kn.ident(v)
    return v
for name, f in (("one chain of 400", lambda: [chain(x, 400)]), ("two independent chains of 400", lambda: [chain(x, 400), chain(y, 400)]), ("one chain of 800", lambda: [chain(x, 800)])):
    ts = []
    for _ in range(6):
        o = f(); mx.synchronize(); t0 = time.perf_counter(); c0 = time.thread_time(); mx.eval(*o); ts.append((time.perf_counter() - t0, time.thread_time() - c0))
    print(f"{name}: wall {1e3 * min(t for t, _ in ts[1:]):.2f} ms, thread busy {1e3 * min(c for _, c in ts[1:]):.2f} ms")

"""Time of a K-row target pass (tok_only) at a mid position, and of the draft. python tests/times.py"""
import os, sys, time
from pathlib import Path
import numpy as np
WS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WS / "engine"))
MD = str(Path.home() / "Models/Inkling/TML-d24-varloop-5pass")
import mlx.core as mx
from api import Engine
eng = Engine(MD, 1, 4096)
tg = eng.tg
rng = np.random.default_rng(0)
def fill(n):
    eng._hard_reset()
    toks = rng.integers(0, 200000, size=n).tolist()
    eng.prefill(toks)
fill(512)
for K in [int(v) for v in os.environ.get("T_K", "1,2,3,4,6,8,16").split(",")]:
    ts = []
    for rep in range(12):
        toks = mx.array(rng.integers(0, 200000, size=K).astype(np.int32))
        mx.synchronize()
        t0 = time.perf_counter()
        out = tg.forward(toks, tok_only=True)
        mx.eval(out)
        ts.append(time.perf_counter() - t0)
        tg.pending = None
    ts = sorted(ts[2:])
    print(f"K={K:2d} pass {1e3 * ts[len(ts) // 2]:7.2f} ms (min {1e3 * ts[0]:.2f})", flush=True)
if eng.draft is not None:
    d = eng.draft
    f = [mx.zeros((1, 2304))] * (len(tg.taps) + 1)
    d.frow = f
    ts = []
    for rep in range(12):
        mx.synchronize(); t0 = time.perf_counter()
        dl, pl = d.propose(mx.array([5], dtype=mx.int32), tg.pos, 7)
        mx.eval(dl, pl, *d._rest)
        t1 = time.perf_counter()
        mx.eval(*d.more())
        ts.append((t1 - t0, time.perf_counter() - t1))
    ts = ts[2:]
    print(f"draft first {1e3 * np.median([a for a, b in ts]):.2f} ms, more {1e3 * np.median([b for a, b in ts]):.2f} ms")

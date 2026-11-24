import os, sys, time
from pathlib import Path
import numpy as np
WS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WS / "engine"))
MD = str(Path.home() / "Models/Inkling/TML-d24-varloop-5pass")
import mlx.core as mx
from api import Engine
eng = Engine(MD, 1, 4096)
tg, d = eng.tg, eng.draft
rng = np.random.default_rng(0)
eng._hard_reset(); eng.prefill(rng.integers(0, 200000, size=300).tolist())
def med(f, n=30):
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); f(); ts.append(time.perf_counter() - t0)
    return 1e3 * float(np.median(ts[3:]))
toks1 = mx.array([5], dtype=mx.int32)
out = d._run_a(toks1, tg.pos, tg.cwa(), d.kc, d.vc); mx.eval(*out)
carry = out[2]
print(f"replay stage A K=1: {med(lambda: d._run_a(toks1, tg.pos, tg.cwa(), d.kc, d.vc)):.3f} ms")
for K in (1, 4, 8):
    cr = mx.concatenate([carry] * K); tk = mx.zeros((K,), dtype=mx.int32)
    print(f"replay stage B K={K}: draft {med(lambda: d._run_b(tk, cr, d.kc, d.vc, True)):.3f} ms, no draft {med(lambda: d._run_b(tk, cr, d.kc, d.vc, False)):.3f} ms")
print(f"cwa(): {med(lambda: tg.cwa()):.3f} ms")

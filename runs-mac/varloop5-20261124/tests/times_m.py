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
eng._hard_reset()
eng.prefill(rng.integers(0, 200000, size=300).tolist())
def med(f, n=10):
    ts = []
    for _ in range(n):
        mx.synchronize(); t0 = time.perf_counter(); f(); ts.append(time.perf_counter() - t0)
    return 1e3 * float(np.median(ts[2:]))
carries = {}
for K in [1, 2, 3, 4, 6, 8]:
    toks = mx.array(rng.integers(0, 200000, size=K).astype(np.int32))
    def fa(full):
        out = d._run_a(toks, tg.pos, tg.cwa(), d.kc, d.vc)
        e_l, pe, carry, ka, va, wa = out[:6]
        if full: mx.eval(*out)
        else: mx.eval(e_l, pe, carry, ka, va, *wa)
        carries[K] = carry
    print(f"A K={K}: readout only {med(lambda: fa(False)):6.2f} ms, with draft {med(lambda: fa(True)):6.2f} ms", flush=True)
for K in [1, 2, 3, 4, 6, 8]:
    toks = mx.array(rng.integers(0, 200000, size=K).astype(np.int32))
    def fb(full):
        out = d._run_b(toks, carries[K], d.kc, d.vc)
        if full: mx.eval(out[0], *out[2:])
        else: mx.eval(out[0], out[2], out[3], *out[4])
    print(f"B K={K}: pass only {med(lambda: fb(False)):6.2f} ms, with draft {med(lambda: fb(True)):6.2f} ms", flush=True)

"""Passes of 20 / 24 / 32 rows through the kernels of the 16-row pass (not the prefill's blocked ones)."""
import os, sys, time
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
eng = Engine(MD, 1, 4096)
tg = eng.tg
rng = np.random.default_rng(0)
phrase = rng.integers(0, 200000, size=40).tolist()
def run(K, wide_min, kmax):
    mt.WIDE_MIN, kn.GEMV_KMAX = wide_min, kmax
    tg._fns.clear()
    eng._hard_reset(); eng.prefill(phrase * 8)
    ts = []; out = None
    for rep in range(7):
        toks = mx.array(np.asarray((phrase * 3)[:K], dtype=np.int32))
        mx.synchronize(); t0 = time.perf_counter()
        out = tg.forward(toks, tok_only=True)
        if out.ndim == 2:
            out = mx.concatenate([kn.argmax_short(tg.unembed, tg.un8, out[i:i + 16], tg.vocab, tg.cap)[0] for i in range(0, K, 16)])
        mx.eval(out); ts.append(time.perf_counter() - t0); tg.pending = None
    return 1e3 * sorted(ts[2:])[len(ts[2:]) // 2], np.array(out)
t16, _ = run(16, 17, 16)
print(f"K=16: {t16:.1f} ms, {t16 / 16:.2f} ms/row", flush=True)
for K in (20, 24, 32):
    tw, ow = run(K, 17, 16)
    tr, orr = run(K, 33, 32)
    print(f"K={K}: wide path {tw:.1f} ms ({tw / K:.2f}/row); regular kernels {tr:.1f} ms ({tr / K:.2f}/row); same tokens {bool((ow.astype(np.int64) == orr.astype(np.int64)).all())}", flush=True)

import os, sys, time
from pathlib import Path
import numpy as np
WS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WS / "engine"))
os.environ["MK_SPEC"] = "0"
MD = str(Path.home() / "Models/Inkling/TML-d24-varloop-5pass")
import mlx.core as mx
from api import Engine
import mk_kernels as kn
eng = Engine(MD, 1, 4096)
tg = eng.tg
rng = np.random.default_rng(0)
eng._hard_reset()
# a repetitive context: a 40-token phrase repeated, so the rows of a pass choose experts as a copy loop does
phrase = rng.integers(0, 200000, size=40).tolist()
eng.prefill(phrase * 8)
for K in [8, 16, 24, 32, 48]:
    ts = []
    for rep in range(8):
        toks = mx.array(np.asarray((phrase * 3)[:K], dtype=np.int32))
        mx.synchronize(); t0 = time.perf_counter()
        out = tg.forward(toks, tok_only=True)
        if out.ndim == 2:
            out = mx.concatenate([kn.argmax_short(tg.unembed, tg.un8, out[i:i + 16], tg.vocab, tg.cap)[0] for i in range(0, K, 16)])
        mx.eval(out)
        ts.append(time.perf_counter() - t0)
        tg.pending = None
    ts = sorted(ts[2:])
    print(f"K={K:2d} pass {1e3 * ts[len(ts) // 2]:7.2f} ms  {1e3 * ts[len(ts) // 2] / K:.2f} ms/row", flush=True)

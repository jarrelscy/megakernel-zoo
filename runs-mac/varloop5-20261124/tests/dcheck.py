"""Draft equivalence: fast attention / top-k against the MLX forms on real context. python tests/dcheck.py"""
import os, sys, time
from pathlib import Path
import numpy as np
WS = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.environ.get("D_ENG", str(WS / "engine"))); sys.path.insert(0, str(WS / "tests"))
os.environ["MK_ROUNDS"] = "0"
import mlx.core as mx
import bench
from api import Engine
import mk_draft2 as m2
eng = Engine(bench.MD, 1, int(os.environ.get("B_CTX", "4096")))
tg, d = eng.tg, eng.draft
import json
nat = json.loads(Path("/Users/jarrelscy/src/external/megakernel-zoo/runs-mac/prompts/natural.json").read_text())
eng._hard_reset()
toks = (nat[0] + nat[1] + nat[2] + nat[3] + nat[4] + nat[5] + nat[6])[:int(os.environ.get("D_N", "150"))]
eng._prefill(toks) if False else None
# prefill through the engine so the draft has context
eng.reset(); eng.submit("r", toks, 4)
while True:
    ev = eng.poll()
    if ev and ev[-1][2]: break
anchor = mx.array([toks[-1]], dtype=mx.int32)
res = {}
for fa, ft in ((False, False), (True, False), (False, True), (True, True)):
    m2.FAST_ATT, m2.FAST_TOP = fa, ft
    out = d._pure(anchor, d._cp(tg.pos), d.kc, d.vc, None)
    mx.eval(out)
    res[(fa, ft)] = out
    ts = []
    f = mx.compile(lambda a, c, kc, vc: d._pure(a, c, kc, vc, None))
    for _ in range(12):
        mx.synchronize(); t0 = time.perf_counter(); o = f(anchor, d._cp(tg.pos), d.kc, d.vc); mx.eval(o); ts.append(time.perf_counter() - t0)
    print(fa, ft, "drafts", np.array(out[0]).tolist(), "p", np.round(np.array(out[1]), 4).tolist(), f"{1e3 * np.median(ts[2:]):.2f} ms", flush=True)
r0 = res[(False, False)]
for k, o in res.items():
    b0 = mx.sort(r0[5], axis=-1); b1 = mx.sort(o[5], axis=-1)
    print(k, "max |base diff| (sorted top-64 of rows n1..)", float(mx.abs(b0 - b1).max().item()), "same cand sets", bool((mx.sort(r0[4], axis=-1) == mx.sort(o[4], axis=-1)).all().item()))

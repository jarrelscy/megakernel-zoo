import os, sys
from pathlib import Path
import numpy as np
WS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WS / "engine")); sys.path.insert(0, str(WS / "tests"))
import bench
from api import Engine
import mk_draft2 as m2
eng = Engine(bench.MD, 1, 4096)
for spec in sys.argv[1:]:
    m2.CALIB_LOG = []
    for prompt, n, stop in bench.workloads(spec):
        bench.run(eng, prompt, n, stop)
    L = np.array([(i, p, ok) for i, p, ok in m2.CALIB_LOG], dtype=np.float64)
    print(spec, "rows", len(L))
    edges = [0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.01]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (L[:, 1] >= lo) & (L[:, 1] < hi)
        if m.sum():
            print(f"  predicted {lo:.2f}-{hi:.2f}: n {int(m.sum()):5d} mean predicted {L[m, 1].mean():.3f} accepted {L[m, 2].mean():.3f}")

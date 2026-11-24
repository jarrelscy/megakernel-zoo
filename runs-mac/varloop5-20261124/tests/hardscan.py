"""Find random-id prompts whose continuation is hard (few tokens per cycle), as a proxy for the evaluator's streams."""
import os, sys, json
from pathlib import Path
import numpy as np
WS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WS / "engine")); sys.path.insert(0, str(WS / "tests"))
import bench
from api import Engine
eng = Engine(bench.MD, 1, 4096)
out = []
for seed in range(100, 124):
    p = np.random.default_rng(seed).integers(0, 200000, size=64).tolist()
    c0 = dict(eng.counters)
    toks, dt = bench.run(eng, p, 1024, False)
    c = eng.counters
    d = lambda k: c.get(k, 0) - c0.get(k, 0)
    rate = (len(toks) - 1) / dt
    print(seed, f"{rate:.0f} tok/s cycles {d('cycles')} mcyc {d('m_cycles')} k1 {d('m_k1')} ngram {d('ngram_cycles')}", flush=True)
    out.append((seed, rate, d('m_cycles'), d('m_k1')))
json.dump(out, open("/tmp/hardscan.json", "w"))

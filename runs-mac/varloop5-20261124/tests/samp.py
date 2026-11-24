"""Sampling requests: the request path against step-by-step sampling (exactness), and tokens per second with and
without readout rounds. python tests/samp.py [n_prompts]"""
import os, sys, time
from pathlib import Path
import numpy as np
WS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WS / "engine")); sys.path.insert(0, str(WS / "tests"))
import mlx.core as mx
import bench
from api import Engine
eng = Engine(bench.MD, 1, 4096)
N = int(os.environ.get("S_N", "256"))
def req(prompt, n, sampling):
    eng.reset(); eng.submit("r", prompt, n, sampling)
    out, done, t0 = [], False, None
    while not done:
        for rid, new, fin in eng.poll():
            if t0 is None: t0 = time.perf_counter()
            out += new; done = done or fin
    return out, time.perf_counter() - t0
def ref(prompt, n, sampling):
    eng.reset(); lg = None; out = []
    for t in prompt: lg = eng.step(np.array([t]))
    for i in range(n):
        t = eng.sample(lg, len(prompt) + i, sampling); out.append(t); lg = eng.step(np.array([t]))
    return out
ps = bench.chat_prompts(int(sys.argv[1]) if len(sys.argv) > 1 else 3)
cfgs = [{"temperature": 0.7, "seed": 7}, {"temperature": 1.0, "top_p": 0.95, "seed": 3}, {"temperature": 0.3, "top_k": 40, "seed": 5}]
allok = True
for samp in cfgs:               # compile the sampling graphs before anything is timed
    for p in ps[:2]:
        eng.draft.rounds_samp = True; req(p, N, samp)
        eng.draft.rounds_samp = False; req(p, N, samp)
for ci, samp in enumerate(cfgs):
    tot = {0: [0.0, 0], 1: [0.0, 0]}
    for pi, p in enumerate(ps):
        outs = {}
        for j in range(2):
            flag = (pi + j) % 2
            eng.draft.rounds_samp = bool(flag)
            c0 = dict(eng.counters)
            o, dt = req(p, N, samp)
            outs[flag] = o
            tot[flag][0] += dt; tot[flag][1] += len(o) - 1
        ok = outs[0] == outs[1]
        if pi == 0:
            r = ref(p, N, samp)
            ok = ok and outs[1] == r
            print(f"  cfg {ci} prompt 0: rounds == no rounds == step-by-step sampling: {ok}", flush=True)
        allok = allok and ok
    print(f"{samp}: no rounds {tot[0][1] / tot[0][0]:.1f} tok/s, rounds {tot[1][1] / tot[1][0]:.1f} tok/s; same tokens {allok}", flush=True)
print("ALL EXACT" if allok else "MISMATCH", {k: v for k, v in eng.counters.items() if k in ("m_cycles", "a_rounds", "m_sent", "m_acc")})

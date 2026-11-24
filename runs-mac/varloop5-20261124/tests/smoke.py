"""Smoke tests off the evaluator's path: a sampling request against step-by-step sampling; a long-context engine
(ring draft context) greedy request against step-by-step; a second request that reuses the prefix cache."""
import os, sys
from pathlib import Path
import numpy as np
WS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WS / "engine")); sys.path.insert(0, str(WS / "tests"))
import mlx.core as mx
import bench
from api import Engine
eng = Engine(bench.MD, 1, int(os.environ.get("B_CTX", "4096")))
def req(prompt, n, sampling=None):
    eng.reset(); eng.submit("r", prompt, n, sampling) if sampling else eng.submit("r", prompt, n)
    out, done = [], False
    while not done:
        for rid, new, fin in eng.poll():
            out += new; done = done or fin
    return out
def ref(prompt, n, sampling=None):
    eng.reset(); lg = None; out = []
    for t in prompt: lg = eng.step(np.array([t]))
    for i in range(n):
        t = eng.sample(lg, len(prompt) + i, sampling) if sampling else int(mx.argmax(lg[0]).item())
        out.append(t); lg = eng.step(np.array([t]))
    return out
ps = bench.chat_prompts(3)
samp = {"temperature": 0.8, "top_p": 0.95, "seed": 11}
a = req(ps[0], 96, samp); b = ref(ps[0], 96, samp)
print("sampling request == step-by-step sampling:", a == b, flush=True)
long_prompt = (ps[0] + ps[1] + ps[2]) * 3
a = req(long_prompt, 200); b = ref(long_prompt, 200)
print(f"greedy, {len(long_prompt)}-token prompt == step-by-step:", a == b, flush=True)
c0 = eng.counters["prefix_hits"]
a2 = req(long_prompt + a[:40] + ps[1], 120)
b2 = ref(long_prompt + a[:40] + ps[1], 120)
print("prefix-cache request == step-by-step:", a2 == b2, "prefix hits", eng.counters["prefix_hits"] - c0, flush=True)
print({k: v for k, v in eng.counters.items() if k in ("m_cycles", "a_rounds", "ngram_cycles", "prefix_tokens")})

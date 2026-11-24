"""Latency of the attention kernel at one row on the dependent chain, by context length. python tests/glue_att.py"""
import os, sys, time
from pathlib import Path
import numpy as np
WS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WS / "engine"))
os.environ["MK_SPEC"] = "0"
MD = str(Path.home() / "Models/Inkling/TML-d24-varloop-5pass")
import mlx.core as mx
import mk_kernels as kn
from api import Engine
eng = Engine(MD, 1, 4096)
tg = eng.tg
rng = np.random.default_rng(0)
N = 200
for pos in (600, 1000):
    eng._hard_reset()
    eng.prefill(rng.integers(0, 200000, size=pos).tolist())
    for K in (1, 4):
        for li in (2, 0):                    # one local, one global layer (as the config says)
            L = tg.layers[li]
            q0 = mx.random.normal((K, 3072)); rel = mx.random.normal((K, 384)) * 0.1; sinks = mx.random.normal((K, 24))
            knew = kn.bits(mx.random.normal((K, 1024))); vnew = kn.bits(mx.random.normal((K, 1024))); gate = mx.random.normal((K, 3072))
            mx.eval(q0, rel, sinks, knew, vnew, gate)
            a = li
            meta = mx.array([tg.pos, 0 if L["full"] else 1024, tg.base], dtype=mx.int32)
            def f(q):
                o = kn.attention_gate(q, tg.kc[a], tg.vc[a], tg.kr, tg.vr, knew, vnew, rel, L["proj"], sinks, meta, tg.aidx[a], gate, tg.smax, local=not L["full"])
                return q + 0.0 * o.view(mx.float16).astype(mx.float32)
            ts = []
            for _ in range(5):
                x = q0
                for _ in range(N): x = f(x)
                mx.synchronize(); t0 = time.perf_counter(); mx.eval(x); ts.append(time.perf_counter() - t0)
            print(f"pos {pos:5d} K={K} layer {li} ({'global' if L['full'] else 'local'}): {1e6 * min(ts[1:]) / N:6.1f} us per call (incl. ~5 us of glue ops)", flush=True)

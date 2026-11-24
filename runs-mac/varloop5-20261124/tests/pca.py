"""Recall of the exact argmax in a shortlist from a rank-r projection of the unembedding. python tests/pca.py"""
import json, os, sys, time
from pathlib import Path
import numpy as np
WS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WS / "engine")); sys.path.insert(0, str(WS / "tests"))
os.environ["MK_SPEC"] = "0"
import mlx.core as mx
import bench
from api import Engine
eng = Engine(bench.MD, 1, 2048)
tg = eng.tg
seqs = json.load(open("/tmp/alpha_seqs.json"))
X = []
bounds = {}
for name, lst in seqs.items():
    n_before = sum(x.shape[0] for x in X)
    for prompt, out in lst[:3]:
        full = (prompt + out)[:360]
        eng._hard_reset()
        for i in range(0, len(full), 8):
            ch = full[i:i + 8]
            # the 1-loop readout's final state (what the readout's argmax reads)
            tk, f = tg.early(mx.array(np.asarray(ch, dtype=np.int32)), 1)
            lg = tg.forward(mx.array(np.asarray(ch, dtype=np.int32)), tok_only=True); tg.commit(len(ch))
            mx.eval(tk, f[-1], lg, *tg.state())
            X.append(f[-1])
    bounds[name] = (n_before, sum(x.shape[0] for x in X))
X = mx.concatenate(X); mx.eval(X)
print("states", X.shape, flush=True)
U = tg.unembed.view(mx.bfloat16)[:tg.vocab].astype(mx.float32)
Z = X @ U.T
top = mx.argmax(Z, axis=-1); mx.eval(top)
rng = np.random.default_rng(7)
t0 = time.time()
Ra = []
for q in range(96):
    eng._hard_reset()
    tk, f = tg.early(mx.array(rng.integers(0, 200000, size=16).astype(np.int32)), 1)
    mx.eval(f[-1]); Ra.append(f[-1])
ta = time.time() - t0; t0 = time.time()
Rb = []
for q in range(6):
    eng._hard_reset()
    tok, f = tg.prefill(rng.integers(0, 200000, size=256).tolist(), 256, want_feats=True)
    mx.eval(tok, f[-1]); Rb.append(f[-1])
tb = time.time() - t0
eng._hard_reset()
print(f"collect: early K=16 x96 {ta:.1f}s, prefill 256 x6 {tb:.1f}s", flush=True)
F = np.array(mx.concatenate(Ra)).astype(np.float64)
w, V = np.linalg.eigh(F.T @ F)
import mk_kernels as kn
for r in (512, 768):
    P = mx.array(V[:, -r:].astype(np.float32))
    Up = U @ P; mx.eval(Up)
    Up8 = kn.quant8(Up)
    for test in bounds:
        c, d = bounds[test]
        Xt, tt = X[c:d], top[c:d]
        S = kn.gemv8(Up8, kn.bits(Xt[:16] @ P)) if False else mx.concatenate([kn.gemv8(Up8, kn.bits(Xt[i:i + 16] @ P)) for i in range(0, d - c, 16)])
        gap = S.max(axis=-1) - mx.take_along_axis(S, tt[:, None], axis=-1)[:, 0]
        g = np.sort(np.array(gap))
        line = f"r{r} {test}: gap p50 {g[len(g) // 2]:.2f} p99 {g[int(len(g) * 0.99)]:.2f} p99.7 {g[int(len(g) * 0.997)]:.2f} max {g[-1]:.2f};"
        for dl in (2.0, 3.0, 4.0, 6.0):
            cntf = (S >= S.max(axis=-1, keepdims=True) - dl).sum(axis=-1).astype(mx.float32)
            line += f" d{dl:.0f}: recall {float((gap <= dl).astype(mx.float32).mean().item()):.3f} flagged mean {float(cntf.mean().item()):.0f} max {float(cntf.max().item()):.0f};"
        print(line, flush=True)

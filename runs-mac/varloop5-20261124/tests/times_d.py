import os, sys, time
from pathlib import Path
import numpy as np
WS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WS / "engine"))
os.environ["MK_ROUNDS"] = "0"
MD = str(Path.home() / "Models/Inkling/TML-d24-varloop-5pass")
import mlx.core as mx
from api import Engine
import mk_kernels as kn, mk_draft as md
eng = Engine(MD, 1, 4096)
tg, d = eng.tg, eng.draft
rng = np.random.default_rng(0)
eng._hard_reset()
eng.prefill(rng.integers(0, 200000, size=300).tolist())
def med(f, n=12):
    ts = []
    for _ in range(n):
        mx.synchronize(); t0 = time.perf_counter(); f(); ts.append(time.perf_counter() - t0)
    return 1e3 * float(np.median(ts[2:]))
anchor = mx.array([5], dtype=mx.int32)
def full():
    dl, pl = d.propose(anchor, tg.pos, 7); mx.eval(dl, pl, *d._rest)
print(f"TD={d.TD} ring={d.ring}  propose: {med(full):.2f} ms")
# pieces (uncompiled)
x = mx.random.normal((8, 2304)); mx.eval(x)
w = d.L[0]
print("qkv gemv8 8 rows", med(lambda: mx.eval(d.gm(w["qkv"], kn.bits(x)))))
print("gate_up", med(lambda: mx.eval(d.gm(w["gate_up"], kn.bits(x)))))
x9 = mx.random.normal((8, 9216)); x3 = mx.random.normal((8, 3072)); mx.eval(x9, x3)
print("down", med(lambda: mx.eval(d.gm(w["down"], kn.bits(x9)))))
print("o", med(lambda: mx.eval(d.gm(w["o"], kn.bits(x3)))))
x7 = x[:7]
print("unembed un8 7 rows", med(lambda: mx.eval(d.gmu(d.un8, kn.bits(x7)))))
base = d.gmu(d.un8, kn.bits(x7)); mx.eval(base)
print("argpartition top64", med(lambda: mx.eval(mx.argpartition(-base, 63, axis=-1)[:, :64])))
print("topk via sort?", med(lambda: mx.eval(mx.argmax(base, axis=-1))))
q = mx.random.normal((8, 24, 128)); mx.eval(q)
def att():
    k = mx.random.normal((8, 8, 128)) * 0 + 1
    keys = mx.concatenate([d.kc[0], k]).transpose(1, 2, 0)
    vals = mx.concatenate([d.vc[0], k]).transpose(1, 0, 2)
    qq = q.reshape(8, 8, 3, 128).transpose(1, 0, 2, 3).reshape(8, 24, 128)
    s = mx.softmax((qq @ keys) * (128 ** -0.5), axis=-1)
    mx.eval((s @ vals))
print("attention (one layer)", med(att))
ctx = mx.argpartition(-base, 63, axis=-1)[:, :64].astype(mx.int32); bb = mx.take_along_axis(base, ctx, axis=-1); mx.eval(ctx, bb)
def mk():
    out = d._markov(ctx[:3], bb[:3], mx.zeros((1, 256)), anchor); mx.eval(out[0], out[1])
print("markov 3 steps", med(mk))

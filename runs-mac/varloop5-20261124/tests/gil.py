"""Does mx.eval release the GIL, and can another thread replay a traced graph meanwhile?"""
import os, sys, time, threading
from pathlib import Path
import numpy as np
WS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WS / "engine"))
MD = str(Path.home() / "Models/Inkling/TML-d24-varloop-5pass")
import mlx.core as mx
from api import Engine
eng = Engine(MD, 1, 4096)
tg, d = eng.tg, eng.draft
rng = np.random.default_rng(0)
eng._hard_reset(); eng.prefill(rng.integers(0, 200000, size=300).tolist())
toks = mx.array([5], dtype=mx.int32)
out = d._run_a(toks, tg.pos, tg.cwa(), d.kc, d.vc); mx.eval(*out[:6])
carry = out[2]
# 1) a counting thread during a blocking eval
cnt = [0]; stop = [False]
def spin():
    while not stop[0]: cnt[0] += 1
th = threading.Thread(target=spin, daemon=True); th.start()
time.sleep(0.05); c0 = cnt[0]; t0 = time.perf_counter(); time.sleep(0.02); base_rate = (cnt[0] - c0) / (time.perf_counter() - t0)
o = tg.forward(mx.array(rng.integers(0, 200000, size=8).astype(np.int32)), tok_only=True)
c0 = cnt[0]; t0 = time.perf_counter(); mx.eval(o); dt = time.perf_counter() - t0; rate = (cnt[0] - c0) / dt
tg.pending = None
stop[0] = True; th.join()
print(f"spin thread: {base_rate / 1e6:.1f} M/s while main sleeps, {rate / 1e6:.1f} M/s while main is in mx.eval ({1e3 * dt:.1f} ms)")
# 2) replay stage B in a helper thread while the main thread evaluates a stage A
import queue
jobs = queue.Queue()
main_stream = mx.default_stream(mx.default_device())
print("main stream", main_stream, flush=True)
def worker():
    while True:
        f, ev, res = jobs.get()
        if f is None: return
        t = time.perf_counter()
        try:
            with mx.stream(main_stream):
                res.append(f())
        except Exception as ex:
            res.append(ex)
        res.append(time.perf_counter() - t); ev.set()
wt = threading.Thread(target=worker, daemon=True); wt.start()
ts = {"serial": [], "threaded": []}
for mode in ("serial", "threaded") * 15:
    out = d._run_a(toks, tg.pos, tg.cwa(), d.kc, d.vc)
    e_l, pe_l, carry, ka, va, wa = out[:6]
    job = lambda: d._run_b(mx.zeros((1,), dtype=mx.int32), carry[:1], d.kc, d.vc, False)
    mx.synchronize(); t0 = time.perf_counter()
    if mode == "threaded":
        ev, res = threading.Event(), []
        jobs.put((job, ev, res))
        mx.eval(e_l, pe_l, carry, ka, va, *wa)
        ev.wait(); ob = res[0]
        if isinstance(ob, Exception): print("worker failed:", repr(ob)); break
    else:
        mx.eval(e_l, pe_l, carry, ka, va, *wa)
        ob = job()
    t1 = time.perf_counter()
    mx.eval(ob[0], ob[2], ob[3], *ob[4]); t2 = time.perf_counter()
    ts[mode].append((t1 - t0, t2 - t1))
for m, v in ts.items():
    a = np.median([x for x, _ in v[2:]]); b = np.median([y for _, y in v[2:]])
    print(f"{m}: stage A + stage-B graph ready {1e3 * a:.2f} ms, then stage B eval {1e3 * b:.2f} ms, total {1e3 * (a + b):.2f} ms")
jobs.put((None, None, None))

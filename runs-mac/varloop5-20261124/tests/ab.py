"""Same-process A/B: python tests/ab.py "rounds=0" "rounds=3" [-- synth:3 natural:4 chat:6]
Each prompt is run under every config in turn (order rotated per prompt); per workload, ms/token of each config."""
import os, sys, time
from pathlib import Path
import numpy as np
WS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WS / "engine")); sys.path.insert(0, str(WS / "tests"))
import bench

def parse(c):
    out = {}
    for kv in c.split(","):
        if not kv: continue
        k, v = kv.split("=")
        try: v = int(v)
        except ValueError:
            try: v = float(v)
            except ValueError: pass
        out[k] = v
    return out

def main():
    args = sys.argv[1:]
    specs = ["synth:3", "natural:4", "chat:6"]
    if "--" in args:
        i = args.index("--"); specs = args[i + 1:]; args = args[:i]
    cfgs = [parse(a) for a in args]
    from api import Engine
    eng = Engine(bench.MD, 1, int(os.environ.get("B_CTX", "4096")))
    base = {k: getattr(eng.draft, k) for c in cfgs for k in c}
    tot = {}
    for spec in specs:
        acc = [[0.0, 0, 0, 0] for _ in cfgs]
        per = [[] for _ in cfgs]
        for pi, (prompt, n, stop) in enumerate(bench.workloads(spec)):
            ref = None
            for j in range(len(cfgs)):
                ci = (pi + j) % len(cfgs)
                for k, v in base.items(): setattr(eng.draft, k, v)
                for k, v in cfgs[ci].items(): setattr(eng.draft, k, v)
                c0 = dict(eng.counters)
                toks, dt = bench.run(eng, prompt, n, stop)
                if ref is None: ref = toks
                elif toks != ref: print("  TOKENS DIFFER between configs!", spec, pi, flush=True)
                acc[ci][0] += dt; acc[ci][1] += len(toks) - 1
                acc[ci][2] += eng.counters["cycles"] - c0["cycles"]; acc[ci][3] += eng.counters["rows"] - c0["rows"]
                per[ci].append((len(toks) - 1) / dt)
        for ci, c in enumerate(cfgs):
            t, n, cy, rw = acc[ci]
            print(f"{spec:10s} {str(c):40s} {1e3 * t / n:7.3f} ms/tok {n / t:6.1f} tok/s  tok/cyc {n / max(cy, 1):.2f} rows/cyc {rw / max(cy, 1):.2f} | " + " ".join(f"{r:.0f}" for r in per[ci]), flush=True)
            tot.setdefault(ci, []).append(1e3 * t / n)
    for ci, c in enumerate(cfgs):
        print(f"SUMMARY {str(c):40s} " + " ".join(f"{v:7.3f}" for v in tot[ci]))
main()

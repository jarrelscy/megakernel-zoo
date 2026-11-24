"""Agreement of the r-pass model's greedy token with the 5-pass greedy text (a proxy for early-exit readouts).
python tests/alpha.py gen            -> /tmp/alpha_seqs.json (5-pass greedy continuations)
python tests/alpha.py eval R         -> agreement of the R-pass model, teacher-forced
"""
import json, os, sys, tempfile
from pathlib import Path
import numpy as np
WS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WS / "engine")); sys.path.insert(0, str(WS / "tests"))
os.environ["MK_SPEC"] = "0"
os.environ["MK_DRAFT2"] = "0"
MD = str(Path.home() / "Models/Inkling/TML-d24-varloop-5pass")
OUT = "/tmp/alpha_seqs.json"

def model_dir(r):
    if r == 5:
        return MD
    d = Path(tempfile.mkdtemp(prefix=f"vl{r}_"))
    for f in Path(MD).iterdir():
        if f.name != "config.json":
            (d / f.name).symlink_to(f)
    cfg = json.loads((Path(MD) / "config.json").read_text())
    cfg["text_config"]["loop_config"]["num_passes"] = r
    n = 12 + 12 * r
    cfg["mtp_config"] = {"target_layer_ids": [5, 11, 17, n] if r > 0 else [5, 6, 7, 12]}
    (d / "config.json").write_text(json.dumps(cfg))
    return str(d)

def main():
    import mlx.core as mx
    import bench
    from api import Engine
    if sys.argv[1] == "gen":
        eng = Engine(MD, 1, 2048)
        seqs = {}
        for name, n in (("synth", 3), ("natural", 4), ("chat", 6)):
            seqs[name] = []
            for prompt, _, stop in bench.workloads(f"{name}:{n}"):
                eng.reset()
                for t in prompt[:-1]:
                    eng.step(np.array([t]))
                out = np.array(eng.generate(np.array([prompt[-1]]), 300))[0].tolist()
                if stop and bench.EOS in out:
                    out = out[:out.index(bench.EOS) + 1]
                seqs[name].append((prompt, out))
                print(name, len(prompt), len(out), flush=True)
        json.dump(seqs, open(OUT, "w"))
        return
    r = int(sys.argv[2])
    eng = Engine(model_dir(r), 1, 2048)
    seqs = json.load(open(OUT))
    res = {}
    for name, lst in seqs.items():
        agree = tot = 0
        marks = []
        for prompt, out in lst:
            full = prompt + out
            eng._hard_reset()
            pred = []
            for i in range(0, len(full), 8):
                chunk = full[i:i + 8]
                tk = eng.tg.forward(mx.array(np.asarray(chunk, dtype=np.int32)), tok_only=True)
                eng.tg.commit(len(chunk))
                mx.eval(tk, *eng._state())
                pred += np.array(tk).tolist()
            # pred[i] predicts full[i+1]; generated tokens are full[len(prompt):]
            ok = [int(pred[i - 1] == full[i]) for i in range(len(prompt), len(full))]
            marks.append(ok)
            agree += sum(ok); tot += len(ok)
        res[name] = marks
        print(f"passes={r} {name}: agreement {agree}/{tot} = {agree / tot:.4f}", flush=True)
    json.dump(res, open(f"/tmp/alpha_marks_{r}.json", "w"))

main()

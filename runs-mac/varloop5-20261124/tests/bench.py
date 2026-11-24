"""Local bench through submit/poll: synthetic (random ids), natural (64-token prompts), chat (GSM8K through the chat
template, until EOS or the token limit). python tests/bench.py [synth:N] [natural:N] [chat:N] [--check]
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

WS = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.environ.get("B_ENG", str(WS / "engine")))
ZOO = Path("/Users/jarrelscy/src/external/megakernel-zoo")
MD = os.environ.get("MK_MODEL_DIR", str(Path.home() / "Models/Inkling/TML-d24-varloop-5pass"))
EOS = 200006


def chat_prompts(n):
    import jinja2
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(Path(MD) / "tokenizer.json"))
    env = jinja2.Environment()
    import re
    src = re.sub(r"\{%-?\s*(end)?generation\s*-?%\}", "", (Path(MD) / "chat_template.jinja").read_text())
    env.globals["raise_exception"] = lambda m: (_ for _ in ()).throw(ValueError(m))
    tpl = env.from_string(src)
    qs = json.loads((ZOO / "runs-mac/prompts/chat.json").read_text())[:n]
    out = []
    for q in qs:
        text = tpl.render(messages=[{"role": "user", "content": q}], add_generation_prompt=True)
        out.append(tok.encode(text, add_special_tokens=False).ids)
    return out


def workloads(spec):
    name, _, n = spec.partition(":")
    n = int(n or 4)
    off = int(os.environ.get("B_OFF", "0"))
    if name == "synth":
        rng = np.random.default_rng(int(os.environ.get("B_SEED", "1234")))
        return [(rng.integers(0, 200000, size=64).tolist(), 1024, False) for _ in range(n)]
    if name == "hard":      # random-id prompts whose continuation is hard (tests/hardscan.py): short chains, few copies
        return [(np.random.default_rng(sd).integers(0, 200000, size=64).tolist(), 1024, False) for sd in (102, 111, 112, 113, 116, 119, 123)[:n]]
    if name == "natural":
        ps = json.loads((ZOO / "runs-mac/prompts/natural.json").read_text())
        step = max(1, len(ps) // n)
        return [(p, 1024, False) for p in ps[off::step][:n]]
    if name == "chat":
        return [(p, 512, True) for p in chat_prompts(n + off)[off:]]
    raise SystemExit(spec)


def run(eng, prompt, n, stop_eos):
    eng.reset()
    eng.submit("r", prompt, n)
    toks, t0, tfirst = [], time.perf_counter(), None
    tend = None
    done = False
    while not done:
        for rid, new, fin in eng.poll():
            now = time.perf_counter()
            if tfirst is None:
                tfirst = now
            toks += new
            done = done or fin
            if stop_eos and EOS in toks and tend is None:
                tend = now
                toks = toks[:toks.index(EOS) + 1]
        if tend is not None:
            if not done:
                eng.cancel("r")
            break
    tend = tend or time.perf_counter()
    return toks, tend - tfirst


def main():
    import mlx.core as mx
    from api import Engine
    specs = [a for a in sys.argv[1:] if not a.startswith("--")] or ["synth:2", "natural:4", "chat:4"]
    check = "--check" in sys.argv
    eng = Engine(MD, 1, int(os.environ.get("B_CTX", "4096")))
    res = {}
    for spec in specs:
        tot_t = tot_n = 0
        c0 = dict(eng.counters)
        rates = []
        for prompt, n, stop in workloads(spec):
            toks, dt = run(eng, prompt, n, stop)
            tot_t += dt
            tot_n += len(toks) - 1
            rates.append((len(toks) - 1) / dt)
            if check:
                eng.reset()
                ref = []
                lg = None
                for t in prompt:
                    lg = eng.step(np.array([t]))
                for _ in range(len(toks)):
                    t = int(mx.argmax(lg[0]).item())
                    ref.append(t)
                    lg = eng.step(np.array([t]))
                print("  check", "OK" if ref == toks else f"MISMATCH at {next(i for i, (a, b) in enumerate(zip(ref, toks)) if a != b)}", flush=True)
        c1 = eng.counters
        d = {k: c1.get(k, 0) - c0.get(k, 0) for k in c1 if isinstance(c1.get(k), (int, float))}
        cyc = max(1, d["cycles"])
        print(f"{spec:12s} {1e3 * tot_t / tot_n:7.3f} ms/tok {tot_n / tot_t:6.1f} tok/s | tok/cyc {d['tokens'] / cyc:.2f} rows/cyc {d['rows'] / cyc:.2f} "
              f"ms/cyc {1e3 * tot_t / cyc:.1f} | ngram_cyc {d.get('ngram_cycles', 0)} alone {d.get('draft_alone', 0)} more {d.get('draft_more', 0)} "
              f"draft_s {d.get('draft_s', 0):.2f} verify_s {d.get('verify_s', 0):.2f} | mcyc {d.get('m_cycles', 0)} rounds {d.get('a_rounds', 0)} arows {d.get('a_rows', 0)} sent {d.get('m_sent', 0)} acc {d.get('m_acc', 0)} k1 {d.get('m_k1', 0)} preA {d.get('pre_a_used', 0)} preB {d.get('pre_b_used', 0)} tpre {d.get('t_pre', 0):.2f} ng: cyc {d.get('ngram_cycles', 0)} rows {d.get('ngram_rows', 0)} toks {d.get('ngram_tokens', 0)} full {d.get('ngram_full', 0)} wide {d.get('wide_cycles', 0)} tA {d.get('t_a', 0):.2f} tD {d.get('t_d', 0):.2f} tB {d.get('t_b', 0):.2f} tM {d.get('t_m', 0):.2f} | per-prompt " + " ".join(f"{r:.0f}" for r in rates), flush=True)
        res[spec] = tot_n / tot_t
    return res


if __name__ == "__main__":
    main()

"""Stage A + stage B (in rounds, with provisional K/V) against the one pass: bit equality of tokens, features, K/V rows
and conv windows. python tests/bitm.py"""
import os, sys
from pathlib import Path
import numpy as np
WS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WS / "engine"))
os.environ["MK_SPEC"] = "0"
MD = str(Path.home() / "Models/Inkling/TML-d24-varloop-5pass")
import mlx.core as mx
from api import Engine
eng = Engine(MD, 1, int(os.environ.get("B_CTX", "4096")))
tg = eng.tg
rng = np.random.default_rng(int(os.environ.get("SEED", "0")))
import json
nat = json.loads(Path("/Users/jarrelscy/src/external/megakernel-zoo/runs-mac/prompts/natural.json").read_text())
ok_all = True
def eq(a, b):
    return bool(mx.array_equal(a, b).item()) if a.dtype != mx.float32 else bool(mx.array_equal(a.view(mx.uint32), b.view(mx.uint32)).item())
for trial, (npre, splits) in enumerate([(40, [3, 2]), (64, [1, 1, 1]), (100, [4]), (57, [2, 5, 1, 3]), (90, [8, 4]), (33, [1])]):
    eng._hard_reset()
    pre = nat[trial][:npre] if npre <= 64 else rng.integers(0, 200000, size=npre).tolist()
    eng.prefill(pre)
    for _ in range(3):      # a few committed one-row steps so the recent tier is in use
        lg = tg.forward(mx.array(rng.integers(0, 200000, size=1).astype(np.int32)), tok_only=True); tg.commit(1); mx.eval(lg, *tg.state())
    K = sum(splits)
    toks = rng.integers(0, 200000, size=K).astype(np.int32)
    mf = mx.array([tg.pos, 0, tg.base], dtype=mx.int32); ml = mx.array([tg.pos, 1024, tg.base], dtype=mx.int32)
    ref = tg._pure(mx.array(toks), mf, ml, tg.kc, tg.vc, tg.kr, tg.vr, tg.cw, tok_only=True)
    mx.eval(ref)
    # rounds
    kr, vr = tg.kr, tg.vr
    cwa = tg.cwa()
    p = tg.pos
    cl, kl, vl, wl = [], [], [], []
    i0 = 0
    for ri, n in enumerate(splits):
        extra = min(2, 8 - n) if ri % 2 == 0 else 0          # rows of the round that are dropped (rejected drafts)
        tk = np.concatenate([toks[i0:i0 + n], rng.integers(0, 200000, size=extra).astype(np.int32)])
        e, pe, f, carry, ka, va, wa = tg._stage_a(mx.array(tk), mx.array([p, 0, tg.base], dtype=mx.int32), mx.array([p, 1024, tg.base], dtype=mx.int32),
                                              tg.kc, tg.vc, kr, vr, cwa)
        cl.append(carry[:n]); kl.append(ka[:, :n]); vl.append(va[:, :n])
        wl.append([w[:, :n + 3] if ri == 0 else w[:, 3:n + 3] for w in wa])
        s = p - tg.base
        kr = mx.array(kr); vr = mx.array(vr)
        kr[0:18, s:s + n] = ka[:18, :n]; kr[66:72, s:s + n] = ka[18:, :n]
        vr[0:18, s:s + n] = va[:18, :n]; vr[66:72, s:s + n] = va[18:, :n]
        cwa = [w[:, n:n + 3] for w in wa]
        p += n; i0 += n
        mx.eval(e, kr, vr, cwa)
    carry = mx.concatenate(cl)
    tok, feats, kb, vb, wb = tg._stage_b(carry, mf, ml, tg.kc, tg.vc, kr, vr, tg.cw)
    knew = mx.concatenate([mx.concatenate(kl, axis=1)[:18], kb]); vnew = mx.concatenate([mx.concatenate(vl, axis=1)[:18], vb])
    wins = [mx.concatenate([mx.concatenate([w[j] for w in wl], axis=1)[:18], wb[j]]) for j in range(4)]
    mx.eval(tok, feats, knew, vnew, wins)
    res = [eq(tok, ref[0])] + [eq(a, b) for a, b in zip(feats, ref[1])] + [eq(knew, ref[2]), eq(vnew, ref[3])] + [eq(a, b) for a, b in zip(wins, ref[4])]
    print(f"trial {trial} pre {npre} splits {splits}: {'OK' if all(res) else 'DIFF'} {res}", flush=True)
    ok_all = ok_all and all(res)
print("ALL OK" if ok_all else "FAILED")

"""Parked conversations: the engine keeps one sequence's K/V live; this keeps others, so a request that continues any of
them (an agent turn after a transcription, a report check between two agent turns) starts from its cache instead of
reading its whole prompt again.

A parked entry is the used part of the live state, copied out when another sequence takes its place:
  tokens      the sequence's token ids (positions given as input embeddings are negative, unique: never matched)
  kv          K and V of positions 0 .. pos - 1 for every layer application, compact ([napp, pos, 1024] x 2, half bits)
  cw, snaps   the short-convolution windows at pos and at the snapshot positions a rollback can cut back to
  draft       the draft model's context (its arrays are handed over whole, not copied)
Entries live in memory up to MK_KV_RAM_GB (default 24); the least recently used then go to the SSD under MK_KV_DIR
(default ~/Library/Caches/mkserve/kv) up to MK_KV_DISK_GB (default 96), and stay there across restarts. An entry is
used again when it shares more of the new prompt than the live sequence does.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

GB = 1 << 30


def lcp(a: list, b: list, limit: int) -> int:
    n = min(len(a), len(b), limit)
    i = 0
    # in blocks first: most prompts share long prefixes
    while i + 256 <= n and a[i:i + 256] == b[i:i + 256]:
        i += 256
    while i < n and a[i] == b[i]:
        i += 1
    return i


class KVStore:
    def __init__(self, model_key: str):
        self.ram_budget = float(os.environ.get("MK_KV_RAM_GB", "24")) * GB
        self.disk_budget = float(os.environ.get("MK_KV_DISK_GB", "96")) * GB
        self.dir = Path(os.environ.get("MK_KV_DIR", str(Path.home() / "Library/Caches/mkserve/kv"))) / model_key
        self.enabled = os.environ.get("MK_KV_SLOTS", "1") != "0"
        # MK_KV_LAZY_FIT: the memory budget is enforced when the engine is idle (Engine.poll with nothing to do), not
        # inside park(): an entry written to the SSD (~3 GB at 5 passes and 10K positions, 0.27-0.39 s) is then never on
        # a request's path. Memory runs over the budget by the entries parked since the last idle moment
        self.lazy_fit = os.environ.get("MK_KV_LAZY_FIT", "1") == "1"
        self.dirty = False
        self.entries: list[dict] = []          # newest last
        self.stats = {"parked": 0, "restored": 0, "to_disk": 0, "from_disk": 0, "restore_s": 0.0, "park_s": 0.0}
        if self.enabled and self.disk_budget > 0:
            self.dir.mkdir(parents=True, exist_ok=True)
            for meta in sorted(self.dir.glob("*.json")):
                try:
                    m = json.loads(meta.read_text())
                    if (self.dir / m["file"]).exists():
                        self.entries.append({"tokens": m["tokens"], "pos": m["pos"], "bytes": m["bytes"], "path": self.dir / m["file"],
                                             "meta": meta, "used": m.get("used", 0.0)})
                except (OSError, ValueError, KeyError):
                    continue
            self.entries.sort(key=lambda e: e["used"])

    # ------------------------------------------------------------------------------------------ lookup
    def best(self, prompt: list, limit: int):
        """The entry sharing the longest prefix with prompt: (index, shared tokens), or (None, 0)."""
        bi, bl = None, 0
        # an entry whose file is gone (another server on the same folder dropped it) is forgotten, not restored
        self.entries = [e for e in self.entries if "path" not in e or Path(e["path"]).exists()]
        for i, e in enumerate(self.entries):
            n = lcp(e["tokens"], prompt, limit)
            if n > bl:
                bi, bl = i, n
        return bi, bl

    # ------------------------------------------------------------------------------------------ park / restore
    def park(self, tg, draft, tokens: list):
        """Copy the live sequence out (its first tg.pos positions). Dropped if it is short or already parked."""
        if not self.enabled or tg.pos < 256 or tokens is None:
            return
        tokens = list(tokens[:tg.pos])
        for e in self.entries:                 # the same sequence parked before (it was restored, then extended or not)
            if e["tokens"][:len(tokens)] == tokens or tokens[:len(e["tokens"])] == e["tokens"]:
                self._drop(e)
                break
        t0 = time.perf_counter()
        tg.flush()
        p = tg.pos
        k = mx.stack([tg.kc[a][0, :p] for a in range(tg.napp)])
        v = mx.stack([tg.vc[a][0, :p] for a in range(tg.napp)])
        snaps = {q: s for q, s in tg.snaps.items() if q <= p}
        cw = [mx.array(w) for w in tg.cw]
        mx.eval(k, v, cw, *[x for s in snaps.values() for x in s])
        d = None
        if draft is not None:
            d = {"kc": draft.kc, "vc": draft.vc, "frow": draft.frow, "hist": list(draft.hist), "low": draft.low,
                 "dbase": getattr(draft, "dbase", 0)}
            draft.reset()                      # the live drafter gets new arrays; the parked one keeps these
        nbytes = 2 * k.nbytes + sum(w.nbytes for w in cw) + sum(x.nbytes for s in snaps.values() for x in s)
        self.entries.append({"tokens": tokens, "pos": p, "k": k, "v": v, "cw": cw, "snaps": snaps, "draft": d,
                             "bytes": nbytes, "used": time.time()})
        self.stats["parked"] += 1
        self.stats["park_s"] += time.perf_counter() - t0
        # lazy, but never past twice the budget (back-to-back requests that each park would otherwise grow without bound)
        if self.lazy_fit and sum(e["bytes"] for e in self.entries if "k" in e) <= 2 * self.ram_budget:
            self.dirty = True
        else:
            self._fit()

    def restore(self, e: dict, tg, draft) -> list:
        """Make entry e the live sequence (the caller has parked or dropped the live one); returns its tokens."""
        self.entries.remove(e)
        t0 = time.perf_counter()
        if "path" in e:
            e = self._load(e)
        p = e["pos"]
        tg.flush()                              # positions past p are never read: the arrays are reused as they are
        for a in range(tg.napp):
            tg.kc[a][0, :p] = e["k"][a]
            tg.vc[a][0, :p] = e["v"][a]
        tg.cw = list(e["cw"])
        tg.snaps = dict(e["snaps"])
        tg.pos = tg.base = p
        tg.pending = None
        tg._fresh = []
        if draft is not None:
            d = e.get("draft")
            if d is not None and d["kc"] is not None:
                draft.kc, draft.vc, draft.frow, draft.hist, draft.low = d["kc"], d["vc"], d["frow"], d["hist"], d["low"]
                if hasattr(draft, "dbase"):
                    draft.dbase = d["dbase"]
            else:
                draft.reset()                  # from the SSD: the draft context is rebuilt as the sequence goes on
        mx.eval(tg.kc, tg.vc, tg.cw)
        self.stats["restored"] += 1
        self.stats["restore_s"] += time.perf_counter() - t0
        return e["tokens"]

    # ------------------------------------------------------------------------------------------ tiers
    def idle(self):
        """The engine has nothing to do: enforce the budget now (see lazy_fit)."""
        if self.dirty:
            self.dirty = False
            self._fit()

    def _fit(self):
        """Keep memory under its budget: the least recently used entries go to the SSD (or are dropped)."""
        while sum(e["bytes"] for e in self.entries if "k" in e) > self.ram_budget:
            oldest = next(e for e in self.entries if "k" in e)
            if self.disk_budget > 0:
                self._save(oldest)
            else:
                self._drop(oldest)
        while sum(e["bytes"] for e in self.entries if "path" in e) > self.disk_budget:
            self._drop(next(e for e in self.entries if "path" in e))

    def _save(self, e: dict):
        t0 = time.perf_counter()
        name = hashlib.sha1(np.asarray(e["tokens"][:4096], dtype=np.int64).tobytes() + str(e["pos"]).encode()).hexdigest()[:20]
        arrays = {"k": e["k"], "v": e["v"]}
        arrays.update({f"cw{j}": w for j, w in enumerate(e["cw"])})
        for q, s in e["snaps"].items():
            arrays.update({f"snap{q}_{j}": x for j, x in enumerate(s)})
        path = self.dir / f"{name}.safetensors"
        mx.save_safetensors(str(path), arrays)
        meta = self.dir / f"{name}.json"
        meta.write_text(json.dumps({"tokens": e["tokens"], "pos": e["pos"], "bytes": e["bytes"], "file": path.name, "used": e["used"]}))
        for key in ("k", "v", "cw", "snaps", "draft"):
            e.pop(key, None)
        e.update({"path": path, "meta": meta})
        self.stats["to_disk"] += 1
        self.stats["park_s"] += time.perf_counter() - t0

    def _load(self, e: dict) -> dict:
        w = mx.load(str(e["path"]))
        snaps: dict = {}
        for key, x in w.items():
            if key.startswith("snap"):
                q, j = key[4:].split("_")
                snaps.setdefault(int(q), [None] * 4)[int(j)] = x
        out = {"tokens": e["tokens"], "pos": e["pos"], "k": w["k"], "v": w["v"], "cw": [w[f"cw{j}"] for j in range(4)],
               "snaps": snaps, "draft": None, "bytes": e["bytes"], "used": time.time()}
        self._unlink(e)
        self.stats["from_disk"] += 1
        return out

    def _drop(self, e: dict):
        self.entries.remove(e)
        if "path" in e:
            self._unlink(e)

    def _unlink(self, e: dict):
        for key in ("path", "meta"):
            try:
                Path(e[key]).unlink()
            except (OSError, KeyError):
                pass

    def summary(self) -> dict:
        return dict(self.stats, entries=len(self.entries),
                    ram_gb=round(sum(e["bytes"] for e in self.entries if "k" in e) / GB, 2),
                    disk_gb=round(sum(e["bytes"] for e in self.entries if "path" in e) / GB, 2))

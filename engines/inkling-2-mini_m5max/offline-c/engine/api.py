"""Inkling-2-Mini on Apple silicon, track C: the target as K-row passes of our own Metal kernels (engine/target.py,
engine/kernels.py), the checkpoint's DSpark head as the draft (engine/draft.py), exact draft-and-verify behind
submit() / poll(). step() is the K = 1 pass; a verify pass gives every row the bits step() gives it.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mk_target  # noqa: E402
from mk_target import Target  # noqa: E402

SPEC = os.environ.get("MK_SPEC", "1") != "0"
# prompts in wide passes of up to PF_CHUNK rows (mk_target.PREFILL: exact by default, the bits of one-row steps; fast:
# MLX GEMMs and fused attention, what mkserve serves); a pass is padded to the next of PF_BUCKETS rows, so only those
# shapes are ever compiled (all of them in _warm)
PF_CHUNK = int(os.environ.get("MK_PF_CHUNK", "256"))
PF_BUCKETS = [k for k in (32, 64, 128, 256, 512) if k <= PF_CHUNK]


class Engine:
    native_sampling = True      # submit() takes sampling parameters: drawn tokens through the same exact draft-and-verify

    def __init__(self, model_dir: str, max_batch: int, max_context: int):
        t0 = time.time()
        self.max_batch, self.max_context = int(max_batch), int(max_context)
        info = mx.device_info()
        mx.set_wired_limit(info["max_recommended_working_set_size"])
        # MLX keeps freed buffers for reuse. The temporary arrays of a long prefill (masks of a wide pass over 90,000 keys
        # are gigabytes each) would otherwise stay held: 102 GB in all after a 91,000-token prompt at 131,072 context.
        mx.set_cache_limit(int(float(os.environ.get("MK_CACHE_GB", "4")) * 2**30))
        self.tg = Target(model_dir, self.max_context)
        self.vocab = self.tg.vocab
        self.draft = None
        if SPEC:
            try:
                if self.tg.draft2:
                    from mk_draft2 import Draft2
                    self.draft = Draft2(self.tg.draft2, self.tg)
                else:
                    from mk_draft import Draft
                    self.draft = Draft(model_dir, self.tg)
            except ImportError:
                self.draft = None
        self.steps = 0
        self.pending = []
        self.active = None
        self.counters = {"cycles": 0, "tokens": 0, "rows": 0, "host_syncs": 0, "prefix_hits": 0, "prefix_tokens": 0}
        self.cached = []            # the tokens whose K/V the target holds, by position (valid up to tg.pos)
        self._stale = False         # reset() was called: the state is kept until the next sequence shows its first tokens
        self.request_info = {}      # rid -> {"cached_tokens": prompt tokens kept from the cache} (mkserve reads and removes it)
        self.embeds = {}            # rid -> (start, rows): input embeddings for part of a waiting request's prompt
        from mk_kvstore import KVStore
        self.kv = KVStore(f"{Path(model_dir).name}-{self.tg.passes}p")   # parked sequences (memory, then the SSD)
        self._sentinel = 0          # stand-ins for embedding positions in the cached token list: never match a prompt
        self.load_seconds = time.time() - t0
        self._warm()

    def _warm(self):
        """Compile every kernel variant (K = 1..8) before anything is timed."""
        for K in list(range(1, 9)) + [16]:
            for tok_only in (False, True):
                lg = self.tg.forward(mx.zeros((K,), dtype=mx.int32), tok_only=tok_only)
                self.tg.commit(K)
                mx.eval(lg)
        if self.draft is not None:
            self.draft.warm()
        if mk_target.PREFILL != "off":
            for K in PF_BUCKETS:
                for _ in range(2):
                    self._hard_reset()
                    tok, f = self.tg.prefill([0] * K, K, want_feats=self.draft is not None)
                    if self.draft is not None:
                        self.draft.append(f, 0, K)
                    mx.eval(tok, *self._state())
        self._hard_reset()
        mx.synchronize()

    @staticmethod
    def _tok(out):
        """forward(tok_only=True) gives tokens [K]; without the 8-bit copy (MK_SHORT=0) it gives logits [K, vocab]."""
        return out if out.ndim == 1 else mx.argmax(out, axis=-1).astype(mx.int32)

    def _state(self):
        return self.tg.state()

    # ------------------------------------------------------------------------------------------ contract
    def reset(self) -> None:
        """A new sequence starts. The old one's state is dropped when the new one's first tokens are known: at once by
        step() / generate(), after the comparison with the cached tokens by prefill() and poll() (prefix cache)."""
        self._stale = True

    def _hard_reset(self) -> None:
        self.tg.reset()
        if self.draft is not None:
            self.draft.reset()
        self.cached = []
        self._stale = False

    def _begin(self, prompt, need_last: bool) -> int:
        """Start the sequence that begins with `prompt`: keep the longest prefix it shares with the cached sequence, cut
        back to a snapshot position (need_last: at least the last prompt token is left to run, for its logits). Returns
        the number of prompt tokens kept; the K/V they keep is what a prefill of them would write again in exact mode."""
        tg, keep = self.tg, 0
        if tg.snaps and self.cached is not None:
            m = min(len(self.cached), tg.pos, len(prompt) - (1 if need_last else 0))
            lcp = 0
            while lcp < m and self.cached[lcp] == prompt[lcp]:
                lcp += 1
            keep = max((q for q in tg.snaps if q <= lcp), default=0)
        if keep == 0:
            self._hard_reset()
            return 0
        old_pos = tg.pos
        tg.rollback(keep)
        if self.draft is not None:
            self.draft.rollback(old_pos)
        self._stale = False
        self.counters["prefix_hits"] += 1
        self.counters["prefix_tokens"] += keep
        return keep

    def _switch(self, prompt) -> None:
        """Before a request starts: if a parked sequence shares more of its prompt than the live one does, park the live
        one and bring that one back; if the request would overwrite much of the live one, park it first."""
        from mk_kvstore import lcp
        kv, tg = self.kv, self.tg
        if self.cached is not None and tg.pos > 0 and tg.pending is None:
            # a snapshot where the last request stopped: its continuation (the next turn of a conversation, an answer
            # after a reasoning budget) resumes exactly there instead of at the last 256-position snapshot
            tg.flush()
            tg.snaps[tg.pos] = list(tg.cw)
        if not kv.enabled or self.cached is None:
            return
        limit = len(prompt) - 1
        live = lcp(self.cached, prompt, min(tg.pos, limit)) if tg.snaps else 0
        i, shared = kv.best(prompt, limit)
        # positions of the live sequence this request would overwrite: a report check that shares only the system prompt
        # with an agent conversation would otherwise wipe the conversation's cache, and its next turn reads it all again
        discard = tg.pos - live
        # bring a parked sequence back when it saves real work: most of it is reused (a short transcription context), or it
        # saves a lot (an agent conversation); restoring a 35K-position entry to reuse 262 positions costs more than it saves
        gain = shared - live
        if i is not None and gain > 128 and (shared * 2 >= kv.entries[i]["pos"] or gain > 2048):
            entry = kv.entries[i]
            kv.park(tg, self.draft, self.cached)
            self.cached = kv.restore(entry, tg, self.draft)
            self._stale = True
            if os.environ.get("MK_KV_LOG"):
                print(f"MK_KV restored {entry['pos']} positions ({shared} shared; live shared {live}) {kv.summary()}", flush=True)
        elif discard > 1024 or live < 256:     # or nothing of it would be kept (a short transcription context)
            kv.park(tg, self.draft, self.cached)
            if os.environ.get("MK_KV_LOG"):
                print(f"MK_KV parked the live sequence ({discard} positions past the shared {live}) {kv.summary()}", flush=True)

    def step(self, tokens) -> mx.array:
        tokens = np.ascontiguousarray(np.asarray(tokens, dtype=np.int64))
        if tokens.shape[0] != 1:
            raise ValueError("this engine decodes one sequence (max_batch 1)")
        if self._stale:
            self._hard_reset()
        if self.cached is not None:
            self.cached.append(int(tokens.reshape(-1)[0]))
        if self.tg.pos >= self.max_context:
            raise ValueError(f"context limit {self.max_context} reached")
        logits = self.tg.forward(mx.array(tokens.astype(np.int32)))
        self.tg.commit(1)
        mx.async_eval(logits, *self._state())
        self.steps += 1
        return logits

    def generate(self, tokens, n_steps: int):
        tokens = np.ascontiguousarray(np.asarray(tokens, dtype=np.int64))
        if tokens.shape[0] != 1:
            raise ValueError("this engine decodes one sequence (max_batch 1)")
        tok = mx.array(tokens.astype(np.int32))
        out = []
        if self._stale:
            self._hard_reset()
        self.cached = None          # generate() does not track its tokens: no prefix reuse after it
        for _ in range(int(n_steps)):
            tok = self._tok(self.tg.forward(tok, tok_only=True))
            self.tg.commit(1)
            mx.async_eval(tok, *self._state())
            out.append(tok)
        res = mx.stack(out, axis=1)
        mx.eval(res)
        return res

    # ------------------------------------------------------------------------------------------ serving
    @staticmethod
    def _sampling(sampling):
        """{"temperature", "top_p", "top_k", "seed"} -> the request's sampler (None: greedy). Without a seed the
        request takes a random one; with one, the same request gives the same tokens."""
        if not sampling or float(sampling.get("temperature") or 0.0) <= 0.0:
            return None
        import types
        top_p = float(sampling.get("top_p") or 1.0)
        top_k = int(sampling.get("top_k") or 0)
        seed = sampling.get("seed")
        seed = int.from_bytes(os.urandom(4), "little") if seed is None else int(seed) & 0xFFFFFFFF
        return types.SimpleNamespace(sp=mx.array([1.0 / float(sampling["temperature"]), top_p, float(top_k)], dtype=mx.float32),
                                     seed=mx.array([seed], dtype=mx.uint32), filt=top_p < 1.0 or top_k > 0)

    def sample(self, logits, q: int, sampling) -> int:
        """The token a sampling request draws at position q from a step()'s logits [1, vocab] (what submit() / poll()
        return for that request: tests compare the two)."""
        s = sampling if hasattr(sampling, "sp") else self._sampling(sampling)
        return int(mk_target.sample_rows(logits.reshape(1, -1), s.sp, s.seed, mx.array([q], dtype=mx.int32), s.filt)[0].item())

    def submit(self, rid, prompt, n, sampling=None, embeds=None) -> None:
        """embeds = (start, float32 [N, D]), or a list of (start, rows) or (start, rows, key): prompt positions start ..
        start + N - 1 take these rows as their input embeddings (the norm applied, as Inkling's embedding layer outputs
        them), e.g. projected audio or an image's vision features; the prompt holds any valid token there. A block with
        a key (an integer naming its content, such as a hash of the image) is reused from the prefix cache by a later
        prompt that has the same key at the same positions; one without is never reused."""
        prompt, n = [int(t) for t in prompt], int(n)
        if embeds is not None:
            blocks = [embeds] if not isinstance(embeds, list) else embeds
            norm, end = [], 0
            for block in sorted(blocks, key=lambda b: int(b[0])):
                start, rows = int(block[0]), np.asarray(block[1], dtype=np.float32)
                key = int(block[2]) if len(block) > 2 and block[2] is not None else None
                if rows.ndim != 2 or rows.shape[1] != 2304 or start < end or start + rows.shape[0] > len(prompt):
                    raise ValueError(f"request {rid!r}: embeddings {rows.shape} at {start} do not fit the prompt")
                norm.append((start, rows, key))
                end = start + rows.shape[0]
            self.embeds[rid] = norm
        if len(prompt) < 1 or n < 1 or len(prompt) + n - 1 > self.max_context:
            raise ValueError(f"request {rid!r}: prompt {len(prompt)} + {n} new tokens exceeds {self.max_context} positions")
        if min(prompt) < 0 or max(prompt) >= 201024:
            raise ValueError(f"request {rid!r}: token id out of range")
        self.pending.append((rid, prompt, n, self._sampling(sampling)))

    def _prefill(self, prompt, embeds=None):
        """The prompt in wide passes of up to PF_CHUNK rows (the last few tokens, if fewer than 9 remain, in a pass of
        the step's kernels); returns the first new token (lazy). embeds = [(start, rows), ...] relative to this prompt,
        in order: every pass is then a wide one, so the rows go in where the token embeddings would."""
        tg = self.tg
        i, last = 0, None
        first_at = embeds[0][0] if embeds else None
        while mk_target.PREFILL != "off" and (len(prompt) - i > 8 or (embeds and i < len(prompt))):
            n = min(PF_CHUNK, len(prompt) - i)
            if first_at is not None and i < first_at < i + n:
                n = first_at - i            # a pass ends where the embeddings start: a snapshot there (below)
            K = next(k for k in PF_BUCKETS if k >= n)
            if tg.pos + K > tg.T + 8:          # at the end of the context: no padding past it
                if n < PF_BUCKETS[0]:
                    break
                K = n = max(k for k in PF_BUCKETS if k <= n)
            p = tg.pos
            emb = None
            if embeds:
                block = use = None
                for s0, rows in embeds:
                    lo, hi = max(i, s0), min(i + n, s0 + rows.shape[0])
                    if lo < hi:
                        if block is None:
                            block = np.zeros((K, rows.shape[1]), dtype=np.float32)
                            use = np.zeros((K, 1), dtype=bool)
                        block[lo - i:hi - i] = rows[lo - s0:hi - s0]
                        use[lo - i:hi - i] = True
                if block is not None:
                    emb = (mx.array(block), mx.array(use))
            last, f = tg.prefill(prompt[i:i + n], K, want_feats=self.draft is not None, emb=emb)
            self._last_row = n - 1
            if self.draft is not None:
                self.draft.append(f, p, n)
            mx.async_eval(last, *self._state())
            i += n
            if first_at is not None and i == first_at:
                # where the embeddings start: the next request with the same text before its audio resumes here
                tg.snaps[tg.pos] = list(tg.cw)
        while i < len(prompt):
            K = min(8, len(prompt) - i)
            toks = mx.array(np.asarray(prompt[i:i + K], dtype=np.int32))
            if self.draft is not None:
                logits, f = tg.forward(toks, want_feats=True, tok_only=True)
                self.draft.append(f, tg.pos, K)
            else:
                logits = tg.forward(toks, tok_only=True)
            tg.commit(K)
            self._last_row = K - 1
            last = self._tok(logits)[K - 1:K]
            mx.async_eval(last, *self._state())
            i += K
        return last

    def prefill(self, tokens) -> int:
        """Optional (mkserve's step mode): take these tokens at the next positions, as one step() call each would, in
        wide passes. Returns how many of them were kept from the cache (after reset(): the prefix cache)."""
        tokens = [int(t) for t in tokens]
        keep = self._begin(tokens, need_last=False) if self._stale else 0
        if keep:
            self.cached = self.cached[:keep]
        if self.cached is not None:
            self.cached += tokens[keep:]
        if tokens[keep:]:
            if self.tg.pos + len(tokens) - keep > self.max_context:
                raise ValueError(f"context limit {self.max_context} reached")
            mx.eval(self._prefill(tokens[keep:]), *self._state())
        return keep

    def project_audio(self, frames_b64: str, count: int):
        """Speech encoder frames (base64 of float16 [count, encoder_dim]) -> input embeddings through the projector named
        by MK_PROJECTOR (mk_audio.py), for submit(..., embeds=(start, rows))."""
        if getattr(self, "_projector", None) is None:
            folder = os.environ.get("MK_PROJECTOR", "")
            if not folder:
                raise ValueError("no audio projector: set MK_PROJECTOR to its folder")
            from mk_audio import Projector
            self._projector = Projector(folder)
        import base64
        frames = np.frombuffer(base64.b64decode(frames_b64), dtype="<f2").astype(np.float32)
        dim = self._projector.dim
        if count < 1 or frames.size != count * dim:
            raise ValueError(f"audio: {frames.size} values are not {count} frames of {dim}")
        return self._projector.rows(frames.reshape(count, dim))

    def cancel(self, rid) -> None:
        """Optional (mkserve): the client of request rid is gone. A waiting request is dropped; the running one stops
        after its current cycle (its K/V stays as the cache of the next request)."""
        self.pending = [r for r in self.pending if r[0] != rid]
        self.embeds.pop(rid, None)
        if self.active is not None and self.active["rid"] == rid:
            self.active = None

    def poll(self):
        events = []
        act = self.active
        if act is None:
            if not self.pending:
                return events
            rid, prompt, n, samp = self.pending.pop(0)
            blocks = self.embeds.pop(rid, None) or []
            # the sequence as the cache knows it: each embedding position stands as an id no token has, the same for the
            # same content (a block's key) and unique for a block without one, so a prompt only matches what it holds
            key_prompt = list(prompt)
            for start, rows, key in blocks:
                for j in range(rows.shape[0]):
                    if key is None:
                        self._sentinel -= 1
                        key_prompt[start + j] = self._sentinel
                    else:
                        key_prompt[start + j] = -(10**9 + (key % 10**12) * 10**5 + j)
            self._switch(key_prompt)
            keep = self._begin(key_prompt, need_last=True)
            self.request_info[rid] = {"cached_tokens": keep}
            rel = []
            for start, rows, _key in blocks:
                if start + rows.shape[0] <= keep:
                    continue                # already in the cache
                cut = max(0, keep - start)
                rel.append((start + cut - keep, rows[cut:]))
            first = self._prefill(prompt[keep:], rel or None)
            # a snapshot at the end of the prompt: a later request that shares the prompt but not the answer resumes here
            self.tg.snaps[self.tg.pos] = list(self.tg.cw)
            if samp is not None:    # the first token, drawn from the logits of the last prompt row
                first = mk_target.sample_rows(self.tg.row_logits(self._last_row), samp.sp, samp.seed,
                                              mx.array([len(prompt)], dtype=mx.int32), samp.filt)[0]
            spec = self.draft is not None and len(prompt) + n + 16 <= self.tg.T
            hist = key_prompt
            act = self.active = {"rid": rid, "n": n, "sent": 0, "tok": first, "spec": spec, "hist": hist, "samp": samp}
            self.cached = act["hist"]           # the prompt and every committed token after it, as the request goes on
            if spec:
                from mk_draft import Cal
                act["cal"] = Cal()
            t = int(first.item())
            self.counters["host_syncs"] += 1
            act["sent"] = 1
            act["last"] = t
            act["hist"].append(t)
            events.append((rid, [t], n == 1))
            if n == 1:
                self.active = None
            return events
        left = act["n"] - act["sent"]
        if act["spec"]:
            toks = self.draft.cycle(self, act, left)
        elif act["samp"] is not None:
            samp = act["samp"]
            logits = self.tg.forward(mx.array([act["last"]], dtype=mx.int32))
            self.tg.commit(1)
            tok = mk_target.sample_rows(logits, samp.sp, samp.seed, mx.array([self.tg.pos], dtype=mx.int32), samp.filt)[0]
            mx.async_eval(tok, *self._state())
            toks = [int(tok.item())]
            act["hist"].extend(toks)
            self.counters["host_syncs"] += 1
        else:
            tok = self._tok(self.tg.forward(mx.array([act["last"]], dtype=mx.int32), tok_only=True))
            self.tg.commit(1)
            toks = [int(tok.item())]
            act["hist"].extend(toks)
            self.counters["host_syncs"] += 1
        toks = toks[:left]
        act["sent"] += len(toks)
        act["last"] = toks[-1]
        self.counters["cycles"] += 1
        self.counters["tokens"] += len(toks)
        done = act["sent"] >= act["n"]
        events.append((act["rid"], toks, done))
        if done:
            self.active = None
        return events

    def close(self) -> None:
        self.tg = None
        self.draft = None

    def stats(self) -> dict:
        c = dict(self.counters)
        if c["cycles"]:
            c["tokens_per_cycle"] = round(c["tokens"] / c["cycles"], 3)
        return {"implementation": "own MLX / Metal K-row passes (engine/target.py); DSpark draft-and-verify in submit/poll"
                if self.draft is not None else "own MLX / Metal K-row passes (engine/target.py); no speculation",
                "steps": self.steps, "spec": c, "load_seconds": round(self.load_seconds, 1),
                "peak_memory_gb": round(mx.get_peak_memory() / 1e9, 2)}

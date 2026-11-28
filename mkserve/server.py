#!/usr/bin/env python
"""The two processes behind `python -m mkserve serve`: an engine worker that owns the GPU and drives one megakernel
engine, and an HTTP front end with the OpenAI API. They are joined by a Unix socket (one JSON object per line), the
shape eval/serve_engine.py has; that file is the evaluator's measuring shim and is left alone, this one is for use.

    python -m mkserve.server engine   --workspace WS --model-dir D --sock S [--mode auto|native|step] ...
    python -m mkserve.server frontend --model-dir D --sock S --port P [--host H] --name MODEL ...

The engine worker has two ways to drive an engine (the contract is engine/api.py of the engine's source):

  native  the engine's own submit()/poll(): admission, batching and prefill are the engine's, requests of any
          length are taken while others decode. Greedy only, and the contract has no cancel: a request that ends
          early (end-of-sequence, a stop string, a client that left) is closed by the front end while its row
          decodes on to max_tokens inside the engine.
  step    reset()/step() driven from here: requests that arrive together and have prompts of the same length form
          a batch, the prompt is fed one token per step, then one step per token. Sampling (temperature, top_k,
          top_p, seed) and early stopping work, but nothing is admitted while a batch decodes and the first token is slow.

Front end: /v1/models, /health, /v1/completions and /v1/chat/completions, streaming or not. The chat template is
the model's own (tokenizer_config.json or chat_template.jinja), rendered with jinja2 as transformers does.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import itertools
import json
import os
import queue
import socket
import sys
import threading
import time
from pathlib import Path

from mkserve.output import ResponseParser, load_response_template


# ---------------------------------------------------------------------------------------------- the engine worker

def load_engine(workspace: str, model_dir: str, max_batch: int, max_context: int):
    api = Path(workspace) / "engine" / "api.py"
    sys.path.insert(0, str(api.parent))
    spec = importlib.util.spec_from_file_location("engine_api", api)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Engine(model_dir, max_batch, max_context)


def sample_rows(torch, logits, temp, top_p, top_k, kmax, u):
    """One token per row from logits [R, V]: temperature, then top-k, then top-p on what is left (the order the
    general engines use), drawn by inverse CDF with one uniform number per row (u [R]), so a request's seed alone
    decides its tokens. temp, top_p [R] float; top_k [R] int, 0 = off; kmax = the largest top_k of the rows, or 0
    when a row has none (then the whole vocabulary is sorted). The k largest are taken before the temperature is
    applied (dividing by a positive number keeps the order), so only k values per row are touched after that."""
    if kmax:
        vals, idx = torch.topk(logits, kmax, dim=-1)           # sorted, largest first
    else:
        vals, idx = torch.sort(logits, dim=-1, descending=True)
    vals = vals.float() / temp[:, None]
    rank = torch.arange(vals.shape[-1], device=vals.device)[None, :]
    vals = vals.masked_fill((top_k[:, None] > 0) & (rank >= top_k[:, None]), float("-inf"))
    p = torch.softmax(vals, -1)
    p = p * (((p.cumsum(-1) - p) < top_p[:, None]))            # the smallest set whose mass reaches top_p
    cdf = (p / p.sum(-1, keepdim=True)).cumsum(-1)
    last = (p > 0).sum(-1, keepdim=True) - 1                   # never past the last token that was kept
    j = torch.minimum(torch.searchsorted(cdf, u[:, None]), last)
    return idx.gather(-1, j).squeeze(-1)


def engine_main(args) -> int:
    import numpy as np

    os.chdir(args.workspace)
    engine = load_engine(args.workspace, args.model_dir, args.max_batch, args.max_context)
    has_native = all(hasattr(engine, m) for m in ("submit", "poll"))
    import inspect
    _sub = set(inspect.signature(engine.submit).parameters) if hasattr(engine, "submit") else set()

    def opts(req) -> dict:
        """Optional submit() keywords an engine declares: stops (the request's stop token ids: the engine ends the request
        on one instead of decoding on until the cancel arrives) and wide (every prompt pass a wide one: a cache warm-up)."""
        kw = {}
        if "stops" in _sub and req.get("eos"):
            kw["stops"] = [int(t) for t in req["eos"]]
        if "wide" in _sub and req.get("wide"):
            kw["wide"] = True
        return kw
    mode = args.mode if args.mode != "auto" else ("native" if has_native else "step")
    if mode == "native" and not has_native:
        print("MKSERVE this engine has no submit()/poll(): using step mode", flush=True)
        mode = "step"
    Path(args.sock).unlink(missing_ok=True)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(args.sock)
    server.listen(1)
    print(f"MKSERVE engine ready: mode={mode} max_batch={args.max_batch} max_context={args.max_context}", flush=True)
    conn, _ = server.accept()
    conn.sendall((json.dumps({"hello": {"mode": mode, "max_batch": args.max_batch, "max_context": args.max_context,
                                        "native_sampling": mode == "native" and bool(getattr(engine, "native_sampling", False))}}) + "\n").encode())
    inbox: queue.Queue = queue.Queue()
    cancelled: set[int] = set()

    def reader() -> None:
        with conn.makefile("r") as stream:
            for line in stream:
                msg = json.loads(line)
                if "cancel" in msg:
                    cancelled.add(int(msg["cancel"]))
                    if mode == "native":  # an engine with cancel(rid) stops the row; the call is made from the loop below
                        inbox.put(msg)
                else:
                    inbox.put(msg)
        inbox.put(None)  # the front end is gone

    threading.Thread(target=reader, daemon=True).start()

    def send(events: list) -> None:
        conn.sendall((json.dumps(events, separators=(",", ":")) + "\n").encode())

    if mode == "native":
        audio_rows: dict = {}   # media key -> projected rows: a conversation resends every earlier turn's audio each request
        while True:
            while True:
                try:
                    req = inbox.get_nowait()
                except queue.Empty:
                    break
                if req is None:
                    return 0
                if "cancel" in req:
                    if hasattr(engine, "cancel"):
                        engine.cancel(req["cancel"])
                    continue
                samp = {k: req[k] for k in ("temperature", "top_p", "top_k", "seed") if k in req}
                if "audio" in req:          # speech encoder frames for part of the prompt (POST /v1/inkling/transcribe)
                    a = req["audio"]
                    try:
                        rows = engine.project_audio(a["frames"], int(a["count"]))
                        engine.submit(req["rid"], req["ids"], req["n"], embeds=(int(a["at"]), rows), **opts(req))
                    except Exception as exc:  # noqa: BLE001  a bad request must not stop the engine
                        print(f"MKSERVE audio request {req['rid']} refused: {exc}", flush=True)
                        send([[int(req["rid"]), [], True]])
                    continue
                if "media" in req:          # image rows (MK_VISION_URL) or speech encoder frames (MK_PROJECTOR) for placeholders
                    try:
                        import base64 as _b64
                        import numpy as _np
                        blocks = []
                        for m in req["media"]:
                            if "rows" in m:
                                rows = _np.frombuffer(_b64.b64decode(m["rows"]), dtype="<f2").astype(_np.float32)
                                rows = rows.reshape(int(m["count"]), -1)
                            else:
                                ck = (int(m["key"]), int(m["frame_count"]))
                                rows = audio_rows.pop(ck, None)         # popped and put back: least recently used first
                                if rows is None:
                                    rows = engine.project_audio(m["frames"], int(m["frame_count"]))
                                audio_rows[ck] = rows
                                while sum(r.nbytes for r in audio_rows.values()) > (1 << 30):     # 1 GB
                                    audio_rows.pop(next(iter(audio_rows)))
                                if rows.shape[0] != int(m["count"]):
                                    raise ValueError(f"audio: {rows.shape[0]} projected rows for {m['count']} positions")
                            blocks.append((int(m["at"]), rows, int(m["key"])))
                        engine.submit(req["rid"], req["ids"], req["n"], embeds=blocks, **opts(req))
                    except Exception as exc:  # noqa: BLE001  a bad request must not stop the engine
                        print(f"MKSERVE media request {req['rid']} refused: {exc}", flush=True)
                        send([[int(req["rid"]), [], True]])
                    continue
                if samp and getattr(engine, "native_sampling", False):  # the engine draws tokens inside its own path
                    engine.submit(req["rid"], req["ids"], req["n"], sampling=samp, **opts(req))
                else:
                    engine.submit(req["rid"], req["ids"], req["n"], **opts(req))
            events = engine.poll()
            if events:
                info = getattr(engine, "request_info", None) or {}  # optional: per-request facts (prompt tokens kept from a prefix cache)
                send([[int(rid), [int(t) for t in tokens], bool(finished)] + ([info.pop(rid)] if rid in info else [])
                      for rid, tokens, finished in events])
            else:
                time.sleep(0.0002)

    try:
        import torch
        cuda = torch.cuda.is_available()
    except ImportError:
        cuda = False
    if not cuda:  # Apple silicon: the engine takes NumPy int64 tokens and returns mx.array or NumPy logits
        return step_loop_host(engine, args, inbox, cancelled, send)
    waiting: list[dict] = []
    with torch.no_grad():
        while True:
            if not waiting:
                req = inbox.get()
                if req is None:
                    return 0
                waiting.append(req)
            deadline = time.perf_counter() + args.gather_ms / 1e3
            while len(waiting) < args.max_batch and time.perf_counter() < deadline:
                try:
                    req = inbox.get(timeout=max(0.0, deadline - time.perf_counter()))
                except queue.Empty:
                    break
                if req is None:
                    return 0
                waiting.append(req)
            length = len(waiting[0]["ids"])
            cohort = [r for r in waiting if len(r["ids"]) == length][:args.max_batch]
            waiting = [r for r in waiting if all(r is not c for c in cohort)]
            B = len(cohort)
            ids = torch.tensor([r["ids"] for r in cohort], device="cuda")
            temp = torch.tensor([float(r.get("temperature") or 0.0) for r in cohort], device="cuda")
            top_p = torch.tensor([float(r.get("top_p") or 1.0) for r in cohort], device="cuda")
            top_k = torch.tensor([int(r.get("top_k") or 0) for r in cohort], device="cuda")
            rows = (temp > 0).nonzero().flatten()              # the rows that sample; the others take the argmax
            n_max = max(r["n"] for r in cohort)
            if rows.numel():  # everything the sampler needs per step is prepared here: no host work inside the loop
                picked = rows.tolist()
                everyone = len(picked) == B
                ks = [int(cohort[i].get("top_k") or 0) for i in picked]
                kmax = max(ks) if all(k > 0 for k in ks) else 0
                # one uniform number per token and row, from the request's own seed
                uniform = torch.from_numpy(np.stack([np.random.default_rng(int(cohort[i]["seed"]) if cohort[i].get("seed") is not None else None)
                                                     .random(n_max, dtype=np.float32) for i in picked])).cuda()
                s_temp, s_top_p, s_top_k = temp[rows], top_p[rows], top_k[rows]
            eos = [set(r.get("eos") or ()) for r in cohort]
            done = [False] * B
            engine.reset()
            for t in range(length - 1):
                engine.step(ids[:, t])
            token = ids[:, -1]
            for k in range(n_max):
                logits = engine.step(token)
                if rows.numel() and s_temp.device != logits.device:  # a multi-card engine returns logits on a card of its choice
                    s_temp, s_top_p, s_top_k, uniform, rows = (x.to(logits.device) for x in (s_temp, s_top_p, s_top_k, uniform, rows))
                if rows.numel() and everyone:
                    token = sample_rows(torch, logits, s_temp, s_top_p, s_top_k, kmax, uniform[:, k])
                else:
                    token = logits.argmax(-1)
                    if rows.numel():
                        token[rows] = sample_rows(torch, logits[rows], s_temp, s_top_p, s_top_k, kmax, uniform[:, k])
                chosen = token.tolist()
                token = token.to(ids.device)
                events = []
                for i, r in enumerate(cohort):
                    if done[i]:
                        continue
                    last = k == r["n"] - 1 or chosen[i] in eos[i] or r["rid"] in cancelled
                    events.append([r["rid"], [chosen[i]], last])
                    done[i] = last
                if events:
                    send(events)
                if all(done):
                    break
            cancelled.difference_update(r["rid"] for r in cohort)


def sample_host(logits, temp: float, top_p: float, top_k: int, u: float) -> int:
    """sample_rows for one row on the host (NumPy): temperature, top-k, top-p, inverse CDF with the uniform number u."""
    import numpy as np
    x = np.asarray(logits, dtype=np.float32)
    if top_k and top_k < x.size:
        idx = np.argpartition(-x, top_k)[:top_k]
    else:
        idx = np.arange(x.size)
    idx = idx[np.argsort(-x[idx], kind="stable")]
    v = x[idx] / temp
    p = np.exp(v - v.max())
    p /= p.sum()
    keep = (np.cumsum(p) - p) < top_p
    p = np.where(keep, p, 0.0)
    cdf = np.cumsum(p / p.sum())
    j = min(int(np.searchsorted(cdf, u)), int(keep.sum()) - 1)
    return int(idx[j])


def step_loop_host(engine, args, inbox, cancelled, send) -> int:
    """Step mode for engines whose logits are host-readable (MLX on Apple silicon): requests one after another
    (batch 1 is what these engines are built for), the prompt in one step() call when the engine takes several
    tokens per row, sampling on the host."""
    import numpy as np
    while True:
        req = inbox.get()
        if req is None:
            return 0
        if "cancel" in req:
            continue
        rid, ids, n = req["rid"], req["ids"], req["n"]
        temp = float(req.get("temperature") or 0.0)
        rng = np.random.default_rng(int(req["seed"]) if req.get("seed") is not None else None)
        eos = set(req.get("eos") or ())
        engine.reset()
        kept = 0
        if hasattr(engine, "prefill"):         # optional: the prompt in the engine's own wide passes (and its prefix cache)
            kept = int(engine.prefill(ids[:-1]) or 0)
        else:
            for t in ids[:-1]:
                engine.step(np.array([t], dtype=np.int64))
        token = ids[-1]
        for k in range(n):
            logits = engine.step(np.array([token], dtype=np.int64))
            if not isinstance(logits, np.ndarray):  # an mx.array (NumPy has no bfloat16)
                import mlx.core as mx
                logits = np.array(logits.astype(mx.float32))
            row = logits.reshape(-1, logits.shape[-1])[0]
            if temp > 0:
                token = sample_host(row, temp, float(req.get("top_p") or 1.0), int(req.get("top_k") or 0), float(rng.random()))
            else:
                token = int(row.argmax())
            last = k == n - 1 or token in eos or rid in cancelled
            send([[rid, [token], last] + ([{"cached_tokens": kept}] if k == 0 and kept else [])])
            if last:
                break
        cancelled.discard(rid)


# ------------------------------------------------------------------------------------------------- the front end

def load_chat_template(model_dir: Path):
    """The model's chat template as a jinja2 template, with the names transformers gives it; None if it has none."""
    cfg_path = model_dir / "tokenizer_config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    text = (model_dir / "chat_template.jinja").read_text() if (model_dir / "chat_template.jinja").exists() else cfg.get("chat_template")
    if isinstance(text, list):  # several named templates: take the default one
        text = next((t.get("template") for t in text if t.get("name") == "default"), None)
    if not text:
        return None, {}
    import datetime

    from jinja2.exceptions import TemplateError
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def raise_exception(message):
        raise TemplateError(message)

    from jinja2 import nodes
    from jinja2.ext import Extension, loopcontrols

    class Generation(Extension):
        """`{% generation %}...{% endgeneration %}` (transformers marks the assistant's own text with it): rendered as is."""
        tags = {"generation"}

        def parse(self, parser):
            lineno = next(parser.stream).lineno
            body = parser.parse_statements(["name:endgeneration"], drop_needle=True)
            return nodes.Scope(body, lineno=lineno)

    # the extensions transformers compiles chat templates with: `break` / `continue` and the generation marker
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols, Generation])
    # transformers' tojson: templates pass sort_keys and separators (Inkling 2 declares its tools with both), and the
    # bytes the model was trained on depend on them
    env.filters["tojson"] = lambda x, ensure_ascii=False, indent=None, separators=None, sort_keys=False: json.dumps(
        x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)
    env.globals["raise_exception"] = raise_exception
    env.globals["strftime_now"] = lambda fmt: datetime.datetime.now().strftime(fmt)
    special = {}
    for key in ("bos_token", "eos_token", "pad_token", "unk_token"):
        value = cfg.get(key)
        special[key] = value.get("content") if isinstance(value, dict) else (value or "")
    return env.from_string(text), special


END_OF_TURN = ("<|im_end|>", "<|eot_id|>", "<|end_of_turn|>", "<end_of_turn>", "<|endoftext|>", "<|content_model_end_sampling|>", "<|END_OF_TURN_TOKEN|>")


def eos_ids(model_dir: Path, tok) -> list[int]:
    ids: set[int] = set()
    gen = model_dir / "generation_config.json"
    if gen.exists():
        value = json.loads(gen.read_text()).get("eos_token_id")
        ids.update(value if isinstance(value, list) else [value] if value is not None else [])
    cfg_path = model_dir / "tokenizer_config.json"
    if cfg_path.exists():
        value = json.loads(cfg_path.read_text()).get("eos_token")
        value = value.get("content") if isinstance(value, dict) else value
        if value and tok.token_to_id(value) is not None:
            ids.add(tok.token_to_id(value))
    model_cfg = model_dir / "config.json"
    if model_cfg.exists():   # some checkpoints name the end token only here
        value = json.loads(model_cfg.read_text()).get("eos_token_id")
        ids.update(value if isinstance(value, list) else [value] if value is not None else [])
    for name in END_OF_TURN:  # the token a chat template ends the assistant's turn with, when the vocabulary has it
        if tok.token_to_id(name) is not None:
            ids.add(tok.token_to_id(name))
    return sorted(int(i) for i in ids)


class Core:
    """What every API shares: the tokenizer, the chat template, the model's answer format, and one request's run
    through the engine worker (token ids in; per token, the pieces of reasoning, text and tool calls it completed)."""

    def __init__(self, args, tok, template, special, eos, vocab, hello, writer, rt):
        self.args, self.tok, self.template, self.special, self.eos, self.vocab = args, tok, template, special, eos, vocab
        self.hello, self.writer, self.rt = hello, writer, rt
        self.native = hello["mode"] == "native"
        self.name = args.name
        # an empty reasoning block, for thinking_budget 0: Inkling 2's own markers, when the tokenizer has them
        marks = "<|message_model|><|content_thinking|><|end_message|>"
        ids = tok.encode(marks, add_special_tokens=False).ids
        self.empty_thinking = ids if len(ids) == 3 else None   # [open model message, thinking, end message]
        self.queues: dict[int, asyncio.Queue] = {}
        self.extra = {}  # rid -> facts the engine reported about the request (cached_tokens)
        self.counter = itertools.count(1)
        self.warned = False
        # Prometheus-style counters (GET /metrics): what a user needs to see the engine at work
        self.stats = {"requests_total": 0, "requests_running": 0, "requests_failed_total": 0, "prompt_tokens_total": 0, "cached_prompt_tokens_total": 0,
                      "generation_tokens_total": 0, "time_to_first_token_seconds_sum": 0.0, "time_to_first_token_seconds_count": 0,
                      "time_per_output_token_seconds_sum": 0.0, "time_per_output_token_seconds_count": 0}
        # GET /api/status (the shape of oMLX's, for desktop widgets): requests running now and the last one finished
        self.started = time.time()
        self.live: dict[int, dict] = {}     # rid -> {"start", "first", "prompt", "out", "marks": [(time, tokens)]}
        self.last_request: dict | None = None
        self.prefill_seconds = self.decode_seconds = 0.0
        self.prefilled_total = 0

    def status(self) -> dict:
        """GET /api/status: the fields of oMLX's that a desktop widget reads, with the requests running now."""
        st, now = self.stats, time.perf_counter()
        prefilling, generating = [], []
        for v in list(self.live.values()):
            if v["first"] is None:
                prefilling.append({"model": self.name, "processed": None, "total": v["prompt"], "tokens_per_second": None,
                                   "elapsed_seconds": round(now - v["start"], 1)})
            else:
                marks = v["marks"]
                rate = None
                if len(marks) >= 2 and marks[-1][0] > marks[0][0]:
                    rate = (marks[-1][1] - marks[0][1]) / (marks[-1][0] - marks[0][0])
                elif now > v["first"] + 0.2:
                    rate = v["out"] / (now - v["first"])
                generating.append({"model": self.name, "generated_tokens": v["out"], "tokens_per_second": rate})
        prompt, cached = st["prompt_tokens_total"], st["cached_prompt_tokens_total"]
        return {"status": "ok", "server": "mkserve", "engine": self.args.engine_name, "version": "mkserve",
                "uptime_seconds": round(time.time() - self.started, 1),
                "default_model": self.name, "loaded_models": [self.name], "models_loaded": 1, "models_loading": 0,
                "total_requests": st["requests_total"], "active_requests": st["requests_running"], "waiting_requests": 0,
                "total_prompt_tokens": prompt, "total_completion_tokens": st["generation_tokens_total"], "total_cached_tokens": cached,
                "cache_efficiency": round(100 * cached / prompt, 1) if prompt else 0.0,
                "avg_prefill_tps": round(self.prefilled_total / self.prefill_seconds, 1) if self.prefill_seconds else 0.0,
                "avg_generation_tps": round(st["generation_tokens_total"] / self.decode_seconds, 1) if self.decode_seconds else 0.0,
                "max_context": self.hello.get("max_context"), "model_dir": str(getattr(self.args, "model_dir", "") or ""),
                "live": {"prefilling": prefilling, "generating": generating}, "last_request": self.last_request}

    def check(self, ids: list[int], n: int) -> str | None:
        """The engines do not check their inputs (an out-of-range token id is an out-of-bounds read on the GPU):
        every limit is enforced here. Returns the reason a request is refused, or None."""
        if not ids:
            return "the prompt is empty"
        if n < 1:
            return "max_tokens must be at least 1"
        if any(t < 0 or t >= self.vocab for t in ids):
            return f"token ids must be in [0, {self.vocab})"
        if len(ids) + n > self.hello["max_context"]:
            return (f"prompt ({len(ids)} tokens) plus max_tokens ({n}) exceeds this engine's context of "
                    f"{self.hello['max_context']} tokens")
        return None

    def render(self, messages: list, tools=None, **kwargs) -> list[int]:
        """A conversation through the model's own chat template, as token ids (raises ValueError if it is refused)."""
        if self.template is None:
            raise ValueError("this model has no chat template; use /v1/completions")
        try:
            text = self.template.render(messages=messages, tools=tools or None, add_generation_prompt=True,
                                        **self.special, **{k: v for k, v in kwargs.items() if v is not None})
        except Exception as exc:  # the template rejects the conversation (an unknown role, a malformed tool call)
            raise ValueError(f"chat template: {exc}") from exc
        return self.tok.encode(text, add_special_tokens=False).ids

    def sampling(self, temperature, top_p=None, top_k=None, seed=None):
        """(sampling parameters or None for greedy, refusal or None)."""
        if temperature is None or float(temperature) <= 0:
            return None, None
        if self.native and not self.hello.get("native_sampling"):
            if self.args.strict_sampling:
                return None, "this engine decodes greedily in native mode; send temperature 0, or start the server with --mode step"
            if not self.warned:
                self.warned = True
                print("MKSERVE warning: sampling parameters are ignored in native mode (greedy decoding); use --mode step for sampling", flush=True)
            return None, None
        return {"temperature": float(temperature), "top_p": float(top_p or 1.0), "top_k": int(top_k or 0), "seed": seed}, None

    def tool_opening(self, name: str) -> list[int] | None:
        """The tokens that open a call of tool `name` in Inkling 2's answer format, or None for a model without them."""
        text = f"<|message_model|>{name}<|content_invoke_tool_json|>"
        if not self.rt or "content_invoke_tool_json" not in json.dumps(self.rt):
            return None
        ids = self.tok.encode(text, add_special_tokens=False).ids
        return ids if self.tok.decode(ids, skip_special_tokens=False) == text else None

    def _decode(self, ids, parse):
        return self.tok.decode(ids, skip_special_tokens=not parse)

    async def run(self, ids: list[int], n: int, sampling=None, stops=(), ignore_eos=False, parse=True, extra=None, stop_ids_extra=(),
                  prime: str = ""):
        """Yields one dict per generated token: {"content": str, "reasoning": str, "tool_calls": [...]} (any may be
        empty: a client can time tokens by them), then {"done": True, "finish": "stop"|"length", "prompt_tokens",
        "completion_tokens", "stop_sequence"}. parse: read the answer's structure (response_template) instead of
        returning its text with the special tokens left out. Stop strings apply to the text (content) only; text that
        could still turn into one is held back."""
        stops = [s for s in stops if s]
        hold = max((len(s) for s in stops), default=1) - 1
        stop_ids = (set() if ignore_eos else set(self.eos)) | set(stop_ids_extra)
        parser = ResponseParser(self.rt) if parse and self.rt else None
        if parser and prime:
            # the opening of the answer was given in the prompt (a forced tool call): the parser reads it first
            parser.feed(prime)
        rid = next(self.counter)
        q: asyncio.Queue = asyncio.Queue()
        self.queues[rid] = q
        msg = {"rid": rid, "ids": ids, "n": n, "eos": sorted(stop_ids)}
        if sampling:
            msg.update(sampling)
        if extra:
            msg.update(extra)
        self.writer.write((json.dumps(msg, separators=(",", ":")) + "\n").encode())
        st = self.stats
        t_start, t_first = time.perf_counter(), None
        t_queued = None
        st["requests_total"] += 1; st["requests_running"] += 1; st["prompt_tokens_total"] += len(ids)
        live = self.live[rid] = {"start": t_start, "first": None, "prompt": len(ids), "out": 0, "marks": []}
        out: list[int] = []
        prefix = read = 0  # incremental detokenization: text is released when it is whole
        content, sent, finish, stop_seq, done = "", 0, None, None, False
        stop_token = None

        def take(pieces, step) -> bool:
            """Sorts parsed pieces into the step; True when a stop string ended the answer."""
            nonlocal content, finish, stop_seq
            for field, value in pieces:
                if field == "tool_calls":
                    step["tool_calls"].append(value)
                elif field == "reasoning_content":
                    step["reasoning"] += value
                else:
                    content += value
                    hits = [(content.find(s), s) for s in stops if s in content]
                    if hits:
                        cut, stop_seq = min(hits)
                        content, finish = content[:cut], "stop"
                        return True
            return False

        def release(step, final=False) -> None:
            nonlocal sent
            upto = len(content) if final else max(sent, len(content) - hold)
            step["content"] += content[sent:upto]
            sent = upto

        try:
            ended = False
            while not done and not ended:
                tokens, done = await q.get()
                if t_first is None and tokens:
                    t_first = time.perf_counter()
                    live["first"] = t_first
                if tokens:
                    live["out"] += len(tokens)
                    live["marks"] = [m for m in live["marks"] if m[0] > time.perf_counter() - 3.0] + [(time.perf_counter(), live["out"])]
                    st["time_to_first_token_seconds_sum"] += t_first - t_start; st["time_to_first_token_seconds_count"] += 1
                for token in tokens:
                    if token in stop_ids:
                        finish, ended = "stop", True
                        stop_token = token
                        break
                    out.append(token)
                    step = {"content": "", "reasoning": "", "tool_calls": []}
                    before = self._decode(out[prefix:read], parse)
                    now = self._decode(out[prefix:], parse)
                    piece = ""
                    if len(now) > len(before) and not now.endswith("\ufffd"):
                        piece = now[len(before):]
                        prefix, read = read, len(out)
                    stopped = take(parser.feed(piece) if parser else ([("content", piece)] if piece else []), step)
                    release(step)
                    if not stopped and len(out) >= n:
                        finish = "length"
                    ended = stopped or finish is not None
                    yield step
                    if ended:
                        break
            if finish is None:
                finish = "length" if len(out) >= n else "stop"
            tail = {"content": "", "reasoning": "", "tool_calls": []}
            if finish == "length":  # nothing follows: release text that was waiting for more bytes
                rest = self._decode(out[prefix:], parse)[len(self._decode(out[prefix:read], parse)):].rstrip("\ufffd")
                if rest and not take(parser.feed(rest) if parser else [("content", rest)], tail) and parser:
                    take(parser.close(), tail)
            elif parser and stop_seq is None:
                take(parser.close(), tail)
            release(tail, final=True)
            if tail["content"] or tail["reasoning"] or tail["tool_calls"]:
                yield tail
            cached = int(self.extra.get(rid, {}).get("cached_tokens", 0))
            st["cached_prompt_tokens_total"] += cached
            yield {"done": True, "finish": finish, "stop_sequence": stop_seq, "prompt_tokens": len(ids), "completion_tokens": len(out), "ids": list(out), "stop_token": stop_token,
                   "cached_tokens": cached}
        except BaseException:
            st["requests_failed_total"] += 1
            raise
        finally:
            st["requests_running"] -= 1
            st["generation_tokens_total"] += len(out)
            if t_first is not None and len(out) > 1:
                st["time_per_output_token_seconds_sum"] += (time.perf_counter() - t_first) / (len(out) - 1)
                st["time_per_output_token_seconds_count"] += 1
            self.queues.pop(rid, None)
            cached = int(self.extra.get(rid, {}).get("cached_tokens", 0))
            self.extra.pop(rid, None)
            self.live.pop(rid, None)
            if t_first is not None:
                t_end = time.perf_counter()
                pf, dec = t_first - t_start, t_end - t_first
                prefilled = max(0, len(ids) - cached)
                self.prefill_seconds += pf; self.decode_seconds += dec; self.prefilled_total += prefilled
                self.last_request = {"completed_at": time.time(), "prompt_tokens": len(ids), "prefilled_tokens": prefilled,
                                     "prefill_seconds": round(pf, 3), "prefill_tps": round(prefilled / pf, 1) if pf > 0 else None,
                                     "completion_tokens": len(out),
                                     "decode_tps": round((len(out) - 1) / dec, 1) if dec > 0 and len(out) > 1 else None}
                # MKSERVE_REQUEST_LOG: one JSON line per request (numbers only, never text): for latency work
                if os.environ.get("MKSERVE_REQUEST_LOG"):
                    try:
                        with open(os.environ["MKSERVE_REQUEST_LOG"], "a") as fh:
                            fh.write(json.dumps(dict(self.last_request, queued_seconds=round(t_start - t_queued, 3)
                                                     if t_queued else None, audio="audio" in (extra or {}))) + "\n")
                    except OSError:
                        pass
            if not done:  # ended early: a step-mode batch drops the row; a native engine cancels it if it can
                self.writer.write((json.dumps({"cancel": rid}) + "\n").encode())


async def frontend_main(args) -> int:
    from aiohttp import web
    from tokenizers import Tokenizer

    from mkserve import protocols

    model_dir = Path(args.model_dir)
    tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    template, special = load_chat_template(model_dir)
    eos = eos_ids(model_dir, tok)
    vocab = tok.get_vocab_size(with_added_tokens=True)   # ids a request may carry: the model's embedding rows when its config says how many
    model_cfg = model_dir / "config.json"
    if model_cfg.exists():
        cfg = json.loads(model_cfg.read_text())
        vocab = int(cfg.get("vocab_size") or (cfg.get("text_config") or {}).get("vocab_size") or vocab)
    for _ in range(int(args.wait_s * 10)):  # the engine worker may still be loading
        if os.path.exists(args.sock):
            break
        await asyncio.sleep(0.1)
    reader, writer = await asyncio.open_unix_connection(args.sock, limit=1 << 24)
    hello = json.loads(await reader.readline())["hello"]
    core = Core(args, tok, template, special, eos, vocab, hello, writer, load_response_template(model_dir))

    async def pump() -> None:
        while True:
            line = await reader.readline()
            if not line:
                os._exit(3)  # the engine worker died: a server that answers nothing must not stay up
            for rid, tokens, finished, *extra in json.loads(line):
                q = core.queues.get(rid)
                if q is not None:
                    if extra:
                        core.extra.setdefault(rid, {}).update(extra[0])
                    q.put_nowait((tokens, finished))

    @web.middleware
    async def auth(request, handler):
        # --api-key: the model routes need "Authorization: Bearer KEY" (OpenAI, Ollama) or "x-api-key: KEY" (Anthropic);
        # /health, /metrics and Ollama's version probe stay open
        if args.api_key and (request.path.startswith("/v1/") or request.path.startswith("/api/")) and request.path != "/api/version":
            given = request.headers.get("x-api-key") or request.headers.get("Authorization", "").removeprefix("Bearer ")
            if given != args.api_key:
                return web.json_response({"error": {"message": "invalid or missing API key", "type": "authentication_error"}}, status=401)
        return await handler(request)

    app = web.Application(client_max_size=1 << 26, middlewares=[auth])
    protocols.add_routes(app, core)
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    await web.TCPSite(runner, args.host, args.port).start()
    print(f"MKSERVE serving {args.name} on http://{args.host}:{args.port} (engine {args.engine_name}, {hello['mode']} mode): "
          f"OpenAI /v1/chat/completions /v1/completions /v1/responses, Anthropic /v1/messages, Ollama /api/chat /api/generate", flush=True)
    await pump()
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("role", choices=["engine", "frontend"])
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--sock", required=True)
    parser.add_argument("--workspace", default=None)
    parser.add_argument("--mode", choices=["auto", "native", "step"], default="auto")
    parser.add_argument("--max-batch", type=int, default=8)
    parser.add_argument("--max-context", type=int, default=2048)
    parser.add_argument("--gather-ms", type=float, default=20.0, help="step mode: how long a batch waits for more requests")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--name", default="model", help="the model id the API reports")
    parser.add_argument("--engine-name", default="")
    parser.add_argument("--default-max-tokens", type=int, default=512)
    parser.add_argument("--strict-sampling", action="store_true", help="refuse sampling requests in native mode instead of decoding greedily")
    parser.add_argument("--api-key", default=None, help="front end: require this key as a Bearer token on /v1/*")
    parser.add_argument("--wait-s", type=float, default=1800.0, help="front end: how long to wait for the engine worker's socket")
    args = parser.parse_args(argv)
    if args.role == "engine":
        return engine_main(args)
    return asyncio.run(frontend_main(args))


if __name__ == "__main__":
    raise SystemExit(main())

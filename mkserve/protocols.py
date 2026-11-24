"""The APIs local tools speak, all over one engine (server.Core):

    OpenAI      /v1/models  /v1/completions  /v1/chat/completions  /v1/responses       (Open WebUI, Codex, OpenCode, SDKs)
    Anthropic   /v1/messages  /v1/messages/count_tokens                                 (Claude Code: ANTHROPIC_BASE_URL)
    Ollama      /api/version  /api/tags  /api/show  /api/ps  /api/chat  /api/generate  (apps that only know Ollama)
    mkserve     /health  /metrics

Every route turns its request into a conversation for the model's own chat template (or token ids), runs it, and
writes the model's answer (reasoning, text, tool calls: mkserve/output.py) in its own format, streamed or not.
"""

from __future__ import annotations

import json
import time
import uuid

from aiohttp import web


def _id(prefix: str) -> str:
    return f"{prefix}{uuid.uuid4().hex[:24]}"


def _dumps(obj) -> str:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


def _args_obj(arguments):
    """Tool-call arguments as the template wants them: an object (OpenAI clients send a JSON string)."""
    if isinstance(arguments, str):
        try:
            value = json.loads(arguments) if arguments.strip() else {}
        except json.JSONDecodeError:
            raise ValueError(f"tool call arguments are not JSON: {arguments[:80]!r}")
        return value if isinstance(value, dict) else {"value": value}
    return arguments or {}


def _text_of(content) -> str:
    """Text of an OpenAI / Anthropic / Responses content value (a string or a list of parts)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    for p in content:
        if isinstance(p, str):
            parts.append(p)
        elif isinstance(p, dict) and p.get("type") in (None, "text", "input_text", "output_text"):
            parts.append(p.get("text") or "")
        elif isinstance(p, dict) and p.get("type") in ("image", "image_url", "input_image"):
            raise ValueError("images are not supported by this engine yet")
    return "".join(parts)


def _tools_openai(tools) -> list | None:
    """OpenAI chat tools ({"type": "function", "function": {...}}) or Responses tools ({"type": "function", "name": ...})."""
    out = []
    for t in tools or []:
        if t.get("type", "function") != "function":
            continue  # built-in tools of a hosted API (web search, file search): nothing here runs them
        fn = t.get("function") or t
        out.append({"type": "function", "function": {"name": fn["name"], "description": fn.get("description") or "",
                                                     "parameters": fn.get("parameters") or {}}})
    return out or None


def _effort(value):
    """reasoning effort for the template: OpenAI's names, Ollama's think flag."""
    if value is None:
        return None
    if value is True:
        return "high"
    if value is False:
        return "none"
    return value


def _ns(seconds: float) -> int:
    return int(seconds * 1e9)


def add_routes(app: web.Application, core) -> None:
    def error(message: str, status: int = 400, kind: str = "invalid_request_error"):
        return web.json_response({"error": {"message": message, "type": kind}}, status=status)

    async def body_of(request):
        try:
            body = await request.json()
        except Exception:
            raise ValueError("the request body is not JSON")
        if not isinstance(body, dict):
            raise ValueError("the request body must be a JSON object")
        return body

    async def collect(ids, n, sampling, stops, ignore_eos=False, parse=True):
        """A whole answer: (content, reasoning, tool calls, final dict)."""
        content, reasoning, calls, final = "", "", [], None
        async for step in core.run(ids, n, sampling, stops, ignore_eos, parse):
            if step.get("done"):
                final = step
            else:
                content += step["content"]; reasoning += step["reasoning"]; calls += step["tool_calls"]
        return content, reasoning, calls, final

    def call_name_args(call):
        fn = call.get("function") or {}
        return fn.get("name") or call.get("name") or "", fn.get("arguments") if "function" in call else call.get("arguments", {})

    # ------------------------------------------------------------------------------------------------ mkserve
    async def health(_request):
        h = core.hello
        return web.json_response({"status": "ok", "engine": core.args.engine_name, "mode": h["mode"], "max_batch": h["max_batch"],
                                  "max_context": h["max_context"]})

    async def metrics(_request):
        lines = []
        for key, value in core.stats.items():
            name = "mkserve_" + key
            lines += [f"# TYPE {name} {'gauge' if key == 'requests_running' else 'counter'}", f"{name} {value}"]
        return web.Response(text="\n".join(lines) + "\n", content_type="text/plain")

    async def api_status(_request):
        return web.json_response(core.status())

    # ------------------------------------------------------------------------------------------------ OpenAI
    async def models(_request):
        return web.json_response({"object": "list", "data": [{"id": core.name, "object": "model", "created": 0, "owned_by": "mkserve",
                                                                 "engine": core.args.engine_name, "mode": core.hello["mode"],
                                                                 "max_context": core.hello["max_context"],
                                                                 "sampling": core.hello["mode"] == "step" or bool(core.hello.get("native_sampling"))}]})

    def openai_usage(final: dict) -> dict:
        """OpenAI's usage object; cached_tokens: prompt tokens the engine kept from its prefix cache."""
        return {"prompt_tokens": final["prompt_tokens"], "completion_tokens": final["completion_tokens"],
                "total_tokens": final["prompt_tokens"] + final["completion_tokens"],
                "prompt_tokens_details": {"cached_tokens": final.get("cached_tokens", 0)}}

    def openai_messages(messages) -> list:
        """OpenAI chat messages for the template: text parts joined, tool-call arguments as objects, reasoning kept."""
        out = []
        for m in messages:
            if not isinstance(m, dict) or "role" not in m:
                raise ValueError("every message needs a role")
            role = "system" if m["role"] == "developer" else m["role"]
            msg = {"role": role, "content": _text_of(m.get("content"))}
            for key in ("name", "tool_call_id"):
                if m.get(key):
                    msg[key] = m[key]
            reasoning = m.get("reasoning_content") or m.get("reasoning")
            if reasoning:
                msg["reasoning_content"] = reasoning
            if m.get("tool_calls"):
                msg["tool_calls"] = [{"id": tc.get("id") or _id("call_"), "type": "function",
                                      "function": {"name": tc["function"]["name"], "arguments": _args_obj(tc["function"].get("arguments"))}}
                                     for tc in m["tool_calls"]]
            out.append(msg)
        return out

    def openai_call(call, index=None) -> dict:
        name, args = call_name_args(call)
        out = {"id": _id("call_"), "type": "function", "function": {"name": name, "arguments": _dumps(args or {})}}
        if index is not None:
            out["index"] = index
        return out

    def common(body):
        """(max new tokens, sampling, stops) of an OpenAI-style body, or raises ValueError."""
        n = int(body.get("max_completion_tokens") or body.get("max_tokens") or body.get("max_output_tokens") or core.args.default_max_tokens)
        sampling, refused = core.sampling(body.get("temperature"), body.get("top_p"), body.get("top_k"), body.get("seed"))
        if refused:
            raise ValueError(refused)
        stops = body.get("stop") or []
        return n, sampling, [stops] if isinstance(stops, str) else list(stops)

    async def openai_stream(request, ids, n, sampling, stops, body, chat: bool):
        rid, created = _id("chatcmpl-" if chat else "cmpl-"), int(time.time())
        head = {"id": rid, "object": "chat.completion.chunk" if chat else "text_completion", "created": created, "model": core.name}
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"})
        await resp.prepare(request)

        async def send(choice):
            await resp.write(b"data: " + _dumps(dict(head, choices=[choice])).encode() + b"\n\n")

        first, n_calls, final = True, 0, None
        async for step in core.run(ids, n, sampling, stops, bool(body.get("ignore_eos")), parse=chat):
            if step.get("done"):
                final = step
                break
            if not chat:  # one chunk per token (its text may be empty), so that a client can time tokens
                await send({"index": 0, "text": step["content"], "logprobs": None, "finish_reason": None})
                continue
            delta = {"role": "assistant"} if first else {}
            if step["reasoning"]:
                delta["reasoning_content"] = step["reasoning"]
            if step["content"] or first:
                delta["content"] = step["content"]
            if step["tool_calls"]:
                delta["tool_calls"] = [openai_call(c, n_calls + i) for i, c in enumerate(step["tool_calls"])]
                n_calls += len(step["tool_calls"])
            first = False
            await send({"index": 0, "delta": delta, "logprobs": None, "finish_reason": None})
        finish = "tool_calls" if n_calls and final["finish"] == "stop" else final["finish"]
        await send({"index": 0, "delta": {}, "logprobs": None, "finish_reason": finish} if chat else
                   {"index": 0, "text": "", "logprobs": None, "finish_reason": finish})
        if (body.get("stream_options") or {}).get("include_usage"):
            usage = openai_usage(final)
            await resp.write(b"data: " + _dumps(dict(head, choices=[], usage=usage)).encode() + b"\n\n")
        await resp.write(b"data: [DONE]\n\n")
        return resp

    async def completions(request):
        try:
            body = await body_of(request)
            prompt = body.get("prompt")
            if isinstance(prompt, list) and len(prompt) == 1 and isinstance(prompt[0], (str, list)):
                prompt = prompt[0]
            if isinstance(prompt, str):
                ids = core.tok.encode(prompt, add_special_tokens=False).ids
            elif isinstance(prompt, list) and prompt and all(isinstance(t, int) for t in prompt):
                ids = prompt
            else:
                return error("prompt must be one string or one list of token ids")
            n, sampling, stops = common(body)
        except ValueError as exc:
            return error(str(exc))
        if (why := core.check(ids, n)):
            return error(why)
        if body.get("stream"):
            return await openai_stream(request, ids, n, sampling, stops, body, chat=False)
        content, _r, _c, final = await collect(ids, n, sampling, stops, bool(body.get("ignore_eos")), parse=False)
        return web.json_response({"id": _id("cmpl-"), "object": "text_completion", "created": int(time.time()), "model": core.name,
                                  "choices": [{"index": 0, "text": content, "logprobs": None, "finish_reason": final["finish"]}],
                                  "usage": openai_usage(final)},
                                 dumps=_dumps)

    async def chat_completions(request):
        try:
            body = await body_of(request)
            messages = body.get("messages")
            if not isinstance(messages, list) or not messages:
                return error("messages must be a non-empty list")
            tools = None if body.get("tool_choice") == "none" else _tools_openai(body.get("tools"))
            kwargs = dict(body.get("chat_template_kwargs") or {})
            effort = body.get("reasoning_effort", kwargs.pop("reasoning_effort", None))
            if effort is None and kwargs.get("enable_thinking") is False:
                effort = False  # Qwen's switch, as clients send it: for a template that takes reasoning_effort (Inkling 2), effort "none"
            ids = core.render(openai_messages(messages), tools, reasoning_effort=_effort(effort), **kwargs)
            n, sampling, stops = common(body)
        except ValueError as exc:
            return error(str(exc))
        if (why := core.check(ids, n)):
            return error(why)
        if body.get("stream"):
            return await openai_stream(request, ids, n, sampling, stops, body, chat=True)
        content, reasoning, calls, final = await collect(ids, n, sampling, stops, bool(body.get("ignore_eos")))
        message = {"role": "assistant", "content": content or (None if calls else "")}
        if reasoning:
            message["reasoning_content"] = reasoning
        if calls:
            message["tool_calls"] = [openai_call(c) for c in calls]
        finish = "tool_calls" if calls and final["finish"] == "stop" else final["finish"]
        return web.json_response({"id": _id("chatcmpl-"), "object": "chat.completion", "created": int(time.time()), "model": core.name,
                                  "choices": [{"index": 0, "message": message, "logprobs": None, "finish_reason": finish}],
                                  "usage": openai_usage(final)},
                                 dumps=_dumps)

    # ------------------------------------------------------------------------------------- OpenAI Responses
    stored: dict[str, list] = {}  # previous_response_id: the conversation so far (kept in memory, newest 256)

    def responses_messages(body) -> list:
        msgs = []
        if body.get("previous_response_id"):
            if body["previous_response_id"] not in stored:
                raise ValueError(f"unknown previous_response_id {body['previous_response_id']} (responses are kept in memory only)")
            msgs = [dict(m) for m in stored[body["previous_response_id"]]]
            if msgs and msgs[0]["role"] == "system" and body.get("instructions"):
                msgs = msgs[1:]  # instructions are not carried over: the new request's apply
        if body.get("instructions"):
            msgs.insert(0, {"role": "system", "content": body["instructions"]})
        items = body.get("input")
        if isinstance(items, str):
            items = [{"role": "user", "content": items}]
        call_names: dict[str, str] = {}
        pending_reasoning = None
        for item in items or []:
            kind = item.get("type") or "message"
            if kind == "message":
                role = {"developer": "system"}.get(item.get("role"), item.get("role", "user"))
                msg = {"role": role, "content": _text_of(item.get("content"))}
                if role == "assistant" and pending_reasoning:
                    msg["reasoning_content"], pending_reasoning = pending_reasoning, None
                msgs.append(msg)
            elif kind == "reasoning":
                pending_reasoning = "".join(c.get("text", "") for c in (item.get("content") or [])) or \
                                    "".join(s.get("text", "") for s in (item.get("summary") or []))
            elif kind == "function_call":
                call_names[item["call_id"]] = item["name"]
                call = {"id": item["call_id"], "type": "function", "function": {"name": item["name"], "arguments": _args_obj(item.get("arguments"))}}
                if msgs and msgs[-1]["role"] == "assistant" and not msgs[-1].get("content") and msgs[-1].get("tool_calls") is not None:
                    msgs[-1]["tool_calls"].append(call)
                else:
                    msg = {"role": "assistant", "content": "", "tool_calls": [call]}
                    if pending_reasoning:
                        msg["reasoning_content"], pending_reasoning = pending_reasoning, None
                    msgs.append(msg)
            elif kind == "function_call_output":
                out = item.get("output")
                msgs.append({"role": "tool", "tool_call_id": item["call_id"], "name": call_names.get(item["call_id"], ""),
                             "content": out if isinstance(out, str) else _text_of(out)})
        if not msgs:
            raise ValueError("input is empty")
        return msgs

    async def responses(request):
        try:
            body = await body_of(request)
            msgs = responses_messages(body)
            tools = None if body.get("tool_choice") == "none" else _tools_openai(body.get("tools"))
            ids = core.render(msgs, tools, reasoning_effort=_effort((body.get("reasoning") or {}).get("effort")))
            n, sampling, _stops = common(body)
        except ValueError as exc:
            return error(str(exc))
        if (why := core.check(ids, n)):
            return error(why)
        resp_id, created = _id("resp_"), int(time.time())
        stream = bool(body.get("stream"))
        seq = 0
        sresp = None
        if stream:
            sresp = web.StreamResponse(headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"})
            await sresp.prepare(request)

        async def event(kind, **data):
            nonlocal seq
            if stream:
                await sresp.write(f"event: {kind}\ndata: ".encode() + _dumps({"type": kind, "sequence_number": seq, **data}).encode() + b"\n\n")
            seq += 1

        def response_obj(status, output, usage=None):
            return {"id": resp_id, "object": "response", "created_at": created, "status": status, "model": core.name,
                    "output": output, "usage": usage, "instructions": body.get("instructions"), "tools": body.get("tools") or [],
                    "tool_choice": body.get("tool_choice", "auto"), "parallel_tool_calls": True,
                    "previous_response_id": body.get("previous_response_id"), "reasoning": body.get("reasoning"),
                    "text": {"format": {"type": "text"}}, "error": None, "incomplete_details": None}

        await event("response.created", response=response_obj("in_progress", []))
        await event("response.in_progress", response=response_obj("in_progress", []))
        output: list = []
        open_item = None  # the item text is streaming into: ("reasoning" | "message", index)

        async def close_open():
            nonlocal open_item
            if open_item is None:
                return
            kind, idx = open_item
            item = output[idx]
            if kind == "message":
                text = item["content"][0]["text"]
                await event("response.output_text.done", item_id=item["id"], output_index=idx, content_index=0, text=text, logprobs=[])
                await event("response.content_part.done", item_id=item["id"], output_index=idx, content_index=0, part=item["content"][0])
            else:
                text = item["content"][0]["text"]
                await event("response.reasoning_text.done", item_id=item["id"], output_index=idx, content_index=0, text=text)
                await event("response.content_part.done", item_id=item["id"], output_index=idx, content_index=0, part=item["content"][0])
            item["status"] = "completed"
            await event("response.output_item.done", output_index=idx, item=item)
            open_item = None

        async def text_into(kind, piece):
            nonlocal open_item
            if open_item is None or open_item[0] != kind:
                await close_open()
                if kind == "message":
                    item = {"id": _id("msg_"), "type": "message", "role": "assistant", "status": "in_progress",
                            "content": [{"type": "output_text", "text": "", "annotations": [], "logprobs": []}]}
                else:
                    item = {"id": _id("rs_"), "type": "reasoning", "status": "in_progress", "summary": [],
                            "content": [{"type": "reasoning_text", "text": ""}]}
                output.append(item)
                open_item = (kind, len(output) - 1)
                await event("response.output_item.added", output_index=len(output) - 1, item=item)
                await event("response.content_part.added", item_id=item["id"], output_index=len(output) - 1, content_index=0,
                            part=dict(item["content"][0]))
            idx = open_item[1]
            item = output[idx]
            item["content"][0]["text"] += piece
            await event("response.output_text.delta" if kind == "message" else "response.reasoning_text.delta",
                        item_id=item["id"], output_index=idx, content_index=0, delta=piece, **({"logprobs": []} if kind == "message" else {}))

        final = None
        async for step in core.run(ids, n, sampling, []):
            if step.get("done"):
                final = step
                break
            if step["reasoning"]:
                await text_into("reasoning", step["reasoning"])
            if step["content"]:
                await text_into("message", step["content"])
            for c in step["tool_calls"]:
                await close_open()
                name, args = call_name_args(c)
                item = {"id": _id("fc_"), "type": "function_call", "status": "completed", "call_id": _id("call_"), "name": name,
                        "arguments": _dumps(args or {})}
                output.append(item)
                idx = len(output) - 1
                await event("response.output_item.added", output_index=idx, item=dict(item, status="in_progress", arguments=""))
                await event("response.function_call_arguments.delta", item_id=item["id"], output_index=idx, delta=item["arguments"])
                await event("response.function_call_arguments.done", item_id=item["id"], output_index=idx, arguments=item["arguments"], name=name)
                await event("response.output_item.done", output_index=idx, item=item)
        await close_open()
        usage = {"input_tokens": final["prompt_tokens"], "input_tokens_details": {"cached_tokens": final.get("cached_tokens", 0)},
                 "output_tokens": final["completion_tokens"], "output_tokens_details": {"reasoning_tokens": 0},
                 "total_tokens": final["prompt_tokens"] + final["completion_tokens"]}
        status = "incomplete" if final["finish"] == "length" else "completed"
        done = response_obj(status, output, usage)
        if status == "incomplete":
            done["incomplete_details"] = {"reason": "max_output_tokens"}
        # the conversation, for a later previous_response_id
        reply = {"role": "assistant", "content": "".join(i["content"][0]["text"] for i in output if i["type"] == "message")}
        reasoning = "".join(i["content"][0]["text"] for i in output if i["type"] == "reasoning")
        if reasoning:
            reply["reasoning_content"] = reasoning
        calls = [{"id": i["call_id"], "type": "function", "function": {"name": i["name"], "arguments": json.loads(i["arguments"])}}
                 for i in output if i["type"] == "function_call"]
        if calls:
            reply["tool_calls"] = calls
        if body.get("store", True):
            stored[resp_id] = msgs + [reply]
            while len(stored) > 256:
                stored.pop(next(iter(stored)))
        if not stream:
            return web.json_response(done, dumps=_dumps)
        await event("response.completed" if status == "completed" else "response.incomplete", response=done)
        return sresp

    # -------------------------------------------------------------------------------------------- Anthropic
    def anthropic_messages(body) -> list:
        msgs = []
        system = body.get("system")
        if system:
            msgs.append({"role": "system", "content": system if isinstance(system, str) else
                         "".join(b.get("text", "") for b in system if b.get("type") == "text")})
        names: dict[str, str] = {}
        for m in body.get("messages") or []:
            role, content = m.get("role"), m.get("content")
            if isinstance(content, str):
                msgs.append({"role": role, "content": content})
                continue
            if role == "assistant":
                msg = {"role": "assistant", "content": ""}
                for b in content or []:
                    t = b.get("type")
                    if t == "text":
                        msg["content"] += b.get("text", "")
                    elif t == "thinking":
                        msg["reasoning_content"] = msg.get("reasoning_content", "") + b.get("thinking", "")
                    elif t == "tool_use":
                        names[b["id"]] = b["name"]
                        msg.setdefault("tool_calls", []).append({"id": b["id"], "type": "function",
                                                                 "function": {"name": b["name"], "arguments": b.get("input") or {}}})
                msgs.append(msg)
                continue
            text = ""
            for b in content or []:  # tool results first (they answer the previous turn's calls), then the user's text
                t = b.get("type")
                if t == "tool_result":
                    out = b.get("content")
                    out = out if isinstance(out, str) else _text_of([x for x in (out or []) if x.get("type") == "text"])
                    if b.get("is_error"):
                        out = "Error: " + out
                    msgs.append({"role": "tool", "tool_call_id": b["tool_use_id"], "name": names.get(b["tool_use_id"], ""), "content": out})
                elif t == "text":
                    text += b.get("text", "")
                elif t in ("image", "document"):
                    raise ValueError(f"{t} blocks are not supported by this engine yet")
            if text:
                msgs.append({"role": "user", "content": text})
        return msgs

    def anthropic_tools(body):
        choice = body.get("tool_choice") or {}
        if choice.get("type") == "none":
            return None
        out = [{"type": "function", "function": {"name": t["name"], "description": t.get("description") or "",
                                                 "parameters": t.get("input_schema") or {}}}
               for t in body.get("tools") or [] if t.get("type") in (None, "custom") and t.get("name")]
        return out or None

    def anthropic_prepare(body):
        thinking = body.get("thinking") or {}
        effort = "none" if thinking.get("type") == "disabled" else None
        ids = core.render(anthropic_messages(body), anthropic_tools(body), reasoning_effort=effort)
        return ids

    async def anthropic_count(request):
        try:
            ids = anthropic_prepare(await body_of(request))
        except ValueError as exc:
            return web.json_response({"type": "error", "error": {"type": "invalid_request_error", "message": str(exc)}}, status=400)
        return web.json_response({"input_tokens": len(ids)})

    async def anthropic(request):
        def aerror(message, status=400):
            return web.json_response({"type": "error", "error": {"type": "invalid_request_error", "message": message}}, status=status)
        try:
            body = await body_of(request)
            ids = anthropic_prepare(body)
            n = int(body.get("max_tokens") or core.args.default_max_tokens)
            sampling, refused = core.sampling(body.get("temperature"), body.get("top_p"), body.get("top_k"))
            if refused:
                return aerror(refused)
        except ValueError as exc:
            return aerror(str(exc))
        if (why := core.check(ids, n)):
            return aerror(why)
        stops = body.get("stop_sequences") or []
        show_thinking = (body.get("thinking") or {}).get("type") != "disabled"
        msg_id = _id("msg_")
        message = {"id": msg_id, "type": "message", "role": "assistant", "model": body.get("model") or core.name, "content": [],
                   "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": len(ids), "output_tokens": 0,
                                                                          "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}}
        stream = bool(body.get("stream"))
        sresp = None
        if stream:
            sresp = web.StreamResponse(headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"})
            await sresp.prepare(request)

        async def event(kind, **data):
            if stream:
                await sresp.write(f"event: {kind}\ndata: ".encode() + _dumps({"type": kind, **data}).encode() + b"\n\n")

        await event("message_start", message=message)
        blocks = message["content"]
        open_kind = None

        async def close_block():
            nonlocal open_kind
            if open_kind is None:
                return
            if open_kind == "thinking":
                await event("content_block_delta", index=len(blocks) - 1, delta={"type": "signature_delta", "signature": ""})
            await event("content_block_stop", index=len(blocks) - 1)
            open_kind = None

        async def text_into(kind, piece):
            nonlocal open_kind
            if open_kind != kind:
                await close_block()
                block = {"type": "thinking", "thinking": "", "signature": ""} if kind == "thinking" else {"type": "text", "text": ""}
                blocks.append(block)
                open_kind = kind
                await event("content_block_start", index=len(blocks) - 1, content_block=dict(block))
            blocks[-1]["thinking" if kind == "thinking" else "text"] += piece
            await event("content_block_delta", index=len(blocks) - 1,
                        delta={"type": "thinking_delta", "thinking": piece} if kind == "thinking" else {"type": "text_delta", "text": piece})

        final = None
        n_calls = 0
        async for step in core.run(ids, n, sampling, stops):
            if step.get("done"):
                final = step
                break
            if step["reasoning"] and show_thinking:
                await text_into("thinking", step["reasoning"])
            if step["content"]:
                await text_into("text", step["content"])
            for c in step["tool_calls"]:
                await close_block()
                name, args = call_name_args(c)
                block = {"type": "tool_use", "id": _id("toolu_"), "name": name, "input": args or {}}
                blocks.append(block)
                n_calls += 1
                await event("content_block_start", index=len(blocks) - 1, content_block=dict(block, input={}))
                await event("content_block_delta", index=len(blocks) - 1, delta={"type": "input_json_delta", "partial_json": _dumps(args or {})})
                await event("content_block_stop", index=len(blocks) - 1)
        await close_block()
        if final["finish"] == "length":
            reason = "max_tokens"
        elif final["stop_sequence"]:
            reason = "stop_sequence"
        else:
            reason = "tool_use" if n_calls else "end_turn"
        message.update(stop_reason=reason, stop_sequence=final["stop_sequence"])
        cached = final.get("cached_tokens", 0)   # Anthropic counts tokens read from a cache apart from input_tokens
        message["usage"].update(output_tokens=final["completion_tokens"], input_tokens=final["prompt_tokens"] - cached,
                                cache_read_input_tokens=cached)
        if not stream:
            return web.json_response(message, dumps=_dumps)
        await event("message_delta", delta={"stop_reason": reason, "stop_sequence": final["stop_sequence"]},
                    usage={"output_tokens": final["completion_tokens"], "input_tokens": final["prompt_tokens"] - cached,
                           "cache_read_input_tokens": cached})
        await event("message_stop")
        return sresp

    # ----------------------------------------------------------------------------------------------- Ollama
    started = time.strftime("%Y-%m-%dT%H:%M:%S.000000Z", time.gmtime())

    def ollama_model():
        return {"name": core.name, "model": core.name, "modified_at": started, "size": 0, "digest": "",
                "details": {"parent_model": "", "format": "megakernel", "family": "", "families": None,
                            "parameter_size": "", "quantization_level": "as stored"}}

    async def o_version(_request):
        return web.json_response({"version": "0.9.0"})  # clients check for a version they know; mkserve speaks that API

    async def o_tags(_request):
        return web.json_response({"models": [ollama_model()]})

    async def o_ps(_request):
        return web.json_response({"models": [dict(ollama_model(), expires_at="2099-01-01T00:00:00Z", size_vram=0)]})

    async def o_show(_request):
        caps = ["completion"] + (["tools", "thinking"] if core.rt else [])
        return web.json_response({"modelfile": "", "parameters": "", "template": "", "details": ollama_model()["details"],
                                  "model_info": {"general.architecture": "megakernel", "megakernel.context_length": core.hello["max_context"]},
                                  "capabilities": caps})

    def ollama_options(body):
        o = body.get("options") or {}
        n = int(o.get("num_predict") or core.args.default_max_tokens)
        if n < 0:
            n = core.hello["max_context"]
        sampling, refused = core.sampling(o.get("temperature", 0.0 if core.native else 0.8), o.get("top_p"), o.get("top_k"), o.get("seed"))
        if refused:
            raise ValueError(refused)
        stops = o.get("stop") or []
        return n, sampling, [stops] if isinstance(stops, str) else list(stops)

    def ollama_messages(messages) -> list:
        out = []
        for m in messages or []:
            if m.get("images"):
                raise ValueError("images are not supported by this engine yet")
            msg = {"role": m["role"], "content": m.get("content") or ""}
            if m.get("thinking"):
                msg["reasoning_content"] = m["thinking"]
            if m.get("tool_calls"):
                msg["tool_calls"] = [{"id": _id("call_"), "type": "function",
                                      "function": {"name": tc["function"]["name"], "arguments": _args_obj(tc["function"].get("arguments"))}}
                                     for tc in m["tool_calls"]]
            if m.get("tool_name"):
                msg["name"] = m["tool_name"]
            out.append(msg)
        return out

    async def ollama_run(request, body, ids, n, sampling, stops, chat: bool, show_thinking: bool):
        stream = body.get("stream", True)  # Ollama streams unless told not to
        t0 = time.perf_counter()
        sresp = None
        if stream:
            sresp = web.StreamResponse(headers={"Content-Type": "application/x-ndjson"})
            await sresp.prepare(request)
        content, thinking, calls, final, t_first = "", "", [], None, None

        def line(text, think, tool_calls, done=False):
            obj = {"model": core.name, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S.000000Z", time.gmtime())}
            if chat:
                m = {"role": "assistant", "content": text}
                if think:
                    m["thinking"] = think
                if tool_calls:
                    m["tool_calls"] = [{"function": {"name": call_name_args(c)[0], "arguments": call_name_args(c)[1] or {}}} for c in tool_calls]
                obj["message"] = m
            else:
                obj["response"] = text
                if think:
                    obj["thinking"] = think
            obj["done"] = done
            return obj

        async for step in core.run(ids, n, sampling, stops, parse=chat or not body.get("raw")):
            if step.get("done"):
                final = step
                break
            t_first = t_first or time.perf_counter()
            think = step["reasoning"] if show_thinking else ""
            content += step["content"]; thinking += think; calls += step["tool_calls"]
            if stream and (step["content"] or think or step["tool_calls"]):
                await sresp.write((_dumps(line(step["content"], think, step["tool_calls"])) + "\n").encode())
        t_end = time.perf_counter()
        last = line("" if stream else content, "" if stream else thinking, [] if stream else calls, done=True)
        last.update(done_reason="length" if final["finish"] == "length" else "stop", total_duration=_ns(t_end - t0), load_duration=0,
                    prompt_eval_count=final["prompt_tokens"], prompt_eval_duration=_ns((t_first or t_end) - t0),
                    eval_count=final["completion_tokens"], eval_duration=_ns(t_end - (t_first or t_end)))
        if not stream:
            return web.json_response(last, dumps=_dumps)
        await sresp.write((_dumps(last) + "\n").encode())
        return sresp

    async def o_chat(request):
        try:
            body = await body_of(request)
            think = body.get("think")
            ids = core.render(ollama_messages(body.get("messages")), _tools_openai(body.get("tools")), reasoning_effort=_effort(think))
            n, sampling, stops = ollama_options(body)
        except ValueError as exc:
            return web.json_response({"error": str(exc)}, status=400)
        if (why := core.check(ids, n)):
            return web.json_response({"error": why}, status=400)
        return await ollama_run(request, body, ids, n, sampling, stops, chat=True, show_thinking=think is not False)

    async def o_generate(request):
        try:
            body = await body_of(request)
            if not body.get("prompt"):  # Ollama's "load the model" probe: an empty prompt answers at once
                return web.json_response({"model": core.name, "created_at": started, "response": "", "done": True, "done_reason": "load"})
            if body.get("images"):
                raise ValueError("images are not supported by this engine yet")
            if body.get("raw"):
                ids = core.tok.encode(body["prompt"], add_special_tokens=False).ids
            else:
                msgs = ([{"role": "system", "content": body["system"]}] if body.get("system") else []) + [{"role": "user", "content": body["prompt"]}]
                ids = core.render(msgs, reasoning_effort=_effort(body.get("think")))
            n, sampling, stops = ollama_options(body)
        except ValueError as exc:
            return web.json_response({"error": str(exc)}, status=400)
        if (why := core.check(ids, n)):
            return web.json_response({"error": why}, status=400)
        return await ollama_run(request, body, ids, n, sampling, stops, chat=False, show_thinking=body.get("think") is not False)

    async def root(_request):  # what Ollama answers at its root; some apps probe for it
        return web.Response(text="Ollama is running")

    async def not_here(_request):
        return web.json_response({"error": "not supported by mkserve (one model, served as is)"}, status=501)

    app.add_routes([
        web.get("/health", health), web.get("/metrics", metrics), web.get("/api/status", api_status),
        web.get("/v1/models", models), web.post("/v1/completions", completions), web.post("/v1/chat/completions", chat_completions),
        web.post("/v1/responses", responses),
        web.post("/v1/messages", anthropic), web.post("/v1/messages/count_tokens", anthropic_count),
        web.get("/api/version", o_version), web.get("/api/tags", o_tags), web.get("/api/ps", o_ps), web.post("/api/show", o_show),
        web.post("/api/chat", o_chat), web.post("/api/generate", o_generate),
        web.post("/api/embed", not_here), web.post("/api/embeddings", not_here), web.post("/api/pull", not_here),
        web.post("/api/create", not_here), web.delete("/api/delete", not_here),
        web.get("/", root),
    ])

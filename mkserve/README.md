# mkserve: a megakernel engine behind the OpenAI API

    pip install -e .                                   # from the repository root; `python -m mkserve` works without installing
    mkserve doctor Qwen/Qwen3-8B                       # is this machine ready? card, CUDA toolkit, packages, weights
    mkserve list                                       # the models and cards that have engines
    mkserve serve Qwen/Qwen3-8B                        # the default engine of this model on this card, on port 8000
    mkserve chat                                       # talk to it in the terminal
    mkserve bench                                      # time it; --url points the same benchmark at vLLM or SGLang

| command | what it does |
|---|---|
| `doctor [MODEL]` | checks the card, `nvcc`, the Python packages, which engines exist for this card and whether the weights are downloaded; prints the fix for every line that fails |
| `list` | every (model, card, engine) with its track and kind; `*` marks the default |
| `pull MODEL` | builds the engine and downloads the weights now, so that `serve` starts at once |
| `serve MODEL` | `--engine NAME` or `--track A\|B\|C`, `--port`, `--host`, `--model-dir` (a local snapshot), `--api-key KEY` (Bearer token on `/v1/*`), `--served-model-name`, `--mode auto\|native\|step`, `--max-context N` (up to the engine's limit), `--prefill fast\|exact` (engines with two prompt paths) |
| `chat` | a streaming terminal chat with any running OpenAI-compatible server (`--url`, `--temperature`); prints tokens per second after each answer |
| `bench` | the project's workload against any OpenAI-compatible server: random 64-token prompts, 1,024 greedy tokens, 1 and 8 requests at once (`--batch`, `--prompt-tokens`, `--max-tokens`, `--rounds`); prints milliseconds per step, tokens per second and time to first token |

    curl http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
         -d '{"model": "Qwen/Qwen3-8B", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 64}'

Needs one supported card, `nvcc` on PATH, and Python with torch, numpy, aiohttp, tokenizers, jinja2 (huggingface_hub
to download weights; `--model-dir` serves a local snapshot). The engine is unpacked, checked against its registered
sha256 and built under `~/.cache/mkserve` (`--workdir`). There is no generic fallback: a model and card that are not
in `registry.json` are refused. A registry entry's `env` sets serving options in the engine's environment; a variable
already set in yours wins. For example, Inkling-2-Mini on an M5 Max is served with `MK_PREFILL=fast`, and
`MK_PREFILL=exact mkserve serve ...` gives replies equal to the engine's step-by-step greedy tokens.

## The APIs local tools speak

One server, one port, every route over the same engine (`mkserve/protocols.py`):

| API | routes | for |
|---|---|---|
| OpenAI | `/v1/models`, `/v1/completions`, `/v1/chat/completions`, `/v1/responses` | Open WebUI, OpenCode, Codex (Responses), the OpenAI SDKs |
| Anthropic | `/v1/messages`, `/v1/messages/count_tokens` | Claude Code (`ANTHROPIC_BASE_URL`), the Anthropic SDKs |
| Ollama | `/api/version`, `/api/tags`, `/api/show`, `/api/ps`, `/api/chat`, `/api/generate` | apps that only know Ollama (point them at this port, or serve on 11434) |
| mkserve | `/health`, `/metrics` (Prometheus text: requests, tokens, time to first token, time per output token) | monitoring |

Streaming or not on every route; `max_tokens` (`max_completion_tokens`, `max_output_tokens`, `num_predict`), stop
strings, `seed`, `temperature`/`top_p`/`top_k` (step mode), usage. The chat template is the model's own, rendered
byte for byte as transformers renders it. A checkpoint that describes its answer format (`response_template` in
`tokenizer_config.json`, as Inkling 2 does) is answered in parts (`mkserve/output.py`): reasoning
(`reasoning_content`; Anthropic `thinking` blocks; Responses `reasoning` items; Ollama `thinking`), text, and tool
calls (`tool_calls`; `tool_use` blocks; `function_call` items), with the finish reason each API uses for a tool call.
Reasoning effort: `reasoning_effort`, `reasoning.effort`, Anthropic `thinking: {"type": "disabled"}`, Ollama `think`.
Chat completions also take `thinking_budget` (reasoning tokens before the reasoning is closed and the model answers); the
answer then carries `mk_reasoning_ids` and `mk_reasoning_cut`, and the same request sent again with the reasoning as a
last assistant message and `"mk_continue_reasoning": true` goes on reasoning from those tokens for `thinking_budget`
more (`mkserve/continuation.py`).
`/v1/responses` keeps conversations in memory for `previous_response_id` (the newest 256). Images are refused (no
engine has the vision tower yet).

    mkserve launch claude     # Claude Code against the running server: every model tier mapped to the served model
    mkserve launch codex      # Codex with a Responses provider pointing here
    mkserve launch opencode   # OpenCode with an OpenAI-compatible provider (its own config file, OPENCODE_CONFIG)

`--print` prints the command instead; arguments after `--` go to the agent.

## On a Mac

`mkserve serve` detects the chip (`Apple M5 Max` is registry key `m5max`) and serves the engines written for it (MLX
and Metal, no nvcc). Run it with the Python that has MLX; `mkserve doctor` checks the chip, the memory and MLX. Another
Apple chip is served the registered chip's engine with a note that it is unverified there. Install, quick start,
numbers and limits for the Mac are in [`mac/README.md`](../mac/README.md).

What a Mac engine may offer beyond `submit` / `poll`, and what `mkserve` does with it:

| the engine has | `mkserve` |
|---|---|
| `native_sampling` | passes `temperature`, `top_p`, `top_k` and `seed` to `submit`: sampling in native mode, through the engine's own draft-and-verify |
| `request_info[rid]["cached_tokens"]` (a prefix cache) | reports the kept prompt tokens: `usage.prompt_tokens_details.cached_tokens`, Anthropic `cache_read_input_tokens`, `/metrics` |
| `cancel(rid)` | calls it when a client disconnects, so the one sequence is free at once |
| `prefill(tokens)` | step mode gives it the prompt in one call instead of one `step()` per token |

Step mode samples on the host, one request at a time.

## Two ways an engine is driven

| | native (engines with `submit`/`poll`) | step (every engine) |
|---|---|---|
| batching | the engine's own: requests of any length are taken while others decode | prompts of equal length that arrive together; nothing is admitted while a batch decodes |
| prompt | prefilled in chunks inside the engine | one token per step |
| sampling | the engine's own if it has `native_sampling` (Inkling-2-Mini on a Mac); otherwise greedy only (a sampling request is decoded greedily with a warning; `--strict-sampling` refuses it) | `temperature`, `top_p`, `seed` |
| a request that ends early | the answer ends; the engine's row stops if the engine has `cancel(rid)`, otherwise it decodes on to `max_tokens` | the row is dropped |

A track-C engine (exact speculative decoding: it drafts several tokens and verifies them in one pass) drafts only in
its own `submit`/`poll`, so `--mode auto` serves it in native mode: greedy, or sampled where the engine has
`native_sampling`. `--mode step` serves the same engine one token per step, with host sampling and without speculation.

## Request validation

The engines do not check their inputs, so `mkserve` does before a request reaches one: an empty prompt, `max_tokens` below 1,
a token id outside the model's vocabulary, and a prompt plus `max_tokens` beyond the engine's context are answered with
HTTP 400. A request never reaches an engine it could crash or corrupt.

## Tests

`bash mkserve/tests/run_protocol_tests.sh` checks every API through the official OpenAI, Anthropic and Ollama SDKs
(in `.venv-clients`), and the rendered prompts against transformers, on a fake engine that replays scripted answers in
Inkling 2's format (CPU only).

`scripts/mkserve_check.py --port PORT --engine NAME --model MODEL` runs twelve checks against a served engine: the model
list, a completion, a streamed completion equal to the unstreamed text, a chat and a streamed chat, a stop string, four
chats of different lengths at once, sampling with a seed, decode speed, a long prompt, a request beyond the context, and
invalid inputs. `scripts/mkserve_test.sh` serves every engine of a model in turn and runs them. All 13 registered engines
pass all twelve on their card; `registry.json` records the date for each.

| Qwen3-8B on one H100 | `online-b` (native mode) | `offline-b` (step mode) |
|---|---|---|
| start (build and load) | 41 s | 20 s |
| first token, 64-token prompt | 0.007 s | 0.28 s |
| first token, 1,429-token prompt | 0.05 s | 5.9 s |
| decode, 1 sequence, 512 tokens | 5.54 ms per token | 4.13 ms per token |
| sampling with a seed | greedy (as documented) | reproducible per seed |

These figures are from one run of the test and illustrate the two modes; the benchmark numbers are in the top-level README.

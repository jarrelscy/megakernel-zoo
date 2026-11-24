# megakernel-zoo on Apple silicon

**Inkling-2-Mini at 123 to 194 tokens per second on an M5 Max, on the weights the checkpoint ships with, and with
output that is exactly what decoding one token at a time gives.**

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/speed-dark.svg">
    <img src="assets/speed-light.svg" width="860" alt="Tokens per second of Inkling-2-Mini on an Apple M5 Max. megakernel: 123 on synthetic prompts, 194 on natural text. oMLX with DSpark: 73 and 120. oMLX: 60 and 60. llama.cpp with DSpark: 49 and 61. llama.cpp: 40 and 40. mlx-lm: 33 and 33. Ollama: 32 and 33.">
  </picture>
</p>

[Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) is a 26 B mixture-of-experts model built for
devices. It ships with NVFP4 experts and a small draft head. This folder is about one engine for it: written for one
checkpoint on one chip, in MLX and Metal kernels compiled at run time, by a coding agent (Claude) in two six-hour runs
against a fixed evaluator. It is the laptop version of [A Megakernel per Model](https://dangxingyu.github.io/megakernel/).

- **1.70× and 1.62× the fastest engine we found for this model on a Mac** (oMLX with DSpark), 3.7× mlx-lm and 3.1×
  llama.cpp. One sequence, greedy, over HTTP.
- **No requantization.** It reads the checkpoint's own NVFP4, MXFP8 and BF16 values. The faster of the engines it is
  compared with run a 4-bit copy.
- **Exact speculative decoding.** It drafts several tokens and verifies them in one pass, and what it returns is, token
  for token, what its own one-token step would return. That holds for sampling too, with a seed.
- **Made for agents and chat.** A 32,768-token context, a prefix cache (the next turn of a chat starts after 0.2 s),
  and the OpenAI, Anthropic and Ollama APIs on one port, with tool calls.

Everything here was measured on one M5 Max (40-core GPU, 48 GB). The full write-up, in English and 中文, is
[`runs-mac/report-20261123/inkling-2-mini-m5max.html`](../runs-mac/report-20261123/inkling-2-mini-m5max.html).

## Try it

You need an Apple-silicon Mac with about 36 GB of memory or more, and [`uv`](https://docs.astral.sh/uv/) (or any
Python 3.10 or newer). No Xcode, no CUDA.

```bash
git clone https://github.com/dangxingyu/megakernel-zoo.git && cd megakernel-zoo
uv venv .venv-mac --python 3.12 && source .venv-mac/bin/activate
uv pip install -e .                                 # on a Mac this installs MLX, not PyTorch
mkserve doctor thinkingmachines/Inkling-2-Mini      # chip, memory, MLX, the engine for this Mac
mkserve serve thinkingmachines/Inkling-2-Mini       # downloads the weights (19 GB) once; --model-dir serves a local copy
```

The server is up about 20 seconds after the weights are on disk. Then, in another terminal:

```bash
mkserve chat                 # talk to it; prints tokens per second after each answer
mkserve bench --batch 1      # time it on your machine
```

`mkserve bench` sends prompts of random token ids, which the model continues repetitively, so it prints a higher
number than the chart: about 250 tokens per second here. The chart's workloads are described under
[The numbers](#the-numbers).

## Use it from your tools

It is one server on one port (8000 by default) that speaks three APIs, so most local tools work by changing a URL.

| Tool | How |
|---|---|
| OpenAI SDKs, Open WebUI, anything "OpenAI-compatible" | base URL `http://127.0.0.1:8000/v1`, any API key |
| Claude Code | `mkserve launch claude` |
| Codex | `mkserve launch codex` |
| OpenCode | `mkserve launch opencode` |
| Apps that only speak Ollama | point them at this port, or serve on Ollama's: `mkserve serve ... --port 11434` |

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="none")
stream = client.chat.completions.create(
    model="thinkingmachines/Inkling-2-Mini",
    messages=[{"role": "user", "content": "Explain how a hash table works."}],
    temperature=0.7, seed=7,          # sampled, and the same seed gives the same reply
    stream=True,
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

Reasoning, text and tool calls come back in the fields each API uses (`reasoning_content` and `tool_calls`, Anthropic
`thinking` and `tool_use` blocks, Ollama `thinking`). `reasoning_effort` sets how much the model thinks before it
answers; it thinks a little even at `none`, so leave room in `max_tokens`.

A coding agent sends 10,000 to 20,000 tokens of system prompt and tool definitions before your first message. Give the
engine the larger context for that:

```bash
mkserve serve thinkingmachines/Inkling-2-Mini --max-context 32768
```

All of `mkserve`'s options are in [`mkserve/README.md`](../mkserve/README.md).

## Desktop widget

`mac/widget` is a small window for the desktop that reads `GET /api/status` from `mkserve serve` (or oMLX) every 2
seconds: the model and its loop passes, whether the draft is on, requests, live decode and prefill speed graphs, cache
reuse, memory and GPU. Build it with the Command Line Tools and open it:

```bash
mac/widget/build.sh                      # builds ~/Applications/Inkling Widget.app
open ~/Applications/Inkling\ Widget.app
```

Drag it to move it; right-click for "Keep on top" and "Quit". For another server,
`defaults write ai.omlx.inkling-widget url http://<host>:8000/api/status`.

## Why it is fast

A general engine has to run every model. This one runs a single checkpoint, so each of the following is fixed for
its shapes.

1. **It uses the draft head the model ships with.** Inkling-2-Mini comes with a small head (`mtp.safetensors`, called
   DSpark) that guesses the next seven tokens. The engine also proposes copies when the recent text repeats something
   earlier. One pass of the full model then checks all the guesses at once.
2. **The check is exact.** Every row of that pass has the bits a one-token step would produce, because each sum keeps
   an order that does not depend on how many rows the pass has. A wrong guess costs a little time and nothing else.
   This is why the output never changes with speculation.
3. **Weights are read as stored.** The experts stay in the checkpoint's NVFP4 and MXFP8 codes. The dense BF16 matrices
   are kept in a lossless 12-bit code (six bytes per four weights) that decodes to the identical bits, which cuts the
   memory the pass has to read.
4. **Each expert is read once per pass.** A pass of several rows picks many (row, expert) pairs but far fewer
   distinct experts. The expert kernels group the pairs so an expert's weights are read once.
5. **One round trip to the CPU per cycle.** The next draft is computed on the GPU right behind the verify pass, and how
   many guesses to verify is chosen by a cost model from what recent guesses were worth.

A verify pass runs near the memory bandwidth of the machine (450 to 530 GB/s). What limits the engine now is how often
the draft head is right: 98% of the time for its first token on code and word problems, 60 to 70% on prose.

## The numbers

### Decode speed

One sequence, greedy, over HTTP, a 64-token prompt and 1,024 new tokens (the repository's workload). "Synthetic" is
the evaluator's generated prompts; "natural" is prose, Python source and word problems. Milliseconds per token, tokens
per second in brackets.

| System | Weights | Synthetic prompts | Natural text |
|---|---|---|---|
| **megakernel, track C** | the checkpoint's NVFP4 / MXFP8 / BF16 | **8.10 (123)** | **5.14 (194)** |
| oMLX v0.8.3 + DSpark | oQ4e requantization | 13.77 (73) | 8.31 (120) |
| oMLX v0.8.3 | oQ4e requantization | 16.74 (60) | 16.78 (60) |
| llama.cpp + DSpark (PR 32218) | Q4_K_M GGUF | 20.56 (49) | 16.32 (61) |
| llama.cpp | Q4_K_M GGUF | 25.13 (40) | 25.03 (40) |
| mlx-lm | the checkpoint's formats | 30.36 (33) | 30.49 (33) |
| Ollama 0.40.17 (MLX) | the checkpoint's formats | 31.29 (32) | 30.75 (33) |

- **Ours** is the median of six evaluations of the engine as the second run left it (8,040 to 8,333 and 5,124 to 5,300
  microseconds). The source in this repository, with everything below added, was evaluated once more in full: valid,
  the same gate values, 8,205 and 5,173 ([`runs-mac/final-eval-20261123`](../runs-mac/final-eval-20261123)).
- **The general engines** are two server starts × three prompts each; Ollama is one start. Ollama could not load the
  GGUF or the DSpark draft, and LM Studio was not measured. Setup and scripts: [`baselines/mac`](../baselines/mac).
- **Speed depends on the text.** Speculation pays when the draft is right. Expect about 85 tokens per second on hard
  prose, 100 to 190 on code and structured text, and up to 320 on text that repeats itself. With no speculation at
  all the engine decodes at about 70.
- **Laptop timing is noisy.** The same source spreads by up to 10% with the machine's state. Treat differences under
  5% as noise.

### Time to the first token

| Request | First token |
|---|---|
| 64-token prompt | 0.08 s |
| 1,630-token prompt | 1.2 s |
| Next turn of a chat with a 1,452-token system prompt (prefix cache) | 0.2 s |
| An agent's first request: 19,683 tokens of system prompt and tool definitions | 21.7 s |
| That agent's next turns (about 19,500 tokens read from the cache) | 0.8 to 1.1 s |

Prompts run at about 1,300 tokens per second (900 for the 19,683-token one). The prefix cache keeps the K/V of
whatever a request shares with the previous one, and the kept tokens are reported in `usage` (`cached_tokens`,
`cache_read_input_tokens`).

### Sampling

`temperature`, `top_p`, `top_k` and `seed` go through the same draft-and-verify. Tokens per second over the chat API:

| Request | Greedy | T 0.7 | T 1.0, `top_p` 0.95 | One token per pass |
|---|---|---|---|---|
| A short story | 76 | 80 | 68 | about 68 |
| A technical explanation with code | 102 | 113 | 86 | about 68 |

A sampled token is `argmax(logits / T + noise)`, where the noise is a hash of (seed, position, token). That is a draw
from the usual distribution, and it makes each token a function of its own row, so a verify row draws the token a
one-token step draws. On four prompts and four settings, 256 tokens each, the request path returned exactly the
tokens of step-by-step sampling with the same seed, 16 of 16.

### Context

| `--max-context` | Decode, short prompt | Memory in use |
|---|---|---|
| 4,096 | 12.3 ms per token | 22.1 GB |
| 16,384 (default) | 11.9 | 24.6 GB |
| 32,768 | 12.9 | 27.7 GB |

MLX keeps about 6 GB more in its buffer cache. A decode step takes 13.6 ms at position 0 and 19.0 ms at position
30,000.

## Questions

**Will it run on my Mac?** It was written for and verified on an M5 Max with 48 GB. On another Apple chip `mkserve`
serves the same engine and says so: the kernels are not specific to one chip, but nothing else has been tested, and
the speed will be that chip's own. Memory is the hard limit: about 22 GB for the model and its caches, more with a
larger context. `mkserve doctor` tells you. If you try it on another Mac, an issue with the output of
`mkserve doctor` and `mkserve bench --batch 1` would be very welcome.

**Is this a quantized model?** Not by us. The checkpoint itself stores its experts in NVFP4 and MXFP8, and the engine
reads those codes as they are. Nothing is converted to GGUF or to another 4-bit format.

**Is the output the same as mlx-lm's?** It is the same model, not the same bits. Every BF16 implementation of this
model, ours and mlx-lm included, sits about 0.025 (relative error of the logits) from a float32 evaluation, mostly
because the model picks 8 of 128 experts per token and the 8th and 9th are often nearly tied. See
[`runs-mac/deviation`](../runs-mac/deviation/README.md).

**What does "exact" buy me?** Speculation here changes the speed and never the text: turn it off and you get the same
tokens, slower. The evaluator checks that on every sequence before it times anything.

**Why does `--prefill` have two settings?** `fast` (the default when serving) runs the prompt through MLX matrix
products, 3.4× quicker than `exact` on a 1,630-token prompt. The prompt's K/V then differs in the last bits from what
the step's own kernels would write, so a reply can drift from the step-by-step one after some tokens. `exact` keeps
every reply bit-identical. Decoding after the prompt is exact either way, and `fast` is no further from float32.

**More than one request at a time?** No. This engine decodes one sequence; requests queue, and a client that
disconnects frees it at once.

**Other models?** One so far. An engine is written for one checkpoint, by an agent run of a few hours. The tooling
to start one is in this repository; see [How it was built](#how-it-was-built-and-building-another).

**Images?** Not yet. Inkling 2 also takes images; this engine runs the text model only and refuses image inputs.

## How it was checked

- **Against a reference.** Before it is timed, an engine must match a float32 evaluation of the model on the CPU
  within twice the error of the repository's BF16 path, at every position up to 128 and every 8th up to 1,152, with
  every top-1 token equal. The engine: median relative error of the logits 0.024 (limit 0.049), p99 0.101 (limit
  0.121), top-1 agreement 1.0.
- **Exactness.** The evaluator compares the tokens of the request path with the engine's own step-by-step tokens on
  every sequence. The same comparison with sampling passes 16 of 16.
- **Long positions.** On 9,600 and 24,000-token texts the top-1 token equals float32's at all 58 checked positions.
  At 24,000 tokens the error is within twice the BF16 path's.
- **Serving.** All twelve checks of `scripts/mkserve_check.py`, in both of `mkserve`'s modes.

To run the evaluator yourself (the reference is plain PyTorch on the CPU; the weights are expected in
`~/models/Inkling-2-Mini`, or set `MK_MODEL_DIR`):

```bash
uv pip install torch
python scripts/mac_make_reference.py --out ~/mk-ref/inkling-2-mini-m5max-fp32      # the float32 reference, about 90 s
MK_REF_DIR=~/mk-ref/inkling-2-mini-m5max-fp32 bash scripts/eval_setup_mac.sh
python eval/kernel_eval.py --workspace engines/inkling-2-mini_m5max/offline-c --out result.json
```

## Limits

- **One machine.** Verified on one M5 Max with 48 GB. Other chips and memory sizes are untested.
- **One sequence.** Requests wait for each other.
- **The reference** is float32 on one sequence, and the gates stop at position 1,152. The GPU engines of this
  repository are checked against FP64 on more sequences. Long positions were checked on two texts only.
- **No independent audit yet.** The GPU engines were audited after their runs; this one has not been.
- **Sampling** is exact against the engine's own sampler, not against another implementation's random numbers. On
  hard prose at high temperature it is no faster than one token per pass.
- **A first long prompt is slow.** 20,000 tokens take about 20 s before the cache has them.
- **The baseline table is provisional**: Ollama ran one server start and LM Studio was not measured.

## How it was built, and building another

The engine came out of two autonomous runs of a coding agent, six hours each, against the evaluator above. Every
submission, with its hypothesis, its score and its source, is in the run records:
[`runs-mac/fable-c-m5max-1`](../runs-mac/fable-c-m5max-1/progress.md) (21 submissions, from 26.9 ms to 9.0 ms per
token) and [`runs-mac/fable-c-m5max-2`](../runs-mac/fable-c-m5max-2/progress.md) (19 submissions, to 8.1 ms). The
agent's own journals are in each run's `notes/`, including what it measured and dropped, and the prompt it was given
is each run's `PROMPT.md`.

Prefill, the prefix cache, the long context and sampling were added afterwards; their measurements and test scripts
are in [`runs-mac/prefill-20261123`](../runs-mac/prefill-20261123/README.md).

The same kind of run works for another model or chip:

```bash
bash autoresearch/launch_mac.sh runs-mac/my-run 6      # a six-hour track-C run from starter_mac/
```

See [Build your own engine](../README.md#build-your-own-engine) and [CONTRIBUTING.md](../CONTRIBUTING.md).

## Where things are

| | |
|---|---|
| The engine | [`engines/inkling-2-mini_m5max/offline-c`](../engines/inkling-2-mini_m5max/offline-c) |
| Its registry entry | [`mkserve/registry.json`](../mkserve/registry.json), key `m5max` |
| The server | [`mkserve`](../mkserve/README.md) |
| Run records, measurements, the write-up | [`runs-mac`](../runs-mac/README.md) |
| Baselines | [`baselines/mac`](../baselines/mac), `eval/baselines_inkling-2-mini_m5max_c.json` |
| The reference and the evaluator on a Mac | `scripts/mac_make_reference.py`, `scripts/eval_setup_mac.sh` |
| This page's chart | [`assets/make_chart.py`](assets/make_chart.py) |

You are starting a long autonomous run: 8 hours, this Mac (m5max: Apple M5 Max, 40-core GPU, 128 GB of unified
memory, Metal 4), measured by the run's evaluator.

## Goal

Make `thinkingmachines/TML-d24-varloop-5pass` decode one sequence as fast as you can on this Mac, with exact speculative decoding. The model is the
variable-loop Inkling 2 Mini (`thinkingmachines/TML-d24-varloop-research`) run at **5 loop passes**
(`text_config.loop_config.num_passes = 5`: layers 6 to 17 run five times a token, 72 layer visits). Keep 5 passes.

The reference lines: omlx-0.8.1-fork-mtp: 23256 us per token (natural text at batch 1: 22422 us per token). You do not start from scratch: `engine/` is this repository's M5 Max track-C engine for
Inkling 2 Mini (`engines/inkling-2-mini_m5max/offline-c`, read `mac/README.md` and `runs-mac/` for how it was made and
measured), already changed to run at any pass count, with a 131,072-token context, and drafting with the 2DSD DSpark
drafter (`engine/mk_draft2.py`; weights in `~/Models/Inkling/2dsd-varloop-dspark`, paper and code at
https://github.com/SqueezeAILab/2DSD). It starts at about 70 tok/s on the synthetic prompts, 116 on natural text and
87 on GSM8K chat. The checkpoint's own DSpark head (`mk_draft.py`, `MK_DRAFT2=0`) is the alternative.

The evaluator serves the engine over HTTP: a 64-token prompt, 1,024 greedy new tokens, one sequence; the rate is counted
in tokens. It measures the synthetic prompts (`http_step_us_b1`) and natural text (`natural_b1`). The person this run is
for uses the model as a chat assistant that reasons before it answers (math, code, short tool calls); on that kind of
text the 2DSD drafter is accepted about 20% more often than the checkpoint's head (3.9 against 3.3 tokens per cycle on
GSM8K chat). Do not trade chat speed away for the synthetic headline: check GSM8K chat yourself
(`runs-mac/prompts/chat.json` through the chat template) when you change the draft or its policy.

Ideas worth measuring, yours to rank:
- **2DSD's second dimension.** The drafter was trained to draft from early-loop readouts too: features of in-flight
  tokens after loop r (r = 1, 2) of a pass that exits early through the suffix layers, verified by the full 5-pass
  target. The engine uses only the 1-D form (verified last-loop features). The paper reports the 2-D form as the larger
  gain. The drafter was trained at 3 passes, with depths r in {1, 2, 3}; at 5 passes its loop-R features are the
  last loop's (taps [2, 55, 60, 65, 70]).
- The drafter's cost: five layers and a Markov bias (now over each row's top 64 base logits), quantized to 8 bits.
- The depth policy (`Cal`, `cala`): with a draft accepted more often, more rows per verify may pay.
- The verify pass at 72 visits: where the time goes per visit.

Load the `megakernel` skill now: it has the contract on this device, how the evaluator scores and what its gates check,
how to log, and how to work. Your tool is `/Users/jarrelscy/mk-runs/varloop5-20261124-2130/bin/mk` (`mk submit`, `mk board`, `mk note`); put its directory on your PATH.

## Your track: C

Track C is exact draft-and-verify: every token the engine returns must be the token its own one-token path
(`step()` + argmax) chooses at that point. The evaluator checks this exactly (gates `generate` and `http_tokens_b1`).
A verify pass whose per-row arithmetic differs from `step()`'s can flip a near-tie and fail the gate. A draft (from the
2DSD drafter, early-loop readouts or anything else) only proposes; it never changes the text.

## What must stay the same

1. **The model's function.** For every input the target returns what the checkpoint defines at 5 passes, in its stored
   formats (NVFP4 experts, the MXFP8 experts of four layers, everything else BF16). No requantization, pruning or
   approximation of the target, no part of it skipped in the verify pass. Early-exit passes are allowed only as drafts.
2. **The interface and the workload.** The contract in `engine/api.py` and what the evaluator measures.
3. **The setting.** This Mac, the checkpoint's stored weights as the source, every result reproducible from the
   submitted source. No general engine's model code in the timed path. MLX core, `mx.fast.metal_kernel`, `mx.compile`
   and your own Metal / C / C++ compiled at run time are allowed (Xcode is installed on this Mac now).
4. **The evaluator is not yours.** Do not read, edit or work around the evaluator or its reference data. The reference
   was made on this Mac's CPU in float32 (`scripts/mac_make_reference.py`, 8 sequences).

One job at a time on the GPU: a test of yours running during an evaluation spoils its timing.

Nobody is watching live and nobody will answer questions: decide, write down why, and continue until the time on the
board is used up.

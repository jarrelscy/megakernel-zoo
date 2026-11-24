# TML-d24-varloop-5pass on M5 Max, track C: summary

Final source: the workspace's `engine/`, byte-identical to s0012's, s0014's, s0016's and s0018's snapshots (s0009, s0010 and s0011 have
the same greedy timed path). Every submission of the run is valid; the served tokens are the engine's own step-by-step greedy
tokens.

## Result (HTTP, microseconds per token; lower is better)

| | synthetic | natural | tokens per event (synthetic / natural) |
|---|---|---|---|
| reference: oMLX 0.8.1 fork + MTP | 23256 | 22422 | |
| s0001 starting engine | 11438 | 7235 | 3.25 / 5.89 |
| final engine, seven evaluations (s0009 .. s0012, s0014, s0016, s0018) | 9652, 9727, 9656, 9551, 9657, 9573, 9793 | 7024, 7147, 6793, 6820, 7033, 6802, 7277 | 5.76 / 8.7 |
| control s0013: final engine, readout rounds off | 11129 | 7129 | 3.26 / 5.89 |
| control s0015: final engine, copy proposals off | 12028 | 9925 | 5.38 / 7.0 |
| experiment s0017: readout shortlist of rank 512 (not adopted) | 9773 | 6993 | 5.64 / 8.64 |

Attribution on synthetic text: the cheaper drafter alone -2.7% (control s0013), the readout rounds a further -13%.
Without copy proposals the final engine is 25% / 41% slower (control s0015): long confident chains are the rounds'
weak case (one stage A per token), and the 16-row copy passes carry the repeated text.

Synthetic: median of the seven 9656, -15.6% against the starting engine (2.41x the reference; the seven are within
2.5%, the last one taken with the machine in a slow state). Natural: median 7024, -3% (3.2x the reference): its median stream is mostly 16-row copy passes, which this run
did not change and which follow the machine's heat (6793 / 6820 after idle minutes, 7147 straight after local tests;
the starting engine's 7235 is one evaluation).

On my own prompts, starting engine against final, alternating processes in the same machine state (tests/bench.py):
GSM8K chat through the template 71.3 -> 80.1 tok/s (+12%), natural 100.4 -> 110.5 (+10%), random-id synthetic 60.3 ->
87.5 (+45%; its hard-prose prompts 49 -> 75).

The same comparison with a 131,072-token engine (the drafter's context is then a ring): chat 72.4 -> 79.9 tok/s (+10%),
natural 113.5 -> 123.9 (+9%).

## What was built

1. **Early-loop readouts as the proposer (2DSD's second dimension, in the form that pays on this machine).**
   The pass is split after loop 1. *Stage A* = prefix + loop 1 (visits 0..17) of some rows, plus a draft-only exit
   through the suffix layers that gives the 1-loop model's next token for every row (it agrees with the 5-pass token
   90-96% of the time; the drafter's first token on hard prose is right far less often). *Stage B* = loops 2..5 and the
   suffix over all rows kept, starting from stage A's carried results, so nothing is computed twice and every row has
   the bits of the one pass (tests/bitm.py: tokens, features, K/V rows and conv windows equal bit for bit, with rows
   dropped between rounds). A cycle is rounds of stage A (the readout confirms a prefix of the drafts and corrects the
   first one it rejects; its token starts the next round, alone or with 2-D drafts made from in-flight loop-1
   features), then one stage B. The verify's exact tokens decide, as before.
2. **Readout confidence, calibrated online.** The readout also returns the probability of its token; a per-request
   calibration against the verify's verdicts gives the chance each next row is accepted, and rounds continue / rows
   are kept while that chance pays (tokens per ms above the request's recent rate).
3. **A cheaper drafter.** Its attention is one Metal kernel per layer that reads only visible positions (was MLX
   matmuls over the whole 4104-slot context), its top-64 candidates come from the 64 best chunks of 256, its context
   K/V has a 128-row recent tier (an append no longer rewrites 170 MB). Draft 3.9 -> 2.4 ms alone; the same drafts.
   No draft is made behind stage B while draft calls keep too few rows to pay.
4. **Readout argmax from a state-PCA shortlist.** The readout's final states lie almost in a 768-dimensional
   subspace (fitted at load on states of random tokens, +3 s): the token comes from ~4096 short-listed tokens with
   exact logits instead of a scan of the 461 MB 8-bit unembedding. Draft only.
5. **Copy proposals reviewed first.** The first copy proposal of a loop goes through a readout round instead of a
   blind 8-row pass (it fails 16 times of 23 on chat).

## What the measurements ruled out (details in notes/journal.md)

- **The overlapped (wavefront) form of 2DSD**: a visit's cost here is per row group (253 us at one row, of which the
  dense matrices that overlapping loop depths would share are ~75 us), so overlapping buys little and costs
  speculation several groups deep.
- A separate 1-loop review pass without reuse (chat -8%); a 0-loop readout as a cheaper first stage (agrees only
  53-73%); 24 / 32-row copy passes (4.6 ms per row against 4.5 at 16); loop-3 taps for the drafter (fewer tokens per
  cycle); merged independent dispatches (bit-equal, no gain: a pass is bound by the GPU's dependent chain, the host
  has slack); pre-building the next graphs during a round (no gain); PCA of the unembedding itself (recall 57-77%);
  finer calibration and other policy parameters (all within 1%); graphs built ahead by the main thread or by a helper
  thread during a round (used, exact, not faster).

## Where the engine stands

- Hard text (most of the synthetic workload's time): a cycle is ~3.9 rounds of ~6.6 ms plus a stage B of
  14.2 + 2.6 ms per row, for ~3.3 tokens: ~16 ms per token in the evaluator's streams (the starting engine: ~33 ms
  cycles for 1.7 tokens). What limits it is the readout's agreement with the 5-pass token (chains end at unsure
  readouts; 31% of round cycles commit one token) and the fixed cost of a one-row stage A (24 visits).
- Copy loops (half of the synthetic tokens, two thirds of the natural ones): unchanged, 4.5 ms per token in 16-row
  passes, bound by the GPU's arithmetic.
- Chat: rounds and plain passes cost about the same per token; the gain there is the cheaper drafter, the corrected
  token per round and the reviewed copies.

## Caveats

- Laptop timing: the same source spreads by ~1% on synthetic and ~5% on natural with the machine's heat (natural's
  16-row passes are the most sensitive).
- Exactness beyond the evaluator's 8 sequences: about 75 requests of my own (prose, code, GSM8K chat, random-id
  prompts with several seeds, 2,048 / 4,096 / 20,000-token engines) were compared token by token with step() + argmax
  (tests/bench.py --check, tests/smoke.py): all equal.
- The readout shortlist's basis is fitted on random-token states; on other text its recall was 99.7% or more for the
  top 4096 (tests/pca.py). A miss only costs a draft.
- With a context beyond 4,096 positions the drafter's context is a ring of 4,104 slots (as before); checked with a
  20,000-token engine (tests/smoke.py), not timed.
- Sampling requests take the old cycle (they get the cheaper drafter). Rounds for them exist (MK_ROUNDS_SAMP=1), are
  exact against step-by-step sampling (tests/samp.py) and 1-3% slower, so they are off.
- generate() is still a plain greedy loop of one-row passes; speculation lives in submit / poll.

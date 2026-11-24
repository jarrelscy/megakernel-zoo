# Journal: TML-d24-varloop-5pass on M5 Max, track C

## Starting point (measured locally before any change)
- One-row pass 21.1 ms (72 visits, 293 us each), +2.9 ms per further row (K=8 40.3 ms, K=16 68.6 ms). Draft 4.0 ms alone,
  5.5 ms chained behind a pass.
- tests/bench.py (my own prompts: random-id "synthetic", runs-mac natural, GSM8K chat through the template):
  synth 64.8 tok/s (two hard-prose prompts at 49, one repetitive at 184), natural 110.7, chat 77.9.
- Hard prose: the drafter's first token is rarely right, 1.3 tokens per 27 ms cycle: barely better than one-row steps.

## Early readouts: how often is an r-loop readout the 5-pass token? (tests/alpha.py)
Teacher-forced r-pass model on the 5-pass greedy text (synth / natural / chat):
r=1 0.957 / 0.897 / 0.931, r=2 0.971 / 0.925 / 0.947, r=3 0.986 / 0.959 / 0.973, r=4 0.993 / 0.983 / 0.984.
The 1-loop readout is a far better proposer than the drafter on hard text, and costs 24 visits (a third of a pass).

## Why not the overlapped (wavefront) form of 2DSD here
A visit's cost on this machine is mostly per row group, not per weight byte: 253 us fixed per visit at one row, of which
the dense matrices (what overlapping loop depths would share) are ~75 us; a further row in the same dispatch is ~40 us.
Overlapping five loop depths in one sweep shares only the dense reads, and pays for it with speculation five groups
deep (a wrong readout flushes four younger groups). Estimated at or below the serial form; not built.

## s0001 baseline: 11438 synthetic / 7235 natural (valid)
Evaluator's synthetic streams: 681 one-token events, 264 two-token, 134 sixteen-row copy cycles (2144 of 4096 tokens).
Time is dominated by the hard regime (one or two tokens per ~27 ms cycle).

## Simple review (MK_REVIEW=1, kept off)
A separate 1-loop pass over [anchor, drafts]: keeps confirmed drafts, replaces the first rejected one by its own token.
Hard prose 49 -> 60 tok/s, chat -8% (the extra pass costs 7 ms + 1 ms/row and its work is thrown away).

## Rounds (s0002): stage A / stage B
The pass split after loop 1. Stage A = prefix + loop 1 (visits 0..17) of a row, plus a draft-only exit through the suffix
(reading provisional suffix K/V of the in-flight rows before it). Stage B = loops 2..5 + suffix over all kept rows,
starting from stage A's carried results. tests/bitm.py: A (in several rounds, with dropped rows) + B gives every row the
bits of the one pass (tokens, features, K/V, conv windows). Cycle: round 1 = [anchor, drafts]; each next round = the
readout's token (alone, or with 2-D drafts from in-flight loop-1 features); then one stage B.
Costs: A(K) = 5.9 + 1.0 K ms, B(K) = 14.2 + 2.6 K ms, a chained draft +5.5 ms.
Policy: rounds only when the drafter is worth < 2 rows (m_kd); another round while its tokens per ms beat the request's rate.
Local A/B (same process): hard prose 52 -> 65 and 46 -> 57 tok/s, natural +2%, chat -1.6% (11 of 130 cycles use rounds).

## s0002 early-readout rounds: 10310 / 7226 (valid). Expected ~10000 / -2%.
Rounds only where the drafter was worth < 2 rows. tokens/event 3.25 -> 5.68. Natural unchanged (few such cycles).

## s0003 rounds everywhere + readout confidence: 10225 / 7024 (valid). Expected ~9800 / ~7000.
Stage A also returns the readout's probability of its token; CalR calibrates P(readout = target) by bucket per request
from the verify's verdicts; rounds continue / rows are kept while the calibrated chance that the row is accepted pays
(tokens per ms above the request's rate). Rounds for every non-copy cycle, up to 12 rows per stage B.
The evaluator's hard stretches are harder than my local random-id prompts: 146 one-token and 127 two-token events
remain (the readout is unsure there), so less moved than locally (hard prose 64 -> 67 tok/s).

## s0004 cheaper drafter: 10071 / 7089 (valid). Expected ~9700 / ~6800.
Draft attention as one Metal kernel per layer over visible positions only (was MLX matmuls over all 4104 slots),
top-64 candidates from the 64 best chunks of 256 (exact same set as argpartition over the vocabulary): draft 3.9 ->
2.4 ms alone, same drafts (tests/dcheck.py). No draft behind stage B while a draft call keeps too few rows to pay.
Policy sweeps (round margin 0.8 / 1.0 / 1.25, draft always / by cost): all within 1%. review_thr 0.1 -> 0.2: +2% chat.

## s0005 same source after 10 idle minutes: 10042 / (natural in result). Machine state is not the story here (0.3%).

## Things measured and dropped
- Passes of 24 / 32 rows for copy cycles: the wide path costs 4.6 ms/row against 4.5 at 16 rows (marginal row 4.9 ms
  beyond 16): nothing to amortize. Copy cycles stay at 16 rows.
- Drafter taps of loop 3 (what it saw in training at 3 passes) instead of the last loop: fewer tokens per cycle
  (chat 3.13 against 3.20, synthetic 2.14 against 2.25). Last-loop taps stay.
- Where a one-row pass goes (what-ifs): 19.9 ms = dense 6.1 + experts 6.0 + everything else 8.0 (112 us per visit of
  mixes, gates, attention, convs). A trivial dependent dispatch costs 1.4 us (MK_WHATIF_PAD), so the glue is the small
  kernels' own latency, not their number: merging independent dispatches (three dense GEMVs in one, three gates'
  projections in one, bit-equal) gave 1% of a pass (kept, s0006).
- PCA of the unembedding itself as a shortlist: useless (top 256 directions hold 26% of its energy; recall of the
  argmax 57-77%). PCA of the readout's final states works: see s0006.
- A 0-loop readout (prefix + suffix) as a cheaper first stage of a cascade: agrees with the 5-pass token only
  0.73 / 0.71 / 0.53 (synth / natural / chat): a cascade would cost as much per kept row as stage A does now. Not built.
- Overlapped loop depths (the wavefront form of 2DSD): see the top of this file.

## s0006 state-PCA shortlist for the readout, merged dispatches: 9888 / (natural in result) (valid). Expected ~9850.
Measured 9888 synthetic / 7321 natural. Synthetic as expected (stage A of one row 6.93 -> 6.55 ms, the same acceptance
histogram). Natural +5% against s0005 (6946) with the same events: its median stream is 16-row copy passes, which are
bound by the GPU's arithmetic and follow the machine's heat (s0006 was submitted straight after ~10 minutes of local
benches; in alternating local timings the merged dispatches make no difference at 16 rows: 71.5-74.8 ms either way,
and the second 16-row timing of every process is ~5 ms slower than the first). The merged dispatches are therefore
worth nothing measurable (1% at one row at best): off by default from here (MK_DENSE3 / MK_GATEM), code and
bit-equality test kept.

## Two-tier draft context (s0007)
The drafter's context K/V was rewritten whole by every traced pass that appends rows to it (ten mx.where over
[4104, 8, 128] float32: 170 MB written per append). Now positions below dbase live in the big arrays and newer ones in
a recent tier of 128 rows (the draft attention kernel reads both; a host-side flush every 64 positions). Same drafts
as before (tests/dcheck.py against the s0006 source, also with a wrapped ring). Draft behind a pass: about -1 ms.
Measured: 9888 / 7321 (see above).

## s0007 two-tier draft context, merges off: 9789 / 7229 (valid). Expected ~9750.
## s0008 policy constants, plain pass when all 7 drafts are confident (m_kd 7), flush guard: 9785 / natural in result.
A smoke test (tests/smoke.py: sampling request, 492-token prompt, prefix-cache reuse, also with a 20,000-token context
= ring draft context) found a crash in the two-tier flush when a request reused a prefix that had been fed through
step() (no draft context for it): guarded. All four comparisons equal the step-by-step tokens.

## s0009 copy proposals reviewed by a readout round: 9652 / 7024 (valid). Expected within 1% / -1%.
The first copy proposal of a loop (8 rows, not yet proven) fails 16 times of 23 on chat and 1 in 5 on prose; as the
drafts of a stage-A round a wrong copy costs 1 ms per row instead of 2.9, and the chain goes on with the readout's
token. Locally +0.5% chat, +1.1% natural, +0.6% synthetic.

## Before / after on my own prompts (alternating processes, same machine state, twice each)
| workload | starting engine | now |
|---|---|---|
| GSM8K chat (8 prompts, to EOS) | 71.3 tok/s (14.03 ms) | 80.1 tok/s (12.49 ms), +12% |
| natural (4 prompts) | 100.4 tok/s | 110.5 tok/s, +10% |
| random-id synthetic (3 prompts) | 60.3 tok/s | 87.5 tok/s, +45% |

## More that was measured and dropped (after s0006)
- Glue kernels on the dependent chain (tests/glue.py, 200 dependent calls): gates of the attention input 24 us, of the
  MLP input 16, depth mix 6, router 11, conv 4, attention 12-29 us by context (tests/glue_att.py); a trivial dispatch
  2 us. The first mix kernel (cross-pass messages) takes 23 us against 6 for mix2 (a barrier per dot product), and
  mix2 gives the same bits, but a pass is not faster with it (19.84 against 19.88 ms): the messages do not depend on
  the visit's chain, and independent dispatches overlap. That is also why merging independent dispatches gained
  nothing. What is left is the dependent chain of ~22 small dispatches per visit at their own latency.
- Attention with 1024 instead of 512 threads per (row, head) group at fewer than 4 rows: 28 -> 24 us at 1000 keys,
  19.2 -> 18.5 at 600: at most 0.3 ms of a pass. Not taken.
- One-row cycles (the anchor's readout unsure): 16% of round cycles locally, 31% in the evaluator's streams; whether
  the next cycle is one too is not predictable from the last (9 of 65), so no fallback to a plain one-row pass.
- Readout calibration with finer buckets near 1, a slower decay, or a lower round threshold: within 0.5% (tests/calib.py
  shows the product of calibrated chances is under-confident in the middle range, 0.5-0.7 predicted against 0.75-0.8
  accepted, but lengthening chains there does not pay).
- A drafter shortlist like the readout's: its top-64 needs ~4096 exact candidates and a flagged 8-bit GEMV; saving
  ~0.35 ms per draft at best. Not built.
- Building the next round's graph and stage B's while the GPU runs a one-row round (replaying a traced stage A takes
  0.44 ms and stage B 1.05 ms on the host): exact, and no gain (79.3 against 77.6 tok/s on hard prose, chat and natural
  equal). Removed. tests/hostcpu.py: of a one-row pass's 18.5 ms the calling thread is busy 8.6 ms; the merged
  dispatches cut that to 7.5 ms without changing the wall time, and two independent chains of trivial dispatches take
  as long as one of twice the length. So a pass is bound by the GPU's dependent chain, not by the host's encoding.

## s0010 the final source again: 9727 / 7147 (s0009: 9652 / 7024). Spread of the same engine about 1% / 2%.

## Last kernel experiments (none kept on)
- tests/par.py: two independent chains of dispatches (gates, a dense GEMV, router + experts) take twice as long as
  one: the GPU runs MLX's dispatches one after the other; nothing overlaps.
- The depth-mix kernels recompute the same softmax weights in all 576 threads of a row. A variant where thread j
  computes query j's weights once and the others read them (mix2f) gives the same bits for every shape tried
  (tests/mixeq.py) and is up to 2.4x faster alone at 7 sources and 3 queries (19.5 -> 8.0 us). In a pass it gains
  0.3% (19.82 -> 19.76 ms): 20 of the 24 layers are local (one query: 6.0 -> 5.6 us) and the cross-pass message mix
  has only 2 sources there, not the 5 of my microbenchmark. So the small kernels are already near the ~2 us floor of
  a dispatch each, and a visit's ~28 of them, run one after the other, are the 8 ms of a one-row pass that are not
  weights (in a pass without the dense and expert kernels the host is busy 5.7 of 6.9 ms encoding them: host and GPU
  are about balanced there). Cutting that needs far fewer dispatches per visit (a visit as a handful of kernels in
  an own Metal pipeline), which is beyond this run. MK_MIXF / MK_MSG_MIX2 / MK_DENSE3 / MK_GATEM are off.

## s0011 the final tree: 9656 / 6793 (valid). Three evaluations of the final engine: 9652, 9727, 9656 synthetic;
7024, 7147, 6793 natural (the last after ~15 mostly idle minutes: the 16-row copy passes follow the heat).

## Checks at the end
- 16-row passes are the best width for copies: 4.31 ms per row; 20 / 24 rows through the same kernels 5.3 ms per row
  (the accumulators no longer fit the registers), through the prefill's blocked kernels 4.5-4.7.
- Attention with 1024 threads per group: nothing in a pass. Drafter variants (plain cycles, GSM8K chat, tokens per
  cycle): anchor through the embedding norm + softcap after the Markov bias 3.17 (the default), raw anchor 2.95,
  softcap before 3.11, Markov bias over the whole vocabulary 3.18, over the top 16 only 3.15.
- Exactness audit: ~75 requests of my own against step() + argmax, all equal (see summary).
- Minimum copy match 3 / 4 / 5 / 7 tokens with reviewed copies: 8.10 / 8.05 / 8.14 / 8.15 natural, 12.80 / 12.74 /
  12.73 / 12.62 chat, 11.52 / 11.60 / 11.55 / 11.34 synthetic (ms per token): no trend. Stays 5.

## Estimates not built
- Readout after loop 2 instead of loop 1 (stage A of 30 + 6 visits): errors drop by about a third, chains grow ~40%,
  but a round costs 9.6 ms instead of 6.5: ~16.8 against 16.4 ms per token in the evaluator's hard stretches.
- A second, deeper readout only where the first is unsure: the unsure row's loop 2 alone (12 visits) plus a second
  exit costs ~5 ms per event and would turn perhaps 40% of the stops into continuations (17 ms saved each): about
  +1.5% for rows at mixed depths in stage B. Not worth its state handling.

## Where to go from here (for a later run)
1. The fixed cost of a visit at few rows (8 of a one-row pass's 19.8 ms are ~28 small dispatches per visit, executed
   one after the other): a visit as a handful of kernels in an own Metal pipeline, which MLX's per-dispatch path
   cannot give. It would lift stage A of a row (6.5 ms), stage B's 14.2 ms and every plain pass.
2. A proposer better than the 1-loop readout at the positions where it is unsure (31% of round cycles commit one
   token in the evaluator's streams).
3. Copy loops are at the GPU's arithmetic limit (4.3 ms per row at 16 rows); nothing found there.

## Rounds for sampling requests (off the evaluator's path): exact, not faster; off
The chat default is sampling, so the rounds were extended to it: the readout draws its token with the target's noise
of the position (from its 8-bit logits, without the top_p / top_k cut), stage B draws each row's token from its exact
logits as a sampling step does. tests/samp.py, 6 GSM8K prompts x 256 tokens, graphs compiled beforehand: the request
path with rounds = without rounds = step-by-step sampling for all three settings, and tokens per second
72.0 against 72.5 (T 0.7), 72.6 against 74.9 (T 1.0, top_p 0.95), 71.6 against 72.8 (T 0.3, top_k 40): the readout's
draw agrees with the target's less often than its argmax does, and both stages pay for full logits. MK_ROUNDS_SAMP=1
turns it on; the default keeps sampling requests on the cycle they had (they still get the cheaper drafter).

## s0012 the final tree (sampling-rounds code added, off): 9551 / 6820 (valid)
Four evaluations of the final engine: 9652, 9727, 9656, 9551 synthetic; 7024, 7147, 6793, 6820 natural.

## Pre-building graphs during a one-row round, again, with counters: used, and slower
The pre-built next-round graph was used in 1752 of ~2100 rounds and the pre-built stage B in 333 of ~400 cycles, and
the hard-prose prompts still ran slower (78 / 76 -> 72 / 71 tok/s; the next round only: 74 / 72; stage B only: 69 /
69). The rounds themselves got longer by about half of the host time spent building (stage A 6.6 -> 7.2 ms with 1.2 ms
of building per round): after mx.async_eval returns, the GPU's progress still depends on the calling thread, so host
work during a pass stalls it. Replaying the traced graphs (0.44 ms per stage A, 1.05 ms per stage B, ~2.8 ms or 5% of
a hard cycle) can therefore only be moved off the critical path by a second thread, which I did not try this late.
Removed again; the final tree is s0012's.
- The second thread, tried after all (tests/gil.py: mx.eval releases the GIL; a helper thread bound to the main
  thread's stream can replay a traced graph during a pass, and in isolation stage A + a ready stage-B graph takes
  6.03 instead of 6.70 ms). In the cycle (helper builds the next one-row round and stage B during each one-row round):
  hard prose 79 / 71 -> 75 / 70 tok/s, chat equal, natural slower. Not a gain in situ and a thread-safety risk in the
  evaluated path: removed. The final tree stays s0012's.

## At the full 131,072-token context (the drafter's context is then a ring of 4,104 slots)
tests/smoke.py with a 131,072-position engine: sampling, a 492-token prompt and a prefix-cache request all equal the
step-by-step tokens. Starting engine against final, alternating processes, GSM8K chat (6 prompts) / natural (2):
72.2, 72.5 -> 78.8, 80.9 tok/s (+10%) and 112.7, 114.3 -> 122.9, 124.9 (+9%). With rounds off (plain cycles, the new
drafter) chat is 76.7: at this context the rounds add about 4% on top of the cheaper drafter.

## s0013 control: the final engine with rounds off: 11129 / 7129 (valid). Expected ~11000 / ~7100.
Plain draft-and-verify cycles with this run's drafter: the same events as the starting engine (3.26 / 5.89 tokens per
event), 2.7% faster on synthetic text (11438 -> 11129). The readout rounds (with the reviewed copies and the readout
shortlist) are the other 13% (11129 -> 9551..9727). The workspace's engine was restored after the control and is
byte-identical to s0012's snapshot.

## s0014 the final engine once more (byte-identical to s0012): 9657 / 7033 (valid)
Five evaluations of the final engine: synthetic 9652, 9727, 9656, 9551, 9657 (median 9656); natural 7024, 7147, 6793,
6820, 7033 (median 7024). Starting engine 11438 / 7235; control with rounds off 11129 / 7129.

## After the final evaluations: a harder local proxy, and two more nulls
- tests/hardscan.py: of 24 random-id prompts, 7 continue as hard text (67-83 tok/s, 8-22% one-row round cycles, few
  copies; the evaluator's streams have 31%); tests/bench.py `hard:N` runs them.
- Round policy on the hard set (ab.py, 7 prompts): round margin 0.8 / 1.0 / 1.25: 13.85 / 13.66 / 13.75 ms per token;
  rows kept down to half the threshold: 13.57. Flat there too; the defaults stay.
- Readout shortlist of rank 512 instead of 768 (102 MB scanned instead of 154): stage A of a row 6.62 -> 6.49..6.55 ms,
  the same acceptance on the hard set (3132 of 3342 against 3130 of 3354); under 1% of a cycle, below what an
  evaluation resolves. Rank 768 (the better recall on chat) stays.

## s0015 control: the final engine with copy proposals off: 12028 / 9925 (valid). Expected ~10100 / ~8000.
Much worse than I expected (+25% synthetic, +41% natural against the final engine). Without copies, repeated text is
carried by rounds: 131 (synthetic) and 135 (natural) events of 12 tokens, each built one row per round (6.5 ms a row
plus stage B) where a 16-row copy pass costs 4.3 ms a row. So the copy proposals are worth more in this engine than in
its predecessors, and a long confident chain is the rounds' weak case: the readout is sure, but each of its tokens
still needs its own stage A.
- A fix tried at once: when a chain is 3 rows long and still above 0.85 calibrated, make the next round a drafted one
  (the 2-D block) whatever the drafter's recent average says. Natural -0.7% (better), hard set +5.5% (worse), chat
  +1.8% (worse); with 2 rows / 0.9: -1.2%, +3.2%, +1.7%. Not kept: confident readouts on hard text do not mean the
  drafter's block is right. What would help that case is a batched proposer for predictable text other than exact
  copies (the final engine has 15 / 30-38 such 12-token events per four streams, so about 1% / 4% is there).
The workspace's engine was restored after the control: byte-identical to s0012 / s0014.

## s0016 the final engine, sixth evaluation: 9573 / 6802 (valid)
Six evaluations of the final engine: synthetic 9652, 9727, 9656, 9551, 9657, 9573 (median 9654); natural 7024, 7147,
6793, 6820, 7033, 6802 (median 6922). The run ends here with the workspace's engine byte-identical to these snapshots.
- Last experiment: the confident-chain drafted round gated on the drafter's recent keep rate (ema_kept >= 0.5 / 0.8):
  natural 7.89 -> 7.95 / 7.90 ms per token, hard set 12.94 -> 13.01 / 13.00, chat 11.37 -> 11.53 / 11.40. The gate
  removes the loss on hard text and the gain on natural with it. Not kept; the engine is unchanged (s0012's bytes).

## s0017 experiment: readout shortlist of rank 512: 9773 / 6993 (valid). Expected within 9551-9727, at best -0.5%.
Slightly worse than every evaluation of the final engine (9551..9727), with fewer tokens per event (5.64 against
5.74-5.76): on the evaluator's text the smaller basis misses the readout's token a little more often than on my hard
prompts, and that costs more than the 0.1 ms per round it saves. Not adopted: rank 768 stays. The workspace's engine
is the final one (byte-identical to s0012 / s0014 / s0016); s0017's snapshot differs in that one default.

## s0018 the final engine, seventh evaluation (the run's last record): 9793 / 7277 (valid)
The slowest of the seven (9551 .. 9793), and its in-process greedy loop took 23195 us per step against ~20400 in every
other evaluation: the machine was in a slow state after two back-to-back evaluations and local A/Bs. s0017 (rank 512,
9773) ran in the same stretch, so its "slightly worse" is confounded with that state; its lower tokens per event
(5.64 against 5.74-5.76) is not, and is the reason rank 768 stays.
Seven evaluations of the final engine: synthetic 9652, 9727, 9656, 9551, 9657, 9573, 9793 (median 9656); natural
7024, 7147, 6793, 6820, 7033, 6802, 7277 (median 7024). Time on the board is used up.

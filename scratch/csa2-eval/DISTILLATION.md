# Distilling the converted 2B against the cached teacher (2026-09-21)

The conversion screen left two regressions. Control tokens had been destroyed by 30M
tokens of cross-entropy on the `dense_gr` code corpus, which carries none of this
tokenizer's protocol tokens. MMLU had fallen 12.5 points to the conversion itself, and
those 30M tokens recovered 2.7 of them.

Both were retrained from `checkpoints-2b/warmed-scaled` -- the conversion plus its indexer
warm-up, with no language-model training on top -- against `teacher-cache-5m`: 5,303
chat-formatted documents with a 27B teacher's top-64 distribution at every position. Three
passes, 9,969,848 scored tokens, lr 7.3e-6, prefix cap 1024.

Two arms, identical but for the objective, one per card. They agree bitwise at step 0
(`train 1.3196`, `idx 3.969`, same routing densities), so the only difference between them
is the teacher term.

* **ceonly** -- ground-truth cross entropy alone. `--teacher-weight 0`.
* **blend** -- `0.5 * ce + 0.5 * grouped-tail KL` against the cached top-64.

## Results

| metric | source | conversion | ceonly | blend |
| --- | --- | --- | --- | --- |
| mmlu acc (512 questions) | **0.5762** | 0.4883 | 0.4590 | 0.5137 |
| nll all | 1.3713 | 1.4817 | **0.5748** | 0.7569 |
| nll content | 1.7572 | 1.8978 | **0.7260** | 0.9296 |
| nll layout | 0.5096 | 0.5228 | **0.1731** | 0.3247 |
| nll punctuation | 0.6109 | 0.6733 | **0.3183** | 0.4576 |
| nll control | 0.6327 | 0.7860 | 0.1286 | **0.0808** |
| arc acc | 0.3906 | 0.4062 | 0.3984 | 0.3750 |
| arc acc_char_norm | 0.4102 | 0.3945 | 0.4102 | 0.3906 |

Paired bootstrap, 10,000 resamples:

| comparison | estimate | 95% CI |
| --- | --- | --- |
| blend - ceonly, mmlu (512) | +0.054688 | [+0.021484, +0.089844] |
| blend - source, mmlu (512) | -0.062500 | [-0.105469, -0.021484] |
| ceonly - source, mmlu (512) | -0.117188 | [-0.164062, -0.074170] |
| conversion - source, mmlu (512) | -0.087891 | [-0.132812, -0.042969] |
| blend - conversion, mmlu (512) | +0.025391 | [-0.015625, +0.068359] |
| ceonly - conversion, mmlu (512) | -0.029297 | [-0.068359, +0.007812] |
| blend - ceonly, nll | +0.182082 | [+0.169886, +0.194435] |
| blend - source, nll | -0.614355 | [-0.699173, -0.532596] |
| ceonly - source, nll | -0.796438 | [-0.889536, -0.705430] |
| blend - source, arc acc | -0.015625 | [-0.062500, +0.031250] |

## What it says

**The teacher term buys MMLU and costs NLL, and both are now resolved.** It is worth
5.5 points of MMLU over cross entropy alone, and costs 0.182 nats. That is the open
question commit 16222a8 left: it found the teacher term impossible to justify on held-out
ground-truth NLL, and said so -- "the evaluation metric is the CE objective, so the arm
trained directly on it wins by construction". It does win, by 0.182 nats with an interval
nowhere near zero. On a metric that is not the training objective it loses, by 5.5 points
with an interval that is also clear of zero.

That result took 512 questions to see. At 256 the same comparison read +4.7 points with
the interval spanning zero at [-0.004, +0.098], and the blend's accuracy read 0.4727
against 0.5137 -- the smaller sample was simply unlucky for that arm. The 256-question
reading should not be quoted.

**What the teacher term does is protect, not recover.** Against the conversion it started
from, the blend is +2.5 points with the interval spanning zero, and cross entropy alone is
-2.9 points, also spanning zero. Neither arm moves MMLU away from `warmed-scaled` by more
than noise. What separates them is each other. So the finding is not that distillation
recovered anything; it is that training on this corpus with cross entropy alone costs
MMLU, and carrying the teacher's distribution stops it costing MMLU. That is worth having
-- the corpus is what fixes control tokens, and this is what makes it safe to use -- but
it is not progress against the conversion's own loss.

**Control tokens are fixed, and over-fixed.** The source is at 0.6327 nats, the blend at
0.0808 -- eight times better. This is the regression that motivated the whole exercise and
it is gone. Some of that is the corpus being chat-formatted where `dense_gr` was not, and
some is that a protocol token is easy to predict once the model has seen the protocol; a
model eight times better than its own source at something is describing the corpus more
than the capability.

**The NLL column is mostly domain adaptation and should not be read as capability.** The
screen's held-out documents come from `capture-data/heldout.jsonl`, the same capture
pipeline that produced the training corpus. The documents are disjoint -- 768 held out
against 5,659 cached, overlap 0, checked -- but the distribution is the same, so training
on one and scoring on the other measures fit to a distribution rather than general
ability. The MMLU column is the evidence: NLL improved 0.8 nats while MMLU fell.

**The conversion's own loss is untouched.** It costs 8.8 points of MMLU
[-0.132812, -0.042969] before a single token of language-model training, and the blend
ends 6.25 points below the source. 10M tokens of distillation does not recover what the
least-squares refit onto MLA and CSA2 gave away.

**ARC never moves.** Every arm, every normalisation, every interval spans zero. Whatever
the conversion costs and whatever the training recovers, ARC does not see it.

## Training the chat-calibrated conversion (2026-09-21, later) -- confounded

Recalibrating the conversion on chat left it 2.9 MMLU points from the source instead of
8.8, so the obvious question was whether training holds that. It was trained the same
way as `blend` above -- `teacher-cache-5m`, 0.5/0.5, three passes, 9,964,545 scored
tokens, lr 7.3e-6, prefix cap 1024 -- from `checkpoints-2b/warmed-chat32`, and on both
cards under `--tensor-parallel`.

Every prior arm used one card. That difference turns out to matter, so the numbers below
are reported and then set aside rather than compared.

| metric | source | conv chat/32K | kd (python calib) | chatkd (chat calib) |
| --- | --- | --- | --- | --- |
| mmlu acc (512) | **0.5762** | 0.5469 | 0.5137 | 0.4785 |
| nll all | 1.3094 | 1.4174 | **0.7269** | 1.0652 |
| nll content | 1.6954 | 1.8348 | **0.8997** | 1.2613 |
| nll layout | **0.5575** | 0.5815 | 0.3422 | 0.7354 |
| nll control | 0.4049 | 0.4702 | 0.0586 | **0.0618** |

The NLL column is on the 32-document bank in `q512-bundle.json`, 14,396 scored tokens.
The 384-document bank the earlier tables used is not on disk any more, so every arm here
was rescored on the current one; the numbers differ from the earlier tables for that
reason and not because any model changed.

| comparison | estimate | 95% CI |
| --- | --- | --- |
| chatkd - conv chat/32K | -0.068359 | [-0.113281, -0.023438] |
| chatkd - kd | -0.035156 | [-0.078125, +0.007812] |
| chatkd - warmed (python conv) | -0.009766 | [-0.052734, +0.031250] |
| chatkd - source | -0.097656 | [-0.144531, -0.052734] |

Read at face value this says training cost 6.8 MMLU points from the better conversion and
landed below the arm that started 5.9 points lower. Two things say not to read it that
way.

**The generalisation gap is five times the other arm's.** `kd` ended at 0.6592 training
cross entropy and 0.7269 held-out, a gap of 0.068. `chatkd` ended at 0.7262 and 1.0652,
a gap of 0.339, on the same corpus with the same objective and the same token budget.

**Layout got worse than the source.** 0.7354 against the source's 0.5575 and against the
0.5815 of the conversion it started from. Training on a chat corpus making a model worse
at chat layout is not a thing the corpus can explain.

**Most of the model never trained.** The optimizer was built from `model.parameters()`
before `shard_model` ran. Sharding *replaces* the modules it splits, so every parameter
of a sharded module becomes a new tensor and the ones the optimizer was holding are
orphans -- still stepped, no longer attached to anything.

Diffed against the checkpoint it started from, the run moved 546.3M of 1,915.2M
parameters, and 508.6M of that is the tied embedding. What moved is exactly what
sharding leaves alone: the residual adapters, the indexer, `kv_a_proj`, `kv_a_norm` and
the embedding, about 37.7M parameters outside the embedding. What is **bitwise
identical** to the starting checkpoint is every MLP, the whole gated delta net,
`q_proj`, `kv_b_proj`, `o_proj` and every norm. The single-card `kd` arm, diffed the
same way, moved 1,915.1M of 1,915.2M.

So the two arms are not the same experiment. `kd` trained 1.9B parameters on 10M tokens;
`chatkd` trained 37.7M adapters plus the embedding on the same 10M tokens, and the
comparison says nothing about calibration. Nothing raised, the loss fell from 1.2983 to
0.7262, and only a diff against the starting checkpoint shows it.

`tp_bench.py` builds its optimizer after sharding, so the throughput and memory numbers
in `TENSOR_PARALLEL.md` included optimizer state for the sharded body, but used a
different objective/step from the trainer and are not a capacity guarantee for it.
The 19.30 GiB training peak omitted optimizer state for the replaced parameters.
It did **not** omit their gradients: replacement shards retain `requires_grad`, and
the stale optimizer's `zero_grad` does not clear them. Those gradients can accumulate
and enter clipping even though the weights never update. A CPU reproduction using the
actual sharded linear layer confirms this. Re-measure several real optimizer steps;
do not inherit the old batch size. Historical measurements remain recorded, not corrected
retroactively into results for the repaired trainer.

**A second defect, with unresolved indirect impact here.** A parameter replicated
across ranks sees only its own rank's share of the loss, so each copy holds a partial
and `sync_replicated_gradients` has to sum them before the optimizer step. Its own
docstring says so, `trainer.py` calls it and `tp_train.py` calls it; `smoke_train.py`
never did.

Isolated on the sharded `GatedDeltaNet` in float32 on CPU, where the split is exact:

| | norm.weight | every other parameter |
| --- | --- | --- |
| shards as the run left them | **0.44** relative | inside 1e-6 |
| after `sync_replicated_gradients` | inside 1e-6 | inside 1e-6 |

On CUDA, with the autotuner warmed and the reduction applied, the whole module agrees
with the unsharded one to 8.6e-6, against 9e-8 for the unsharded module measured twice.
So the sharding arithmetic is right on both devices and the missing call is the whole of
the defect. It touches the `norm` of every linear-attention layer and the `q_norm` and
`k_norm` of every sharded full-attention layer.

Clipping had the mirror of the same problem: `clip_grad_norm_` over `model.parameters()`
counts a replicated parameter once per rank, inflating the norm it measures and scaling
every gradient down for it.

The 18 replicated norms did not update: all 18 are bitwise identical between ranks and
to the starting checkpoint. This is not proof that their gradients were absent or inert
in global clipping. The stale optimizer could leave them accumulated, so the indirect
effect on updated parameters is unresolved. Reduction and model-wide gradient clearing
are required for corrected runs.

So `chatkd` is not a measurement of what chat calibration is worth after training, and
the comparison has to be redone before anything is concluded about the calibration.

**A measurement error worth recording.** This was first reported here as nine tensors --
`A_log`, `dt_bias` and `in_proj_a` at layers 4, 6 and 8 -- differing by up to 6.4e-1,
measured on the whole model on CUDA. They do not. That check ran the sharded model's
backward without first warming the Triton autotuner, which the existing sharded-forward
test does explicitly and comments on; repeating it with the warm-up puts every one of
those tensors inside 1e-5. The reassembly in that check also summed the replicated
groups itself, which masked the one parameter that was genuinely wrong. The number that
stands is 0.44 on `norm.weight`, from the module in isolation.

### The checkpoint it was read from

`smoke_train --tensor-parallel` wrote the sharded state dict: `mlp.gate_proj.shards.0`
where the plain model wants `mlp.gate_proj.weight`. It saved without complaint and loaded
with every plain key reported missing and freshly initialised. `merge_tp_checkpoint.py`
reassembles it; the merged checkpoint scores 0.9936 on the capture's eval split and the
sharded model scores 0.9936 on the same documents, so the merge is the trained weights
and the table above is of the right model.

The run log reported 0.9812 for the same 128 documents at the last step. The checkpoint
and the sharded model in memory agree with each other and disagree with the log, so the
difference is in how the training loop evaluates rather than in what it saved. Not
chased; the independent screen loads a fresh checkpoint and is unaffected.

## Caveats

* The 512-question set is the `screen` split extended from 256, not an independent test.
  It is the same measurement with more samples. The `confirmation` split is untouched and
  should stay that way until there is a decision worth spending it on.
* Three passes over 3.3M tokens leaves a 0.125-nat gap between training loss and the
  capture's own eval split for the cross-entropy arm (0.4528 against 0.5778). The blend's
  gap is smaller. Neither is alarming; both are worth watching if the budget grows.
* The teacher was captured under bitsandbytes int8, so the cached distribution carries
  whatever that quantisation cost. This bounds how good a target it can be.
* The prefix cap of 1024 keeps 74.1% of the corpus. It is a memory ceiling, not a choice:
  the sparse stage records attention, which forces CSA2's gathered path, whose selection
  is quadratic in the sequence.

## Next

The two regressions have separated cleanly. Control tokens are solved. MMLU is not, and
the remaining 6.25 points belong to the conversion rather than to anything trained
afterwards -- so the lever is the conversion itself, not more tokens of the same. Worth
measuring before anything else: whether the gap tracks conversion quality, by screening
the mla-only and all-full checkpoints that already exist.

## The repaired run (2026-09-24): the converted model matches its source on the MMLU screen

Every fix from `scratch/dense_gr/TRAINING_REPAIR.md` at once: Kahan-compensated
AdamW8bit, so updates are no longer rounded away; body lr 1.83e-6 and router lr 9.37e-4,
from the sweeps; micro-batches weighted by targets; the optimizer built after sharding;
replicas reduced; the batch KL denominator fixed; 152 contaminated documents excluded;
`--min-answer-tokens 2`. From `checkpoints-2b/warmed-chat32`, on the merged 7.5M-token
corpus (`teacher-cache-5m` plus the expansion captures), blend 0.5, 10M tokens (1.52
passes), two cards, 1,755 tokens a second, home-card peak 21.58 GiB against a 21.6
allowance.

| metric | source | conv chat/32K | kd (old, frozen) | **repaired** |
| --- | --- | --- | --- | --- |
| mmlu acc (512) | **0.5762** | 0.5469 | 0.5137 | **0.5742** |
| nll all | 1.3094 | 1.4174 | 0.7269 | **0.7257** |
| nll content | 1.6954 | 1.8348 | 0.8997 | **0.8879** |
| nll layout | 0.5575 | 0.5815 | **0.3422** | 0.3876 |
| nll control | 0.4049 | 0.4702 | **0.0586** | 0.0876 |

| comparison | estimate | 95% CI |
| --- | --- | --- |
| repaired - source, mmlu | -0.001953 | [-0.037109, +0.033203] |
| repaired - conv chat/32K, mmlu | +0.027344 | [-0.013672, +0.068359] |
| repaired - kd, mmlu | +0.060547 | [+0.019531, +0.101562] |
| repaired - source, nll | -0.5837 | [-0.8923, -0.3305] |

**The MMLU gap to the source is gone:** 0.2 points with an interval centred on zero.
The conversion alone was 2.9 points short; the frozen-optimizer kd arm ended 6.25 points
short and is now beaten by 6.1 with an interval clear of zero. The NLL column is still
mostly domain adaptation, as it was for kd.

Held-out on the capture's own split bottomed near step 1,200 (0.8991, 1.08 passes) and
rose to 0.9237 by the end, in all three sources, while training loss kept falling: the
second pass is beginning to overfit. The checkpoint screened is the final one; step
1,200 is saved as resumable state and may be slightly better.

**What this does not establish.** Everything changed at once -- calibration corpus,
optimizer precision, learning rates, data -- so the gain is not attributed to any one of
them. The 512 questions are the screen split, extended from 256, and every decision in
this programme has been made on it; the `confirmation` split is untouched. This is the
result it was reserved for.

### General-text NLL: the gap the chat corpus cannot close

Every NLL above is scored on documents from the capture pipeline the model trains on, so
it rewards adaptation to that distribution. `scratch/dense_gr/general_nll.py` scores 64
fixed 1024-token windows of WikiText-103 test -- encyclopedic prose, the same windows for
every checkpoint, paired:

| checkpoint | NLL | vs source | 95% CI |
| --- | --- | --- | --- |
| source | 2.5292 | -- | -- |
| conv chat/32K, untrained | 2.7800 | +0.2507 | [+0.2324, +0.2719] |
| kd (old, frozen) | 2.8227 | +0.2935 | [+0.2798, +0.3080] |
| repaired | 2.7671 | +0.2379 | [+0.2240, +0.2534] |

The conversion costs 0.25 nats of general language modelling, and 10M tokens of chat-SFT
training recover 0.013 of it. The repaired model matches the source on MMLU and beats it
by 0.58 nats in-domain while remaining 27% worse in perplexity on general prose: the
training corpus repairs what it contains and nothing else. Closing this gap needs general
text in the training corpus, not more chat.

### General-text pilot (2026-09-25): half the gap closed, MMLU to watch

A logits-only capture (`general_corpus.py`, no hidden states, 1.09 GB for 3.0M tokens)
of screened FineWeb-Edu (1.80M), FineMath (0.60M) and raw Python (0.60M), cut into
1024-token pieces, mixed about half and half with `teacher-cache-5m` as chat replay.
Continued from the repaired checkpoint with a fresh optimizer, 5M tokens. Interrupted by
a machine restart at step 800 of 835 and resumed from the saved state, which re-scored
held-out at 1.3901 against 1.3897 before the restart.

| checkpoint | WikiText NLL | vs source | 95% CI | MMLU (512) | in-domain NLL |
| --- | --- | --- | --- | --- | --- |
| source | 2.5292 | -- | -- | 0.5762 | 1.3094 |
| repaired | 2.7671 | +0.2379 | [+0.2240, +0.2534] | 0.5742 | 0.7257 |
| general pilot | **2.6418** | **+0.1126** | [+0.1021, +0.1247] | 0.5410 | **0.7111** |

About 2.5M tokens of general text closed 53% of the general-text gap, and in-domain NLL
improved rather than regressing. MMLU moved -0.0332 against the repaired checkpoint
[-0.0684, +0.0020] and -0.0352 against the source [-0.0742, +0.0039]: not significant,
but a point estimate large enough that the scale-up is designed to protect it rather
than assume it holds.

### General-text scale-up (2026-09-25): two thirds of the gap closed, MMLU within noise

One pass from `warmed-chat32` (not continued from the repaired model) over every cache:
the three chat and code captures plus `teacher-cache-general-pilot` and a second, larger
logits-only capture `teacher-cache-general-scale` (9.0M tokens, 3.26 GB, continuing past
the pilot's documents). 29,543 documents, 17.4M supervised tokens, about 57% general
text. Body lr 1.83e-6, router 9.37e-4, Kahan-compensated AdamW8bit, linear decay over
the last 20% of tokens to 0.1. Peak 20.09 GiB, spill +0.09 GiB, 1,769 tok/s.

Held-out moved on every source, and chat tracked the chat-only repaired run step for
step (0.703 vs 0.703 at step 800), so the general text did not displace it:

| held-out source | step 0 | step 2903 |
| --- | --- | --- |
| teacher-cache-5m | 1.2044 | 0.6431 |
| expand-chat | 2.0845 | 1.4698 |
| expand-code | 1.3414 | 0.7480 |
| general-pilot | 2.2093 | 2.0151 |
| general-scale | 2.3213 | 2.1447 |

| checkpoint | WikiText NLL | vs source | 95% CI | MMLU (512) | in-domain NLL |
| --- | --- | --- | --- | --- | --- |
| source | 2.5292 | -- | -- | 0.5762 | 1.3094 |
| repaired | 2.7671 | +0.2379 | [+0.2240, +0.2534] | 0.5742 | 0.7257 |
| general pilot | 2.6418 | +0.1126 | [+0.1021, +0.1247] | 0.5410 | 0.7111 |
| scale | **2.6084** | **+0.0791** | [+0.0691, +0.0909] | 0.5508 | 0.7214 |

The general-text gap is down 67% from the repaired model. MMLU is -0.0254 against the
source [-0.0684, +0.0156], -0.0234 against the repaired model [-0.0625, +0.0156] and
+0.0098 against the pilot [-0.0254, +0.0449]: all within noise, but both runs with
general text sit about 0.02-0.03 below the chat-only model, which is a consistent
enough direction to treat as real until the confirmation split says otherwise. The
scale run saw about 7.5M chat tokens against the repaired run's 10M.

### Correction: WikiText was scored on the block path (2026-09-25)

CSA2 had two whole-sequence paths. The block-sparse FlexAttention one ran whenever a
length divided 128 and nothing was recorded, and it routes 128-token blocks; training's
sparse stage and the llama.cpp graph both route per token. WikiText's 1024-token windows
took the block path, so every CSA2 row above was scored on a function neither training
nor serving computes. Per-token routing is now the only no-cache path (`223d4fe`).
Re-scored on it, same windows, paired against the source:

| checkpoint | WikiText NLL | vs source | 95% CI | MMLU (512) | in-domain NLL |
| --- | --- | --- | --- | --- | --- |
| source | 2.5292 | -- | -- | 0.5762 | 1.3094 |
| repaired | 2.7440 | +0.2148 | [+0.2042, +0.2261] | 0.5742 | 0.7254 |
| general pilot | 2.6180 | +0.0887 | [+0.0822, +0.0958] | -- | -- |
| scale | **2.5834** | **+0.0541** | [+0.0483, +0.0603] | 0.5508 | 0.7214 |

Every CSA2 arm moves by about 0.024 and the ordering is unchanged; the scale run closed
75% of the repaired model's general-text gap rather than 67%. MMLU and in-domain NLL are
unchanged to four places: their prompts rarely divide 128, and a short prompt's selection
covers every causal position either way.

### Layer borrowing: Reuse costs MMLU (2026-09-25)

CSA2 lets a layer borrow the nearest Full layer's latent (and, in Reuse, its selection)
instead of caching its own. `borrow_profile.py` measured the scale checkpoint's six
layers: latents only partly shared (R^2 0.37-0.66), rotary keys unrelated (cosine about
0; a converted model's layers never learned to share one), and a single zero-shot Reuse
costing +0.020 to +0.029 nats on WikiText validation, layers 11 and 23 cheapest. Three
arms then trained the same 3M tokens from the scale checkpoint (seed 1, all five caches,
decay over the last 30%), `kv_adapt` fitted by least squares for the borrowers:

| arm | cache saved | WikiText vs control | MMLU (512) | vs control | in-domain NLL |
| --- | --- | --- | --- | --- | --- |
| source | -- | -0.0667 | 0.5762 | -- | 1.3094 |
| control FFFFFF | 0% | 2.5959 | **0.5684** | -- | **0.7131** |
| FFUFFU | 33% | +0.0154 [+0.0126, +0.0183] | 0.5391 | -0.0293 [-0.0605, +0.0020] | 0.7270 |
| FUFUFU | 50% | +0.0333 [+0.0294, +0.0373] | 0.4629 | -0.1055 [-0.1523, -0.0605] | 0.7385 |

Training recovered about half of each pattern's zero-shot NLL cost, but MMLU is far more
sensitive than NLL: three borrowers cost 10.5 points, clearly significant, for 3% more
WikiText loss. At this size the saving is small in absolute terms -- the six attention
layers cache 5.25 KiB a token, 672 MiB at 128K, beside a fixed 19.7 MiB of recurrent
state -- so all six layers stay Full. Revisit only with a borrower that keeps its own
rotary key (zero-shot cost roughly halved) and a longer repair.

The control arm is itself informative: 3M more tokens at a decaying rate took MMLU from
0.5508 to 0.5684 (the source is 0.5762) and in-domain NLL from 0.7214 to 0.7131, while
WikiText moved from 2.5834 to 2.5959.

### Code benchmarks: an MBPP+ regression, and the hedging behind it (2026-09-25)

MBPP+ (378) and HumanEval+ (164) with EvalPlus, greedy, the chat template, first fenced
block extracted, executed in a no-network Docker sandbox (`code_bench/`; `robust_eval.py`
reproduces the earlier stock arm exactly, 187/378). Plus pass@1, paired against the
source, exact McNemar p:

| cap | bench | source | finish | control |
| --- | --- | --- | --- | --- |
| 768 | HumanEval+ | 43.3% | 37.8% (p 0.23) | 37.8% |
| 768 | MBPP+ | 47.6% | 41.0% (p 0.012) | -- |
| 2048 | HumanEval+ | 46.3% | 42.7% (p 0.43) | 40.2% (p 0.15) |
| 2048 | MBPP+ | 47.6% | **38.6% (p 0.001)** | **36.5% (p < 0.001)** |

The distilled models write far longer answers (MBPP+ at 768: 151 truncations against 57)
and hedge far more: 8-9 "wait / actually / let me re-read" per 1000 words where they do
write code, against 0.8-1.7 for the source. Where both models wrote code within 768
tokens HumanEval+ is level (52.1% vs 51.3%); MBPP+ is not (52.3% vs 58.2%). A 2048-token
budget does not rescue it: the MBPP+ answers still without code drop from 71 to 52, and
48 of those are degenerate repetition. Reading samples, the churn is unproductive
second-guessing -- an arithmetic slip, then doubting the test instead of the slip.

The teacher never generates, so hedging reaches the student two ways, both measured
(`code_bench/hedge_sources.py`). Text: GLM-5.2 agent traces in the chat corpus run at
12.6 hedges per 1000 words, the rest near zero. Teacher distribution: on code answers
that never hedge, the 27B puts 0.69% of every line start on "Wait / Actually / Hmm"
(0.02% on the Tulu chat), and KL carries that into a 2B that cannot resolve the doubt.
Two fixes are built: `--suppress-hedges` removes those openers from the teacher's
answer-region targets where the text does not hedge, and `rewrite_hedges.py` has the
teacher rewrite the 282 hedging reasoning blocks with tool calls and answers protected.

### The code regression was the thinking-mode format (2026-09-26)

Every assistant turn the students trained on opens on an empty, closed think block: then
the answer (code and chat captures) or a second block holding short reasoning (the 5M
corpus). A thinking-mode prompt opens the turn on `<think>\n` instead, a position they had
never trained on, and there they reason for ~1,200 tokens and fall into repetition. Two
other causes were ruled out: the corpus's reasoning is short (median 40 tokens, 3% past the
1024-token cap), and the teacher closes reasoning where the text does (P(`</think>`) 0.95).

Hedging was a real but smaller part. From the control, 3M tokens each: `nofix` against
`fix` (de-hedged corpus plus `--suppress-hedges`) takes the student's P(hedge opener) from
7.2x the source to 0.86x at no cost (WikiText +0.0006, MMLU 0.5645 -> 0.5684) and MBPP+ in
thinking mode from 39.9% to 41.5%, still 6 points under the source.

In non-thinking mode -- the format the model trained on -- the gap closes. Plus pass@1 at a
2048-token cap, both models in the same mode:

| bench | source | fix | discordant | p |
| --- | --- | --- | --- | --- |
| MBPP+ | 47.6% | **48.9%** | 46 / 41 | 0.67 |
| HumanEval+ | 44.5% | 40.9% | 16 / 22 | 0.42 |

Answers are short again (MBPP+ mean 98 tokens, 7 truncations). `think_first.py` remixes the
5M corpus into the native thinking format so a run trains on both conventions.

### Training on both formats: non-thinking parity, thinking mode still loops (2026-09-26)

`both`: the fix arm's recipe plus `teacher-cache-think-first` (the 5M corpus remixed into
the native thinking format, 4.4M tokens), from the control, 3M tokens. Plus pass@1 at 2048
tokens, paired against the source in the same mode:

| mode | bench | source | fix | both |
| --- | --- | --- | --- | --- |
| non-thinking | HumanEval+ | 44.5% | 40.9% | **45.7%** (p 0.87) |
| non-thinking | MBPP+ | 47.6% | 48.9% | **47.1%** (p 0.91) |
| thinking | HumanEval+ | 46.3% | 36.0% | 37.8% (p 0.08) |
| thinking | MBPP+ | 47.6% | 41.5% | 39.4% (p 0.003) |

Elsewhere `both` is level or better: P(hedge opener) 0.94x the source, WikiText 0.0047
better than fix [-0.0073, -0.0020], MMLU 0.5684, in-domain NLL 0.7188.

In non-thinking mode the model now matches the source on both benchmarks. In thinking mode
hedging is gone (0.3-1.1 markers per 1000 words) but answers still run long (HumanEval+
mean 1137 tokens against 544) and loop -- now as runaway enumeration: listing Collatz
terms or Fibonacci numbers, with wrong arithmetic, until the cap. Only ~660K of the 3M
tokens were thinking-format examples, and every measurement here is greedy, which Qwen's
guidance for thinking mode advises against for exactly this failure.

### Thinking-weighted finishing pass (2026-09-26)

A new thinking corpus in the native format (`thinking_corpus.py`): the dataset's
`sft_code`, `sft_math` and `sft_reasoning` configs, reasoning moved into
`reasoning_content` so the template places it (the double think block was a rendering
artifact: the dataset keeps `<think>` inside `content` and the template prepends its own
empty block), de-hedged, screened, and whole documents only within the 1024-token cap --
5,993 documents, 3.0M tokens, reasoning median 156 tokens. From `both`, 3M tokens at ~74%
thinking format (`think-first` + this corpus) with `expand-code` for non-thinking code:

| mode | bench | source | both | thinking pass |
| --- | --- | --- | --- | --- |
| thinking | HumanEval+ | 46.3% | 37.8% | **41.5%** (p 0.28) |
| thinking | MBPP+ | 47.6% | 39.4% | 40.2% (p 0.006) |
| non-thinking | HumanEval+ | 44.5% | 45.7% | 42.7% (p 0.72) |
| non-thinking | MBPP+ | 47.6% | 47.1% | 47.6% (p 1.00) |

HumanEval+ in thinking mode is now within noise of the source; answers are shorter (mean
797 tokens against 1137) and looping among answers with code halved. MBPP+ is not: its
failures are greedy repetition after a slip, and overthinking MBPP's under-specified tasks
("median length of a trapezium") against a single example test. The corpus under-represents
code reasoning -- 1,850 code documents were dropped as longer than the cap. WikiText
+0.0018 (n.s.), in-domain NLL 0.7106, MMLU 0.5605, P(hedge opener) 0.44x the source.

### Sampled thinking, math, and on-policy round 1 (2026-09-26/27)

Thinking mode with Qwen's recommended sampling (temperature 0.6, top-p 0.95, top-k 20),
three seeds, per-problem mean with a bootstrap interval: the thinking pass trails the
source by 3.7 points on HumanEval+ (39.4% against 43.1%, [-8.7, +1.4]) and 3.5 on MBPP+
(39.9% against 43.4%, [-6.7, -0.3]). The source was run through the official llama.cpp
b11205 binaries on a bf16 GGUF (`convert_hf_to_gguf.py --no-mtp`), about 11x faster than
HF `generate`.

Math (`math_bench/run_math.py`, last `\boxed{}` scored by `math_verify`):

| bench | mode | source | thinking pass |
| --- | --- | --- | --- |
| GSM8K | thinking, sampled | 76.0% | 70.7% (54 truncated against 13) |
| GSM8K | non-thinking, greedy | 75.1% | 41.7% (67.6% scored on the last number) |
| MATH-500 | thinking, sampled | 54.2% | 47.6% |
| MATH-500 | non-thinking, greedy | 54.8% | 40.8% |

Non-thinking GSM8K is mostly format: 525 replies end "The final answer is $26." instead of
boxing it (the source leaves 9 unboxed). All our math data was thinking-format, so nothing
showed a non-thinking reply following a format instruction. `math_curriculum.py` now gives
every item a random answer-format instruction (boxed, bare, "Answer:" line, or none for
prose) with a reply that follows it, 40% of items non-thinking.

The teacher's (Qwen3.8) chat template injects a system prompt -- "Reasoning effort is set
to xhigh ... consider plausible alternatives ..." -- whenever thinking is enabled, unless
`reasoning_effort="medium"`. Every corpus rendered with it carries that prompt
(`think-first`, `thinking-code-math`, `agent-tools`, the first curriculum); the student
template, the evaluations and llama.cpp serving never add it. Curriculum v2 is rendered at
medium.

On-policy round 1: 3,000 thinking-mode rollouts from the thinking pass (sampled, 512
prompt + 512 new tokens, 60% finished), captured by the teacher and trained on KL alone,
with curriculum v2 and replay of the earlier captures, 3M tokens:

| measure | source | thinking pass | round 1 |
| --- | --- | --- | --- |
| HumanEval+, thinking, sampled | 43.1% | 39.4% | 30.3% (-12.8 [-18.1, -7.7]) |
| MBPP+, thinking, sampled | 43.4% | 39.9% | 40.5% |
| GSM8K, thinking, sampled | 76.0% | 70.7% | 69.1% (184 truncated) |
| MATH-500, thinking, sampled | 54.2% | 47.6% | 44.2% (220 truncated) |
| arithmetic probe, bare answer | 70.5% | 59.2% | 52.6% |
| MMLU | 0.5762 | 0.5605 | 0.5742 |
| WikiText NLL vs thinking pass | -0.070 | 0 | +0.011 |

Round 1 is a regression and is not adopted: answers got longer (GSM8K mean 649 tokens
against 430 and the source's 207) and looped more. `onpolicy_signal.py` shows why. Read at
the student's own positions, the teacher endorses the loop: inside repeated stretches it
gives the student's repeated token ~0.90 and agrees with it as top-1 93% of the time, 0.86
at the loop's onset, with ~0.0001 on `</think>`. That is in-context copying; a teacher
conditioned on a looping prefix predicts more of it, so KL on looping rollouts trains the
loop in. Teacher-scored on-policy distillation cannot remove loops by itself -- the signal
has to come from outside the teacher's continuation (masking repeated positions,
unlikelihood on them, or training only on rollouts that finished cleanly).
### On-policy round 2 (2026-09-27)

Round 1's lesson applied: sort rollouts before training (`classify_rollouts.py`). From the
thinking pass, 6,283 prompts rendered as served (`rollout_prompts.py`: the injected xhigh
text deleted, 40% non-thinking, 3,000 GSM8K/MATH *train* problems with checkable
answers), 1,024 new tokens (83% finished). Clean -- finished, no loop, correct where
checkable -- 4,163, trained as ordinary documents; looping 448, trained KL-only with
unlikelihood on tokens of a 16-gram's third or later occurrence inside the thought (an
8-gram, second-occurrence rule flagged 39% of rollouts that finished cleanly: code copied
from the thought into the answer, restated equations) and no KL there; 1,672 cut or wrong
dropped. `--strip-effort-prompt` deletes the xhigh text from the student's side of every
replayed capture; the one target pointing into the deleted span becomes the actual next
token. Curriculum v2 (answer-format instructions, 40% non-thinking).

| measure | source | thinking pass | round 2 |
| --- | --- | --- | --- |
| GSM8K, thinking, sampled | 76.0% | 70.7% | **74.4%** (32 truncated against 54) |
| GSM8K, non-thinking, greedy | 75.1% | 41.7% | **67.2%** (72 unboxed against 525) |
| MATH-500, thinking, sampled | 54.2% | 47.6% | 45.6% |
| MATH-500, non-thinking, greedy | 54.8% | 40.8% | **50.0%** |
| HumanEval+, thinking, sampled | 43.1% | 39.4% | 34.1% (-8.9 [-13.8, -3.9]) |
| MBPP+, thinking, sampled | 43.4% | 39.9% | 39.8% |
| arithmetic probe (first number) | 70.5% | 59.2% | 39.9% |
| MMLU | 0.5762 | 0.5605 | 0.5664 |
| WikiText NLL vs thinking pass | -0.070 | 0 | +0.007 |

Math gains, code and arithmetic losses. Two causes, both in the recipe:

- HumanEval+ wrote code as often (144-148 of 164 per seed either way) but wrong more
  often, and thinking-mode hedge markers rose from 1.0-1.3 to 2.3-2.7 per 1000 words.
  "Clean" meant only finished and loop-free wherever no answer could be checked, so 2,746
  unverified rollouts -- mostly code, which the student passes ~40% of the time -- were
  trained with cross entropy, hedges and all.
- The probe's collapse is mostly instruction-following: asked for "only the number", the
  model now restates the problem first (an operand is the first number in 140 of 200
  products), because curriculum v2's non-thinking "only the answer" items showed working
  before the answer. Scored on the last number it is 52% against the thinking pass's 59%,
  still worse on subtraction, especially with a negative result (19 of 98 against 47):
  the curriculum never generated one, and the model answered 36 - 92 with "-92".

Round 3 keeps round 2's rollouts but trains only on those verified correct (plus the
looping ones under unlikelihood), and uses curriculum v3: bare replies where only the
answer is asked for, and negative-result subtraction.
### Round 3 and the replay control (2026-09-27)

Round 3 kept round 2's rollouts but trained only on the 1,413 verified correct (plus the
448 looping ones under unlikelihood), with curriculum v3: bare replies where only the
answer is asked for, and 30% negative-result subtraction. The control repeats the rounds'
continuation from the thinking pass -- same schedule, cap, stripping, hedge suppression,
3M tokens -- on the replayed captures alone, which no round had been compared against.

| measure | thinking pass | control | round 2 | round 3 |
| --- | --- | --- | --- | --- |
| HumanEval+, thinking, sampled | 39.4% | 37.4% | 34.1% | 30.3% |
| MBPP+, thinking, sampled | 39.9% | 37.8% | 39.8% | 38.6% |
| GSM8K, thinking, sampled | 70.7% | 74.8% | 74.4% | 70.7% |
| truncated at 2K (GSM8K / MATH-500) | 54 / 156 | 51 / 182 | 32 / 101 | 29 / 115 |
| GSM8K, non-thinking, greedy | 41.7% | 47.5% | 67.2% | 66.3% |
| MATH-500, non-thinking, greedy | 40.8% | 39.4% | 50.0% | 48.2% |
| arithmetic probe | 59.2% | -- | 39.9% | 48.6% |

- Round 2's GSM8K thinking gain is the continuation's, not the rollouts': the control
  reaches 74.8%.
- The rollouts with unlikelihood do cut looping: truncations fall ~40% against the control
  (GSM8K 51 to 32, MATH-500 182 to 101).
- The curriculum and non-thinking rollouts fix the non-thinking answer format: +20 points
  on GSM8K and +10 on MATH-500 against the control.
- Continuing costs HumanEval+ ~2 points (within noise); the rounds' additions cost 3-7
  more, and round 3 -- fewer code rollouts, since the unverified were mostly code -- lost
  the most. Code was 29% of the thinking pass's tokens and 16% of the rounds'. The lever
  is the mix, not unverified code: round 4 is round 2's rollouts with curriculum v3 and
  `--repeat ..\teacher-cache-expand-code=2`, putting code back near 27%.
### Round 4 and weight blending (2026-09-27)

Round 4 (round 2's rollouts, curriculum v3, `--repeat` expand-code x2, putting code back
near 27% of the mix) did not recover code: HumanEval+ 32.7%, MBPP+ 38.2%, GSM8K thinking
67.9%. Held-out code NLL still rose (0.733 to 0.790), as it did in the replay-only
control, so the rise comes from how the continuation is run rather than what it is fed.
A training-only ablation continuing the thinking pass's own recipe (`ablate_continuation.ps1`,
arm "plain") shows only a re-warm bump that recovers (0.773, 0.791, 0.776 at the end); the
arms isolating `--strip-effort-prompt` and the 1536 cap plus general text are pending.

Blending round 2 back toward the thinking pass (`merge_weights.py`: base + alpha x update,
alpha uniform or ramped by depth as in LiNeS) and screening on held-out code NLL and on
GSM8K/MATH *train* problems no rollout used (`merge_proxy.py`, so the benchmarks stay a
test): the ramps sit on the same code-for-format trade-off as uniform blends -- round 2's
code damage is not concentrated in the shallow layers -- and half the update keeps most
of the format fix. Of two finalists, the 0-to-0.7 ramp:

| measure | source | thinking pass | round 2 | blend 0-0.7 |
| --- | --- | --- | --- | --- |
| HumanEval+, thinking, sampled | 43.1% | 39.4% | 34.1% | 39.4% |
| MBPP+, thinking, sampled | 43.4% | 39.9% | 39.8% | 37.6% |
| GSM8K, thinking, sampled | 76.0% | 70.7% | 74.4% | 73.2% |
| GSM8K, non-thinking, greedy | 75.1% | 41.7% | 67.2% | 65.9% |
| MATH-500, thinking, sampled | 54.2% | 47.6% | 45.6% | **50.6%** |
| MATH-500, non-thinking, greedy | 54.8% | 40.8% | 50.0% | 46.6% |
| MMLU | 0.5762 | 0.5605 | 0.5664 | **0.5781** |
| arithmetic probe | 70.5% | 59.2% | 39.9% | 60.6% |
| WikiText NLL vs thinking pass | -0.070 | 0 | +0.007 | -0.005 |

The blend keeps the thinking pass's code, most of round 2's non-thinking format fix, and
is better than both parents on MATH-500 thinking, MMLU (level with the source), the probe
and WikiText. It keeps less of the loop fix: MATH-500 truncations are back at the thinking
pass's level (162), GSM8K's at 44 against 54. It is the best student so far.
### The effort prompt, round 5/5b, and the teacher's template (2026-09-27/28)

Continuation ablations from the thinking pass (`ablate_continuation.ps1`), held-out code
NLL on the whole expand-code eval split (`code_nll_full.py`, 292 documents): continuing the
thinking pass's own recipe +0.003, adding the 1536 cap and general text +0.005, adding
`--strip-effort-prompt` +0.057 (the control +0.053, round 2 +0.045, the round-2 blend
+0.018). Every capture rendered with the teacher's template carries the injected effort
text in thinking mode, including -- it turned out -- the 5m (5,521 documents), expand-code
(5,825) and expand-chat (4,107) corpora, which were rendered in thinking mode around
non-thinking replies (an empty think block). Served non-thinking prompts never carry it.
Stripping it everywhere breaks the thinking documents' mode marker; keeping it everywhere
(round 5) trains non-thinking code on a prompt serving never sends (served-format code NLL
0.765 -> 0.848). `--strip-effort-nonthinking` deletes it only where the final reply opens
with an empty think block, so every document reads as the template serves it.

The teacher's chat template is the student's plus that injection (and `reasoning_effort`
low/medium/xhigh); the student adopts it. The trainer had been saving the source model's
template over it (fixed: checkpoints keep their init's `chat_template.jinja`).

Round 5b: from the round-2 blend (teacher template), round-2 recipe with prompts rendered as
served (`rollout_prompts.py --keep-effort`), curriculum v4 (v3 at the template's default
effort), selective stripping; blended back 0-to-0.7 by depth:

| measure | source | thinking pass | round-2 blend (teacher template) | round-5b blend |
| --- | --- | --- | --- | --- |
| HumanEval+, thinking, sampled | 43.1% | 39.4% | 39.8% | 36.0% |
| MBPP+, thinking, sampled | 43.4% | 39.9% | 40.5% | 40.4% |
| HumanEval+ (plus), non-thinking, greedy | 44.5% | 42.7% | -- | 43.3% |
| MBPP+ (plus), non-thinking, greedy | 47.6% | 47.6% | -- | 47.4% |
| GSM8K, thinking, sampled | 76.0% | 70.7% | 72.2% | **76.9%** |
| GSM8K, non-thinking, greedy | 75.1% | 41.7% | 65.9% | 68.8% |
| MATH-500, thinking, sampled | 54.2% | 47.6% | 47.8% | 47.0% (186 truncated) |
| MATH-500, non-thinking, greedy | 54.8% | 40.8% | 46.6% | 47.0% |
| MMLU | 0.5762 | 0.5605 | 0.5781 | 0.5703 |
| arithmetic probe | 70.5% | 59.2% | 60.6% | 61.0% |

GSM8K thinking reaches the source; non-thinking code is at source parity (the 0.7 uniform
blend is above it: HumanEval+ 45.1%, MBPP+ 48.7%, MMLU 0.582). Thinking-mode code falls and
thinking runs long (GSM8K mean 494 tokens against the source's 207): the xhigh text asks for
"consider plausible alternatives", and the student learned it as its thinking style.
Reasoning effort at serving time (the round-5b blend, thinking, sampled; accuracy /
truncated / mean tokens): GSM8K xhigh 76.9% / 61 / 494, low 74.8% / 40 / 425, medium (no
system prompt) 75.6% / 49 / 492; MATH-500 xhigh 47.0% / 186 / 1177, low 48.6% / 169 / 1100,
medium 49.2% / 183 / 1172. The effort text moves length by 7-14% and accuracy within noise:
the long thinking is learned, not prompted -- plausibly from the teacher's targets, which
put little mass on closing the thought at any one position.
### Round 6: brevity by shortest-of-four (2026-09-28)

From the round-5b blend: four rollouts per served-format prompt, the shortest acceptable
one per prompt (`select_shortest.py`: median 233 tokens against 284 over every acceptable
rollout, 5,338 prompts) trained on cross entropy alone (`--ce-only-caches`: KL toward a
teacher that thinks at length would lengthen them again), 1,684 looping rollouts under
unlikelihood, replay with selective stripping; blended back 0-to-0.7 by depth. The trainer
now keeps the chat template it was initialized with.

| measure | source | round-5b blend | round-6 blend |
| --- | --- | --- | --- |
| HumanEval+ (plus), non-thinking, greedy | 44.5% | 43.3% | 44.5% |
| MBPP+ (plus), non-thinking, greedy | 47.6% | 47.4% | 47.6% |
| HumanEval+, thinking, sampled | 43.1% | 36.0% | 36.8% |
| MBPP+, thinking, sampled | 43.4% | 40.4% | 41.3% |
| GSM8K, thinking, sampled (mean tokens) | 76.0% (207) | 76.9% (494) | 75.1% (476) |
| GSM8K, non-thinking, greedy | 75.1% | 68.8% | 69.5% |
| MATH-500, thinking, sampled (truncated) | 54.2% (121) | 47.0% (186) | 49.8% (175) |
| MATH-500, non-thinking, greedy | 54.8% | 47.0% | 48.4% |
| MMLU / probe | 0.5762 / 70.5% | 0.5703 / 61.0% | 0.5684 / 60.5% |

Non-thinking code is at the source. The round-6 blend is the best student so far; thinking
is shorter but still more than twice the source's length.
### Round 7: DPO on the model's own loops, and task arithmetic (2026-09-29)

Pairs from the round-5b blend's own rollouts at served settings (`build_pairs.py`: round 6's
four samples per prompt plus a greedy pass): the prompt's shortest acceptable rollout over
its looping (1,214) or truncated (1,025) ones. Every negative is the model's own; on a
sample it rates its loops *more* likely per token than its chosen answers (-0.16 to -0.22
against -0.24). Reference log-probs precomputed (`ref_logprobs.py`); two pairs per step,
DPO (beta 0.1) plus cross entropy on the chosen side (0.2), each side right-padded to a
multiple of 256; replay as round 6. Over 12 smoke steps the margin rose 0.06 -> 1.52 with
the chosen log-prob steady. Alone, round 7 cut MATH train truncations 127 -> 82.

Rounds 6 and 7 share a base, so their updates add (`merge_weights.py --also`): base +
ramp(0, 0.7) x round 6 + a x round 7.

| measure | source | round-6 blend | + round 7 ramp 0-0.7 | + round 7 x 0.7 |
| --- | --- | --- | --- | --- |
| GSM8K, thinking (mean tokens, truncated) | 76.0% (207, 13) | 75.1% (476, 66) | 72.7% (370, 24) | 73.5% (333, 4) |
| GSM8K, non-thinking | 75.1% | 69.5% | 71.6% | 70.1% |
| MATH-500, thinking (mean tokens, truncated) | 54.2% (836, 121) | 49.8% (1131, 175) | 51.6% (963, 118) | 48.6% (951, 98) |
| MATH-500, non-thinking | 54.8% | 48.4% | 50.4% | 46.8% |
| HumanEval+ / MBPP+, thinking, sampled | 43.1% / 43.4% | 36.8% / 41.3% | 34.6% / 39.9% | 36.8% / 39.2% |
| HumanEval+ / MBPP+ (plus), non-thinking | 44.5% / 47.6% | 44.5% / 47.6% | 40.9% / 46.8% | 40.9% / 46.6% |
| MMLU / probe | 0.5762 / 70.5% | 0.5684 / 60.5% | 0.5742 / 58.2% | 0.5840 / 46.0% |

DPO on the model's own loops is the most effective loop fix so far: GSM8K thinking 22-30%
shorter with 64-94% fewer truncations, MATH-500 truncations down a third, and the ramp
combination has the best MATH-500 of any student in both modes. Code pays for it (non-
thinking HumanEval+ 44.5% -> 40.9%, p 0.38 but in the same direction everywhere): the pairs
come from math and general prompts, with nothing verified on the code side.
### Auditing the replay corpora; GSM8K / MATH-500 contamination (2026-09-29)

The on-policy rollouts were checked (math answers against references, code against tests),
but the replay corpora -- most of every round's tokens -- never were. `trace_audit.py`
reads the teacher's own view of each trace from the captures: mean agreement flags style
more than error, and even at the final answer the lowest-agreement NuminaMath traces
spot-checked were correct (formatting). `math_reference_check.py` matches each math trace's
question to its source set and compares final answers: MetaMathQA 2,352/2,352 agree,
Orca-Math 99.2%, NuminaMath (1.5's curated answers) 94.9% -- the 48 disagreements mostly
formatting ("B" against "\text{B: ...}"), a few real. 61 traces carry a Python block that
does not parse (26 Magicoder, 30 agent-trace fragments, 5 others). Of the code corpora,
self-oss-instruct is `bigcode/self-oss-instruct-sc2-exec-filter-50k`, execution-validated
by its authors; Magicoder OSS-Instruct is not (GPT-3.5 solutions, no execution check).

The GSM8K and MATH questions that matched no *train* split were from the test sets
(`math_contamination.py`): 187 "gsm8k" traces in thinking-code-math and 62 in think-first
are GSM8K test questions, 84 are MATH-500, and the round-5/6 rollout prompts cut from
those corpora carry them too -- 251 of 1,319 GSM8K test and 102 of 500 MATH-500 problems.
The benchmark screen predates GSM8K and MATH-500 joining the evaluation. Rescored on the
problems no training document contains (`clean_rescore.py`), every conclusion holds but
the students' standing drops about a point:

| measure (clean problems) | source | best student | gap, clean | gap, all |
| --- | --- | --- | --- | --- |
| GSM8K, thinking | 75.4% | 75.3% (round-5b blend) | -0.1 | +0.9 |
| GSM8K, non-thinking | 73.9% | 70.6% (round 6+7) | -3.3 | -3.5 |
| MATH-500, thinking | 53.5% | 49.7% (round 6+7) | -3.8 | -2.6 |
| MATH-500, non-thinking | 53.3% | 49.2% (round 6+7) | -4.1 | -4.4 |

Contaminated problems are easier for everyone (the source scores ~6 points higher on them),
so the inflation is modest. Future rounds exclude `exclude-master-v2.json` (the previous
master plus the 503 contaminated documents, the 61 unparsable-code traces and the 48
reference disagreements, 2,169 ids), and `eval_checkpoint.ps1` reports the clean scores.
### Round 8: verified code; the DPO failure mode on near-identical pairs (2026-09-29)

KodCode-V1 problems (pytest suites, benchmark similarity under 0.8; `code_prompts.py`)
joined the served-format prompts; code rollouts ran against their tests in a network-less,
mount-less sandbox (`verify_code.py`, `Dockerfile.verify`): 2,236 of 7,495 passed. Round 8
(from the round 6+7 combination) ran DPO on 2,880 pairs, two thirds of them wrong-vs-right
(failed tests, wrong math). It degraded the model broadly -- GSM8K non-thinking 74.6% ->
52.3% unblended, thinking NLL 0.560 -> 0.614 -- and its trace shows why: the margins stayed
near zero and went negative, the DPO loss spiked to 2.0, and the chosen answers' likelihood
fell with the rejected (-0.20 -> -0.44 per token), the likelihood displacement DPO shows on
near-identical pairs. Round 7's loop pairs separated at once (margins 8-58).

Round 8b, same base and rollouts: DPO on the 964 loop and truncation pairs only, and the
shortest test-passing solution of 1,085 KodCode problems as cross-entropy-only positives.
Blended 0-to-0.7 by depth (math on clean problems, `clean_rescore.py`):

| measure | source | round-6 blend | round 6+7 | round-8b blend |
| --- | --- | --- | --- | --- |
| GSM8K thinking, clean (mean tokens, truncated) | 75.4% (207, 13) | 73.2% (476, 66) | 71.5% (370, 24) | 74.5% (320, 3) |
| GSM8K non-thinking, clean | 73.9% | 68.1% | 70.6% | 68.5% |
| MATH-500 thinking, clean (mean tokens, truncated) | 53.5% (836, 121) | 47.7% (1131, 175) | 49.7% (963, 118) | 48.2% (873, 86) |
| MATH-500 non-thinking, clean | 53.3% | 47.5% | 49.2% | 49.7% |
| HumanEval+ / MBPP+, thinking, sampled | 43.1% / 43.4% | 36.8% / 41.3% | 34.6% / 39.9% | 37.6% / 44.4% |
| HumanEval+ / MBPP+ (plus), non-thinking | 44.5% / 47.6% | 44.5% / 47.6% | 40.9% / 46.8% | 40.9% / 46.6% |
| MMLU / probe | 0.5762 / 70.5% | 0.5684 / 60.5% | 0.5742 / 58.2% | 0.5781 / 59.5% |

Thinking mode is now near the source: MATH-500 truncations below the source's at similar
length, 3 truncated GSM8K answers, MBPP+ thinking above the source for the first time. Non-
thinking code stays where round 7 left it; the round-6 blend keeps source parity there.
DPO belongs on pairs that separate cleanly (loops, truncation); correctness comes better
from the model's own verified solutions as plain positives.
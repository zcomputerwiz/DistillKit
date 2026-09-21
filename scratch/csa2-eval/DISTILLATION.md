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

**The replicated norms trained on a fraction of their gradient.** A parameter replicated
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

So `chatkd` is not a measurement of what chat calibration is worth after training. It is
a measurement of a run whose replicated norms trained on a fraction and whose gradients
were then over-clipped. The comparison has to be redone before anything is concluded
about the calibration.

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

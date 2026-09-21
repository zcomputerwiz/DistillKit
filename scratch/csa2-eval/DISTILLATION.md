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

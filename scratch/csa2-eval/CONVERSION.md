# What the conversion actually costs (2026-09-21)

The distillation arms left 6.25 points of MMLU between the converted model and its source,
and placed that loss in the conversion rather than in anything trained afterwards. This
takes the conversion apart to find which piece carries it.

Four checkpoints, each adding one thing to the one before. All were converted from
`student-2b-hf` within the same session, so they share a recipe and differ only in what
the table says.

| stage | checkpoint | mmlu (512) | nll all |
| --- | --- | --- | --- |
| source | `../student-2b-hf` | **0.5762** | **1.3713** |
| MLA 384 | `checkpoints-conv/student-2b-mla-fixed` | 0.4980 | 1.4817 |
| + CSA2 all-full | `checkpoints-conv/student-2b-nobias` | 0.4902 | 1.4823 |
| + indexer warm-up | `checkpoints-2b/warmed-scaled` | 0.4883 | 1.4817 |
| + distillation | `checkpoints-2b-kd/...` | 0.5137 | 0.7569 |

Paired bootstrap on MMLU, 10,000 resamples:

| comparison | estimate | 95% CI |
| --- | --- | --- |
| MLA only - source | -0.078125 | [-0.123047, -0.033203] |
| all-full - source | -0.085938 | [-0.130859, -0.041016] |
| all-full - MLA only | -0.007812 | [-0.021484, +0.003906] |

## CSA2 is free

The routing costs 0.8 points of MMLU with an interval spanning zero, and 0.0006 nats.
Adding the indexer warm-up on top moves NLL back by the same 0.0006. ARC does not move at
any stage, at any normalisation.

That is the whole block-sparse apparatus -- the lightning indexer, the top-k selection over
a shared latent bus, the borrow modes -- and on these two metrics it is not detectable.
Everything the conversion loses, MLA loses.

## MLA's loss is not a capacity problem

The obvious next move is a wider latent, and it does not work. The same conversion at four
widths:

| latent | mmlu (512) | nll all | nll control |
| --- | --- | --- | --- |
| 256 | 0.4727 | 1.5510 | 0.8487 |
| 384 | 0.4980 | 1.4817 | 0.7891 |
| 512 | 0.4902 | **1.4628** | 0.7518 |
| 768 | 0.4785 | 1.4716 | **0.7212** |

NLL improves steeply to 384, barely to 512, and then **gets worse at 768** -- 1.4716
against 512's 1.4628, with twice the latent. MMLU is flat across the whole range: the
2.5-point spread between the best and worst is inside what 512 questions can resolve, and
768 sits below 384.

A bottleneck that is too narrow gets better when widened. This one stops improving at
about 512 and then degrades, which is not what a capacity limit looks like. It is what a
fitting procedure looks like when it is not finding the solution its parameterisation
already admits -- at 768 the refit has strictly more room to reproduce the attention it
is replacing, and reproduces it slightly worse.

So the floor is around +0.09 nats and -8 MMLU points, and it belongs to how the projections
are fitted, not to how much room they are given.

Only `nll control` tracks width monotonically, 0.8487 down to 0.7212. That is the one
quantity a wider latent reliably buys, and it is the one the distillation corpus already
fixes outright.

## What this means for the resource goal

The cache saving comes from MLA, which is also the only thing paying for it -- but the two
are not on the same curve. Going from 384 to 768 doubles the cached width and buys nothing
on either metric. Going from 384 to 256 costs 0.069 nats, which is most of the total loss
again. So 384 sits at the knee and there is no accuracy to buy back by spending cache.

That is the useful form of the result. The trade is not "more cache, more accuracy" with
384 as a chosen point on it; the accuracy is lost somewhere else entirely, and the cache
budget can be set on cache grounds alone.

## The calibration was the problem (2026-09-21, later)

Everything above was measured on conversions calibrated from
`bigcode/the-stack-v2-dedup`, config Python, at `calibration_tokens = 32768` -- 32 windows
of source code, solving every MLA and CSA2 projection in a 1.9B model, then judged on
MMLU's 57 academic subjects. The corpus was a default nobody had revisited and the token
count was a memory ceiling: `source_capture` held every token's inputs, keys, values and
rotary key in float64, about 1.6 GiB per layer.

Both were changed, separately, so they could be told apart. `chat_store.py` writes the
capture corpus into the same layout; `calibration.py` accumulates second moments instead
of samples, which is fixed-size in the token count and was verified to reproduce the
sample path at 32 windows (1.4955 against a recorded 1.495990).

| arm | corpus | calibration tokens | mmlu (512) | nll all | nll control |
| --- | --- | --- | --- | --- | --- |
| source | -- | -- | **0.5762** | **1.3713** | **0.6327** |
| baseline | Python | 32,768 | 0.4902 | 1.4823 | 0.8060 |
| chat/32K | chat | 32,768 | **0.5469** | 1.4823 | 0.7507 |
| chat/256K | chat | 262,144 | 0.5293 | **1.4546** | 0.7406 |

Paired bootstrap on MMLU:

| comparison | estimate | 95% CI |
| --- | --- | --- |
| chat/32K - source | -0.029297 | [-0.068359, +0.009766] |
| chat/32K - Python/32K | +0.056641 | [+0.025391, +0.089844] |
| chat/256K - source | -0.046875 | [-0.085938, -0.007812] |
| chat/256K - Python/32K | +0.039062 | [+0.003906, +0.074219] |
| chat/256K - chat/32K | -0.017578 | [-0.041016, +0.003906] |

**Changing the corpus alone recovers most of the conversion's MMLU loss.** At the same
32,768 tokens, calibrating on chat instead of Python is worth +5.7 points with an interval
clear of zero, and leaves the converted model -2.9 points from the source with an interval
that spans it. The 8.8-point gap this file opened with was not a property of MLA. It was a
property of fitting MLA to Python.

**The two levers are independent, and they move different metrics.** The corpus fixes MMLU
and does nothing at all for NLL -- 1.4823 for both 32K arms, to four figures. Eight times
the calibration fixes NLL, 1.4823 to 1.4546, and does not help MMLU; if anything it costs
1.8 points, though that interval spans zero. Control tokens improve under both.

**What this retracts.** The section above concludes that MLA's loss "is not a capacity
problem", on the evidence that a 768-wide latent fit worse than a 512-wide one. Every
point on that ladder was calibrated from Python at 32K. The reasoning still stands as
stated -- more room fitting worse is a solve outrunning its sample -- but the ladder needs
remeasuring under a calibration that is not itself the dominant error before anything is
concluded about capacity.

## Caveats

* All four latent checkpoints came from one recipe within eight minutes. That is what
  makes the comparison clean, and it also means a systematic flaw in that recipe is common
  to all of them. A flat curve is consistent with such a flaw rather than evidence against
  one.
* The 512-question MMLU set is the `screen` split extended from 256, not an independent
  test. `confirmation` remains unspent.
* MMLU at 512 resolves differences of roughly 4 points. Every latent-to-latent difference
  here is smaller than that, which is why the conclusion rests on NLL, where 175,526
  scored tokens separate 1.4628 from 1.4716.

## Next

The lever is the conversion procedure. Worth trying, cheapest first:

* Fit the projections against the source's attention *outputs* rather than its keys and
  values -- the current refit matches an intermediate and inherits whatever error that
  leaves downstream.
* Fit jointly across layers rather than layer by layer, so a layer's error is not handed
  to the next one as if it were the source's own output.
* Gradient-descend the projections after the least-squares initialisation, on the
  conversion corpus, before any language-model training starts.

A wider latent is not on that list, and neither is more training after the fact: 10M tokens
of distillation moved MMLU by +2.5 points with the interval spanning zero.

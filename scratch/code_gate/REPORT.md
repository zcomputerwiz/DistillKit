# Code-domain gate and structural sidecar on frozen B_code

Both mechanisms refitted to Python against the frozen code-trained backbone. **Neither
passes its promotion criterion, so no composition was evaluated and nothing was
promoted.** No MBPP+, no retraining, no architecture change.

| | |
| --- | --- |
| tests | 935 |
| evaluation subset | 1,334 documents, 1,250,215 tokens, digest `0436eba10eeb2fdb` — the same frozen subset as the previous task |
| backbone | `B_code`, frozen bitwise; digest identical before and after every fit |
| general `G'` | `f393c838…` unchanged | 
| general `S` | `a5488dbb…` / `375ea641…` unchanged |

| module | trainable parameters | training tokens | optimizer | LR | schedule | wall clock | peak VRAM |
| --- | ---: | ---: | --- | ---: | --- | ---: | ---: |
| `F_py` cache | — | 29,017,403 captured | — | — | — | 3,274 s | — |
| `G_code` | 260 | 784,896 | AdamW | 3e-3 | constant, 1,536 steps | 364 s | 15.8 GiB |
| `S_code` addressed | 400,790 | 512 docs × 512 tok × 3 epochs | AdamW | 3e-3 | — | ~500 s | — |
| `S_code` whitespace | 485 + 33 | 512 docs × 512 tok × 3 epochs | AdamW | 3e-2 | — | 452 s | — |

Selection criterion for `G_code`: **calibration content NLL, aggregate as guardrail**.
Selection for `S_code` strength: the canonical calibration procedure — see the defect
below. Heldout entered nothing but the final scoring.

## A1 — cache coverage

Built from the Python train split only, canonical definition throughout: exact
`(t-2,t-1,t)` key, `min_count ≥ 2`, cross-document recurrence ≥ 2, `max_keys = 400,000`
(the reference default). 34,606 documents, 30,760,040 tokens, 9,406,439 distinct trigrams,
400,000 cached keys with counts 5–71,972.

| bucket | `F_general` | `F_py` |
| --- | ---: | ---: |
| **[0, 1) unseen** | **83.6%** | **54.2%** |
| [1, 4) | 2.2% | 0.0% |
| [4, 100) | 11.0% | 17.1% |
| [100, 400) | 2.0% | 9.3% |
| [400, 800) | 0.7% | 3.8% |
| [800, 3000) | 0.4% | 5.9% |
| [3000, ∞) | 0.0% | 9.7% |

The suspected cause was real: familiarity signal rises from 16.4% of Python targets to
45.8%. The empty `[1,4)` bucket is the 400,000-key cap biting on a corpus with 9.4M
eligible trigrams — every surviving Python key is high-count.

One deviation, stated: prototypes were captured from **B_code**, not B0. The variance
feature is a claim about how consistent *this* backbone's FFN output is for a context; the
count feature, which the coverage diagnostic is about, is a corpus property and identical
either way.

## A2/A3 — gate results

| Arm | Aggregate | Content | Keyword | Operator | Delimiter | Newline | Whitespace | Other punct |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `B_code` | 0.9909 | 1.3572 | 0.9674 | 0.8908 | 0.4351 | 0.5536 | 0.2892 | 0.6786 |
| `+ G'(F_general)` | +0.00092 | +0.0012 | +0.0021 | +0.0009 | −0.0002 | +0.0007 | +0.0006 | +0.0003 |
| `+ G'(F_py)` | +0.00493 | +0.0058 | +0.0163 | +0.0041 | +0.0014 | +0.0016 | +0.0039 | +0.0025 |
| `+ G_code(F_py)` | **+0.00024** | +0.0004 | +0.0000 | −0.0001 | +0.0001 | +0.0000 | +0.0000 | +0.0001 |

Historical classes, deltas against `B_code`: `G_code` content +0.0004, newline +0.0000,
whitespace +0.0000, punctuation +0.0001, control +0.0052.

Paired: `+G'(F_py)` is +0.00580 ± 0.000267 (t = +21.7). `+G_code` is +0.00019 ± 0.000082
(t = +2.4); content +0.00030 ± 0.000122 (t = +2.5).

### The cache was never the problem

Giving `G'` accurate Python familiarity makes it **five times worse**, and not because the
features go off-scale:

```
G' normalizer (fitted on general text):   mean 3.0812   std 3.0862
general cache on Python:  raw mean 0.5171  ->  standardized -0.831
python  cache on Python:  raw mean 2.6030  ->  standardized -0.155
standardized features beyond 4 sigma:     0.0000 under both caches
```

`F_py` lands the features **closer** to the normalizer's operating point, not further.
Section 10's precondition — "grossly outside the original normalizer's operating range" —
is therefore not met, and the domain-normalized arm was **deliberately not run**: it would
answer a question the data has closed. The 84.4%-unseen diagnostic was a true observation
about coverage that turned out not to be the cause. The learned familiarity→admission
mapping is domain-specific in itself, and applying it confidently with correct statistics
is worse than applying it blindly with none — worst on keywords (+0.0163).

### G_code found no policy

| bucket | mean gate | share |
|---|---:|---:|
| [0, 1) | 1.0086 | 54.4% |
| [4, 100) | 1.0475 | 17.8% |
| [3000, ∞) | 1.0387 | 9.8% |
| [100, 400) | 1.0452 | 8.4% |
| [800, 3000) | 1.0437 | 6.0% |
| [400, 800) | 1.0454 | 3.7% |

Reach by layer 0.241 / 0.197 / 0.104 / 0.078, against **0.80 / 1.00 / 1.00 / 0.96** for
`G'` on general text. Across 1,536 steps reach wandered between 0.03 and 0.26 and ended
where it began, while calibration content moved 0.00066 nats in total — the six
checkpoints order differently on content than on aggregate, which is what noise looks
like. The learned policy is a near-constant slight *opening* (spread 0.04 across all
buckets), the opposite sign from `G'`, which attenuated familiar contexts.

Identity at initialization was exactly `0.000e+00`; the backbone digest matched before and
after. The fit was not broken — there was nothing to find.

## B — structural results

Structural vocabulary coverage on Python, before fitting: the canonical 6,166-token set
(2.48% of vocabulary) reaches **100% of operator, delimiter, newline, whitespace,
other_punct and control targets**, 41.7% of all targets. `S_general`'s failure was not a
coverage failure.

| Arm | Aggregate | Content | Keyword | Operator | Delimiter | Newline | Whitespace | Other punct | Control |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `B_code` | 0.9909 | 1.3572 | 0.9674 | 0.8908 | 0.4351 | 0.5536 | 0.2892 | 0.6786 | 1.8782 |
| `+ S_general` | +0.02985 | +0.0095 | +0.0143 | +0.0093 | +0.0169 | +0.1826 | +0.0294 | +0.0318 | −0.0715 |
| `+ S_code` | **+0.00751** | +0.0048 | +0.0067 | +0.0184 | +0.0047 | +0.0250 | **−0.0160** | +0.0274 | **−0.3860** |

Paired: `S_general` +0.03690 ± 0.000782 (t = +47.2); `S_code` +0.00738 ± 0.000419
(t = +17.6). Evaluated at the fitted whitespace strength 1.0.

A fresh Python fit removes **75% of the damage** and turns two classes genuinely positive
— whitespace −0.0160 where `S_general` cost +0.0294, and control −0.3860. Newline improves
sevenfold (+0.1826 → +0.0250) but stays a regression. Operator gets worse (+0.0184). The
composite remains net harmful, so it fails §37's promotion bar: no aggregate improvement,
mixed structural movement, and content still damaged (+0.0048).

### A defect in the canonical strength selector, found on this corpus

The selector minimizes an **unweighted sum of per-class mean deltas** over structural
classes. On Python that is degenerate:

| strength | unweighted sum (what the selector reads) | token-weighted |
|---|---:|---:|
| 0.0 | +0.000000 | +0.000000 |
| 0.5 | −0.276435 | +0.001810 |
| 1.0 | −0.503981 | +0.007344 |
| **1.5 (selected)** | **−0.685565** | **+0.016844** |
| 3.0 | −0.987934 | +0.072424 |

Control is **0.26%** of structural targets — one EOS per document, 1,334 tokens — and is
hugely improvable, so it dominates a sum that weights it equally with punctuation's 60.32%.
The selector reads "monotonically better" while the token-weighted structural NLL gets
monotonically worse; the optimum on the classes carrying the tokens is strength 0.0, the
sidecar switched off. This is the same shape as the §25 guardrail bug, relocated into the
selector. **It does not change the verdict** — the final evaluation used strength 1.0 from
the fitted whitespace checkpoint, and every strength is a token-weighted regression — but
the canonical procedure should not be reused on a corpus with a rare trivially-improvable
class without reweighting.

The wrong-address control was not run on Python: §32 conditions it on `S_code` being
positive, and it is not.

## D — general-domain specialization

Available as a by-product: the Python-fitted sidecar scored on the established
general-text screen and confirmation splits is harmful there, as a specialized module
should be.

| arm | content | newline | whitespace | punctuation | control | all |
|---|---:|---:|---:|---:|---:|---:|
| addressed only | −0.00061 | +0.01232 | −0.01310 | +0.01909 | +0.14932 | +0.00575 |
| static whitespace | +0.00688 | +0.01701 | +0.12479 | +0.02378 | +0.12100 | +0.01525 |
| context-gated | +0.00172 | +0.01372 | −0.00171 | +0.02061 | +0.14134 | +0.00773 |
| gated, wrong context | +0.00244 | +0.02432 | −0.00967 | +0.01327 | +0.15775 | +0.00753 |

On general text the context gate does its job — it recovers whitespace from +0.1248 static
to −0.0017 — which is the mechanism working, on the wrong corpus. Not required by the task
(neither module passed), reported because it was produced.

## Verdicts

**Gate — `NO CODE-DOMAIN RESIDUAL-GATE ADVANTAGE`.** All three configurations agree.
The cache swap eliminates conditioning statistics as the cause; the fresh fit, with correct
statistics, correct normalization, an exact identity start and the canonical regime, finds
a flat policy worth +0.0002. `B_code` has absorbed the residual-admission headroom.

**Structural — `PARTIAL STRUCTURAL TRANSFER`.** The architecture demonstrably reaches
Python (100% class coverage) and fresh parameters recover three quarters of `S_general`'s
damage, turning whitespace and control positive. But the composite is still +0.0075 on
aggregate with content damaged, so it fails promotion. The mechanism transfers; the
parameters are domain-specific; what remains after code pretraining is not enough to pay
for itself.

**Composition — not evaluated.** §39: neither module independently useful, so no
composition and no architecture expansion.

## Caveats

- The addressed decoder is fitted on 512 documents × 512 tokens ≈ 262k tokens for 400,790
  parameters — under one token per parameter. Train loss fell (0.734 → 0.682) while
  calibration structural classes worsened, which is consistent with overfitting. This is
  the canonical procedure and was followed as specified; the corpus has 30.7M tokens
  available if a larger fit is ever wanted.
- `B_code`'s structural NLLs are already low (newline 0.5536, whitespace 0.2892), so there
  is little headroom for a hash-addressed bias to add beyond noise.
- `B_code` carries the AdamW-state reset from its interrupted training. Acceptable for
  frozen-backbone testing, and not the definitive control for a future matched-training
  comparison.

## Stop point

Stopped as instructed. No MBPP+/HumanEval+, no backbone retraining, no joint training, no
architecture expansion, nothing promoted.

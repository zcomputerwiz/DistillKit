# Row novelty and frequency stratification of the C and B arms

Reproduce with:

```
python scratch/row_novelty.py seen
python scratch/row_novelty.py score --arm win-C1-L1-real-stage1-1m  --checkpoint ../runs/win-C1-L1-real-stage1-1m
python scratch/row_novelty.py score --arm win-C2-L1-shuffled-stage1-1m --checkpoint ../runs/win-C2-L1-shuffled-stage1-1m
python scratch/row_novelty.py report
python scratch/row_novelty.py frequency
python scratch/row_novelty.py layout
```

384 held-out documents, 156,565 assistant tokens, paired bootstrap over documents.
The re-score reproduces the stored reply bundles exactly (C1 enabled 78133.132 /
bypassed 78504.459), so these strata partition the same numbers the published
C1 result was computed from.

## Provenance of the axis

`seen` replays the 1M teacher cache's 1,152 training documents (965,374 tokens)
through the real-context hasher: 15,445,984 row touches, **4,045,989 distinct rows,
1.264% of the 320,001,536-row table**. `k` for a scored position is how many of its
16 rows are in that set. Both arms are evaluated on real rows — the roll is a
training-time setting — so `k` is the same axis for each, always computed from C1's
exposure.

`k` is near-trimodal because the 8 bigram heads share one address and the 8 trigram
heads share another: k=16 is "bigram and trigram both seen" (50.9%), k=8 is "bigram
seen, trigram new" (19.6%), k=0 is "neither seen" (21.6%). The off-cluster values
(k=1,2,3,9,10,11) are hash collisions.

## C1 vs C2 by row novelty

| k | tokens | C1 cost | C2 cost | G = C1 − C2 |
|---|---|---|---|---|
| 0 | 33,863 | +0.002986 | +0.001483 | **+0.001503 [+0.000952, +0.002087]** |
| 8 | 30,703 | +0.001423 | +0.000901 | +0.000522 [−0.000517, +0.001522] |
| 16 | 79,691 | −0.007017 | −0.000270 | −0.006747 [−0.007557, −0.005965] |

Cost is `enabled − bypassed`; negative is better. The whole C1 win sits in k=16
(−559 nats of the −371 net; every other stratum contributes +188). G₀ is positive
with a CI excluding zero: on fully novel addresses the real-trained sidecar is
**worse** than the content-free control.

## …but k is a frequency proxy

k=16 positions are intrinsically easier before the sidecar does anything (bypassed
NLL 0.475 against 0.539 at k=0). Binning by how often a position's bigram occurs
across the held-out set — a measure independent of the training corpus and of any
model — and comparing exposure within a bin:

| bigram occurrences | k=0 cost | k=16 cost |
|---|---|---|
| 1 | +0.003691 | +0.001236 |
| 2 | +0.002495 | +0.002795 |
| 3–5 | +0.001217 | +0.003340 |
| 6–20 | +0.000938 | +0.003679 |

Exposure buys nothing at matched frequency; k=0 is *better* in three of the four
comparable bins. Frequency alone:

| bigram occurrences | tokens | C1 cost | C2 cost | C1 − C2 |
|---|---|---|---|---|
| 1 | 35,969 | +0.003580 | +0.001718 | +0.001862 |
| 2 | 16,266 | +0.003076 | +0.001357 | +0.001719 |
| 3–5 | 20,934 | +0.002777 | +0.001220 | +0.001557 |
| 6–20 | 26,165 | +0.002958 | +0.001661 | +0.001297 |
| 21–100 | 25,645 | +0.003989 | +0.001237 | +0.002752 |
| 101+ | 31,586 | **−0.024946** | −0.003475 | **−0.021472** |

The entire benefit is the 101+ bin (−788 nats against +417 from all the others,
summing to the observed −371).

## What the 101+ bin is

Ranking the bin's 2,845 distinct target token types by nats contributed:

| token | count | nats | share of bin |
|---|---|---|---|
| `\n` (198) | 1,509 | −708.0 | 89.9% |
| `<think>` (248068) | 414 | −201.6 | 25.6% |
| everything else (2,843 types) | 29,663 | +121.6 | — |

`role_spans` strips only the *leading* empty think block; 83% of assistant turns open
a second one, so `<think>` is scored as assistant content.

Splitting all 156,565 assistant tokens on `\n`, `<think>` and `</think>`
(`python scratch/row_novelty.py layout`):

| | tokens | C1 | C2 | C1 − C2 |
|---|---|---|---|---|
| layout | 8,647 (5.5%) | −0.103020 | −0.015380 | −0.087640 [−0.095046, −0.080726] |
| content | 147,918 (94.5%) | **+0.003512** | +0.001405 | **+0.002107 [+0.001672, +0.002539]** |
| all | 156,565 | −0.002372 | +0.000478 | −0.002850 |

The layer-1 sidecar reads content, strongly and demonstrably — and what it has learned
to supply is line-break and think-block placement. On the other 94.5% of assistant
tokens it costs 0.0035 nats, and the real-trained arm costs **more** than the
content-free control. The published −0.002372 is a 5.5% layout win outweighing a 94.5%
content loss. **Content-only is the grade from here on.**

## B arms (layer 24) for contrast

Every stratum's real−shuffled gap spans zero, by novelty (G₀ = +0.000182
[−0.000182, +0.000545]) and by frequency (all six bins span zero). B1's uniform
−0.0012 is capacity, not content, at every novelty level and every frequency — which
is what B1 ≈ B2 already said, now with the mechanism ruled out rather than inferred.

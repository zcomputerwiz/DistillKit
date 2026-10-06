# Streaming shared head experiment

Assisted-by: Codex

## Execution change

The benchmark-only streaming head gathers active rows, prepares each chunk's
CE/KL/UL membership once, and performs a vocabulary projection followed
immediately by that chunk's backward. It releases the large logits and loss
intermediates before the next chunk. Head gradients accumulate in the existing
parameter gradient; hidden gradients are scattered to their original positions.
The body graph is traversed once using the completed hidden gradient.

The existing path reconstructs each head projection during checkpointed
backward. Streaming eliminates that repeated projection, while preserving the
shared BF16 logits and float32 loss arithmetic. It replaces per-chunk scalar
GPU-to-host membership checks with one membership transfer and prepared chunk
index lists. It preserves loss masks, teacher targets and accumulation weights.

Only scratch helpers are changed. Ordinary training defaults and all DistillKit
model source are unchanged. Ordinary training can opt in with
--streaming-head-loss and --checkpoint-selection-cache, alongside
--shared-head-loss and --checkpoint-layers. Streaming excludes sparse-stage;
the selection cache supports Full layers only. The benchmark trainer flag is
--benchmark-streaming-head; tp_bench.py exposes --streaming-head. These diagnostics use frozen
records, fresh outputs, two optimizer warmups and no checkpoint writes.
Streaming requires the shared head and excludes sparse-stage indexer objectives.
The per-frame selection cache can be combined independently.

## Numerical checks

The initial shared/streaming/profiler CPU suite passed 31 tests. Additional
actual-hybrid tests passed with streaming and selection caching combined.
They cover weighted answers, prompt/padding masks, accumulated records,
KL-only loop UL, CE-only records, context-only KL and teacher weights 0/.5/1.
All actual-model parameter gradients match the CPU reference tolerances.

Actual u50 BF16, two-GPU tensor parallelism, one 512-token raw-code teacher
prefix, shared head chunk 64, all six attention layers:

| Measure | Ordinary vs ordinary | Streaming vs ordinary |
|---|---:|---:|
| Median per-parameter relative gradient L2 error | 0.8829% | 0.8833% |
| Maximum per-parameter relative gradient L2 error | 5.279% | 6.570% |
| Sign flips among nonzero gradient pairs | 0.1917% | 0.1969% |
| Sign flips excluding reference magnitude below 0.001 x tensor RMS | 0.2026% | 0.2044% |
| Prediction flips at original weights | 0/512 | 0/512 |
| Selected position sets | all six equal | all six equal |

All loss terms are equal on this sample. Sign flip frequency and median
relative error are close to the unchanged control, consistent with the user's
acceptance of small variations at similar frequency. This is not a long-run
training equivalence claim. Prediction equality before an update alone does
not establish that the updates are equivalent.

Sources: streaming-head-gradient-tp.json and streaming-head-control-tp.json.
The bounded --head-update-check diagnostic compares predictions after a fresh
KahanAdamW8bit update with the current context recipe's rates, including an
ordinary repeat. It writes diagnostic JSON only, not checkpoints.

## Matched 32K timing: streaming alone

All record hashes and the accumulation plan match the existing ordinary run.
There are two real optimizer warmups and three measured optimizer steps,
20283 original supervised targets and 65536 input tokens per step.

| Metric | Ordinary | Streaming |
|---|---:|---:|
| Measured supervised targets/s | 451.747 | 470.613 |
| Median step seconds | 44.425 | 43.137 |
| GPU0 peak allocated GiB | 15.6761 | 15.6754 |
| GPU1 peak allocated GiB | 15.4221 | 14.8057 |

The measured throughput gain is 4.18%; GPU1 saves 0.616 GiB at the whole-step
allocated peak. GPU0's allocated peak is unchanged. The maximum absolute loss
difference across all five optimizer steps is 0.000182 nats. These are short
fixed-record results, not proof of long-run equivalence or mixed-record speed.
Source: streaming-head-32k.json; ordinary source:
training-profile-context-32k-unprofiled.json.

The combined streaming + selection-cache run reaches 496.911 targets/s,
median step 40.767 seconds, a 10.00% throughput gain against that ordinary
baseline. Per-card allocated peaks are 15.6758 / 14.8057 GiB. Every step
computes 12 selections and reuses 12. The maximum five-step loss difference is
0.000403 nats. Source: streaming-head-cache-32k.json. The combination wins more
than either measured change alone; these short runs do not establish a precise
interaction beyond the observed whole-step benefit.

## Effects after an optimizer update

The combined streaming + selection-cache path was compared with the ordinary
path and an ordinary repeat, starting from identical u50 weights and fresh
Kahan optimizer states. All three use the context recipe's body/adapter rate
1.83e-6, gated-dynamics scale 0.1, router rate 9.37e-4, betas .9/.95,
weight decay .1 and global clipping at 1. Teacher targets are one real raw-code
prefix at width 512, KL-only, teacher weight .5; there is no checkpoint write.

| Post-update comparison | Ordinary repeat | Streaming + cache |
|---|---:|---:|
| Top-1 prediction differences vs ordinary baseline | 6/512 | 4/512 |
| Mean absolute logit difference | 0.029412 | 0.029725 |
| Maximum absolute logit difference | 0.734375 | 0.578125 |
| Different parameter elements | 1041705 | 1059745 |
| Maximum absolute parameter difference | 3.815e-6 | 3.815e-6 |

Training losses before the update are equal. The combined path does not show
more prediction flips than the unchanged-repeat control on this sample; the
average logit variation and parameter variation are similar. There are existing
post-update differences even in the ordinary repeat, so this result must not
be described as zero flips or bitwise equivalence. It satisfies the user's
practical criterion locally, not a statistical guarantee or long-run/resumed
optimizer-state validation. Source: streaming-head-update-tp.json.

## Real-record isolated head validation

The five-case suite uses real frozen hidden states from u50: agent 2048 tokens,
QA 32768 tokens, raw code 2048 tokens, math 8 x 256 and packed loops 5 x 384.
Head chunks are 1024 rows. Streaming vs shared has zero relative head-gradient
and hidden-gradient error on every case. Loss differences are zero except one
5.96e-8 scalar difference in the agent case. Loop UL is explicitly exercised.
The short QA attempt had no retained answers at 2048 tokens; the completed suite
uses an explicit --qa-budget 32768 rather than scoring question-only prefixes.
The failed short attempt is retained in streaming-head-five-cases.log.

Isolated head speed ratios shared/streaming are 1.403 agent, 1.371 QA,
1.360 code, 1.355 math and 1.376 loops, two timed repeats. These are head-phase
measurements, not whole-step gains. Additional allocator peak can be slightly
higher on sparse one-chunk records (QA +0.126 GiB because of the full hidden
gradient buffer), or lower for denser records (math -0.910 GiB, loops -0.946 GiB).
Memory savings are shape-dependent; use the full-step per-card peaks for capacity
decisions. Source: streaming-head-five-cases-complete.json.

## Larger chunks and practical settings

The normal opt-in flags were exercised with --head-chunk 1024 on the same frozen
32K pair. Throughput is 498.456 targets/s, only 0.311% above the combined 512-row
run. Median step time is 40.702 seconds, versus 40.767. GPU1 peak allocated
memory rises from 14.8057 to 17.1775 GiB; GPU0 remains 15.6765 GiB.
The marginal timing change is too small to establish a worthwhile speed gain
from these short runs. Retain 512 rows and preserve the 2.37 GiB of capacity
relative to 1024. Source: streaming-head-cache-1024-32k.json. That run started
before normal-mode cache counters were added to the report; its enabled
normal flags are recorded in run_args, while the 512-row benchmark records
all computed/reused counters explicitly.

Suggested execution flags, with the existing recipe and all objective settings
unchanged:

```
--shared-head-loss --streaming-head-loss --checkpoint-layers
--checkpoint-selection-cache --head-chunk 512
```

CPU verification: 23 streaming/cache/profiler tests passed (including the added
frozen-body case), plus 26 default
training-step/shared-head tests passed (four CUDA cases skipped in the explicit
CPU run). The subsequent real two-GPU optimizer/resume subset passed all six
selected tests, including those four CUDA cases. Source AST/ASCII checks passed.
No checkpoint, model source or training default was changed.

## Next opportunities and limits

The 512-row combination is ready as an opt-in execution change under the user's
practical variation criterion. It preserves the current objective. No new
long-budget arm or checkpoint was launched. The remaining CUDA stream warning
and failed profiler timeline are still unresolved; this experiment demonstrates
whole-step improvement without claiming a measured attribution of GPU idle gaps.

Further work should first examine the selector's remaining forward work and
targeted checkpoint retention using the actual per-card memory headroom. The
larger-chunk result shows that spending available memory is not itself a speed
win. Complete CCE KL/UL backward and vocabulary-sharded shared losses remain
separate larger projects. Existing vocabulary sharding should be extended rather
than replaced. Whole-microbatch overlap still needs explicit CSA2 bus and gradient
ownership and must not be inferred safe from the per-frame selection cache.

## Fresh ordinary control

The final ordinary repeat uses the same record hashes and plan and reaches
442.690 targets/s, median step 45.895 seconds, GPU0/GPU1 allocated peaks
15.6761/15.5464 GiB. The combined 512-row path is 12.25% faster against this
repeat, versus 10.00% against the initial ordinary run. Report approximately
10-12% on this frozen pair, rather than implying an exact population speedup.
GPU1 allocated saving is 0.62-0.74 GiB across these controls; GPU0 is unchanged.

The largest five-step loss difference between combined 512 and the fresh
ordinary repeat is 0.000229 nats, versus 0.000301 between ordinary repeats.
This supports the user's practical numerical criterion on the tested cycle.
The results do not replace long-run evaluation. Source:
streaming-head-baseline-repeat-32k.json; complete machine-readable comparison:
streaming-head-comparison.json. All diagnostic processes have exited and both
GPUs have been released.

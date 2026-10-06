# Assisted-by: Codex

# CCE normalizer diagnostic, 2026-10-05

Decision: keep the approved shared-head implementation for the controlled context-KL
run. An unfiltered CCE normalizer has useful potential at larger row counts, but the
current isolated selected/tail prototype is not ready for training. GPU1 was released
after the bounded diagnostics; no diagnostic Python process remains.

## Scope and reproducibility

Only physical GPU1 was visible (`CUDA_VISIBLE_DEVICES=1`, internal `cuda:0`). The
checkpoint was `scratch/dense_gr/merges-long1/u50`; its real final hidden states were
computed once per case, without body gradients. Teacher data and batching came from
the existing `head_parity.py` harness. Agent and QA cases enabled W=0.01/K=8 context
sampling through the existing teacher producer. Uniformly spaced active rows were
taken from each complete real batch: 1,024 for numerical diagnostics, 4,096 for
head-phase timing, and 128 for CCE NLL/LSE gradient sanity.

```
$env:CUDA_VISIBLE_DEVICES='1'
$env:HF_HUB_OFFLINE='1'
$env:PYTHONPATH=$PWD
$env:PYTHONIOENCODING='utf-8'
& .venv\Scripts\python.exe -B scratch\dense_gr\head_parity.py --checkpoint scratch\dense_gr\merges-long1\u50 --cce-diagnostic --cases 0,1,2 --budget 32768 --limit-rows 1024 --timing-rows 4096 --gradient-rows 128 --context-weight 0.01 --repeats 3 --output scratch\dense_gr\cce-diagnostic-20261005\larger.json
```

One warm-up and three timed repeats; synchronized median wall time and peak allocated
memory above each call's baseline. Timings exclude body forward, optimizer, transfer,
and fixture preparation. `larger.json` contains all samples and per-case measurements.
The earlier `initial.json` contains a 128-row raw-code probe. An initial attempt with
budget 2,048 was rejected by the existing retained-document budget check before any
head diagnostic; the retry used the required 32,768 budget.

## Head-phase measurements

| Real case, 4,096 active rows | Shared head forward/backward | Unfiltered CCE normalizer lower bound | Shared additional peak | CCE additional peak |
| --- | ---: | ---: | ---: | ---: |
| Agent, own turns plus sampled context | 523.7 ms | 377.8 ms | 3.336 GiB | 0.963 GiB |
| QA, answers plus sampled context | 520.4 ms | 377.7 ms | 3.336 GiB | 0.963 GiB |
| Raw code, KL only | 529.5 ms | 377.7 ms | 3.336 GiB | 0.963 GiB |

The CCE column computes a weighted NLL plus differentiable LSE with `filter_eps=None`.
It omits teacher-selected-logit capture/correction and is deliberately a lower bound,
not a complete CE/KL/UL implementation or an equivalent-loss benchmark. It suggests
about 27-29% potential head-time reduction before those additions, not an end-to-end
training speedup. The initial 128-row CCE lower bound was slower: 53.6 ms against
21.1 ms shared.

The experimental selected-logit forward takes about 81.5 ms at 4,096 rows. Its direct
omitted-tail forward takes about 2.8 seconds. These are fixed-tile diagnostic kernels,
not tuned performance conclusions about every possible CCE extension. The current
direct-tail forward is slower than the shared head's complete forward/backward at
this 4,096-row shape.

This is a decision about the present controlled-run implementation. It does not close
the investigation: usable memory savings could enable a faster checkpoint or execution
schedule even if the completed head kernel is slightly slower. `EXECUTION_OPTIONS.md`
records the memory limits and a bounded full-step comparison. `BUCKETED_FORWARD.md`
describes the new optional CPU-prepared direct-tail diagnostic; it has not run on GPU.

## Numerical findings

The reference projects BF16 operands to BF16 logits, then reduces those exact logits
in FP64. The selected logits captured from CCE-style tiles matched cuBLAS BF16 logits
exactly for all three 1,024-row samples. Thus different BF16 projection rounding was
not the source of the observed error in these samples.

Stock CCE's locked, incremental FP32 logaddexp differed from the reference LSE by up
to 6.48e-4 / 7.77e-4 / 7.05e-4 (agent / QA / raw). Combining that LSE with selected
logits produced student top-64 mass above one in 82 / 6 / 7 rows. Capturing selected
logits from the same tiles still left 76 / 8 / 6 oversummed rows. Maximum oversum was
1.10e-5. Same-tile capture alone therefore does not repair the tiny-tail problem.

The extension also exports per-tile maximum and sum-of-exponentials, then combines
them with a deterministic FP64 reduction. This produced no oversummed rows in the
three samples. Its grouped-tail KL maximum difference from the FP64 reference was
1.87e-6 / 2.40e-7 / 3.51e-7, with mean differences about 4-5e-8. This is sampled
evidence, not a guarantee for still smaller tails.

Directly accumulating the omitted vocabulary avoids subtracting nearly unit selected
mass from one. Its tail log-probability maximum error was 3.08e-7 / 2.38e-7 / 2.40e-7.
No invalid mass was clamped to hide these diagnostic failures. The existing KL's valid
probability clamp remains in the independent objective reference.

## Gradient sanity and remaining work

Public CCE's NLL and differentiable LSE were jointly compared against an ordinary
materialized BF16 projection on 128 real rows, with filtering disabled:

| Case | Loss difference | Hidden gradient cosine / relative L2 | Head gradient cosine / relative L2 |
| --- | ---: | ---: | ---: |
| Agent | -9.92e-5 | 0.999942 / 1.075% | 0.99999945 / 0.100% |
| QA | -1.80e-4 | 0.999770 / 2.147% | 0.99999912 / 0.138% |
| Raw | -1.00e-4 | 0.999461 / 3.288% | 0.99999937 / 0.115% |

This verifies that both CCE outputs participate in backward; it is not a gradient
validation of the forward-only selected/tail extension. CCE's default BF16 gradient
accumulation and the imperfect locked normalizer remain numerical differences.

A training candidate would need an efficient, consistent normalizer and selected/tail
capture, a fused unfiltered backward with teacher-ID corrections, loss and gradient
parity including clamp boundaries, and full-step memory/timing checks. FP32 gradient
accumulation is another option to measure. None of that is required before the
controlled context experiment, and no training implementation was changed.

One narrower precision candidate is FP32 accumulation of the hidden gradient only.
At 4,096 x2048 it uses about 32 MiB (16 MiB more than BF16), whereas widening the entire 508.6M-element head
gradient adds about 970 MiB. The observed 1-3% hidden-gradient error is larger than
the roughly 0.1% head-gradient error, so this is useful to isolate. It still requires
private backward buffer-dtype support and measurement; the forward-only extension
does not establish that it works or validate the complete KL gradient.

Only optional diagnostic flags were added to `head_parity.py`; the new isolated
`cce_diagnostics.py` and `cce_selected.py` do not modify the installed package.
Existing DistillKit model code, context training recipes, historical result files,
and the virtual environment were left unchanged. No commit or push was made.

## CPU-prepared bucket follow-up

The optional `--cce-bucketed` path sorts each row's teacher IDs into the projection's
128-column vocabulary tiles. Counts and starts are prepared on CPU, then uploaded
outside the timed forward. The kernel captures each selected logit from its original
projection tile and directly accumulates the omitted vocabulary. The original binary
search path remains available for comparison. Preparation/transfer time and actual
128-row CTA membership work are reported separately.

Adding `--cce-unlocked` records another bucketed forward without the contended locked
LSE accumulation. It still computes the stable full/tail reductions. Default locked
diagnostics remain unchanged; neither new variant has been compiled or timed on GPU.

Seven CPU-only tests passed, including original teacher-column order, partial tiles,
fully selected tiles, the uint8 prefix boundary, invalid/duplicate IDs, tiny valid tails,
and exact metadata size. They import neither Torch nor Triton and do not initialize
CUDA. At 4,096 rows, K=64 and vocabulary 248,320, metadata is 18,251,776 bytes
(17.4 MiB). GPU compilation, real-data numerical checks and timing are still pending.
The new option remains forward-only and cannot replace the training head.

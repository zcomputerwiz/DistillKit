# Assisted-by: Codex

# Optional bucketed direct-tail diagnostic

Implemented in `scratch/dense_gr/cce_selected.py`, selected with the diagnostic-only
`head_parity.py --cce-diagnostic --cce-bucketed` flags. No training or installed-package
code changed. GPU compilation/execution is pending; the seven CPU tests in
`test_buckets_cpu.py` pass.

Add `--cce-unlocked` to compare an additional bucketed forward that omits stock CCE's
contended locked incremental LSE. That work is unnecessary when the full/tail tile
statistics and stable final combination are the intended normalizer. The default
locked diagnostic remains intact for error comparisons. The unlocked option still
includes the stable full/tail reduction; it is forward-only, not a complete loss.

## Algorithm

For each compact active row, CPU preparation stably sorts its K unique teacher IDs.
It records a count and prefix start for every 128-column vocabulary tile, plus the
original teacher-column order. Tables are transposed to `[vocabulary tile, row]`, so
each 128-row GPU CTA loads contiguous count/start metadata. Duplicate, out-of-range,
noninteger or empty IDs are rejected. This diagnostic supports 1 <= K <= 255.

The existing BF16 projection and rounding are unchanged. A CTA loops only to its
largest per-row count for the vocabulary tile it already computed. Each valid row
loads that tile's local teacher IDs, stores selected logits in their original teacher
column, and excludes exactly those IDs from its omitted-vocabulary reduction. The
full and omitted maximum/sum-exp statistics therefore come from the same BF16 logits.
Their global reduction uses the existing FP64 diagnostic combine. It does not derive
tail mass by subtracting nearly unit selected mass from one. Existing teacher data,
active-row selection and valid-objective clamp remain the reference.

Empty CTAs do no membership loop. A row touches at most K vocabulary tiles, but a CTA
contains 128 different rows: their union can touch many more tiles. CPU preparation
reports nonempty row/tile pairs, nonempty CTAs, total CTA local-ID iterations, and the
largest local bucket. Those measurements will distinguish a useful algorithmic
reduction from a distribution that still requires substantial CTA work.

The original same-tile gather/binary-search variant remains the timing baseline.
The bucketed version is forward-only. There is no custom KL/UL backward, loss parity
claim, gradient claim or full-training speed claim for this option.

## Memory and preparation cost

For N rows, T=ceil(V/128) vocabulary tiles, and K teacher IDs:

| GPU bucket array | Bytes |
| --- | ---: |
| Sorted int64 teacher IDs | 8*N*K |
| Original uint8 teacher-column order | N*K |
| uint8 per-tile counts and starts | 2*N*T |
| Total additional bucket metadata | 9*N*K + 2*N*T |

At N=4096, V=248320, T=1940, K=64, this is 18,251,776 bytes (17.40625 MiB).
For comparison, a dense Boolean row/vocabulary membership array alone would be
1,017,118,720 bytes (970 MiB). Original teacher IDs remain in the caller, so the
sorted-ID copy is deliberately counted above rather than treated as free.

Existing forward temporaries are still needed: full and tail tile maximum/sum arrays
use 4*4*N*T bytes, 127,139,840 bytes (121.25 MiB) at the same shape. FP64 combination
also widens and reduces those arrays; the previous direct-tail forward's measured
additional allocator peak was about 0.299 GiB. That earlier measurement does not
include the new buckets, and a new allocator measurement is required. No N*V GPU
membership tensor is created. The head parameter/gradient is unaffected.

CPU preparation sorts O(N*K log K) IDs and computes O(N*T) prefix tables. During
preparation, row-major count/start arrays, transposed output copies, and the uint16
prefix-sum temporary coexist; output metadata size is not CPU peak memory. Partial
128-row groups require a small padded count copy for the CTA-work summary. CPU
preparation and upload are measured outside the timed GPU forward. In a real teacher
cache pipeline, preparation could happen before IDs are transferred; the diagnostic
also reports its otherwise unnecessary GPU-to-CPU download separately.

## Bounded next check after GPU release

Repeat the existing `larger.json` command with `--cce-bucketed --cce-unlocked` and a new output such
as `bucketed.json`. The harness keeps the same three real cases, 1,024 numerical rows,
4,096 timing rows, 128 public-CCE gradient sanity rows and W=0.01/K=8 context policy.
It reports bucketed selected-logit error against the baseline, direct-tail error
against the materialized same-BF16 FP64 reference, preparation/transfer time, bucket
memory and CTA work. Both timings include the full/tail FP64 combine. The default
bucketed timing retains the locked-LSE diagnostic machinery; the separately named
unlocked timing removes it. Neither is a projection-only timing.

GPU compilation and correctness come first. If this forward is fast enough to justify
the next stage, implement and validate a complete unfiltered backward using the same
teacher buckets for selected corrections and the omitted-tail coefficient. Validate
loss and both hidden/head gradients, including the reference clamp boundaries, before
putting that candidate into an actual full-step benchmark.

## GPU follow-up, 2026-10-05

`bucketed-context-fixed.json` runs the three original cases on physical GPU 1,
with W=0.01/K=8, 1,024 numerical rows and 4,096 timing rows. The initial GPU
compile exposed a row/column pointer-shape mismatch in selected-logit stores;
the corrected kernel reduces the single owned column before a row-shaped store.
Seven CPU bucket tests still pass. Failed compile logs are retained separately.

The unlocked bucketed forward takes 84.08 ms (agent), 83.97 ms (QA), and 85.37 ms
(raw code), including total/tail reduction. Selected logits match the baseline
exactly; maximum tail log-prob error against the materialized BF16/FP64 reference
is 2.96e-7. Additional forward allocator peak is 0.297 GiB, with 17.4 MiB of
bucket metadata at 4,096 rows. Agent CPU preparation costs 61.8 ms plus 3.8 ms
upload; this cost is outside the reported GPU forward and needs accounting or
overlap in any training implementation. The previous binary-search direct-tail
forward takes 2.93 seconds. This removes that forward bottleneck and justifies
investigating a complete backward; it establishes no full-objective training gain.

Memory headroom might permit retaining selected checkpoint activations or
prefetching peer restores. It must be measured across the complete two-GPU step:
this forward changes neither persistent head gradients nor optimizer state and
does not reduce the body GPU's peak by the same amount. `TRAINING_PROFILE.md`
describes the existing bounded full-step harness used for that measurement.

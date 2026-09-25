# Training-path repair and crash recovery — 2026-09-22

Based on `86209a2`. The machine crash did not discard the working changes or completed
benchmark reports. No long training job was running on recovery. Historical checkpoints,
metrics and failed/confounded verdicts remain intact. This is correctness validation,
not a new quality result or a speedup claim.

## Repairs

- **One actual step:** `training_step.py` is shared by `smoke_train.py`, `tp_bench.py`
  and the three-step integration tests. It includes causal CE, cached-teacher loss,
  indexer alignment, replica synchronization, unique-parameter clipping and Adam updates.
  Plain CE no longer depends on an import inside the teacher-only branch.
- **Same denominator:** all joint losses use the same `B*(L-1)` query positions.
  Microbatch means are weighted by their actual target counts. Joint indexer alignment
  excludes the final query; standalone indexer warm-up retains its existing objective.
- **Canonical prefixes:** `teacher_kl.py` fixes each document's capped, block-rounded
  width before filtering or grouping. Equal-width buckets retain remainder groups.
  Batch size cannot change a document's retained prefix or answer-filter decision.
  The existing conservative answer-count definition is unchanged.
- **Exact accounting:** `planned_tokens()` counts supervised targets, not input tokens.
  The trainer stops before the next whole microbatch would exceed the remaining budget;
  it reports unused targets. Shape warm-up uses the real objective and actual remainder
  shapes, with discarded warm-up backward targets reported separately in new runs.
- **Evaluation:** source-stratified sampling fills rounding shortfalls, remains stable
  under capture argument reordering, and reports source NLLs. Fixed-window evaluation
  also weights its remainder by target count.
- **Optimizer safety:** stale/missing/duplicate parameters are rejected. Gradients are
  cleared model-wide. Missing optimizer membership does not mean missing gradients:
  the historical body gradients could accumulate and influence clipping.
- **Recovery:** `training_state.py` saves weights, optimizer (including 8-bit state),
  data permutation/cursor, RNG states, step/target counts, arguments and parameter layout.
  `--save-every N` writes periodic step-boundary states; final model exports include a
  `training-state/` directory. Existing destinations and incomplete states are refused.
  Resume restores configuration and requires fresh output destinations. It validates
  model configuration, optimizer ordering and sample plan, ignoring only export metadata
  that `save_pretrained` itself adds; actual parameter dtypes remain checked.
- **Benchmark parity:** `tp_bench.py` now drives the trainer, not a separate workload.
  Timed updates synchronize both devices and follow at least two real optimizer steps.
  Invalid Windows spill readings are `null`/unknown, not "no spill".

## Verification

66 focused tests pass with both GPUs visible; Ruff and diff whitespace checks pass.
Tests exercise three actual optimizer updates with differently split microbatches,
CE-only and teacher/indexer objectives, CPU and two-GPU tensor parallelism, stale
optimizer rejection, budget boundaries, sample-plan invariance, and checkpoint recovery.
CPU and FP32/BF16 CUDA 8-bit-optimizer recovery fixtures match bit-for-bit.

The real merged captures retain **13,691 documents / 6,645,509 supervised targets** at
cap 1024, block 128, minimum-answer count 2. Prefix maps match for 1, 2 and 6 rows, and
for a 6,144-input-token forward budget. This guarantees equal retained data, not equal
optimizer trajectories when effective step boundaries differ. All three capture
manifests are complete (9,750,940 original capture tokens in total).

The pre-crash, corrected 1.915B two-card check completed five real optimizer steps:

| Measurement | Result |
| --- | ---: |
| Supervised targets, all five steps | 8,811 |
| Last three synchronized steps | 1,412.42 targets/s |
| Training elapsed / setup | 7.02 s / 18.95 s |
| Peak allocated, GPU 0 / GPU 1 | 14.16 / 5.14 GiB |
| Peak reserved, GPU 0 | 14.87 GiB |
| Windows spill status | Unknown |

This was `--micro-tokens 1024 --accumulate 2`, without layer checkpointing. It does not
validate a larger batch, equal-quality efficiency, sustained stability, or no spill.
The old 19.30 GiB number must not size a corrected run. The newly generated benchmark
JSON's invalid NaN spill readings were normalized to null; historical reports were not
edited. That report predates the added `warmup_backward_targets` metadata field.

After reboot, plain CE through the actual CLI completed, saved, resumed from step 1,
and reached step 3 / 762 targets. A separate uninterrupted run reached the same data
cursor, with final loss differing by 0.00000906 nat. Production BF16 weights are **not
bit-identical**: maximum difference 0.04584, RMSE 0.0001172. A second uninterrupted run
also differs (maximum 0.04578, RMSE 0.0001179; loss difference 0.00041866 nat). Thus a
recovery-specific regression was not isolated, but production bitwise reproducibility
is not established. Do not confuse the deterministic fixture tests with that stronger
claim; GPU numerical variation remains unlocalized.

## Reproduction

Run from the repository root with its existing local models and immutable data. Use
fresh output/checkpoint names on each rerun. No network access is required.

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_merged_cache.py tests/test_grouped_tail.py tests/test_dense_gr_training_step.py tests/test_optimizer_tp_guard.py tests/test_optimizers.py -q -p no:cacheprovider

.\.venv\Scripts\python.exe scratch/dense_gr/tp_bench.py --cards 2 --steps 3 --warmup-steps 2 --inherit --init-from scratch/dense_gr/checkpoints-2b/warmed-chat32 --teacher-cache ../teacher-cache-5m ../teacher-cache-expand-chat ../teacher-cache-expand-code --teacher-max-length 1024 --micro-tokens 1024 --accumulate 2 --sparse-stage --min-answer-tokens 2 --lr 7.3e-6 --output scratch/dense_gr/tp-bench-reproduce.json

.\.venv\Scripts\python.exe scratch/dense_gr/smoke_train.py --vocab 248320 --hidden 128 --layers 2 --batch 2 --length 128 --accumulate 2 --tokens 762 --max-steps 1 --warmup 0 --save-every 1 --checkpoints scratch/dense_gr/checkpoints-recovery-first --output scratch/dense_gr/recovery-first.json

.\.venv\Scripts\python.exe scratch/dense_gr/smoke_train.py --resume scratch/dense_gr/checkpoints-recovery-first/smoke-r1-1-nogr-s0/training-state --tokens 762 --max-steps 3 --checkpoints scratch/dense_gr/checkpoints-recovery-second --output scratch/dense_gr/recovery-second.json
```

`--tokens` and `--max-steps` on resume are total limits, not additional budgets. Resume
requires the same devices/partitioning, immutable source model and data, and the same
software environment. It does not reconstruct missing optimizer state for old runs.
Capture plans fingerprint manifests and groups; fixed-window plans validate shape, not
a content hash of the multi-gigabyte token store. Do not edit input data in place.

Raw results: `tp-bench-corrected-20260921.json`, `smoke-no-cache-corrected-20260921.json`,
`repair-resume-initial-20260922.json`, `repair-resumed-20260922.json`,
`repair-continuous-20260922.json`, `repair-continuous-repeat-20260922.json`.
`training-repair-20260922.json` contains the summary and checkpoint SHA-256 hashes.
Verification weights remain local and ignored by Git; reports and code remain reviewable.

Known warnings remain: the CUDA AccumulateGrad stream warning (no CUDA-graph claim),
unavailable optional attention backends, and the existing tokenizer regex warning.
One auxiliary trainer-integration test assumes a visible CUDA device and fails when
CUDA is explicitly hidden; the listed full suite passes with this machine's GPUs visible.

## Follow-up: spill telemetry repaired, 2026-09-22

Windows counters were healthy. The Python reader launched Windows PowerShell 5.1 with
an inherited PowerShell 7 module path; loading that diagnostics module failed, and the
reader discarded stderr before turning the empty output into NaN. It now imports the
child shell's own diagnostics module explicitly, checks exit status and sample validity,
uses structured JSON, and reports errors. No execution-policy, registry or driver
settings were changed.

Readings remain process-only, summed across the process's adapter instances. The PID
pattern has an underscore boundary so PID 12 cannot match PID 123. Adapter-wide fallback
is removed: it could mix unrelated jobs into the guard or compare incompatible baselines.

The watcher can establish a late baseline and guard subsequent growth, but records the
unobserved interval permanently. Missing samples produce an unknown clean-run verdict,
not false; an observed breach still produces true. Reports include scope, sample/failure
counts, last error and the actual guard threshold. A final bounded counter read covers
short runs that end before their first background poll. All watcher consumers use these
fields; the standalone sizing benchmark refuses an invalid initial baseline and excludes
unqualified/spilled points from its best-result selection.

Live verification from the previously failing environment initialized both GPUs and
collected three valid process-scoped readings: 156 MiB baseline/latest/peak, zero growth,
zero errors. No training or intentional spill was performed. The six-file focused suite,
including `tests/test_spill_watch.py`, passes **89 tests** (23 spill tests). Existing
historical measurements above remain unknown; this repair cannot recover missing samples.
Details are in `spill-telemetry-repair-20260922.json`.

## Review of `9cdfcf3`, and what the preflight left open — 2026-09-24

The repair was checked against its own claims rather than taken as read. The shared step
rejects stale, missing and duplicate optimizer parameters, clears the whole model's
gradients, weights micro-batches by targets, normalizes the teacher KL by `B*(L-1)`,
reduces replicas before a deduplicated clip and clips the router separately. Every
earlier fix survived the rewrite; the ungrouped `epochs` path is gone rather than
guarded, so every run has a finite, planned shape set and warm-up covers exactly it.
Ten suites, 124 tests, pass with both GPUs visible.

It also found what the earlier diagnosis missed: in the broken tensor-parallel run the
orphaned body parameters still required gradients and `optimizer.zero_grad()` only
clears its own, so their gradients accumulated for 1,663 steps and fed the clipping norm.

**The reload gate had not been run.** The preflight config defines it -- the same
held-out documents, reload within 0.002 nat in aggregate and per source -- and no report
recorded it. `reload_probe.py` on both preflight checkpoints:

| checkpoint | in-loop | reloaded | delta | worst source delta |
| --- | --- | --- | --- | --- |
| initial (step 3) | 1.641797 | 1.640701 | -0.001096 | -0.001672 |
| resumed (step 6) | 1.551504 | 1.551649 | +0.000145 | +0.001358 |

Both pass. The 0.012 nat gap the broken run showed between its log and its checkpoint
does not appear under the new pipeline. The body trains: 1,915.1M of 1,915.2M parameters
changed over six steps.

## bf16 weights discard most updates

That count hides the largest remaining defect. The trainer holds weights in bfloat16
(`smoke_train.py`, `from_pretrained(..., dtype=torch.bfloat16)`) and `bnb.AdamW8bit`
writes into them with no float32 master copy. bf16 keeps 8 significant bits, so an Adam
step of about `lr` survives rounding only when it exceeds half an ulp, `|w| * 2^-9`. At
lr 7.3e-6 that is `|w| < 0.002`. Measured on the single-card kd arm after ~2,000 steps,
all MLP weights:

| \|w\| | elements | changed |
| --- | --- | --- |
| < 2.4e-4 | 20.3M | 99.9% |
| 2.4e-4 - 9.8e-4 | 60.5M | 99.6-99.8% |
| 9.8e-4 - 2.0e-3 | 79.0M | 96.7% |
| 2.0e-3 - 3.9e-3 | 148.7M | 0.7% |
| >= 3.9e-3 | 597.4M | **0.0%** |

Across the whole model the kd arm changed **13.3%** of its elements; every norm, `A_log`
and `dt_bias` changed none, the embedding 4.1%. The six-step preflight had already reached
11.6%, so the movable set is fixed by magnitude within a few steps and more tokens do not
widen it. Which weights train is decided by their size, not their gradient.

Every training result in this project was produced under this, including the finding that
distillation "protects rather than recovers": most of the model could not move. It needs
fixing before another comparison. The options, none installed as a library here
(`torchao` and `optimi` are absent):

* float32 master weights with bf16 compute -- the standard fix; costs 4 bytes a
  parameter on top, about 3.8 GiB per card once split, and the home card carries the
  un-split embedding;
* stochastic rounding of the bf16 write -- unbiased, no extra memory, but needs an
  optimizer that exposes the update, which `bnb.AdamW8bit` applies inside its kernel;
* a bf16 Kahan compensation buffer -- 2 bytes a parameter, same constraint on the update.

Raw results: `reload-probe-initial-20260924.json`, `reload-probe-resumed-20260924.json`.

## Kahan-compensated AdamW8bit -- 2026-09-24

bitsandbytes has no supported fix. `AdamW8bit` (0.50.2) has no rounding or master-weight
option, and its kernel receives the parameter pointer and writes the update in place,
round-to-nearest. Stochastic rounding for its optimizers was requested in 2024 as
bitsandbytes #1165 and is open, labelled low priority. It is not an oversight on their
side: like `torch.optim.AdamW` it updates whatever dtype the weights are, on the standard
mixed-precision assumption that masters stay fp32. This trainer loads straight into
bf16, which is what exposes it.

`KahanAdamW8bit` (in `training_step.py`) does what optimi does for its own optimizers.
Each bf16 weight carries a bf16 buffer holding what rounding drops; per parameter, the
step hands bitsandbytes' own fp32 kernel `weight + compensation`, rounds back and keeps
the remainder. The 8-bit moments and the kernel are unchanged -- the moments do not
depend on the parameter dtype -- and weight decay runs inside the compensated sum. It is
the default; `--no-kahan` reproduces earlier runs.

Against the same kernel on fp32 weights fed the same gradients, 200 steps:

| \|w\| | compensated movement / fp32 movement | RMS error | stock bf16 movement / fp32 |
| --- | --- | --- | --- |
| < 1e-3 | 1.0000 | 0.00% | 1.02 |
| 1e-3 - 1e-2 | 1.0000 | 0.01% | 0.47-0.49 |
| 1e-2 - 1e-1 | 1.0000-1.0001 | 0.10-0.11% | **0.00** |
| 1e-1 - 1 | 1.0037-1.0045 | 2.6-2.7% | **0.00** |

Unbiased everywhere. The few percent near |w| = 1 is the limit of a bf16 buffer -- the
residual it holds grows to half an ulp of the weight, so its own ulp approaches one step --
and is noise around the right answer, not lost movement. That band is norm-sized
parameters; the matrices sit at 0.002-0.05.

On the real model, the preflight configuration for six steps: 99.9% of 1,915M elements
now carry an update that rounding would have dropped, 99.8-100% in every matrix family
and the embedding, against 11.6% for the stock optimizer at six steps and 13.3% at 2,000.
Peak memory on the home card is **18.47 GiB against 14.16** -- the buffers, plus one
parameter's fp32 working copy and gradient at a time, 4 GB for the 508M-row embedding.
Chunking that one update is the lever if the batch has to grow.

The 102 parameters with no buffer are all accounted for: 48 fp32 `branch_gain_delta`
take the stock path; 48 decoder layernorms take no gradient because the gated residual
route replaces the block norm with `branch_gain_delta` by design; 6 `index_gate`s are
detached by the isolated-indexer policy. Held-out over six steps on 12 documents is not a
quality signal -- the stock preflight swings 1.82, 2.15, 1.64, 1.67, 1.86, 1.55.

`copy_train.py`, `sidecar_train.py`, `tp_train.py`, `benchmark.py` and `profile_step.py`
still build stock `AdamW8bit` on bf16 weights and have the same defect.

Raw result: `kahan-probe-20260924.json`.

## Audit: other silent failures of the same class -- 2026-09-24

The bf16 freeze was a silent no-op: the loss fell, nothing raised, and most of the model
could not change. This section lists the symptoms the project ran into and what could
cause each, looking for more of that class.

**The freeze was seen before and misdiagnosed.** `ple_forensics/RESULTS.md` records
"`sharpness` sitting bit-identical for 72 steps" and explains it as AdamW "moving ~`lr`
per element regardless of gradient size". Adam's normalization still moves a float32
weight by about `lr` a step; bit-identical for 72 steps is the rounding signature. Those
arms trained whole decoder layers in bf16 with `torch.optim.AdamW` at 1e-5 to 3e-5, where
every weight above |w| = 0.005-0.015 is frozen. The same document's learning-rate sweep --
optimum at 1e-5, "3e-5 was past the edge" -- was confounded: raising `lr` raises the
threshold under which a weight can move at all, so the sweep changed which parameters
trained, not only how far. Its verdicts rest on arms trained under the freeze.

| symptom | cause found or candidate | status |
| --- | --- | --- |
| distillation "protects, not recovers"; 10M tokens moved MMLU +2.5 | bf16 freeze: 13.3% of elements trainable at lr 7.3e-6 | fixed, `KahanAdamW8bit` |
| `sharpness` bit-identical 72 steps; forensics lr sweep optimum | the same freeze, read as an Adam property | forensics verdicts need re-running |
| `idx` flat at 3.0-4.1 through the chat run | router frozen, and trained jointly at 7.3e-6 against the 1e-3 its warm-up used | open: give router parameters their own learning rate |
| other trainers | `code_training/train.py` (bnb, bf16, 2e-5), `ple_forensics/*_arms.py` (torch AdamW, bf16, 1e-5-3e-5), `copy_train`, `sidecar_train`, `tp_train`, `benchmark`, `profile_step` | open: same freeze |
| lr 7.3e-6 | chosen and only ever validated under the freeze; an update now reaches ~7x more of the model | open: re-validate |
| 8-bit state on the tied 508M embedding | bitsandbytes recommends 32-bit state for embeddings (`StableEmbedding`, or `GlobalOptimManager.override_config(..., "optim_bits", 32)`); not used. Tied, so the head gives every row a dense gradient, which removes the main failure it addresses | open, measure; +3 GB on home |
| latent ladder: 768 fits worse than 512 | candidate: variance of a 32K-token least-squares calibration. 256K tokens improved NLL at 384; predicts a monotone ladder at 256K | open, testable |
| identical runs differ by up to 0.046 | gradient noise flipping bf16 roundings on elements at the boundary; should shrink to continuous differences under compensation | open, testable |
| in-loop held-out on 12 documents swings 0.3 nats a step | sample too small to decide anything | use >= 128 documents |
| gradient accumulation in bf16 | contributions under ~0.2% of the running sum are lost; negligible at `--accumulate 2` with comparable micro-batches | low |
| weight decay | also frozen before; now live, 0.15% shrink over 2,000 steps at this rate | negligible |
| teacher captured under bnb int8 | bounds target quality; not measurable locally at bf16 | unquantified |

Checked and clean: the teacher KL upcasts each logit chunk itself (pinned by
`test_bf16_outputs.py`); the indexer warm-up trains in float32 at lr 1e-3; the conversion
solves in float64 and rounds once on write.

## Learning-rate sweep under compensated updates -- 2026-09-24

lr 7.3e-6 was chosen, and only ever run, while the bf16 freeze decided which weights
could move -- and under the freeze a lower rate meant fewer weights moving at all. This
re-measures it with `KahanAdamW8bit`.

Seven arms, identical but for `lr`: `warmed-chat32`, `teacher-cache-5m`, blend 0.5,
indexer weight 1, prefix cap 1024, `--min-answer-tokens 2`, 6,144 tokens a step
(`--micro-tokens 3072 --accumulate 2`), two cards, warm-up 20, 120 steps, 713,108 tokens,
seed 0 -- so the same data in the same order, and every difference is paired. Scored
afterwards in one process on 128 held-out documents (73,088 targets, prefixes floored to
the routing block), paired bootstrap over documents, 10,000 draws.

| lr | held-out NLL | vs 3.65e-6 | 95% CI |
| --- | --- | --- | --- |
| start (untrained) | 1.3457 | +0.5561 | [+0.4449, +0.6751] |
| 9.1e-7 | 0.7984 | +0.0088 | [-0.0086, +0.0274] |
| **1.83e-6** | **0.7813** | -0.0083 | [-0.0192, +0.0048] |
| 3.65e-6 | 0.7896 | -- | -- |
| 7.3e-6 | 0.8367 | +0.0471 | [+0.0362, +0.0585] |
| 1.46e-5 | 0.8972 | +0.1076 | [+0.0892, +0.1251] |
| 2.92e-5 | 0.9798 | +0.1902 | from the first pass, paired against 7.3e-6: +0.1431 |
| 5.84e-5 | 1.2026 | +0.4130 | first pass: +0.3659 against 7.3e-6 |

**The optimum is a flat basin from about 0.9e-6 to 3.65e-6**, whose three points are
inside each other's intervals. 7.3e-6 is off it by 0.047 nats with an interval clear of
zero, and every doubling above costs more; 5.84e-5 spikes to 1.86 training loss by step 20.

The sweep is 120 steps; a full run is about fourteen times longer, and the optimum moves
down with horizon, not up. So the long-run choice is the low half of the basin:
**1.83e-6**, a quarter of the old rate. Caveats: one seed; held-out NLL on the capture's
own distribution, not MMLU; the router trains at the same rate and has not been swept.

Setting it up exposed one more silent failure. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`,
set by every run here, is not supported on Windows -- PyTorch warns and ignores it -- so
the caching allocator fragments. The first probe at this batch died asking for 1.89 GiB,
the embedding's float32 working copy, with 9.51 GiB reserved but unallocated. Large
parameters are now updated in 64M-element chunks aligned to bitsandbytes' 256-element
state blocks, which a test pins as bit-identical to one pass. Home-card peak at this
batch: 19.46 GiB; 1,780-1,820 tokens a second.

Raw results: `sweep-eval-20260924.json`, `sweep-eval-low-20260924.json`, `sweep-lr-*.json`.

## Router and adapter learning rates -- 2026-09-24

With the body at 1.83e-6, the router (6.5M parameters, the indexer projections) and the
residual adapters (25.8M, created at conversion) were each given their own rate, one at
a time, at 8x, 64x and 512x the body's. Same protocol as the body sweep: 120 steps, same
data order, scored afterwards on the same 128 held-out documents. The 1x arm is the
body sweep's 1.83e-6 arm.

Held-out NLL does not respond to either, across three decades:

| arm | NLL | vs 1x | 95% CI |
| --- | --- | --- | --- |
| 1x | 0.7813 | -- | -- |
| router 8x / 64x / 512x | 0.7857 / 0.7803 / 0.7805 | +0.004 / -0.001 / -0.001 | all span zero |
| adapter 8x / 64x / 512x | 0.7788 / 0.7841 / 0.7789 | -0.003 / +0.003 / -0.002 | all span zero |

That is the wrong instrument for the router: at a 1024-token cap it is one input among
many and a better choice of blocks is buried. `router_eval.py` scores the router's own
objective -- the KL of its scores against the attention over the positions it selected,
what training minimizes -- on the same held-out documents:

| arm | router KL | vs 1x | 95% CI |
| --- | --- | --- | --- |
| untrained start | 23.1288 | +0.4015 | [+0.3411, +0.4629] |
| 1x | 22.7273 | -- | -- |
| router 8x | 22.5652 | -0.1621 | [-0.1716, -0.1529] |
| router 64x | 22.2110 | -0.5163 | [-0.5419, -0.4942] |
| **router 512x** | **22.0990** | **-0.6284** | [-0.6572, -0.6012] |
| adapter 512x | 22.9091 | +0.1817 | [+0.1527, +0.2123] |

**The router wants its own rate; the adapters do not.** At the body's rate the router
improves 0.40 on the untrained start; at 512x it improves 1.03 in the same steps, with
NLL unmoved. Returns flatten -- 64x to 512x adds 0.11 -- and 512x is about the 1e-3 the
router was warmed at, so `--router-lr` now defaults to 9.37e-4. Higher was not tried;
it would be beyond the rate the router has ever been validated at. The adapters stay at
the body's rate: no rate helped NLL, and at 512x they set back the router's alignment.

Raw results: `sweep-eval-groups-20260924.json`, `router-eval-20260924.json`.

## Dropped features to re-test after the freeze -- 2026-09-24

Every documented verdict in the repository was checked for what it trained, in what
precision, at what rate. A training result is suspect when bf16 weights were updated
without compensation at a rate whose freeze threshold, `lr * 2^9`, sits inside the range
of the weights it was meant to move.

| verdict | trained | precision, rate | freeze exposure | re-test? |
| --- | --- | --- | --- | --- |
| PLE responsibility transfer, "branch closed, negative" (`ple_forensics`) | whole decoder-layer windows plus sidecar | bf16, torch AdamW, 1e-5 to 3e-5 | severe: frozen above \|w\| 0.005-0.015; its lr sweep confounded | **yes** |
| GR and PLE stage-1 retrofits, "hurting text NLL" (`eval-20260910`) | sidecar, backbone frozen | bf16 -- `loader.py` loads the student in bf16 by default -- at 1e-4; 1e-3 for the lr arm | partial: frozen above \|w\| 0.05, most likely the gains near 1; checkpoints not on disk to measure | yes, if GR/PLE are still candidates, after fixing the loader |
| Python-specialized backbone B_code (`code_training`) | whole backbone | bf16, AdamW8bit, 2e-5 | severe: frozen above \|w\| 0.01 | only if the code-domain line is reopened |
| code residual gate and structural sidecar, "closed" (`code_gate`, `mbpp_plus`) | gates and branches | **fp32**, 3e-3 | none in the fits; their "already absorbed by B_code" premise inherits B_code's | via B_code only |
| state sidecar, "closed" (`state_sidecar`) | sidecar | **fp32** (cast explicitly), 3e-3 | none; same B_code premise | via B_code only |
| gate diagnosis | calibration fits | fp32 | none | no |
| modular phase 1 / 1b, "ineffective integration" | its own package | AdamW; parameter precision not verified | unverified | check before relying on it |
| `dense_gr.md` "measured and rejected" | nothing -- compile and CUDA-graph speed | -- | none | no |
| conversion calibration, latent ladder, refit | least-squares solves, not gradient steps | float64 | none | no (ladder has its own candidate cause) |

The main trainer is still exposed: `distillkit/models/loader.py` loads the student in
bf16 unless `model_kwargs.torch_dtype` overrides it, and nothing there compensates the
update. Any stage-1 re-test through `distillkit.main` needs that fixed first.

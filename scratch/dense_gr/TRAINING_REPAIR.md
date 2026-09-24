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

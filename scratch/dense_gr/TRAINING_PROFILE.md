# Bounded real-objective training profiles

Assisted-by: Codex

Run only after the context arm releases both GPUs. These options are opt-in;
ordinary training and DistillKit model source are unchanged. The profiler uses
the existing CE, grouped-tail teacher KL, unlikelihood, context masks, sharding,
checkpointing and compensated optimizer. It does not substitute a CE surrogate.

From the DistillKit directory, after `train-ctl-context.json` exists:

```powershell
& .venv/Scripts/python.exe -B scratch/dense_gr/tp_bench.py `
  --recipe-from scratch/dense_gr/train-ctl-context.json --cards 2 `
  --fixed-shape 1 32768 --warmup-steps 2 --steps 3 `
  --profile-dir scratch/csa2-eval/training-profile-context-32k `
  --output scratch/csa2-eval/training-profile-context-32k.json
```

`--recipe-from` reuses the completed run's `run_args`, including all cache roles,
answer weights, context sampling, unlikelihood sources, learning rates and model
settings. Explicit trainer flags override recipe values. Checkpoint/output paths
and run-control fields are not reused. No checkpoint is written. Evaluations and
generation probes are disabled in fixed-record benchmarks.

`--fixed-shape ROWS WIDTH` chooses real canonical cache groups at that exact
planned shape, in seed order, one per accumulation microbatch. It refuses an
insufficient matching set. Alternatively, `--group-indices I J` chooses exact
canonical plan indices; repeats are permitted. `--fixed-records` alone freezes
the normal seed-selected first accumulation cycle. Every optimizer step replays
the same records without altering IDs, teacher targets, padding or loss weights.
Inspect `records.json` for selected document IDs, target weights, objective-row
counts, the plan fingerprint and SHA256 hashes of every record. `ce_rows` counts
rows eligible for CE in the objective; it is zero for KL-only records even though
the shared head still computes and reports their NLL. `original_supervised_rows`
counts the original positive supervision mask, excluding the final position.
`kl_rows` excludes negative spans and includes sampled context, while
`unlikelihood_objective_rows` follows the current KL-only objective branch.
`shared_head_rows` counts the union actually projected by the shared head.
A full recipe does not imply one selected record exercises every loss term:
choose an assistant/context record and a negative rollout if those are the
operations being investigated. The recorded counts make this explicit.

Exactly two real optimizer warmups precede the default three measured steps;
fixed-record mode skips the separate discarded largest-shape warmup. Thus every
measured shape is already warm. Bounds require fresh paths, `--no-checkpoint`, a
finite step limit, adequate target budget, no resume and no preference pairs.
Use the usual external process timeout/spill guard as well: the step bound is
not a wall-clock timeout.

Each measured step exports a Chrome trace and a saved-tensor JSON. CPU ranges
identify body/layers/blocks, MLA projection/routing/attention chunks, head chunks,
backward, optimizer and peer save/load hooks. The trace retains device/stream
identities and CUDA launch, copy and
synchronization events. CUDA activity support is required; the JSON flags traces
with no CUDA device events as incomplete rather than calling them GPU timelines.
The hardware runs on October 5 produced no usable CUDA timeline: Nsight Systems
2026.1.3 failed twice resolving device 0's UUID; PyTorch/CUPTI reported
CUPTI_ERROR_NOT_INITIALIZED and exported CPU events only. Both devices were
explicitly exposed with CUDA_VISIBLE_DEVICES=0,1. This is native Windows, not WSL.
The installed Systems CLI has no --devices option (that is a Compute option).
Its --gpu-metrics-devices=help check separately reports insufficient counter
privileges. CPU inclusive ranges cannot establish GPU idle-time attribution.
See ../csa2-eval/TRAINING_PROFILE_RESULTS.md for ordinary timing results.

For the stream warning investigation, optionally add `--grad-streams`. During
measured steps only, public leaf post-accumulation hooks record the parameter
name, device, current CUDA stream ID, CPU thread and timestamp, with a matching
CPU trace range. Registering these hooks stores them on leaf tensors; the helper
does not request gradient edges or pre-create AccumulateGrad nodes. The callback
stream is the leaf accumulation stream, not proof of the incoming gradient's
producer stream. Correlate the callback ranges with CUDA trace flows/waits. This
option adds per-parameter overhead, so leave it out of throughput A/B runs.

Saved-tensor accounting wraps original hook constructors and preserves their
pack/unpack functions and payloads, including nested checkpoints and peer
offloading. It retains no extra tensor storage. Logical save traffic is separate
from peak **live unique underlying storage**: overlapping tensor views count
their shared storage once. Parameter storage is separated from other storage;
the latter includes activations and copied parameters. Opaque checkpoint holders
are counted but their internal retained objects are not inferred. This is an
accounting scope, not the CUDA allocator peak. Per-card allocated/reserved native
peaks since optimizer warmup are included as the authoritative device measure.

Timings include profiler/accounting overhead, but exclude trace export. For a
throughput A/B, run the same frozen-record command without `--profile-dir`, with
a fresh output path, and require matching record hashes. Use traces to explain
the difference rather than treating instrumented timings as production speed.

Questions to answer before changing the schedule:

- Is replay of router/projections, inner SDPA or head chunks on the critical path?
- Are peer restores/cross-device reductions waiting visibly, and on which stream?
- Do AccumulateGrad stream-mismatch warnings coincide with event waits and GPU
  gaps, or only occur on warmup/first use? The TP code has no dedicated streams;
  do not attribute this warning's cost from utilization samples or suppress it.
- Do outer-checkpoint changes expose many copied K/V prefix views? Compare
  logical traffic, unique retained storage and actual per-card allocator peaks.
- Does a completed CCE objective reduce the whole-step peak or only the head
  phase? Any selective checkpoint/prefetch/vocabulary-sharding experiment follows
  measured headroom and must preserve the objective and gradients.

A separate benchmark-only hypothesis is to clear the CSA2 bus after a complete
optimizer step and synchronized backward. The current body clears it at the
start of each forward, while the last published latent/index/rotary tensors can
retain graph references between steps. Earlier clearing might release those
references sooner or affect the AccumulateGrad stream warning. This is not
implemented here and is not an established source of idle time. Before trying
it, verify all supported reuse/reindex layouts with `use_cache=False`, finish
indexer targets and reporting, and establish exact gradient/update parity. Never
clear the bus while a checkpoint replay or microbatch backward can still read
it. Compare native peak/live-storage reports and stream waits against the same
frozen records, including a multi-step control; cached inference needs separate
ownership rules and is outside this experiment.

CPU verification covers identical gradients through nested checkpoints/custom
save hooks, alias accounting, frozen masks/target hashes, recipe/CLI forwarding,
objective coverage counts, post-accumulation hook cleanup/unchanged gradients,
and a CPU timeline of the actual shared CE/KL/UL/context objective.

An opt-in frozen-record experiment adds --replay-selection-cache to tp_bench.py.
The selector's integer positions and validity masks belong to each outer
checkpoint frame; replay reuses that frame's selection without consulting the
shared bus. It supports Full CSA2 layers in checkpointed training only. Ordinary
training and model source are unchanged. Three CPU tests cover independent frame
ownership and actual hybrid shared CE/KL/UL/context gradients; together with the
eight profiler tests, 11 pass. GPU gradient checks use head_parity.py with
--selection-replay-check --selection-tp; --selection-repeat-control provides an
unchanged-versus-unchanged backward control. No checkpoint or optimizer update is
written by that diagnostic.

The tested streaming head is available to ordinary training with
--shared-head-loss --streaming-head-loss. Full CSA2 layers with outer checkpoints
can also use --checkpoint-selection-cache. Defaults remain unchanged; the
original benchmark-only flags remain supported. These optimizations preserve
the objective, not bitwise GPU backward ordering. The actual update diagnostic
uses head_parity.py --head-update-check --selection-tp --update-replay-cache and
compares with an ordinary repeat. See ../csa2-eval/STREAMING_HEAD_RESULTS.md for
numerics, post-update prediction differences, timing and supported scope.

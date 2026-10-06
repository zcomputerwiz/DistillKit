# Training execution diagnostics, October 5, 2026

Assisted-by: Codex

Follow-up: STREAMING_HEAD_RESULTS.md records the user's practical acceptance
criterion, post-update ordinary-repeat controls and the combined whole-step
gain. Both paths are now opt-in trainer flags with defaults unchanged. The
initial bitwise-parity cautions below describe the earlier diagnostic stage,
not a requirement to reject small variations comparable to ordinary repeats.

## Matched whole-step experiment

Both runs use the actual context arm recipe, two GPUs, two fixed 32768-token
records per optimizer step, two optimizer warmups and three measured steps.
The record hashes and plan fingerprint match exactly. Each step contains 20283
original supervised targets and 25930 shared-head rows, including 5647 context
rows; this particular record pair has no unlikelihood rows. No checkpoint or
full-budget training run was written.

| Metric | Ordinary replay | Cached selection replay |
|---|---:|---:|
| Measured supervised targets/s | 451.747 | 474.140 |
| Median step seconds | 44.425 | 42.612 |
| GPU0 peak allocated GiB | 15.6761 | 15.6754 |
| GPU1 peak allocated GiB | 15.4221 | 15.4221 |
| Peak reserved GiB | 17.4668 | 16.9668 |

Measured throughput improved 4.96%; median step time fell 4.08%. These are short
fixed-record runs, not an estimate of an entire mixed-record training job.
Allocator reservation differs between processes; actual allocated peaks are
essentially unchanged. All five cached steps computed 12 selections and reused
12 selections during checkpoint replay. The cache belongs to each outer
checkpoint frame and retains only nondifferentiable position/mask results.
There is no global bus lookup. The experiment changes neither the model source
nor ordinary training defaults.

Sources: training-profile-context-32k-unprofiled.json and
training-profile-context-32k-replay-cache.json. CPU gradient and ownership tests
passed. The actual u50 BF16 two-GPU check at 512 tokens produced identical losses
and all six selected-position sets, but parameter gradients were not bitwise
equal. In the unchanged-versus-unchanged two-GPU control, gradients also differ:
median relative L2 error 0.849%, versus 1.081% for the cache; maxima are 5.47%
and 7.61%, respectively, across 722 parameter gradients. The single-GPU cache
check likewise has identical loss and selections but median gradient difference
0.942% and maximum 2.65% across 470 gradients. These results do not prove that
the cache adds no backward error. It remains benchmark-only; no ordinary training
adoption is justified by identical losses or CPU parity alone.

The unchanged single-GPU repeat has median relative L2 gradient difference
0.928% and maximum 2.73%, with identical loss and selected sets. Thus backward
variability persists without tensor parallelism; it cannot be attributed solely
to peer transfers. The cache comparison is close to this control on this sample,
but repeated controls or deterministic backward kernels are needed to isolate
any additional cache effect. Both GPU checks cover KL-only real code; the CPU
actual-model test additionally covers shared CE/KL/UL/context gradients.

## Utilization and profiler limits

Training divides work between a sharded body and a tied head resident on GPU1.
Outer layer checkpoints replay body work during backward; MLA attention chunks
also use inner checkpoints. CSA2 selection forms dense historical index scores
and applies top-k without gradients. CPU utilization alone does not distinguish
GPU waits from fast CPU kernel launches. Alternating high-utilization bursts
can accompany these phases, but the critical waits have not been measured.

The CPU-only instrumented step took 44.328 seconds, with inclusive backward
32.302, body 8.442 and head 1.562 seconds. Selector call ranges total 15.309
seconds across forward and replay; these inclusive ranges overlap GPU work and
waits and are not an additive estimate of removable GPU time. Likewise the
peer-hook CPU ranges do not measure actual device copy latency. The observed
5% cache gain is a better measure of its whole-step benefit.

Installed Nsight Systems 2026.1.3 failed twice with "Can't find UUID for CUDA
device 0", both for a CUDA profiler API capture range and tracing from startup
with CUDA event tracing disabled. These are native Windows executions with
CUDA_VISIBLE_DEVICES=0,1. PyTorch Kineto/CUPTI separately returned
CUPTI_ERROR_NOT_INITIALIZED; its trace contains zero CUDA device events.
Consequently there is no GPU idle-time or stream-wait attribution. Systems'
local profile help provides --gpu-metrics-devices, but no --devices option;
its device enumeration reports insufficient hardware-counter privileges.
That counter permission failure is separate from the UUID trace failure.
No driver, counter permission or system configuration was changed.

## CCE and opportunities unlocked by memory

The stable bucketed CCE forward computes full and omitted-tail normalizers in
84-85 ms for 4096 rows. Selected logits match the BF16 reference exactly and
omitted-tail log-probability maximum errors stay below 3e-7 on the three cases.
Bucket CPU preparation took 61.8 ms and upload 3.8 ms outside that GPU timing;
these costs must be included or demonstrably overlapped in a whole-step run.
Metadata occupies 17.4 MiB and forward workspace about 0.297 GiB.

The unfiltered CCE NLL/LSE backward benchmark uses about 2.37 GiB less head-phase
memory than the current shared head. It is a lower bound: complete teacher KL
and unlikelihood backward have not been implemented or verified. It is not a
replacement training objective. Most sampled normalizer error came from
incremental float32 reduction, rather than selected-logit dot rounding; stable
tile summaries with a float64 final reduction eliminated sampled mass oversums.

That head-phase memory saving could permit larger head chunks, prefetched
activations or fewer checkpoints, even if a complete replacement head were
slower in isolation. It does not automatically free the same space on GPU0,
and it does not shrink persistent weights, gradients or optimizer state.
Any scheduling experiment needs a complete gradient-verified objective and a
matched whole-step measurement before adoption. See the existing CCE diagnostic
BUCKETED_FORWARD.md and EXECUTION_OPTIONS.md for the numerical evidence and
remaining implementation work.

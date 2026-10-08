# Completion curriculum: prepared, not launched

Assisted-by: Codex

Updated 2026-10-07 (local). This prepares the next adjustment; it does not promote
the previous candidate or start the measured training arms.

## Why change the targets

The completed execution-grounded pilot improved long greedy agent success from
35/72 (u50) and 34/72 (matched replay control) to 56/72. False completions fell
from 4 and 7 respectively to zero. Recovery remained 2/9 and aggregation 0/9.
Recovery often stops honestly after renewing or inspecting a still-pending job.
Aggregation usually stops on the first page despite a continuation cursor; the
primary gap is complete retrieval, rather than arithmetic alone.

There is no measured plateau: the pilot has only a final step-40 evaluation, and
its falling preference losses use different examples. Further improvement is a
hypothesis to test with complete continuations and milestone evaluations.

Preservation remains part of the objective. Independent paired retention showed
code NLL +0.002483 (95% CI +0.002001 to +0.003013) and thinking NLL +0.000954
(+0.000566 to +0.001357), below the existing investigation thresholds but
measurably worse. The broader atlas also shows that targeted training misses
some replay-control gains. Math results vary across repetitions; no functional
code preservation claim follows from these NLL checks. The prior candidate had
one unauthorized sampled case in seed 2, so it is not a promoted safe baseline.

## Prepared data and implementation

`scratch/dense_gr/completion_curriculum.py` creates 66 training and 22 held-out
complete trajectories across 11 families: recovery, recovery with inspection,
pending and committed timeouts, permanent denial, pagination, overlapping
pages, read-only retrieval, already-completed tasks, empty search, and direct
answers without tools. There are 222 training and 72 held-out assistant-turn
preferences. Every chosen continuation is checked against executed state.

The proposed 80-turn schedule contains all turns of 26 complete trajectories,
ordered within each trajectory, in four 20-turn blocks. Prefixes and tool
results remain context, outside the response loss. `completion_train.py` adds a
finite, inventory-bound pair schedule to the existing trainer. It neither
cycles pairs nor adds checkpoint resume support. Model code is unchanged.

After a recoverable error, the positive continuation renews, retries the
authorized write and confirms observed success. After permanent denial, the
positive continuation reports the blocker honestly. A truthful unfinished
report is a negative only when an authorized tool action can still resolve the
task. Pagination examples follow every cursor, read every unique record once,
deduplicate overlapping pages, and report the complete sum. Read-only,
already-done and no-tool examples constrain needless calls and writes.

Existing rollout and reference-scoring tools were reused. The current candidate
generated responses on 80 clean reference prefixes. Thirteen certified student
negatives cover invalid schema, unhelpful lookup/cursor choices, and an invented
recovery token. None is an observed abandonment on these prefixes; abandonment
contrasts remain curated. This is prefix-conditioned collection, not full
on-policy trajectory sampling.

An audit caught a numeric-labeling error before training: a correct worked sum
was initially read as its first summand. The corrected checker accepts explicit
valid additive equations and leaves ambiguous expressions unlabeled. Corrected
reference scores retain all 222 pair IDs/order, matching prefixes, nonempty
response spans and finite scores. Hashes and individual evidence are recorded
in `completion-v2/prepared.json` and `data/student-audit.json`; superseded
artifacts are retained locally under `numeric-audit-correction/`.

## Proposed measured comparison

Both arms independently start from the previous targeted HF checkpoint with a
fresh optimizer. This is an additional stage, not exact optimizer continuation.
Both use the same 80 frozen replay microbatches / 966,679 weighted targets,
assistant-only conversational masks, full-token raw text/code, flat depth rate
0.1375, ten warmup steps, teacher weight 0.5, and existing tensor-parallel,
selection-cache and shared/streamed-head paths. The completion arm additionally
uses two pairs per step, weight 0.1, DPO beta 0.1 and chosen CE weight 1.0.

The proposed stage is 40 steps with state snapshots at 10/20/30/40. Existing
`export_state.py` exports intermediate snapshots after the stage ends. This
provides a learning curve, not early stopping inside this stage: paired resume
is unsupported. Run training, export and GPU evaluation sequentially.

Compare both arms and their starting candidate on the 44-case new closed-loop
held-out suite, existing safety/agent screens and math at each milestone. Review
paired code/reasoning retention for milestone candidates. Report expected
permission-blocked handling separately from completed goals. Select the earliest
checkpoint with reliable recovery/aggregation gains and acceptable preservation;
do not automatically extend training or promote a checkpoint.

The new worlds vary IDs, argument keys, tool namespaces, padding and page counts,
but share synthetic task templates. They are development proxies, not evidence
of general real-world agent performance. Existing broader screens remain
necessary, and any unauthorized behavior requires case-level review.

## Validation and launch boundary

22 focused tests pass. A discarded one-step GPU validation exercised the actual
replay, two longest scheduled pairs (maximum 3015 tokens), backward and optimizer
path. Initial DPO was 0.691928 versus log(2) = 0.693147; reference-relative margin
was +0.002486. Losses were finite, memory telemetry valid, shared-memory delta
0.086 GiB and no spill detected. Overall peak was 14.71 GiB; measured training
peaks were 12.86/8.99 GiB. No validation checkpoint was saved.

The proposed arms and export commands are concrete in `completion-v2/plan.json`.
Data/reference preparation and GPU preflight are complete. Before a measured
launch, freeze source/runtime/checkpoint hashes, exact milestone evaluation
commands and semantic safety case review. Approximately 1.20 TB is currently
free on D:, but snapshot/export capacity must be checked again at launch.
Measured arms have not started. No new candidate has been promoted.

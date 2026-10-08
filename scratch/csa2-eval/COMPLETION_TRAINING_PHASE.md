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

The arms and export commands are concrete in `completion-v2/plan.json`.
Data/reference preparation, GPU preflight, runtime/checkpoint hash freeze,
milestone commands and safety case review are complete. The frozen launch
reserves 180 GiB for snapshots, exports and evidence; 1,114 GiB was free at
freeze. A one-shot worker now reruns baselines before measured training.
No new candidate has been promoted.

## Launch preparation update

`completion_run.py` now specifies a one-shot bounded worker using the existing
trainer, exporter, live/scenario/math evaluators and paired-retention tools. It
binds source, checkpoint/data hashes and package versions, rejects changed inputs
or implicit retries, reserves 180 GiB for snapshots/exports/evidence, and stops
on spill, nonfinite loss or incomplete exposure. All 27 focused tests pass.
Every milestone has a matching control comparison; independent retention uses
`keep=1` for a direct reference contrast. Baselines are evaluated before training.
There is no automatic extension, selection or promotion.

The remaining safety-case review exposed an evaluator contradiction. In sampled
`scenario:empty:2:0`, a completed empty directory search is followed by successful
alias resolution and inspection of an existing record. The evaluator then calls
a requested write unauthorized based on its hidden task kind. This is ambiguous
evidence of unauthorized behavior; ignoring the user's empty-search stopping
condition and inventing recovery tokens after denial are separate issues.
Historical scores remain unchanged. `completion-v2/safety-review.json` records
the evidence and the completed investigation. The user authorized correction
if unintentional. The original fixture was introduced in `98b1659`; the later
change only bounded prefill. There is no documented intent for the contradictory
lookup results. The generic alias/read handlers had no empty-world branch, and
the reference validation stopped after searching, never exploring those handlers.
This is evidence of a fixture gap, not a deliberate alternative-lookup challenge.

`completion_scenarios.py` supplies the versioned `consistent-no-match-v2`
environment. Empty-world aliases, inspection and writes return NOT_FOUND; guessed
record IDs remain invalid attempts, without a misleading unauthorized-write label.
All other worlds delegate to the unchanged historical evaluator. All 72 original
reference paths still pass, and all prompts/schemas/IDs are unchanged. The new
completion curriculum's empty-world read/write gap is also closed. Chosen paths,
training response spans and reference scores are unaffected by this correction.

`completion-v2/launch.json` and `evaluation.json` freeze sources (including local
fused kernels), packages, data and starting checkpoints. Verification passed.
The bounded worker started at 2026-10-08 04:46 UTC, beginning with fresh u50 and
starting-candidate evaluations on the corrected version, then the two arms and
all milestone comparisons. Its stages/logs/receipts live in `completion-v2/`.
Old scores are historical and must not be pooled with the corrected scores.
No automatic checkpoint promotion or further stage is scheduled.

# Execution-grounded training pilot

Assisted-by: Codex

Research and plan frozen October 7, 2026. u50 remains the reference and adopted
checkpoint. The preceding masked-replay checkpoint is not promoted.

## Research and choice

[TRACE (July 2026)](https://arxiv.org/html/2607.13988v1) assigns credit at tool-call
boundaries while retaining an outcome verifier. Its actual algorithm uses frozen
reference gold-answer probabilities, temporal differences and GRPO. This pilot
adopts the principle of local credit, not the TRACE algorithm: a database state
transition provides stronger evidence of a successful write than answer probability.

[Agent-R](https://arxiv.org/html/2501.11425v3) trains revisions from failed prefixes
and successful continuations, mixed with general data. That supports showing the
student the state after an error, rather than only flawless expert trajectories.
Our recovery does not insert an artificial user approval or require an explanation
before every action. Available tools should resolve recoverable failures autonomously.

[EigenData (January 2026)](https://arxiv.org/html/2601.22607v1) combines executable
environment verification, supervised initialization and subsequent verifiable-reward
RL. We use actual state changes, permissions and observations as evidence. A final
sentence saying "done" does not establish success.

[Why Multi-Step Tool-Use RL Collapses (June 2026)](https://arxiv.org/html/2606.26027v1)
reports structural collapse in small tool-using models and stabilization from
supervisory signals. This motivates retaining chosen-response CE and general replay
instead of starting unrestricted reward-only RL. Its learning rates are not
transplanted to this different student.

[Agent Error Dataset (September 30, 2026)](https://arxiv.org/html/2609.40111v1)
retains execution evidence and separates diagnosis training from action repair.
It supports post-error action targets without requiring reflection text, but its
actor gains are task-dependent and its real-environment evaluation regresses.
Its offline DPO pilot changes likelihood margins without establishing task recovery;
those historical pairs also fail its stricter lineage criterion. Our exact shared
prefixes address that eligibility issue, not the absence of proven recovery benefit.
Thus DPO is exploratory here, chosen-response CE is retained, and actual held-out
execution determines usefulness. A later chosen-CE-only ablation could isolate DPO
if this combined pilot is promising; the present contrast cannot do so.

These papers support the design direction, not a guarantee, a universal best method,
or the particular pilot weights. The existing DPO/CCE trainer and rollout tools
already support a bounded test. A new GRPO backend and inference integration for the
hybrid architecture would add substantial work before testing this specific failure.

## What changes

Two 40-step arms independently start from u50:

| Setting | Masked control | Targeted arm |
| --- | --- | --- |
| Frozen replay microbatches | Same 80 | Same 80 |
| Weighted replay targets | 966,679 | 966,679 |
| Flat depth multiplier | 0.1375 | 0.1375 |
| Warmup | 10 steps | 10 steps |
| Teacher weight | 0.5 | 0.5 |
| Additional pair loss | None | 2 pairs/step; weight 0.1 |
| DPO beta / chosen CE coefficient | N/A | 0.1 / 1.0 |

The rate is half the prior 0.275 pilot multiplier, an exploratory preservation
choice. The matched control isolates the additional targeted objective at that
rate; comparisons with u50 and the prior arm remain necessary. This does not isolate
the effect of lowering the rate from every other run difference.

The targeted objective is existing replay loss plus 0.1 times the mean of
`-logsigmoid(0.1 * ((logp_chosen - ref_chosen) - (logp_rejected - ref_rejected)))`
and chosen-response mean CE. DPO uses summed response log probabilities; chosen CE
uses mean response loss. It therefore has a response-length preference to monitor.
Pair records do not enter the replay token denominator or token-budget accounting.
Response tokens, including assistant end-of-turn, receive the pair objective;
system/user/tool-result tokens and earlier assistant turns in the shared prefix do
not. The model still conditions on tool results and must understand them.

The original 16-source replay, source weights, code/math/QA preservation, assistant
masking, raw-text objective, parameter-specific rate scaling, architecture and model
source are retained. Training uses the already validated TP, streamed head and
checkpoint selection replay path. Additional short pair forwards are sequential;
they do not introduce a second resident reference model. Reference scores are
computed beforehand with the existing `ref_logprobs.py`.

## Data and proof

`execution_pairs.py` builds 96 training-only pairs: eight variants of renewal,
post-renewal retry, confirmed completion, permanent permission failure, uncertain
write without commit, uncertain write with commit, read-only injected tool text,
pagination, complete empty search, already-desired state, known-ID read and direct
arithmetic. Tool names and identifier fields vary. Timeout and pagination fixtures
include both the observation action and its subsequent conditional action.

Every chosen tool branch executes in a local state machine. The audit stores the
shared prefix, actual observations and before/after state, mutation count, reads,
pages and errors. A known final report is constructed from that state. Curated
counterfactual negatives are declared; they are not represented as student samples.
For actions requiring follow-up, the audit executes the successful continuation too.
Positive turns across the fixture families cover the recovery sequence.

The existing `onpolicy_rollouts.py` collects one greedy assistant turn per fixture
from u50 (batch 4, prefix width 1024, budget 192). Only completed, certifiably bad
turns replace curated negatives. Schema violations, unauthorized/redundant writes,
retrying an unknown outcome before observation, invented IDs/tokens, and explicit
affirmative completion against an unchanged state are eligible. Unknown prose,
questions, planned actions and truncated generations are not automatically labeled.
This is prefix-conditioned student data, not full on-policy trajectory collection
or iterative self-training. The replacement count and every decision are audited.

Frozen development cases, IDs, tool schemas and trajectories are not used for
training. The data builder does not import `agentic_scenarios.py`. Its separate
transfer namespace tests schema transfer at the unit level, not live task success.
The frozen 80-pair training schedule covers all 12 families and is recorded in the
plan. Reference preparation must retain all 96 pairs and identical chosen/rejected
prefixes, with finite scores and nonempty assistant-only spans.

## Limits and gates

The new simulator is still a small record-management proxy. It does not establish
real browsing, coding-agent, concurrent transaction or arbitrary API reliability.
Timeout semantics are intentional: unknown outcome requires observation, whereas
STALE_SESSION explicitly guarantees no write. Transferring indiscriminate retries
to real tools could duplicate non-idempotent operations. No amount of this training
guarantees "always"; serving-time claim/state checks remain a separate enforcement
mechanism and are not being added to model source here.

The existing pair iterator does not checkpoint its shuffle state. Paired resume is
now rejected explicitly instead of silently replaying the wrong pairs. Invalid,
negative or non-finite preference settings fail before allocating the model. A
failed paired run needs inspection and a fresh, separately recorded restart.

Preparation and training use existing stage receipts and logs. Frozen input and
runtime hashes are verified. The trainer warms pair shapes before updates, measures
VRAM/spill and synchronizes/clips gradients. Preparation finishing does not imply
training or evaluation finished.

Evaluate both arms on the same nine frozen agent/math commands, including long
contexts with bounded cached prefill, sampled trials, readonly/injection, empty
search and no-tool controls. Review false claims alongside actual final state,
invalid writes, loops, tool-action correctness and final-report semantics. Do not
reward merely suppressing the word "done". Run paired code/reasoning NLL against
u50 and the matched control before considering promotion. Meaningful improvements
must survive the existing seed/budget review and broader reserved benchmark gates.
No automatic promotion or follow-on training is authorized by a stage exit alone.

## Previous diagnostic now complete

At fixed 4K budget, seed 1, batch 8 and isolated compile cache, the u50 MATH repeat
is token-identical on 64/64 cases and remains 37/64 correct. GSM8K is identical on
41/64, changes 46 to 49 correct, gains three answers and loses none. These are two
runs, not an estimate of all inference variance. The tool recovery failures remain
verified independently. `phase3-masking/diagnostics/review.json` retains case details.

## Run locations

Plan and evaluation protocol: `scratch/dense_gr/execution-grounded/`.
Generated data/reference/proofs: its `data/` directory. Stage receipts/logs:
`runs/`. Commands use the existing `.venv/Scripts/python.exe`:

```text
scratch/dense_gr/execution_grounded_run.py prepare
scratch/dense_gr/execution_grounded_run.py train-control
scratch/dense_gr/execution_grounded_run.py train-targeted
scratch/dense_gr/execution_grounded_run.py eval-control
scratch/dense_gr/execution_grounded_run.py eval-targeted
```

CPU validation: eleven executable-world, preference-loss, checkpoint-selection
replay and settings-validation tests passed, including multiple simultaneous
checkpoint frames and invalid options rejected before model allocation.

Preparation completed: all 96 student turns ended within budget; 18 certified
failures replaced curated negatives (11 invented IDs, two invented tokens, four
redundant writes and one invalid schema). No student false completion was certified
on these short fixtures; explicit false-completion contrasts remain curated. All
96 reference pairs were retained with finite scores and matching prefixes.

Audit correction: the first labeling pass incorrectly treated "Done" on four
correct read-only reports as a failed write. Inspection caught this before
training. The classifier now restricts completion-state checks to mutation tasks;
read-only and no-tool regression tests cover the distinction. Previous labels,
scores and receipts are retained in `audit-correction/`; corrected reference
preparation is complete. The unchanged student rollouts were reused by hash.

A two-step GPU preflight completed before the bounded arms. It used a fresh u50
copy, the actual pair/replay path and no checkpoint output. Both steps completed
with valid memory telemetry and finite losses: measured training peaks were
15.67/14.77 GiB, shared-memory delta 0.086 GiB, and no spill was detected. Initial
mean DPO was 0.69972 versus log(2) = 0.69315, with reference-relative mean margin
-0.01292. This checks two real pairs before the first update, not bitwise reference
equivalence on the entire dataset. The continuation stops on any failure and runs the
frozen control/targeted arms, nine evaluations per arm, paired retention evidence
and CPU comparisons. Its launcher hash and current stage are recorded in
`execution-status.json`. This is a one-shot worker, not a recurring automation.

Both 40-step arms and their bounded evaluation worker have completed. Long greedy
success was 35/72 for u50, 34/72 for control and 56/72 for targeted; false
completions were 4, 7 and 0. Targeted recovery was still 2/9 and aggregation 0/9,
usually due to stopping before retry or further pagination. One unauthorized
sampled case remains; no candidate is promoted. Across the targeted schedule's
80 pairs, 16 negatives were certified student failures and 2,057 chosen assistant
tokens received the added objective; 52,216 prefix tokens remained outside it.
Independent code/thinking retention worsened slightly, below the investigation
thresholds. See [the completion-phase preparation](COMPLETION_TRAINING_PHASE.md)
for completed results, limitations and the next proposed comparison.

The first preflight was stopped by the trainer's finite-plan guard before updates:
two steps cannot consume the original 40-step ordered schedule. Its failed receipt
is retained in `preflight-plan-check/`. The preflight now has its own four canonical
microbatches, copied from the frozen schedule and bound to its source hash. The
full 80-batch schedules for the measured arms are unchanged.

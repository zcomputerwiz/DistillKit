# Next phase: masked replay, compared with u50

Assisted-by: Codex

Date: 2026-10-07. **Planning and validation only; no training has started.**
The user selected masked replay only, compared with u50. There is no runnable
legacy context-loss arm. The older `agentic-v3-masked-plan` is superseded by
`scratch/dense_gr/phase3-masking/plan.json`.

## Question and scope

Does a short continuation using masked conversational replay preserve or
improve useful assistant behavior? Train no new synthetic agent curriculum in
this phase. Compare the candidate with the unchanged u50 checkpoint. This
measures the combined replay update, not the isolated causal effect of masking.
If retention is acceptable, plan a separate curriculum experiment afterward.
Do not combine a loss-policy change and a new data-dose change again.

All conversational replay scores assistant text, reasoning, tool calls and
assistant turn endings. User/system/tool-result text remains visible as input
but unscored. Raw code and general-text continuation retains full-token loss.
Existing CE/KL/UL assignments and the streamed-head/selection-cache optimizations
are retained. No model source or llama.cpp changes are required.

## Frozen training schedule

| Setting | Planned value |
|---|---|
| Start | `scratch/dense_gr/merges-long1/u50` |
| Steps | 40, with two microbatches per step |
| Data | Existing replay only; no new agent examples |
| Actual supervised targets | 966,679 |
| Batch order | 80 frozen canonical groups, recorded document IDs and widths |
| Input budget | Up to 32,768 tokens per microbatch |
| LR | Existing 1.83e-6 base, flat 0.275 depth multiplier; existing parameter-specific multipliers remain |
| Warmup/decay | 10 warmup steps; flat afterward, decay disabled |
| Body peak LR | 5.0325e-7 before parameter-specific multipliers |
| Initialization/seed | Fresh u50; seed 25 |
| Outputs | `phase3-masking/masked/`, with ordinary resume state |

The 10M `--tokens` argument is an upper bound, not an instruction to train to
10M targets: the finite batch plan and 40-step cap terminate the run. The trainer
rejects a budget or step limit that would truncate the frozen plan. Resume binds
the selected groups, cache manifests and objective weights, and never reshuffles
or repeats after exhaustion.

The planner selects existing source-homogeneous groups, uses masked target mass
for source balancing, and gives small replay sources a share floor before
normalization. Whole batches mean realized shares are approximate, not exact
token quotas. Source coverage is now materially better than the earlier tiny
prefix: raw general text contributes 65,266 targets instead of 381; teacher-code
contributes 51,271 instead of 15,351. Raw code contributes 324,423. All 16 replay
sources are represented; the full source ledger is in the plan.

CPU preflight read all 80 planned microbatches through the actual cache reader:
966,679 targets, **zero conversational user/system/tool-result targets**, and an
exact match to the planned fingerprint. It performed no model update. Training
memory/throughput still needs the normal first-step checks; no new performance
number is claimed.

## Evaluation corrections

1. **Live grading v2** separates recovered errors from completion and retains
   invalid-attempt counts. It verifies the target ID and recorded tool results,
   rejects false completion, accepts reasonable clarification wording without
   demanding both direction words, and only provides a user choice after actual
   ambiguous results. Legacy mode remains explicit for reproducing old results.
   Free-form explanations and uncertain final wording are review items, never
   automatic passes. Announcement-without-call detection is a review heuristic,
   not a formal guarantee about all natural language.
2. **Independent agent development suite:** 72 frozen tasks across eight families
   and three wordings, at three distractor lengths. New families include pagination,
   indirect IDs, multi-record aggregation, stale-session recovery and untrusted
   instructions in tool results. All gold trajectories execute successfully.
   Use the existing 48 live tasks as additional coverage of genuine user choices
   and conditional/no-op writes. The length stress is synthetic, not a substitute
   for natural long-context QA.
3. **Retention NLL:** all 292 eligible code and 300 thinking eval documents at a
   4,096-token prefix cap, rather than the first 64 IDs at 1,024 tokens. Report
   assistant/reasoning/tool-call loss separately from observations. Retain the
   existing broader atlas for QA, raw code, general text and real agent traces.
4. **Math development:** the old 512-question bank left only 15 GSM8K and 9 MATH
   questions after conservative matching against every local capture JSONL.
   These are potential capture overlaps, not proof every case entered an optimizer
   update. Expanding to all train questions left 235 GSM8K and 11 MATH candidates.
   Therefore MATH development is carved from the remaining MATH test split,
   excluding MATH-500 normalized exact/13-word overlaps and local capture overlaps.
   MATH-500 remains reserved for final confirmation. Never label this new subset
   as full MATH test accuracy or use its questions as training rollouts.
   The final bank contains 64 GSM8K and 64 MATH cases; MATH covers all seven
   subjects. The screened candidate pools contain 235 and 3,302 questions.
5. **Generation protocol:** both EOS tokens (248044/248046), raw text, token IDs,
   per-case answers, formatting and truncation flags, bank/prompt hashes, and
   independent per-case RNG streams are retained. Old generation behavior remains
   available explicitly for historical comparison. No paid judge API is required.

The earlier explanation review is now recorded separately: u50 and replay each
explain review correctly in 3/4 cases, candidate v2 in 1/4. Combined with the
saved-trace audit, outcomes are 36/48, 37/48, 36/48; clean outcomes are 33/48,
30/48, 33/48. This is a non-blind single-assistant semantic review, not an
independent judge. It reinforces the decision not to promote v2.

## Execution order and decision rules

1. Freeze and verify all evaluation assets before training. Run the u50 baseline
   first. Inspect review flags and any protocol failures; correct an evaluator
   defect for both checkpoints, never just one arm.
2. Run the 72 new tasks greedily and the 48 corrected live tasks. Run the 24 short
   independent tasks with seeds 1, 2 and 3 for sampled consistency (batch size 1;
   each task/turn receives its own seed). Report per-family outcome, clean success,
   unauthorized attempts, false reports, needless handbacks and announcement flags.
   Do not call deterministic repeated decoding a reliability experiment.
3. Run frozen math development with seeds 0, 1 and 2 at 2,048 new tokens. GSM8K
   is greedy/nonthinking; MATH is sampled thinking (T=.6, p=.95, k=20). Repeat
   the first 16 cases per subject at 4,096 tokens with seed 0 to diagnose cap
   sensitivity. Grade strict boxed answers, but report unboxed/truncated cases
   separately and retain text for review. Do not silently change answer extraction.
4. Train only the masked arm from the serialized argv after the baseline is
   valid. Save the final model and resume state. Do not automatically continue
   beyond 40 steps or promote the result.
5. Evaluate candidate with the identical frozen protocols. Compute paired
   document intervals for own-turn NLL and paired case comparisons for outcomes.
   Re-run neither bank selection nor source sampling based on candidate scores.
6. Stop progression on any new unauthorized write or false-completion failure
   without explanation, or a recurring new tool-handback pattern. An own-turn
   NLL rise over 0.01 nat in code or 0.02 in reasoning/QA is an investigation
   trigger, not an automatic proof of capability loss. A decline of 3/64 math
   answers in any seed triggers case and budget analysis. These are conservative
   pilot triage thresholds, not powered non-inferiority claims.
7. A promising candidate still needs executable code confirmation using the
   existing HumanEval+/MBPP+ generator and installed no-network Docker sandbox,
   plus final GSM8K-test/MATH-500 confirmation. Use both stop tokens explicitly.
   NLL alone cannot certify code correctness. Keep u50 unless the complete
   evidence supports adoption; do not tune another recipe on final-test results.

Small development sets and synthetic environments cannot establish universal
agent reliability. This scope is enough to detect the specific observed failures
and decide whether to proceed to a separate, broader curriculum experiment.
If the masked replay phase fails retention, adjust its rate or exposure before
adding new agent data. If it passes, keep its verified replay schedule as the
reference for the next curriculum experiment, matching replay documents and LR
rather than only optimizer-step counts.

## Tools and commands

`phase3-masking/evaluation.json` freezes asset hashes and exact baseline/candidate
commands. Verification checks all 107 local capture-file hashes and their
inventory as well as the selected evaluation assets. Run
`next_phase_plan.py --verify-evaluations` before beginning evaluations.

Validation: 29 focused tests pass, including finite-order resume, both EOS
tokens, independent case RNG, grading and executable reference trajectories.
The real u50 math runner completed a two-case plumbing smoke; both cases reached
its deliberately short 32-token cap, with raw output and truncation flags saved.
This is not a capability score. The agent smoke exposed an unnecessarily
required first-page cursor; the tool schema now makes it optional and explains
pagination. The corrected suite was reference-validated and re-frozen before
any baseline evaluation.
The corrected two-case u50 agent smoke completed: the known-ID read passed
cleanly. Pagination ultimately reached the correct state, but an unauthorized
write attempt and invented recovery token correctly prevented a success grade.
These are debugging cases, not a representative baseline. Their saved traces
are `phase3-eval/smoke-agent-optional-cursor-output.json`.

All paths below are relative to DistillKit; use `.venv/Scripts/python.exe`.

```text
scratch/dense_gr/next_phase_plan.py --verify
scratch/dense_gr/agentic_scenarios.py run --checkpoint CHECKPOINT --output OUTPUT.json
scratch/dense_gr/agentic_live_eval.py --data scratch/dense_gr/agentic-v2-data/trajectories.jsonl --checkpoint CHECKPOINT --output OUTPUT.json --grading-version 2
scratch/dense_gr/math_dev_eval.py --checkpoint CHECKPOINT --output OUTPUT_DIR --seed 0
scratch/dense_gr/atlas.py nll --domains scratch/dense_gr/phase3-eval/retention/domains.pt --arm base=BASE --arm masked=CANDIDATE --output-dir OUTPUT_DIR --save-token-evidence
```

For sampled agent runs, add `--short-only --sample --seed SEED --batch-size 1`
to `agentic_scenarios.py`. For code generation, add
`--eos-token-ids 248044 248046` and, when sampling with the compiled runner,
`--case-seeds`. Execute generated code only through the existing
`scratch/downstream/code_bench/run_docker.ps1` sandbox.

The exact training argv is in `phase3-masking/plan.json`; there is no legacy
arm in that file. Do not invoke the old `agentic_arm.py run` pipeline for this
single-arm phase: it expects two differently named historical arms.

## Rationale

The emphasis on verified environment outcomes and repeated trials follows
[tau-bench](https://arxiv.org/abs/2406.12045). Keeping code execution tests in
the final check follows [EvalPlus](https://arxiv.org/abs/2305.01210), which shows
that weak tests can mis-rank models. The masking decision and local evidence
are recorded in [the evaluation audit](EVALUATION_AUDIT.md) and
[the role-loss review](ROLE_LOSS_REVIEW.md). The specific 40-step schedule and
triage thresholds are experimental choices for this project, not prescriptions
claimed from those papers.

# Existing checkpoints: tool behavior screen

Completed 2026-10-05. All five checkpoints completed the frozen offline screen
and exploratory system-instruction follow-up. No training was launched; no
generated tools or code were executed.

## Decision

Keep assistant-only CE/KL. Do not advance teacher context-KL as a preservation
recipe. Simple returned-value reading is retained, but choosing to emit a tool
call has regressed on some held-out traces. Context-KL does not repair that
behavioral gap. A clear tool-use system instruction recovers part of it.

Flat 0.55 is the preferred confirmation candidate, not a proven long-run
winner. It matches or slightly exceeds ramp on these tool screens while
retaining about 90% of round-5's QA gain versus ramp's 75% on the broader QA
set. Fix the intended serving instruction before comparing subsequent arms.
If an auxiliary is needed, anchor protected assistant decisions conditioned
on real history rather than prediction of tool outputs. Do not combine a new
anchor with rate/data changes in the same first experiment.

## Protocol

`scratch/dense_gr/tool_behavior_eval.py` reuses the existing HF loading/generation
path and `frontier/tool_tasks.py` schema validator. Actual runs use eager HF
generation: greedy BF16, batch 8, 512 new tokens, native chat templates with
`enable_thinking=False`, stopping on either 248044 or 248046. Every checkpoint
must reproduce each frozen rendered prompt exactly. GPU 0 evaluated base and
round-5; GPU 1 evaluated ramp, flat and context, one worker per GPU. All ended.

Checkpoints: base `merges-long1/u50`, round-5 `control-arms/long5-step100`, and
`checkpoints-ctl-{ramp,flat,context}/smoke-r1-1-gr-s25-csa2`. The committed
`summary.json` records full SHA256 hashes of their safetensors and the fixtures.
Raw generations, frozen inputs, logs and common `graded.json` rescoring remain
local beside this report; generated datasets are not added to Git.

The primary fixture has 72 cases: 48 new synthetic cases and 24 reference-call
targets from the frontier-tools cache's eval IDs, excluding its train IDs, one
target per document. A second fixture contains all 12 eligible held-out
documents with a call after a tool result. Three exact targets overlap the
primary set; do not add denominators as independent samples. ID exclusion does
not establish absence of paraphrases, cross-corpus duplicates or pretraining
exposure.

Controlled cases include eight each of direct extraction, dependent-ID calls,
error branches, longer extraction and initial calls, plus four each of no-call
and missing-information requests. The first four groups form 16 paired
contexts whose changed results alter the correct answer/action. Long inputs
reach 4,739 tokens, with the result preceding unrelated logs. These are simple
diagnostic templates rather than broad task coverage. Four identical
missing-information prompts count as one bootstrap unit.

All outputs are rescored identically. Grader corrections accept imperative
requests without question marks, interpret XML strings using schema types,
and permit independent parallel calls in a different order. No prompts or
generations changed. Six grader tests pass. Earlier trainer/profile validation
passed 53 tests; a further profiler/grader subset passed 14.

## Primary results

Every checkpoint passes all direct and longer extraction, dependent-ID, initial
call, no-call and missing-information cases. Differences occur elsewhere:

| Checkpoint | Error action / 8 | Both variants / 16 pairs | Reference calls / 24 | Schema-valid calls / 24 | Post-tool reference calls / 12 |
|---|---:|---:|---:|---:|---:|
| Base u50 | 4 | 12 | 8 | 16 | 6 |
| Round-5 step 100 | 5 | 13 | 1 | 3 | 3 |
| Ramp | 8 | 16 | 4 | 11 | 7 |
| Flat 0.55 | 8 | 16 | 4 | 11 | 8 |
| Context-KL | 4 | 12 | 2 | 4 | 3 |

One base primary replay truncates; no other primary or post-tool response does.
Truncated responses cannot pass. Schema validity checks declared tools and
argument schemas, not correct selection or successful execution. Exact replay
matches are not task-success scores: alternative Python programs or API plans
may be correct. Primary emitted-call counts are base 18, round-5 4, ramp 13,
flat 13 and context 5, corroborating omissions beyond reference mismatches.

Paired document-bootstrap primary exact-match deltas versus base: round-5
-29.2 percentage points (95% interval -45.8 to -12.5), ramp/flat -16.7 (-33.3 to
-4.2), context -25.0 (-41.7 to -8.3). Post-tool: flat +16.7 (0 to +41.7), ramp
+8.3 (0 to +25.0), round-5/context -25.0 (-50.0 to 0). These describe a small
one-seed screen, not full-budget or benchmark significance.

Manual examples distinguish behavior from context prediction:

- On RATE_LIMIT, base wrongly calls `refresh_token` with the public ticket
  number. Context reads the delay correctly but announces a retry without
  emitting `schedule_retry`; ramp and flat emit it correctly.
- Context says it cannot change a display name despite having
  `update_display_name`; base emits that tool with the supplied new name.
- Context says it cannot open a returned URL despite having `fetch_page`;
  base emits the call with that URL.
- On a DNS follow-up, context suggests PTR instead of the requested CNAME
  query despite having `dns_query`.

## Exploratory prompt ablation

After observing omissions, a separate fixture prepends a generic instruction:
use supplied tools when needed, emit native calls rather than announcing them,
use returned values, and ask for genuinely missing information. It contains 33
unique replay targets across 27 documents, plus the eight error cases, with
unchanged targets. It is exploratory, not independent confirmation.

| Checkpoint | Error action / 8 | Reference calls / 33 | Schema-valid calls / 33 | Emitted calls / 33 |
|---|---:|---:|---:|---:|
| Base u50 | 4 | 13 | 28 | 31 |
| Round-5 step 100 | 8 | 9 | 16 | 19 |
| Ramp | 8 | 14 | 24 | 26 |
| Flat 0.55 | 8 | 14 | 25 | 28 |
| Context-KL | 8 | 9 | 14 | 16 |

On those same 33 targets without the instruction, exact matches are base 14,
round-5 4, ramp 11, flat 12 and context 5. For overlapping targets, the matched
union uses the separate post-tool run's output. Prompting helps trained arms
but not every checkpoint: base loses one net exact match. One base framed
response truncates; all others finish. Framed bootstrap units are documents,
keeping multiple targets from a document together.

Restored error actions show some failures are sensitive to elicitation rather
than complete loss of ability. Prompting does not close the replay gap, and
context-KL does not outperform ordinary round-5 on reference-call matches.

## Limits and reproduction

Track generated call omissions and result-conditioned decisions alongside
teacher-forced call NLL: the latter improved even where generated behavior
worsened. There is no executable task-success, stochastic reliability, 32K
agent, or multi-seed full-budget result here. Novel-value tests are near
ceiling. Confirm the candidate with broader sandbox agent tasks while
retaining QA/code outcomes.

From the DistillKit root, the runner's `build` subcommand takes `--tokenizer`,
`--cache`, `--source`, and `--output`. The recorded inputs are
`scratch/dense_gr/merges-long1/u50`, `../teacher-cache-frontier-tools`, and
`../capture-data/frontier/tools.jsonl`. Seed 20261006 is fixed. Build the second
fixture with `--post-tool-only`; create the follow-up with `framed --directory
scratch/csa2-eval/tool-behavior --tokenizer ... --output .../framed-frozen.json`.

Run each arm/fixture with:

```powershell
.venv/Scripts/python.exe scratch/dense_gr/tool_behavior_eval.py run --fixture scratch/csa2-eval/tool-behavior/frozen.json --checkpoint scratch/dense_gr/merges-long1/u50 --arm base --device cuda:0 --output scratch/csa2-eval/tool-behavior/base
.venv/Scripts/python.exe scratch/dense_gr/tool_behavior_eval.py report --directory scratch/csa2-eval/tool-behavior --output scratch/csa2-eval/tool-behavior/summary.json
```

`report` requires all five arms for each fixture. Fixtures and completed runs
refuse overwrite; use fresh directories for reruns. Exact fixture reproduction
requires the same tokenizer/template and source/cache manifests.

Assisted-by: Codex

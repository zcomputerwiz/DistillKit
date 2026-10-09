# Replay data and next-phase assessment

Assisted-by: Codex

2026-10-08. Assessment only: no new model generation, training or promotion.
CPU cache scans and eight solution checks in the existing Docker sandbox completed.
The approved policy is **exclude confirmed matching tasks only**. Conversational
context stays masked; raw code/text retains full-token loss.

## Findings

Use the existing data more carefully before adding a large dataset. The 475
retained, execution-verified teacher-code conversations include 392 outside the
recent replay prefix. Raw code already supplies 33.56% of weighted replay targets,
but from only 17 long documents. Greater document variety and verified solution
coverage are better-supported adjustments than raising those documents' weight.

Aggregation needs fresh executed transitions and complete trajectories. Its four
trained aggregation worlds were learned but did not transfer. More copies of
those paths or more generic code is not an established repair.

After removing confirmed exposed tasks from the retained-pool code comparison,
sampled means still decline, but both intervals cross zero. This weakens the claim
of a statistically established independent functional-code regression; it neither
erases the paired code-NLL concern nor establishes equivalence. Keep u50 as the
reference and the current candidate experimental.

## Exclusion indexes and confirmation

The documented master is `../capture-data/exclude-master-v2.json`: 2,169 IDs,
described in `DISTILLATION.md` around line 799. The current run index is
`../capture-data/exclude-long-r5.json`: 4,175 IDs. `long_round5.ps1` combines nine
lists; `exclusion_for.py` intersects the union with loaded cache IDs. Recent run
arguments correctly use that derived index.

The automatic banks in `expand_corpus.py` cover MMLU/ARC; `math_contamination.py`
covers GSM8K/MATH-500. Its old docstring overstated HumanEval/MBPP screening. The
teacher-code pool received a separate code screen at review, which did not cover
all inherited thinking corpora. The inaccurate docstring is corrected without
changing screening behavior or frozen results.

All 16 current replay caches were screened: 42,620 retained training documents,
81,783,841 input tokens. Six code/reasoning caches were decoded in the initial
audit, the other ten in the follow-up. Inputs decoded during follow-up were
checked against their manifest token hashes. This is not an exhaustive semantic
paraphrase or historical-training audit.

Confirmation requires a copied complete HumanEval function contract plus its
definition, or an MBPP stem, matching interface and at least two copied concrete
benchmark tests. Corpus names and shared phrases alone never trigger exclusion.
Moving an import does not change a copied contract.

There are **104 confirmed documents covering 103 tasks** (42 HumanEval, 61 MBPP):
70 thinking documents, 33 think-first documents and one on-policy rollout alias.
Thirteen other document/task flags stay unconfirmed. Several connect distinct
tasks, including prefix/substring filtering or sorting second/third positions.
A generic sorting stem with pancake-sort tests is not a comb-sort match.

Four confirmed documents occur in the recent frozen prefix: HumanEval/161 and
MBPP/398, /226 and /232. The additional MBPP/427 rollout copy is outside that
prefix. Exposure does not causally identify the stage responsible for a change.

Future-use files in `../dense_gr/completion-gauntlet/dataset-audit/`:

- `overlap-review.json`: decisions, evidence, manifest and input hashes.
- `exclude-code-confirmed-20261008.json`: 104 confirmed document IDs.
- `exclude-replay-next-20261008.json`: old 4,175 plus 104, **4,279 IDs**.
- `exclusion-build.json`: existing builder invocation and union verification.

The original master, run index, frozen 80-batch schedule and scores are unchanged.
A future schedule must be regenerated with the new index: the old schedule would
restore four excluded documents. Different cache combinations must be trimmed
again with `exclusion_for.py`. Screen new aliases under actual captured IDs.

## Separately screened functional results

Reuse existing final-answer diagnostic scores; no new generation or benchmark
execution. Both historical extraction results and the unscreened final-answer
diagnostic remain available. Intervals resample tasks across three matched seeds;
they are exploratory, unadjusted pass@1 comparisons.

| Final-answer sampled mean | Tasks | u50 | Step 40 | Delta and paired 95% CI, percentage points |
| --- | ---: | ---: | ---: | --- |
| HumanEval+, recent-prefix unmatched | 163 | 42.94% | 38.45% | -4.50 [-8.79, -0.41] |
| MBPP+, recent-prefix unmatched | 375 | 45.33% | 43.20% | -2.13 [-4.98, +0.71] |
| HumanEval+, retained-pool unmatched | 122 | 42.08% | 38.52% | -3.55 [-8.20, +1.09] |
| MBPP+, retained-pool unmatched | 317 | 43.85% | 41.96% | -1.89 [-4.94, +1.05] |

On the retained-pool subset, greedy HumanEval is 52/122 in both arms; MBPP is
146/317 versus 144/317. HumanEval sampled seeds have two losses and one gain;
MBPP has two losses and one tie. Smaller subsets are less precise and change the
task distribution. These screens do not rule out older student or foundation
exposure. Evidence: `screened-comparison.json`.

## Existing pool and measured exposure

| Source | Retained train docs | Recent distinct docs | Recent nonempty thinking | Outside recent prefix |
| --- | ---: | ---: | ---: | ---: |
| Raw code | 768 | 17 | 0 | 751 |
| Verified teacher code | 475 | 83 | 39 | 392 |
| Expanded code | 5,507 | 256 | 0 | 5,251 |
| Verified short code | 1,085 | 51 | 41 | 1,034 |
| Thinking | 5,290 | 87 | 87 | 5,203 |
| Think-first | 3,673 | 115 | 115 | 3,558 |

Counts precede the new supplement. Outside this prefix does not mean never used
historically. Captured-length truncation was absent in these six sources. A final
newline token is not evidence that a conversation lacked a turn delimiter.

The 966,679 weighted replay targets contain raw code 33.56%, expanded code 5.94%,
teacher code 5.30%, short code 1.69%, thinking 4.70% and think-first 3.63%. These
are actual targets, not cache sizes or nominal shares. The 17 raw documents span
12 repository/package origins.

All 475 teacher rows have historical passing execution labels, completed outputs,
closed thinking boundaries and valid/kept judge labels. Origins: Leetcode 94,
Code Contests 79, Taco 133, Algorithm 58, Filter 105 and Docs 6. Nonempty thinking
occurs in 188, median 574.5 and p90 1,038.6 reasoning tokens; 287 have empty thinking.
Of the 392 outside-prefix prompts, 43 mention duplicates/frequency and 91 mention
boundaries/empty/signed cases. Overlapping keywords and test names are coverage
hints, not proof of semantic or mutation-test coverage.

Teacher generation used medium effort and no system message; the gauntlet used
the template's xhigh default. Old project notes explicitly retained this mismatch
for comparison. A versioned effort-matched probe could distinguish it from a
training-induced length change. Thinking/sampling are also confounded here.

### Verification and format alignment

The inherited verifier chooses the last final fence; the benchmark diagnostic
chooses the first. 473/475 teacher and 1,079/1,085 short-code documents have one
final fence. Four documents change extracted code. Eight first/last solutions
were checked with the existing no-network, no-host-mount `code-verify-sandbox`:
all historical last solutions pass; two first extractions fail.

One recent teacher document puts an unlabeled DP formula before a correct
explicit Python solution. Preferring the explicit Python block resolves this
format mismatch. One short-code document outside the recent prefix has a failing
first Python solution and a passing revision. The other two changed short-code
solutions pass either way. This small finding does not explain the benchmark gap.

A read-only scan found no explicit-Python-preference changes in current completed
benchmark outputs, so that preference does not affect the table above. Before
reusing data, qualify the exact final code the served protocol consumes, target
one complete Python solution, and report format separately from correctness.
Legitimate revisions can become explicit tool-assisted turns; a passing last
block does not certify earlier blocks. Historical labels/captures were preserved.

Evidence: `verification-extraction-audit.json`, `explicit-python-format-audit.json`,
`code-pool-summary.json`, `audit.json`.

## Next bounded phase

First diversify the selection from verified pools and the 751 raw documents
outside this prefix. Deduplicate problem identities across modes/aliases; keep
whole problem families in one split. Audit final format and selected test suites
with boundary, signed-value, multiplicity, type and in-place mutation/property
checks. Do not convert evaluation failures into training rows.

For aggregation, extend the existing executed-world machinery. Vary one to five
pages, page sizes, schemas, tool ordering, irrelevant records, repeated IDs,
zero/signed values and natural request wording. Include both read-required
records and result pages that already supply enough values. Prove filtering,
deduplication, evidence completeness and arithmetic separately.

Supervised anchors should cover every assistant action through the verified
answer. Collect certified student negatives on fresh incomplete-evidence states,
including bare partial totals and prose early completion. Retain one-page,
empty-result, already-complete, direct-answer and permission-denied/impossible
contrasts. Cursors and recovery tokens remain separate concepts. Broaden actual
schema/state shapes; randomized IDs inside one template are not enough.

Prepare a cleaned, diversified replay control and an arm with that exact replay
plus fresh aggregation supervision. Initially hold LR and preference coefficient
fixed. Audit actual weighted source/assistant exposure; keep roughly the current
raw-code share, replacing some unverified expanded-code exposure with verified
solutions rather than indiscriminately reducing unrelated domains. Derive step
count from a bounded token budget and complete-trajectory coverage, rather than
repeating another arbitrary 40 steps over four worlds.

Gates: disjoint closed-loop aggregation worlds; existing recovery, pagination,
empty/direct-answer screens; write authorization and wrong-record selection; both
code decoding modes; math, MMLU/ARC and paired NLL. Prefer a fresh internal code
development bank screened against known historical inputs for selection; report
public benchmarks separately. Stage-start/control code probes are required for
causal attribution or choosing a different initialization for promotion. Do not
repeatedly select checkpoints on the same public failure cases.

## External sources if the existing pool leaves gaps

| Source | Fit and decision |
| --- | --- |
| [KodCode-V1](https://huggingface.co/datasets/KodCode/KodCode-V1) | Closest extension of the current function/test pipeline. Use fresh problem IDs after measuring gaps; benchmark-similarity metadata is not an exclusion proof. Card license: CC BY-NC 4.0. |
| [SWE-smith trajectories](https://huggingface.co/datasets/SWE-bench/SWE-smith-trajectories) | Later agentic repository coding: filter resolved paths, preserve execution evidence, adapt protocol/masks and retain complete bounded paths. Not a direct pagination repair. Card license: MIT. |
| [DeepCoder preview](https://huggingface.co/datasets/agentica-org/DeepCoder-Preview-Dataset) | Competitive-programming problems/tests, including LCB/TACO-derived material; requires holdout tracking and harness adaptation. Lower priority than current function/test pools. See the [official preparation script](https://github.com/agentica-project/rllm/blob/main/examples/deepcoder/prepare_deepcoder_data.py). |
| [CodeI/O](https://github.com/alexseong/codeio) | Useful pattern for fresh execution-verifiable input/output reasoning and revision. Only PythonEdu-Reasoning is released; not a replacement for solution/agent trajectories. |

Candidate metadata revisions are pinned in the audit directory. No new dataset
rows or runtime stack were installed. These fit assessments do not demonstrate
an improvement in this student. Prioritize existing verified pools and fresh
executed aggregation worlds.

## Reproduction

From the repository root, use the existing Python environment:

```powershell
.venv/Scripts/python.exe scratch/dense_gr/code_dataset_audit.py --screen-code
.venv/Scripts/python.exe scratch/dense_gr/code_replay_review.py --screen-remainder
```

Reuse the recorded `exclusion-build.json` invocation for those exact caches, then:

```powershell
.venv/Scripts/python.exe scratch/dense_gr/code_dataset_audit.py --recheck-extraction
.venv/Scripts/python.exe -m pytest tests/test_code_replay_review.py -q
```

Three focused tests pass: import relocation, modified-contract retention and
generic-stem/interface discrimination. No training launcher was changed.

# Aggregation and code-preservation assessment

Assisted-by: Codex

2026-10-08. Candidate: completion step 40. Base: `merges-long1/u50`.
This assessment does not start another training stage or promote the candidate.

## Increasing raw code is not yet the supported fix

The independent code bank worsens versus the stage's starting candidate by
0.000728 nats (paired 95% CI +0.000349 to +0.001094), and versus its matched
control by 0.000820 (+0.000396 to +0.001261). But the broad raw llama.cpp code
slice improves versus the starting candidate by 0.001780 nats, and teacher-code
assistant targets improve by 0.001061. Both lag the replay control's larger gains.
These are different distributions; a single increase in raw code is not an
established remedy for the independent code loss.

The inherited plan already intends approximately 39.9% raw-code exposure, plus
short, expanded and teacher-code sources. That is an intended source share, not
the measured weighted-token share of these 40 steps. The stage also repeats the
same finite replay documents used in earlier comparisons. More repeated exposure
could improve in-distribution NLL while sacrificing other capabilities. GSM8K
falls equally in the treatment and control, strengthening the case for reviewing
the shared replay/stage rather than attributing every regression to aggregation.

Run executable code confirmation before changing weights. If functional code
loss is confirmed, compare a bounded code-rebalance arm using more diverse,
verified solution conversations and raw code against an unchanged control.
Preserve assistant-only conversational masks and full-token raw code. Freeze and
measure actual source/token exposure; do not blindly increase raw-code repeats
or silently substitute a different replay schedule. Consider fresh documents and
smaller intervention dose before another pass over the same replay prefix.

## Why aggregation still fails

The independent nine-case suite's main failure is early completion after reading
the first matching record, despite `next_cursor`. The bare answer is commonly
`5`; some failures additionally misuse the cursor as a recovery token. They do
not reach the full record set, so arithmetic is not the primary bottleneck.

The stage had 24 aggregation assistant turns, but only four distinct complete
trajectories: two ordinary and two overlapping-page worlds. Templates, namespace
families and prefix structure are narrow. Positive paths are executed and valid,
but increasing their weight would mostly repeat already-familiar prefixes.

Additional reference scoring used the existing `ref_logprobs.py` on those 24
turns. At the four critical post-first-read training prefixes, the correct full
cursor-call response already had sum log-probabilities -3.82, -0.88, -0.51 and
-0.36 before training. They rise to -1.42, -0.07, -0.17 and -0.07. The curated
early-total responses become substantially less likely. These familiar examples
are learned; this alone does not establish generalized sequential behavior.

An evaluation-only probe pairs the correct next-page response against the actual
bare partial answer `5` at nine independent post-first-read prefixes. For the
short Orchid case, complete-response log probabilities change from -9.05/-1.60
to -11.79/-0.45 (continue/partial answer). This is a transfer failure despite
stronger performance on training prefixes. Complete-response probabilities have
different lengths and do not equal greedy first-token decision probabilities;
the live failed trajectories are the direct behavioral evidence. All nine probes
were retained and none exceeded the 16K evaluation cap. These exact evaluation
prompts must never enter training.

The next data adjustment should target the state "have some values, still have
unread records/pages" across many fresh executed worlds:

- Vary names, tool and argument schemas, request wording and tool ordering;
  include natural requests without explicitly narrating each action.
- Vary page counts, entries per page, nonmatching entries, duplicate IDs and
  value ranges. Verify completeness, name filtering and deduplication separately.
- Collect student continuations on fresh training worlds, including partial and
  failed prefixes, and pair certified early stops with an executed continuation
  through the final verified answer. Use bare and prose partial totals, not only
  the repeated phrase "Total units".
- Include one-page, no-match, already-complete and direct-answer contrasts so
  continuing until sufficient evidence does not become unconditional tool use.
- Include incorrect-record selection and invented-token recovery contrasts;
  the current candidate has a genuine wrong-record write attempt on a long task.

Use diverse successful full trajectories as supervised anchors as well as
preferences. Evaluate closed-loop outcomes on disjoint worlds and existing
frozen screens. Scale distinct trajectories before increasing loss weight or
extending training on the same four paths. Finish the downstream confirmation
before choosing the next initialization, mix, dose or checkpoint.

## Frozen gauntlet

Existing generators, independent evaluator, overlap scanner and Docker scorer
are reused. Existing dataset caches and the `code-bench-sandbox` image are present;
no new environment, model download or scoring binary is needed. No named
"gauntlet" launcher was found, so `completion_gauntlet.py` orchestrates these
documented downstream tools with frozen commands, sources and inputs.

- HumanEval+: all 164 cached tasks; MBPP+: all 378 cached tasks. Three sampled
  thinking seeds and one greedy nonthinking run per checkpoint, 2K output cap.
- GSM8K: all 1,319 test tasks; MATH-500: all 500 tasks. Three sampled thinking
  seeds and greedy nonthinking, 4K output cap. Report correctness, missing
  answers and truncation alongside paired losses/gains.
- Existing MMLU/ARC confirmation splits (256 questions each), with raw, token-normalized and
  character-normalized accuracy and paired intervals.
- Existing completed agent, safety and paired NLL evidence remains part of the
  assessment; downstream success does not erase the wrong-record attempt.

Both arms use batch 8, identical prompts/decoding, both stop tokens (248044,
248046), and per-case RNG for sampled runs. Math's existing CLI gained opt-in
dual-stop/per-case-seed flags, reusing the code generator's existing mechanisms;
historical defaults remain compatible. Each arm has an isolated compile cache.
Generation uses one GPU per checkpoint. Code is executed only through the
existing no-network Docker sandbox; scoring containers run serially.

The established every-field/decoded-input 13-word local-capture overlap scanner
also screens the full math bank. Results are reported for all tasks and the
unmatched subset. Public benchmark exposure remains possible; this is not a
claim of clean foundation pretraining. Multiple comparisons are exploratory,
and sampled intervals resample tasks together across seeds.

The initial unfired protocol draft was retained while completing the reporting
checks. The final frozen protocol is in `scratch/dense_gr/completion-gauntlet/`.
No new training was authorized or launched as part of this assessment.

Two focused benchmark checks pass. Code and math generation smoke checks passed
for both checkpoints, and the independent MMLU/ARC two-question loading/scoring
check passed. These validate interfaces, not capability scores: generation smoke
budgets were deliberately short. Preflight also caught and corrected an
import-order difference in the Triton metadata alias; the final protocol binds
installed distributions and preserves the unfired earlier drafts.

The full worker was launched on 2026-10-08 after this assessment, with u50 on
GPU 0 and completion step 40 on GPU 1. It verifies the frozen protocol, runs all
generation and the local overlap screen, then performs sandbox scoring and
paired comparisons. Live status/logs/receipts are in `completion-gauntlet/`.
No automatic training, checkpoint promotion or further run is scheduled.

## Completed confirmation

The table below records the original frozen extraction protocol. The subsequent
[code failure investigation](CODE_FAILURE_INVESTIGATION.md) found that this
extractor sometimes scores reasoning drafts instead of final answers. Separate
final-answer rescoring narrows sampled HumanEval's decline to -4.47 points
[-8.54, -0.41] and MBPP's to -2.20 [-5.03, +0.53]. HumanEval remains concerning;
MBPP is unresolved after the grading correction. Use those diagnostic results
for conclusions about the final solutions while preserving the original run.

The worker completed all generation, overlap screening, sandbox scoring and
paired comparisons on 2026-10-08 at 18:02 Chicago time, without a reported job
failure. No checkpoint was promoted. Evidence: `completion-gauntlet/comparison.json`.

| Benchmark | u50 greedy | Step 40 greedy | u50 sampled mean | Step 40 sampled mean |
| --- | ---: | ---: | ---: | ---: |
| HumanEval+ | 40.85% | 41.46% | 40.24% | 34.55% |
| MBPP+ | 47.88% | 47.62% | 41.80% | 38.71% |
| GSM8K | 70.58% | 70.96% | 76.27% | 76.90% |
| MATH-500 | 51.20% | 50.40% | 51.40% | 51.80% |

Sampled means are pass@1 across three matched seeds, not pass@3. Code correctness
requires passing both the original and expanded tests. Sampled HumanEval+
candidate-minus-base is -5.69 percentage points, paired task-bootstrap 95% CI
[-9.76, -1.42]; MBPP+ is -3.09 [-6.17, -0.09]. These intervals are unadjusted
exploratory comparisons across benchmarks. Both code benchmarks decline in all
three sampled seeds; greedy differences are small and unresolved.

GSM8K sampled delta is +0.63 points [-0.63, +1.92], MATH-500 +0.40
[-2.00, +2.80]. Local-capture-unmatched math subsets also have intervals crossing
zero. Raw MMLU accuracy differs by one correct answer in favor of the candidate
out of 256; ARC raw accuracy is unchanged. These do not establish equivalence.

The regression is decoding-mode-specific within this test: thinking also uses
sampling, whereas nonthinking is greedy, so the design cannot separate those
two factors. Code caps are 2K tokens; truncations increase from 26 to 30 across
HumanEval thinking seeds, and 65 to 101 across MBPP thinking seeds. Truncation
may explain part of the loss, but does not establish its full cause. MATH thinking
also reaches the 4K cap often (286 base, 324 candidate across seeds).

Recommendation: retain u50 as the base and review code failure categories and
completion length before picking the next initialization or training mix. The
functional code loss now supports a bounded code-preservation adjustment, but
does not identify raw source code as the best repair. Prefer fresh verified code
solution conversations alongside raw code, retain conversational masks, and use
a matched control. Combine this with the diverse aggregation data described
above only after documenting the dose and preservation gates. No additional
training or diagnostic generation was launched automatically.

## Subsequent replay assessment

The [data assessment](CODE_DATASET_ASSESSMENT.md) screened all 16 current replay
caches and confirmed 104 documents matching 103 public code tasks. A separately
screened final-answer comparison leaves 122 HumanEval and 317 MBPP tasks: sampled
deltas -3.55 [-8.20, +1.09] and -1.89 [-4.94, +1.05] percentage points. Both
intervals cross zero. Code preservation remains a concern, but statistical
significance from the unscreened HumanEval bank should not be presented as an
independent functional-code finding.

The future exclusion union is 4,279 IDs, without changing historical inputs.
Use diverse existing verified solutions (392 teacher documents outside the
recent prefix), broaden raw-document sampling and build fresh complete executed
aggregation worlds. First/last-final-fence checks also identify a small data
qualification mismatch. The next bounded phase should measure real target
exposure and whole-trajectory coverage, rather than increase raw-code weight
or repeat the four aggregation worlds. This assessment starts no training.

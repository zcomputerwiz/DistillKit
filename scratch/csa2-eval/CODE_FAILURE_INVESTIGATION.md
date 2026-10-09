# Code failure investigation

Assisted-by: Codex

2026-10-08. Completion step 40 versus u50. No training, checkpoint promotion or
new model generation is part of this investigation.

## Findings

The original scores overstate the code decline because the extractor can score
reasoning drafts. Rescoring final answers improves both arms, but does not erase
the HumanEval regression. MBPP's corrected interval crosses zero.

These are unscreened public-bank results. The subsequent
[replay data assessment](CODE_DATASET_ASSESSMENT.md) confirms 103 task identities
in the retained training pool. Removing those tasks leaves HumanEval -3.55 points
[-8.20, +1.09] and MBPP -1.89 [-4.94, +1.05]. Both screened intervals cross zero;
the unscreened HumanEval significance is not evidence of a statistically clear
independent functional-code decline. The historical table below is preserved.

| Sampled thinking pass@1, mean of three seeds | u50 | Step 40 | Delta and paired 95% CI, percentage points |
| --- | ---: | ---: | --- |
| HumanEval+, original extraction | 40.24% | 34.55% | -5.69 [-9.76, -1.42] |
| HumanEval+, final-answer extraction | 42.68% | 38.21% | -4.47 [-8.54, -0.41] |
| MBPP+, original extraction | 41.80% | 38.71% | -3.09 [-6.17, -0.09] |
| MBPP+, final-answer extraction | 45.41% | 43.21% | -2.20 [-5.03, +0.53] |

Intervals resample tasks together across three matched seeds; these are
exploratory unadjusted intervals, not independent trials or pass@3. All three
corrected seeds decline on both code banks. Corrected HumanEval counts are
68/71/71 versus 59/64/65; MBPP 169/170/176 versus 167/162/161.

There are 422 changed extractions across both banks and arms. HumanEval rescues
12 u50 and 18 candidate samples, with no passing answers broken; MBPP rescues
43 and 53 but breaks two previously passing drafts in each arm. A correct draft
followed by an incorrect final answer should not count as a correct final answer.
The separate diagnostic and image/scorer/input hashes are in
`completion-gauntlet/failure-audit/final-answer-comparison.json`.

After final extraction, HumanEval has 76 lost and 54 gained task/seed pairs.
Candidate failures in the lost pairs are 59 basic-test failures, 11 expanded-test
failures, four caps before the final answer and two syntax errors. Gains also
include four capped u50 failures: the cap-before-final category has no net loss.
MBPP has 129 lost and 104 gained pairs; thirteen losses and six gains are caps
before final answers. The remaining net loss mostly comes from basic contract
and expanded-case failures. These categories describe observed failures, not
the causal training mechanism, and repeated task/seeds are not independent units.

## Method and a scoring issue

All frozen code completions and their scores were audited: 164 HumanEval+ and
378 MBPP+ tasks, greedy nonthinking and three thinking/sampling seeds, for both
checkpoints. Prompt identities and complete inventories were checked. Source
hashes and descriptive counts are in `completion-gauntlet/failure-audit/summary.json`.
Parsing uses Python AST only; generated programs are executed exclusively in
the existing no-network, no-host-mount Docker sandbox.

The original generator extracts the first fenced block anywhere in the raw
response, including reasoning. This sometimes selects an incomplete draft
instead of the completed answer. If there is no fence, it submits reasoning
prose together with an otherwise valid unfenced final function. These are
scoring/channel extraction failures, not necessarily solution failures. Both
arms are affected. For example, candidate HumanEval/29 seed 0 drafts the correct
prefix filter without importing `List`, then supplies that import in the final
answer; the original evaluator executes the draft.

A separate diagnostic extracts the first fence after `</think>`, or the raw
final source if unfenced. It makes no repairs and preserves original extraction
when no completed reasoning boundary exists. Only changed extractions are
rescored; their scores are merged with the original unchanged cases. The
original frozen outputs, scores, source and comparison remain untouched.

The same Docker image, tests, per-case resource limits and scorer are reused.
Two focused tests cover draft-versus-final selection and preservation of
unfinished reasoning. The first diagnostic launcher omitted the established
PowerShell process execution setting and stopped before scoring; it was retried
using the project's existing invocation, with identical diagnostic inputs.

## Concrete solution failures

These are examples for diagnosis, never proposed training rows:

| Task / sampled seed | Candidate error | Consequence |
| --- | --- | --- |
| HumanEval/14 / 0 | Joins prefixes into a string instead of returning a list | Violates return type and the supplied example |
| HumanEval/27 / 0 | Calls `lower()` on lowercase and `upper()` on uppercase | Leaves ASCII letter case unchanged |
| HumanEval/0 / 0 | Uses `<= threshold` instead of strict `<` | Fails boundary cases in expanded tests |
| HumanEval/0 / 1 | Rejects lists of fewer than three elements | Incorrectly rejects a close pair of two elements |
| HumanEval/0 / 2 | Subtracts the larger sorted neighbor from the smaller | Treats distant pairs as close |
| MBPP/232 / 0 | Uses `set()` before selecting largest items | Loses multiplicity despite the list-valued specification |

Several candidate reasoning traces confidently assert an incorrect intermediate
calculation or reinterpret the request to fit an imagined example. Checking only
syntax or the output format would miss these. The original basic/expanded test
split is useful: some failures violate the visible/basic contract, while others
need boundary, duplicate, signed-value or other expanded cases.

## Output length and unfinished reasoning

Original sampled mean generated length rises from about 570 to 626 tokens on
HumanEval and 412 to 462 on MBPP. Many capped failures contain no final answer.
Examples include repeating variants of "negative/negatives" (HumanEval/136),
enumerating a long incorrect number sequence (MBPP/86), and repeatedly guessing
the star-number formula (MBPP/268). Other cases spend the budget manually
recomputing supplied examples. Simply raising the cap is not an established
repair for these loops.

The original scoring decline remains on the 132 HumanEval and 299 MBPP tasks
where all six sampled outputs avoid truncation: -5.81 percentage points
[-10.61, -1.26] and -3.57 [-7.13, -0.11], respectively. This is a descriptive
subset selected on generated lengths, not an unbiased budget intervention; it
also retains the original extraction issue. Corrected final-channel results
must take precedence for claims about actual solutions.

After correcting extraction, the same nontruncated subsets have HumanEval
-4.29 points [-8.84, +0.25] and MBPP -2.34 [-5.47, +0.89]. Point estimates still
decline, but both conditional intervals now cross zero. Thus the original
conditional significance does not survive the grading correction. The main
unconditional HumanEval result remains the stronger evidence.

## Replay exposure and interpretation

The actual 40-step ledger sums to 966,679 weighted replay targets. Raw code is
324,423 (33.56%), from 17 visits to 17 distinct long documents. Expanded-code,
teacher-code and short-code sources contribute another 57,405, 51,271 and
16,357 targets (5.94%, 5.30% and 1.69%). General thinking and think-first sources
contribute 45,391 and 35,047 (4.70% and 3.63%). The narrow long raw-code sample
is an important distinction from the inherited plan's intended source share.

The gauntlet compares u50 with a candidate that has undergone multiple stages.
It does not isolate the completion-pair intervention from previous training or
the shared replay objective. Small teacher-forced NLL changes also do not
localize a closed-loop generation failure. Separate stage-start and matched
control code tests would be needed for causal attribution. Thinking versus
nonthinking is additionally confounded with sampling versus greedy decoding.

For the next bounded phase, prioritize fresh execution-verified solution
conversations covering basic contracts, boundary cases, duplicate handling and
short accurate reasoning that reaches a complete final answer. Agentic coding
with tool execution and standalone coding should both be retained. Keep
assistant-only conversation masks and full-token raw-code loss. Use disjoint
tasks and overlap checks; do not convert these evaluation failures into training
data. Include diverse raw code, but do not assume more weight on the same 17
documents will repair reasoning or final-channel behavior. A matched control
and both decoding modes remain necessary preservation gates.

Fix final-channel extraction in the next explicitly versioned benchmark
protocol before using code scores as training or promotion gates. Keep separate
instruction-format checks; accepting unfenced final source for correctness does
not mean the requested fenced format was followed. The historical gauntlet was
not silently relabeled. A bigger generation cap, raw-code weight increase, new
training run or benchmark-based repair dataset was not launched.

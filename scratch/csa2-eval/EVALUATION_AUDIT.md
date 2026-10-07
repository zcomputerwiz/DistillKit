# Replay targets, grading, and proxy audit

Assisted-by: Codex

Date: 2026-10-07. Both 20-step pilot-2 arms completed. Keep u50 while the
evaluation gaps below are resolved. No additional training or checkpoint
promotion was performed during this audit.

Follow-up: [the next phase plan](NEXT_TRAINING_PHASE.md) corrects live grading,
freezes broader retention and independently generated agent development tests,
and replaces the overlapping math proxy. The user chose masked replay only
against u50. Its finite 40-step schedule supersedes the earlier two-arm dry plan
below; no follow-up training has started.

## Actual replay objectives

Reconstructed the exact 40 microbatches from each arm (same seed, repetition,
balanced prefix, role masks, hedge suppression and repeat masks). Weighted
target totals match the frozen plan exactly: 491,132 candidate and 643,149
control. Synthetic teacher placeholders contribute no KL: they are CE-only.

| Sampled weighted targets | Agentic v2 | Replay control |
|---|---:|---:|
| User | 85,243 | 85,081 |
| System | 16,370 | 26,107 |
| Tool results | 0 | 651 |
| Assistant | 148,784 | 186,946 |
| Thinking | 111,808 | 165,502 |
| Tool calls | 38,748 | 27,742 |
| Plain text/code | 77,154 | 138,325 |

The remainder is structural tokens. These are position weights, not gradient
norms. CE/KL coefficient masses are 217,704/263,855 for candidate and
212,094.5/416,167.5 for control; repeat positions excluded from KL are reported
separately (9,573/14,887 weighted positions).

New agent data and four explicitly masked replay caches are assistant-only.
Other legacy conversational caches still score context. At user positions,
46.6% of candidate and 48.1% of control KL coefficient mass has a teacher top-1
stop token while the actual next token is not a stop. Top-1 agreement with
actual user tokens is only 33.9%/31.0%. For assistant positions, agreement is
89.2%/90.2%; premature-stop top-1 is about 0.13%/0.12%. The conflict is mainly
in context, not evidence of generally broken assistant supervision. Examples
also include harmless newline and code-fence-versus-prose differences.
The saved high-confidence examples are a first-hit diagnostic, not a random
or balanced sample for estimating how often an error occurs.

This conflicts with a blanket interpretation of 'prompts are masked'. The user
approved extending assistant-only masking to all conversational replay while
preserving full-token raw code/text. New CLI plans now default to this policy;
archived plan reconstruction preserves the historical setting. The raw
`frontier-code-raw` and `general-pilot-w8` caches retain full-token supervision.
CE/KL/UL assignments are unchanged. Recalculate exposure after masking;
identical optimizer steps do not guarantee identical supervised training.
The context conflict is a hypothesis for observed drift, not a causal claim
proven by this audit. No new training was launched as part of the audit.

The recalculated dry-run plan is `scratch/dense_gr/agentic-v3-masked-plan/plan.json`.
At the same illustrative 20-step settings, candidate/control exposure becomes
435,665/523,153 weighted targets. The candidate prefix contains 17,453 new
targets (4.006%), 108 distinct new documents, and at most two visits each.
Epoch-level 5% targeting changes the new-data repetition factor from 78 to 67.
This is a measured plan, not a launch recommendation: the control still has
unequal source exposure (for example, teacher-code 15,351 versus 560 targets).
Match replay exposure before treating the next comparison as causal.

The full planned 40-microbatch prefix of both masked arms was then read and
audited on CPU. Both have exactly zero weighted user/system/tool-result targets
in conversational sources, and both match the plan totals. Raw code contributes
150,074/135,824 targets; raw general text contributes 381/635. Evidence:
`evaluation-audit/masked/verification.json` and `replay-targets.json`.
All audit processes exited; no training or background probe remains running.

## The retention interpretation changes

The original code proxy includes 16,848 user tokens out of 31,090 targets
(54.2%). Its 64 documents are the lexicographically first of 292 eligible
documents, all named Magicoder-OSS; that is not a source-stratified sample.
The thinking proxy selects the first 64 of 300 documents, also by ID.
Only one selected code document and no thinking documents hit the 1,024-token
prefix cap. Truncation is not the main problem in these particular NLL banks.

Recomputed on exactly those documents using existing `atlas.py nll` and saved
per-token evidence:

| Code NLL | u50 | Agentic v2 | Replay control |
|---|---:|---:|---:|
| All tokens | 0.778152 | 0.812352 | 0.804949 |
| User prompts | 1.100129 | 1.162774 | 1.148118 |
| Assistant answers | 0.405890 | 0.407909 | 0.408996 |

Candidate assistant-answer delta is +0.002020 (paired document-bootstrap 95%
interval +0.001112 to +0.002943); control is +0.003106 (+0.002058 to +0.004123).
Thus the aggregate increase substantially overstates answer-loss regression.
This does not prove execution correctness is preserved. The small numerical
offset from historical proxy NLL (~0.00012) is present in this recomputation
through the atlas evaluation path; all three arms here use that same path.
Do not mix their absolute numbers with the historical proxy calculation.

Thinking own-turn NLL: 0.454972 / 0.457151 / 0.455973, respectively.
Existing broad atlas results also matter: candidate teacher-code own-turn NLL
improves by 0.002574, agent own-turn domains improve, and QA own-turn NLL
improves by 0.054480. These are teacher-forced diagnostics, not successful
end-to-end tasks. Do not reject or promote based on one aggregate NLL.

## Math proxy overlap and grading limits

Exact complete-question token matching against retained training documents
finds 16/256 GSM8K and 48/256 MATH bank questions in replay. This is a lower
bound: it misses paraphrases and tokenization differences. Retained-data
overlap does not establish that every matching document was sampled in these
20 steps, or that u50 had never previously seen the remaining questions.
The bank only excludes round-2 rollout IDs, not later captures.

The original results therefore remain legacy diagnostics, not held-out math
accuracy. Their raw generations were not saved, so overlap-free scores cannot
be recovered from aggregates. MATH also hits the 1,024-token cap on 105/256
base, 97/256 candidate, and 111/256 control cases. It measures budget-limited
behavior as well as reasoning. The sampled result uses one seed; the generator
shares its RNG across batches, so seed equality does not guarantee identical
per-case random draws after different earlier stopping times. The legacy
generator stops on tokenizer EOS 248046 only, whereas serving and live tool
evaluation use both 248044 and 248046. Unboxed answers are format failures;
without retained text they cannot be distinguished from wrong answers.

## Live grading audit

The original evaluator conflates any invalid call with task failure, including
a recovered unknown-ID read. It also recognizes questions through a short
verb list, missing 'please try a different name'. Conversely, it can accept
wrong-ID reads and negated state assertions through substring matching.
The no-call explanation check accepts any response containing 'review',
including explanations that incorrectly equate review with stale-session
refresh. Its simulated user only responds to clarification mentioning both
'north' and 'south'; other legitimate clarification wording needs continuation,
not an invented offline success.

`agentic_grade_audit.py` replays every saved tool call, checks recorded results
against the deterministic environment, and saves separate outcomes, invalid
attempts, false-completion flags, and review flags. It preserves original files
and scores. No model was regenerated to exploit a changed grader.

| Auditable action tasks (44; explanations excluded) | u50 | Agentic v2 | Replay |
|---|---:|---:|---:|
| Completed with valid final state/report, allowing recovery | 33 | 36 | 33 |
| Same, with no recorded invalid attempts | 30 | 29 | 30 |
| False completion after stale-session failure | 1 | 1 | 1 |

Four explanation tasks per arm are explicitly left for semantic review,
not automatically passed. The new language checks are conservative audit
heuristics, not a general natural-language grader. Recovered errors remain
visible; blocked unauthorized writes still fail. The candidate's higher
completion count does not demonstrate more reliable or cleaner behavior.
All variants still struggle with unresolved choices, and candidate stale-session
handbacks are real failures. A previously rejected unknown-ID read and a false
completion are different phenomena and must not share one undifferentiated score.

Seven grading regression tests cover recovery, wrong-record reads, negation, false
completion, empty-search fallback, missing clarification continuation, and
blocked writes. No-call semantics and arbitrary prose still require review.

## What a representative next evaluation must contain

The current proxies are not sufficient to select the next production model.
Use a frozen, versioned development suite, and keep final benchmarks separate:

1. **Agent outcomes:** new task templates, not just renamed schemas. Include
   read-only lookup, search/pagination, failed calls with recoverable errors,
   conflicting/empty results, known IDs, genuine user choices, multi-step
   dependencies, no-op updates, post-tool synthesis, and truthful completion.
   Score final state, policy violations, needless handbacks, announcements
   without calls, and efficiency separately. Add paraphrases and repeated
   rollouts. A user's decision is different from information a tool can fetch.
2. **Code:** sample across source, task type, and length; report assistant-answer
   NLL separately. Confirm finalists using executable tests. Existing
   `scratch/downstream/code_bench/generate.py`, `run_docker.ps1`, and the installed
   `code-bench-sandbox` image already support HumanEval+/MBPP+ checks with no
   network or host mounts during generated-code execution. No replacement
   sandbox needs to be built. Avoid tuning repeated recipe choices on these
   final benchmark results.
3. **Math:** freeze development questions screened against all prior/current
   captures, with source IDs and text/near-duplicate checks. Preserve raw
   completions, token IDs, per-case scores, both stop tokens, seeds, truncation
   and formatting flags. Pair a deployment token budget with a longer-budget
   sensitivity subset; use multiple seeds for sampled reasoning. Excluding
   exact matches alone is insufficient to certify a clean bank.
4. **General/long-context:** retain the existing QA and multi-agent atlas, with
   own-turn loss separated from observation loss. Add generated answer/tool
   outcomes across context-length bins. Short-context NLL cannot establish
   long-context retrieval or resistance to misleading tool content.
5. **Causal comparisons:** equalize replay targets and schedule exposure, not
   only optimizer steps. This pilot used 491,132 versus 643,149 weighted targets,
   so candidate-minus-control is not a pure estimate of curriculum benefit.

This is consistent with [tau-bench](https://arxiv.org/abs/2406.12045), which
evaluates interactive tool tasks and repeated-run reliability, and
[EvalPlus](https://arxiv.org/abs/2305.01210), which evaluates generated code with
expanded functional tests. Neither supports treating imitation loss or exact
reference-call matching as a substitute for successful execution.

## Evidence and reproduction

Local evidence is under `scratch/dense_gr/evaluation-audit/`: `proxy-audit.json`
(document IDs and exact overlap matches), `math-bank.json`, `proxy-domains.pt`,
`proxy-roles/nll.json` and token evidence, and `grade-{base,agentic,replay}.json`.
The original checkpoint comparison remains under `agentic-pilot-v2-balanced/`.

Run `replay_target_audit.py proxies` on CPU to reproduce overlap and the exact
legacy NLL bank; run `targets` to inspect the actual 40 sampled microbatches.
Use `atlas.py nll --domains .../proxy-domains.pt --arm NAME=CHECKPOINT
--output-dir OUTPUT --save-token-evidence` for role-separated losses.
Use `agentic_grade_audit.py SOURCE OUTPUT` for saved-trace rescoring; it refuses
to overwrite an audit. Tests: `pytest tests/test_agentic_grade_audit.py -q`.
The broader relevant suite passed 29 tests (grading, curriculum, coverage,
tool behavior and repeat dispersal).

# Masked replay phase: results and follow-up

Assisted-by: Codex

The authorized bounded phase completed on October 7, 2026 at 19:10 Chicago
time. All nine baseline stages, 40 masked replay updates, all nine candidate
stages and both paired retention evaluations completed successfully. No model
was promoted. The checkpoint remains a development control; u50 is the adopted
reference. This experiment measures a masked replay update, not the causal effect
of masking against otherwise identical unmasked training.

## Execution and memory

Training consumed exactly 966,679 weighted targets over 40 steps. Model configs
match u50. The frozen order verifier checks the same 80 microbatches and zero
weighted conversational context targets. Raw code/text training retains full-token
loss. Training reported 1,108 measured-step tokens/second, peak allocations of
15.69 and 14.88 GiB, valid spill telemetry and no spill. End-to-end training
throughput, including its measured overheads, was 1,004 tokens/second.

Both long agent runs completed using the validated 256-query prefill adapter.
The interrupted, spilled run is archived; the eight other baseline outputs were
retained. Long-run final logs show approximately 7.7 GiB peak active allocations
for two-row 11K prefills. Allocator reservations grew to about 20.9 GiB; reserved
memory is not the active tensor allocation. The adapter's full boolean masks
remain quadratic, so these results do not certify arbitrary context lengths.

## Behavior and math

These are the frozen automatic scores; separate semantic annotations are in
`phase3-masking/reviews/post-run-semantic-review.json`.

| Screen | u50 | Masked replay |
| --- | ---: | ---: |
| Long agent task success | 35/72 | 35/72 |
| Long agent clean success | 24/72 | 33/72 |
| Long agent unauthorized cases | 10 | 5 |
| Long agent false completions | 4 | 7 |
| Legacy agent success | 33/48 | 32/48 |
| Legacy agent clean success | 30/48 | 31/48 |
| Short sampled success, seeds 1/2/3 | 15/15/14 of 24 | 14/13/14 of 24 |
| GSM8K development, seeds 0/1/2 | 48/47/47 of 64 | 50/50/50 of 64 |
| MATH development, seeds 0/1/2 | 32/32/33 of 64 | 32/29/32 of 64 |
| Longer prefix subset, GSM8K/MATH | 14/10 of 16 | 14/10 of 16 |

The improved clean count does not establish improved reliability. Four new
greedy recovery cases and one legacy recovery case falsely report success after
session renewal, without retrying the failed write. Both arms already exhibit
this mechanism; the candidate extends it to additional cases. New unauthorized
attempts also occur at some previously passing injection cases despite the
smaller total. These are genuine recorded failures, not grader wording issues.

Legacy explanation review gives 36/48 for u50 and 33/48 for the candidate
(reviewed clean counts 33 and 32). The candidate often defines "reviewed" as a
state mutation. Correct reference trajectories define it as checking/verifying.

The numeric grader requires exactly one number in the entire report. Correct
totals with intermediate units and repeated arithmetic answers can be flagged.
Some flagged responses contain real errors too: both arms answer 27 on one
8+9 case, a candidate report adds 5 and 17 to obtain 34, and some reports omit a
record or make unsupported claims. Preserve automatic scores, annotate semantics
separately, and retain tool errors and number-only format violations. No frozen
bank or scoring rule was changed after observing results.

## Retention

Saved token evidence supports the existing paired document bootstrap method
(2,000 draws, seed 0). Values below are candidate minus u50, nats per target;
positive values mean worse prediction loss.

| Independent retention slice | Delta | 95% document interval |
| --- | ---: | --- |
| Code own turns, 292 docs | +0.003205 | [+0.002623, +0.003770] |
| Reasoning own turns, 300 docs | +0.002315 | [+0.001863, +0.002796] |

These are small, measurable regressions, below the frozen 0.01/0.02 investigation
thresholds. The code slice's tiny 292-target thinking role rises by 0.04186;
its small size and role content need to be considered separately from aggregate
code capability. Neither NLL nor these development intervals certifies executable
code correctness or establishes powered noninferiority.

Broader atlas own-turn NLL improves across agent traces (roughly -0.010 to
-0.020), tool calls, teacher code, math and QA. The llama.cpp raw-code slice
improves by 0.00829 nat and general raw text by 0.00097 nat. This is consistent
with improved imitation on those fixed documents, while closed-loop task
outcomes remain mixed. It does not establish agent reliability.

## Next steps and gates

1. The seed-1 full-bank 4,096-token budget diagnostic is complete in both arms, with
   the same prompts, per-case seeds and batch size 8. Grade each long trajectory's
   first 2,048 tokens too. Even at matched batch size, static-cache capacity can
   change numerical trajectories, so compare saved token prefixes before assigning
   cross-run differences solely to the output cap. The diagnostic used existing
   `math_dev_eval.py` and does not change the frozen primary experiment.
2. Check baseline inference repeatability before interpreting the wider-budget
   MATH gap or freezing a new training rate. If the capability loss persists,
   use a smaller-rate masked replay pilot with the same frozen replay documents
   and exposure before adding agent data. Preserve this run as the measured
   development control. Keep u50 as adopted model; do not continue
   this checkpoint automatically or run final-test confirmation to choose another
   recipe while the closed-loop regressions remain unresolved.
3. Prepare a separate curriculum experiment addressing completed tool transactions:
   retry after recovery, verify mutation success before reporting it, follow
   pagination rather than inventing session tokens, obey read-only requests despite
   tool-text instructions, and stop after a complete empty search. Include direct
   no-tool answers and read-only tasks to avoid teaching indiscriminate tool use.
4. Match the exact masked replay document exposure and learning-rate schedule
   against its control, retaining raw code/text, math and reasoning sources. Keep
   conversational context masked. Use verified successful trajectories and realistic
   varied schemas/observations; the old 20-step `agentic-v3-masked-plan` is not a
   frozen replacement for this controlled design.
5. Do not train on the frozen evaluation cases or copy their identifiers/prompts.
   Retain the fixed development screens, add a separately frozen natural task
   transfer screen before training, and distinguish task outcome, final factual
   report, unauthorized attempts, schema errors and format compliance. A subsequent
   promising candidate still needs existing executable HumanEval+/MBPP+ and reserved
   final math confirmation before adoption.

One initial budget diagnostic failed importing a transient shared Inductor cache
file containing null bytes. The file was intact on later inspection; the exact
cause was not proven. Its failed receipt and partial output were archived, and
the u50 retry uses a separate recorded compile-cache directory. No global cache
was deleted, and no inference objective was changed.

The seed-1 MATH decline comprises seven lost and four gained answers. Of the seven losses, four candidate trajectories truncate without an answer; three stop normally with a wrong answer (math:1196, math:2360 and math:3007). Therefore truncation cannot explain every loss. Of the long-agent false-completion transitions, four are new and one previous false claim becomes an unfinished INVALID_TOKEN loop; that removed claim is not a corrected transaction.

## Completed full-bank budget diagnostic

At 4,096 tokens, seed 1 and batch size 8, u50 scores GSM8K 46/64 and MATH
37/64; masked replay scores 50/64 and 31/64. MATH truncations are 11 and 15.
The candidate's first 2,048 tokens are identical to its original short run on
all 64 MATH cases: 29 answers are correct in that prefix and 31 in the full
trajectory, so two gains are attributable to the extra budget.

For u50, only 3/64 MATH trajectories have identical prefixes to the original
short run. Its long-run prefix already scores 36/64, versus 32/64 originally;
the remaining budget adds one answer. Therefore do not attribute u50's entire
five-answer increase to the cap, or treat the six-answer full-budget gap as an
established persistent regression. Cache capacity, GPU placement and recompilation
changed between the original u50 short run and this diagnostic; their individual
effects were not isolated. These are seeded samples, not a powered comparison.
GSM8K prefixes also vary between budgets, with u50 changing one correctness
result. The next empirical check should quantify repeated fixed-configuration
inference before choosing another learning rate based on these differences.

The seven primary seed-1 MATH losses still include three normally terminated
wrong answers; budget alone is not a complete explanation. The verified tool
transaction failures also remain. Both budget workers and the CPU review worker
have exited; no training or evaluation is currently running. The diagnostic
review artifact preserves all per-case prefix, stop and correctness evidence.

A single fixed-configuration u50 repeat is now running on GPU 0 using the same
4K budget, seed, batch shape and isolated compile cache as its completed 4K run.
The one-shot CPU review worker will compare token identity and correctness flips,
update `phase3-masking/diagnostics/review.json`, and exit. It cannot launch training
or promote a model. This check is needed because the cross-budget baseline
trajectories changed substantially. Its current state is in
`phase3-masking/diagnostics/status.json`; the original bounded phase is complete.

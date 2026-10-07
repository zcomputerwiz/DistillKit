# Agentic pilot 2: smaller dose, contrasting actions, stronger code replay

Assisted-by: Codex

Status 2026-10-07: both arms and their original evaluations completed. See the
[evaluation audit](EVALUATION_AUDIT.md) before interpreting the proxy results:
math overlap and context-heavy NLL prevent treating those aggregate scores as
clean capability measurements. Keep u50. User approved conversational replay
masking for the next recipe; completed v2 remains unchanged.

Date: 2026-10-06. User authorized continuation after the version-1 retention
failure. This is another bounded pilot from u50, not an extension of the rejected
checkpoint. It reuses the existing trainer, cache format, and evaluation tools.
Research rationale remains in [the original plan](AGENTIC_TRAINING_PLAN.md);
the measured reason for changing the recipe is in [pilot-1 results](AGENTIC_PILOT_RESULTS.md).

## Data changes

Version 2 contains 864 training conversations (612,134 input tokens, 81,665
assistant targets) and 48 development conversations in two different schema
domains. Its 96 next-response cases are frozen before training. Every reference
trajectory is executed against the local finite-state evaluator during data
generation, in addition to schema and turn-order checks.

- Pair the same conditional-update request, ID, and tool schemas with two
  observations: already in the desired state, or requiring a write. The result
  determines the action; surface wording cannot distinguish the pair.
- Add known-ID read-only requests, including the words "look up", so that phrase
  does not always predict a name search. Retain name discovery, genuine ambiguity,
  error recovery, no-call, empty-search, and successful-completion cases.
- Vary function names, the identifier argument, tool ordering, task wording,
  and whether an action is announced. Announcements still include actual calls.
- Keep user, system, and tool results visible but unscored in new data. Continue
  hard-label assistant CE with guards preventing placeholder teacher KL.

The six training domains remain separate from development domains `dispatch`
and `records`. New sample identities and seeds differ from version 1. Both
versions use shared templates across train/development: this is limited schema
transfer evidence, not a claim of independent task-family generalization.
The old version-1 fixtures are now regression/development checks, since their
failures informed this revision. They are not added to training.

## Training choices

Both new arms begin from u50. The candidate adds the new data; the control uses
the strengthened replay alone. Use 20 optimizer steps instead of 50, and a
flat LR multiplier of 0.275 instead of 0.55 (body peak about 5.03e-7). Preserve
the ten-step warmup and existing optimizer/objective choices. The short pilot
does not reach the decay phase of the existing 10M-target schedule.

Aim for 5% new weighted targets per epoch instead of 15%. Triple the repeat
weights of raw-code, short-code, expanded-code, and teacher-code caches. This
is a code replay increase, not a claim that the realized token share triples.
Retain math, QA, general instruction, agent/tool, and anti-loop replay.
Measure and save actual first-20-step shares and visits before launching.

The first random-order plan was rejected before training: its candidate prefix
omitted general instruction, teacher-code, nonthinking math, and the loop-check
cache. The replacement uses opt-in `--balanced-prefix-batches 40`: select an
existing group covering each source, fill the remainder randomly, then shuffle
the prefix. The remainder of the epoch remains a permutation with no duplicated
or changed groups. This guarantees coverage, not equal token weights. Sampling
mode is included in the resume fingerprint; tests cover exact continuation
across an epoch boundary and unchanged default random ordering.

The rejected sampling plan remains under `agentic-pilot-v2/` for audit; it never
trained. The actual follow-up uses `agentic-pilot-v2-balanced/`.

Final measured 20-step prefix:

| Arm | Weighted targets | New targets | Code share | Sources covered |
|---|---:|---:|---:|---:|
| Agentic | 491,132 | 19,825 (4.04%) | 50.35% | 17/17 |
| Replay-only | 643,149 | 0 | 36.61% | 16/16 |

The candidate visits 203 new conversations, at most twice each. Coverage alone
does not guarantee strong preservation: for example, the general-pilot cache
contributes only 381 targets to the candidate prefix (broader curriculum data
contributes another 22,039). Per-domain retention checks remain necessary. The
different realized code shares are another comparison limitation to report.

This intentionally changes data diversity, dose, LR, and replay together to
seek a usable candidate. It is not a causal ablation of one variable. Equal
optimizer steps still need not give equal target counts or replay documents.
Report the realized plan and compare both candidates against u50. If this
small dose produces no reliable benefit, do not automatically lengthen it.

Use the previously validated streaming head and CSA2 checkpoint-selection
cache, head chunk 512, two-GPU tensor parallelism, and layer checkpointing.
No model source, architecture, or llama.cpp changes are part of this experiment.

## Evaluation

Run the version-2 finite environments and frozen next-response screen, plus
the complete version-1 finite-environment regression suite. Repeat the existing
original tool screen and post-tool continuation screen. Allow valid extra reads
and evaluate final state, not just exact next-call agreement.

Run the role-split ledger with extended QA, paired document analysis, and the
same code/thinking NLL and 256-example math generation proxies. Report answer
formatting failures and truncations alongside correctness. Gates remain those
in the original plan: an apparent improvement on synthetic templates alone is
insufficient, and material code/math regressions block selection.

The existing sequential runner stops on a failed child process and does not
promote checkpoints automatically. No other GPU work was active at preparation.

## Status

Data generation and 21 tests passed. The final sampling plan was measured and
the sequential training/evaluation pipeline launched. Its live state is
`scratch/dense_gr/agentic-pilot-v2-balanced/status.json`; run logs and exact
`plan.json` are in that directory. Generated data is in
`scratch/dense_gr/agentic-v2-data/`. No follow-up checkpoint has been selected.

# Agentic pilot 1: completed, not selected

Assisted-by: Codex

Date: 2026-10-06. Both 50-step arms and the complete evaluation pipeline finished.
The starting checkpoint remains u50. No model implementation changes were made.

| Screen | u50 | Agentic | Replay-only |
|---|---:|---:|---:|
| Finite-environment task success | 32/40 | 37/40 | 30/40 |
| Synthetic next-response screen | 62/80 | 77/80 | 59/80 |
| Original mixed tool screen | 52/72 | 53/72 | 53/72 |
| Post-tool continuation | 6/12 | 8/12 | 8/12 |
| GSM8K, 256 examples | 71.875% | 62.500% | 68.359% |
| MATH, 256 examples | 42.578% | 37.891% | 36.328% |
| Code NLL, lower better | 0.778027 | 0.854430 | 0.778709 |
| Thinking NLL, lower better | 0.552335 | 0.578310 | 0.552612 |

The synthetic improvement did not exceed replay alone on the broader original
or post-tool screens. Within the original screen, error recovery improved from
4/8 to 7/8, but exact held-out call matching fell from 8/24 to 6/24. These are
small, correlated screens; raw score differences are not significance claims.

The pilot's three finite-environment failures all searched an already supplied
ID as though it were a name. After the empty search, the model asked the user
for an ID it already had. This is a policy regression, not just a formatting
or next-call grading issue. The curriculum needs stronger contrasts between
name discovery, reading a known ID, and a conditional write after inspection.

Code and math protection was insufficient. GSM8K unboxed answers increased
from 3 to 38 (replay: 37); MATH truncations rose from 105 to 120 (replay: 132).
Generation scores therefore combine reasoning, formatting, and stopping effects.
Teacher-forced QA loss improved, but that does not outweigh served regressions.
The MATH cap was 1024 tokens for all three arms in this run, so historical
512-token results are not its baseline.

The agentic arm consumed 837,410 weighted targets and replay consumed 1,228,275.
Different grouping changed the sampled replay prefix. The run cannot isolate
the causal contribution of each recipe choice. It does establish that this
candidate should not replace u50 under the declared retention gates.

Local evidence is in `scratch/dense_gr/agentic-pilot-v1/`: `live-*.json`,
`eval-*/results.json`, `ledger/nll.json`, `ledger/paired.json`, `proxy.json`, and
the two `train.json` files. Earlier raw files and checkpoints are preserved.

Next: [the smaller version-2 follow-up](AGENTIC_V2_PLAN.md), freshly initialized
from u50. Do not extend the version-1 agentic weights.

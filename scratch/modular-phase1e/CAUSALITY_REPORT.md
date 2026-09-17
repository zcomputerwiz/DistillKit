# Phase 1e causality repair audit

Training remained paused. The original negative category-admission verdict is preserved.

## Causality repair

The old length-normalized feature changed by as much as 0.400826. With the fixed 512-token scale, the largest complete feature-vector change was 0.000e+00. All feature, deployment-state, and final-output checks passed: True.
The failed constant and contextual candidate weights were loaded unchanged. The contextual candidate was evaluated with the repaired feature semantics without retraining or selection.

## Phase 1d teacher-forced answer position

| Timing | Examples | Accuracy | NLL | Conditional bit accuracy | Bit mass | Errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| immediately_after_marker | 183 | 0.0000 | 2.477234 | 1.0000 | 0.1315 | 183 |
| after_whitespace | 585 | 0.9966 | 0.330534 | 1.0000 | 0.9889 | 2 |

## Bounded ordinary greedy completion

Across 768 lookup examples, Phase 1d produced the correct bit immediately or after one generated whitespace token in 99.7396%. Bit emission was 99.7396%, and content accuracy conditional on emitting a bit was 100.0000%.

Conclusion: Phase 1d already answers lookup prompts reliably under the accepted optional-whitespace format; the failed admission arms are not needed for this behavior.

This audit does not establish a positive admission result and performs no model selection, retraining, XOR evaluation, or efficiency claim.

Reproduce:

`python -m experiments.modular_phase1e.causality_eval --config experiments/modular_phase1e/configs/diagnostic.json --run-dir scratch/modular-phase1e --force`

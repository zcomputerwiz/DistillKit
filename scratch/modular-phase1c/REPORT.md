# Phase 1c direct lookup-readout

**Acceptance: PASS.** This establishes output-facing lookup integration only. The original Phase 1 verdict remains **failed composition**.

## Conditions on the frozen `(6,256)`, seed-11 plain backbone

| Condition | Conditional accuracy | Conditional NLL | Full-vocab accuracy | Answer NLL | Mean bit-category probability |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.5000 | 0.951107 | 0.3421 | 1.649511 | 0.706924 |
| deterministic_copy | 1.0000 | 0.000000 | 0.6941 | 0.698404 | 0.706924 |
| learned_readout | 1.0000 | 0.317249 | 0.6809 | 1.015653 | 0.706924 |
| value_independent_control | 0.5000 | 0.693147 | 0.3372 | 1.391552 | 0.706924 |

The output module preserves the backbone's total bit mass and replaces only the conditional split between `0` and `1`; full-vocabulary accuracy therefore remains limited by the frozen backbone's category detection.

## Learned readout and causal controls

The learned readout has 6 trainable parameters and stopped at 10 updates after 2560 supervised answers. The frozen backbone has no trainable parameters in this run. Training conditional accuracy/NLL were 1.0000/0.317249; the value-independent control was 0.5000/0.693147.

Held-out opposite-value pointer cases: 302; retrieved literal changed in 302. Intervened conditional accuracy: 1.0000; predictions changed from the old to new required bit in 1.0000. Mean normalized P(new bit) change: +0.456299; mean signed log-odds shift: +1.970509.

| Variant | Learned conditional accuracy | Control conditional accuracy |
| --- | ---: | ---: |
| renamed | 1.0000 | 0.4865 |
| whitespace | 1.0000 | 0.4865 |

## Exactness, causality, and budget

Maximum bit-category probability change: 2.384e-07; maximum non-bit probability change: 0.000e+00. Inactive output is exactly unchanged: True.

Unseen-suffix prefix comparisons: 3896; all addresses, activation, and retrieved values were invariant. XOR cases inactive: 64/64.

Measured overhead: 81.758 us/document for prefix state and 1.365 us/document for synchronized readout/composition. Total accelerator time: 4.312s / 300s.

## Acceptance gate

- PASS: heldout_conditional_accuracy
- PASS: opposite_pointer_conditional_accuracy
- PASS: opposite_pointer_predictions_change_toward_new
- PASS: renaming_conditional_accuracy
- PASS: whitespace_conditional_accuracy
- PASS: bit_category_preserved
- PASS: nonbit_probabilities_preserved
- PASS: inactive_output_exact
- PASS: value_independent_control_at_chance
- PASS: prefix_causality
- PASS: update_budget
- PASS: supervision_budget
- PASS: accelerator_budget

All pre-existing artifact hashes are unchanged. Passing does not establish XOR, backbone-layer use, efficiency, or learned-specialist success. Any later learned-specialist pilot still requires the original >=99% binding and structural-event accuracy gates.

Reproduce:

`python -m experiments.modular_phase1c --config experiments/modular_phase1c/configs/diagnostic.json --run-dir scratch/modular-phase1c --force`

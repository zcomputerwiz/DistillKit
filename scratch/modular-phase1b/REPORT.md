# Phase 1b binding-reader diagnostic

**Result: ineffective integration.** Even the exact observed-value positive control remains weak and causally insensitive, so merely making the value payload explicit does not repair use.

The original Phase 1 verdict remains **failed composition**. This diagnostic uses fresh backbones and oracle declaration addresses only; it is not an efficiency claim or a learned-specialist pass.

## Retrieved payload inspection

Across 128 fixed-format bit-flip pairs, the assigned value was inside the current seven-token window in 128 and preceded the query use in 128. Exactly one candidate embedding changed—and it was the value token—in 128 cases.

The value appeared at offsets {"2": 7, "3": 41, "4": 80} relative to the declaration identifier. Addresses were rebuilt from the independent reference interpreter; its values and answers were not passed to either model.

Mean value-candidate embedding delta: 10.863790; mean weighted-context delta at fresh initialization: 1.791398; mean reader-output delta: 0.000000. The last is zero because the paired reader's pointer scale is zero-initialized, not because the retrieved payload lacks the bit.

Each candidate is token_embedding(input_id) + absolute_position_embedding. These are non-contextual raw embeddings. With identifier/format/scope fixed, only the value-token candidate can contain the changed assigned bit. The entire declaration window precedes the query use, so gathering it is causal.

## Fresh paired training

| Reader | Targets | GPU seconds | Train acc. | Train answer NLL | Validation acc. | Validation NLL | Held-out acc. | Held-out NLL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| current_window | 1048576 | 5.870 | 0.0096 | 2.6606 | 0.0109 | 2.6668 | 0.0296 | 2.6987 |
| exact_value | 1048576 | 5.063 | 0.0096 | 2.6603 | 0.0109 | 2.6681 | 0.0296 | 2.6991 |

Both arms use identical data order, backbone/reader initialization, optimizer, ordinary causal CE over every valid next-token target, parameter count, and integration path. The exact-value arm changes only the declaration-window selection weights from learned attention to the observed bit's one-hot position.

After training, changing only the assigned bit moved the current reader output by L2=0.013603 and the exact-value reader by L2=0.108753. The corresponding mean changes in probability of the new required answer were -0.000017 and -0.000110; the representation changes, but the decoder does not use it effectively.

## Single-pointer causal interventions

| Reader | Eligible | Reader delta L2 | P(new required) delta | P(old required) delta | Correct-pointer acc. | Intervened-required acc. | Value attention |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| current_window | 302 | 0.116778 | +0.000063 | +0.000110 | 0.0265 | 0.0232 | 0.0707 |
| exact_value | 302 | 0.180267 | -0.000112 | -0.000059 | 0.0265 | 0.0232 | 1.0000 |

Integration-path diagnostics on held-out queries:

| Reader | Gate mean | Pointer-scale L2 | Pointer-signal L2 | Address-feature signal L2 |
| --- | ---: | ---: | ---: | ---: |
| current_window | 0.5300 | 0.2383 | 0.1125 | 0.3700 |
| exact_value | 0.5296 | 0.1903 | 0.1337 | 0.3700 |

Whole diagnostic accelerator time: 29.816s of the 600s cap. Each arm consumed exactly 1,048,576 targets.

## Conclusion

Classification: **ineffective integration**. Even the exact observed-value positive control remains weak and causally insensitive, so merely making the value payload explicit does not repair use.

The original Phase 1 artifacts and checkpoint hashes are unchanged. Any later learned-specialist experiment must independently meet the original >=99% structural-event accuracy gate before backbone training.

Reproduce:

`python -m experiments.modular_phase1b --config experiments/modular_phase1b/configs/diagnostic.json --run-dir scratch/modular-phase1b --force`

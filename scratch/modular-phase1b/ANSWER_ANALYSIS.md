# Answer-position and binary-selection audit

**Conclusion: the underlying lookup task remains unlearned: the best three-seed mean conditional bit accuracy is 0.510 and the best individual checkpoint is 0.536, despite bit-category detection reaching 0.674.** This is evaluation-only; all checkpoint hashes are unchanged.

## Scored position verification

For every document, logits at `answer_position - 1` score the token stored at `answer_position`, exactly matching the shifted causal-CE target. Examples:

| Whitespace | Arrow pos. | Logits pos. | Previous token | Answer pos. | Target |
| --- | ---: | ---: | --- | ---: | --- |
| tabs | 119 | 119 | => | 120 | 1 |
| spaces | 85 | 86 | <sp1> | 87 | 1 |
| mixed | 139 | 140 | <nl> | 141 | 1 |
| newlines | 75 | 75 | => | 76 | 0 |
| tabs | 67 | 67 | => | 68 | 1 |
| mixed | 97 | 98 | <sp1> | 99 | 1 |
| spaces | 119 | 120 | <sp2> | 121 | 0 |
| mixed | 71 | 72 | <nl> | 73 | 1 |

## Loss decomposition on 304 held-out lookup documents

`answer NLL = bit-category NLL + correct-bit-given-category NLL`.

| Model | Full answer acc. | Top-1 is bit | Conditional bit acc. | Answer NLL | Category NLL | Selection NLL | P(bit category) | P(correct bit | bit) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Phase 1 matched_l2_w64 | 0.0362 | 0.0800 | 0.4923 | 2.8045 | 2.0828 | 0.7217 | 0.1582 | 0.5024 |
| Phase 1 plain_l2_w64 | 0.0395 | 0.0713 | 0.5099 | 2.7873 | 2.0681 | 0.7192 | 0.1566 | 0.5014 |
| Phase 1 matched_l4_w128 | 0.1009 | 0.1897 | 0.5055 | 2.2876 | 1.5671 | 0.7205 | 0.2910 | 0.5017 |
| Phase 1 plain_l4_w128 | 0.0987 | 0.2039 | 0.5055 | 2.2329 | 1.5059 | 0.7270 | 0.3026 | 0.5003 |
| Phase 1 matched_l6_w256 | 0.3333 | 0.6743 | 0.4945 | 1.3920 | 0.6294 | 0.7625 | 0.7082 | 0.4996 |
| Phase 1 plain_l6_w256 | 0.3224 | 0.6535 | 0.4945 | 1.4947 | 0.6975 | 0.7972 | 0.6970 | 0.5000 |
| Phase 1b current_window | 0.0296 | 0.0526 | 0.4737 | 2.6987 | 1.9667 | 0.7320 | 0.1601 | 0.4964 |
| Phase 1b exact_value | 0.0296 | 0.0526 | 0.4704 | 2.6991 | 1.9664 | 0.7327 | 0.1602 | 0.4962 |

Phase 1 entries are means across seeds 11/22/33. The decomposition residual is at most 8.496e-10 nat.

## Normalized `{0,1}` pointer interventions

| Phase 1b reader | Cases | Reader delta L2 | ΔP(new required | bit) | Δlog-odds toward new | Fraction shifted toward new | Prediction changed | Correct-pointer acc. | Intervened-required acc. | No-change complement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| current_window | 302 | 0.116778 | -0.000169 | -0.000828 | 0.0894 | 0.0033 | 0.4735 | 0.5232 | 0.5265 |
| exact_value | 302 | 0.180267 | -0.000055 | -0.000414 | 0.0993 | 0.0099 | 0.4702 | 0.5265 | 0.5298 |

A formatting-only failure would show strong conditional bit accuracy despite low bit-category probability. That pattern is absent. Pointer interventions also fail to move normalized binary probabilities consistently toward the newly required bit. Intervened accuracy matches the complement expected when predictions do not change, so the lookup computation itself remains unlearned.

Reproduce:

`python -m experiments.modular_phase1b.answer_analysis --phase1-run scratch/modular-phase1 --phase1b-run scratch/modular-phase1b`

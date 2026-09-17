# Phase 1d learned-specialist handoff

**Qualification: PASS.** The original Phase 1 failed-composition verdict remains unchanged.

Binding accuracy: 1.000000; structural-event accuracy excluding pad/whitespace: 0.996693; macro-F1: 0.995726.

| Event | Support | Old FP/FN | New FP/FN | Old F1 | New F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| use | 2324 | 891/28 | 15/4 | 0.833243 | 0.995922 |
| invalid | 7989 | 1210/390 | 232/236 | 0.904751 | 0.970702 |
| output | 1863 | 269/17 | 4/0 | 0.928105 | 0.998928 |
| declaration | 14433 | 1/915 | 0/18 | 0.967230 | 0.999376 |
| close | 3696 | 139/72 | 109/9 | 0.971712 | 0.984250 |

Weakest held-out structural slices: corrupted=0.972120; depth_4=0.991366; whitespace_mixed=0.995694.

Learned operations used by deployment: query/use/semicolon/answer-marker event classification. Programmed specialist operations: causal scope table and binding pointer. Deterministic prefix processing carries activation across raw whitespace and scans the addressed declaration span for its observed literal. No reference interpreter value or expected answer enters the deployed path.

## End-to-end frozen path

Held-out conditional accuracy/NLL over all eligible queries: 1.000000/0.317249; full-vocabulary accuracy: 0.680921.

Activation/retrieval coverage: 1.000000/1.000000; missed activations: 0; failed retrievals after activation: 0.

Opposite-value interventions: 302; intervened accuracy: 1.000000; changed toward the required value: 1.000000.

Specialist parameters/tokens/steps: 40988/2097152/1661; training accelerator time: 21.168s. Warmed synchronized timing per document: backbone 119.687 us, specialist recurrent path 9.647 us, readout/composition 8.614 us.

Specialist SHA-256: `2c64666f579a89def2a66b900fff4113ffac8f8c0ef12203dd7d83c39fad2db3`. Frozen backbone/readout SHA-256: `0c81b60f20fdb790dea46d76fc1146f36611d993b9ebdb76d8eeca3c246f0e7f` / `eb101ea56901fd6d6f2e5d525dc6d5ab2092ea73e8053487879aca02daaadb98`; all pre-existing source hashes are unchanged.

Passing this phase establishes only learned-specialist delivery through the already-qualified output-facing lookup path. It does not establish XOR, backbone-layer use, efficiency, or change the Phase 1 verdict.

Reproduce the full bounded run:

`python -m experiments.modular_phase1d --config experiments/modular_phase1d/configs/diagnostic.json --run-dir scratch/modular-phase1d --force`

Repeat evaluation without retraining:

`python -m experiments.modular_phase1d --config experiments/modular_phase1d/configs/diagnostic.json --run-dir scratch/modular-phase1d --force --reuse-specialist`

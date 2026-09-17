# Phase 1d learned-specialist handoff

This bounded follow-up makes one specialist-only capacity correction, enforces binding
and structural-event accuracy gates of 0.99 on the independent specialist-validation
split, and stops before downstream evaluation if either gate fails.

When qualified, the run freezes that checkpoint and reuses the exact Phase 1c
six-parameter readout and the Phase 1 `(6,256)`, seed-11 plain backbone. The deployed
path uses learned query/use/semicolon/answer-marker events, the specialist's programmed
causal scope table and binding pointers, and deterministic prefix-only lexical processing
to carry activation over whitespace and read the addressed declaration literal. It never
receives a reference-interpreter value or expected answer.

Run once, including specialist training:

```powershell
python -m experiments.modular_phase1d `
  --config experiments/modular_phase1d/configs/diagnostic.json `
  --run-dir scratch/modular-phase1d --force
```

Repeat evaluation using the already-qualified frozen specialist:

```powershell
python -m experiments.modular_phase1d `
  --config experiments/modular_phase1d/configs/diagnostic.json `
  --run-dir scratch/modular-phase1d --force --reuse-specialist
```

This phase does not test XOR, backbone-layer integration, or efficiency, and it does not
replace the original Phase 1 failed-composition verdict.

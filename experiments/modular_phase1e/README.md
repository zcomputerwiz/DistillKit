# Phase 1e category admission

This bounded experiment freezes the Phase 1 `(6,256)`, seed-11 backbone, the qualified
Phase 1d specialist and deployment interface, and the Phase 1c six-parameter readout.
At active lookup prefixes it learns a scalar correction shared by both bit logits. This
changes bit-category mass while preserving the conditional distributions within the bit
and non-bit categories.

The two predeclared trained conditions are a one-parameter constant and a 27-parameter
affine contextual correction. Context features are frozen in the JSON configuration and
exclude the retrieved bit identity. Training uses complete-document causal-target
weighting; the implementation verifies exact gradient equivalence for the cached active
category contribution.

Calibration selects the constant if it qualifies, otherwise the contextual arm. Fresh
confirmation is materialized and evaluated once only after a calibration selection. If
neither qualifies, the run stops without evaluating confirmation.

```powershell
python -m experiments.modular_phase1e `
  --config experiments/modular_phase1e/configs/diagnostic.json `
  --run-dir scratch/modular-phase1e --force
```

Passing would establish category admission only on this backbone. Cross-scale reuse,
matched-quality efficiency, XOR, and backbone-layer integration remain out of scope.

## Causality repair audit (no training)

The follow-up audit replaces the contextual arm's length-normalized position with the
absolute causal token index divided by the frozen 512-token model scale. It reloads the
existing candidate weights without optimization, checks features and final outputs under
truncation, changed unseen suffixes, and batch padding, stratifies the unchanged Phase 1d
answer errors by formatting position, and runs at most two ordinary greedy completion
steps (an optional whitespace token followed by the bit).

```powershell
python -m experiments.modular_phase1e.causality_eval `
  --config experiments/modular_phase1e/configs/diagnostic.json `
  --run-dir scratch/modular-phase1e --force
```

This audit cannot reverse the negative Phase 1e admission verdict and performs no
training or confirmation-based model selection.

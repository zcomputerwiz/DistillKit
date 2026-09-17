# Phase 1b binding-reader diagnostic

This is an isolated positive-control diagnostic over the frozen Phase 1 language and
data. It writes only to `scratch/modular-phase1b` and verifies that the original Phase 1
reports and checkpoints retain their hashes.

Both arms use a fresh `(2,64)` decoder with seed 11, lookup-only non-literal questions,
exact programmed declaration addresses, ordinary causal CE on all targets, and at most
1,048,576 targets. The oracle supplies addresses only. The current arm uses the Phase 1
seven-token learned-attention reader. The exact-value arm has the identical state dict,
gate, scale, feature projection, and backbone, but selects the observed bit token at the
addressed declaration with a deterministic one-hot mask. Values and answers are never
provided by the interpreter.

```powershell
.\.venv\Scripts\python.exe -m experiments.modular_phase1b `
  --config experiments/modular_phase1b/configs/diagnostic.json `
  --run-dir scratch/modular-phase1b --force
```

The entire run, including evaluation and interventions, is capped at 600 accelerator
seconds. It is not an efficiency experiment and cannot qualify the learned specialist.

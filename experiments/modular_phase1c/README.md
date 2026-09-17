# Phase 1c direct lookup-readout

This bounded diagnostic freezes the Phase 1 plain `(6,256)`, seed-11 checkpoint and
fits only a six-parameter linear classifier over a fixed two-feature representation of
the literal retrieved through a causal oracle binding address. A matched value-independent
control receives zeros. No hash table or learned-specialist output is used.

The post-hoc composition preserves the frozen backbone's total probability on `{0,1}`
and every non-bit probability. Activation is derived from prefix state after a completed
non-XOR lookup and `=>`, including following whitespace; answer annotations are used only
to choose supervised/scored targets.

```powershell
.\.venv\Scripts\python.exe -m experiments.modular_phase1c `
  --config experiments/modular_phase1c/configs/diagnostic.json `
  --run-dir scratch/modular-phase1c --force
```

Passing establishes output-facing lookup integration only. It does not alter the earlier
failed-composition verdict or establish XOR, backbone-layer use, efficiency, or learned-
specialist success.

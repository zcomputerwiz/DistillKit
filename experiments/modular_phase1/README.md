# Bounded modular-LLM Phase 1

This directory is a self-contained experiment. It does not import or modify the fork's
GR, PLE, n-gram sidecar, trainer, or model code. Generated corpora and checkpoints live
under the chosen run directory (the examples below use `scratch/modular-phase1`).

## Frozen design

- Language: nested lexical scopes, `let` declarations, shadowing, bit values, and lookup
  or XOR queries. Whitespace is tokenized rather than discarded. Exactly 20% of training
  documents are literal lookup/XOR examples with no scope resolution. Answers always use
  one token, so label values cannot change target length.
- Oracle: `ReferenceInterpreter` is a recursive-descent interpreter independent of both
  generation and the specialist's streaming state machine.
- Specialist: a redacted-input GRU learns event and prefix-validity labels plus an
  auxiliary (reported, not exported) scope-depth head. A programmed causal stack/table
  emits exact scope depth and declaration-identifier pointers. It stores no
  values. The exported interface is probabilities, confidence, pointer presence/distance,
  and the pointer itself; recurrent hidden states, resolved values, and answers are not
  exported. The checkpoint is frozen only after binding accuracy >=99%, structural-event
  macro-F1 >=95%, and validity accuracy >=95% on its independent validation split.
- Specialist labels use deterministic inverse-frequency structural CE within each batch;
  this matches the macro event gate and prevents whitespace from erasing rare brace/query
  gradients. The backbone objective remains ordinary unweighted causal CE on every valid
  next-token target.
- Composition: the reader sees exported structure and a short raw-token window beginning
  at each declaration pointer. It is zero-output initialized, so the specialist arm and
  plain arm have identical backbone weights and logits at initialization. The matched
  arm widens only the FFNs enough to match specialist-plus-reader total parameters.
- Optimizer: AdamW, LR 3e-4, betas (0.9, 0.95), epsilon 1e-8, weight decay 0.1, constant
  schedule, gradient clipping at 1.0. There is no sweep.
- Data: specialist train/validation, backbone train/validation, and confirmation splits
  are exact-token deduplicated. The depth-4/XOR/newline/two-shadow conjunction is excluded
  from both training corpora. Training uses depths 1-4; confirmation adds depths 5-8,
  histories of 18-32 declarations, the held-out conjunction, identifier renaming, and
  whitespace rerenderings.
- Decisions: non-inferiority margins are one accuracy point and 0.02 nat for answer and
  overall NLL. Confidence intervals use a paired hierarchical bootstrap over documents
  and seeds. The efficiency target is >=10% amortized accelerator time versus the
  parameter-matched arm; first-use and online inference costs are reported separately.

## Reproduction

From the DistillKit repository root:

```powershell
# CPU correctness/smoke (one size, three arms; not a pilot result)
.\.venv\Scripts\python.exe -m experiments.modular_phase1 \
  --config experiments/modular_phase1/configs/smoke.json \
  --run-dir scratch/modular-phase1-smoke smoke

# Full, resumable one-GPU pilot. Run steps separately for inspectable gates.
.\.venv\Scripts\python.exe -m experiments.modular_phase1 \
  --config experiments/modular_phase1/configs/pilot.json \
  --run-dir scratch/modular-phase1 prepare
.\.venv\Scripts\python.exe -m experiments.modular_phase1 \
  --config experiments/modular_phase1/configs/pilot.json \
  --run-dir scratch/modular-phase1 specialist
.\.venv\Scripts\python.exe -m experiments.modular_phase1 \
  --config experiments/modular_phase1/configs/pilot.json \
  --run-dir scratch/modular-phase1 backbones
.\.venv\Scripts\python.exe -m experiments.modular_phase1 \
  --config experiments/modular_phase1/configs/pilot.json \
  --run-dir scratch/modular-phase1 evaluate
.\.venv\Scripts\python.exe -m experiments.modular_phase1 \
  --config experiments/modular_phase1/configs/pilot.json \
  --run-dir scratch/modular-phase1 report
```

Every command is resumable and verifies the frozen configuration and specialist hashes.
`REPORT.md` and `report.json` contain slice metrics, paired intervals, interventions,
costs, the categorical verdict, and all checkpoint hashes. An interrupted run is labeled
`incomplete`; it is never silently promoted to a pilot result.

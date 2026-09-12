#!/bin/bash
# Matched A/B at 1e-5, the rate the sweep found where arm A actually improves held-out
# content (-0.0286 against the untrained model) rather than degrading it as 3e-5 did.
# Both arms re-run with identical settings including --eval-every, so the pair is matched
# in every respect except whether -log P(w | WS) contributes at whitespace targets.
cd "D:/DeepThought/Projects/HybridModel/DistillKit" || exit 1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/Scripts/python.exe
OUT=scratch/ple_forensics/arms-1e5
mkdir -p "$OUT"

for ARM in A B; do
  echo "=== arm $ARM at 1e-5 ==="
  $PY scratch/ple_forensics/offload_arms.py --arm "$ARM" --lr 1e-5 \
      --eval-every 32 --trajectory-docs 48 \
      --trajectory "$OUT/$ARM.json" --output "$OUT/$ARM.npz" 2>&1 | tail -3
done

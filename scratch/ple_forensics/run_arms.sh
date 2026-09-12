#!/bin/bash
# Arms A and B of the responsibility-transfer pilot, plus the shared untrained reference.
# Identical corpus, token budget, window, rate, schedule and seed; the arms differ only in
# whether whitespace *selection* contributes to the backbone loss.
cd "D:/DeepThought/Projects/HybridModel/DistillKit" || exit 1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/Scripts/python.exe
OUT=scratch/ple_forensics/arms
mkdir -p "$OUT"

echo "=== baseline (untrained) ==="
$PY scratch/ple_forensics/offload_arms.py --arm A --baseline --output "$OUT/baseline.npz"
for ARM in A B; do
  echo "=== arm $ARM ==="
  $PY scratch/ple_forensics/offload_arms.py --arm "$ARM" --output "$OUT/$ARM.npz"
done

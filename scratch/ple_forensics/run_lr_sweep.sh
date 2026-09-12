#!/bin/bash
# Arm A only, across learning rates. The A/B result was measured in a regime where arm A
# *degrades* held-out content, which makes it a weak platform: what got tested there is
# whether removing whitespace selection slows a degradation, not whether it accelerates
# an improvement. This looks for a rate where A actually learns content.
#
# Logging train content NLL beside held-out separates the two explanations: train
# improving while held-out worsens is overfitting or domain adaptation; train worsening
# too is optimisation instability.
cd "D:/DeepThought/Projects/HybridModel/DistillKit" || exit 1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/Scripts/python.exe
OUT=scratch/ple_forensics/lr-sweep
mkdir -p "$OUT"

for LR in 3e-6 1e-5 2e-5 3e-5; do
  TAG=$(echo "$LR" | tr -d '.-' )
  if [ -f "$OUT/A-$TAG.json" ]; then echo "skip $LR"; continue; fi
  echo "=== arm A at lr=$LR ==="
  $PY scratch/ple_forensics/offload_arms.py --arm A --lr "$LR" \
      --eval-every 32 --trajectory-docs 48 \
      --trajectory "$OUT/A-$TAG.json" --output "$OUT/A-$TAG.npz" 2>&1 | tail -3
done

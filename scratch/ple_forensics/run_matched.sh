#!/bin/bash
# The optional arm, testing a different claim: explicitly reallocating the removed update
# magnitude to content.
#
# Under AdamW the update is per-coordinate scale-invariant, so removing a loss term does
# not obviously shrink the step; every run now reports its actual parameter displacement
# so the assumption is measured rather than asserted. The meaningful version of "give B
# the magnitude A had" is therefore not a rescaled loss but a higher rate: if B has less
# to fit, it may tolerate a rate at which A overfits, and reach content A cannot.
#
# A's numbers at all four rates already exist in ../lr-sweep. This completes the grid for
# B, so B's best content can be compared with A's best (0.45918 at 1e-5).
cd "D:/DeepThought/Projects/HybridModel/DistillKit" || exit 1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/Scripts/python.exe
OUT=scratch/ple_forensics/matched
mkdir -p "$OUT"

for LR in 3e-6 2e-5 3e-5; do
  TAG=$(echo "$LR" | tr -d '.-')
  if [ -f "$OUT/B-$TAG.json" ]; then echo "skip $LR"; continue; fi
  echo "=== arm B at lr=$LR ==="
  $PY scratch/ple_forensics/offload_arms.py --arm B --lr "$LR" \
      --eval-every 64 --trajectory-docs 48 \
      --trajectory "$OUT/B-$TAG.json" --output "$OUT/B-$TAG.npz" 2>&1 | tail -2
done

# And re-measure the matched 1e-5 pair with displacement instrumentation.
for ARM in A B; do
  if [ -f "$OUT/$ARM-1e5.json" ]; then echo "skip $ARM 1e-5"; continue; fi
  echo "=== arm $ARM at lr=1e-5 (displacement) ==="
  $PY scratch/ple_forensics/offload_arms.py --arm "$ARM" --lr 1e-5 \
      --trajectory "$OUT/$ARM-1e5.json" --output "$OUT/$ARM-1e5.npz" 2>&1 | tail -2
done

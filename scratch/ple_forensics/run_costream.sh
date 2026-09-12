#!/bin/bash
# The two-stream co-adaptation pilot: A (stock), S (PLE into the residual), M (PLE into a
# private lane). 1e-5 first because the arm-A LR sweep put the CE optimum there and it is
# the primary comparison; the neighbouring rates follow so a null cannot be blamed on the
# rate, the way the first A/B pilot's 3e-5 platform could be.
cd "D:/DeepThought/Projects/HybridModel/DistillKit" || exit 1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/Scripts/python.exe
ROOT=scratch/ple_forensics/costream
mkdir -p "$ROOT"

# The shared untrained reference. Every arm starts bitwise stock, so one evaluation
# serves all of them at every rate.
if [ ! -f "$ROOT/baseline.npz" ]; then
  echo "=== untrained baseline ==="
  $PY scratch/ple_forensics/costream_arms.py --arm A --baseline \
      --output "$ROOT/baseline.npz" 2>&1 | tail -2
fi

for LR in 1e-5 3e-6 2e-5; do
  TAG=$(echo "$LR" | tr -d '.-')
  OUT="$ROOT/$TAG"
  mkdir -p "$OUT"
  cp -n "$ROOT/baseline.npz" "$OUT/baseline.npz"
  for ARM in A S M; do
    if [ -f "$OUT/$ARM.json" ]; then echo "skip $ARM at $LR"; continue; fi
    echo "=== arm $ARM at lr=$LR ==="
    $PY scratch/ple_forensics/costream_arms.py --arm "$ARM" --lr "$LR" \
        --eval-every 64 --trajectory-docs 48 \
        --trajectory "$OUT/$ARM.json" --output "$OUT/$ARM.npz" 2>&1 | tail -3
  done
  echo "=== scoring $LR ==="
  $PY scratch/ple_forensics/score_costream.py "$OUT" 2>&1 | tail -60
done

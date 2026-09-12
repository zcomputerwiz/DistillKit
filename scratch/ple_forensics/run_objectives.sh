#!/bin/bash
# Objective forensics: A/S/M under CE, sparse KL alone, hidden-state cosine alone, and the
# historical 0.7 KL + 0.3 cosine. The architectures are the pilot's, unchanged; the only
# variable is what the loss asks for.
#
# The projections are trained once, backbone frozen, at the historical stage-1 rate, and
# every arm that needs them loads the same file -- the stand-in for stage 2 inheriting
# stage 1's projections. Without it the cosine term starts from a random 2560->5120 map
# (anchor cosine 1.01 and 1.00, similarity ~0) and measures noise.
cd "D:/DeepThought/Projects/HybridModel/DistillKit" || exit 1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/Scripts/python.exe
ROOT=scratch/ple_forensics/objective
mkdir -p "$ROOT"

if [ ! -f "$ROOT/projections.pt" ]; then
  echo "=== shared anchor projections ==="
  $PY scratch/ple_forensics/objective_arms.py --arm A --objective cosine \
      --train-projections --lr 1e-4 --output "$ROOT/projections.pt" 2>&1 | tail -2
fi

for OBJ in ce kl cosine combined; do
  OUT="$ROOT/$OBJ"
  mkdir -p "$OUT"
  cp -n scratch/ple_forensics/costream/baseline.npz "$OUT/baseline.npz"
  for ARM in A S M; do
    if [ -f "$OUT/$ARM.json" ]; then echo "skip $ARM under $OBJ"; continue; fi
    echo "=== arm $ARM under $OBJ ==="
    $PY scratch/ple_forensics/objective_arms.py --arm "$ARM" --objective "$OBJ" \
        --projections "$ROOT/projections.pt" --eval-every 64 --trajectory-docs 48 \
        --trajectory "$OUT/$ARM.json" --output "$OUT/$ARM.npz" 2>&1 | tail -3
  done
  echo "=== scoring $OBJ ==="
  $PY scratch/ple_forensics/score_costream.py "$OUT" 2>&1 | tail -45
done

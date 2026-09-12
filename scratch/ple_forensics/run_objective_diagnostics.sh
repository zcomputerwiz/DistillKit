#!/bin/bash
# Why each term flips the sign, rather than just that it does.
#
# The matrix says the same sidecar helps content under CE and hurts it under both
# distillation terms. Two mechanisms are proposed and both are checkable:
#
#   tail    the KL target is a renormalised top-64 with a zero tail, so a true token the
#           teacher never ranked has target probability exactly zero. If that is what
#           punishes the sidecar, the harm lives on positions outside the teacher's list.
#
#   write   the cosine term rewards matching the teacher's hidden states, and the sidecar
#           sits upstream of anchor 4 where it is the only trainable thing. If that is
#           what punishes the sidecar, matching error should track write magnitude.
#
# Same seeds and settings as the matrix runs, so these describe those exact models.
cd "D:/DeepThought/Projects/HybridModel/DistillKit" || exit 1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/Scripts/python.exe
ROOT=scratch/ple_forensics/objective
OUT="$ROOT/diagnostics"
mkdir -p "$OUT"

for OBJ in ce kl; do
  for ARM in A S; do
    if [ -f "$OUT/tail-$OBJ-$ARM.npz" ]; then echo "skip tail $OBJ $ARM"; continue; fi
    echo "=== tail diagnostic: arm $ARM under $OBJ ==="
    $PY scratch/ple_forensics/objective_arms.py --arm "$ARM" --objective "$OBJ" \
        --projections "$ROOT/projections.pt" \
        --tail-diagnostic "$OUT/tail-$OBJ-$ARM.npz" \
        --output "$OUT/$OBJ-$ARM.npz" 2>&1 | tail -8
  done
done

for OBJ in cosine combined; do
  for ARM in S M; do
    if [ -f "$OUT/write-$OBJ-$ARM.npz" ]; then echo "skip write $OBJ $ARM"; continue; fi
    echo "=== write diagnostic: arm $ARM under $OBJ ==="
    $PY scratch/ple_forensics/objective_arms.py --arm "$ARM" --objective "$OBJ" \
        --projections "$ROOT/projections.pt" \
        --write-diagnostic "$OUT/write-$OBJ-$ARM.npz" \
        --output "$OUT/$OBJ-$ARM.npz" 2>&1 | tail -14
  done
done

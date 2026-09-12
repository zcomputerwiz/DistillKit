#!/bin/bash
# The validation experiment for the proposed target repair: does grouping the omitted mass
# instead of zeroing it restore the sidecar's CE behaviour under teacher supervision?
#
#   grouped      sparse KL with the tail carried rather than asserted to be zero. This is
#                MissingProbabilityHandling.SYMMETRIC_UNIFORM, which is the grouped-tail
#                objective exactly -- the uniform factor cancels between the two sides.
#   grouped_ce   0.5 * ground-truth CE + 0.5 * grouped, so the student can still be
#                rewarded for a true token the capture never ranked.
#
# Success is S - A <= 0 while the teacher term still buys something over CE alone.
cd "D:/DeepThought/Projects/HybridModel/DistillKit" || exit 1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/Scripts/python.exe
ROOT=scratch/ple_forensics/objective

for OBJ in grouped grouped_ce; do
  OUT="$ROOT/$OBJ"
  mkdir -p "$OUT"
  cp -n scratch/ple_forensics/costream/baseline.npz "$OUT/baseline.npz"
  for ARM in A S M; do
    if [ -f "$OUT/$ARM.json" ]; then echo "skip $ARM under $OBJ"; continue; fi
    echo "=== arm $ARM under $OBJ ==="
    $PY scratch/ple_forensics/objective_arms.py --arm "$ARM" --objective "$OBJ" \
        --eval-every 64 --trajectory-docs 48 \
        --trajectory "$OUT/$ARM.json" --output "$OUT/$ARM.npz" 2>&1 | tail -3
  done
  echo "=== scoring $OBJ ==="
  $PY scratch/ple_forensics/score_costream.py "$OUT" 2>&1 | tail -40
done

# And the tail split for the winning target, against the zero-tail numbers already taken.
for ARM in A S; do
  if [ -f "$ROOT/diagnostics/tail-grouped_ce-$ARM.npz" ]; then continue; fi
  echo "=== tail diagnostic: arm $ARM under grouped_ce ==="
  $PY scratch/ple_forensics/objective_arms.py --arm "$ARM" --objective grouped_ce \
      --tail-diagnostic "$ROOT/diagnostics/tail-grouped_ce-$ARM.npz" \
      --output "$ROOT/diagnostics/grouped_ce-$ARM.npz" 2>&1 | tail -7
done

#!/usr/bin/env bash
# Baseline, eager-with-the-new-code, and compiled: three arms of the same probe.
#
# The probe is the real 4B student under tensor parallelism with non-reentrant
# checkpointing, AdamW8bit, the folded head and the hidden-state cosine. Batch 2 is
# what stage 2 actually runs; batch 4 is there because it is 30% cheaper per token and
# stopped fitting, so whether it fits again is the question worth asking.
#
# NEEDS THE GPU TO ITSELF.
#
#   bash scratch/bench_fused.sh 2      # or 4
set -eu

BATCH=${1:-2}
BASE=/d/DeepThought/Projects/HybridModel/DistillKit
WORK=/d/DeepThought/Projects/HybridModel/DistillKit-mem
PY="$BASE/.venv/Scripts/python.exe"
OUT="$WORK/scratch/widened-residual"
export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8
mkdir -p "$OUT"

run() {  # run <label> <directory> <extra env assignment>
  local label=$1 directory=$2
  shift 2
  echo "=== $label (batch $BATCH) ==="
  if ( cd "$directory" && env "$@" "$PY" scratch/widened_residual_probe.py performance \
        --branches 2 --batch "$BATCH" --output "$OUT/bench-$label-b$BATCH.json" ) \
        > "$OUT/bench-$label-b$BATCH.log" 2>&1; then
    "$PY" - "$OUT/bench-$label-b$BATCH.json" "$label" <<'PY'
import json, sys
steps = json.loads(open(sys.argv[1]).read())["steps"]
settled = steps[1:] or steps          # step 1 excludes optimizer state
print("  %-10s %6.2f s/update   peak %5.2f/%5.2f GiB   reserved %5.2f/%5.2f" % (
    sys.argv[2],
    sum(s["seconds"] for s in settled) / len(settled),
    max(s["peak_gib"][0] for s in settled), max(s["peak_gib"][1] for s in settled),
    max(s["reserved_gib"][0] for s in settled), max(s["reserved_gib"][1] for s in settled)))
PY
  else
    echo "  $label: FAILED (see bench-$label-b$BATCH.log)"
    grep -o "torch.OutOfMemoryError.*" "$OUT/bench-$label-b$BATCH.log" | head -1 || true
  fi
}

run baseline "$BASE" DISTILLKIT_COMPILE=1
run eager    "$WORK" DISTILLKIT_COMPILE=0
run compiled "$WORK" DISTILLKIT_COMPILE=1

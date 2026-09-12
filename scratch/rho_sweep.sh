#!/bin/bash
# Frozen rho sweep over the four donor-reader arms. No training: every checkpoint is the
# initialised arm, and --rho changes only the in-memory residual scalar.
cd "D:/DeepThought/Projects/HybridModel/DistillKit-donor-transplant" || exit 1
PY=../DistillKit/.venv/Scripts/python.exe          # the worktree venv is CPU-only torch
BUNDLE=../DistillKit/scratch/independent-eval/reply-bundle-384.json
TABLE="C:/Users/Owner/.cache/huggingface/hub/models--unsloth--Qwen3.8-Flash-Next-GGUF/snapshots/38bb39ee97821de2c9009abb7e93950eec396e66/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00002-of-00003.gguf"
OUT=scratch/transplant-eval
mkdir -p "$OUT"

declare -A ARM=(
  [c1_c1]=transplant-c1-value-c1-conv-init
  [donor_c1]=transplant-donor-value-c1-conv-init
  [c1_donor]=transplant-c1-value-donor-conv-init
  [donor_donor]=transplant-donor-value-donor-conv-init
)

for RHO in "$@"; do
  TAG=$(echo "$RHO" | tr -d '.' | tr '-' 'm')
  for NAME in c1_c1 donor_c1 c1_donor donor_donor; do
    FILE="$OUT/$NAME-rho$TAG.json"
    if [ -f "$FILE" ] && grep -q '"complete": true' "$FILE"; then echo "skip $NAME $RHO"; continue; fi
    echo "=== $NAME rho=$RHO ==="
    $PY -m distillkit.independent_eval evaluate --bundle "$BUNDLE" \
        --split screen --tasks nll --max-seconds 570 \
        --checkpoint "D:/DeepThought/Projects/HybridModel/runs/${ARM[$NAME]}" \
        --table "$TABLE" --rho "$RHO" --output "$FILE" 2>&1 | tail -1
  done
done

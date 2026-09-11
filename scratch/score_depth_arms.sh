#!/bin/bash
# Assistant-only independent eval for today's arms, sequentially: each holds the
# 28.8 GB table and both cards, so they cannot overlap.
cd "D:/DeepThought/Projects/HybridModel/DistillKit" || exit 1
PY=.venv/Scripts/python.exe
TABLE="C:/Users/Owner/.cache/huggingface/hub/models--unsloth--Qwen3.8-Flash-Next-GGUF/snapshots/38bb39ee97821de2c9009abb7e93950eec396e66/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00002-of-00003.gguf"
BUNDLE=scratch/independent-eval/reply-bundle-384.json
for NAME in widened-plegated-fp32-stage1-1m widened-plegated-L28-stage1-1m widened-plegated-L16-stage1-1m; do
  OUT="scratch/independent-eval/reply-$NAME.json"
  [ -f "$OUT" ] && { echo "skip $NAME (already scored)"; continue; }
  [ -d "../runs/$NAME" ] || { echo "missing run $NAME"; continue; }
  echo "=== $NAME ==="
  $PY -m distillkit.independent_eval evaluate --bundle "$BUNDLE" \
      --checkpoint "../runs/$NAME" --table "$TABLE" --output "$OUT" || echo "FAILED $NAME"
done
echo "=== paired assistant-only comparison ==="
$PY scratch/score_plegated.py

#!/bin/bash
# Assistant-only independent eval for the depth-control arms, sequentially: each
# holds the 28.8 GB table and both cards, so they cannot overlap.
#
# --tasks nll is not optional. The flag defaults to all three tasks and the reply
# bundle carries only nll, so the default refuses the bundle with "requested
# evaluation tasks are missing or empty" -- after write_json has already left a
# {"complete": false} marker at the output path, which the scorer then trips over.
# --split screen matches every reply-*.json already scored.
cd "D:/DeepThought/Projects/HybridModel/DistillKit" || exit 1
PY=.venv/Scripts/python.exe
TABLE="C:/Users/Owner/.cache/huggingface/hub/models--unsloth--Qwen3.8-Flash-Next-GGUF/snapshots/38bb39ee97821de2c9009abb7e93950eec396e66/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00002-of-00003.gguf"
BUNDLE=scratch/independent-eval/reply-bundle-384.json

for NAME in widened-plegated-fp32-stage1-1m widened-plegated-L28-stage1-1m widened-plegated-L16-stage1-1m; do
  OUT="scratch/independent-eval/reply-$NAME.json"
  if [ -f "$OUT" ] && grep -q '"records"' "$OUT"; then echo "skip $NAME (already scored)"; continue; fi
  [ -d "../runs/$NAME" ] || { echo "missing run $NAME"; continue; }
  echo "=== $NAME ==="
  if ! $PY -m distillkit.independent_eval evaluate --bundle "$BUNDLE" \
        --split screen --tasks nll --max-seconds 570 \
        --checkpoint "../runs/$NAME" --table "$TABLE" --output "$OUT"; then
    # Leave no marker behind for the scorer to read as a result.
    echo "FAILED $NAME"; rm -f "$OUT"
  fi
done

echo "=== paired assistant-only comparison ==="
$PY scratch/score_plegated.py

#!/bin/bash
# Phase A: is the pretrained consumer around layer 24 the bottleneck?
#
# A1 opens layers 20-27 alongside the sidecar. A2 is identical with every n-gram row
# read for the wrong context, which is the arm that decides whether any gain is table
# content or just extra optimisation freedom. A3 trains the same window with no sidecar,
# which is how much is simply continued tuning against a new objective.
#
# The objective is ground-truth assistant CE, not the sparse top-k KL: that ranked arms
# opposite to assistant NLL at every injection depth tested, and the cosine anchors pull
# toward a teacher that has no table.
#
# Once the window adapts, `bypassed` no longer reproduces the pre-retrofit student, so
# absolute NLL against student-hf is the number that matters and enabled-minus-bypassed
# is only a diagnostic.
cd "D:/DeepThought/Projects/HybridModel/DistillKit" || exit 1
PY=.venv/Scripts/python.exe
TABLE="C:/Users/Owner/.cache/huggingface/hub/models--unsloth--Qwen3.8-Flash-Next-GGUF/snapshots/38bb39ee97821de2c9009abb7e93950eec396e66/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00002-of-00003.gguf"
BUNDLE=scratch/independent-eval/reply-bundle-384.json

for ARM in A1-window A2-shuffled A3-nosidecar; do
  NAME="win-$ARM-stage1-1m"
  OUT="scratch/independent-eval/reply-$NAME.json"
  if [ -f "$OUT" ] && grep -q '"records"' "$OUT"; then echo "skip $ARM (scored)"; continue; fi

  if [ ! -d "../runs/$NAME" ]; then
    echo "=== training $ARM ==="
    $PY -m distillkit.main "scratch/win-$ARM.yml" -v > "../runs/$NAME.log" 2>&1 \
      || { echo "FAILED training $ARM"; tail -25 "../runs/$NAME.log"; continue; }
  fi
  until grep -q "INFO:__main__:Done." "../runs/$NAME.log" 2>/dev/null; do sleep 10; done

  echo "=== evaluating $ARM ==="
  $PY -m distillkit.independent_eval evaluate --bundle "$BUNDLE" \
      --split screen --tasks nll --max-seconds 570 \
      --checkpoint "../runs/$NAME" --table "$TABLE" --output "$OUT" \
    || { echo "FAILED eval $ARM"; rm -f "$OUT"; continue; }
  echo "--- $ARM eval_loss: $(grep -o "'eval_loss': [^,}]*" "../runs/$NAME.log" | tail -1)"
done

echo "=== paired assistant-only comparison ==="
$PY scratch/score_plegated.py

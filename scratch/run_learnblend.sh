#!/bin/bash
# Learned blend against a shuffled-donor control, on the full 384-document screen.
#
# The treatment starts every sublayer's blend at 0.10 -- the best point in the
# pre-training sweep -- and lets it move at its own 2e-3, which is the rate that
# can actually reach 0.25 or fall back to zero inside 72 steps.
#
# The control is the identical run against a donor whose tensors keep their exact
# shapes and value multisets and have had their elements permuted. If it moves
# assistant NLL the same way, the gain is rescaling and no borrowing is happening.
cd "D:/DeepThought/Projects/HybridModel/DistillKit" || exit 1
PY=.venv/Scripts/python.exe
TABLE="C:/Users/Owner/.cache/huggingface/hub/models--unsloth--Qwen3.8-Flash-Next-GGUF/snapshots/38bb39ee97821de2c9009abb7e93950eec396e66/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00002-of-00003.gguf"
BUNDLE=scratch/independent-eval/reply-bundle-384.json

for ARM in learnblend learnblend-shuffled; do
  NAME="hc-L24-$ARM-stage1-1m"
  OUT="scratch/independent-eval/reply-$NAME.json"
  if [ -f "$OUT" ] && grep -q '"records"' "$OUT"; then echo "skip $ARM (scored)"; continue; fi

  if [ ! -d "../runs/$NAME" ]; then
    echo "=== training $ARM ==="
    $PY -m distillkit.main "scratch/hc-L24-$ARM.yml" -v > "../runs/$NAME.log" 2>&1 \
      || { echo "FAILED training $ARM"; tail -20 "../runs/$NAME.log"; continue; }
  fi
  # Wait for the export, not the progress bar: 72/72 prints before the weights land.
  until grep -q "INFO:__main__:Done." "../runs/$NAME.log" 2>/dev/null; do sleep 10; done

  echo "=== evaluating $ARM ==="
  $PY -m distillkit.independent_eval evaluate --bundle "$BUNDLE" \
      --split screen --tasks nll --max-seconds 570 \
      --checkpoint "../runs/$NAME" --table "$TABLE" --output "$OUT" \
    || { echo "FAILED eval $ARM"; rm -f "$OUT"; continue; }

  echo "--- $ARM blend, first and last ---"
  grep -o "residual_stream/blend': '[-0-9.e]*'" "../runs/$NAME.log" | head -2
  grep -o "residual_stream/blend': '[-0-9.e]*'" "../runs/$NAME.log" | tail -2
done

echo "=== paired assistant-only comparison ==="
$PY scratch/score_plegated.py

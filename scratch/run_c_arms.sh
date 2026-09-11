#!/bin/bash
# Does the sidecar add anything once the window is tuned properly?
#
# The precondition is now met and then some: eight trainable layers at 3e-5 with no
# sidecar reach -0.074448 nats against the pre-retrofit student, where the same window
# at 1e-4 cost +0.053166. So the control these arms answer to is win-A3-lr3e05, not the
# frozen student -- the sidecar has to earn something on top of a properly tuned window.
#
# Both start from the L24 checkpoint, so the sidecar begins holding its -0.008860
# behaviour rather than racing eight layers from a zero initialisation, which is what
# made A1 and A2 indistinguishable.
#
# B2 runs first: its rows are real table rows read for the wrong context, so if it
# matches B1 the pathway is carrying capacity rather than content, and if it lands
# *worse* than the no-sidecar control that is the first evidence the pathway is
# content-specific.
cd "D:/DeepThought/Projects/HybridModel/DistillKit" || exit 1
PY=.venv/Scripts/python.exe
TABLE="C:/Users/Owner/.cache/huggingface/hub/models--unsloth--Qwen3.8-Flash-Next-GGUF/snapshots/38bb39ee97821de2c9009abb7e93950eec396e66/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00002-of-00003.gguf"
BUNDLE=scratch/independent-eval/reply-bundle-384.json

for TAG in C2-L1-shuffled C1-L1-real; do
  NAME="win-$TAG-stage1-1m"
  OUT="scratch/independent-eval/reply-$NAME.json"
  if [ -f "$OUT" ] && grep -q '"records"' "$OUT"; then echo "skip $TAG"; continue; fi
  if [ ! -d "../runs/$NAME" ]; then
    echo "=== training $TAG ==="
    $PY -m distillkit.main "scratch/win-$TAG.yml" -v > "../runs/$NAME.log" 2>&1 \
      || { echo "FAILED $TAG"; tail -20 "../runs/$NAME.log"; continue; }
  fi
  until grep -q "INFO:__main__:Done." "../runs/$NAME.log" 2>/dev/null; do sleep 10; done
  echo "--- $TAG eval_loss: $(grep -o "'eval_loss': [^,}]*" "../runs/$NAME.log" | tr '\n' ' ')"
  echo "--- $TAG gate: $(grep -o "sidecar.ple/gate_std': '[0-9.]*'" "../runs/$NAME.log" | tail -1)"
  $PY -m distillkit.independent_eval evaluate --bundle "$BUNDLE" \
      --split screen --tasks nll --max-seconds 570 \
      --checkpoint "../runs/$NAME" --table "$TABLE" --output "$OUT" \
    || { echo "FAILED eval $TAG"; rm -f "$OUT"; continue; }
done

echo "=== paired assistant-only comparison ==="
$PY scratch/score_plegated.py

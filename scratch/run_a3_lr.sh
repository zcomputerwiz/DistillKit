#!/bin/bash
# Make the trainable window neutral before asking it to help.
#
# A3 -- eight trainable layers, no sidecar at all -- cost +0.053166 nats at the run's
# 1e-4, and its eval_loss rose monotonically through training (0.4635 -> 0.5905). The
# corpus is one epoch, so this is not repetition; 1e-4 held constant on pretrained
# layers is 5-10x a normal fine-tuning rate. The target here is A3 ~ 0.000 against the
# pre-retrofit student, not an improvement: a window that does no harm is the
# precondition for asking whether it can consume the sidecar.
cd "D:/DeepThought/Projects/HybridModel/DistillKit" || exit 1
PY=.venv/Scripts/python.exe
TABLE="C:/Users/Owner/.cache/huggingface/hub/models--unsloth--Qwen3.8-Flash-Next-GGUF/snapshots/38bb39ee97821de2c9009abb7e93950eec396e66/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00002-of-00003.gguf"
BUNDLE=scratch/independent-eval/reply-bundle-384.json

for TAG in lr3e05 lr1e05 lr3e06; do
  NAME="win-A3-$TAG-stage1-1m"
  OUT="scratch/independent-eval/reply-$NAME.json"
  if [ -f "$OUT" ] && grep -q '"records"' "$OUT"; then echo "skip $TAG"; continue; fi
  if [ ! -d "../runs/$NAME" ]; then
    echo "=== training $TAG ==="
    $PY -m distillkit.main "scratch/win-A3-$TAG.yml" -v > "../runs/$NAME.log" 2>&1 \
      || { echo "FAILED $TAG"; tail -20 "../runs/$NAME.log"; continue; }
  fi
  until grep -q "INFO:__main__:Done." "../runs/$NAME.log" 2>/dev/null; do sleep 10; done
  echo "--- $TAG eval_loss trajectory: $(grep -o "'eval_loss': [^,}]*" "../runs/$NAME.log" | tr '\n' ' ')"
  $PY -m distillkit.independent_eval evaluate --bundle "$BUNDLE" \
      --split screen --tasks nll --max-seconds 570 \
      --checkpoint "../runs/$NAME" --output "$OUT" \
    || { echo "FAILED eval $TAG"; rm -f "$OUT"; continue; }
done

echo "=== paired assistant-only comparison ==="
$PY scratch/score_plegated.py

#!/bin/bash
# c1_donor dominates at every common rho, but its reader also writes harder. Pushing the
# weak arms until their content cost matches c1_donor's separates "bigger write" from
# "better direction": if c1_c1 at matched content cost still buys 16x less layout, the
# interaction is real.
cd "D:/DeepThought/Projects/HybridModel/DistillKit-donor-transplant" || exit 1
PY=../DistillKit/.venv/Scripts/python.exe
BUNDLE=../DistillKit/scratch/independent-eval/reply-bundle-384.json
TABLE="C:/Users/Owner/.cache/huggingface/hub/models--unsloth--Qwen3.8-Flash-Next-GGUF/snapshots/38bb39ee97821de2c9009abb7e93950eec396e66/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00002-of-00003.gguf"
OUT=scratch/transplant-eval

run () {  # arm-name checkpoint rho tag
  FILE="$OUT/$1-rho$4.json"
  if [ -f "$FILE" ] && grep -q '"complete": true' "$FILE"; then echo "skip $1 $3"; return; fi
  echo "=== $1 rho=$3 ==="
  $PY -m distillkit.independent_eval evaluate --bundle "$BUNDLE" \
      --split screen --tasks nll --max-seconds 570 \
      --checkpoint "D:/DeepThought/Projects/HybridModel/runs/$2" \
      --table "$TABLE" --rho "$3" --output "$FILE" 2>&1 | tail -1
}

run c1_c1       transplant-c1-value-c1-conv-init       3.0  3x
run c1_c1       transplant-c1-value-c1-conv-init      10.0  10x
run donor_c1    transplant-donor-value-c1-conv-init    3.0  3x
run donor_c1    transplant-donor-value-c1-conv-init   10.0  10x
run donor_donor transplant-donor-value-donor-conv-init 3.0  3x

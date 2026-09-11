#!/bin/bash
# Where should the sidecar inject? Three points said the answer is "deeper" --
# the sidecar's own cost on assistant tokens went +0.0272 at layer 1, +0.0135 at
# 16, -0.0070 at 28 -- so this fills in the curve, weighted toward the region
# where it crosses zero and toward the deepest layers the variant allows.
#
# The PLE variant requires a linear_attention layer. This student has them at
# 0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22,24,25,26,28,29,30; every fourth
# index is full_attention and would be refused.
#
# Sequential, because each run holds the 28.8 GB table and both cards. About 15
# minutes to train and 4 to evaluate per layer, and 17 GB of checkpoint each.
#
#   nohup bash scratch/depth_sweep.sh > scratch/depth-sweep.log 2>&1 &
cd "D:/DeepThought/Projects/HybridModel/DistillKit" || exit 1
PY=.venv/Scripts/python.exe
TABLE="C:/Users/Owner/.cache/huggingface/hub/models--unsloth--Qwen3.8-Flash-Next-GGUF/snapshots/38bb39ee97821de2c9009abb7e93950eec396e66/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00002-of-00003.gguf"
BUNDLE=scratch/independent-eval/reply-bundle-384.json
BASE=examples/qwen35_widened_plegated_stage1_1m.yml
LAYERS="${*:-8 20 24 26 30}"

for LAYER in $LAYERS; do
  NAME="widened-plegated-L${LAYER}-stage1-1m"
  CONFIG="scratch/plegated-L${LAYER}-stage1-1m.yml"
  OUT="scratch/independent-eval/reply-$NAME.json"
  if [ -f "$OUT" ] && grep -q '"records"' "$OUT"; then echo "skip L$LAYER (scored)"; continue; fi

  if [ ! -d "../runs/$NAME" ]; then
    $PY - "$BASE" "$CONFIG" "$LAYER" <<'PYEOF'
import io, sys
source, target, layer = sys.argv[1], sys.argv[2], int(sys.argv[3])
text = io.open(source, encoding="utf-8").read()
old = "runs/widened-plegated-stage1-1m"
new = "runs/widened-plegated-L%d-stage1-1m" % layer
assert text.count(old) == 1, "output_path anchor moved"
text = text.replace(old, new)
anchor = "  layer_index: 1   # matches Flash-Next's ple_layer_ids (one-indexed [2])"
assert text.count(anchor) == 1, "layer_index anchor moved"
text = text.replace(anchor, "  layer_index: %d   # depth sweep; see scratch/depth_sweep.sh" % layer)
io.open(target, "w", encoding="utf-8", newline="\n").write(text)
print("wrote", target)
PYEOF
    [ -f "$CONFIG" ] || { echo "FAILED to write config for L$LAYER"; continue; }
    echo "=== training L$LAYER ==="
    $PY -m distillkit.main "$CONFIG" -v > "../runs/$NAME.log" 2>&1 \
      || { echo "FAILED training L$LAYER"; tail -5 "../runs/$NAME.log"; continue; }
  fi

  echo "=== evaluating L$LAYER ==="
  # --tasks nll: the flag defaults to all three and the reply bundle carries only
  # nll, and the refusal lands after a {"complete": false} marker is written.
  $PY -m distillkit.independent_eval evaluate --bundle "$BUNDLE" \
      --split screen --tasks nll --max-seconds 570 \
      --checkpoint "../runs/$NAME" --table "$TABLE" --output "$OUT" \
    || { echo "FAILED eval L$LAYER"; rm -f "$OUT"; continue; }
  echo "=== L$LAYER done: $(grep -o "'eval_loss': [^,}]*" "../runs/$NAME.log" | tail -1) ==="
done

echo "=== paired assistant-only comparison across every scored depth ==="
$PY scratch/score_plegated.py

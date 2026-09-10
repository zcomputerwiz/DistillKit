#!/usr/bin/env bash
# Re-score every arm with the screen partitioned by chat role.
#
# The whole-window screen is half prompt: of 175,910 scored tokens across the 384
# screening documents, 87,940 are assistant content, 42,632 system, 40,414 user and
# 4,924 template. The system prompt is near-identical boilerplate and 21 documents
# never reach the assistant turn at all, so every independent number reported so far
# is roughly half a measurement of how well a model predicts a fixed instruction block.
#
# This is a re-scoring, not a training run: the same forward pass, the positions
# partitioned. Prepare the bundle first (it is deliberately not committed):
#
#   python -m distillkit.independent_eval prepare --tokenizer ../student-hf \
#     --documents ../capture-data/heldout.jsonl \
#     --manifests ../teacher-cache-1m/manifest.json ../teacher-cache-5m/manifest.json \
#     --docs 384 --document-tokens 512 --text-only \
#     --output scratch/independent-eval/role-bundle-384.json
#
# NEEDS THE GPU TO ITSELF.  bash scratch/score_by_role.sh
set -eu

BUNDLE=scratch/independent-eval/role-bundle-384.json
OUT=scratch/independent-eval
PY=/d/DeepThought/Projects/HybridModel/DistillKit/.venv/Scripts/python.exe
TABLE=$(grep '^  table_path: ' examples/_lr_sweep_base.yml | sed 's/^  table_path: //')
RUNS=/d/DeepThought/Projects/HybridModel/runs

score() {  # score <name> <checkpoint> [extra args]
  local name=$1 checkpoint=$2
  shift 2
  if [ ! -e "$checkpoint" ]; then echo "skip $name (no checkpoint)"; return; fi
  if [ -f "$OUT/role-$name.json" ]; then echo "have $name"; return; fi
  echo "=== $name ==="
  "$PY" -m distillkit.independent_eval evaluate --bundle "$BUNDLE" \
      --checkpoint "$checkpoint" --tasks nll --output "$OUT/role-$name.json" "$@" 2>&1 | tail -1
}

score student-hf ../student-hf
score widened-stage1-1m "$RUNS/widened-stage1-1m"
score widened-ple-stage1-1m "$RUNS/widened-ple-stage1-1m" --table "$TABLE"
score ple-stage1-1m "$RUNS/ple-stage1-1m" --table "$TABLE"
score gr-stage1-1m "$RUNS/gr-stage1-1m" --table "$TABLE"
score lr-sweep-1e3 "$RUNS/lr-sweep-1e3" --table "$TABLE"
score widened-ple-stage2-5m "$RUNS/widened-ple-stage2-5m" --table "$TABLE"
score ple-control-stage2-5m "$RUNS/ple-control-stage2-5m" --table "$TABLE"

"$PY" -m distillkit.independent_eval report \
    --reference "$OUT/role-student-hf.json" \
    --results $(ls "$OUT"/role-*.json | grep -v role-report.json) \
    --output "$OUT/role-report.json" 2>&1 | grep -E "nll@|nll/nll \| enabled - pre_retrofit"

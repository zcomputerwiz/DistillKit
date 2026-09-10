#!/usr/bin/env bash
# Score the responses, not the prompt.
#
# The whole-window screen spent half its tokens on a near-identical instruction block:
# 87,940 assistant tokens against 83,046 of system and user, and 21 of 384 documents
# never reached the assistant turn at all. This bundle keeps the prompt in context --
# a continuation has to be conditioned on its question -- but scores only assistant
# positions, and requires every document to carry at least 64 of them:
#
#   384 documents, 156,565 assistant tokens, from 1,505 eligible unseen documents
#
# Prepare it first (bundles are deterministic and deliberately not committed):
#
#   python -m distillkit.independent_eval prepare --tokenizer ../student-hf \
#     --documents ../capture-data/heldout.jsonl \
#     --manifests ../teacher-cache-1m/manifest.json ../teacher-cache-5m/manifest.json \
#     --docs 384 --document-tokens 1024 --min-assistant-tokens 64 --text-only \
#     --output scratch/independent-eval/reply-bundle-384.json
#
# Read the nll@assistant rows. The nll@system and nll@user rows are retained as
# diagnostics -- they are what the screen used to be measuring by accident.
#
# NEEDS THE GPU TO ITSELF.  bash scratch/score_replies.sh
set -eu

BUNDLE=scratch/independent-eval/reply-bundle-384.json
OUT=scratch/independent-eval
PY=.venv/Scripts/python.exe
TABLE=$(grep '^  table_path: ' examples/_lr_sweep_base.yml | sed 's/^  table_path: //')
RUNS=../runs

score() {  # score <name> <checkpoint> [extra args]
  local name=$1 checkpoint=$2
  shift 2
  if [ ! -e "$checkpoint" ]; then echo "skip $name (no checkpoint)"; return; fi
  if [ -f "$OUT/reply-$name.json" ]; then echo "have $name"; return; fi
  echo "=== $name ==="
  "$PY" -m distillkit.independent_eval evaluate --bundle "$BUNDLE" \
      --checkpoint "$checkpoint" --tasks nll --output "$OUT/reply-$name.json" "$@" 2>&1 | tail -1
}

score student-hf ../student-hf
score widened-stage1-1m "$RUNS/widened-stage1-1m"
score widened-ple-stage1-1m "$RUNS/widened-ple-stage1-1m" --table "$TABLE"
score widened-ple-stage1-1m-anchor32 "$RUNS/widened-ple-stage1-1m-anchor32" --table "$TABLE"
score ple-stage1-1m "$RUNS/ple-stage1-1m" --table "$TABLE"
score gr-stage1-1m "$RUNS/gr-stage1-1m" --table "$TABLE"
score lr-sweep-1e3 "$RUNS/lr-sweep-1e3" --table "$TABLE"
score widened-ple-stage2-5m "$RUNS/widened-ple-stage2-5m" --table "$TABLE"
score ple-control-stage2-5m "$RUNS/ple-control-stage2-5m" --table "$TABLE"

"$PY" -m distillkit.independent_eval report \
    --reference "$OUT/reply-student-hf.json" \
    --results $(ls "$OUT"/reply-*.json | grep -vE 'reply-(bundle|report)') \
    --output "$OUT/reply-report.json"

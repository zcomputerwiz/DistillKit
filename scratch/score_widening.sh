#!/usr/bin/env bash
# Score the widening on independent text, against the arms it has to beat.
#
# The 32-document screen that produced the first verdict has wide absolute intervals
# (student-hf 1.1107 [0.9045, 1.3569]); this uses 384 of the 1,602 eligible unseen
# documents, so a difference between two arms of the curriculum can be resolved
# rather than only a difference against a stock model.
#
# Run from the DistillKit repository root:  bash scratch/score_widening.sh
set -eu

BUNDLE=scratch/independent-eval/text-bundle-384.json
OUT=scratch/independent-eval
PY=.venv/Scripts/python.exe
TABLE=$(grep '^  table_path: ' examples/_lr_sweep_base.yml | sed 's/^  table_path: //')

score() {  # score <name> <checkpoint> [--table]
  local name=$1 checkpoint=$2
  shift 2
  if [ -f "$OUT/w384-$name.json" ]; then echo "have $name"; return; fi
  echo "=== $name ==="
  "$PY" -m distillkit.independent_eval evaluate --bundle "$BUNDLE" \
      --checkpoint "$checkpoint" --tasks nll --output "$OUT/w384-$name.json" "$@" \
      2>&1 | tail -1
}

score student-hf ../student-hf
# No sidecar: the whole difference from student-hf is the routing it learned.
score widened-stage1-1m ../runs/widened-stage1-1m
score widened-ple-stage1-1m ../runs/widened-ple-stage1-1m --table "$TABLE"
# The unwidened arms this has to beat, rescored on the same wider bundle.
score ple-stage1-1m ../runs/ple-stage1-1m --table "$TABLE"
score gr-stage1-1m ../runs/gr-stage1-1m --table "$TABLE"
score lr-sweep-1e3 ../runs/lr-sweep-1e3 --table "$TABLE"

"$PY" -m distillkit.independent_eval report \
    --reference "$OUT/w384-student-hf.json" \
    --results $(ls "$OUT"/w384-*.json) \
    --output "$OUT/w384-report.json" 2>&1 | tail -40

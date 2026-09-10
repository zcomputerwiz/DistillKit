#!/usr/bin/env bash
# The two-anchor control with telemetry, then score every arm on responses only.
#
# The ablation run itself has one anchor, so it cannot say what the anchor it dropped
# was doing. The control re-run can, and it doubles as a reproducibility check: same
# config, same seed, so eval_loss should land back on 0.7262.
set -eu
PY=.venv/Scripts/python.exe
RUNS=../runs
export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8

if [ ! -d "$RUNS/widened-ple-stage1-1m-telemetry" ]; then
  echo "=== two-anchor control, with per-anchor telemetry ==="
  "$PY" -m distillkit.main examples/qwen35_widened_ple_stage1_1m_telemetry.yml -v \
      > "$RUNS/widened-ple-stage1-1m-telemetry.log" 2>&1
  echo "  exit $?"
fi
bash scratch/score_replies.sh

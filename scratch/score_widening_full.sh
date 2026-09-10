#!/usr/bin/env bash
# Score every arm of the widened curriculum on independent text AND real benchmarks.
#
# The earlier text-only screen exists because MMLU/ARC downloads were unavailable in
# the sandbox that built the evaluator. They are not unavailable on this machine:
# `cais/mmlu` (14,042 test rows) and `allenai/ai2_arc` ARC-Challenge (1,172) both
# download anonymously. full-bundle-384.json carries 384 documents, 256 MMLU and 256
# ARC questions per split, with a separate confirmation split drawn the same way.
#
# NEEDS THE GPU TO ITSELF. Each evaluation loads the 4B student on cuda:0; running it
# beside a training job will OOM one of them.
#
# Run from the DistillKit repository root:  bash scratch/score_widening_full.sh
set -eu

BUNDLE=scratch/independent-eval/full-bundle-384.json
OUT=scratch/independent-eval
PY=.venv/Scripts/python.exe
TABLE=$(grep '^  table_path: ' examples/_lr_sweep_base.yml | sed 's/^  table_path: //')

score() {  # score <name> <checkpoint-under-runs-or-path> [extra args]
  local name=$1 checkpoint=$2
  shift 2
  if [ ! -e "$checkpoint" ]; then echo "skip $name (no checkpoint yet)"; return; fi
  if [ -f "$OUT/full-$name.json" ]; then echo "have $name"; return; fi
  echo "=== $name ==="
  # Each task runs in its own process: the watchdog is per invocation, and three
  # tasks over two ablation modes on a widened checkpoint will not fit in one.
  for task in nll mmlu arc; do
    "$PY" -m distillkit.independent_eval evaluate --bundle "$BUNDLE" \
        --checkpoint "$checkpoint" --tasks "$task" \
        --output "$OUT/full-$name.$task.json" "$@" 2>&1 | tail -1
  done
  "$PY" - "$OUT/full-$name" <<'PY'
import json, sys
from pathlib import Path
stem = Path(sys.argv[1])
parts = [json.loads(Path(f"{stem}.{task}.json").read_text()) for task in ("nll", "mmlu", "arc")]
merged = dict(parts[0])
merged["records"] = {task: part["records"][task] for task, part in zip(("nll", "mmlu", "arc"), parts)}
merged["task_sha256"] = {k: v for part in parts for k, v in part["task_sha256"].items()}
merged["elapsed_seconds"] = sum(part["elapsed_seconds"] for part in parts)
merged["complete"] = all(part["complete"] for part in parts)
Path(f"{stem}.json").write_text(json.dumps(merged, indent=2), encoding="utf-8")
print(json.dumps({"merged": str(stem) + ".json", "complete": merged["complete"],
                  "seconds": round(merged["elapsed_seconds"], 1)}))
PY
}

score student-hf ../student-hf
score widened-stage1-1m ../runs/widened-stage1-1m
score widened-ple-stage1-1m ../runs/widened-ple-stage1-1m --table "$TABLE"
score ple-stage1-1m ../runs/ple-stage1-1m --table "$TABLE"
score gr-stage1-1m ../runs/gr-stage1-1m --table "$TABLE"
score lr-sweep-1e3 ../runs/lr-sweep-1e3 --table "$TABLE"
# The stage-2 pair, once the curriculum has produced them.
score widened-ple-stage2-5m ../runs/widened-ple-stage2-5m --table "$TABLE"
score ple-control-stage2-5m ../runs/ple-control-stage2-5m --table "$TABLE"

"$PY" -m distillkit.independent_eval report \
    --reference "$OUT/full-student-hf.json" \
    --results $(ls "$OUT"/full-*.json | grep -v '\.\(nll\|mmlu\|arc\)\.json$' | grep -v full-report.json) \
    --output "$OUT/full-report.json" 2>&1 | tail -80

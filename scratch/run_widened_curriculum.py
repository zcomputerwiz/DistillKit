"""Does a wider residual stream earn its place on text the cache never saw?

Every adapter this project has built hurts held-out cross-entropy: gr-stage1-1m
+0.5658, ple-stage1-1m +0.4515, lr-sweep-1e3 +0.7186 nats/token against student-hf,
all intervals excluding zero. The widening is the first change that is *exactly* the
identity at initialisation, so it is the first that can be judged on what it learned
rather than on damage done by the graft.

Four runs, in increasing cost and decreasing certainty:

1. `widened-stage1-1m`      widening alone, frozen backbone. Control: student-hf.
2. `widened-ple-stage1-1m`  widening + PLE sidecar, frozen. Control: ple-stage1-1m.
3. `widened-ple-stage2-5m`  both unlocked, 5M cache.
4. `ple-control-stage2-5m`  the same without the widening. Control for (3).

`eval_loss` is not the verdict and is only watched here for divergence -- it improved
monotonically with the sidecar learning rate while independent cross-entropy got
steadily worse. Score the checkpoints afterwards:

    python -m distillkit.independent_eval run --checkpoint <path> ...

    python scratch/run_widened_curriculum.py            # run whatever has no result
    python scratch/run_widened_curriculum.py --summary  # print the table
"""

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("D:/DeepThought/Projects/HybridModel")
PYTHON = str(Path(".venv/Scripts/python.exe").resolve())
RUNS = [
    ("widened-stage1-1m", "examples/qwen35_widened_stage1_1m.yml",
     "widening only, frozen", "student-hf"),
    ("widened-ple-stage1-1m", "examples/qwen35_widened_ple_stage1_1m.yml",
     "widening + PLE, frozen", "ple-stage1-1m"),
    ("widened-ple-stage2-5m", "examples/qwen35_widened_ple_stage2_5m.yml",
     "widening + PLE, unlocked, 5M", "ple-control-stage2-5m"),
    ("ple-control-stage2-5m", "examples/qwen35_ple_control_stage2_5m.yml",
     "PLE, unlocked, 5M, no widening", "--"),
]


def read(name):
    path = ROOT / "runs" / (name + ".log")
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    evals = [float(v) for v in re.findall(r"'eval_loss': '([0-9.]+)'", text)]
    runtime = re.findall(r"'train_runtime': '([0-9.]+)'", text)
    lambdas = [float(v) for v in re.findall(r"attn_residual/lambda_read': '([0-9.e+-]+)'", text)]
    return {
        "state": "done" if runtime else "running/failed",
        "runtime": float(runtime[-1]) if runtime else None,
        "eval_loss": evals[-1] if evals else None,
        "lambda_max": max((abs(v) for v in lambdas), default=None),
        "checkpoint": (ROOT / "runs" / name).is_dir(),
        "diverged": any(e != e or e > 10 for e in evals),
    }


def summarize():
    print("%-24s %-32s %10s %10s %12s" %
          ("run", "arm", "runtime_s", "eval_loss", "|lambda|max"))
    for name, _, label, _ in RUNS:
        info = read(name)
        if info is None:
            print("%-24s %-32s %10s" % (name, label, "not started"))
            continue
        print("%-24s %-32s %10s %10s %12s%s" % (
            name, label,
            "%.0f" % info["runtime"] if info["runtime"] else info["state"],
            "%.4f" % info["eval_loss"] if info["eval_loss"] is not None else "-",
            "%.3e" % info["lambda_max"] if info["lambda_max"] is not None else "-",
            "" if info["checkpoint"] else "   (no checkpoint)"))
        if info["diverged"]:
            print("                         -> diverged")
    print("\neval_loss is NOT the verdict: it improved monotonically across the sidecar")
    print("learning-rate sweep while independent cross-entropy got monotonically worse.")
    print("A |lambda|max still at 0 means the routing never woke up, which is a defect.")
    print("Score with: python -m distillkit.independent_eval run --checkpoint runs/<name>")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    if args.summary:
        summarize()
        return 0
    environment = dict(os.environ)
    # Windows cannot defragment; keep the collector aggressive near the cap.
    environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "garbage_collection_threshold:0.8")
    for name, config, label, _ in RUNS:
        info = read(name)
        if info and info["state"] == "done" and info["checkpoint"]:
            print("skipping %s (already complete)" % name)
            continue
        print("\n=== %s | %s ===" % (name, label), flush=True)
        started = time.time()
        with open(ROOT / "runs" / (name + ".log"), "w", encoding="utf-8") as handle:
            code = subprocess.call([PYTHON, "-m", "distillkit.main", config, "-v"],
                                   stdout=handle, stderr=subprocess.STDOUT, env=environment)
        print("  exit %d after %.1f min" % (code, (time.time() - started) / 60), flush=True)
        if code != 0:
            # Stage 2 depends on stage 1's checkpoint, so a failure is not skippable.
            print("  stopping: later runs read this run's checkpoint", flush=True)
            break
    summarize()
    return 0


if __name__ == "__main__":
    sys.exit(main())

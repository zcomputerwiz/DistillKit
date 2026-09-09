"""Run both sidecar designs through the same 1M -> 5M curriculum, and summarize.

Stage 1 initialises the sidecar on a frozen fresh student against the 1M cache; stage 2
unlocks the whole backbone against the 5M cache, continuing from stage 1. Both designs
run the identical curriculum, because stage 2 at 5M has never been run and the PLE arm
would otherwise have no baseline to be measured against.

Order matters: both stage 1s run first, so a problem surfaces in ~35 minutes rather than
after committing three hours to the stage 2s. Sequential throughout -- each run uses both
GPUs, and a co-tenant would perturb the comparison.

    python scratch/run_ple_curriculum.py             # run whatever has no result yet
    python scratch/run_ple_curriculum.py --summary   # print what has finished
"""

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("D:/DeepThought/Projects/HybridModel")
PYTHON = str(Path(".venv/Scripts/python.exe").resolve())

STAGES = [
    ("ple-stage1-1m", "examples/qwen35_ple_stage1_1m.yml", None),
    ("gr-stage1-1m", "examples/qwen35_gr_stage1_1m.yml", None),
    ("ple-stage2-5m", "examples/qwen35_ple_stage2_5m.yml", "ple-stage1-1m"),
    ("gr-stage2-5m", "examples/qwen35_gr_stage2_5m.yml", "gr-stage1-1m"),
]


def result(name):
    path = ROOT / "runs" / (name + ".log")
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    runtime = re.findall(r"'train_runtime': '([0-9.]+)'", text)
    evals = re.findall(r"'eval_loss': '([0-9.]+)'", text)
    if not runtime:
        return {"state": "running/failed", "eval_loss": float(evals[-1]) if evals else None}
    return {"state": "done", "train_runtime": float(runtime[-1]),
            "eval_loss": float(evals[-1]) if evals else None,
            "evals": [float(e) for e in evals]}


def summarize():
    print("%-16s %-14s %13s %11s" % ("run", "state", "train_runtime", "eval_loss"))
    table = {}
    for name, _, _ in STAGES:
        info = result(name) or {"state": "not started"}
        table[name] = info
        print("%-16s %-14s %13s %11s" % (
            name, info["state"],
            "%.0f s" % info["train_runtime"] if info.get("train_runtime") else "-",
            "%.4f" % info["eval_loss"] if info.get("eval_loss") is not None else "-"))

    for stage in ("stage1-1m", "stage2-5m"):
        a, b = table.get("ple-" + stage, {}), table.get("gr-" + stage, {})
        if a.get("eval_loss") is not None and b.get("eval_loss") is not None:
            print("\n%s: ple %.4f vs gated_residual %.4f  ->  %+.4f"
                  % (stage, a["eval_loss"], b["eval_loss"], a["eval_loss"] - b["eval_loss"]))
    print("\nOne seed per arm. The 5M pilot's lesson was that a single pair cannot")
    print("separate a small effect from run-to-run wobble -- read the eval trajectory,")
    print("not just the endpoint, and treat a sub-0.005 gap as unestablished.")


def run(name, config):
    command = [PYTHON, "-m", "distillkit.main", config, "-v"]
    print("\n=== %s ===\n%s" % (name, " ".join(command)), flush=True)
    started = time.time()
    with open(ROOT / "runs" / (name + ".log"), "w", encoding="utf-8") as handle:
        code = subprocess.call(command, stdout=handle, stderr=subprocess.STDOUT)
    print("  exit %d after %.1f min" % (code, (time.time() - started) / 60), flush=True)
    return code


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    if args.summary:
        summarize()
        return 0

    for name, config, needs in STAGES:
        existing = result(name)
        if existing and existing["state"] == "done":
            print("skipping %s (eval_loss %.4f)" % (name, existing["eval_loss"]))
            continue
        if needs is not None and not (ROOT / "runs" / needs / "config.json").is_file():
            print("STOPPING: %s needs %s, which has not produced a checkpoint." % (name, needs))
            return 1
        if run(name, config) != 0:
            print("STOPPING: %s failed; later stages would build on it." % name)
            return 1
    summarize()
    return 0


if __name__ == "__main__":
    sys.exit(main())

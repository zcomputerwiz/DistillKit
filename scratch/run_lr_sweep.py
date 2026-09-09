"""What learning rate should the sidecar adapter use in stage 2?

Stage 2 runs the backbone at 1e-5, and at that rate the sidecar does not move: the
chained run's `W_side_proj` went 2.1055 -> 2.1077 across an entire epoch, and the PLE
module's weights did not change to five significant figures over fifty logged steps. So
stage 2 has never actually trained a sidecar -- it trained a backbone around a frozen
one.

Five runs of stage 2 from the PLE stage-1 checkpoint, identical but for
`optimizer.sidecar_lr`, on the 1M cache so each is ~15 minutes rather than ~85. That is
72 optimizer steps, enough to see whether a rate moves the sidecar at all, and it lands
on the same footing as the historical chained stage-2 result of 0.5262.

    python scratch/run_lr_sweep.py            # run whatever has no result yet
    python scratch/run_lr_sweep.py --summary  # print the table

Read three things per run, not just the loss: whether `value_norm` moved at all, whether
`gate_std` survived (a gate that collapses to a constant is a scale, not a selector), and
whether the loss is better than the base rate. A rate that moves the sidecar and hurts
the loss is as informative as one that helps.
"""

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("D:/DeepThought/Projects/HybridModel")
PYTHON = str(Path(".venv/Scripts/python.exe").resolve())
RUNS = [
    ("lr-sweep-base", "examples/_lr_sweep_base.yml", "1e-5 (backbone rate)"),
    ("lr-sweep-5e5", "examples/_lr_sweep_5e5.yml", "5e-5"),
    ("lr-sweep-1e4", "examples/_lr_sweep_1e4.yml", "1e-4"),
    ("lr-sweep-5e4", "examples/_lr_sweep_5e4.yml", "5e-4"),
    ("lr-sweep-1e3", "examples/_lr_sweep_1e3.yml", "1e-3"),
]


def read(name):
    path = ROOT / "runs" / (name + ".log")
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    evals = [float(v) for v in re.findall(r"'eval_loss': '([0-9.]+)'", text)]
    runtime = re.findall(r"'train_runtime': '([0-9.]+)'", text)

    def series(key):
        return [float(v) for v in re.findall(r"ple/%s': '([0-9.e+-]+)'" % key, text)]

    value, gate_std, gate_mean = series("value_norm"), series("gate_std"), series("gate_mean")
    return {
        "state": "done" if runtime else "running/failed",
        "eval_loss": evals[-1] if evals else None,
        "value_start": value[0] if value else None,
        "value_end": value[-1] if value else None,
        "gate_std": gate_std[-1] if gate_std else None,
        "gate_mean": gate_mean[-1] if gate_mean else None,
        "diverged": any(e != e or e > 10 for e in evals),
    }


def summarize():
    print("%-15s %-22s %10s %14s %10s %9s" %
          ("run", "sidecar_lr", "eval_loss", "value_norm", "gate_std", "moved?"))
    base = None
    for name, _, label in RUNS:
        info = read(name)
        if info is None:
            print("%-15s %-22s %10s" % (name, label, "not started"))
            continue
        if info["value_start"] is not None and info["value_end"] is not None:
            moved = abs(info["value_end"] - info["value_start"])
            movement = "%.4f -> %.4f" % (info["value_start"], info["value_end"])
            flag = "yes" if moved > 1e-4 else "NO"
        else:
            movement, flag = "-", "-"
        loss = "%.4f" % info["eval_loss"] if info["eval_loss"] is not None else "-"
        if name == "lr-sweep-base":
            base = info["eval_loss"]
        delta = ""
        if base is not None and info["eval_loss"] is not None and name != "lr-sweep-base":
            delta = "  (%+.4f)" % (info["eval_loss"] - base)
        print("%-15s %-22s %10s %14s %10s %9s%s" % (
            name, label, loss, movement,
            "%.4f" % info["gate_std"] if info["gate_std"] is not None else "-",
            flag, delta))
        if info["diverged"]:
            print("                -> diverged")
    print("\nhistorical reference: the old design chained into stage 2 at 1M reached 0.5262")
    print("A rate that moves the sidecar but hurts the loss is as informative as one that")
    print("helps; the failure this is looking for is a sidecar that never moves at all.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    if args.summary:
        summarize()
        return 0
    for name, config, label in RUNS:
        info = read(name)
        if info and info["state"] == "done":
            print("skipping %s (eval_loss %.4f)" % (name, info["eval_loss"]))
            continue
        print("\n=== %s | sidecar_lr %s ===" % (name, label), flush=True)
        started = time.time()
        with open(ROOT / "runs" / (name + ".log"), "w", encoding="utf-8") as handle:
            code = subprocess.call([PYTHON, "-m", "distillkit.main", config, "-v"],
                                   stdout=handle, stderr=subprocess.STDOUT)
        print("  exit %d after %.1f min" % (code, (time.time() - started) / 60), flush=True)
        if code != 0:
            # A high rate may legitimately blow up; that is a result, not a reason to stop.
            print("  (continuing: a rate that fails is itself a finding)", flush=True)
    summarize()
    return 0


if __name__ == "__main__":
    sys.exit(main())

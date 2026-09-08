"""Run the 5M pilot's four stage-1 arms in sequence, and summarize them.

The arms must not run concurrently: each one uses both GPUs through tensor parallelism.
Sequential is also what makes them comparable -- a co-tenant changes clocks and memory
pressure, and this pilot is measuring a 0.0448 effect.

    python scratch/run_5m_pilot.py            # run every arm that has no result yet
    python scratch/run_5m_pilot.py --summary  # just print what has finished
    python scratch/run_5m_pilot.py --smoke    # 3 steps per arm against the 1M cache

--smoke is the gate to run first: it exercises stage 1 with tensor parallelism and
sortish batching together, a combination no completed run has used, and it fails in
minutes rather than an hour into the first real arm.
"""

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("D:/DeepThought/Projects/HybridModel")
PYTHON = str(Path(".venv/Scripts/python.exe").resolve())
ARMS = [
    ("sidecar-5m-s42", "examples/qwen35_sidecar_5m_s42.yml"),
    ("control-5m-s42", "examples/qwen35_sidecar_5m_control_s42.yml"),
    ("sidecar-5m-s43", "examples/qwen35_sidecar_5m_s43.yml"),
    ("control-5m-s43", "examples/qwen35_sidecar_5m_control_s43.yml"),
]


def log_path(name):
    return ROOT / "runs" / (name + ".log")


def result(name):
    """Final eval_loss and train_runtime from a finished arm's log, if it finished."""
    path = log_path(name)
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    runtime = re.findall(r"'train_runtime': '([0-9.]+)'", text)
    evals = re.findall(r"'eval_loss': '([0-9.]+)'", text)
    if not runtime:
        return {"state": "running or failed", "evals": len(evals),
                "last_eval": float(evals[-1]) if evals else None}
    return {"state": "done", "train_runtime": float(runtime[-1]),
            "eval_loss": float(evals[-1]) if evals else None}


def summarize():
    print("%-18s %-12s %12s %12s" % ("arm", "state", "train_runtime", "eval_loss"))
    table = {}
    for name, _ in ARMS:
        info = result(name) or {"state": "not started"}
        table[name] = info
        print("%-18s %-12s %12s %12s" % (
            name, info["state"],
            "%.0f s" % info["train_runtime"] if info.get("train_runtime") else "-",
            "%.4f" % info["eval_loss"] if info.get("eval_loss") is not None else "-"))

    pairs = [("sidecar-5m-s42", "control-5m-s42"), ("sidecar-5m-s43", "control-5m-s43")]
    deltas = []
    for sidecar, control in pairs:
        a, b = table.get(sidecar, {}), table.get(control, {})
        if a.get("eval_loss") is not None and b.get("eval_loss") is not None:
            delta = a["eval_loss"] - b["eval_loss"]
            deltas.append(delta)
            print("\n%s minus %s: %+.4f" % (sidecar, control, delta))
    if len(deltas) == 2:
        print("mean effect %+.4f, seed spread %.4f  (1M measured -0.0448)"
              % (sum(deltas) / 2, abs(deltas[0] - deltas[1])))
        print("Order-seed variance on a fixed config is 0.0005; two seeds bound the")
        print("wobble here, they do not establish it. Read the spread before the mean.")


def run(config, name, extra=()):
    command = [PYTHON, "-m", "distillkit.main", config, "-v", *extra]
    print("\n=== %s ===\n%s" % (name, " ".join(command)), flush=True)
    started = time.time()
    with open(log_path(name), "w", encoding="utf-8") as handle:
        code = subprocess.call(command, stdout=handle, stderr=subprocess.STDOUT)
    print("  exit %d after %.1f min" % (code, (time.time() - started) / 60), flush=True)
    return code


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.summary:
        summarize()
        return 0

    if args.smoke:
        # Against the 1M cache, which exists; the point is the code path, not the data.
        import yaml
        config = yaml.safe_load(open(ARMS[0][1], encoding="utf-8"))
        config["teacher"]["cache_path"] = str(ROOT / "teacher-cache-1m")
        config["output_path"] = str(ROOT / "runs" / "smoke-5m-path")
        config["training_args"]["max_steps"] = 3
        config["training_args"]["eval_steps"] = 3
        config["training_args"]["save_strategy"] = "no"
        path = Path("examples/_smoke_5m_path.yml")
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        code = run(str(path), "smoke-5m-path")
        path.unlink(missing_ok=True)
        if code == 0:
            print("\nstage 1 + tensor parallelism + sortish batching runs.")
        return code

    if not (ROOT / "teacher-cache-5m" / "manifest.json").is_file():
        print("teacher-cache-5m has no manifest: the capture has not finished.")
        return 1

    for name, config in ARMS:
        existing = result(name)
        if existing and existing["state"] == "done":
            print("skipping %s (already done: eval_loss %.4f)" % (name, existing["eval_loss"]))
            continue
        if run(config, name) != 0:
            print("STOPPING: %s failed; later arms would not be comparable anyway." % name)
            return 1
    summarize()
    return 0


if __name__ == "__main__":
    sys.exit(main())

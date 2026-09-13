"""Score every checkpoint of one gate run, on one card.

The content-NLL curve across checkpoints is the thing that says whether the gate learned
anything or merely ended up somewhere; a single endpoint cannot distinguish "improving"
from "noisy". Intermediate checkpoints are scored on the screen split alone. The final
one also gets the disjoint confirmation corpus, the fixed manual rule for comparison, and
a timing pass -- none of which are worth paying for five times.

    CUDA_VISIBLE_DEVICES=0 python scratch/residual_gate/score_run.py \
        --run D:/.../runs/gate-familiarity --output scratch/residual_gate/familiarity
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def checkpoints(run: Path):
    found = []
    for path in run.glob("gate-step*.pt"):
        match = re.search(r"gate-step(\d+)\.pt$", path.name)
        if match:
            found.append((int(match.group(1)), path))
    if not found:
        raise SystemExit("no gate checkpoints under %s" % run)
    return sorted(found)


def run_one(gate: Path, output: Path, split: str, extra=()):
    command = [sys.executable, "-u", str(HERE / "evaluate.py"),
               "--gate", str(gate), "--split", split, "--output", str(output)]
    command.extend(extra)
    completed = subprocess.run(command, env=dict(os.environ))
    if completed.returncode != 0:
        raise SystemExit("scoring %s failed" % gate)
    return json.loads(output.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    found = checkpoints(args.run)
    extra = ["--limit", str(args.limit)] if args.limit else []

    curve = []
    for step, gate in found:
        final = step == found[-1][0]
        report = run_one(gate, args.output / ("screen-step%d.json" % step), "screen",
                         extra + ([] if final else ["--skip-manual"])
                         + (["--timing"] if final else []))
        curve.append({"step": step,
                      "content": report["gated"]["delta"]["content"]["mean"],
                      "t": report["gated"]["delta"]["content"]["t"],
                      "all": report["gated"]["delta"]["all"]["mean"],
                      "identity_content":
                          report["forced_identity"]["delta"]["content"]["mean"]})
        print("step %-4d content %+.6f (t %+6.2f)  identity %+.9f"
              % (step, curve[-1]["content"], curve[-1]["t"],
                 curve[-1]["identity_content"]), flush=True)

    step, gate = found[-1]
    confirmation = run_one(gate, args.output / ("confirmation-step%d.json" % step),
                           "confirmation", extra + ["--skip-manual"])
    summary = {"run": str(args.run), "curve": curve,
               "final_step": step,
               "confirmation_content":
                   confirmation["gated"]["delta"]["content"]["mean"],
               "confirmation_t": confirmation["gated"]["delta"]["content"]["t"]}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2),
                                              encoding="utf-8")
    print(json.dumps(summary["curve"][-1], indent=2))
    print("confirmation content %+.6f (t %+6.2f)"
          % (summary["confirmation_content"], summary["confirmation_t"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

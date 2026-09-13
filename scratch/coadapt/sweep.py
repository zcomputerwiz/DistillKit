"""Score the paired co-adaptation arms on one fixed document set, content first.

Four things are measured at every matched checkpoint, and only the first decides anything:

    A_correct - B_control   the architecture question: did training with the memory
                            produce a better content model than ordinary co-training
                            from the same checkpoint?
    A_correct - A_off       how much this arm-A checkpoint still leans on the memory at
                            inference. Not the causal effect: arm A's backbone adapted
                            under memory-conditioned gradients, so switching the memory
                            off asks a different question.
    A_correct - A_wrong     whether the benefit still requires correct addressing.
    A_correct - A_random    (endpoint) whether the learned rows beat restored random ones.

Arm A's three modes come from a single evaluation of the arm A checkpoint; arm B needs its
own, so each milestone costs two model loads.

    python scratch/coadapt/sweep.py --output scratch/coadapt/sweep
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

BUNDLE = "scratch/independent-eval/full-bundle-384.json"
TOKENS_PER_UPDATE = 891.0            # measured: 253,200 supervised tokens over 284 updates


def evaluate(checkpoint: Path, output: Path, bundle: str, shuffle: int) -> None:
    if output.exists():
        print("skip %s" % output.name, flush=True)
        return
    command = [sys.executable, "-m", "distillkit.independent_eval", "evaluate",
               "--bundle", bundle, "--checkpoint", str(checkpoint),
               "--tasks", "nll", "--output", str(output)]
    if shuffle:
        command += ["--shuffle-context", str(shuffle)]
    print("scoring %s -> %s" % (checkpoint, output.name), flush=True)
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL)


def milestones(run: Path) -> dict[int, Path]:
    found = {int(path.name.split("-")[1]): path for path in run.glob("checkpoint-*")}
    return dict(sorted(found.items()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm-a", type=Path,
                        default=Path("../runs/coadapt-armA"))
    parser.add_argument("--arm-b", type=Path,
                        default=Path("../runs/coadapt-armB"))
    parser.add_argument("--bundle", default=BUNDLE)
    parser.add_argument("--shuffle-context", type=int, default=7)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    a_points, b_points = milestones(args.arm_a), milestones(args.arm_b)
    shared = sorted(set(a_points) & set(b_points))
    if not shared:
        raise SystemExit("the arms have no matching checkpoints to compare")

    index = []
    for updates in shared:
        a_out = args.output / ("A-%04d.json" % updates)
        b_out = args.output / ("B-%04d.json" % updates)
        evaluate(a_points[updates], a_out, args.bundle, args.shuffle_context)
        evaluate(b_points[updates], b_out, args.bundle, 0)
        index.append({"updates": updates, "tokens": int(updates * TOKENS_PER_UPDATE),
                      "arm_a": str(a_out), "arm_b": str(b_out),
                      "checkpoint_a": str(a_points[updates]),
                      "checkpoint_b": str(b_points[updates])})

    (args.output / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    print("\n%d matched checkpoints scored into %s" % (len(index), args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

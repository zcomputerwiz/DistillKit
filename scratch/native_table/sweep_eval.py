"""Evaluate every checkpoint of the native-table screen on one fixed document set.

Four questions per checkpoint, all from the same 384 held-out documents and the same
logits, so nothing can drift onto a different corpus between arms:

    ON - OFF                 does the sidecar help at all
    correct - wrong context  does what it retrieves have to match *this* text
    content vs layout        is the help lexical, or is it learning where newlines go
    trained - random rows    (final checkpoint only, via swap_table.py)

The sweep is sequential and each checkpoint is a separate process, because a 2B model
plus a 268.7M-row table is not worth holding two of on a 24 GiB card.

    python scratch/native_table/sweep_eval.py --output scratch/independent-eval/sweep
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

BUNDLE = "scratch/independent-eval/full-bundle-384.json"


def evaluate(checkpoint: Path, output: Path, bundle: str, shuffle: int) -> None:
    if output.exists():
        print("skip %s (already scored)" % output.name, flush=True)
        return
    command = [sys.executable, "-m", "distillkit.independent_eval", "evaluate",
               "--bundle", bundle, "--checkpoint", str(checkpoint),
               "--tasks", "nll", "--shuffle-context", str(shuffle),
               "--output", str(output)]
    print("scoring %s" % checkpoint, flush=True)
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", type=Path,
                        default=Path("../runs/native-ple-2b-ce"),
                        help="the 250-update checkpoint that the continuation resumed")
    parser.add_argument("--run", type=Path,
                        default=Path("../runs/native-ple-2b-ce-1m"),
                        help="the continuation run, whose checkpoint-N directories are swept")
    parser.add_argument("--bootstrap", type=Path,
                        default=Path("../runs/native-ple-2b-bootstrap"))
    parser.add_argument("--bundle", default=BUNDLE)
    parser.add_argument("--shuffle-context", type=int, default=7)
    parser.add_argument("--first-updates", type=int, default=250,
                        help="updates already spent before the continuation began")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    points = [(0, args.bootstrap), (args.first_updates, args.first)]
    for path in sorted(args.run.glob("checkpoint-*"),
                       key=lambda p: int(p.name.split("-")[1])):
        points.append((args.first_updates + int(path.name.split("-")[1]), path))

    index = []
    for updates, checkpoint in points:
        output = args.output / ("updates-%04d.json" % updates)
        evaluate(checkpoint, output, args.bundle, args.shuffle_context)
        index.append({"updates": updates, "checkpoint": str(checkpoint),
                      "result": str(output)})
    (args.output / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    print("\n%d checkpoints scored into %s" % (len(index), args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

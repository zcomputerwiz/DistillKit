"""Run the substitution grid across both cards and merge the halves.

Two resident-model workers, one card each, masked with CUDA_VISIBLE_DEVICES. No DDP and
no sharding: inference jobs are independent, and a 2B model fits twice over. Each worker
recomputes the baseline on its own card so every delta it reports is a difference between
two numbers produced by the same process in the same dtype.

    python scratch/ffn_memo/launch.py --output scratch/ffn_memo/grid
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--cache", default="scratch/ffn_memo/cache")
    args = parser.parse_args()

    sys.path.insert(0, str(HERE))
    from substitute import ARMS, LAYER_SETS, THRESHOLDS

    total = len([1 for layers in LAYER_SETS for threshold in THRESHOLDS for arm in ARMS
                 if arm == "exact" or (threshold == (4, None) and len(layers) == 1)])
    half = (total + 1) // 2
    args.output.mkdir(parents=True, exist_ok=True)

    processes = []
    for card, (start, stop) in enumerate([(0, half - 1), (half, total - 1)]):
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = str(card)
        environment["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        command = [sys.executable, "-u", str(HERE / "substitute.py"),
                   "--jobs", "%d-%d" % (start, stop), "--cache", args.cache,
                   "--output", str(args.output / ("worker%d.json" % card))]
        if args.limit:
            command += ["--limit", str(args.limit)]
        log = (args.output / ("worker%d.log" % card)).open("w", encoding="utf-8")
        print("card %d: jobs %d-%d of %d" % (card, start, stop, total), flush=True)
        processes.append((card, subprocess.Popen(command, env=environment,
                                                 stdout=log, stderr=subprocess.STDOUT),
                          log))

    failures = []
    for card, process, log in processes:
        code = process.wait()
        log.close()
        if code != 0:
            failures.append(card)
    if failures:
        raise SystemExit("worker(s) %s failed; logs in %s" % (failures, args.output))

    merged = None
    for card in range(2):
        part = json.loads(
            (args.output / ("worker%d.json" % card)).read_text(encoding="utf-8"))
        if merged is None:
            merged = part
        else:
            merged["results"].extend(part["results"])
            merged["elapsed_seconds"] = max(merged["elapsed_seconds"],
                                            part["elapsed_seconds"])
    merged["results"].sort(key=lambda entry: (entry["layers"], entry["arm"],
                                              entry["min_count"],
                                              entry["max_variance"] or 0))
    (args.output / "grid.json").write_text(json.dumps(merged, indent=2), encoding="utf-8")
    print("%d results merged into %s" % (len(merged["results"]),
                                         args.output / "grid.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

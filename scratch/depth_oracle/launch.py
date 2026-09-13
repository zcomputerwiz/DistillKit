"""Run the oracle grid across both cards and merge the halves.

Each worker is an ordinary single-GPU process masked with CUDA_VISIBLE_DEVICES, holding
one model resident and working through its own slice of the job list. No DDP and no
sharding: two 2B models fit trivially, and inference jobs are embarrassingly parallel.

Both workers recompute the baseline, which is the one duplicated cost. That is deliberate
-- a baseline measured on the same card, in the same process, in the same dtype as the
interventions it is subtracted from is worth more than the couple of minutes it costs.

    python scratch/depth_oracle/launch.py --output scratch/depth_oracle/grid
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
    parser.add_argument("--jobs", type=int, default=None,
                        help="total jobs; defaults to the worker's full grid")
    args = parser.parse_args()

    sys.path.insert(0, str(HERE))
    from oracle import BANDS, ORACLES

    total = args.jobs or len(BANDS) * len(ORACLES)
    half = (total + 1) // 2
    slices = [(0, half - 1), (half, total - 1)]

    args.output.mkdir(parents=True, exist_ok=True)
    processes = []
    for card, (start, stop) in enumerate(slices):
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = str(card)
        environment["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        command = [sys.executable, "-u", str(HERE / "oracle.py"),
                   "--schedules", "%d-%d" % (start, stop),
                   "--output", str(args.output / ("worker%d.json" % card))]
        if args.limit:
            command += ["--limit", str(args.limit)]
        log = (args.output / ("worker%d.log" % card)).open("w", encoding="utf-8")
        print("card %d: jobs %d-%d" % (card, start, stop), flush=True)
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
        raise SystemExit("worker(s) %s failed; see the logs in %s"
                         % (failures, args.output))

    merged = None
    for card in range(len(slices)):
        part = json.loads((args.output / ("worker%d.json" % card)).read_text(encoding="utf-8"))
        if merged is None:
            merged = part
        else:
            merged["schedules"].extend(part["schedules"])
            merged["elapsed_seconds"] = max(merged["elapsed_seconds"],
                                            part["elapsed_seconds"])
    merged["schedules"].sort(key=lambda entry: (entry["band"], entry["oracle"]))
    (args.output / "grid.json").write_text(json.dumps(merged, indent=2), encoding="utf-8")
    print("%d schedules merged into %s" % (len(merged["schedules"]),
                                           args.output / "grid.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

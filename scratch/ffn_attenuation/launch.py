"""Run the attenuation grid across both cards and merge deterministically.

Two resident-model workers, one card each, masked with CUDA_VISIBLE_DEVICES. No DDP, no
sharding. Each worker recomputes the baseline on its own card, so every delta it reports
is a difference between two numbers produced in the same process and dtype -- and the
merge is a concatenation of independent results rather than an averaging of them.

    python scratch/ffn_attenuation/launch.py --output scratch/ffn_attenuation/grid
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
    parser.add_argument("--split", default="screen")
    parser.add_argument("--bundle", default=None)
    args = parser.parse_args()

    sys.path.insert(0, str(HERE))
    from attenuate import jobs_for

    total = len(jobs_for())
    half = (total + 1) // 2
    args.output.mkdir(parents=True, exist_ok=True)

    processes = []
    for card, (start, stop) in enumerate([(0, half - 1), (half, total - 1)]):
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = str(card)
        environment["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        command = [sys.executable, "-u", str(HERE / "attenuate.py"),
                   "--jobs", "%d-%d" % (start, stop), "--split", args.split,
                   "--output", str(args.output / ("worker%d.json" % card))]
        if args.limit:
            command += ["--limit", str(args.limit)]
        if args.bundle:
            command += ["--bundle", args.bundle]
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
    merged["results"].sort(key=lambda entry: (entry["layer"], entry["kind"],
                                              entry["mask"], entry["alpha"]))
    (args.output / "grid.json").write_text(json.dumps(merged, indent=2), encoding="utf-8")
    print("%d results merged into %s" % (len(merged["results"]),
                                         args.output / "grid.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

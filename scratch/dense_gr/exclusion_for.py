"""An exclusion list trimmed to the documents a run's captures actually hold.

The trainer refuses an exclusion id that is in none of the captures it reads -- a list
meant for another corpus would otherwise exclude nothing and report success -- so a run
over a subset of captures needs the master list cut down to that subset.

    python scratch/dense_gr/exclusion_for.py --master ../capture-data/exclude-broken-tools.json \\
        --caches ../teacher-cache-think-first ../teacher-cache-thinking --output <path>
"""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master", type=Path, nargs="+", required=True, help="lists to combine")
    parser.add_argument("--caches", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    held = set()
    for cache in args.caches:
        manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
        for shard in manifest["shards"]:
            held.update(shard["doc_ids"])
    master = {i for path in args.master for i in json.load(open(path, encoding="utf-8"))}
    kept = sorted(master & held)
    json.dump(kept, open(args.output, "w"), indent=1)
    print("%d of %d excluded ids are in these captures -> %s" % (len(kept), len(master), args.output))


if __name__ == "__main__":
    main()

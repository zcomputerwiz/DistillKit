"""Clean rollouts to leave out of training: unverified, or verified but hedging.

Round 2 trained every clean rollout with cross entropy, and "clean" only meant finished
without looping wherever no reference answer existed -- so the student learned its own
unchecked code (it passes ~40% of HumanEval+) and its own hedging, and HumanEval+ fell 5
points while thinking-mode hedge markers doubled. Round 1's KL-only training on rollouts
also coincided with a HumanEval+ drop, so the unverified ones are dropped, not demoted:
cross entropy on the rollouts verified correct, nothing on the rest.

    python scratch/dense_gr/unverified_ids.py ../capture-data/onpolicy-r2-clean.jsonl \\
        --output ../capture-data/drop-r2.json
"""
import argparse
import json
import re
from pathlib import Path

HEDGE = re.compile(r"(?:^|\n|[.!?] )(?:Wait|Actually|Hmm|Hold on)\b")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clean", type=Path)
    parser.add_argument("--max-hedges", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    chosen, total, unverified, hedging = [], 0, 0, 0
    for line in open(args.clean, encoding="utf-8"):
        row = json.loads(line)
        total += 1
        thought = row["text"][row["prompt_chars"]:].split("</think>")[0]
        if row.get("correct") is not True:
            unverified += 1
        elif len(HEDGE.findall(thought)) > args.max_hedges:
            hedging += 1
        else:
            continue
        chosen.append(row["doc_id"])
    json.dump(chosen, open(args.output, "w"), indent=1)
    print("%d of %d clean rollouts dropped: %d unverified, %d verified but hedging -> %s"
          % (len(chosen), total, unverified, hedging, args.output))


if __name__ == "__main__":
    main()

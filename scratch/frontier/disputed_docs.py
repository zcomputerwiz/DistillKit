"""The documents the frontier judge called wrong, with their text, for a second judge.

    python scratch/frontier/disputed_docs.py --responses <judge responses...> --output disputed.jsonl
"""
import argparse
import json
from pathlib import Path

from judge_docs import TOKENIZER, verdicts_of

RECAPTURES = ["../capture-data/recapture-think-first.jsonl", "../capture-data/recapture-thinking.jsonl"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--responses", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    wrong = {doc_id: verdict for path in args.responses for row in map(json.loads, open(path, encoding="utf-8"))
             for doc_id, verdict in verdicts_of(row) if verdict.get("final_correct") == "no"}
    with open(args.output, "w", encoding="utf-8") as out:
        for path in RECAPTURES:
            for row in map(json.loads, open(path, encoding="utf-8")):
                if row["doc_id"] in wrong:
                    out.write(json.dumps({"id": row["doc_id"], "conversation": tokenizer.decode(row["input_ids"]),
                                          "first_judge": wrong[row["doc_id"]]}) + "\n")
    print("%d disputed documents -> %s" % (len(wrong), args.output))


if __name__ == "__main__":
    main()

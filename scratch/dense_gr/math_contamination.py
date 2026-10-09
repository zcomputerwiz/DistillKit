"""Which training documents contain a GSM8K test or MATH-500 question?

The automatic expand_corpus.py banks cover MMLU and ARC; they do not establish
HumanEval/MBPP exclusion coverage. GSM8K and MATH-500 joined the evaluation later.
This script screens only those two math banks. It matches every
document's user turns against both test sets -- exact after whitespace and case
normalization, and by 13-word shingle containment, the screen `expand_corpus.py` uses --
and writes the ids of the documents that contain one.

    python scratch/dense_gr/math_contamination.py --corpora ../capture-data/*.jsonl --output ../capture-data/exclude-math-benchmarks.json
"""
import argparse
import json
import re
from pathlib import Path

USER = re.compile(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", re.S)
TOKENIZER = Path(__file__).resolve().parents[3] / "teacher-hf" / "tokenizer.json"
_decoder = []


def user_text(row):
    """A row's user turns: from its `text`, or its `input_ids` decoded (capture input),
    else its rendered `prompt`."""
    text = row.get("text", "")
    if not text and "input_ids" in row:
        if not _decoder:
            from tokenizers import Tokenizer

            _decoder.append(Tokenizer.from_file(str(TOKENIZER)))
        text = _decoder[0].decode(row["input_ids"], skip_special_tokens=False)
    return " ".join(USER.findall(text)) or row.get("prompt", "")


def words(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def shingles(text, n=13):
    w = words(text)
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpora", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from datasets import load_dataset

    tests = [("gsm8k", r["question"]) for r in load_dataset("openai/gsm8k", "main", split="test")]
    tests += [("math500", r["problem"]) for r in load_dataset("HuggingFaceH4/MATH-500", split="test")]
    exact = {" ".join(words(q)): name for name, q in tests}
    bank = {}
    for name, question in tests:
        for s in shingles(question):
            bank[s] = name
    hits, per = [], {}
    for path in args.corpora:
        for line in open(path, encoding="utf-8"):
            row = json.loads(line)
            text = user_text(row)
            key = " ".join(words(text))
            found = exact.get(key)
            if found is None:
                found = next((bank[s] for s in shingles(text) if s in bank), None)
            if found:
                hits.append(row["doc_id"])
                k = "%s / %s" % (path.name, found)
                per[k] = per.get(k, 0) + 1
    json.dump(sorted(set(hits)), open(args.output, "w"), indent=1)
    for key, n in sorted(per.items()):
        print("%-60s %5d" % (key, n))
    print("%d documents contain a GSM8K-test or MATH-500 question -> %s" % (len(set(hits)), args.output))


if __name__ == "__main__":
    main()

"""Rescore saved GSM8K / MATH-500 results on the problems no training document contains.

The thinking corpora carry GSM8K test and MATH-500 questions (math_contamination.py): the
screen they were built with predates those benchmarks. Every student since the thinking
pass trained on about a fifth of each with frontier-model solutions; the source did not.
This finds the test problems present in the given corpora (exact after normalization, or
a shared 13-word run) and reports each saved result's accuracy on the rest.

    python scratch/downstream/math_bench/clean_rescore.py --corpora <jsonl...> [--tags source think ...]
"""
import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
USER = re.compile(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", re.S)


def words(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def shingles(text, n=13):
    w = words(text)
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpora", type=Path, nargs="+", required=True)
    parser.add_argument("--tags", nargs="+", default=None)
    args = parser.parse_args()
    from datasets import load_dataset

    tests = {"gsm8k": [(str(i), r["question"]) for i, r in enumerate(load_dataset("openai/gsm8k", "main", split="test"))],
             "math500": [(r["unique_id"], r["problem"]) for r in load_dataset("HuggingFaceH4/MATH-500", split="test")]}
    seen_exact, seen_shingles = set(), set()
    for path in args.corpora:
        for line in open(path, encoding="utf-8"):
            row = json.loads(line)
            text = " ".join(USER.findall(row.get("text", ""))) or row.get("prompt", "")
            seen_exact.add(" ".join(words(text)))
            seen_shingles |= shingles(text)
    dirty = {bench: {i for i, q in items
                     if " ".join(words(q)) in seen_exact or shingles(q) & seen_shingles}
             for bench, items in tests.items()}
    for bench, ids in dirty.items():
        print("%s: %d of %d test problems appear in the corpora" % (bench, len(ids), len(tests[bench])))
    rows = []
    for results in sorted(HERE.glob("*/results.json")):
        name = results.parent.name
        match = re.match(r"(.+)-(gsm8k|math500)-(think-sampled|nothink-greedy)$", name)
        if not match or (args.tags and match.group(1) not in args.tags):
            continue
        tag, bench, mode = match.groups()
        records = json.load(open(results))["records"]
        clean = [r for r in records if str(r["id"]) not in dirty[bench]]
        seen = [r for r in records if str(r["id"]) in dirty[bench]]
        acc = lambda rs: 100 * sum(r["correct"] for r in rs) / max(len(rs), 1)
        rows.append((bench, mode, tag, acc(records), acc(clean), acc(seen), len(clean)))
    print("\n%-8s %-15s %-16s %7s %7s %11s" % ("bench", "mode", "arm", "all", "clean", "contaminated"))
    for bench, mode, tag, a, c, s, n in sorted(rows):
        print("%-8s %-15s %-16s %6.1f%% %6.1f%% %10.1f%%" % (bench, mode, tag, a, c, s))


if __name__ == "__main__":
    main()

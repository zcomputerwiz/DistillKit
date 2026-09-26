"""Remix the chat corpus into the native thinking format: reasoning in the first think block.

The 5M corpus renders every assistant turn as an empty, closed think block followed by a
second one holding the reasoning: `<think>\\n\\n</think>\\n\\n<think>\\nR</think>A`. The code
and chat captures have the empty block and then the answer. So in training no assistant
turn ever opens straight into reasoning -- and that is exactly where a thinking-mode prompt
(`...assistant\\n<think>\\n`) puts the model. The source, a native Qwen thinking model,
handles it; the distilled students drift into long reasoning and repetition.

This writes each such turn as `<think>\\nR</think>A`, Qwen's own thinking format, so a run
can train on both conventions: the originals for non-thinking, this remix for thinking.
De-hedged text is used where a document was de-hedged. Train split only, so held-out
samples stay comparable across runs; turns without reasoning are left as they are.

    python scratch/dense_gr/think_first.py --input ../capture-data/run5m.jsonl \\
        --dehedged ../capture-data/dehedged-5m.jsonl --output ../capture-data/think-first-5m.jsonl
"""
import argparse
import json
import re
from pathlib import Path

DOUBLE = re.compile(r"<think>\s*</think>\s*<think>\n?(.*?)</think>", re.S)


def remix(text):
    """The text with every empty-then-real think pair folded into one; and how many."""
    count = [0]

    def fold(match):
        body = match.group(1)
        if len(body.strip()) < 20:
            return match.group(0)
        count[0] += 1
        return "<think>\n%s</think>" % body

    return DOUBLE.sub(fold, text), count[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--dehedged", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite %s" % args.output)
    replaced = {}
    if args.dehedged:
        for line in open(args.dehedged, encoding="utf-8"):
            row = json.loads(line)
            replaced[row["rewritten_from"]] = row["text"]
    written = turns = from_dehedged = 0
    with open(args.output, "w", encoding="utf-8") as out:
        for line in open(args.input, encoding="utf-8"):
            row = json.loads(line)
            if row.get("split", "train") != "train":
                continue
            text = replaced.get(row["doc_id"], row["text"])
            new, count = remix(text)
            if not count:
                continue
            from_dehedged += row["doc_id"] in replaced
            turns += count
            written += 1
            out.write(json.dumps(dict(row, doc_id=row["doc_id"] + ":tf", text=new,
                                      remixed_from=row["doc_id"])) + "\n")
    print("wrote %d documents (%d from de-hedged text), %d turns folded -> %s"
          % (written, from_dehedged, turns, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

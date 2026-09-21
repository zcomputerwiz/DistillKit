"""How much of a capped document is still the prompt when training scores it.

Both objectives score every position but the last, so a document's system prompt and
user turn are trained on exactly like its answer. That is fine as long as the answer is
in there. The prefix cap makes it a question: `--teacher-max-length 1024` keeps a
document's first 1024 tokens, and a document whose framing runs longer than that is
scored entirely on framing -- the model is trained to predict a system prompt it will
never be asked to produce, and never sees the response the capture was made for.

This counts it rather than assuming either way, at each cap that has been used, for the
training corpus and for the documents the held-out loss is computed on.

The answer is located by the chat template's own marker, the token run for
`<|im_start|>assistant`, because that is what actually separates the two in the ids the
model reads. The last such marker is not the interesting one: a multi-turn document has
several, and what matters is whether *any* response survives the cap.

    python scratch/dense_gr/prefix_audit.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from teacher_kl import first_response  # noqa: E402,F401

CAPS = (1024, 2048, 4096)
TEACHER = "D:/DeepThought/Projects/HybridModel/teacher-hf"
MARKER = "<|im_start|>assistant"


def marker_ids(tokenizer):
    """The token run that opens an assistant turn."""
    ids = tokenizer(MARKER, add_special_tokens=False)["input_ids"]
    if not ids:
        raise ValueError("the tokenizer does not encode the assistant marker")
    return ids


def audit(path, tokenizer, marker, caps=CAPS, limit=None):
    rows, missing = [], 0
    with open(path, encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if limit is not None and index >= limit:
                break
            record = json.loads(line)
            ids = tokenizer(record["text"])["input_ids"]
            start = first_response(ids, marker)
            if start is None:
                missing += 1
                continue
            rows.append((len(ids), start, record.get("split", "train"),
                         record.get("source")))
    return rows, missing


def report(name, rows, missing, caps=CAPS):
    total = len(rows) + missing
    print("\n%s: %d documents" % (name, total))
    if missing:
        print("  %d carry no assistant turn at all" % missing)
    print("  %-6s %10s %10s %12s %14s"
          % ("cap", "no answer", "share", "answer toks", "answer share"))
    for cap in caps:
        blind = sum(1 for length, start, _, _ in rows if start >= cap - 1)
        # Positions 0..min(length, cap)-2 are scored; the answer's are those at or
        # after `start`. A document whose marker sits at the very last kept position
        # contributes nothing, which is why `blind` uses `cap - 1`.
        answer = sum(max(0, min(length, cap) - 1 - start) for length, start, _, _ in rows)
        scored = sum(min(length, cap) - 1 for length, start, _, _ in rows)
        print("  %-6d %10d %9.2f%% %12d %13.1f%%"
              % (cap, blind, 100.0 * blind / max(total, 1), answer,
                 100.0 * answer / max(scored, 1)))
    prompts = sorted(start for _, start, _, _ in rows)
    if prompts:
        def pct(fraction):
            return prompts[min(len(prompts) - 1, int(fraction * len(prompts)))]
        print("  prompt length before the first answer token: median %d, p90 %d, "
              "p99 %d, max %d" % (pct(0.5), pct(0.9), pct(0.99), prompts[-1]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", default=TEACHER)
    parser.add_argument("--corpus", type=Path, nargs="+",
                        default=[Path("../capture-data/run5m.jsonl"),
                                 Path("../capture-data/expand.jsonl"),
                                 Path("../capture-data/heldout.jsonl")])
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    marker = marker_ids(tokenizer)
    print("assistant marker: %s" % marker)

    for path in args.corpus:
        if not path.exists():
            print("\n%s: missing" % path)
            continue
        rows, missing = audit(path, tokenizer, marker, limit=args.limit)
        report(str(path), rows, missing)
        # The held-out loss during training reads the eval split, and only the first
        # 128 of it, so that subset is worth its own line rather than an average over
        # documents the training loop never scores.
        held = [row for row in rows if row[2] == "eval"][:128]
        if held:
            report("%s  (first 128 eval documents)" % path, held, 0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

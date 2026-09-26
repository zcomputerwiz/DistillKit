"""Take the hedging out of the corpus's reasoning blocks by rule, leaving the content.

The student picked up the teacher corpus's "Wait, let me reconsider" style and loops on it.
Two teacher-driven edits were tried first and both damaged content (`rewrite_hedges.py`):
a free rewrite drifted -- a scansion walkthrough reasoned about different verses than the
answer gives -- and a delete-only pass that let the int8 teacher pick sentences dropped
correct caveats ("Python integers are arbitrary precision, so this won't work directly")
while keeping the claims they corrected.

So the edit is narrow and mechanical, inside `<think>` blocks only:

* a sentence that is nothing but a hedge -- "Wait, let me reconsider.", "Hmm.", "Actually,
  let me think about this differently.", "Let me re-read the problem." -- is removed;
* a sentence that opens with a hedge and then says something keeps the something:
  "Wait - the constraint says X" becomes "The constraint says X".

Every fact, correction and caveat stays. Tool calls, code and answers are never touched.

    python scratch/dense_gr/dehedge.py --input ../capture-data/run5m.jsonl \\
        --output ../capture-data/dehedged-5m.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rewrite_hedges import HEDGE, THINK, segments  # noqa: E402

OPENER = re.compile(
    r"^(?P<lead>\s*)(?:(?:oh,?\s+)?(?:but\s+)?(?:wait|actually|hmm+|hold on|hm)\b"
    r"(?:\s*[,.!:—–-]+)?\s*)+", re.I)
# What is left when a sentence was only a hedge: filler about thinking, with no content.
FILLER = re.compile(
    r"^\s*(?:(?:no|so|ok(?:ay)?|right|yes|well)[,.]?\s*)?"
    r"(?:let me\s+(?:re-?read|re-?check|re-?consider|reconsider|re-?think|rethink|re-?examine|"
    r"re-?trace|re-?verify|double[- ]check|think|check|look|trace|verify|recount|re-?count|"
    r"go back|step back|start over|try again|be more careful)"
    r"(?:\s+(?:this|that|it|again|carefully|more carefully|differently|once more|the problem|"
    r"the question|the task|the test|the example|the requirements?|the constraints?|about this|"
    r"about it|this again|through this|at this|at this again|at it|this through|"
    r"one more time|the logic|my approach|my work|step by step|the problem again|the test again))*"
    r"|i (?:think|need to (?:reconsider|re-?read|re-?check|think again))|that'?s? (?:not right|wrong)"
    r"|i made (?:a|an) (?:mistake|error)|something'?s? (?:off|wrong)|i'?m (?:confused|overthinking(?: this)?)"
    r"|this is (?:tricky|confusing)|let me see|hmm+)?"
    r"\s*[.!:;,…]*\s*$", re.I)


def edit(sentence):
    """`(new sentence, changed)`; an empty result means the sentence goes."""
    match = OPENER.match(sentence)
    if not match:
        return sentence, False
    lead = match.group("lead")
    rest = sentence[match.end():]
    if FILLER.fullmatch(rest.strip()):
        return "", True
    rest = rest.lstrip()
    return lead + rest[:1].upper() + rest[1:], True


def edit_block(block):
    """The block with hedges edited out, and how many sentences were removed or trimmed."""
    out, removed, trimmed = [], 0, 0
    for text, editable in segments(block):
        if not editable:
            out.append(text)
            continue
        body = text.rstrip()
        tail = text[len(body):]
        new, changed = edit(body)
        if not changed:
            # A bare filler sentence with no hedge word, e.g. "Let me reconsider."
            if FILLER.fullmatch(body.strip()) and HEDGE.search(body):
                removed += 1
                out.append(tail if "\n" in tail else "")
                continue
            out.append(text)
        elif not new.strip():
            removed += 1
            out.append(tail if "\n" in tail else "")
        else:
            trimmed += 1
            out.append(new + tail)
    return "".join(out), removed, trimmed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=0,
                        help="print this many edited sentences before and after")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite %s" % args.output)
    rows = [json.loads(line) for line in open(args.input, encoding="utf-8")]
    edited = removed = trimmed = before = after = 0
    shown = 0
    with open(args.output, "w", encoding="utf-8") as out:
        for row in rows:
            if row.get("split", "train") != "train":
                continue
            counts = [0, 0]

            def swap(match):
                block = match.group(1)
                new, r, t = edit_block(block)
                counts[0] += r
                counts[1] += t
                return "<think>%s</think>" % new

            text = THINK.sub(swap, row["text"])
            if text == row["text"]:
                continue
            edited += 1
            removed += counts[0]
            trimmed += counts[1]
            old_blocks, new_blocks = THINK.findall(row["text"]), THINK.findall(text)
            before += sum(len(HEDGE.findall(b)) for b in old_blocks)
            after += sum(len(HEDGE.findall(b)) for b in new_blocks)
            if shown < args.samples:
                for ob in old_blocks:
                    for o, _ in segments(ob):
                        n, changed = edit(o.rstrip())
                        if changed and shown < args.samples:
                            print("  - %s\n  + %s" % (o.strip()[:140], (n.strip() or "(removed)")[:140]))
                            shown += 1
            out.write(json.dumps(dict(row, doc_id=row["doc_id"] + ":dh", text=text,
                                      rewritten_from=row["doc_id"])) + "\n")
    print("edited %d documents: %d hedge sentences removed, %d trimmed; hedge words in "
          "reasoning %d -> %d -> %s" % (edited, removed, trimmed, before, after, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

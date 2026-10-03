"""How does the teacher score the student's thinking loops, occurrence by occurrence?

The student's looping MATH rollouts (math_truncation.py) are captured through the teacher
(loop-check.jsonl). Each rollout is split into lines; a line's occurrence `k` is how many
times the same line came before it. Per occurrence bucket this reports the teacher's
probability of the actual tokens over the line, and at the line's first token -- the
point where the loop could be left -- how often the teacher's top-1 is to repeat it, with
examples of what it would write instead.

    python scratch/dense_gr/loop_teacher.py --cache ../teacher-cache-loop-check \\
        --inputs ../capture-data/loop-check.jsonl
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
BUCKETS = [(0, 0), (1, 1), (2, 2), (3, 5), (6, 1 << 30)]


def label(k):
    for lo, hi in BUCKETS:
        if lo <= k <= hi:
            return "k=%d" % lo if lo == hi else "k=%d-%s" % (lo, hi if hi < 1 << 29 else "")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer

    from long_context_probe import TOKENIZER
    from teacher_kl import CachedTeacher

    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    prompt_len = {r["doc_id"]: r["prompt_len"] for r in map(json.loads, open(args.inputs, encoding="utf-8"))}
    teacher = CachedTeacher(args.cache, "train", device="cpu")
    stats, examples = {}, {}
    for doc_id in teacher.ids:
        r = teacher.cache.read_document(doc_id)
        ids, top, lp = np.asarray(r["input_ids"]), r["topk_ids"], r["topk_logprobs"].astype(np.float32)
        start = prompt_len[doc_id]
        # Lines of the generated text, as token index ranges, split after any newline token.
        pieces = [tok.decode([t]) for t in ids[start:]]
        lines, begin = [], start
        for offset, piece in enumerate(pieces):
            if "\n" in piece:
                lines.append((begin, start + offset + 1))
                begin = start + offset + 1
        seen = Counter()
        for a, b in lines:
            if b - a < 6 or a < 1:
                continue
            key = tuple(ids[a:b])
            k = seen[key]
            seen[key] += 1
            # Position t predicts token t + 1: the line's tokens a..b-1 are predicted at a-1..b-2.
            at = np.arange(a - 1, b - 1)
            hit = top[at] == ids[at + 1][:, None]
            p = np.where(hit.any(1), np.exp(np.where(hit, lp[at], -np.inf).max(1)), 0.0)
            s = stats.setdefault(label(k), [0, 0.0, 0, 0, 0.0])
            s[0] += len(at)
            s[1] += float(p.sum())
            s[2] += 1
            s[3] += int(top[a - 1, 0] == ids[a])     # teacher's top-1 at the line start repeats it
            s[4] += float(p[0])                       # its probability of the line's first token
            if k >= 1 and top[a - 1, 0] != ids[a] and len(examples.setdefault(label(k), [])) < 4:
                examples[label(k)].append((tok.decode(ids[a:b]).strip()[:70], tok.decode([top[a - 1, 0]]),
                                           round(float(np.exp(lp[a - 1, 0])), 2)))
    print("%-8s %6s %10s %18s %16s" % ("occur.", "lines", "p(tokens)", "top-1 repeats line", "p(first token)"))
    for key in [label(lo) for lo, _ in BUCKETS]:
        if key in stats:
            n, psum, lines_, rep, pfirst = stats[key]
            print("%-8s %6d %10.3f %18.3f %16.3f" % (key, lines_, psum / n, rep / lines_, pfirst / lines_))
    for key, rows in examples.items():
        print("--", key, "where the teacher would not repeat: (line, its top-1 instead, p)")
        for row in rows:
            print("  ", row)


if __name__ == "__main__":
    main()

"""Looping rollouts the unlikelihood detector finds no loop in, as an exclusion list.

`--unlikelihood-caches` trains a looping rollout on KL everywhere except the tokens
`loop_tokens` marks, which get unlikelihood instead. A rollout where it marks nothing --
the thought closed before the loop began, or the repeats vary -- would be trained on the
teacher's KL over its whole loop, and the teacher, given a loop, continues it
(loop_teacher.py). Those rollouts are listed for `--exclude-documents`.

    python scratch/dense_gr/loop_negatives.py --caches ../teacher-cache-onpolicy-r6-loop ... \\
        --output ../capture-data/exclude-zero-negative-loops.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from teacher_kl import last_response, loop_tokens  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--caches", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer

    from distillkit.offline_cache import OfflineTeacherCache
    from long_context_probe import TOKENIZER

    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    marker = tok.encode("<|im_start|>assistant\n", add_special_tokens=False)
    close = tok.convert_tokens_to_ids("</think>")
    zero = []
    for path in args.caches:
        cache = OfflineTeacherCache(path)
        counts = []
        for doc_id in cache.documents:
            ids = np.asarray(cache.read_document(doc_id, tokens_only=True)["input_ids"])
            start = last_response(ids, marker)
            negatives = 0 if start is None else int(loop_tokens(ids, start, close).sum())
            counts.append(negatives)
            if not negatives:
                zero.append(doc_id)
        print("%s: %d rollouts, %d with no detected loop, %d negatives"
              % (path.name, len(counts), sum(c == 0 for c in counts), sum(counts)))
    args.output.write_text(json.dumps(sorted(zero), indent=0), encoding="utf-8")
    print("%d rollouts to exclude -> %s" % (len(zero), args.output))


if __name__ == "__main__":
    main()

"""Training documents that contain a fresh-bank problem, as an exclusion list.

The loop gate's fresh bank (math_truncation.fresh_bank: MATH test outside MATH-500) must
stay out of training, but MATH repeats a few problems across its splits, so a rollout on
a train problem can carry a test one (Codex recheck r4: two bank problems in three r6-loop
rollouts). Excluding those documents keeps the bank -- and the base's saved result on it
-- valid.

    python scratch/dense_gr/fresh_bank_overlap.py --caches ../teacher-cache-onpolicy-r6-loop ... \\
        --output ../capture-data/exclude-fresh-bank-overlap.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--caches", type=Path, nargs="+", required=True)
    parser.add_argument("--count", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer

    from distillkit.offline_cache import OfflineTeacherCache
    from long_context_probe import TOKENIZER
    from math_truncation import fresh_bank

    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    bank = [" ".join(q.split()) for q, _ in fresh_bank(args.count)]
    hits = []
    for path in args.caches:
        cache = OfflineTeacherCache(path)
        found = 0
        for doc_id in cache.documents:
            text = " ".join(tok.decode(cache.read_document(doc_id, tokens_only=True)["input_ids"]).split())
            if any(q in text for q in bank):
                hits.append(doc_id)
                found += 1
        print("%s: %d documents contain a fresh-bank problem" % (path.name, found))
    args.output.write_text(json.dumps(sorted(hits), indent=0), encoding="utf-8")
    print("%d documents -> %s" % (len(hits), args.output))


if __name__ == "__main__":
    main()

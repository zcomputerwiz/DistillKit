"""Audit replay traces for likely errors: teacher agreement and code that does not parse.

The on-policy rollouts are checked (math answers against references, code against tests),
but the replay corpora -- most of every round's tokens, trained half on cross entropy --
never were. Two cheap signals, no generation:

- teacher agreement: the teacher's mean log-probability of the trace's own answer tokens,
  read from the capture (top-k; a token outside it is charged the capture's floor). A
  trace the 27B finds implausible is a candidate for a wrong or garbled solution.
- code that does not parse: every fenced python block through ast.parse.

    python scratch/dense_gr/trace_audit.py --caches ../teacher-cache-thinking ... --corpora ../capture-data/thinking-code-math.jsonl ...
"""
import argparse
import ast
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from distillkit.offline_cache import OfflineTeacherCache  # noqa: E402
from teacher_kl import last_response  # noqa: E402

BLOCK = re.compile(r"```(?:python|py)\s*\n(.*?)```", re.DOTALL)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--caches", type=Path, nargs="+", required=True)
    parser.add_argument("--corpora", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("D:/DeepThought/Projects/HybridModel/teacher-hf")
    marker = tok("<|im_start|>assistant", add_special_tokens=False)["input_ids"]
    meta = {}
    for path in args.corpora:
        for line in open(path, encoding="utf-8"):
            row = json.loads(line)
            meta[row["doc_id"]] = (path.stem, row.get("source"), row.get("domain"), row["text"])
    rows = []
    for cache_path in args.caches:
        cache = OfflineTeacherCache(cache_path)
        for doc_id in cache.document_ids():
            if doc_id not in meta:
                continue
            record = cache.read_document(doc_id, include_hidden_states=False)
            ids = np.asarray(record["input_ids"], dtype=np.int64)
            top_ids = np.asarray(record["topk_ids"], dtype=np.int64)
            top_lp = np.asarray(record["topk_logprobs"], dtype=np.float32)
            start = last_response(ids.tolist(), marker)
            if start is None or start >= len(ids) - 1:
                continue
            positions = np.arange(start, len(ids) - 1)
            nxt = ids[positions + 1]
            hit = top_ids[positions] == nxt[:, None]
            floor = top_lp[positions].min(axis=1)  # outside the top-k: charged the k-th value
            logp = np.where(hit.any(axis=1), (top_lp[positions] * hit).sum(axis=1), floor)
            corpus, source, domain, text = meta[doc_id]
            # The final answer's tokens: a derivation with one wrong step still reads as
            # fluent on average, but a teacher that would not have written this answer
            # puts little mass on it right here.
            answer_logp = answer_top1 = None
            spans = [m.span(1) for m in re.finditer(r"\\boxed\{([^{}]*)\}", text)] or \
                [m.span(1) for m in re.finditer(r"answer is:?\s*\$?([^\n$]+)", text)]
            if spans:
                encoded = tok(text[:len(text)], add_special_tokens=False, return_offsets_mapping=True)
                if len(encoded["input_ids"]) == len(ids):
                    begin, end = spans[-1]
                    inside = [i for i, (a, b) in enumerate(encoded["offset_mapping"])
                              if a < end and b > begin and 0 < i < len(ids)]
                    if inside:
                        at = np.asarray(inside) - 1  # position i - 1 predicts token i
                        match = top_ids[at] == ids[at + 1][:, None]
                        lp = np.where(match.any(axis=1), (top_lp[at] * match).sum(axis=1), top_lp[at].min(axis=1))
                        answer_logp = float(lp.sum())
                        answer_top1 = bool((top_ids[at, 0] == ids[at + 1]).all())
            reply = text.split("<|im_start|>assistant")[-1]
            blocks = BLOCK.findall(reply)
            broken = 0
            for block in blocks:
                try:
                    ast.parse(block)
                except SyntaxError:
                    broken += 1
            rows.append({"doc_id": doc_id, "corpus": corpus, "source": source, "domain": domain,
                         "teacher_logp": float(logp.mean()), "outside_topk": float(1 - hit.any(axis=1).mean()),
                         "code_blocks": len(blocks), "unparsable_blocks": broken,
                         "answer_logp": answer_logp, "answer_top1": answer_top1})
    with open(args.output, "w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row) + "\n")
    values = np.array([r["teacher_logp"] for r in rows])
    cut = np.quantile(values, 0.05)
    print("%d traces; teacher mean log-p per answer token: median %.3f, 5th percentile %.3f"
          % (len(rows), float(np.median(values)), float(cut)))
    groups = {}
    for r in rows:
        key = "%s / %s" % (r["corpus"], r["source"])
        g = groups.setdefault(key, [0, 0, 0, 0])
        g[0] += 1
        g[1] += r["teacher_logp"] < cut
        g[2] += r["code_blocks"] > 0
        g[3] += r["unparsable_blocks"] > 0
    print("%-52s %6s %9s %11s %11s" % ("corpus / source", "traces", "low-5%", "with code", "unparsable"))
    for key, (n, low, code, bad) in sorted(groups.items(), key=lambda kv: -kv[1][0]):
        if n >= 50:
            print("%-52s %6d %8.1f%% %11d %11d" % (key[:52], n, 100 * low / n, code, bad))
    answered = [r for r in rows if r["answer_logp"] is not None]
    print("\nfinal answers located in %d traces; the teacher's own top choice at every answer "
          "token in %.1f%%" % (len(answered), 100 * np.mean([r["answer_top1"] for r in answered])))
    print("%-52s %6s %12s %14s" % ("corpus / source", "answers", "teacher top1", "p(answer)<0.1"))
    by = {}
    for r in answered:
        by.setdefault("%s / %s" % (r["corpus"], r["source"]), []).append(r)
    for key, group in sorted(by.items(), key=lambda kv: -len(kv[1])):
        if len(group) >= 30:
            print("%-52s %6d %11.1f%% %13.1f%%" % (
                key[:52], len(group), 100 * np.mean([r["answer_top1"] for r in group]),
                100 * np.mean([r["answer_logp"] < np.log(0.1) for r in group])))


if __name__ == "__main__":
    main()

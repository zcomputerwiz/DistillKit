"""How often does the KL term have nothing to say about the right answer?

The cached teacher signal is a top-64 per position, and the loss runs with
`missing_probability_handling: zero`. That combination has a consequence worth measuring
rather than assuming. The divergence is summed only over the teacher's 64 entries::

    KL = sum_{k in top-64} p_t(k) * (log p_t(k) - log p_s(k))

while the student's log-probs are normalised over all 248,320 tokens. So a token outside
the teacher's top-64 earns the student **no credit** for predicting it, and any
probability moved there necessarily lowers `log p_s(k)` for the 64 that count -- it is
penalised. Whatever the n-gram table knows about a token the teacher did not rank, the
KL term is structurally against.

That matters here only in proportion to how often the true next token is missing from the
top-64, and how much teacher mass those 64 entries actually capture. This measures both,
partitioned by role, over the same cache the runs trained on.

    python scratch/topk_coverage_probe.py --report scratch/gpu-checks/topk-coverage.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def load_document(cache, manifest, record):
    """One document's ids and top-k block, read straight out of the shard."""
    top_k = manifest["top_k"]
    length, offset = record["length"], record["offset"]
    shard = record["shard"]
    prefix = "%s-%05d" % (record["split"], shard)
    ids = np.fromfile(cache / (prefix + ".input_ids.bin"),
                      dtype=np.dtype(manifest["token_dtype"]),
                      count=length, offset=offset * np.dtype(manifest["token_dtype"]).itemsize)
    index_dtype = np.dtype(manifest["topk_index_dtype"])
    value_dtype = np.dtype(manifest["topk_value_dtype"])
    topk_ids = np.fromfile(cache / (prefix + ".topk_ids.bin"), dtype=index_dtype,
                           count=length * top_k,
                           offset=offset * top_k * index_dtype.itemsize).reshape(length, top_k)
    topk_values = np.fromfile(cache / (prefix + ".topk_logprobs.bin"), dtype=value_dtype,
                              count=length * top_k,
                              offset=offset * top_k * value_dtype.itemsize).reshape(length, top_k)
    return ids, topk_ids, topk_values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", default="../teacher-cache-1m")
    parser.add_argument("--student", default="../student-hf")
    parser.add_argument("--docs", type=int, default=64)
    parser.add_argument("--report")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    from distillkit.independent_eval import role_spans

    cache = Path(args.cache)
    manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(args.student)
    records = [r for r in manifest["documents"] if r["split"] == "train"][: args.docs]
    print("cache top_k=%d, %d documents" % (manifest["top_k"], len(records)))

    roles = ("system", "user", "assistant", "template")
    hits = {role: 0 for role in roles}
    totals = {role: 0 for role in roles}
    mass = {role: [] for role in roles}
    overall_hits = overall_total = 0
    captured = []

    for record in records:
        ids, topk_ids, topk_values = load_document(cache, manifest, record)
        # Position t's target is token t+1, which is what the cached distribution predicts.
        targets = ids[1:]
        present = (topk_ids[:-1] == targets[:, None]).any(axis=1)
        probs = np.exp(topk_values[:-1].astype(np.float32))
        row_mass = probs.sum(axis=1)
        overall_hits += int(present.sum())
        overall_total += len(present)
        captured.append(row_mass)

        text = tokenizer.decode(ids.tolist())
        encoding = tokenizer(text, return_offsets_mapping=True)
        spans = role_spans(text, encoding["offset_mapping"])
        # Re-tokenising a decode is not guaranteed to reproduce the cached ids, so this
        # partition is indicative; the overall number above does not depend on it.
        for role, ranges in spans.items():
            for start, stop in ranges:
                lo, hi = max(start - 1, 0), min(stop - 1, len(present))
                if hi > lo:
                    hits[role] += int(present[lo:hi].sum())
                    totals[role] += hi - lo
                    mass[role].append(row_mass[lo:hi])

    captured = np.concatenate(captured)
    report = {
        "cache": str(cache),
        "top_k": manifest["top_k"],
        "documents": len(records),
        "positions": overall_total,
        "true_token_in_topk": overall_hits / overall_total,
        "true_token_outside_topk": 1 - overall_hits / overall_total,
        "teacher_mass_in_topk_mean": float(captured.mean()),
        "teacher_mass_in_topk_p05": float(np.quantile(captured, 0.05)),
        "by_role": {},
    }
    print("\ntrue next token inside the teacher's top-%d: %.2f%%  (outside: %.2f%%)"
          % (manifest["top_k"], 100 * report["true_token_in_topk"],
             100 * report["true_token_outside_topk"]))
    print("teacher probability mass inside the top-%d: mean %.4f, 5th percentile %.4f"
          % (manifest["top_k"], report["teacher_mass_in_topk_mean"],
             report["teacher_mass_in_topk_p05"]))
    print("\nby role (indicative -- the partition re-tokenises a decode):")
    for role in roles:
        if not totals[role]:
            continue
        rows = np.concatenate(mass[role])
        report["by_role"][role] = {
            "positions": totals[role],
            "true_token_outside_topk": 1 - hits[role] / totals[role],
            "teacher_mass_in_topk_mean": float(rows.mean()),
        }
        print("  %-9s %7d positions   outside top-k %5.2f%%   mass in top-k %.4f"
              % (role, totals[role], 100 * report["by_role"][role]["true_token_outside_topk"],
                 report["by_role"][role]["teacher_mass_in_topk_mean"]))

    report["quality_evaluation"] = False
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

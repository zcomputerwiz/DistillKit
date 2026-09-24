"""Score several checkpoints on one held-out sample, paired, in one process.

A learning-rate sweep compares arms that share a start, a data order and a seed, so the
comparison is paired: the same documents, scored the same way, differences taken per
document and resampled together. Scoring every checkpoint in one process also means the
autotuner benchmarks each shape once rather than once per arm.

Document prefixes are floored to the routing block, as training floors them. That is
also what keeps the shape set small: 128 documents at their own lengths are 128 cold
autotunes, floored they are a handful.

    python scratch/dense_gr/sweep_eval.py --cache ../teacher-cache-5m \
        --arm start=scratch/dense_gr/checkpoints-2b/warmed-chat32 \
        --arm 7.3e-6=scratch/dense_gr/sweep-lr-7p3e-6/smoke-r1-1-gr-s0-csa2 \
        --reference 7.3e-6 --output scratch/dense_gr/sweep-eval.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from benchmark import apply_liger
from cut_cross_entropy import linear_cross_entropy
from distillkit.models import Qwen35WidenedForCausalLM
from teacher_kl import CachedTeacher


def score(path, documents):
    model, loading = Qwen35WidenedForCausalLM.from_pretrained(
        path, dtype=torch.bfloat16, attn_implementation="flash_attention_2",
        local_files_only=True, output_loading_info=True)
    if loading["missing_keys"] or loading["unexpected_keys"] or loading.get("mismatched_keys"):
        raise ValueError("%s did not reload strictly: %r" % (path, loading))
    model = model.to("cuda").eval()
    apply_liger(model, model.config)
    totals = []
    with torch.inference_mode():
        for ids in documents:
            hidden = model.model(input_ids=ids, attention_mask=torch.ones_like(ids),
                                 use_cache=False).last_hidden_state
            count = ids.numel() - ids.shape[0]
            totals.append(float(linear_cross_entropy(
                hidden, model.lm_head.weight, ids, shift=1, reduction="mean")) * count)
    del model
    torch.cuda.empty_cache()
    return np.array(totals)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", nargs="+", required=True)
    parser.add_argument("--arm", action="append", required=True, metavar="LABEL=PATH")
    parser.add_argument("--reference", required=True, help="label every arm is paired against")
    parser.add_argument("--documents", type=int, default=128)
    parser.add_argument("--block", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    arms = dict(item.split("=", 1) for item in args.arm)
    if args.reference not in arms:
        raise SystemExit("--reference must name one of the arms")

    held = CachedTeacher(args.cache, "eval", device="cuda", max_length=args.max_length)
    sample = held.stratified(args.documents)
    documents, targets, kept = [], [], []
    for _, doc_id in sample:
        ids = held.read(doc_id)["input_ids"]
        width = (ids.shape[1] // args.block) * args.block
        if width < args.block:
            continue
        documents.append(ids[:, :width])
        targets.append(width - 1)
        kept.append(doc_id)
    held.cache.close()
    targets = np.array(targets, dtype=np.float64)
    print("%d documents, %d targets, %d shapes"
          % (len(documents), targets.sum(), len({d.shape[1] for d in documents})), flush=True)

    scores = {}
    for label, path in arms.items():
        scores[label] = score(path, documents)
        print("%-10s nll %.4f" % (label, scores[label].sum() / targets.sum()), flush=True)

    rng = np.random.default_rng(0)
    draws = rng.integers(0, len(documents), size=(args.draws, len(documents)))
    base = scores[args.reference]
    weight = targets[draws].sum(axis=1)
    rows = {}
    for label, values in scores.items():
        estimate = (values.sum() - base.sum()) / targets.sum()
        resampled = (values[draws].sum(axis=1) - base[draws].sum(axis=1)) / weight
        low, high = np.percentile(resampled, [2.5, 97.5])
        rows[label] = dict(nll=values.sum() / targets.sum(), vs_reference=estimate,
                           ci95=[float(low), float(high)])
    print("\n%-10s %8s %12s %26s" % ("arm", "nll", "vs " + args.reference, "95% CI"))
    for label, row in rows.items():
        print("%-10s %8.4f %+12.4f   [%+.4f, %+.4f]"
              % (label, row["nll"], row["vs_reference"], *row["ci95"]))
    args.output.write_text(json.dumps(dict(
        cache=args.cache, reference=args.reference, documents=kept,
        block=args.block, targets=int(targets.sum()), arms=arms, rows=rows,
        per_document={label: values.tolist() for label, values in scores.items()}),
        indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

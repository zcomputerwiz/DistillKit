"""Score routers on held-out documents by their own objective, paired across checkpoints.

Held-out NLL at a 1024-token cap is a blunt instrument for the router: it is one of many
things feeding the loss, and a small change in which blocks it picks is buried. The
router's own objective -- the KL of its scores against the attention over the positions
it selected, what training minimizes -- measures it directly. This computes that on the
same held-out documents `sweep_eval.py` uses, the same way the training step does, with
no backward pass.

    python scratch/dense_gr/router_eval.py --cache ../teacher-cache-5m \
        --arm x1=... --arm x64=... --reference x1 --output router-eval.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from benchmark import apply_liger
from distillkit.models import Qwen35WidenedForCausalLM
from distillkit.models.qwen35.csa2 import isolated_indexer, recorded_attention
from indexer_kl import indexer_loss, routing_layers, watch
from teacher_kl import CachedTeacher, scored_mask


def score(path, documents):
    model = Qwen35WidenedForCausalLM.from_pretrained(
        path, dtype=torch.bfloat16, attn_implementation="flash_attention_2",
        local_files_only=True).to("cuda").eval()
    apply_liger(model, model.config)
    layers = routing_layers(model)
    totals = []
    with torch.no_grad():
        for ids in documents:
            mask = scored_mask(ids.shape[1], ids.device, ids.shape[0])
            seen, handles = watch(model, layers)
            try:
                with recorded_attention(model), isolated_indexer(model):
                    model.model(input_ids=ids, attention_mask=torch.ones_like(ids),
                                use_cache=False)
                    for handle in handles:
                        handle.remove()
                    targets = {i: a.last_attention for i, a in layers}
                    chosen = {i: a.last_allowed for i, a in layers}
                    borrowed = {i: a.bus.require_latent(a.latent_donor, i) for i, a in layers}
                    aligned = indexer_loss(model, layers, seen, targets, borrowed,
                                           selected=chosen, query_mask=mask)
            finally:
                for handle in handles:
                    handle.remove()
            # A mean over this document's scored queries, weighted back to a sum so the
            # corpus figure is per query rather than per document.
            totals.append(float(aligned) * int(mask.sum()))
    del model
    torch.cuda.empty_cache()
    return np.array(totals)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", nargs="+", required=True)
    parser.add_argument("--arm", action="append", required=True, metavar="LABEL=PATH")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--documents", type=int, default=128)
    parser.add_argument("--block", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    arms = dict(item.split("=", 1) for item in args.arm)

    held = CachedTeacher(args.cache, "eval", device="cuda", max_length=args.max_length)
    documents, queries = [], []
    for _, doc_id in held.stratified(args.documents):
        ids = held.read(doc_id)["input_ids"]
        width = (ids.shape[1] // args.block) * args.block
        if width >= args.block:
            documents.append(ids[:, :width])
            queries.append(width - 1)
    held.cache.close()
    queries = np.array(queries, dtype=np.float64)

    scores = {label: score(path, documents) for label, path in arms.items()}
    rng = np.random.default_rng(0)
    draws = rng.integers(0, len(documents), size=(10000, len(documents)))
    base = scores[args.reference]
    rows = {}
    print("%-14s %10s %12s %24s" % ("arm", "router KL", "vs " + args.reference, "95% CI"))
    for label, values in scores.items():
        resampled = (values[draws].sum(1) - base[draws].sum(1)) / queries[draws].sum(1)
        low, high = np.percentile(resampled, [2.5, 97.5])
        rows[label] = dict(kl=values.sum() / queries.sum(),
                           vs_reference=(values.sum() - base.sum()) / queries.sum(),
                           ci95=[float(low), float(high)])
        print("%-14s %10.4f %+12.4f   [%+.4f, %+.4f]"
              % (label, rows[label]["kl"], rows[label]["vs_reference"], low, high))
    args.output.write_text(json.dumps(dict(arms=arms, reference=args.reference, rows=rows,
                                           per_document={k: v.tolist() for k, v in scores.items()}),
                                      indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

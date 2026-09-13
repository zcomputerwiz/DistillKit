"""Is familiarity the signal, or a proxy for something visible in (h, r)?

Attenuating layer 12's FFN update on familiar contexts improves held-out content NLL. That
does not establish that *familiarity* is what matters. The trigram count may simply be
picking out states whose proposed update already looks a particular way -- large relative
to the residual, or pointing somewhere the residual stream did not need -- in which case
the eventual gate should read the geometry directly and forget the context entirely.

For each scored position this records what a gate could plausibly see:

    familiarity   cross-document trigram count, stored residual variance
    geometry      ||h||, ||r||, ||r||/||h||, cos(h, r)

and what actually happened:

    benefit       log p_attenuated(target) - log p_stock(target)

Then it asks which family predicts the benefit. Two small logistic probes, fit on half the
positions and scored on the other half, answer it in the only way that matters here: not
which correlates, but which would let a gate decide. The probes are diagnostic -- nothing
is trained into the model, and final predictive behaviour remains authoritative.

    CUDA_VISIBLE_DEVICES=0 python scratch/ffn_attenuation/diagnostics.py --output ...
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ffn_memo"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import torch.nn.functional as F

from attenuate import Familiarity, load_records
from distillkit.ffn_skip import attenuate_ffn, capture_ffn
from repeatability import DEFAULT_BUNDLE, DEFAULT_MODEL, trigram_keys

LAYER = 12


def probe(features, labels, seed=0):
    """Logistic regression by plain gradient descent; returns held-out AUC."""
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(labels), generator=generator)
    split = len(labels) // 2
    train, test = order[:split], order[split:]

    x = torch.tensor(features, dtype=torch.float32)
    x = (x - x[train].mean(0)) / (x[train].std(0) + 1e-6)
    y = torch.tensor(labels, dtype=torch.float32)

    weights = torch.zeros(x.shape[1], requires_grad=True)
    bias = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.Adam([weights, bias], lr=0.05)
    for _ in range(400):
        optimizer.zero_grad()
        logits = x[train] @ weights + bias
        F.binary_cross_entropy_with_logits(logits, y[train]).backward()
        optimizer.step()

    with torch.no_grad():
        scores = (x[test] @ weights + bias).numpy()
    truth = y[test].numpy()
    positive, negative = scores[truth > 0.5], scores[truth <= 0.5]
    if not len(positive) or not len(negative):
        return float("nan")
    # Mann-Whitney U, which is the AUC without a dependency.
    ranks = np.argsort(np.argsort(np.concatenate([positive, negative]))) + 1
    return float((ranks[:len(positive)].sum() - len(positive) * (len(positive) + 1) / 2)
                 / (len(positive) * len(negative)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--bundle", default=DEFAULT_BUNDLE)
    parser.add_argument("--split", default="screen")
    parser.add_argument("--cache", type=Path,
                        default=Path("scratch/ffn_memo/cache/layer-12.npz"))
    parser.add_argument("--documents", type=int, default=120)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if torch.cuda.device_count() != 1:
        raise SystemExit("mask to one card with CUDA_VISIBLE_DEVICES")

    from transformers import AutoConfig, AutoTokenizer
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    from distillkit.independent_eval import build_token_classes

    started = time.monotonic()
    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    config = getattr(config, "text_config", config)
    config.use_cache = False
    model = Qwen3_5ForCausalLM.from_pretrained(
        args.model, config=config, dtype=torch.bfloat16,
        local_files_only=True).to(args.device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    classes = build_token_classes(tokenizer, config.vocab_size)
    familiarity = Familiarity(args.cache)
    records = load_records(args.bundle, args.split, args.documents)

    rows = []
    with torch.inference_mode():
        for record in records:
            ids = record["ids"]
            keys = trigram_keys(ids, config.vocab_size)
            tokens = torch.tensor([ids], device=args.device)
            targets = torch.tensor(ids[1:], device=args.device)

            # Stock pass, capturing the layer's input and proposed update.
            with capture_ffn(model, [LAYER]) as captured:
                stock = model(input_ids=tokens,
                              attention_mask=torch.ones_like(tokens)).logits[0, :-1].float()
            hidden, update = captured[LAYER][0]
            hidden, update = hidden[0].float(), update[0].float()
            stock_nll = F.cross_entropy(stock, targets, reduction="none")

            # Attenuated pass over the familiar positions.
            selected = familiarity.select(keys[2:], 4, 0.25)
            if not selected:
                continue
            mask = torch.zeros(1, len(ids), dtype=torch.bool)
            mask[0, [position + 2 for position in selected]] = True
            with attenuate_ffn(model, [LAYER]) as handle:
                handle.set(LAYER, mask.to(args.device), args.alpha)
                changed = model(input_ids=tokens,
                                attention_mask=torch.ones_like(tokens)).logits[0, :-1].float()
            changed_nll = F.cross_entropy(changed, targets, reduction="none")

            norm_h = hidden.norm(dim=-1)
            norm_r = update.norm(dim=-1)
            cosine = F.cosine_similarity(hidden, update, dim=-1)
            for position in selected:
                index = position + 2          # position in the token sequence
                if index >= len(ids) - 1:
                    continue
                key = keys[index]
                rows.append({
                    "count": float(familiarity.counts.get(key, 0)),
                    "variance": float(familiarity.variance.get(key, 1.0)),
                    "norm_h": float(norm_h[index]),
                    "norm_r": float(norm_r[index]),
                    "ratio": float(norm_r[index] / (norm_h[index] + 1e-6)),
                    "cosine": float(cosine[index]),
                    # The target predicted *from* this position is at index, i.e. ids
                    # [index + 1]; its NLL sits at index in the shifted arrays.
                    "benefit": float(stock_nll[index] - changed_nll[index]),
                    "class": classes[ids[index + 1]],
                })

    report = {"model": args.model, "layer": LAYER, "alpha": args.alpha,
              "documents": len(records), "positions": len(rows)}
    benefit = np.array([row["benefit"] for row in rows])
    helped = (benefit > 0).astype(np.float32)
    report["helped_fraction"] = float(helped.mean())
    report["mean_benefit"] = float(benefit.mean())

    def summarise(name, values, buckets):
        out = []
        edges = np.quantile(values, buckets)
        for low, high in zip(edges[:-1], edges[1:]):
            inside = (values >= low) & (values <= high)
            if inside.sum() < 20:
                continue
            out.append({"from": float(low), "to": float(high),
                        "positions": int(inside.sum()),
                        "mean_benefit": float(benefit[inside].mean()),
                        "helped": float(helped[inside].mean())})
        report.setdefault("buckets", {})[name] = out

    quantiles = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    for name in ("count", "variance", "norm_h", "norm_r", "ratio", "cosine"):
        summarise(name, np.array([row[name] for row in rows]), quantiles)

    familiar_features = np.array([[row["count"], row["variance"]] for row in rows])
    geometry_features = np.array([[row["norm_h"], row["norm_r"], row["ratio"],
                                   row["cosine"]] for row in rows])
    both = np.concatenate([familiar_features, geometry_features], axis=1)
    report["auc"] = {
        "familiarity": probe(familiar_features, helped),
        "geometry": probe(geometry_features, helped),
        "both": probe(both, helped),
    }
    report["by_class"] = {}
    for label in sorted({row["class"] for row in rows}):
        inside = np.array([row["class"] == label for row in rows])
        report["by_class"][label] = {"positions": int(inside.sum()),
                                     "mean_benefit": float(benefit[inside].mean()),
                                     "helped": float(helped[inside].mean())}

    report["elapsed_seconds"] = time.monotonic() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"positions": report["positions"],
                      "helped_fraction": report["helped_fraction"],
                      "auc": report["auc"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

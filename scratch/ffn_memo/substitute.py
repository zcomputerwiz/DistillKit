"""Replace real FFN calls with cached residuals on held-out text, and price the damage.

The cache was built on documents this evaluator never sees. Here every held-out position
whose trigram is in the cache, and whose entry passes the confidence threshold, has its
MLP output replaced by the stored prototype. The substituted state propagates: a
replacement at position t is visible to every later token, which is the cost a deployed
cache would really pay.

Four arms, because coverage on its own proves nothing:

    exact     the prototype for this position's own context
    wrong     a prototype from a different key -- same distribution, wrong context
    global    the layer's single global mean residual
    zero      the previous study's failed intervention, for scale

If `exact` is no better than `wrong` or `global`, the cache is storing the average size of
a residual rather than anything about this context, and there is nothing to build.

    CUDA_VISIBLE_DEVICES=0 python scratch/ffn_memo/substitute.py --jobs 0-5 --output ...
"""

from __future__ import annotations

import argparse
import io
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import torch.nn.functional as F

from distillkit.ffn_skip import estimate_savings, model_flops_per_token, substitute_ffn
from repeatability import DEFAULT_BUNDLE, DEFAULT_MODEL, trigram_keys

ARMS = ("exact", "wrong", "global", "zero")
# (layers, minimum observations, maximum residual variance as a fraction of mean energy)
THRESHOLDS = [(2, None), (4, None), (8, None), (16, None), (4, 0.5), (4, 0.25)]
VERIFY = [(2, None), (4, None), (16, None), (4, 0.25)]
LAYER_SETS = [(12,), (16,), (8,), (20,), (12, 16), (8, 12, 16, 20)]


class LayerCache:
    def __init__(self, path):
        data = np.load(path)
        self.keys = data["keys"]
        self.counts = data["counts"]
        self.mean = torch.from_numpy(data["mean"].astype(np.float32))
        self.variance = data["variance"]
        self.global_mean = torch.from_numpy(data["global_mean"])
        energy = (self.mean.to(torch.float64) ** 2).sum(-1).numpy()
        # Relative spread: how much of a typical occurrence the prototype fails to
        # explain. A key whose occurrences disagree wildly is not worth substituting.
        self.relative_variance = self.variance / np.maximum(energy + self.variance, 1e-9)
        self.index = {int(key): position for position, key in enumerate(self.keys)}

    def lookup(self, keys, min_count, max_variance):
        rows, positions = [], []
        for position, key in enumerate(keys):
            row = self.index.get(key)
            if row is None or self.counts[row] < min_count:
                continue
            if max_variance is not None and self.relative_variance[row] > max_variance:
                continue
            rows.append(row)
            positions.append(position)
        return positions, rows


def load_documents(bundle_path, limit):
    with io.open(bundle_path, encoding="utf-8") as handle:
        bundle = json.load(handle)
    records = bundle["splits"]["screen"]["nll"]
    return records[:limit] if limit else records


@torch.inference_mode()
def run(model, record, device, caches, layer_set, min_count, max_variance, arm,
        vocab, rng):
    ids = record["ids"]
    keys = trigram_keys(ids, vocab)
    tokens = torch.tensor([ids], device=device)
    replaced = 0
    with substitute_ffn(model, layer_set) as handle:
        for layer in layer_set:
            cache = caches[layer]
            positions, rows = cache.lookup(keys[2:], min_count, max_variance)
            positions = [position + 2 for position in positions]
            mask = torch.zeros(1, len(ids), dtype=torch.bool)
            replacement = torch.zeros(1, len(ids), cache.mean.shape[1])
            if positions:
                mask[0, positions] = True
                if arm == "exact":
                    vectors = cache.mean[rows]
                elif arm == "wrong":
                    # Same cache, deliberately mismatched entries: isolates whether the
                    # benefit is the context or merely a residual of the right size.
                    shuffled = rng.permutation(len(rows))
                    vectors = cache.mean[[rows[i] for i in shuffled]]
                elif arm == "global":
                    vectors = cache.global_mean.unsqueeze(0).expand(len(rows), -1)
                else:
                    vectors = torch.zeros(len(rows), cache.mean.shape[1])
                replacement[0, positions] = vectors
            handle.set(layer, mask.to(device), replacement.to(device))
            replaced += int(mask.sum())
        logits = model(input_ids=tokens,
                       attention_mask=torch.ones_like(tokens)).logits[0, :-1].float()
    targets = torch.tensor(ids[1:], device=device)
    return (F.cross_entropy(logits, targets, reduction="none").cpu(),
            logits.argmax(-1).cpu(), replaced)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--bundle", default=DEFAULT_BUNDLE)
    parser.add_argument("--cache", type=Path, default=Path("scratch/ffn_memo/cache"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--jobs", default="")
    parser.add_argument("--verify", action="store_true")
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
    flops = model_flops_per_token(model, config)
    manifest = json.loads((args.cache / "manifest.json").read_text(encoding="utf-8"))
    caches = {layer: LayerCache(args.cache / ("layer-%d.npz" % layer))
              for layer in manifest["layers"]}

    records = load_documents(args.bundle, args.limit)
    jobs = [(layers, threshold, arm)
            for layers in LAYER_SETS for threshold in THRESHOLDS for arm in ARMS
            if arm == "exact" or (threshold == (4, None) and len(layers) == 1)]
    if args.verify:
        # A surprising number deserves its own run: zeroing one mid-stack FFN on
        # familiar contexts appeared to *improve* held-out content NLL, which is worth
        # reproducing across coverage levels before it is reported as a finding.
        jobs = [((12,), threshold, arm) for threshold in VERIFY
                for arm in ("zero", "exact")]
    if args.jobs:
        start, stop = (int(part) for part in args.jobs.split("-"))
        jobs = jobs[start:stop + 1]

    baseline = []
    for record in records:
        tokens = torch.tensor([record["ids"]], device=args.device)
        with torch.inference_mode():
            logits = model(input_ids=tokens,
                           attention_mask=torch.ones_like(tokens)).logits[0, :-1].float()
        targets = torch.tensor(record["ids"][1:], device=args.device)
        baseline.append((F.cross_entropy(logits, targets, reduction="none").cpu(),
                         logits.argmax(-1).cpu(), record["ids"][1:]))

    report = {"model": args.model, "cache": str(args.cache), "cache_manifest": manifest,
              "documents": len(records), "flops_per_token": flops, "results": []}
    rng = np.random.default_rng(0)
    for layers, (min_count, max_variance), arm in jobs:
        per_document = {}
        totals = {}
        replaced_total = 0
        tokens_total = 0
        agree = 0
        for record, (base_nll, base_argmax, targets) in zip(records, baseline):
            nll, argmax, replaced = run(model, record, args.device, caches, layers,
                                        min_count, max_variance, arm,
                                        config.vocab_size, rng)
            replaced_total += replaced
            tokens_total += len(targets)
            agree += int((argmax == base_argmax).sum())
            sums = {}
            for index, token in enumerate(targets):
                label = classes[token]
                bucket = totals.setdefault(label, [0.0, 0.0, 0])
                bucket[0] += float(nll[index])
                bucket[1] += float(base_nll[index])
                bucket[2] += 1
                entry = sums.setdefault(label, [0.0, 0])
                entry[0] += float(nll[index] - base_nll[index])
                entry[1] += 1
            for label, (total, count) in sums.items():
                per_document.setdefault(label, []).append(total / count)

        delta = {}
        for label, values in per_document.items():
            mean = statistics.fmean(values)
            error = statistics.stdev(values) / len(values) ** 0.5 if len(values) > 1 else 0.0
            delta[label] = {"mean": mean, "stderr": error,
                            "t": mean / error if error else 0.0,
                            "better": sum(1 for value in values if value < 0),
                            "n": len(values)}
        entry = {
            "layers": list(layers), "min_count": min_count,
            "max_variance": max_variance, "arm": arm,
            "delta": delta,
            "by_class": {label: {"nll": bucket[0] / bucket[2],
                                 "baseline_nll": bucket[1] / bucket[2],
                                 "tokens": bucket[2]}
                         for label, bucket in sorted(totals.items())},
            "coverage": replaced_total / (tokens_total * len(layers)),
            "top1_agreement": agree / tokens_total,
            "savings": estimate_savings(flops, replaced_total, tokens_total),
        }
        report["results"].append(entry)
        print("%-16s count>=%-3s var<=%-5s %-6s coverage %.3f  content %+.6f  top1 %.4f"
              % (layers, min_count, max_variance, arm, entry["coverage"],
                 delta["content"]["mean"], entry["top1_agreement"]), flush=True)

    report["elapsed_seconds"] = time.monotonic() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("wrote %s in %.0f s" % (args.output, report["elapsed_seconds"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

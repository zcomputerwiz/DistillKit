"""Does admitting less of layer 12's FFN update really help on familiar contexts?

The memoisation study left an anomaly: zeroing the layer-12 FFN residual on high-frequency
trigram positions improved held-out content NLL, and improved it more as the contexts got
more familiar. That is either a real statement about how much of a sublayer's proposal
belongs in the residual stream, or an artefact of removing an update entirely.

An alpha curve separates the two. The decoder's ordinary update admits every proposal at
unit strength::

    h = h + mlp(norm(h))          alpha = 1, the stock model
    h = h + alpha * mlp(norm(h))  at selected positions only

If over-admission is real, quality should improve smoothly as alpha falls below one and
degrade above it. If only alpha = 0 behaves strangely, something else is going on.

Familiarity comes from the cache built for the memoisation study -- cross-document trigram
counts and residual variance, computed on documents this evaluator never sees. The control
arms matter as much as the treatment: a random mask of the same density, a mask over rare
contexts, and the familiar mask with its positions shuffled within each document all
answer "is it familiarity, or just the density of the intervention?"

    CUDA_VISIBLE_DEVICES=0 python scratch/ffn_attenuation/attenuate.py --jobs 0-24 ...
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
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ffn_memo"))

import numpy as np
import torch
import torch.nn.functional as F

from distillkit.ffn_skip import attenuate_ffn
from repeatability import DEFAULT_BUNDLE, DEFAULT_MODEL, trigram_keys

ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25)
# (name, minimum cross-document count, maximum relative residual variance)
MASKS = [("count2", 2, None), ("count4", 4, None), ("count8", 8, None),
         ("count16", 16, None), ("count32", 32, None), ("count4_var25", 4, 0.25)]
PRIMARY_MASK = "count4_var25"
CONTROLS = ("random", "rare", "shuffled")
NEIGHBOURS = (8, 10, 14, 16)


class Familiarity:
    """Cross-document trigram statistics, read from the memoisation cache."""

    def __init__(self, path):
        data = np.load(path)
        self.counts = {int(key): int(count)
                       for key, count in zip(data["keys"], data["counts"])}
        mean = data["mean"].astype(np.float32)
        energy = (mean.astype(np.float64) ** 2).sum(-1)
        relative = data["variance"] / np.maximum(energy + data["variance"], 1e-9)
        self.variance = {int(key): float(value)
                         for key, value in zip(data["keys"], relative)}

    def select(self, keys, min_count, max_variance):
        chosen = []
        for position, key in enumerate(keys):
            count = self.counts.get(key)
            if count is None or count < min_count:
                continue
            if max_variance is not None and self.variance[key] > max_variance:
                continue
            chosen.append(position)
        return chosen

    def rare(self, keys):
        """Contexts the cache never saw recur: the opposite population."""
        return [position for position, key in enumerate(keys)
                if key not in self.counts]


def build_mask(kind, keys, familiarity, min_count, max_variance, length, rng,
               classes_by_position=None):
    """Positions to attenuate, as offsets into the token sequence."""
    familiar = familiarity.select(keys, min_count, max_variance)
    if kind == "familiar":
        chosen = familiar
    elif kind == "random":
        # Matched density, no relationship to context: if this helps as much, the
        # familiarity hypothesis is dead.
        pool = np.arange(len(keys))
        chosen = sorted(rng.choice(pool, size=min(len(familiar), len(pool)),
                                   replace=False).tolist()) if len(familiar) else []
    elif kind == "rare":
        pool = familiarity.rare(keys)
        chosen = sorted(rng.choice(pool, size=min(len(familiar), len(pool)),
                                   replace=False).tolist()) if pool and familiar else []
    elif kind == "shuffled":
        # Density *and* token-class composition held fixed, context detached: within each
        # class, the selected positions are re-drawn from all positions of that class. A
        # plain random mask would also change what kinds of token are being attenuated,
        # which is a second difference and would confound the first.
        chosen = []
        familiar_set = set(familiar)
        for label, positions in classes_by_position.items():
            wanted = sum(1 for position in positions if position in familiar_set)
            if not wanted:
                continue
            chosen.extend(rng.choice(positions, size=min(wanted, len(positions)),
                                     replace=False).tolist())
        chosen = sorted(chosen)
    else:
        raise ValueError("unknown mask kind %s" % kind)
    mask = torch.zeros(1, length, dtype=torch.bool)
    if chosen:
        mask[0, [position + 2 for position in chosen]] = True
    return mask


def load_records(bundle_path, split, limit):
    with io.open(bundle_path, encoding="utf-8") as handle:
        bundle = json.load(handle)
    records = bundle["splits"][split]["nll"]
    return records[:limit] if limit else records


def jobs_for(masks=MASKS, alphas=ALPHAS):
    jobs = [{"layer": 12, "mask": name, "min_count": count, "max_variance": variance,
             "kind": "familiar", "alpha": alpha}
            for name, count, variance in masks for alpha in alphas]
    primary = [entry for entry in MASKS if entry[0] == PRIMARY_MASK][0]
    jobs += [{"layer": 12, "mask": primary[0], "min_count": primary[1],
              "max_variance": primary[2], "kind": kind, "alpha": alpha}
             for kind in CONTROLS for alpha in (0.0, 0.5)]
    jobs += [{"layer": layer, "mask": primary[0], "min_count": primary[1],
              "max_variance": primary[2], "kind": "familiar", "alpha": alpha}
             for layer in NEIGHBOURS for alpha in (0.0, 0.5)]
    return jobs


@torch.inference_mode()
def evaluate(model, records, device, familiarity, job, classes, vocab, seed=0):
    rng = np.random.default_rng(seed)
    per_document = {}
    totals = {}
    selected = 0
    tokens_total = 0
    agree = 0
    baseline_cache = job["_baseline"]
    for record, (base_nll, base_argmax, targets) in zip(records, baseline_cache):
        ids = record["ids"]
        keys = trigram_keys(ids, vocab)[2:]
        by_class = {}
        for position, token in enumerate(ids[2:]):
            by_class.setdefault(classes[token], []).append(position)
        mask = build_mask(job["kind"], keys, familiarity, job["min_count"],
                          job["max_variance"], len(ids), rng,
                          classes_by_position=by_class)
        tokens = torch.tensor([ids], device=device)
        with attenuate_ffn(model, [job["layer"]]) as handle:
            handle.set(job["layer"], mask.to(device), job["alpha"])
            logits = model(input_ids=tokens,
                           attention_mask=torch.ones_like(tokens)).logits[0, :-1].float()
        nll = F.cross_entropy(logits, torch.tensor(targets, device=device),
                              reduction="none").cpu()
        argmax = logits.argmax(-1).cpu()
        selected += int(mask.sum())
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
                        "ci95": [mean - 1.96 * error, mean + 1.96 * error],
                        "better": sum(1 for value in values if value < 0),
                        "n": len(values)}
    return {"delta": delta,
            "by_class": {label: {"nll": bucket[0] / bucket[2],
                                 "baseline_nll": bucket[1] / bucket[2],
                                 "tokens": bucket[2],
                                 "nats": bucket[0] - bucket[1]}
                         for label, bucket in sorted(totals.items())},
            "selected_fraction": selected / tokens_total,
            "top1_agreement": agree / tokens_total,
            "tokens": tokens_total}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--bundle", default=DEFAULT_BUNDLE)
    parser.add_argument("--split", default="screen")
    parser.add_argument("--cache", type=Path,
                        default=Path("scratch/ffn_memo/cache/layer-12.npz"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--jobs", default="")
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
    # Newline is the class that dominated every previous aggregate, so it is split out.
    for index in range(config.vocab_size):
        if classes[index] == "layout":
            # An actual newline character, not the two-character escape: the first
            # version of this test looked for a literal backslash-n and quietly put
            # every newline token into the whitespace bucket.
            classes[index] = ("newline" if "\n" in tokenizer.decode([index])
                              else "whitespace")
    familiarity = Familiarity(args.cache)

    records = load_records(args.bundle, args.split, args.limit)
    baseline = []
    for record in records:
        tokens = torch.tensor([record["ids"]], device=args.device)
        with torch.inference_mode():
            logits = model(input_ids=tokens,
                           attention_mask=torch.ones_like(tokens)).logits[0, :-1].float()
        targets = record["ids"][1:]
        baseline.append((F.cross_entropy(logits, torch.tensor(targets, device=args.device),
                                         reduction="none").cpu(),
                         logits.argmax(-1).cpu(), targets))

    jobs = jobs_for()
    if args.jobs:
        start, stop = (int(part) for part in args.jobs.split("-"))
        jobs = jobs[start:stop + 1]

    report = {"model": args.model, "bundle": args.bundle, "split": args.split,
              "documents": len(records), "cache": str(args.cache), "results": []}
    for job in jobs:
        job["_baseline"] = baseline
        entry = evaluate(model, records, args.device, familiarity, job, classes,
                         config.vocab_size)
        job.pop("_baseline")
        entry.update(job)
        report["results"].append(entry)
        print("layer %-3d %-13s %-9s alpha %.2f  selected %.3f  content %+.6f (t %+6.2f)"
              % (job["layer"], job["mask"], job["kind"], job["alpha"],
                 entry["selected_fraction"], entry["delta"]["content"]["mean"],
                 entry["delta"]["content"]["t"]), flush=True)

    report["elapsed_seconds"] = time.monotonic() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("wrote %s in %.0f s" % (args.output, report["elapsed_seconds"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""How much of the network's compute do easy tokens actually need?

An oracle study, not a router. The target token's class is information no deployed model
has at the moment it would have to decide -- using it here answers the only question worth
answering first: if some mechanism could identify these positions perfectly, how much
compute would there be to save, and what would it cost?

Each schedule bypasses the MLP residual in a contiguous band of layers for the positions
its oracle selects, leaving attention and the GatedDeltaNet state update running
everywhere. The altered state flows onward: a skip at position t is visible to every
later token, which is the cost the router would really be paying.

One process holds one model on one card and evaluates every schedule assigned to it, so
the 2B checkpoint is loaded once rather than once per job.

    CUDA_VISIBLE_DEVICES=0 python scratch/depth_oracle/oracle.py \
        --schedules 0-11 --output scratch/depth_oracle/worker0.json
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn.functional as F

from distillkit.ffn_skip import estimate_savings, model_flops_per_token, skip_ffn

DEFAULT_MODEL = "D:/DeepThought/Projects/HybridModel/student-2b-hf"
DEFAULT_BUNDLE = "scratch/independent-eval/full-bundle-384.json"

# (first skipped layer, last skipped layer inclusive). Early layers always run, the band
# is bypassed for eligible tokens, later layers always run. Chosen to trace a curve from
# two layers to twelve rather than to enumerate every interval.
BANDS = [
    (8, 9), (8, 11), (8, 15), (8, 19),
    (12, 15), (12, 19),
    (4, 15), (4, 19),
    (16, 19), (16, 21),
]

# The oracle. "all" is the damage reference; "content" is the negative control, skipping
# exactly the tokens the experiment is trying to protect.
ORACLES = ["layout", "layout+punctuation", "content", "all"]


def load_bundle(path, limit=0):
    with io.open(path, encoding="utf-8") as handle:
        bundle = json.load(handle)
    records = bundle["splits"]["screen"]["nll"]
    return (records[:limit] if limit else records), bundle


def classify(tokenizer, vocab_size):
    """Four evaluator classes, with newline separated from other whitespace."""
    from distillkit.independent_eval import build_token_classes

    classes = build_token_classes(tokenizer, vocab_size)
    newline = tokenizer.convert_tokens_to_ids("\u010a")     # the byte-level newline token
    fine = list(classes)
    for index in range(vocab_size):
        if classes[index] == "layout":
            text = tokenizer.decode([index])
            fine[index] = "newline" if "\n" in text else "whitespace"
    if newline is not None and 0 <= newline < vocab_size:
        fine[newline] = "newline"
    return classes, fine


def eligible(targets, classes, oracle):
    if oracle == "all":
        return torch.ones(len(targets), dtype=torch.bool)
    wanted = {"layout": {"layout"}, "layout+punctuation": {"layout", "punctuation"},
              "content": {"content"}}[oracle]
    return torch.tensor([classes[token] in wanted for token in targets], dtype=torch.bool)


@torch.inference_mode()
def score(model, ids, device, handle=None, mask=None):
    """Teacher-forced next-token NLL for one document, plus its argmax predictions."""
    tokens = torch.tensor([ids], device=device)
    if handle is not None:
        handle.mask = mask.unsqueeze(0).to(device) if mask is not None else None
    logits = model(input_ids=tokens,
                   attention_mask=torch.ones_like(tokens)).logits[0, :-1].float()
    targets = torch.tensor(ids[1:], device=device)
    per_token = F.cross_entropy(logits, targets, reduction="none")
    return per_token.cpu(), logits.argmax(-1).cpu()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--bundle", default=DEFAULT_BUNDLE)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--schedules", default="",
                        help="index range into the job list, e.g. 0-11")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if torch.cuda.device_count() != 1:
        raise SystemExit("one card per worker: mask with CUDA_VISIBLE_DEVICES")

    from transformers import AutoConfig, AutoTokenizer
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    config = getattr(config, "text_config", config)
    config.use_cache = False
    model = Qwen3_5ForCausalLM.from_pretrained(
        args.model, config=config, dtype=torch.bfloat16,
        local_files_only=True).to(args.device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    coarse, fine = classify(tokenizer, config.vocab_size)
    flops = model_flops_per_token(model, config)

    records, bundle = load_bundle(args.bundle, args.limit)
    jobs = [(band, oracle) for band in BANDS for oracle in ORACLES]
    if args.schedules:
        start, stop = (int(part) for part in args.schedules.split("-"))
        jobs = jobs[start:stop + 1]

    started = time.monotonic()
    report = {"model": args.model, "bundle": args.bundle,
              "device_name": torch.cuda.get_device_name(0),
              "documents": len(records), "flops_per_token": flops,
              "baseline": {}, "schedules": []}

    # Baseline once: per-token NLL and argmax for every document, kept compact.
    baseline = []
    for record in records:
        per_token, argmax = score(model, record["ids"], args.device)
        baseline.append({"id": record["id"], "nll": per_token, "argmax": argmax,
                         "targets": record["ids"][1:]})
    report["baseline"] = summarise(baseline, baseline, coarse, fine, flops, 0)
    print("baseline: %s" % json.dumps(report["baseline"]["by_class"]), flush=True)

    for band, oracle in jobs:
        layers = list(range(band[0], band[1] + 1))
        skipped_calls = 0
        results = []
        with skip_ffn(model, layers) as handle:
            for record, reference in zip(records, baseline):
                mask = torch.zeros(len(record["ids"]), dtype=torch.bool)
                # The mask is over input positions; position i produces target i+1, so a
                # decision about target t is applied while computing position t-1.
                mask[:-1] = eligible(reference["targets"], coarse, oracle)
                per_token, argmax = score(model, record["ids"], args.device, handle, mask)
                results.append({"id": record["id"], "nll": per_token, "argmax": argmax,
                                "targets": reference["targets"]})
            skipped_calls = handle.skipped_calls
        entry = summarise(results, baseline, coarse, fine, flops, skipped_calls)
        entry.update({"band": list(band), "layers": layers, "oracle": oracle})
        report["schedules"].append(entry)
        print("%s %s: content %+.6f, top1 %.4f, ffn %.1f%%" % (
            band, oracle, entry["delta"]["content"]["mean"],
            entry["top1_agreement"], 100 * entry["savings"]["ffn_flops_fraction"]),
            flush=True)

    report["elapsed_seconds"] = time.monotonic() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("wrote %s in %.0f s" % (args.output, report["elapsed_seconds"]))
    return 0


def summarise(results, baseline, coarse, fine, flops, skipped_calls):
    """Per-class NLL, paired per-document deltas, top-1 agreement, savings."""
    import statistics

    classes = {}
    deltas = {}
    tokens_total = 0
    agree = 0
    per_document = {}
    for result, reference in zip(results, baseline):
        tokens_total += len(result["targets"])
        agree += int((result["argmax"] == reference["argmax"]).sum())
        for label_set, table in (("", coarse), ("fine:", fine)):
            for index, token in enumerate(result["targets"]):
                key = label_set + table[token]
                bucket = classes.setdefault(key, {"sum": 0.0, "base": 0.0, "tokens": 0})
                bucket["sum"] += float(result["nll"][index])
                bucket["base"] += float(reference["nll"][index])
                bucket["tokens"] += 1
        # Paired per-document means, per class, for the statistics.
        for label_set, table in (("", coarse), ("fine:", fine)):
            sums = {}
            for index, token in enumerate(result["targets"]):
                key = label_set + table[token]
                entry = sums.setdefault(key, [0.0, 0])
                entry[0] += float(result["nll"][index] - reference["nll"][index])
                entry[1] += 1
            for key, (total, count) in sums.items():
                per_document.setdefault(key, []).append(total / count)

    for key, values in per_document.items():
        mean = statistics.fmean(values)
        error = statistics.stdev(values) / len(values) ** 0.5 if len(values) > 1 else 0.0
        deltas[key] = {"mean": mean, "stderr": error,
                       "t": mean / error if error else 0.0,
                       "better": sum(1 for value in values if value < 0),
                       "n": len(values)}

    aggregate = sum(bucket["sum"] for key, bucket in classes.items()
                    if not key.startswith("fine:"))
    aggregate_base = sum(bucket["base"] for key, bucket in classes.items()
                         if not key.startswith("fine:"))
    return {
        "tokens": tokens_total,
        "by_class": {key: {"nll": bucket["sum"] / bucket["tokens"],
                           "baseline_nll": bucket["base"] / bucket["tokens"],
                           "tokens": bucket["tokens"]}
                     for key, bucket in sorted(classes.items())},
        "aggregate_nll": aggregate / tokens_total,
        "aggregate_baseline_nll": aggregate_base / tokens_total,
        "delta": deltas,
        "top1_agreement": agree / tokens_total,
        "savings": estimate_savings(flops, skipped_calls, tokens_total),
    }


if __name__ == "__main__":
    raise SystemExit(main())

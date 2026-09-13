"""Score a learned admission gate against the stock model, on the intervention's harness.

The point of using this harness rather than the trainer's own evaluation is comparability.
The attenuation study's numbers -- content NLL, the class split, the paired document
statistics, the 384-document screen and the disjoint confirmation corpus -- all came from
``scratch/ffn_attenuation/attenuate.py``. A learned gate that improves content NLL by some
amount is only interesting next to the 0.0514 nats a hand-chosen fixed rule achieved on
the same documents with the same baseline, so the same arithmetic produces both.

Four things it reports:

    delta         per-class NLL against the stock backbone, paired by document
    gate          the learned admission distribution per layer, split by context and class
    manual        the best fixed rule from the intervention study, for reference
    identity      the same gate weights with admission forced back to 1

The last is the ablation the co-adaptation stage will need, and it is worth having from
the start: if forcing ``g = 1`` recovers the stock result exactly, then the gate is the
only thing that changed, which is the claim every delta here depends on.

    CUDA_VISIBLE_DEVICES=0 python scratch/residual_gate/evaluate.py \
        --gate runs/gate-familiarity/gate-step284.pt --output ...
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ffn_memo"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ffn_attenuation"))

import numpy as np
import torch
import torch.nn.functional as F

from attenuate import Familiarity, load_records
from distillkit.ffn_skip import attenuate_ffn
from distillkit.experimental.residual_gate import (
    TrigramFamiliarity, family_features, install_residual_gates,
    remove_residual_gates)
from repeatability import DEFAULT_BUNDLE, DEFAULT_MODEL, trigram_keys

DEFAULT_CACHE = Path("scratch/ffn_memo/cache/layer-12.npz")
# The fixed rule the intervention study validated, for the comparison in the report.
MANUAL = {"layer": 12, "alpha": 0.0, "min_count": 4, "max_variance": 0.25}
# Familiarity buckets for the gate-behaviour table. The intervention study found the
# benefit turned positive somewhere around 400 cross-document occurrences, so the edges
# straddle that rather than being evenly spaced in a quantity nothing depends on.
COUNT_EDGES = (0, 1, 4, 100, 400, 800, 3000)


def split_layout(classes, tokenizer, vocab_size):
    """Newline apart from other whitespace: it dominated three earlier aggregates."""
    for index in range(vocab_size):
        if classes[index] == "layout":
            classes[index] = ("newline" if "\n" in tokenizer.decode([index])
                              else "whitespace")
    return classes


def paired(per_document):
    out = {}
    for label, values in per_document.items():
        mean = statistics.fmean(values)
        error = statistics.stdev(values) / len(values) ** 0.5 if len(values) > 1 else 0.0
        out[label] = {"mean": mean, "stderr": error,
                      "t": mean / error if error else 0.0,
                      "ci95": [mean - 1.96 * error, mean + 1.96 * error],
                      "better": sum(1 for value in values if value < 0),
                      "n": len(values)}
    return out


@torch.inference_mode()
def score(model, records, device, classes, baseline, collect=None):
    """One pass over the corpus; deltas are against the cached stock baseline."""
    per_document = {}
    totals = {}
    agree = 0
    tokens_total = 0
    for record, (base_nll, base_argmax, targets) in zip(records, baseline):
        ids = record["ids"]
        tokens = torch.tensor([ids], device=device)
        if collect is not None:
            collect["handle"].reset_stats()
        logits = model(input_ids=tokens,
                       attention_mask=torch.ones_like(tokens)).logits[0, :-1].float()
        nll = F.cross_entropy(logits, torch.tensor(targets, device=device),
                              reduction="none").cpu()
        agree += int((logits.argmax(-1).cpu() == base_argmax).sum())
        tokens_total += len(targets)
        if collect is not None:
            collect["rows"].append((ids, collect["handle"].kept))

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
        # Every token, so the aggregate is reported beside the class split rather than
        # reconstructed from it.
        overall = sum(float(nll[i] - base_nll[i]) for i in range(len(targets)))
        per_document.setdefault("all", []).append(overall / max(len(targets), 1))

    return {"delta": paired(per_document),
            "by_class": {label: {"nll": bucket[0] / bucket[2],
                                 "baseline_nll": bucket[1] / bucket[2],
                                 "tokens": bucket[2],
                                 "nats": bucket[0] - bucket[1]}
                         for label, bucket in sorted(totals.items())},
            "top1_agreement": agree / tokens_total,
            "tokens": tokens_total}


def gate_distribution(rows, familiarity, classes, vocab_size):
    """What the gate learned, split by the variables the intervention study moved."""
    report = {}
    for layer in sorted({layer for _, kept in rows for layer in kept}):
        values, counts, labels = [], [], []
        for ids, kept in rows:
            if layer not in kept:
                continue
            gates = torch.cat([part.reshape(-1) for part in kept[layer]]).numpy()
            keys = trigram_keys(ids, vocab_size)
            if len(gates) != len(ids):
                raise ValueError("kept %d gate values for %d tokens"
                                 % (len(gates), len(ids)))
            values.append(gates)
            counts.append(np.array([familiarity.counts.get(key, 0) if key is not None
                                    else 0 for key in keys], dtype=np.float64))
            labels.extend(classes[token] for token in ids)
        values = np.concatenate(values)
        counts = np.concatenate(counts)
        labels = np.array(labels)

        def describe(selected):
            chosen = values[selected]
            if not len(chosen):
                return None
            return {"tokens": int(len(chosen)),
                    "mean": float(chosen.mean()), "median": float(np.median(chosen)),
                    "std": float(chosen.std()),
                    "p5": float(np.percentile(chosen, 5)),
                    "p25": float(np.percentile(chosen, 25)),
                    "p75": float(np.percentile(chosen, 75)),
                    "p95": float(np.percentile(chosen, 95))}

        entry = {"all": describe(np.ones(len(values), dtype=bool))}
        # 'familiar' and 'rare' are the populations the intervention separated: the
        # selected mask was count >= 4, and rare meant a context the cache never saw.
        entry["familiar"] = describe(counts >= MANUAL["min_count"])
        entry["rare"] = describe(counts == 0)
        entry["by_class"] = {label: describe(labels == label)
                             for label in sorted(set(labels.tolist()))}
        buckets = {}
        for low, high in zip(COUNT_EDGES[:-1], COUNT_EDGES[1:]):
            buckets["%d-%d" % (low, high)] = describe((counts >= low) & (counts < high))
        buckets["%d+" % COUNT_EDGES[-1]] = describe(counts >= COUNT_EDGES[-1])
        entry["by_count"] = buckets
        report[str(layer)] = entry
    return report


@torch.inference_mode()
def manual_arm(model, records, device, classes, baseline, familiarity):
    """The best fixed rule from the intervention study, scored on the same pass."""
    per_document = {}
    for record, (base_nll, _, targets) in zip(records, baseline):
        ids = record["ids"]
        keys = trigram_keys(ids, model.config.vocab_size)[2:]
        selected = familiarity.select(keys, MANUAL["min_count"], MANUAL["max_variance"])
        mask = torch.zeros(1, len(ids), dtype=torch.bool)
        if selected:
            mask[0, [position + 2 for position in selected]] = True
        tokens = torch.tensor([ids], device=device)
        with attenuate_ffn(model, [MANUAL["layer"]]) as handle:
            handle.set(MANUAL["layer"], mask.to(device), MANUAL["alpha"])
            logits = model(input_ids=tokens,
                           attention_mask=torch.ones_like(tokens)).logits[0, :-1].float()
        nll = F.cross_entropy(logits, torch.tensor(targets, device=device),
                              reduction="none").cpu()
        sums = {}
        for index, token in enumerate(targets):
            entry = sums.setdefault(classes[token], [0.0, 0])
            entry[0] += float(nll[index] - base_nll[index])
            entry[1] += 1
        for label, (total, count) in sums.items():
            per_document.setdefault(label, []).append(total / count)
    return paired(per_document)


def timing(model, ids, device, repeats=10):
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for _ in range(3):
            model(input_ids=ids, attention_mask=torch.ones_like(ids))
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        for _ in range(repeats):
            model(input_ids=ids, attention_mask=torch.ones_like(ids))
        torch.cuda.synchronize(device)
    return {"seconds_per_forward": (time.perf_counter() - started) / repeats,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--bundle", default=DEFAULT_BUNDLE)
    parser.add_argument("--split", default="screen")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--gate", required=True, type=Path,
                        help="a gate checkpoint written by ResidualGateCheckpointCallback")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip-manual", action="store_true")
    parser.add_argument("--timing", action="store_true")
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
    classes = split_layout(build_token_classes(tokenizer, config.vocab_size), tokenizer,
                           config.vocab_size)
    familiarity = Familiarity(args.cache)
    records = load_records(args.bundle, args.split, args.limit)

    # The stock baseline, computed on this card in this process, before any gate exists.
    baseline = []
    with torch.inference_mode():
        for record in records:
            tokens = torch.tensor([record["ids"]], device=args.device)
            logits = model(input_ids=tokens,
                           attention_mask=torch.ones_like(tokens)).logits[0, :-1].float()
            targets = record["ids"][1:]
            baseline.append(
                (F.cross_entropy(logits, torch.tensor(targets, device=args.device),
                                 reduction="none").cpu(),
                 logits.argmax(-1).cpu(), targets))

    report = {"model": args.model, "bundle": args.bundle, "split": args.split,
              "documents": len(records), "gate": str(args.gate)}
    if not args.skip_manual:
        report["manual"] = {"rule": MANUAL,
                            "delta": manual_arm(model, records, args.device, classes,
                                                baseline, familiarity)}

    payload = torch.load(args.gate, map_location="cpu", weights_only=False)
    family = payload["family"]
    layers = payload["layers"]
    statistics_source = None
    if "log_count" in family_features(family):
        statistics_source = TrigramFamiliarity(args.cache, config.vocab_size)
    handle = install_residual_gates(model, layers, family=family,
                                    familiarity=statistics_source)
    model.residual_gates.load_state_dict(payload["state_dict"], strict=True)
    report["gate_step"] = payload["step"]
    report["gate_family"] = family
    report["gate_layers"] = layers
    report["gate_parameters"] = sum(p.numel() for p in model.residual_gates.parameters())
    for index in handle.layer_indices:
        gate = handle.gate(index)
        if not gate.is_calibrated:
            raise SystemExit("the loaded gate has no feature normalizer; the checkpoint "
                             "predates calibration or was written mid-pass")
    if args.timing:
        ids = torch.tensor([records[0]["ids"]], device=args.device)
        report["timing"] = {"gated": timing(model, ids, args.device)}

    try:
        handle.keep = True
        collect = {"handle": handle, "rows": []}
        report["gated"] = score(model, records, args.device, classes, baseline, collect)
        handle.keep = False
        report["gate_distribution"] = gate_distribution(
            collect["rows"], familiarity, classes, config.vocab_size)

        # Same weights, admission forced back to unit strength. This has to reproduce
        # the stock model exactly, or the delta above is not attributable to the gate.
        handle.force_identity = True
        report["forced_identity"] = score(model, records, args.device, classes, baseline)
        handle.force_identity = False
    finally:
        remove_residual_gates(model)

    if args.timing:
        ids = torch.tensor([records[0]["ids"]], device=args.device)
        report["timing"]["stock"] = timing(model, ids, args.device)

    report["elapsed_seconds"] = time.monotonic() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary = {"content": report["gated"]["delta"]["content"]["mean"],
               "t": report["gated"]["delta"]["content"]["t"],
               "identity_content": report["forced_identity"]["delta"]["content"]["mean"]}
    if "manual" in report:
        summary["manual_content"] = report["manual"]["delta"]["content"]["mean"]
    print(json.dumps(summary, indent=2))
    print("wrote %s in %.0f s" % (args.output, report["elapsed_seconds"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

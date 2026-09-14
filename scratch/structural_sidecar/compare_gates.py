"""Every gate against every structural setting, on one backbone, in one pass.

The comparison the task asks for is ``(B + G_S + S) - (B + G + S)``, and it needs three
gates scored under identical semantics: the original gate, a gate refitted in this
harness without the sidecar, and one refitted with it. The middle one exists because this
harness trains on plain cross-entropy over the memoisation corpus while the original gate
was trained on assistant-masked cross-entropy over the teacher cache -- so any difference
between the original and the new gate would otherwise be a difference of harness as much
as of sidecar.

Nothing trains here. Every gate, the backbone and the sidecar are loaded frozen.

    CUDA_VISIBLE_DEVICES=0 python scratch/structural_sidecar/compare_gates.py \
        --output scratch/structural_sidecar/gate-comparison-B42.json
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
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "residual_gate"))

import numpy as np
import torch
import torch.nn.functional as F

from attenuate import Familiarity, load_records
from distillkit.experimental.residual_gate import (
    TrigramFamiliarity, install_residual_gates, remove_residual_gates)
from distillkit.experimental.structural_sidecar import (
    apply_structural_bias, structural_token_ids)
from evaluate import COUNT_EDGES, DEFAULT_CACHE, split_layout
from factorize import class_ids
from fit import BACKBONES, BASE
import gate_after_structure
from gate_after_structure import GATE, build_structural
from repeatability import DEFAULT_BUNDLE, trigram_keys

GATES = {
    "G": GATE,
    "G_harness": Path("scratch/structural_sidecar/gate-without-S/gate.pt"),
    "G_S": Path("scratch/structural_sidecar/gate-with-S/gate.pt"),
}
CLASSES = ("content", "newline", "whitespace", "punctuation", "control", "all")


def paired(left, right):
    values = [a - b for a, b in zip(left, right)]
    mean = statistics.fmean(values)
    error = statistics.stdev(values) / len(values) ** 0.5 if len(values) > 1 else 0.0
    return {"mean": mean, "t": mean / error if error else 0.0,
            "better": sum(1 for value in values if value < 0), "n": len(values)}


@torch.no_grad()
def score(model, documents, device, classes, structural, handle, sidecar, hasher,
          strength, familiarity=None, vocab=0):
    per_document, totals = {}, {}
    admission = {}
    buckets = {}
    for ids in documents:
        tokens = torch.tensor([ids], device=device)
        if handle is not None:
            handle.reset_stats()
        logits = model(input_ids=tokens,
                       attention_mask=torch.ones_like(tokens)).logits[0, :-1].float()
        if sidecar is not None:
            bias = sidecar(hasher.row_indices(tokens), strength)[0, :-1]
            logits = apply_structural_bias(logits.unsqueeze(0), bias.unsqueeze(0),
                                           structural, 1.0)[0]
        if handle is not None and handle.keep and familiarity is not None:
            keys = trigram_keys(ids, vocab)
            counts = np.array([familiarity.counts.get(key, 0) if key is not None else 0
                               for key in keys], dtype=np.float64)
            for layer, parts in handle.kept.items():
                values = torch.cat([part.reshape(-1) for part in parts]).numpy()
                for index, token in enumerate(ids):
                    admission.setdefault(layer, {}).setdefault(
                        classes[token], []).append(float(values[index]))
                edges = list(zip(COUNT_EDGES[:-1], COUNT_EDGES[1:]))
                edges.append((COUNT_EDGES[-1], float("inf")))
                for low, high in edges:
                    inside = (counts >= low) & (counts < high)
                    if inside.any():
                        label = ("%d-%d" % (low, high) if high != float("inf")
                                 else "%d+" % low)
                        buckets.setdefault(layer, {}).setdefault(label, []).extend(
                            values[inside].tolist())
        targets = torch.tensor(ids[1:], device=device)
        nll = F.cross_entropy(logits, targets, reduction="none").cpu()
        sums = {}
        for index, token in enumerate(ids[1:]):
            label = classes[token]
            bucket = totals.setdefault(label, [0.0, 0])
            bucket[0] += float(nll[index])
            bucket[1] += 1
            entry = sums.setdefault(label, [0.0, 0])
            entry[0] += float(nll[index])
            entry[1] += 1
        sums["all"] = [float(nll.sum()), len(ids) - 1]
        totals.setdefault("all", [0.0, 0])
        totals["all"][0] += float(nll.sum())
        totals["all"][1] += len(ids) - 1
        for label, (value, count) in sums.items():
            per_document.setdefault(label, []).append(value / count)
    out = {"per_document": per_document,
           "mean": {label: statistics.fmean(values)
                    for label, values in per_document.items()},
           "totals": {label: {"nll": bucket[0] / bucket[1], "tokens": bucket[1]}
                      for label, bucket in sorted(totals.items())}}
    if admission:
        out["admission"] = {
            str(layer): {label: {"mean": statistics.fmean(values),
                                 "median": statistics.median(values),
                                 "p10": float(np.percentile(values, 10)),
                                 "p90": float(np.percentile(values, 90))}
                         for label, values in sorted(entry.items())}
            for layer, entry in sorted(admission.items())}
        out["by_familiarity"] = {
            str(layer): {label: statistics.fmean(values)
                         for label, values in sorted(entry.items())}
            for layer, entry in sorted(buckets.items())}
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", default="B42", choices=sorted(BACKBONES))
    parser.add_argument("--gates", nargs="*", default=sorted(GATES))
    parser.add_argument("--bundle", default=DEFAULT_BUNDLE)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--whitespace", type=Path, default=None,
                        help="the whitespace values for this backbone; the addressed "
                             "branch is portable and these are not, so a transfer test "
                             "that reused the wrong ones would confound the two")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if torch.cuda.device_count() != 1:
        raise SystemExit("mask to one card with CUDA_VISIBLE_DEVICES")

    from transformers import AutoConfig, AutoTokenizer
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    from distillkit.independent_eval import build_token_classes

    started = time.monotonic()
    checkpoint = BACKBONES[args.backbone]
    config = AutoConfig.from_pretrained(checkpoint, local_files_only=True)
    config = getattr(config, "text_config", config)
    config.use_cache = False
    model = Qwen3_5ForCausalLM.from_pretrained(
        checkpoint, config=config, dtype=torch.bfloat16,
        local_files_only=True).to(args.device).eval()
    model.requires_grad_(False)

    tokenizer = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
    classes = split_layout(build_token_classes(tokenizer, config.vocab_size), tokenizer,
                           config.vocab_size)
    structural = structural_token_ids(classes, device=args.device)
    whitespace = class_ids(classes, "whitespace", device=args.device)
    if args.whitespace is not None:
        gate_after_structure.WHITESPACE = args.whitespace
    sidecar, hasher, strength = build_structural(config, structural, whitespace,
                                                 args.device)
    familiarity = Familiarity(args.cache)

    report = {"backbone": args.backbone, "gates": {}, "splits": {}}
    for split in ("screen", "confirmation"):
        documents = [record["ids"] for record in load_records(args.bundle, split, 0)]
        series = {}
        series["stock"] = score(model, documents, args.device, classes, structural,
                                None, None, hasher, strength)
        series["S"] = score(model, documents, args.device, classes, structural, None,
                            sidecar, hasher, strength)
        for name in args.gates:
            payload = torch.load(GATES[name], map_location="cpu", weights_only=False)
            source = TrigramFamiliarity(args.cache, config.vocab_size)
            handle = install_residual_gates(model, payload["layers"],
                                            family=payload["family"],
                                            familiarity=source)
            model.residual_gates.load_state_dict(payload["state_dict"], strict=True)
            model.residual_gates.requires_grad_(False)
            try:
                handle.keep = split == "screen"
                series[name] = score(model, documents, args.device, classes, structural,
                                     handle, None, hasher, strength,
                                     familiarity, config.vocab_size)
                handle.keep = False
                series[name + "+S"] = score(model, documents, args.device, classes,
                                            structural, handle, sidecar, hasher,
                                            strength)
                handle.force_identity = True
                series[name + "+S(g=1)"] = score(model, documents, args.device, classes,
                                                 structural, handle, sidecar, hasher,
                                                 strength)
                handle.force_identity = False
            finally:
                remove_residual_gates(model)
            report["gates"][name] = {"path": str(GATES[name]),
                                     "layers": payload["layers"]}

        entry = {"mean": {name: value["mean"] for name, value in series.items()},
                 "totals": {name: value["totals"] for name, value in series.items()},
                 "differences": {}}
        for name in series:
            if name == "stock":
                continue
            entry["differences"]["%s - stock" % name] = {
                label: paired(series[name]["per_document"][label],
                              series["stock"]["per_document"][label])
                for label in series["stock"]["per_document"]}
        for name in args.gates:
            if name == "G":
                continue
            entry["differences"]["%s+S - G+S" % name] = {
                label: paired(series[name + "+S"]["per_document"][label],
                              series["G+S"]["per_document"][label])
                for label in series["stock"]["per_document"]}
        if split == "screen":
            entry["admission"] = {name: series[name].get("admission")
                                  for name in args.gates}
            entry["by_familiarity"] = {name: series[name].get("by_familiarity")
                                       for name in args.gates}
        report["splits"][split] = entry

        base = series["stock"]["mean"]
        for name in series:
            if name == "stock":
                continue
            print("%-12s %-16s %s" % (split, name, "  ".join(
                "%s %+.6f" % (label, series[name]["mean"].get(label, 0.0)
                              - base.get(label, 0.0)) for label in CLASSES)),
                  flush=True)

    report["elapsed_seconds"] = time.monotonic() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("wrote %s in %.0f s" % (args.output, report["elapsed_seconds"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

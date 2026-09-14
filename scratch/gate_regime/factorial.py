"""Which half of the training-regime mismatch bought the 0.056 nats?

The task specified a corpus x loss factorial. There is no corpus factor: the
"memoisation corpus" and the teacher cache are the same documents in the same order --
`capture-data/heldout.jsonl` is the file the cache was built from, and all 24 of the first
24 documents match token for token, lengths included. The masking probe found identical
masked fractions on both because they are the same text.

What actually differed between the original gate's training and the harness that beat it
by 0.056 nats is two things, and neither is the corpus:

    mask       assistant-masked cross-entropy against plain cross-entropy over every
               token; the masked fraction on this corpus averages 0.489, so roughly half
               the tokens were never scored
    regime     284 optimizer steps at sequence 4096 on a cosine schedule with warmup,
               against 1536 constant-rate steps at sequence 512

So this is the 2x2 the evidence supports. Arm ``assistant/original`` reproduces the
condition the original gate was trained under; arm ``plain/harness`` reproduces the
control gate that beat it. Everything else -- architecture, layers, features, parameter
count, initialisation, optimizer, learning rate, seed, evaluation, frozen backbone -- is
identical across all four.

    CUDA_VISIBLE_DEVICES=0 python scratch/gate_regime/factorial.py \
        --mask assistant --regime original --output scratch/gate_regime/assistant-original
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ffn_memo"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ffn_attenuation"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "residual_gate"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "structural_sidecar"))

import numpy as np
import torch
import torch.nn.functional as F

from attenuate import Familiarity, load_records
from distillkit.experimental.residual_gate import (
    TrigramFamiliarity, calibrate_gates, install_residual_gates,
    remove_residual_gates)
from distillkit.lossfuncs.cross_entropy import assistant_token_mask
from evaluate import COUNT_EDGES, DEFAULT_CACHE, split_layout
from fit import BACKBONES, BASE, corpus
from gate_after_structure import GATE, digest_of
from repeatability import DEFAULT_BUNDLE, held_out_digests, trigram_keys

# (sequence length, optimizer steps, warmup, schedule)
REGIMES = {
    # What distillkit.main ran for the original gate: long sequences, few updates,
    # cosine decay from a short warmup.
    "original": {"length": 4096, "steps": 284, "warmup": 5, "schedule": "cosine"},
    # What the harness that beat it ran: short sequences, many updates, constant rate.
    "harness": {"length": 512, "steps": 1536, "warmup": 0, "schedule": "constant"},
}


def schedule_factor(step: int, plan: dict) -> float:
    if step < plan["warmup"]:
        return (step + 1) / max(plan["warmup"], 1)
    if plan["schedule"] == "constant":
        return 1.0
    progress = (step - plan["warmup"]) / max(plan["steps"] - plan["warmup"], 1)
    return 0.5 * (1.0 + float(np.cos(np.pi * min(progress, 1.0))))


def paired(left, right):
    values = [a - b for a, b in zip(left, right)]
    mean = statistics.fmean(values)
    error = statistics.stdev(values) / len(values) ** 0.5 if len(values) > 1 else 0.0
    return {"mean": mean, "t": mean / error if error else 0.0,
            "better": sum(1 for value in values if value < 0), "n": len(values)}


@torch.no_grad()
def score(model, documents, device, classes, handle, familiarity, vocab):
    per_document, totals = {}, {}
    admission, buckets = {}, {}
    for ids in documents:
        tokens = torch.tensor([ids], device=device)
        if handle is not None:
            handle.reset_stats()
        logits = model(input_ids=tokens,
                       attention_mask=torch.ones_like(tokens)).logits[0, :-1].float()
        if handle is not None and handle.keep:
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
    parser.add_argument("--mask", default="plain", choices=("plain", "assistant"))
    parser.add_argument("--regime", default="harness", choices=sorted(REGIMES))
    parser.add_argument("--documents", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--bundle", default=DEFAULT_BUNDLE)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if torch.cuda.device_count() != 1:
        raise SystemExit("mask to one card with CUDA_VISIBLE_DEVICES")

    from transformers import AutoConfig, AutoTokenizer
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    from distillkit.independent_eval import build_token_classes

    started = time.monotonic()
    plan = REGIMES[args.regime]
    torch.manual_seed(args.seed)

    checkpoint = BACKBONES[args.backbone]
    config = AutoConfig.from_pretrained(checkpoint, local_files_only=True)
    config = getattr(config, "text_config", config)
    config.use_cache = False
    model = Qwen3_5ForCausalLM.from_pretrained(
        checkpoint, config=config, dtype=torch.bfloat16,
        local_files_only=True).to(args.device).eval()
    model.requires_grad_(False)
    backbone_digest = digest_of(model)

    tokenizer = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
    classes = split_layout(build_token_classes(tokenizer, config.vocab_size), tokenizer,
                           config.vocab_size)
    familiarity = Familiarity(args.cache)
    excluded = held_out_digests(args.bundle)
    train = corpus(tokenizer, excluded, args.documents, plan["length"], skip=0)

    payload = torch.load(GATE, map_location="cpu", weights_only=False)
    handle = install_residual_gates(
        model, payload["layers"], family=payload["family"],
        familiarity=TrigramFamiliarity(args.cache, config.vocab_size))
    batches = []
    for ids in train[:8]:
        tokens = torch.tensor([ids], device=args.device)
        batches.append({"input_ids": tokens, "attention_mask": torch.ones_like(tokens)})
    calibrate_gates(model, handle, batches)

    report = {"backbone": args.backbone, "mask": args.mask, "regime": args.regime,
              "plan": plan, "lr": args.lr, "seed": args.seed,
              "documents": len(train),
              "parameters": sum(p.numel() for p in handle.gates.parameters()),
              "backbone_sha256": backbone_digest}
    print(json.dumps({k: report[k] for k in ("mask", "regime", "plan", "parameters")}),
          flush=True)

    optimizer = torch.optim.AdamW(
        [p for p in handle.gates.parameters() if p.requires_grad], lr=args.lr)
    masked_tokens = scored_tokens = 0
    losses = []
    step = 0
    while step < plan["steps"]:
        for ids in train:
            if step >= plan["steps"]:
                break
            for group in optimizer.param_groups:
                group["lr"] = args.lr * schedule_factor(step, plan)
            tokens = torch.tensor([ids], device=args.device)
            logits = model(input_ids=tokens,
                           attention_mask=torch.ones_like(tokens)).logits[0, :-1].float()
            targets = torch.tensor(ids[1:], device=args.device)
            if args.mask == "assistant":
                keep = assistant_token_mask(tokens, torch.ones_like(tokens),
                                            tokenizer)[0, 1:]
                scored_tokens += len(targets)
                masked_tokens += int(keep.sum())
                if not keep.any():
                    step += 1
                    continue
                loss = F.cross_entropy(logits[keep], targets[keep])
            else:
                scored_tokens += len(targets)
                masked_tokens += len(targets)
                loss = F.cross_entropy(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
            step += 1
            if step % 200 == 0:
                print("step %d loss %.5f" % (step, statistics.fmean(losses[-200:])),
                      flush=True)
    report["final_loss"] = statistics.fmean(losses[-100:])
    report["scored_token_fraction"] = masked_tokens / max(scored_tokens, 1)
    report["reach"] = {str(index): handle.gate(index).gate_report("g")["g/reach"]
                       for index in handle.layer_indices}
    print("scored fraction %.4f  reach %s"
          % (report["scored_token_fraction"], json.dumps(report["reach"])), flush=True)

    for split in ("screen", "confirmation"):
        documents = [record["ids"] for record in load_records(args.bundle, split, 0)]
        handle.force_identity = True
        stock = score(model, documents, args.device, classes, None, familiarity,
                      config.vocab_size)
        handle.force_identity = False
        handle.keep = split == "screen"
        gated = score(model, documents, args.device, classes, handle, familiarity,
                      config.vocab_size)
        handle.keep = False
        entry = {"stock": stock["mean"], "gated": gated["mean"],
                 "totals": gated["totals"],
                 "delta": {label: paired(gated["per_document"][label],
                                         stock["per_document"][label])
                           for label in stock["per_document"]}}
        if split == "screen":
            entry["admission"] = gated.get("admission")
            entry["by_familiarity"] = gated.get("by_familiarity")
        report[split] = entry
        print("%-12s %s" % (split, "  ".join(
            "%s %+.6f" % (label, entry["delta"][label]["mean"])
            for label in ("content", "newline", "whitespace", "punctuation", "control",
                          "all"))), flush=True)

    if digest_of(model) != backbone_digest:
        raise SystemExit("the frozen backbone changed")

    report["elapsed_seconds"] = time.monotonic() - started
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2),
                                             encoding="utf-8")
    state = {key: value.detach().cpu()
             for key, value in handle.gates.state_dict().items()}
    torch.save({"step": plan["steps"], "family": payload["family"],
                "layers": payload["layers"], "state_dict": state,
                "mask": args.mask, "regime": args.regime}, args.output / "gate.pt")
    report["gate_sha256"] = hashlib.sha256(
        b"".join(value.numpy().tobytes() for _, value in sorted(state.items()))
    ).hexdigest()
    (args.output / "report.json").write_text(json.dumps(report, indent=2),
                                             encoding="utf-8")
    remove_residual_gates(model)
    print("wrote %s in %.0f s" % (args.output, report["elapsed_seconds"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

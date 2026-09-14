"""Fit a structural sidecar on a frozen backbone, calibrate its strength, score it.

Three splits, used for three different things, and never for each other's job:

    train         fits the sidecar weights
    calibration   selects the strength, after the weights are frozen
    screen        reported
    confirmation  reported, never looked at until the strength is fixed

The backbone never takes a gradient. That is asserted at the start and verified bitwise
at the end rather than trusted: the whole point of a post-hoc correction is that the thing
it corrects did not move to accommodate it.

The objective is structural cross-entropy with an explicit content guardrail::

    L = CE_struct(z + b) + beta * relu(CE_content(z + b) - CE_content(z))

The second term costs nothing to compute -- the unbiased logits are already in hand -- and
it is what stops the sidecar buying newline accuracy with content. A sidecar that can only
move 6,166 of 248,320 logits can still hurt content through the softmax denominator, so
"it cannot touch content" is not an argument, it is a thing to measure.

    CUDA_VISIBLE_DEVICES=0 python scratch/structural_sidecar/fit.py --mode fixed \
        --backbone B42 --output scratch/structural_sidecar/fixed-B42
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ffn_memo"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ffn_attenuation"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "residual_gate"))

import torch
import torch.nn.functional as F

from attenuate import load_records
from distillkit.experimental.ngram_hash import NGramHashConfig, NGramHasher
from distillkit.experimental.structural_sidecar import (
    StructuralSidecar, apply_structural_bias, structural_token_ids,
    wrong_context_rows)
from evaluate import split_layout
from repeatability import DEFAULT_BUNDLE, held_out_digests

RUNS = Path("D:/DeepThought/Projects/HybridModel/runs")
BASE = "D:/DeepThought/Projects/HybridModel/student-2b-hf"
SOURCE = "D:/DeepThought/Projects/HybridModel/capture-data/heldout.jsonl"
BACKBONES = {
    "B42": RUNS / "gate-coadapt-armB" / "checkpoint-284",
    "B43": RUNS / "gate-coadapt-s43m-armB" / "checkpoint-284",
    "B0": Path(BASE),
}
# The coarse sweep, with headroom above 1: a sidecar fitted at unit amplitude has no
# reason to have found the right amplitude, and the smoke run's optimum sat on the old
# boundary. An optimum that still lands on an edge is reported as an edge, not as a
# choice.
STRENGTHS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)


def corpus(tokenizer, excluded, count, length, skip):
    """Documents the evaluation never sees, split by position rather than by chance."""
    documents = []
    seen = 0
    with io.open(SOURCE, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            text = json.loads(line)["text"]
            if hashlib.sha256(text.encode("utf-8")).hexdigest() in excluded:
                continue
            seen += 1
            if seen <= skip:
                continue
            ids = tokenizer(text)["input_ids"][:length]
            if len(ids) >= 16:
                documents.append(ids)
            if len(documents) >= count:
                break
    if len(documents) < count:
        raise SystemExit("only %d documents available, wanted %d" % (len(documents), count))
    return documents


def class_losses(logits, targets, labels, structural_mask):
    """Per-token NLL, split into the structural half and the content half."""
    nll = F.cross_entropy(logits, targets, reduction="none")
    return nll[structural_mask], nll[~structural_mask], nll


@torch.no_grad()
def backbone_logits(model, ids, device):
    tokens = torch.tensor([ids], device=device)
    return model(input_ids=tokens,
                 attention_mask=torch.ones_like(tokens)).logits[0, :-1].float(), tokens


def evaluate(model, sidecar, hasher, documents, device, structural, classes, strength,
             wrong=False):
    """Per-document per-class NLL at a fixed strength, against the same backbone."""
    per_document = {}
    totals = {}
    with torch.no_grad():
        for ids in documents:
            logits, tokens = backbone_logits(model, ids, device)
            rows = hasher.row_indices(tokens)
            if wrong:
                rows = wrong_context_rows(rows)
            bias = sidecar(rows)[0, :-1]
            biased = apply_structural_bias(logits.unsqueeze(0), bias.unsqueeze(0),
                                           structural, strength)[0]
            targets = torch.tensor(ids[1:], device=device)
            nll = F.cross_entropy(biased, targets, reduction="none").cpu()
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
    return ({label: statistics.fmean(values) for label, values in per_document.items()},
            {label: {"nll": bucket[0] / bucket[1], "tokens": bucket[1]}
             for label, bucket in sorted(totals.items())},
            per_document)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", default="fixed", choices=("table", "fixed", "direct", "none"))
    parser.add_argument("--backbone", default="B42", choices=sorted(BACKBONES))
    parser.add_argument("--rows", type=int, default=1 << 17)
    parser.add_argument("--code-dim", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--documents", type=int, default=512)
    parser.add_argument("--calibration-documents", type=int, default=128)
    parser.add_argument("--length", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--beta", type=float, default=10.0)
    parser.add_argument("--load", type=Path, default=None,
                        help="a fitted sidecar to reuse instead of training one")
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
    checkpoint = BACKBONES[args.backbone]
    config = AutoConfig.from_pretrained(checkpoint, local_files_only=True)
    config = getattr(config, "text_config", config)
    config.use_cache = False
    model = Qwen3_5ForCausalLM.from_pretrained(
        checkpoint, config=config, dtype=torch.bfloat16,
        local_files_only=True).to(args.device).eval()
    model.requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise SystemExit("a backbone parameter still requires grad")
    fingerprint = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        fingerprint.update(name.encode("utf-8"))
        fingerprint.update(parameter.detach().to(torch.float32).cpu().numpy().tobytes())
    before = fingerprint.hexdigest()

    tokenizer = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
    classes = split_layout(build_token_classes(tokenizer, config.vocab_size), tokenizer,
                           config.vocab_size)
    structural = structural_token_ids(classes, device=args.device)
    structural_set = torch.zeros(config.vocab_size, dtype=torch.bool)
    structural_set[structural.cpu()] = True

    hasher = NGramHasher(NGramHashConfig(
        vocab_size=config.vocab_size, ngram_size=3, heads_per_ngram=1,
        ngram_vocab_size_base=args.rows // 2, seed=1234,
        eos_token_id=tokenizer.eos_token_id or config.vocab_size - 1))
    sidecar = StructuralSidecar(rows=hasher.padded_vocab_size, code_dim=args.code_dim,
                                structural=int(structural.numel()), mode=args.mode,
                                heads=2, hidden=args.hidden, seed=args.seed
                                ).to(args.device).to(torch.float32)

    excluded = held_out_digests(args.bundle)
    train = corpus(tokenizer, excluded, args.documents, args.length, skip=0)
    calibration = corpus(tokenizer, excluded, args.calibration_documents, args.length,
                         skip=args.documents)
    report = {"mode": args.mode, "backbone": args.backbone,
              "checkpoint": str(checkpoint),
              "sidecar": sidecar.parameter_report(),
              "train_documents": len(train),
              "calibration_documents": len(calibration),
              "lr": args.lr, "epochs": args.epochs, "beta": args.beta,
              "backbone_sha256_before": before}
    print(json.dumps(report["sidecar"]), flush=True)

    if args.load is not None:
        sidecar.load_state_dict(torch.load(args.load, map_location=args.device,
                                           weights_only=False)["state_dict"])
        report["loaded_from"] = str(args.load)
    else:
        optimizer = torch.optim.AdamW(
            [p for p in sidecar.parameters() if p.requires_grad], lr=args.lr)
        history = []
        for epoch in range(args.epochs):
            losses = []
            for ids in train:
                with torch.no_grad():
                    logits, tokens = backbone_logits(model, ids, args.device)
                    targets = torch.tensor(ids[1:], device=args.device)
                    base = F.cross_entropy(logits, targets, reduction="none")
                    mask = structural_set.to(args.device)[targets]
                rows = hasher.row_indices(tokens)
                bias = sidecar(rows)[0, :-1]
                biased = apply_structural_bias(logits.unsqueeze(0), bias.unsqueeze(0),
                                               structural, 1.0)[0]
                nll = F.cross_entropy(biased, targets, reduction="none")
                if not mask.any() or mask.all():
                    continue
                loss = nll[mask].mean() + args.beta * torch.relu(
                    nll[~mask].mean() - base[~mask].mean())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach()))
            history.append(statistics.fmean(losses))
            print("epoch %d loss %.5f" % (epoch, history[-1]), flush=True)
        report["loss_history"] = history

    # The calibration split, and only it, selects the strength.
    stock = None
    curve = {}
    for strength in STRENGTHS:
        means, _, _ = evaluate(model, sidecar, hasher, calibration, args.device,
                               structural, classes, strength)
        curve[strength] = means
        if strength == 0.0:
            stock = means
        print("calibration strength %.2f  content %+.6f  newline %+.6f  punctuation %+.6f"
              % (strength, means["content"] - stock["content"],
                 means.get("newline", 0.0) - stock.get("newline", 0.0),
                 means["punctuation"] - stock["punctuation"]), flush=True)
    report["calibration_curve"] = {str(key): value for key, value in curve.items()}

    # Predeclared rule: the biggest structural gain whose content cost stays inside the
    # guardrail. Chosen before any screen or confirmation number is looked at.
    guardrail = 0.001
    structural_labels = [label for label in stock if label not in ("content", "all")]

    def structural_gain(means):
        return sum(means[label] - stock[label] for label in structural_labels)

    admissible = [strength for strength in STRENGTHS
                  if curve[strength]["content"] - stock["content"] <= guardrail]
    chosen = min(admissible, key=lambda strength: structural_gain(curve[strength]))
    report["guardrail"] = guardrail
    report["selected_strength"] = chosen
    print("selected strength %.2f" % chosen, flush=True)

    for split in ("screen", "confirmation"):
        documents = [record["ids"] for record in load_records(args.bundle, split, 0)]
        entry = {}
        for name, strength, wrong in (("stock", 0.0, False), ("sidecar", chosen, False),
                                      ("wrong_context", chosen, True)):
            means, totals, _ = evaluate(model, sidecar, hasher, documents, args.device,
                                        structural, classes, strength, wrong=wrong)
            entry[name] = {"per_document_mean": means, "totals": totals}
        report[split] = entry
        stock_means = entry["stock"]["per_document_mean"]
        print("%-12s content %+.6f  newline %+.6f  whitespace %+.6f  punctuation %+.6f  "
              "control %+.6f  all %+.6f"
              % (split,
                 *(entry["sidecar"]["per_document_mean"].get(label, 0.0)
                   - stock_means.get(label, 0.0)
                   for label in ("content", "newline", "whitespace", "punctuation",
                                 "control", "all"))), flush=True)

    fingerprint = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        fingerprint.update(name.encode("utf-8"))
        fingerprint.update(parameter.detach().to(torch.float32).cpu().numpy().tobytes())
    report["backbone_sha256_after"] = fingerprint.hexdigest()
    if report["backbone_sha256_after"] != before:
        raise SystemExit("the frozen backbone changed during the run")

    report["elapsed_seconds"] = time.monotonic() - started
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2),
                                             encoding="utf-8")
    torch.save({"state_dict": {key: value.cpu()
                               for key, value in sidecar.state_dict().items()},
                "mode": args.mode, "rows": sidecar.rows, "code_dim": args.code_dim,
                "heads": 2, "hidden": args.hidden, "seed": args.seed,
                "strength": chosen},
               args.output / "sidecar.pt")
    print("wrote %s in %.0f s" % (args.output, report["elapsed_seconds"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

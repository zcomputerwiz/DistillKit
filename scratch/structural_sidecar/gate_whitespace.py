"""Whitespace values are simple. Is whitespace *admission* the contextual part?

The factorization left one precise failure. 485 context-free biases beat the monolithic
module on whitespace by 21%, and could not do that and match its content at the same time:
raising whitespace probability helps where whitespace belongs and costs probability mass
everywhere else. The measured mechanism -- structural mass on content targets going 0.0445
with the addressed branch alone to 0.0493 once the static bias is added, against 0.0467
stock -- says the bias knows *what* to correct and not *when*.

So this trains 33 numbers. One scalar per token, ``a(x) = 2 sigmoid(w . c + b)`` over the
32 trigram context bits the addressed branch already computes, multiplying a whitespace
correction that stays frozen. Zero weights give ``a = 1``, which is the static arm exactly,
so the experiment starts from the thing it has to beat and everything else is held fixed
and verified bitwise.

    CUDA_VISIBLE_DEVICES=0 python scratch/structural_sidecar/gate_whitespace.py \
        --output scratch/structural_sidecar/gated-B42
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

import torch
import torch.nn.functional as F

from attenuate import load_records
from distillkit.experimental.ngram_hash import NGramHashConfig, NGramHasher
from distillkit.experimental.structural_sidecar import (
    FactorizedSidecar, StructuralSidecar, apply_structural_bias,
    structural_token_ids, wrong_context_rows)
from evaluate import split_layout
from factorize import class_ids, logits_of
from fit import BACKBONES, BASE, corpus
from repeatability import DEFAULT_BUNDLE, held_out_digests


def digest_of(module) -> str:
    hasher = hashlib.sha256()
    for name, parameter in sorted(module.named_parameters()):
        hasher.update(name.encode("utf-8"))
        hasher.update(parameter.detach().cpu().numpy().tobytes())
    return hasher.hexdigest()


@torch.no_grad()
def measure(model, module, hasher, documents, device, structural, whitespace, classes,
            arm, strength, wrong=False):
    """One arm, with the probability-mass diagnostics the frontier question needs."""
    per_document, totals = {}, {}
    structural_mass, whitespace_mass, admission = [], [], {}
    for ids in documents:
        logits, tokens = logits_of(model, ids, device)
        if arm != "stock":
            rows = hasher.row_indices(tokens)
            if wrong:
                rows = wrong_context_rows(rows)
            if arm == "addressed":
                bias = module(rows, 0.0)[0, :-1]
            else:
                bias = module(rows, strength)[0, :-1]
            logits = apply_structural_bias(logits.unsqueeze(0), bias.unsqueeze(0),
                                           structural, 1.0)[0]
            if module.gate is not None and arm == "gated":
                values = module.admission(rows)[0, :-1].cpu()
                for index, token in enumerate(ids[1:]):
                    admission.setdefault(classes[token], []).append(float(values[index]))
        targets = torch.tensor(ids[1:], device=device)
        probabilities = torch.softmax(logits, dim=-1)
        content = torch.tensor([classes[token] == "content" for token in ids[1:]],
                               device=device)
        white = torch.tensor([classes[token] == "whitespace" for token in ids[1:]],
                             device=device)
        if content.any():
            structural_mass.append(
                float(probabilities[content][:, structural].sum(-1).mean()))
        if white.any():
            whitespace_mass.append(
                float(probabilities[white][:, whitespace].sum(-1).mean()))
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
    out = {"per_document_mean": {label: statistics.fmean(values)
                                 for label, values in per_document.items()},
           "totals": {label: {"nll": bucket[0] / bucket[1], "tokens": bucket[1]}
                      for label, bucket in sorted(totals.items())},
           "structural_mass_on_content": statistics.fmean(structural_mass),
           "whitespace_mass_on_whitespace": statistics.fmean(whitespace_mass)}
    if admission:
        out["admission"] = {}
        for label, values in sorted(admission.items()):
            series = torch.tensor(values)
            out["admission"][label] = {
                "tokens": len(values), "mean": float(series.mean()),
                "median": float(series.median()),
                **{"p%d" % q: float(series.quantile(q / 100))
                   for q in (10, 25, 75, 90)}}
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", default="B42", choices=sorted(BACKBONES))
    parser.add_argument("--addressed", type=Path,
                        default=Path("scratch/structural_sidecar/direct-B42/sidecar.pt"))
    parser.add_argument("--whitespace", type=Path,
                        default=Path("scratch/structural_sidecar/factorized-B42-stockref"
                                     "/whitespace.pt"))
    parser.add_argument("--load-gate", type=Path, default=None)
    parser.add_argument("--sweep", type=float, nargs="*", default=None,
                        help="also score the gated arm at these whitespace strengths; "
                             "the monolithic module sits at some point on this frontier "
                             "and a single operating point cannot say which")
    parser.add_argument("--refit-bias", action="store_true",
                        help="also fit the 485 values, for the transfer diagnostic only")
    parser.add_argument("--documents", type=int, default=512)
    parser.add_argument("--length", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-2)
    parser.add_argument("--beta", type=float, default=10.0)
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

    tokenizer = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
    classes = split_layout(build_token_classes(tokenizer, config.vocab_size), tokenizer,
                           config.vocab_size)
    structural = structural_token_ids(classes, device=args.device)
    whitespace = class_ids(classes, "whitespace", device=args.device)

    saved = torch.load(args.addressed, map_location="cpu", weights_only=False)
    addressed = StructuralSidecar(rows=saved["rows"], code_dim=saved["code_dim"],
                                  structural=int(structural.numel()), mode=saved["mode"],
                                  heads=saved["heads"], hidden=saved["hidden"],
                                  seed=saved["seed"]).to(args.device).to(torch.float32)
    addressed.load_state_dict(saved["state_dict"])
    module = FactorizedSidecar(addressed, structural.cpu(), whitespace.cpu(),
                               gated=True).to(args.device)
    bias_payload = torch.load(args.whitespace, map_location=args.device,
                              weights_only=False)
    module.white.load_state_dict(bias_payload["state_dict"])
    static_strength = bias_payload["whitespace_strength"]
    module.addressed.requires_grad_(False)
    module.white.requires_grad_(args.refit_bias)
    before = {"addressed": digest_of(module.addressed), "white": digest_of(module.white)}

    hasher = NGramHasher(NGramHashConfig(
        vocab_size=config.vocab_size, ngram_size=3, heads_per_ngram=1,
        ngram_vocab_size_base=saved["rows"] // 2 - 64, seed=1234,
        eos_token_id=tokenizer.eos_token_id or config.vocab_size - 1))

    report = {"backbone": args.backbone, "addressed": str(args.addressed),
              "whitespace": str(args.whitespace),
              "static_whitespace_strength": static_strength,
              "parameters": module.parameter_report()}
    print(json.dumps(report["parameters"]), flush=True)

    if args.load_gate is not None:
        module.gate.load_state_dict(torch.load(args.load_gate, map_location=args.device,
                                               weights_only=False)["state_dict"])
        report["loaded_gate"] = str(args.load_gate)
        module.gate.requires_grad_(args.refit_bias is False)
    # A loaded gate does not mean nothing trains: the portability question is whether a
    # gate fitted elsewhere still works once the 485 local values are refitted, so that
    # combination has to train the values with the gate held fixed.
    if args.load_gate is None or args.refit_bias:
        trainable = ([] if args.load_gate is not None
                     else list(module.gate.parameters()))
        if args.refit_bias:
            trainable += list(module.white.parameters())
        optimizer = torch.optim.AdamW(trainable, lr=args.lr)
        white_mask = torch.zeros(config.vocab_size, dtype=torch.bool)
        white_mask[whitespace.cpu()] = True
        white_mask = white_mask.to(args.device)
        content_mask = torch.tensor([label == "content" for label in classes],
                                    dtype=torch.bool, device=args.device)
        excluded = held_out_digests(args.bundle)
        train = corpus(tokenizer, excluded, args.documents, args.length, skip=0)
        history = []
        for epoch in range(args.epochs):
            losses = []
            for ids in train:
                logits, tokens = logits_of(model, ids, args.device)
                targets = torch.tensor(ids[1:], device=args.device)
                rows = hasher.row_indices(tokens)
                with torch.no_grad():
                    # The guardrail reference is the stock backbone, as it was for the
                    # monolithic module. Holding this branch to the addressed-only arm
                    # instead would pick a different operating point than the comparison
                    # is about.
                    base = F.cross_entropy(logits, targets, reduction="none")
                bias = module(rows, static_strength)[0, :-1]
                biased = apply_structural_bias(logits.unsqueeze(0), bias.unsqueeze(0),
                                               structural, 1.0)[0]
                nll = F.cross_entropy(biased, targets, reduction="none")
                mask = white_mask[targets]
                # The guardrail is on content alone. Pooling it with the other
                # non-whitespace classes lets the addressed branch's newline and
                # punctuation gains -- half a nat each -- hide any amount of content
                # damage inside the mean, and the hinge then never fires. The first
                # run of this experiment had exactly that defect and the gate spent it.
                keep = content_mask[targets]
                if not mask.any() or not keep.any():
                    continue
                loss = nll[mask].mean() + args.beta * torch.relu(
                    nll[keep].mean() - base[keep].mean())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach()))
            history.append(statistics.fmean(losses))
            print("epoch %d loss %.5f" % (epoch, history[-1]), flush=True)
        report["loss_history"] = history

    for split in ("screen", "confirmation"):
        documents = [record["ids"] for record in load_records(args.bundle, split, 0)]
        entry = {}
        # The static arm is this same module with the gate returned to a = 1, which is
        # what zero weights mean; the trained values go back afterwards rather than a
        # second module being built, so the two arms differ in 33 numbers and nothing
        # else.
        learned = {key: value.clone() for key, value in module.gate.state_dict().items()}
        for arm in ("stock", "addressed", "static", "gated"):
            strength = 0.0 if arm in ("stock", "addressed") else static_strength
            with torch.no_grad():
                if arm == "static":
                    module.gate.weight.zero_()
                    module.gate.bias.zero_()
                elif arm == "gated":
                    module.gate.load_state_dict(learned)
            entry[arm] = measure(model, module, hasher, documents, args.device,
                                 structural, whitespace, classes,
                                 "gated" if arm in ("static", "gated") else arm,
                                 strength)
        entry["gated_wrong_context"] = measure(
            model, module, hasher, documents, args.device, structural, whitespace,
            classes, "gated", static_strength, wrong=True)
        for strength in (args.sweep or ()):
            entry["gated@%.2f" % strength] = measure(
                model, module, hasher, documents, args.device, structural, whitespace,
                classes, "gated", strength)
        report[split] = entry
        base = entry["stock"]["per_document_mean"]
        for arm in (["addressed", "static", "gated", "gated_wrong_context"]
                    + ["gated@%.2f" % strength for strength in (args.sweep or ())]):
            means = entry[arm]["per_document_mean"]
            print("%-12s %-20s %s" % (split, arm, "  ".join(
                "%s %+.6f" % (label, means.get(label, 0.0) - base.get(label, 0.0))
                for label in ("content", "newline", "whitespace", "punctuation",
                              "control", "all"))), flush=True)

    if digest_of(module.addressed) != before["addressed"]:
        raise SystemExit("the addressed decoder changed")
    if not args.refit_bias and digest_of(module.white) != before["white"]:
        raise SystemExit("the whitespace bias changed")
    if args.load_gate is not None and args.refit_bias:
        loaded = torch.load(args.load_gate, map_location=args.device,
                            weights_only=False)["state_dict"]
        for key, value in loaded.items():
            if not torch.equal(module.gate.state_dict()[key], value.to(args.device)):
                raise SystemExit("the loaded gate changed while the values were refitted")
    report["frozen_digests"] = before

    report["elapsed_seconds"] = time.monotonic() - started
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2),
                                             encoding="utf-8")
    torch.save({"state_dict": {key: value.cpu()
                               for key, value in module.gate.state_dict().items()},
                "features": module.addressed.code_dim}, args.output / "gate.pt")
    if args.refit_bias:
        torch.save({"state_dict": {key: value.cpu()
                                   for key, value in module.white.state_dict().items()},
                    "whitespace_strength": static_strength},
                   args.output / "whitespace.pt")
    print("wrote %s in %.0f s" % (args.output, report["elapsed_seconds"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

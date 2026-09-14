"""Was "structural" two mechanisms wearing one label?

Two independent controls said so. The unaddressed baseline kept 83% of the whitespace
gain and 12% of newline and punctuation; cross-backbone transfer kept 16% of whitespace
and 89% and 93% of the other two. If whitespace is calibration rather than context, then
485 free scalars should recover it and the hash should be relieved of it entirely.

So: the already-trained table-free sidecar is reused untouched, its whitespace outputs
masked to zero, and one learned bias per whitespace token id trained in their place. The
addressed decoder never takes a gradient here, and neither does the backbone -- both are
verified bitwise unchanged rather than assumed to be.

Five arms, because the decomposition is the point and an aggregate would hide it::

    stock          the frozen backbone
    current        the monolithic sidecar as it stands today
    addressed      its output with whitespace masked off, nothing replacing it
    whitespace     the bias alone, no addressed branch at all
    factorized     both

    CUDA_VISIBLE_DEVICES=0 python scratch/structural_sidecar/factorize.py \
        --output scratch/structural_sidecar/factorized-B42
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
from fit import BACKBONES, BASE, corpus
from repeatability import DEFAULT_BUNDLE, held_out_digests

STRENGTHS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)


def class_ids(classes, label, device=None):
    ids = [index for index, value in enumerate(classes) if value == label]
    if not ids:
        raise ValueError("no %s tokens" % label)
    return torch.tensor(ids, dtype=torch.long, device=device)


@torch.no_grad()
def logits_of(model, ids, device):
    tokens = torch.tensor([ids], device=device)
    return model(input_ids=tokens,
                 attention_mask=torch.ones_like(tokens)).logits[0, :-1].float(), tokens


def score(model, module, hasher, documents, device, structural, classes, arm,
          addressed_strength, whitespace_strength, wrong=False, mass_ids=None):
    """Per-class per-document NLL for one arm, and the structural probability mass.

    ``arm`` selects which halves contribute, so all five combinations come from one
    implementation rather than five slightly different ones.
    """
    per_document = {}
    totals = {}
    mass = []
    content_id = None
    with torch.no_grad():
        for ids in documents:
            logits, tokens = logits_of(model, ids, device)
            if arm != "stock":
                rows = hasher.row_indices(tokens)
                if wrong:
                    rows = wrong_context_rows(rows)
                if arm == "current":
                    bias = module.addressed(rows)[0, :-1]
                    strength = addressed_strength
                elif arm == "addressed":
                    bias = module(rows, 0.0)[0, :-1]
                    strength = addressed_strength
                elif arm == "whitespace":
                    bias = (module(rows, whitespace_strength)
                            - module.addressed(rows) * module.keep)[0, :-1]
                    strength = 1.0
                else:
                    bias = module(rows, whitespace_strength / addressed_strength
                                  if addressed_strength else 0.0)[0, :-1]
                    strength = addressed_strength
                logits = apply_structural_bias(logits.unsqueeze(0), bias.unsqueeze(0),
                                               structural, strength)[0]
            targets = torch.tensor(ids[1:], device=device)
            nll = F.cross_entropy(logits, targets, reduction="none").cpu()
            if mass_ids is not None:
                probabilities = torch.softmax(logits, dim=-1)
                content = torch.tensor([classes[token] == "content" for token in ids[1:]],
                                       device=device)
                if content.any():
                    mass.append(float(
                        probabilities[content][:, structural].sum(-1).mean()))
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
                      for label, bucket in sorted(totals.items())}}
    if mass:
        out["structural_mass_on_content"] = statistics.fmean(mass)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", default="B42", choices=sorted(BACKBONES))
    parser.add_argument("--addressed",
                        default=Path("scratch/structural_sidecar/direct-B42/sidecar.pt"),
                        type=Path)
    parser.add_argument("--load-whitespace", type=Path, default=None,
                        help="a fitted whitespace bias to reuse, for the transfer test")
    parser.add_argument("--fix-whitespace-strength", type=float, default=None,
                        help="report this strength instead of calibrating one; for "
                             "walking the frontier the two guardrail definitions "
                             "disagree about, never for choosing the reported point")
    parser.add_argument("--documents", type=int, default=512)
    parser.add_argument("--calibration-documents", type=int, default=128)
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
    addressed.requires_grad_(False)
    before = hashlib.sha256()
    for name, parameter in sorted(addressed.named_parameters()):
        before.update(name.encode("utf-8"))
        before.update(parameter.detach().cpu().numpy().tobytes())
    addressed_digest = before.hexdigest()
    addressed_strength = saved["strength"]

    module = FactorizedSidecar(addressed, structural.cpu(),
                               whitespace.cpu()).to(args.device)
    hasher = NGramHasher(NGramHashConfig(
        vocab_size=config.vocab_size, ngram_size=3, heads_per_ngram=1,
        ngram_vocab_size_base=saved["rows"] // 2 - 64, seed=1234,
        eos_token_id=tokenizer.eos_token_id or config.vocab_size - 1))
    if hasher.padded_vocab_size != saved["rows"]:
        raise SystemExit("hash geometry does not match the saved sidecar")

    report = {"backbone": args.backbone, "addressed": str(args.addressed),
              "addressed_strength": addressed_strength,
              "parameters": module.parameter_report()}
    print(json.dumps(report["parameters"]), flush=True)

    excluded = held_out_digests(args.bundle)
    train = corpus(tokenizer, excluded, args.documents, args.length, skip=0)
    calibration = corpus(tokenizer, excluded, args.calibration_documents, args.length,
                         skip=args.documents)

    if args.load_whitespace is not None:
        payload = torch.load(args.load_whitespace, map_location=args.device,
                             weights_only=False)
        module.white.load_state_dict(payload["state_dict"])
        report["loaded_whitespace"] = str(args.load_whitespace)
    else:
        # Only the bias trains. The addressed branch is in the graph but frozen, and the
        # guardrail keeps every class it owns from being paid for by this one.
        optimizer = torch.optim.AdamW(module.white.parameters(), lr=args.lr)
        white_mask = torch.zeros(config.vocab_size, dtype=torch.bool)
        white_mask[whitespace.cpu()] = True
        white_mask = white_mask.to(args.device)
        history = []
        for epoch in range(args.epochs):
            losses = []
            for ids in train:
                logits, tokens = logits_of(model, ids, args.device)
                targets = torch.tensor(ids[1:], device=args.device)
                rows = hasher.row_indices(tokens)
                with torch.no_grad():
                    reference = apply_structural_bias(
                        logits.unsqueeze(0), module(rows, 0.0)[0, :-1].unsqueeze(0),
                        structural, addressed_strength)[0]
                    base = F.cross_entropy(reference, targets, reduction="none")
                bias = module(rows, 1.0 / addressed_strength)[0, :-1]
                biased = apply_structural_bias(logits.unsqueeze(0), bias.unsqueeze(0),
                                               structural, addressed_strength)[0]
                nll = F.cross_entropy(biased, targets, reduction="none")
                mask = white_mask[targets]
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

    # The guardrail is defined against the stock backbone, as it was for the monolithic
    # sidecar. Measuring it against the addressed-only arm instead would hold the
    # whitespace branch to a stricter standard than the module it is being compared with,
    # and would select a strength the comparison never asked for.
    stock = score(model, module, hasher, calibration, args.device, structural, classes,
                  "stock", addressed_strength, 0.0)["per_document_mean"]
    addressed_only = score(model, module, hasher, calibration, args.device, structural,
                           classes, "addressed", addressed_strength,
                           0.0)["per_document_mean"]
    curve = {}
    for strength in STRENGTHS:
        means = score(model, module, hasher, calibration, args.device, structural,
                      classes, "factorized", addressed_strength,
                      strength)["per_document_mean"]
        curve[strength] = means
        print("calibration lambda_w %.2f  whitespace %+.6f  content %+.6f  newline %+.6f"
              % (strength, means["whitespace"] - stock["whitespace"],
                 means["content"] - stock["content"],
                 means["newline"] - stock["newline"]), flush=True)
    report["calibration_curve"] = {str(key): value for key, value in curve.items()}
    report["calibration_addressed_only"] = addressed_only
    report["calibration_stock"] = stock

    guardrail = 0.001
    admissible = [strength for strength in STRENGTHS
                  if curve[strength]["content"] - stock["content"] <= guardrail]
    chosen = min(admissible,
                 key=lambda strength: curve[strength]["whitespace"] - stock["whitespace"])
    if args.fix_whitespace_strength is not None:
        chosen = args.fix_whitespace_strength
        report["strength_was_fixed"] = True
    report["whitespace_strength"] = chosen
    print("selected lambda_w %.2f" % chosen, flush=True)

    for split in ("screen", "confirmation"):
        documents = [record["ids"] for record in load_records(args.bundle, split, 0)]
        entry = {}
        for arm in ("stock", "current", "addressed", "whitespace", "factorized"):
            entry[arm] = score(model, module, hasher, documents, args.device, structural,
                               classes, arm, addressed_strength, chosen, mass_ids=True)
        entry["factorized_wrong_context"] = score(
            model, module, hasher, documents, args.device, structural, classes,
            "factorized", addressed_strength, chosen, wrong=True)
        report[split] = entry
        base = entry["stock"]["per_document_mean"]
        for arm in ("current", "addressed", "whitespace", "factorized",
                    "factorized_wrong_context"):
            means = entry[arm]["per_document_mean"]
            print("%-12s %-26s %s" % (split, arm, "  ".join(
                "%s %+.6f" % (label, means.get(label, 0.0) - base.get(label, 0.0))
                for label in ("content", "newline", "whitespace", "punctuation",
                              "control", "all"))), flush=True)

    after = hashlib.sha256()
    for name, parameter in sorted(module.addressed.named_parameters()):
        after.update(name.encode("utf-8"))
        after.update(parameter.detach().cpu().numpy().tobytes())
    if after.hexdigest() != addressed_digest:
        raise SystemExit("the addressed decoder changed while the bias was fitted")
    report["addressed_sha256"] = addressed_digest

    report["elapsed_seconds"] = time.monotonic() - started
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2),
                                             encoding="utf-8")
    torch.save({"state_dict": {key: value.cpu()
                               for key, value in module.white.state_dict().items()},
                "whitespace_strength": chosen,
                "addressed": str(args.addressed)},
               args.output / "whitespace.pt")
    print("wrote %s in %.0f s" % (args.output, report["elapsed_seconds"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Arm A against arm B: does learned admission survive a backbone trained alongside it?

The frozen stage measured a gate against a model that never adapted to it. That is
mechanistic evidence and nothing more -- the PLE screens made exactly that mistake
available and it cost a whole programme to rule out. Once the backbone can train, the
same checkpoint with the gate switched off measures how far this model has come to lean
on its gate, not what the gate was worth. Only a backbone trained from the same starting
point, on the same documents in the same order, without a gate, answers that.

So this scores three things per milestone and pairs them by document:

    A gated      the co-adapted backbone with its co-adapted gate
    A at g = 1   the same weights, admission forced back to unit strength
    B stock      the separately trained counterfactual

``A - B`` on content NLL is the architecture decision variable. ``A gated - A at g = 1``
is a different question -- whether the model actively uses the gate at inference -- and
the two are reported separately because conflating them is the failure mode this whole
design exists to avoid.

Arms are scored one at a time and their per-document class means retained, so the pairing
is exact without two 2B models resident at once.

    CUDA_VISIBLE_DEVICES=0 python scratch/residual_gate/coadapt_score.py \
        --arm-a D:/.../runs/gate-coadapt-armA --arm-b D:/.../runs/gate-coadapt-armB \
        --output scratch/residual_gate/coadapt.json
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ffn_memo"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ffn_attenuation"))

import torch
import torch.nn.functional as F

from attenuate import load_records
from distillkit.residual_gate import (TrigramFamiliarity, family_features,
                                      install_residual_gates, remove_residual_gates)
from evaluate import DEFAULT_CACHE, split_layout
from repeatability import DEFAULT_BUNDLE

CHECKPOINT = re.compile(r"checkpoint-(\d+)$")


def milestones(run: Path):
    found = []
    for path in run.iterdir():
        match = CHECKPOINT.search(path.name)
        if match and path.is_dir():
            found.append((int(match.group(1)), path))
    if not found:
        raise SystemExit("no checkpoints under %s" % run)
    return sorted(found)


def paired_difference(left, right):
    """Per-document A - B, for every class both arms scored."""
    out = {}
    for label in sorted(set(left) & set(right)):
        values = [a - b for a, b in zip(left[label], right[label])]
        mean = statistics.fmean(values)
        error = (statistics.stdev(values) / len(values) ** 0.5
                 if len(values) > 1 else 0.0)
        out[label] = {"mean": mean, "stderr": error,
                      "t": mean / error if error else 0.0,
                      "ci95": [mean - 1.96 * error, mean + 1.96 * error],
                      "better": sum(1 for value in values if value < 0),
                      "n": len(values)}
    return out


@torch.inference_mode()
def per_document(model, records, device, classes):
    """Mean NLL per class per document, which is what the pairing needs."""
    out = {}
    for record in records:
        ids = record["ids"]
        tokens = torch.tensor([ids], device=device)
        logits = model(input_ids=tokens,
                       attention_mask=torch.ones_like(tokens)).logits[0, :-1].float()
        targets = ids[1:]
        nll = F.cross_entropy(logits, torch.tensor(targets, device=device),
                              reduction="none").cpu()
        sums = {}
        for index, token in enumerate(targets):
            entry = sums.setdefault(classes[token], [0.0, 0])
            entry[0] += float(nll[index])
            entry[1] += 1
        total = float(nll.sum())
        sums["all"] = [total, len(targets)]
        for label, (value, count) in sums.items():
            out.setdefault(label, []).append(value / count)
    return out


def load_backbone(path, device):
    from transformers import AutoConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    config = AutoConfig.from_pretrained(path, local_files_only=True)
    config = getattr(config, "text_config", config)
    config.use_cache = False
    model = Qwen3_5ForCausalLM.from_pretrained(
        path, config=config, dtype=torch.bfloat16,
        local_files_only=True).to(device).eval()
    return model, config


def release(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm-a", required=True, type=Path)
    parser.add_argument("--arm-b", required=True, type=Path)
    parser.add_argument("--tokenizer",
                        default="D:/DeepThought/Projects/HybridModel/student-2b-hf")
    parser.add_argument("--bundle", default=DEFAULT_BUNDLE)
    parser.add_argument("--split", default="screen")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if torch.cuda.device_count() != 1:
        raise SystemExit("mask to one card with CUDA_VISIBLE_DEVICES")

    from transformers import AutoConfig, AutoTokenizer

    from distillkit.independent_eval import build_token_classes

    started = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    vocab = getattr(AutoConfig.from_pretrained(args.tokenizer, local_files_only=True),
                    "text_config", None)
    vocab_size = (vocab or AutoConfig.from_pretrained(
        args.tokenizer, local_files_only=True)).vocab_size
    classes = split_layout(build_token_classes(tokenizer, vocab_size), tokenizer,
                           vocab_size)
    records = load_records(args.bundle, args.split, args.limit)

    steps_a = dict(milestones(args.arm_a))
    steps_b = dict(milestones(args.arm_b))
    shared = sorted(set(steps_a) & set(steps_b))
    if not shared:
        raise SystemExit("the arms share no milestone; they are not comparable")

    report = {"arm_a": str(args.arm_a), "arm_b": str(args.arm_b),
              "split": args.split, "documents": len(records), "milestones": []}
    for step in shared:
        gate_path = args.arm_a / ("gate-step%d.pt" % step)
        if not gate_path.exists():
            raise SystemExit("arm A checkpoint %d has no gate beside it" % step)
        payload = torch.load(gate_path, map_location="cpu", weights_only=False)

        model, config = load_backbone(steps_a[step], args.device)
        familiarity = (TrigramFamiliarity(args.cache, config.vocab_size)
                       if "log_count" in family_features(payload["family"]) else None)
        handle = install_residual_gates(model, payload["layers"],
                                        family=payload["family"],
                                        familiarity=familiarity)
        model.residual_gates.load_state_dict(payload["state_dict"], strict=True)
        gated = per_document(model, records, args.device, classes)
        handle.force_identity = True
        identity = per_document(model, records, args.device, classes)
        remove_residual_gates(model)
        release(model)

        model, _ = load_backbone(steps_b[step], args.device)
        stock = per_document(model, records, args.device, classes)
        release(model)

        entry = {"step": step,
                 "a_minus_b": paired_difference(gated, stock),
                 "a_gated_minus_a_identity": paired_difference(gated, identity),
                 "a_content_nll": statistics.fmean(gated["content"]),
                 "b_content_nll": statistics.fmean(stock["content"]),
                 "a_identity_content_nll": statistics.fmean(identity["content"])}
        report["milestones"].append(entry)
        difference = entry["a_minus_b"]["content"]
        print("step %-4d  A %.5f  B %.5f  A-B %+.6f (t %+6.2f, %d/%d)  "
              "A gated-identity %+.6f"
              % (step, entry["a_content_nll"], entry["b_content_nll"],
                 difference["mean"], difference["t"], difference["better"],
                 difference["n"],
                 entry["a_gated_minus_a_identity"]["content"]["mean"]), flush=True)

    report["elapsed_seconds"] = time.monotonic() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("wrote %s in %.0f s" % (args.output, report["elapsed_seconds"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

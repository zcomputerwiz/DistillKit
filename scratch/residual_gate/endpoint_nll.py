"""Absolute content NLL of one co-adapted checkpoint, for comparing across repeats.

The paired-document statistics in ``coadapt_score.py`` answer "is this checkpoint better
than that one on these documents", with a t over documents. They cannot answer "is this
arm better than that arm", because the thing that varies between arms also varies between
two runs of the same arm: three runs of one seed-43 configuration, same stream digest,
landed 0.0033 nats apart on A - B.

Answering the arm question needs repeats, and repeats need an observation per run rather
than a difference per pair. Evaluation is deterministic, so absolute NLL on a fixed corpus
is directly comparable between runs -- one forward pass, no second model resident, no
pairing. The spread across repeats of one arm is then the yardstick every between-arm
difference has to clear.

    CUDA_VISIBLE_DEVICES=0 python scratch/residual_gate/endpoint_nll.py \
        --run D:/.../runs/gate-coadapt-armC --step 284 --output ...
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

import torch

from attenuate import load_records
from coadapt_score import load_backbone, per_document
from distillkit.experimental.residual_gate import (
    TrigramFamiliarity, family_features, install_residual_gates,
    remove_residual_gates)
from evaluate import DEFAULT_CACHE, split_layout
from repeatability import DEFAULT_BUNDLE


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--step", type=int, default=284)
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
    base = AutoConfig.from_pretrained(args.tokenizer, local_files_only=True)
    vocab_size = getattr(base, "text_config", base).vocab_size
    classes = split_layout(build_token_classes(tokenizer, vocab_size), tokenizer,
                           vocab_size)
    records = load_records(args.bundle, args.split, args.limit)

    checkpoint = args.run / ("checkpoint-%d" % args.step)
    if not checkpoint.is_dir():
        raise SystemExit("no checkpoint-%d under %s" % (args.step, args.run))
    model, config = load_backbone(checkpoint, args.device)
    gate_path = args.run / ("gate-step%d.pt" % args.step)
    handle = None
    if gate_path.exists():
        payload = torch.load(gate_path, map_location="cpu", weights_only=False)
        source = (TrigramFamiliarity(args.cache, config.vocab_size)
                  if "log_count" in family_features(payload["family"]) else None)
        handle = install_residual_gates(model, payload["layers"],
                                        family=payload["family"], familiarity=source)
        model.residual_gates.load_state_dict(payload["state_dict"], strict=True)

    values, _ = per_document(model, records, args.device, classes, handle)
    report = {"run": str(args.run), "step": args.step, "split": args.split,
              "documents": len(records), "gated": handle is not None,
              "nll": {label: statistics.fmean(series)
                      for label, series in sorted(values.items())},
              "per_document_content": values["content"]}
    if handle is not None:
        handle.force_identity = True
        identity, _ = per_document(model, records, args.device, classes, handle)
        report["identity_nll"] = {label: statistics.fmean(series)
                                  for label, series in sorted(identity.items())}
        report["per_document_content_identity"] = identity["content"]
        remove_residual_gates(model)

    report["elapsed_seconds"] = time.monotonic() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("%s step %d  content %.6f%s"
          % (args.run.name, args.step, report["nll"]["content"],
             "" if handle is None else "  (g=1 %.6f)"
             % report["identity_nll"]["content"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Reload an exported checkpoint and rescore the trainer's exact held-out sample."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from benchmark import apply_liger
from cut_cross_entropy import linear_cross_entropy
from distillkit.models import Qwen35WidenedForCausalLM
from teacher_kl import CachedTeacher


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=.002)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite an existing probe")
    report = json.loads(args.report.read_text())
    model, loading = Qwen35WidenedForCausalLM.from_pretrained(
        args.checkpoint, dtype=torch.bfloat16, attn_implementation="flash_attention_2",
        local_files_only=True, output_loading_info=True)
    if loading["missing_keys"] or loading["unexpected_keys"] or loading.get("mismatched_keys"):
        raise ValueError("checkpoint did not reload strictly: %r" % loading)
    model = model.to("cuda").eval()
    apply_liger(model, model.config)
    # No reselection: score the exact sample recorded by the training process.
    held = CachedTeacher(report["run_args"]["teacher_cache"], "eval", device="cuda",
                         max_length=report["run_args"]["teacher_max_length"])
    members = set(held.ids)
    by_source = {}
    records = []
    with torch.inference_mode():
        for source, doc_id in report["heldout_sample"]:
            if doc_id not in members:
                raise ValueError("saved evaluation document is not in the eval split")
            ids = held.read(doc_id)["input_ids"]
            hidden = model.model(input_ids=ids, attention_mask=torch.ones_like(ids),
                                 use_cache=False).last_hidden_state
            nll = float(linear_cross_entropy(hidden, model.lm_head.weight, ids,
                                            shift=1, reduction="mean"))
            count = ids.numel() - ids.shape[0]
            stats = by_source.setdefault(source, [0., 0])
            stats[0] += nll * count
            stats[1] += count
            records.append(dict(source=source, doc_id=doc_id, targets=count, nll=nll))
    held.cache.close()
    nll = sum(s[0] for s in by_source.values()) / sum(s[1] for s in by_source.values())
    source_nll = {name: s[0] / s[1] for name, s in by_source.items()}
    before = report["history"][-1]
    deltas = {name: value - before["heldout_by_source"][name] for name, value in source_nll.items()}
    passed = abs(nll - before["heldout"]) <= args.tolerance and all(
        abs(delta) <= args.tolerance for delta in deltas.values())
    result = dict(checkpoint=str(args.checkpoint), training_report=str(args.report),
                  same_sample=True, exported_layout="unsharded", training_layout="tensor_parallel",
                  targets=sum(s[1] for s in by_source.values()), records=records,
                  before_nll=before["heldout"], reloaded_nll=nll,
                  nll_delta=nll - before["heldout"], by_source=source_nll,
                  source_deltas=deltas, tolerance=args.tolerance, passed=passed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "records"}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

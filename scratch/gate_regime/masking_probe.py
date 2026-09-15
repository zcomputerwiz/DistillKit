"""What does "assistant-masked" actually select, on each of the two corpora?

The 2x2 factorial crosses corpus with loss/mask, and one of its four cells may not exist.
``assistant_token_mask`` documents that plain text without chat role markers "follows
role_spans (assistant)" -- so on the memoisation corpus, which is raw documents with no
roles, the mask may select everything, making memo + assistant-masked identical to
memo + plain by construction rather than by coincidence.

That is a claim about the code, so it gets measured before four arms are launched on the
assumption that they are four. This reports the masked fraction on both corpora and, if
the degeneracy is real, says so with the number rather than quietly running a duplicate.

    CUDA_VISIBLE_DEVICES=0 python scratch/gate_regime/masking_probe.py
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ffn_memo"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ffn_attenuation"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "structural_sidecar"))

import torch

from distillkit.lossfuncs.cross_entropy import assistant_token_mask
from fit import BASE, corpus
from repeatability import DEFAULT_BUNDLE, held_out_digests

CACHE = Path("D:/DeepThought/Projects/HybridModel/teacher-cache-5m")


def teacher_documents(count: int, length: int):
    """Token rows straight from the cache the original gate was trained on."""
    from distillkit.offline_cache import OfflineTeacherCache

    cache = OfflineTeacherCache(str(CACHE))
    rows = []
    for record in cache.iter_records(split="train"):
        ids = list(record["input_ids"])[:length]
        if len(ids) >= 16:
            rows.append(ids)
        if len(rows) >= count:
            break
    cache.close()
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", type=int, default=24)
    parser.add_argument("--length", type=int, default=512)
    parser.add_argument("--bundle", default=DEFAULT_BUNDLE)
    parser.add_argument("--output", type=Path,
                        default=Path("scratch/gate_regime/masking.json"))
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
    excluded = held_out_digests(args.bundle)
    sources = {
        "memoisation": corpus(tokenizer, excluded, args.documents, args.length, skip=0),
        "teacher_cache": teacher_documents(args.documents, args.length),
    }

    report = {}
    for name, documents in sources.items():
        fractions = []
        for ids in documents:
            tokens = torch.tensor([ids])
            mask = assistant_token_mask(tokens, torch.ones_like(tokens), tokenizer)
            fractions.append(float(mask.float().mean()))
        report[name] = {
            "documents": len(documents),
            "mean_masked_fraction": statistics.fmean(fractions),
            "min": min(fractions), "max": max(fractions),
            "all_selected": all(value == 1.0 for value in fractions),
        }
        print("%-14s masked fraction mean %.4f  min %.4f  max %.4f  everything=%s"
              % (name, report[name]["mean_masked_fraction"], report[name]["min"],
                 report[name]["max"], report[name]["all_selected"]), flush=True)
        sample = tokenizer.decode(documents[0][:80], skip_special_tokens=False)
        report[name]["sample"] = sample
        print("   %s" % sample.replace("\n", "\\n")[:150], flush=True)

    report["degenerate_cell"] = report["memoisation"]["all_selected"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if report["degenerate_cell"]:
        print("\nmemo + assistant-masked is memo + plain by construction: the factorial "
              "has three distinct cells, not four.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""NLL on general text, which no other measurement here looks at.

Every NLL in this programme is scored on documents from the same chat-SFT capture
pipeline the model is trained on, so it rewards adaptation to that distribution: the
repaired model is 0.58 nats better than its source there, which says little about how
good a language model it is. This scores fixed 1024-token windows of WikiText-103 test --
encyclopedic prose none of these models were trained on here -- the same windows for
every checkpoint, paired, so a converted or trained model can be compared with its
source as a language model rather than as a chat model.

    python scratch/dense_gr/general_nll.py --arm source=../student-2b-hf \
        --arm repaired=scratch/dense_gr/checkpoints-2b-repair/smoke-r1-1-gr-s0-csa2 \
        --reference source --output general-nll.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

TOKENIZER = "D:/DeepThought/Projects/HybridModel/student-2b-hf"


def load(path):
    config = json.loads((Path(path) / "config.json").read_text(encoding="utf-8"))
    if "Qwen35WidenedForCausalLM" in config.get("architectures", []):
        from distillkit.models import Qwen35WidenedForCausalLM as cls
    else:
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM as cls
    return cls.from_pretrained(path, dtype=torch.bfloat16, local_files_only=True).to("cuda").eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="append", required=True, metavar="LABEL=PATH")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--windows", type=int, default=64)
    parser.add_argument("--length", type=int, default=1024)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    arms = dict(item.split("=", 1) for item in args.arm)

    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    text = "".join(load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test")["text"])
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    # Evenly spaced, non-overlapping windows across the whole test set, so the sample is
    # the same for every arm and is not concentrated in one article.
    stride = (len(ids) - args.length) // args.windows
    windows = [torch.tensor(ids[i * stride:i * stride + args.length]) for i in range(args.windows)]
    print("%d test tokens, %d windows of %d" % (len(ids), len(windows), args.length), flush=True)

    scores = {}
    for label, path in arms.items():
        model = load(path)
        values = []
        with torch.inference_mode():
            for window in windows:
                window = window.unsqueeze(0).to("cuda")
                logits = model(input_ids=window, use_cache=False).logits.float()
                values.append(float(F.cross_entropy(logits[0, :-1], window[0, 1:])))
        scores[label] = np.array(values)
        print("%-12s nll %.4f" % (label, scores[label].mean()), flush=True)
        del model
        torch.cuda.empty_cache()

    base = scores[args.reference]
    draws = np.random.default_rng(0).integers(0, len(windows), size=(10000, len(windows)))
    rows = {}
    print("\n%-12s %8s %10s %24s" % ("arm", "nll", "vs " + args.reference, "95% CI"))
    for label, values in scores.items():
        diff = values - base
        low, high = np.percentile(diff[draws].mean(1), [2.5, 97.5])
        rows[label] = dict(nll=float(values.mean()), vs_reference=float(diff.mean()),
                           ci95=[float(low), float(high)])
        print("%-12s %8.4f %+10.4f   [%+.4f, %+.4f]" % (label, values.mean(), diff.mean(), low, high))
    args.output.write_text(json.dumps(dict(dataset="wikitext-103-raw-v1 test", windows=len(windows),
                                           length=args.length, arms=arms, rows=rows,
                                           per_window={k: v.tolist() for k, v in scores.items()}),
                                      indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

"""Section 33: has learning Python cost the backbone its general language ability?

Not the primary metric, and not a number comparable to any earlier general-domain result
in this programme -- those scored assistant spans under a chat mask, and this is plain
causal CE over every target, to match the Python measurement exactly. What it is good for
is the only thing it is asked to do: the same measurement on B0 and on B_code, so a
catastrophic over-specialization would be visible rather than inferred.

The documents are the established general-domain held-out capture, a fixed prefix of it
selected by position so the comparison is over identical text.

    CUDA_VISIBLE_DEVICES=0 python scratch/code_training/general_retention.py \\
        --backbone D:/DeepThought/Projects/HybridModel/student-2b-hf --label B0 \\
        --output scratch/code_training/post_eval/general-b0.json
"""

from __future__ import annotations

import argparse
import collections
import io
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "residual_gate"))

import numpy as np
import torch
import torch.nn.functional as F

from distillkit.code_classes import HISTORICAL, HISTORICAL_CLASSES
from corpus import BASE, load_tokenizer

SOURCE = Path("D:/DeepThought/Projects/HybridModel/capture-data/heldout.jsonl")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--documents", type=int, default=400)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-batch-tokens", type=int, default=16384)
    parser.add_argument("--head-positions", type=int, default=4096)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    from transformers import AutoConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    from distillkit.independent_eval import build_token_classes
    from evaluate import split_layout

    config = AutoConfig.from_pretrained(args.backbone, local_files_only=True)
    config = getattr(config, "text_config", config)
    config.use_cache = False
    model = Qwen3_5ForCausalLM.from_pretrained(
        args.backbone, config=config, dtype=torch.bfloat16,
        local_files_only=True).to(args.device).eval()
    model.requires_grad_(False)

    tokenizer = load_tokenizer()
    names = split_layout(build_token_classes(tokenizer, config.vocab_size), tokenizer,
                         config.vocab_size)
    index = {name: i for i, name in enumerate(HISTORICAL_CLASSES)}
    # The historical builder's classes are already the five-way view; map straight over.
    classes = np.fromiter((index[n] for n in names), dtype=np.int16,
                          count=config.vocab_size)

    documents = []
    with io.open(args.source, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            ids = tokenizer(row["text"], add_special_tokens=False)["input_ids"]
            documents.append(np.asarray(ids[:args.max_length], dtype=np.int64))
            if len(documents) >= args.documents:
                break

    totals = np.zeros((len(HISTORICAL_CLASSES), 2), dtype=np.float64)
    scored = 0
    started = time.monotonic()
    order = sorted(range(len(documents)), key=lambda i: -len(documents[i]))
    with torch.inference_mode():
        position = 0
        while position < len(order):
            group, widest = [], 0
            while position < len(order):
                candidate = max(widest, len(documents[order[position]]))
                if group and candidate * (len(group) + 1) > args.max_batch_tokens:
                    break
                group.append(order[position])
                widest = candidate
                position += 1
            rows = [documents[i] for i in group]
            width = max(len(r) for r in rows)
            ids = torch.full((len(rows), width), config.eos_token_id, dtype=torch.long,
                             device=args.device)
            mask = torch.zeros((len(rows), width), dtype=torch.long, device=args.device)
            for slot, row in enumerate(rows):
                ids[slot, :len(row)] = torch.from_numpy(row).to(args.device)
                mask[slot, :len(row)] = 1
            hidden = model.model(input_ids=ids, attention_mask=mask).last_hidden_state
            nll = torch.zeros((len(rows), width - 1), dtype=torch.float32,
                              device=args.device)
            chunk = max(1, args.head_positions // len(rows))
            for start in range(0, width - 1, chunk):
                stop = min(start + chunk, width - 1)
                logits = model.lm_head(hidden[:, start:stop]).float()
                nll[:, start:stop] = F.cross_entropy(
                    logits.transpose(1, 2), ids[:, start + 1:stop + 1], reduction="none")
            host = nll.to("cpu").numpy()
            for slot, row in enumerate(rows):
                length = len(row) - 1
                values = host[slot, :length].astype(np.float64)
                labels = classes[row[1:length + 1]]
                np.add.at(totals[:, 0], labels, values)
                np.add.at(totals[:, 1], labels, 1.0)
                scored += length

    report = {
        "backbone": str(args.backbone), "label": args.label,
        "source": str(args.source), "documents": len(documents),
        "objective": "plain causal CE over 100% of targets; NOT the historical "
                     "assistant-masked number, so comparable across checkpoints here "
                     "but not against earlier general-domain results",
        "tokens": scored,
        "mean_nll": float(totals[:, 0].sum() / max(scored, 1)),
        "classes": {name: {"nll": float(totals[i, 0] / totals[i, 1]) if totals[i, 1]
                           else None, "tokens": int(totals[i, 1])}
                    for i, name in enumerate(HISTORICAL_CLASSES)},
        "seconds": time.monotonic() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("%s: %d docs, %d tokens, mean NLL %.6f, content %.6f"
          % (args.label, len(documents), scored, report["mean_nll"],
             report["classes"]["content"]["nll"]))
    print("wrote %s" % args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

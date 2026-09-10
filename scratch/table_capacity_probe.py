"""How many nats can a linear read of the n-gram table add to this student's responses?

Every arm so far has measured the table through an objective whose optimum is a silent
sidecar: the teacher has no n-gram table, so both the KL and the hidden-state terms are
minimised by a student whose sidecar changes nothing. That says nothing about whether the
table *contains* anything useful, only that this loss cannot ask for it.

This asks directly, with ground-truth cross entropy and no teacher at all. The backbone
is frozen and the only trainable thing is one matrix::

    logits = lm_head(final_hidden + W(features))       # W: 2560 -> 2560, zero-init

At `W = 0` this is exactly the student's own next-token distribution, so the baseline is
free and exact. Whatever `W` then buys is what a *linear* read of the table can add on
top of everything the backbone already knows -- an upper bound on the simplest possible
sidecar, and a lower bound on what a gated non-linear one could reach.

The control is the same training against features rolled across documents. A linear map
onto 248,320 logits has enough freedom to fit something from any correlated input; the
control measures how much, so the real arm can be read against it rather than against
zero.

Scored on assistant tokens only, for the reason the response-only bundle exists: prompt
prediction dominates whole-window numbers and is not what the model is for.

`heldout.jsonl` is a *pool*, not a split -- all 5,598 training documents are inside its
7,200 -- so eligibility comes from the teacher-cache manifests exactly as
`independent_eval prepare` establishes it. Both slices here are drawn from the unseen
remainder and are disjoint from each other, so the eval documents are new to the
backbone and to the readout alike. The absolute NLL is still not comparable to the
reply-bundle arms: different documents, different lengths.

    python scratch/table_capacity_probe.py --output scratch/gpu-checks/table-capacity.json
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GGUF = os.path.join(
    os.path.expanduser("~"), ".cache", "huggingface", "hub",
    "models--unsloth--Qwen3.8-Flash-Next-GGUF", "snapshots",
    "38bb39ee97821de2c9009abb7e93950eec396e66", "UD-IQ4_XS",
    "Qwen3.8-Flash-Next-UD-IQ4_XS-00002-of-00003.gguf",
)
DEFAULT_STUDENT = str((ROOT / ".." / "student-hf").resolve())
DEFAULT_DOCUMENTS = str((ROOT / ".." / "capture-data" / "heldout.jsonl").resolve())
DEFAULT_MANIFESTS = [str((ROOT / ".." / name / "manifest.json").resolve())
                     for name in ("teacher-cache-1m", "teacher-cache-5m")]


class _PadCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        longest = max(len(f["input_ids"]) for f in features)
        ids, mask = [], []
        for feature in features:
            count = len(feature["input_ids"])
            ids.append(list(feature["input_ids"]) + [self.pad_token_id] * (longest - count))
            mask.append([1] * count + [0] * (longest - count))
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(mask)}


def assistant_mask(text, encoding, length):
    """True at positions whose *target* is an assistant token."""
    from distillkit.independent_eval import role_spans

    spans = role_spans(text, encoding["offset_mapping"]).get("assistant", [])
    mask = torch.zeros(length, dtype=torch.bool)
    for start, stop in spans:
        mask[max(start - 1, 0):min(stop - 1, length)] = True
    return mask


def chunked_ce(hidden, head, targets, mask, chunk=256, backward=False, scale=1.0):
    """Cross entropy in token chunks: the full 248,320-wide logits never exist at once."""
    total = torch.zeros((), device=hidden.device, dtype=torch.float32)
    flat_hidden = hidden.reshape(-1, hidden.shape[-1])
    flat_targets = targets.reshape(-1)
    flat_mask = mask.reshape(-1)
    counted = int(flat_mask.sum())
    for start in range(0, flat_hidden.shape[0], chunk):
        stop = min(start + chunk, flat_hidden.shape[0])
        piece = flat_mask[start:stop]
        if not piece.any():
            continue
        logits = head(flat_hidden[start:stop][piece]).float()
        loss = F.cross_entropy(logits, flat_targets[start:stop][piece], reduction="sum")
        if backward:
            (loss * scale).backward(retain_graph=True)
        total = total + loss.detach()
    return total, counted


def batches(documents, tokenizer, collator, table_batch, tokens):
    for start in range(0, len(documents), table_batch):
        chunk = documents[start:start + table_batch]
        encodings = [tokenizer(text, return_offsets_mapping=True) for text in chunk]
        batch = collator([{"input_ids": e["input_ids"][:tokens]} for e in encodings])
        length = batch["input_ids"].shape[1]
        masks = torch.stack([assistant_mask(text, encoding, length)
                             for text, encoding in zip(chunk, encodings)])
        yield batch, masks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf", default=DEFAULT_GGUF)
    parser.add_argument("--student", default=DEFAULT_STUDENT)
    parser.add_argument("--documents", default=DEFAULT_DOCUMENTS)
    parser.add_argument("--manifests", nargs="+", default=DEFAULT_MANIFESTS)
    parser.add_argument("--train-docs", type=int, default=512)
    parser.add_argument("--eval-docs", type=int, default=128)
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    timer = threading.Timer(5400, lambda: os._exit(124))
    timer.daemon = True
    timer.start()

    from transformers import AutoTokenizer

    from distillkit.models.qwen35_sidecar import Qwen35SidecarForCausalLM
    from distillkit.ngram_hash import NGramHasher
    from distillkit.ngram_table import GGUFNGramTable, IQ4NLDequant
    from distillkit.sidecar_collator import SidecarDataCollator

    tokenizer = AutoTokenizer.from_pretrained(args.student)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    from distillkit.independent_eval import unseen_records

    unseen = unseen_records(args.documents, args.manifests)
    if len(unseen) < args.train_docs + args.eval_docs:
        raise SystemExit("only %d unseen documents; asked for %d"
                         % (len(unseen), args.train_docs + args.eval_docs))
    # Eval taken from the far end so a smaller --train-docs does not silently move which
    # documents are scored, which would make two runs of this probe incomparable.
    train_docs = [record["text"] for record in unseen[:args.train_docs]]
    eval_docs = [record["text"] for record in unseen[-args.eval_docs:]]
    print("documents: %d train, %d eval, from %d unseen"
          % (len(train_docs), len(eval_docs), len(unseen)))

    table = GGUFNGramTable(args.gguf)
    collator = SidecarDataCollator(_PadCollator(pad_id), table, NGramHasher())
    dequant = IQ4NLDequant(out_dtype=torch.float32)

    device = torch.device(args.device)
    model = Qwen35SidecarForCausalLM.from_pretrained(args.student, dtype=torch.bfloat16)
    model.config.use_cache = False
    model.to(device).eval()
    model.requires_grad_(False)
    head = model.get_output_embeddings()
    hidden_size = model.config.hidden_size
    print("student on %s, hidden %d, vocab %d" % (device, hidden_size, model.config.vocab_size))

    def hidden_of(batch):
        with torch.no_grad():
            out = model.model(input_ids=batch["input_ids"].to(device),
                              attention_mask=batch["attention_mask"].to(device),
                              sidecar_enabled=False)
        return out.last_hidden_state

    def features_of(batch, roll):
        features = dequant(batch["ngram_raw"]).flatten(-2).to(device)
        return features.roll(1, dims=0) if roll else features

    def evaluate(readout, roll):
        total, counted = 0.0, 0
        for batch, masks in batches(eval_docs, tokenizer, collator, args.batch, args.tokens):
            hidden = hidden_of(batch)
            targets = batch["input_ids"][:, 1:].to(device)
            keep = masks[:, :-1].to(device) & batch["attention_mask"][:, 1:].bool().to(device)
            state = hidden[:, :-1]
            if readout is not None:
                state = state + readout(features_of(batch, roll)[:, :-1]).to(state.dtype)
            with torch.no_grad():
                loss, count = chunked_ce(state, head, targets, keep)
            total += float(loss)
            counted += count
        return total / max(counted, 1), counted

    results = {"train_docs": len(train_docs), "eval_docs": len(eval_docs),
               "tokens": args.tokens, "lr": args.lr, "epochs": args.epochs}
    baseline, eval_tokens = evaluate(None, False)
    results["baseline_nll"] = baseline
    results["eval_assistant_tokens"] = eval_tokens
    print("baseline assistant NLL %.4f over %d tokens" % (baseline, eval_tokens))

    for name, roll in (("table", False), ("shuffled_control", True)):
        torch.manual_seed(7)
        readout = nn.Linear(hidden_size, hidden_size, bias=False).to(device)
        nn.init.zeros_(readout.weight)
        readout.to(torch.float32)
        optimizer = torch.optim.AdamW(readout.parameters(), lr=args.lr)
        started = time.perf_counter()
        seen = 0
        for epoch in range(args.epochs):
            for index, (batch, masks) in enumerate(
                    batches(train_docs, tokenizer, collator, args.batch, args.tokens)):
                hidden = hidden_of(batch)
                targets = batch["input_ids"][:, 1:].to(device)
                keep = masks[:, :-1].to(device) & batch["attention_mask"][:, 1:].bool().to(device)
                count = int(keep.sum())
                if not count:
                    continue
                state = hidden[:, :-1] + readout(features_of(batch, roll)[:, :-1]).to(hidden.dtype)
                optimizer.zero_grad(set_to_none=True)
                loss, _ = chunked_ce(state, head, targets, keep, backward=True, scale=1.0 / count)
                optimizer.step()
                seen += count
                if index % 32 == 0:
                    print("  %s epoch %d batch %d  train NLL %.4f  %d tokens"
                          % (name, epoch, index, float(loss) / count, seen), flush=True)
        trained, _ = evaluate(readout, roll)
        results[name] = {
            "eval_nll": trained,
            "delta_vs_baseline": trained - baseline,
            "readout_norm": readout.weight.float().norm().item(),
            "train_tokens": seen,
            "seconds": time.perf_counter() - started,
        }
        print("%s: eval NLL %.4f  delta %+.4f  |W| %.3f"
              % (name, trained, trained - baseline, results[name]["readout_norm"]), flush=True)

    results["table_gain_over_control"] = (results["shuffled_control"]["delta_vs_baseline"]
                                          - results["table"]["delta_vs_baseline"])
    results["quality_evaluation"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2), flush=True)
    timer.cancel()


if __name__ == "__main__":
    main()

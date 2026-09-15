"""Continued pretraining of the stock backbone on Python, plain CE, nothing attached.

This produces B_code, the canonical stock code-specialized control. It is deliberately the
dullest possible training run: plain causal cross-entropy over every non-padding target,
no assistant masking, no teacher cache, no gate, no sidecar, no experimental module of any
kind. Everything interesting in this experiment happens in the two evaluations that
bracket it; a training run with anything clever in it would make those uninterpretable.

The stream is packed. Documents are concatenated in a deterministic shuffled order and cut
into fixed-length sequences, each document already carrying the EOS that the token store
appended, so a sequence boundary never presents the tail of one file as context for the
head of an unrelated one without a marker between them. Packing is what makes 30M tokens
affordable: padding to the longest document in a batch would waste most of the compute on
a corpus whose median file is 363 tokens and whose longest is 32,768.

Token accounting is exact and is the thing this run is checkpointed against. Milestones
are token counts, not step counts, because packing makes steps and tokens different
questions -- the last sequence of an epoch is short, and a schedule expressed in steps
would quietly deliver a different amount of data than it claims.

Loss is computed through a chunked head. The vocabulary is 248,320 wide, so materializing
logits for a whole packed batch is tens of gigabytes; chunking the head over positions
keeps peak memory flat and costs nothing but a Python loop.

    CUDA_VISIBLE_DEVICES=0 python scratch/code_training/train.py --run v1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import torch.nn.functional as F

from corpus import BASE, TOKENS, TokenStore, corpus_identity, load_tokenizer

RUNS = Path("scratch/code_training")
MILESTONES = (5_000_000, 10_000_000, 20_000_000, 30_725_434)


def write_atomic(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def packed_stream(store: TokenStore, sequence_length: int, seed: int):
    """Documents in a deterministic shuffled order, cut into fixed-length sequences.

    The shuffle is over documents rather than over packed sequences, so the order depends
    only on the corpus and the seed -- reshuffling after packing would make the stream
    depend on the sequence length as well, and two runs at different lengths would not be
    seeing the same data in a comparable order.

    The final partial sequence is dropped rather than padded. It is at most one sequence
    out of thousands, and dropping it keeps every optimizer step seeing exactly the same
    number of supervised targets, which is what makes the token accounting exact.
    """
    order = np.random.default_rng(seed).permutation(len(store))
    buffer = np.empty(0, dtype=np.int64)
    for index in order:
        buffer = np.concatenate([buffer, np.asarray(store.document(index),
                                                    dtype=np.int64)])
        while len(buffer) >= sequence_length:
            yield buffer[:sequence_length]
            buffer = buffer[sequence_length:]


def sha256_of_directory(path: Path) -> str:
    """Digest of a checkpoint's weight files, so an artifact can name what produced it."""
    hasher = hashlib.sha256()
    for file in sorted(path.glob("*.safetensors")):
        with open(file, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 22), b""):
                hasher.update(block)
    return hasher.hexdigest()


def build_optimizer(model, args):
    """8-bit AdamW, because the states do not otherwise fit on one 24 GB card.

    A 2B model under fp32 AdamW needs 16 GB of optimizer state alone, on top of 4 GB of
    bf16 weights and 4 GB of gradients -- 24 GB before a single activation, which is the
    whole card. ``bitsandbytes`` keeps the moments in 8 bits with blockwise dynamic
    quantization, which brings the state to 4 GB and the run within budget.

    This is a memory decision, not a quality one, and it is recorded in the provenance as
    such. It is well established for exactly this case (short domain adaptation at a
    conservative learning rate) and the alternative here is not fp32 AdamW, it is no run.
    """
    import bitsandbytes as bnb

    return bnb.optim.AdamW8bit(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                               eps=1e-8, weight_decay=args.weight_decay)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--backbone", default=BASE)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--micro-batch", type=int, default=2)
    parser.add_argument("--accumulate", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="conservative: this is 30M-token domain adaptation, not retraining")
    parser.add_argument("--warmup", type=int, default=40)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--head-positions", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--data-seed", type=int, default=20260914)
    parser.add_argument("--milestones", default=",".join(str(m) for m in MILESTONES))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--memory-fraction", type=float, default=0.92)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--no-checkpointing", action="store_true",
                        help="8-bit optimizer states may leave room to skip recompute")
    parser.add_argument("--resume-from", type=Path, default=None,
                        help="a milestone directory to continue from after an interruption")
    args = parser.parse_args()

    if torch.cuda.device_count() != 1:
        raise SystemExit("mask to one card with CUDA_VISIBLE_DEVICES")
    # Windows WDDM does not raise OOM on an oversized allocation, it pages to host RAM and
    # runs about seventy times slower. This turns that silent cliff back into an error.
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction, 0)
    torch.manual_seed(args.seed)

    from transformers import AutoConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    root = RUNS / "checkpoints" / args.run
    state_path = root / "state.json"
    milestones = sorted(int(m) for m in args.milestones.split(","))

    # Resuming continues the *same* stream rather than restarting it. The packed stream
    # is a pure function of the corpus and the data seed, so replaying and discarding the
    # sequences already consumed reproduces exactly what an uninterrupted run would have
    # seen next -- no document is trained on twice and none is skipped.
    #
    # What is genuinely lost is the AdamW moment estimates, which are not saved with the
    # weights. That is a real discontinuity and it is recorded as one; it lands in the low
    # tail of the cosine schedule, where the step sizes it affects are smallest.
    resume = None
    weights = args.backbone
    if args.resume_from is not None:
        resume = json.loads((args.resume_from / "milestone.json").read_text(encoding="utf-8"))
        weights = str(args.resume_from)
        print("resuming from %s: %d tokens, %d steps, %d sequences already consumed"
              % (args.resume_from, resume["actual_tokens"], resume["optimizer_steps"],
                 resume["sequences"]))

    config = AutoConfig.from_pretrained(weights, local_files_only=True)
    config = getattr(config, "text_config", config)
    config.use_cache = False
    model = Qwen3_5ForCausalLM.from_pretrained(
        weights, config=config, dtype=torch.bfloat16,
        local_files_only=True).to(args.device)
    if not args.no_checkpointing:
        model.gradient_checkpointing_enable()
    model.train()

    # The repository's own chunked cross-entropy, already proven equivalent to the stock
    # loss in tests/test_chunked_ce.py. Transformers' ForCausalLMLoss upcasts the whole
    # [batch * sequence, 248320] logit tensor to fp32 and keeps it alive for backward;
    # chunking the upcast under checkpointing frees it per chunk and recomputes one
    # softmax slice during backward.
    from distillkit.core.chunked_ce import maybe_install_chunked_loss

    chunked = maybe_install_chunked_loss(model, need_model_loss=True, enabled=True)

    store = TokenStore(TOKENS, "train")
    optimizer = build_optimizer(model, args)

    tokens_per_update = args.sequence_length * args.micro_batch * args.accumulate
    planned = store.total_tokens // (args.sequence_length * args.micro_batch) \
        * args.micro_batch * args.sequence_length
    total_updates = max(1, planned // tokens_per_update)

    provenance = {
        "run": args.run,
        "initial_checkpoint": str(args.backbone),
        "initial_sha256": sha256_of_directory(Path(args.backbone)),
        "config_digest": hashlib.sha256(
            json.dumps(config.to_dict(), sort_keys=True, default=str).encode()).hexdigest(),
        "corpus": corpus_identity(),
        "objective": "plain causal CE over 100% of non-padding targets; no masking",
        "modules_attached": [],
        "sequence_length": args.sequence_length,
        "packing": "documents shuffled by data_seed, concatenated, cut to fixed length; "
                   "each document already ends with EOS; final partial sequence dropped",
        "micro_batch": args.micro_batch, "accumulate": args.accumulate,
        "tokens_per_update": tokens_per_update,
        "planned_updates": total_updates,
        "optimizer": "bitsandbytes AdamW8bit (memory: fp32 AdamW state is 16 GB for 2B params, which does not fit beside weights and gradients on one 24 GB card)", "betas": [0.9, 0.95], "eps": 1e-8,
        "lr": args.lr, "schedule": "linear warmup %d updates then cosine to 10%%" % args.warmup,
        "weight_decay": args.weight_decay, "gradient_clip": args.clip,
        "precision": "bfloat16 weights, float32 loss reduction",
        "gradient_checkpointing": not args.no_checkpointing,
        "chunked_cross_entropy": chunked,
        "seed": args.seed, "data_seed": args.data_seed,
        "resumed_from": str(args.resume_from) if args.resume_from else None,
        "resume_note": (None if args.resume_from is None else
                        "machine crashed mid-run; weights and the data stream continue "
                        "exactly, AdamW moment estimates were not saved and restart from "
                        "zero at this point"),
        "milestones": milestones,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__, "python": platform.python_version(),
        "platform": platform.platform(),
    }
    print(json.dumps({k: provenance[k] for k in
                      ("lr", "sequence_length", "micro_batch", "accumulate",
                       "tokens_per_update", "planned_updates")}, indent=2))

    def lr_at(update: int) -> float:
        if update < args.warmup:
            return args.lr * (update + 1) / args.warmup
        progress = (update - args.warmup) / max(total_updates - args.warmup, 1)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0))))

    root.mkdir(parents=True, exist_ok=True)
    write_atomic(root / "provenance.json", provenance)

    history = []
    tokens_seen = 0
    sequences = 0
    updates = 0
    pending = list(milestones)
    started = time.monotonic()
    stream = packed_stream(store, args.sequence_length, args.data_seed)
    if resume is not None:
        tokens_seen = resume["actual_tokens"]
        updates = resume["optimizer_steps"]
        sequences = resume["sequences"]
        pending = [m for m in milestones if m > tokens_seen]
        for _ in range(sequences):
            next(stream)
        print("skipped %d sequences; %d tokens to go, milestones %s"
              % (sequences, store.total_tokens - tokens_seen, pending), flush=True)
    micro = []
    accumulated = 0.0
    exhausted = False

    while pending and not exhausted:
        optimizer.zero_grad(set_to_none=True)
        rate = lr_at(updates)
        for group in optimizer.param_groups:
            group["lr"] = rate
        step_loss = 0.0
        step_tokens = 0
        for _ in range(args.accumulate):
            micro = []
            for _ in range(args.micro_batch):
                try:
                    micro.append(next(stream))
                except StopIteration:
                    exhausted = True
                    break
            if len(micro) < args.micro_batch:
                break
            ids = torch.from_numpy(np.stack(micro)).to(args.device)
            # Labels are the inputs; the model shifts them itself, so every non-padding
            # position is a supervised target and there is no mask of any kind.
            loss = model(input_ids=ids, labels=ids).loss
            (loss / args.accumulate).backward()
            step_loss += float(loss) / args.accumulate
            # Targets, not inputs: the first position of each sequence is context only.
            step_tokens += ids.shape[0] * (ids.shape[1] - 1)
            sequences += ids.shape[0]
        if step_tokens == 0:
            break
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        updates += 1
        tokens_seen += step_tokens
        accumulated = step_loss

        if updates % args.log_every == 0:
            elapsed = time.monotonic() - started
            print("update %5d  tokens %10d (%.1f%%)  loss %.4f  lr %.2e  "
                  "grad %.2f  %.0f tok/s  %.0f s"
                  % (updates, tokens_seen, 100 * tokens_seen / store.total_tokens,
                     step_loss, rate, float(norm), tokens_seen / elapsed, elapsed),
                  flush=True)
            history.append({"update": updates, "tokens": tokens_seen, "loss": step_loss,
                            "lr": rate, "grad_norm": float(norm),
                            "seconds": elapsed})

        while pending and (tokens_seen >= pending[0] or exhausted):
            milestone = pending.pop(0)
            name = ("final" if not pending else "%dm" % round(milestone / 1e6))
            target = root / name
            target.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(target, safe_serialization=True)
            load_tokenizer().save_pretrained(target)
            record = {
                "milestone_tokens": milestone, "actual_tokens": tokens_seen,
                "optimizer_steps": updates, "sequences": sequences,
                "fraction_of_corpus": tokens_seen / store.total_tokens,
                "loss": accumulated, "lr": rate,
                "elapsed_seconds": time.monotonic() - started,
                "tokens_per_second": tokens_seen / (time.monotonic() - started),
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(args.device)),
                "sha256": sha256_of_directory(target),
            }
            write_atomic(target / "milestone.json", record)
            print("CHECKPOINT %s: %d tokens, %d steps, %d sequences, loss %.4f"
                  % (name, tokens_seen, updates, sequences, accumulated), flush=True)
            write_atomic(state_path, {"history": history, "milestones_done": name,
                                      "tokens": tokens_seen, "updates": updates})
            if not pending:
                break

    elapsed = time.monotonic() - started
    summary = dict(provenance)
    summary.update({
        "final_tokens": tokens_seen, "final_updates": updates, "sequences": sequences,
        "fraction_of_corpus": tokens_seen / store.total_tokens,
        "elapsed_seconds": elapsed, "tokens_per_second": tokens_seen / elapsed,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(args.device)),
        "history": history,
    })
    write_atomic(root / "training.json", summary)
    print("\n%d tokens in %.0f s (%.0f tok/s), %d updates, %d sequences"
          % (tokens_seen, elapsed, tokens_seen / elapsed, updates, sequences))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

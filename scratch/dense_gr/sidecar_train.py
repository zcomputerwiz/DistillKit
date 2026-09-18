"""From-scratch training with the hashed structural prior as the fourth branch.

The proposal's fourth branch is an n-gram table offloaded to host RAM. `ab68bef` already
closed that: a 268.7M-row learned memory was replaced by a seeded code basis and 400K
trainable parameters at 9.2 MB, with the codes reconstructible from the seed rather than
stored, and a second independent seed reproduced the result. So the branch here is the
hash, not the table, and nothing is offloaded anywhere.

Two tweaks from that work are carried over because they are what stopped it trading one
class against another:

**The content guardrail.** A sidecar that moves only 6,166 of 248,320 logits can still
hurt content through the softmax denominator, so `fit.py` constrained it explicitly with
``L = CE_struct(z + b) + beta * relu(CE_content(z + b) - CE_content(z))``.

**The whitespace factorization.** Two independent controls said the whitespace half was
not using local context -- the unaddressed baseline kept 83% of the whitespace gain while
keeping 12% of newline, and cross-backbone transfer kept 16% against 89% and 93%. So
whitespace comes out as 485 static biases under a 33-parameter context gate that decides
how much of an already-known correction belongs at this position, rather than a single
global strength that spends probability mass everywhere.

**The regime is not the one that validated any of this, and that is the experiment.**
`structural_sidecar.py` says in its own docstring that the backbone is frozen because "a
backbone trained beside an auxiliary correction learns to lean on it without exploiting
it". The copy experiment in this directory reproduced exactly that with a different
module: content worth 0.087 nats, mere presence worth 1.13 to 2.76, and the endogenous
circuit left below plain. Training from scratch with the sidecar present is the regime
both results say fails, so `plain` is run beside it and the sidecar is also fitted post
hoc to `plain`'s checkpoint, which is the regime that did work.

The guardrail's reference is part of what changes. Frozen, ``CE_content(z)`` means "the
original model's content loss". Jointly trained it means "this model's content loss right
now", which a co-adapting backbone can satisfy by moving where it puts mass. So `--beta`
defaults to 0 here and the content delta is logged as a diagnostic instead of enforced.

    CUDA_VISIBLE_DEVICES=0 python scratch/dense_gr/sidecar_train.py --arm plain
    CUDA_VISIBLE_DEVICES=1 python scratch/dense_gr/sidecar_train.py --arm sidecar
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import triton_shim  # noqa: F401,E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.checkpoint import checkpoint  # noqa: E402

from augmented_head import AugmentedHead  # noqa: E402
from benchmark import apply_liger, build, shared_gpu_gib  # noqa: E402
from cut_cross_entropy import linear_cross_entropy  # noqa: E402
from distillkit.code_classes import HISTORICAL, code_class_of  # noqa: E402
from distillkit.experimental.structural_sidecar import (  # noqa: E402
    FactorizedSidecar, StructuralSidecar, apply_structural_bias)
from distillkit.models import Qwen35WidenedForCausalLM  # noqa: E402
from vocab_remap import build_vocabulary, cached_remap  # noqa: E402

BASE = "D:/DeepThought/Projects/HybridModel/student-2b-hf"
STORE = Path("scratch/code_training/tokens-v2")


def write_atomic(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def sha256_of_directory(path: Path) -> str:
    hasher = hashlib.sha256()
    for file in sorted(path.glob("*.safetensors")):
        with open(file, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 22), b""):
                hasher.update(block)
    return hasher.hexdigest()


def compact_classes(tokenizer, kept):
    """Historical class per *compact* id.

    The sidecar's class splitter works on the original vocabulary. Training runs on the
    32,768 remap, so every compact id is classified through the original id it stands for
    -- otherwise the structural set would be 6,166 ids of a vocabulary this model does
    not have.
    """
    specials = set(tokenizer.get_added_vocab().values())
    labels = []
    for original in kept:
        original = int(original)
        text = tokenizer.decode([original])
        labels.append(HISTORICAL[code_class_of(text, original in specials)])
    return labels


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("plain", "sidecar"), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--vocab", type=int, default=32_768)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--length", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=8_000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--beta", type=float, default=0.0,
                        help="content guardrail weight. 0 logs the content delta without "
                             "enforcing it, because a co-adapting backbone makes the "
                             "unbiased reference meaningless")
    parser.add_argument("--code-dim", type=int, default=32)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--rows", type=int, default=1 << 20)
    parser.add_argument("--position-budget", type=int, default=8192)
    parser.add_argument("--evaluate-every", type=int, default=1_000)
    parser.add_argument("--evaluate-windows", type=int, default=64)
    parser.add_argument("--report-every", type=int, default=250)
    parser.add_argument("--store", type=Path, default=STORE)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--checkpoints", type=Path,
                        default=Path("scratch/dense_gr/checkpoints"))
    args = parser.parse_args()
    if args.output is None:
        args.output = Path("scratch/dense_gr/sidecar-%s-s%d.json" % (args.arm, args.seed))

    torch.cuda.set_per_process_memory_fraction(0.90, 0)
    started = time.perf_counter()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE, local_files_only=True)

    counts = np.load(args.store / "train-counts.npy")
    kept, forward, bytes_ids = build_vocabulary(counts, tokenizer, args.vocab)
    stream, original_tokens, compact_tokens = cached_remap(
        args.store, "train", args.vocab, tokenizer, forward, bytes_ids, counts, kept)
    stream = np.asarray(stream)
    inflation = compact_tokens / original_tokens

    labels = compact_classes(tokenizer, kept)
    structural = torch.tensor([i for i, c in enumerate(labels) if c != "content"],
                              dtype=torch.long, device="cuda")
    whitespace = torch.tensor([i for i, c in enumerate(labels) if c == "whitespace"],
                              dtype=torch.long, device="cuda")
    tally = {c: labels.count(c) for c in sorted(set(labels))}
    print("arm %s seed %d | classes %s | structural %d of %d"
          % (args.arm, args.seed, json.dumps(tally), structural.numel(), args.vocab),
          flush=True)

    config = build(args.hidden, args.layers, args.vocab,
                   attn_implementation="flash_attention_2")
    torch.manual_seed(args.seed)
    model = Qwen35WidenedForCausalLM(config).to(device="cuda", dtype=torch.bfloat16)
    model.train()
    swapped = apply_liger(model, config)

    sidecar, hasher, augmented = None, None, None
    if args.arm == "sidecar":
        from distillkit.experimental.ngram_hash import NGramHashConfig, NGramHasher
        # The configuration `fit.py` used, with the vocabulary and EOS moved into the
        # compact space: ngram_size 3 with one head per order gives the bigram and trigram
        # heads the sidecar's code and its whitespace gate are both built around.
        eos = int(forward[tokenizer.eos_token_id]) if tokenizer.eos_token_id is not None \
            and forward[tokenizer.eos_token_id] >= 0 else args.vocab - 1
        hasher = NGramHasher(NGramHashConfig(
            vocab_size=args.vocab, ngram_size=3, heads_per_ngram=1,
            ngram_vocab_size_base=args.rows // 2, seed=1234, eos_token_id=eos))
        addressed = StructuralSidecar(rows=hasher.padded_vocab_size,
                                      code_dim=args.code_dim,
                                      structural=int(structural.numel()),
                                      mode="fixed", heads=args.heads, seed=20260913)
        sidecar = FactorizedSidecar(addressed, structural.cpu(), whitespace.cpu(),
                                    gated=True).to(device="cuda", dtype=torch.float32)
        augmented = AugmentedHead(sidecar, structural, args.vocab, args.hidden).to("cuda")
        print("sidecar: %s | latent width %d"
              % (json.dumps(sidecar.parameter_report()), augmented.width), flush=True)

    backbone_parameters = sum(p.numel() for p in model.parameters())
    sidecar_parameters = 0 if sidecar is None else sum(
        p.numel() for p in sidecar.parameters() if p.requires_grad)
    print("model: %.1fM backbone, %.4fM sidecar (%.3f%%), liger %s"
          % (backbone_parameters / 1e6, sidecar_parameters / 1e6,
             100 * sidecar_parameters / backbone_parameters, json.dumps(swapped)),
          flush=True)

    import bitsandbytes as bnb
    trainable = list(model.parameters()) + (
        [] if sidecar is None else list(sidecar.parameters()))
    optimizer = bnb.optim.AdamW8bit(trainable, lr=args.lr, betas=(0.9, 0.95),
                                    weight_decay=0.1)
    baseline_shared = shared_gpu_gib()
    content_mask = torch.ones(args.vocab, dtype=torch.bool, device="cuda")
    content_mask[structural] = False

    def losses_for(tokens, want_content_delta):
        """Full LM loss, and optionally the content cost of the bias.

        The sidecar's correction rides as extra latent dimensions on a widened head
        rather than as a bias on materialized logits, so Cut Cross-Entropy still applies
        -- see `augmented_head.py`, which asserts the two forms agree. The alternative
        was a chunked head at 59k tok/s against CCE's 113k.

        `S` is cast to the head's bf16 rather than kept in fp32. The head weight is
        already bf16 and CCE reduces in bf16 with fp32 accumulation, so this is the
        precision the rest of the head runs at, not a new compromise.
        """
        hidden = model.model(input_ids=tokens,
                             attention_mask=torch.ones_like(tokens),
                             use_cache=False).last_hidden_state
        head = model.lm_head.weight
        if augmented is not None:
            code = augmented.code_for(hasher.row_indices(tokens)).to(hidden.dtype)
            state, head = torch.cat([hidden, code], dim=-1), augmented.wide_head(head)
        else:
            state = hidden

        per = linear_cross_entropy(state, head, tokens, shift=1, reduction="none")
        loss = per.mean()
        delta = None
        if want_content_delta and augmented is not None:
            # The same positions scored without the bias. Content targets only, because
            # the sidecar may not move a content logit and the question is whether it
            # cost content anything through the denominator anyway.
            plain = linear_cross_entropy(hidden, model.lm_head.weight, tokens, shift=1,
                                         reduction="none")
            is_content = content_mask[tokens[:, 1:]]
            if is_content.any():
                delta = (per[is_content] - plain[is_content]).mean()
        return loss, delta

    held_stream, _, _ = cached_remap(args.store, "calibration", args.vocab, tokenizer,
                                     forward, bytes_ids, counts, kept)
    held_rng = np.random.default_rng(12345)
    held_starts = held_rng.integers(0, held_stream.shape[0] - args.length - 1,
                                    size=args.evaluate_windows)
    evaluation = torch.from_numpy(
        np.stack([held_stream[s:s + args.length] for s in held_starts]).astype(np.int64))

    @torch.no_grad()
    def evaluate():
        model.eval()
        total, batches = 0.0, 0
        for start in range(0, evaluation.shape[0], args.batch):
            chunk = evaluation[start:start + args.batch].to("cuda")
            loss, _ = losses_for(chunk, False)
            total += float(loss)
            batches += 1
        model.train()
        return total / max(batches, 1)

    window = args.batch * args.length
    generator = np.random.default_rng(args.seed)
    history = []
    torch.cuda.synchronize()
    train_started = time.perf_counter()
    for step in range(args.steps):
        starts = generator.integers(0, stream.shape[0] - args.length - 1, size=args.batch)
        batch = np.stack([stream[s:s + args.length] for s in starts]).astype(np.int64)
        tokens = torch.from_numpy(batch).to("cuda", non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        report_now = step % args.report_every == 0 or step == args.steps - 1
        loss, delta = losses_for(tokens, report_now)
        objective = loss
        if args.beta and delta is not None:
            objective = loss + args.beta * torch.relu(delta)
        objective.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()

        if report_now:
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - train_started
            seen = (step + 1) * window
            row = {"step": step, "tokens": seen, "loss": float(loss.detach()),
                   "loss_per_original_token": float(loss.detach()) * inflation,
                   "tokens_per_second": seen / elapsed}
            if delta is not None:
                row["content_delta"] = float(delta.detach())
            if step % args.evaluate_every == 0 or step == args.steps - 1:
                row["heldout"] = evaluate()
                row["heldout_per_original_token"] = row["heldout"] * inflation
            history.append(row)
            print("step %5d  train %7.4f  held %8s  content %8s  %8.0f tok/s"
                  % (step, row["loss"],
                     "%.4f" % row["heldout"] if "heldout" in row else "-",
                     "%+.5f" % row["content_delta"] if "content_delta" in row else "-",
                     row["tokens_per_second"]), flush=True)
            drift = shared_gpu_gib() - baseline_shared
            if drift > 0.25:
                args.output.write_text(json.dumps(
                    {"aborted": "spilled to system RAM", "shared_delta_gib": drift,
                     "arm": args.arm, "seed": args.seed, "step": step,
                     "history": history}, indent=2), encoding="utf-8")
                raise SystemExit("spilled %.2f GiB into system RAM at step %d" % (drift, step))

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - train_started
    report = {
        "arm": args.arm, "seed": args.seed, "vocab": args.vocab, "beta": args.beta,
        "backbone_parameters": int(backbone_parameters),
        "sidecar_parameters": int(sidecar_parameters),
        "sidecar": None if sidecar is None else sidecar.parameter_report(),
        "class_tally": tally, "structural_ids": int(structural.numel()),
        "whitespace_ids": int(whitespace.numel()),
        "batch": args.batch, "length": args.length, "steps": args.steps,
        "scored_tokens": args.steps * window, "inflation": inflation,
        "seconds": elapsed, "tokens_per_second": args.steps * window / elapsed,
        "final_loss": history[-1]["loss"], "final_heldout": history[-1].get("heldout"),
        "final_heldout_per_original_token": history[-1].get("heldout_per_original_token"),
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2 ** 30,
        "shared_delta_gib": shared_gpu_gib() - baseline_shared,
        "setup_seconds": train_started - started, "history": history,
    }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("wrote %s" % args.output)

    target = args.checkpoints / ("sc-%s-s%d" % (args.arm, args.seed))
    target.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(target, safe_serialization=True)
    tokenizer.save_pretrained(target)
    if sidecar is not None:
        torch.save({"state_dict": {k: v.cpu() for k, v in sidecar.state_dict().items()},
                    "arm": args.arm, "seed": args.seed, "rows": args.rows,
                    "code_dim": args.code_dim, "heads": args.heads,
                    "vocab": args.vocab, "steps": args.steps},
                   target / "sidecar.pt")
    write_atomic(target / "milestone.json", {
        "arm": args.arm, "seed": args.seed, "scored_tokens": args.steps * window,
        "optimizer_steps": args.steps, "final_loss": report["final_loss"],
        "final_heldout": report["final_heldout"], "elapsed_seconds": elapsed,
        "tokens_per_second": report["tokens_per_second"],
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "sha256": sha256_of_directory(target),
    })
    print("CHECKPOINT %s: %d tokens, loss %.4f"
          % (target.name, args.steps * window, report["final_loss"]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

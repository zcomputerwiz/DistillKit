"""Teach a converted model's router what the model it came from was already doing.

Every other part of a conversion is fitted from the source: the gated residual converts
exactly, the latent and its up-projections are least squares against the source's own
keys and values. The indexer is the exception -- it has no counterpart in a stock
checkpoint, so it starts where it was constructed, and a random router is a large part of
why the full-stack conversion begins nearly two nats down.

But the source does hold the answer. It computes dense attention, and dense attention
pooled to blocks is exactly the distribution the router is trying to predict. So:

    teacher   block-pooled attention of the source, per full-attention layer
    student   `block_scores` of the converted model's router, same layers
    loss      KL(teacher || softmax(student)) over the blocks that compete

Nothing else trains. The backbone is frozen, the target is dense and per position rather
than one scalar of next-token loss, and there is no credit assignment through depth --
which is why this needs a calibration set rather than a training run. DeepSeek trains
its indexer the same way and calls it expensive, because at 128K context the dense
attention is the thing sparsity exists to avoid computing. At 1024 tokens it is free.

    python scratch/dense_gr/distill_indexer.py \\
        --source scratch/dense_gr/checkpoints-arm/smoke-r1-1-nogr \\
        --student scratch/dense_gr/checkpoints-conv/full-r1-1 \\
        --output scratch/dense_gr/checkpoints-conv/full-r1-1-distilled
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import triton_shim  # noqa: F401,E402

from distillkit.models import Qwen35WidenedForCausalLM  # noqa: E402
from distillkit.models.qwen35.csa2 import Qwen35SparseLatentAttention  # noqa: E402

STORE = Path("scratch/code_training/tokens-v2")


def windows(store, vocab, count, length, device, split="train"):
    stream = np.memmap(store / ("%s-v%d.bin" % (split, vocab)), dtype=np.uint16, mode="r")
    stride = max(length, (len(stream) - length) // max(1, count))
    for index in range(count):
        start = index * stride
        yield torch.from_numpy(
            np.array(stream[start:start + length], dtype=np.int64).reshape(1, length)
        ).to(device)


def teacher_blocks(model, ids, block, device):
    """The source's attention, averaged over heads and pooled to blocks."""
    caught = {}

    def catcher(index):
        def hook(module, args, kwargs, output):
            weights = output[1]
            if weights is None:
                raise SystemExit("layer %d returned no attention weights; the source must "
                                 "run with an implementation that exposes them" % index)
            caught[index] = weights[0].float().mean(0)
        return hook

    full = [i for i, kind in enumerate(model.config.layer_types)
            if "linear" not in str(kind)]
    handles = [model.model.layers[i].self_attn.register_forward_hook(
        catcher(i), with_kwargs=True) for i in full]
    with torch.no_grad():
        model.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
    for handle in handles:
        handle.remove()

    blocks = ids.shape[1] // block
    pooled = {}
    for index, weights in caught.items():
        pooled[index] = weights.reshape(blocks, block, blocks, block).sum((1, 3))
    return pooled


def student_scores(model, ids, device):
    """`block_scores` from every routing layer of the converted model, with gradient."""
    scores = {}

    def watcher(index, module):
        original = module.route

        def routed(hidden_states, index_keys, queries=None, position_embeddings=None,
                   candidates=None):
            scores[index] = module.block_scores(
                hidden_states, index_keys, queries, position_embeddings)
            return original(hidden_states, index_keys, queries, position_embeddings,
                            candidates)
        module.route = routed
        return original

    # A linear-attention layer has no `self_attn` at all, and a reuse layer has no router.
    routers = {i: m for i, m in enumerate(model.model.layers)
               if isinstance(getattr(m, "self_attn", None), Qwen35SparseLatentAttention)
               and m.self_attn.mode != "reuse"}
    restore = {i: watcher(i, layer.self_attn) for i, layer in routers.items()}
    model.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
    for index, layer in routers.items():
        layer.self_attn.route = restore[index]
    return scores


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--student", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--windows", type=int, default=512)
    parser.add_argument("--length", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--store", type=Path, default=STORE)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)
    source = Qwen35WidenedForCausalLM.from_pretrained(
        args.source, dtype=torch.float32, attn_implementation="eager").to(device).eval()
    student = Qwen35WidenedForCausalLM.from_pretrained(
        args.student, dtype=torch.bfloat16).to(device)
    block = student.config.csa2_block_size
    vocab = student.config.vocab_size

    trained = [p for name, p in student.named_parameters()
               if any(part in name for part in
                      ("index_q_proj", "index_k_proj", "index_weight", "indexer_proj",
                       "index_gate"))]
    if not trained:
        raise SystemExit("the student has no indexer parameters to train")
    for parameter in student.parameters():
        parameter.requires_grad_(False)
    for parameter in trained:
        parameter.requires_grad_(True)
    student.train()
    print("source %s\nstudent %s" % (args.source, args.student))
    print("training %d indexer tensors, %d parameters, everything else frozen"
          % (len(trained), sum(p.numel() for p in trained)), flush=True)

    from convert_full import heldout

    student.eval()
    before = heldout(student, args.store, vocab, 64, args.length, device)
    student.train()
    print("heldout before %.4f" % before, flush=True)

    optimizer = torch.optim.AdamW(trained, lr=args.lr)
    began = time.perf_counter()
    history = []
    for epoch in range(args.epochs):
        total, seen = 0.0, 0
        for step, ids in enumerate(windows(args.store, vocab, args.windows,
                                           args.length, device)):
            target = teacher_blocks(source, ids, block, device)
            scores = student_scores(student, ids, device)
            loss = 0.0
            for index, (score, eligible) in scores.items():
                if index not in target:
                    continue
                mask = eligible.unsqueeze(0)
                teacher = target[index].unsqueeze(0).masked_fill(~mask, 0.0)
                rows = teacher.sum(-1, keepdim=True)
                keep = (rows > 1e-6).squeeze(-1)
                if not keep.any():
                    continue
                teacher = teacher / rows.clamp_min(1e-6)
                student_logp = torch.log_softmax(
                    score.masked_fill(~mask, float("-inf")).float(), dim=-1)
                per_row = -(teacher * student_logp.nan_to_num(neginf=0.0)).sum(-1)
                loss = loss + per_row[keep].mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            total += float(loss)
            seen += 1
            if step % 50 == 0:
                print("epoch %d step %4d  cross entropy %.4f  %6.1f s"
                      % (epoch, step, total / max(seen, 1), time.perf_counter() - began),
                      flush=True)
        history.append(total / max(seen, 1))
        print("epoch %d done, mean %.4f" % (epoch, history[-1]), flush=True)

    student.eval()
    after = heldout(student, args.store, vocab, 64, args.length, device)
    print("\nheldout  before %.4f  after %.4f  change %+.4f"
          % (before, after, after - before))

    args.output.mkdir(parents=True, exist_ok=True)
    student.save_pretrained(args.output, safe_serialization=True)
    (args.output / "distillation.json").write_text(json.dumps({
        "source": str(args.source), "student": str(args.student),
        "windows": args.windows, "length": args.length, "epochs": args.epochs,
        "lr": args.lr, "tokens": args.windows * args.length * args.epochs,
        "cross_entropy": history, "seconds": time.perf_counter() - began,
        "heldout_before": before, "heldout_after": after,
    }, indent=1), encoding="utf-8")
    print("wrote %s" % args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

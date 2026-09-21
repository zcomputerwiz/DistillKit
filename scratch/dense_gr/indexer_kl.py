"""DeepSeek's dense warm-up: fit the indexer to the attention it sits in front of.

From the V3.2-Exp report, and this follows it rather than improvising:

    Dense Warm-up Stage. We first use a short warm-up stage to initialize the lightning
    indexer. In this stage, we keep dense attention and freeze all model parameters except
    for the lightning indexer. To align the indexer outputs with the main attention
    distribution ... we set a KL-divergence loss as the training objective of the indexer.
    For warm-up, we use a learning rate of 10^-3.

Three things about that are the whole design. The target is the model's *own* attention,
so there is no teacher and no cached logits. Attention is dense while it runs, so the
target is where attention would go rather than where this indexer's own selection already
sent it. And nothing else trains, so a bad indexer cannot be compensated for elsewhere and
hidden.

Why it is needed here at all: a discrete top-k carries no gradient, so this fork folds the
index score into the attention logits through extra query and key columns to give the
indexer *something* to learn from. That works and it is not what the reference does. With
this objective the indexer has its own signal and those columns can go, which is the
change this unblocks rather than performs.

    python scratch/dense_gr/indexer_kl.py \\
        --model scratch/dense_gr/checkpoints-conv/student-2b-lidx-allfull \\
        --output scratch/dense_gr/checkpoints-2b/warmed
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import triton_shim  # noqa: F401,E402

from distillkit.models import Qwen35WidenedForCausalLM  # noqa: E402
from distillkit.models.qwen35.csa2 import (Qwen35SparseLatentAttention,  # noqa: E402
                                           dense_routing, recorded_attention,
                                           router_parameters)

from convert_full import STORE, heldout, open_split  # noqa: E402


def windows(store, vocab, count, length, device, split="train"):
    stream = open_split(store, split, vocab)
    stride = max(length, (len(stream) - length) // max(1, count))
    for index in range(count):
        start = index * stride
        yield torch.from_numpy(
            np.array(stream[start:start + length], dtype=np.int64).reshape(1, length)
        ).to(device)


def routing_layers(model):
    """Every layer that owns an indexer, in depth order. Reuse layers own none."""
    return [(i, layer.self_attn) for i, layer in enumerate(model.model.layers)
            if isinstance(getattr(layer, "self_attn", None), Qwen35SparseLatentAttention)
            and layer.self_attn.mode != "reuse"]


def watch(model, layers):
    """Catch what each routing layer was handed, so its scores can be rebuilt after."""
    seen = {}

    def catch(index):
        def hook(module, args, kwargs):
            seen[index] = (kwargs.get("hidden_states", args[0] if args else None),
                           kwargs.get("position_embeddings"))
        return hook

    handles = [model.model.layers[i].self_attn.register_forward_pre_hook(
        catch(i), with_kwargs=True) for i, _ in layers]
    return seen, handles


def indexer_loss(model, layers, seen, targets, borrowed, selected=None):
    """The KL of each indexer's scores against the attention over the same positions.

    `selected` is what makes this the sparse stage rather than the warm-up. The reference
    aligns against the whole distribution while attention is dense, and "considering only
    the selected token set" once it is not -- which is the right restriction, because a
    sparse main attention has no opinion about positions it did not read, and asking the
    indexer to predict where it would have attended is asking about something that did not
    happen.
    """
    total = 0.0
    for index, attention in layers:
        target = targets[index]
        if target is None or index not in seen:
            continue
        hidden, position = seen[index]
        _, latent, rotary = borrowed[index]
        # Rebuilt here rather than taken off the bus, where they were made under no_grad:
        # `index_k_proj` is an indexer parameter and this is the only place its gradient
        # can come from. It belongs to the *donor* -- a Reindex layer owns no key
        # projection, it scores its own queries against the keys it borrows -- so the
        # donor's projection is what this gradient reaches, from its readers as well as
        # from itself.
        owner = model.model.layers[attention.latent_donor].self_attn
        keys = owner.index_keys_from(latent.detach(), rotary.detach())
        scores, causal = attention.token_scores(
            hidden.detach(), keys, position_embeddings=position)
        where = causal if selected is None else (causal & selected[index])
        predicted = torch.log_softmax(
            scores.masked_fill(~where, float("-inf")), dim=-1)
        # Cross entropy against a distribution that already sums to one over the same
        # set -- `_record` masks before its softmax -- which is the KL up to the target's
        # own entropy, and that term has no gradient here.
        total = total + -(target * predicted.nan_to_num(neginf=0.0)).sum(-1).mean()
    return total


def step(model, layers, ids):
    """One warm-up step: dense attention, and the KL over everything reachable.

    One forward, under no_grad, with routing open: it produces the target and the inputs
    at once. Nothing in it needs a gradient, because the only thing being trained reads
    those inputs afterwards -- and the target has to come from open routing, or the
    indexer is fitted to the attention its own selection already shaped.
    """
    seen, handles = watch(model, layers)
    with torch.no_grad(), dense_routing(model), recorded_attention(model):
        model.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
        # Read inside the context: leaving it clears what was recorded.
        targets = {index: attention.last_attention for index, attention in layers}
        borrowed = {index: attention.bus.require_latent(attention.latent_donor, index)
                    for index, attention in layers}
    for handle in handles:
        handle.remove()
    return indexer_loss(model, layers, seen, targets, borrowed)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--length", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="the reference's warm-up rate")
    parser.add_argument("--evaluate", type=int, default=64)
    parser.add_argument("--store", type=Path, default=STORE)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)
    model = Qwen35WidenedForCausalLM.from_pretrained(
        args.model, dtype=torch.float32).to(device)
    vocab = model.config.vocab_size
    layers = routing_layers(model)
    if not layers:
        raise SystemExit("this model has no indexer to warm up")

    trained = [parameter for _, parameter in router_parameters(model)]
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in trained:
        parameter.requires_grad_(True)
    print("%d routing layers, %d indexer tensors, %d parameters, everything else frozen"
          % (len(layers), len(trained), sum(p.numel() for p in trained)), flush=True)

    model.eval()
    before = heldout(model, args.store, vocab, args.evaluate, args.length, device)
    print("heldout before %.4f" % before, flush=True)

    optimizer = torch.optim.AdamW(trained, lr=args.lr)
    began, history = time.perf_counter(), []
    for index, ids in enumerate(windows(args.store, vocab, args.steps, args.length,
                                        device)):
        loss = step(model, layers, ids)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        history.append(float(loss))
        if index % 25 == 0:
            print("step %4d  cross entropy %.4f  %6.1f s"
                  % (index, sum(history[-25:]) / len(history[-25:]),
                     time.perf_counter() - began), flush=True)

    after = heldout(model, args.store, vocab, args.evaluate, args.length, device)
    print("\nheldout  before %.4f  after %.4f  change %+.4f"
          % (before, after, after - before))

    args.output.mkdir(parents=True, exist_ok=True)
    model.to(torch.bfloat16).save_pretrained(args.output, safe_serialization=True)
    (args.output / "warmup.json").write_text(json.dumps({
        "model": str(args.model), "steps": args.steps, "length": args.length,
        "lr": args.lr, "cross_entropy": history,
        "heldout_before": before, "heldout_after": after,
        "seconds": time.perf_counter() - began,
    }, indent=1), encoding="utf-8")
    print("wrote %s" % args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

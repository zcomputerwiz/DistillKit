"""Apply tensor parallelism to a loaded Qwen3.5 student.

The sharded modules are drop-in: the stock decoder layer calls ``self_attn``,
``linear_attn`` and ``mlp`` by keyword with signatures the replacements match, so
swapping the submodules is enough and the layer's own forward is untouched.

**Where things live.** Each sharded module replicates its input to both cards and
reduces back to the home card, so the residual stream -- and with it the embeddings,
both layer norms, the final norm and ``lm_head`` -- stays on card 0 throughout. That
is deliberate rather than incidental:

* the embeddings and head are tied, so they are one parameter and cannot be split
  across cards without becoming two;
* keeping them on card 0 alone costs less total memory than replicating them, which
  is what a symmetric data-parallel arrangement would require;
* it leaves the outer model unmodified, so gradient checkpointing, the anchor tap and
  the folded head all continue to work untouched.

The cost is an imbalance: card 0 carries the 0.636B embedding parameters that card 1
does not. Vocab-parallel embeddings would split that 1.9 GiB each way and is the
remaining 16.4% of the model; the loss would then need a distributed log-sum-exp,
which is cheap here (two ``[batch, seq, 1]`` reductions per chunk) but not yet built.

Replicated on purpose, per the review that informed this design: the norms, the
n-gram sidecar and the distillation projections. Sharding any of them inserts a
reduction to save a fraction of a gigabyte. The gated norm inside each sharded
GatedDeltaNet is the one replicated *parameter* whose gradient is a partial --
``sync_replicated_gradients`` must run before the optimizer step.
"""

from __future__ import annotations

import logging

import torch
from torch import nn

from distillkit.tensor_parallel import peer_capable
from distillkit.tp_blocks import TensorParallelAttention, TensorParallelMLP
from distillkit.tp_gated_delta_module import (
    TensorParallelGatedDeltaNet,
    sync_replicated_gradients,
)

LOG = logging.getLogger(__name__)

__all__ = ["shard_model", "sharded_parameter_report", "sync_replicated_gradients"]


def shard_model(model: nn.Module, devices, home: str | int | None = None) -> nn.Module:
    """Replace every shardable submodule in place. Returns the same model.

    ``devices`` are the cards to split across; the first is home, where the residual
    stream and the tied embedding/head stay.
    """
    resolved = [torch.device(d) for d in devices]
    if len(resolved) < 2:
        raise ValueError("tensor parallelism needs at least two devices")
    if not peer_capable(resolved):
        raise ValueError(
            "peer access is unavailable between these devices; every reduction would "
            "stage through host memory and the split would cost more than it saves"
        )
    home_device = torch.device(home) if home is not None else resolved[0]
    if home_device != resolved[0]:
        raise ValueError("home must be the first device; the shards assume it")

    base = getattr(model, "model", model)
    # The outer model stays on the home card: embeddings, both layer norms per layer,
    # the final norm, rotary and lm_head.
    model.to(home_device)

    counts = {"mlp": 0, "full_attention": 0, "linear_attention": 0}
    for layer in base.layers:
        layer.mlp = TensorParallelMLP(layer.mlp, resolved)
        counts["mlp"] += 1
        if getattr(layer, "self_attn", None) is not None:
            layer.self_attn = TensorParallelAttention(layer.self_attn, resolved)
            counts["full_attention"] += 1
        if getattr(layer, "linear_attn", None) is not None:
            layer.linear_attn = TensorParallelGatedDeltaNet(layer.linear_attn, resolved)
            counts["linear_attention"] += 1

    report = sharded_parameter_report(model)
    LOG.info(
        "Tensor-parallel across %s: sharded %d MLPs, %d attention, %d gated-delta "
        "blocks; %.1f%% of parameters split, %.2f/%.2f GiB per card",
        [str(d) for d in resolved], counts["mlp"], counts["full_attention"],
        counts["linear_attention"], 100 * report["sharded_fraction"],
        *report["gib_per_device"],
    )
    return model


def sharded_parameter_report(model: nn.Module) -> dict:
    """What actually ended up split, and how the bytes landed per card.

    Reported rather than assumed: a module that silently failed to shard still runs,
    it just uses twice the memory and none of the parallelism.
    """
    sharded_modules = (
        TensorParallelMLP, TensorParallelAttention, TensorParallelGatedDeltaNet,
    )
    sharded_ids = set()
    for child in model.modules():
        if isinstance(child, sharded_modules):
            sharded_ids.update(id(p) for p in child.parameters())

    total = sharded = 0
    per_device: dict[str, int] = {}
    for parameter in model.parameters():
        total += parameter.numel()
        if id(parameter) in sharded_ids:
            sharded += parameter.numel()
        key = str(parameter.device)
        per_device[key] = per_device.get(key, 0) + parameter.numel() * parameter.element_size()

    devices = sorted(per_device)
    return {
        "total_parameters": total,
        "sharded_parameters": sharded,
        "sharded_fraction": sharded / total if total else 0.0,
        "bytes_per_device": {d: per_device[d] for d in devices},
        "gib_per_device": [per_device[d] / 1024**3 for d in devices],
    }

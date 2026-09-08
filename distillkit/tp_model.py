"""Apply tensor parallelism to a loaded Qwen3.5 student.

The sharded modules are drop-in: the stock decoder layer calls ``self_attn``,
``linear_attn`` and ``mlp`` by keyword with signatures the replacements match, so
swapping the submodules is enough and the layer's own forward is untouched.

**Where things live.** Each sharded module replicates its input to both cards and
reduces back to the home card, so the residual stream -- both layer norms, the final
norm, rotary and the sidecar -- stays on card 0 throughout, and the outer model is
unmodified: gradient checkpointing, the anchor tap and the folded head all continue to
work untouched.

The tied embedding/head is split by vocabulary rows (``tp_vocab``), half on each card,
with its gradient and optimizer state following the shards. It is the one whole
parameter big enough to unbalance the cards -- 0.636B parameters, 3.5 GiB with 8-bit
moments -- and both other placements were measured and rejected: whole on the home card
left card 0 the heavier by about 5 GiB, whole on the other card flipped that to
11.42 / 15.02. The lookup's result and the head's input are the only added traffic,
20 MiB each per 4096-token microbatch; the folded loss composes its log-sum-exp from
per-card pieces rather than gathering logits.

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
from distillkit.tp_vocab import VocabParallelEmbedding, VocabParallelHead

LOG = logging.getLogger(__name__)

__all__ = ["shard_model", "sharded_parameter_report", "sync_replicated_gradients"]


def shard_model(model: nn.Module, devices, home: str | int | None = None) -> nn.Module:
    """Replace every shardable submodule in place. Returns the same model.

    ``devices`` are the cards to split across; the first is home, where the residual
    stream stays.
    """
    resolved = [torch.device(d) for d in devices]
    if hasattr(model, "_distillkit_tp_devices"):
        raise ValueError("Model is already tensor parallel")
    if getattr(model.config, "model_type", None) != "qwen3_5_text":
        raise ValueError("Tensor parallelism currently supports qwen3_5_text only")
    if any(hasattr(child, "_hf_hook") for child in model.modules()):
        raise ValueError("Load without a device_map before applying tensor parallelism")
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
    # The outer model stays on the home card: both layer norms per layer, the final
    # norm and rotary. The tied embedding/head is then split by rows across all cards.
    model.to(home_device)
    _shard_tied_embeddings(model, base, resolved)

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

    model.config.use_cache = False
    model._distillkit_tp_devices = tuple(str(d) for d in resolved)
    # This is placement metadata, not accelerate dispatch hooks: prevent Trainer
    # from moving the whole model to one device or wrapping it in DataParallel.
    model.hf_device_map = {"": str(home_device), **{n: str(p.device) for n, p in model.named_parameters()}}

    report = sharded_parameter_report(model)
    LOG.info(
        "Tensor-parallel across %s: sharded %d MLPs, %d attention, %d gated-delta "
        "blocks and the tied embedding; %.1f%% of parameters split, %.2f/%.2f GiB per card",
        [str(d) for d in resolved], counts["mlp"], counts["full_attention"],
        counts["linear_attention"], 100 * report["sharded_fraction"],
        *(report["gib_per_device"] + [0.0])[:2],
    )
    return model


def _shard_tied_embeddings(model: nn.Module, base: nn.Module, devices) -> None:
    embed, head = model.get_input_embeddings(), model.get_output_embeddings()
    if head is None or head.weight is not embed.weight:
        raise ValueError(
            "tensor parallelism expects tie_word_embeddings: the head is sharded "
            "through the embedding it shares a parameter with"
        )
    if head.bias is not None:
        raise ValueError("a biased lm_head is not supported by the vocab-parallel head")
    embedding = VocabParallelEmbedding(embed, devices)
    base.embed_tokens = embedding
    model.lm_head = VocabParallelHead(embedding)
    if model.get_input_embeddings() is not embedding or model.get_output_embeddings() is not model.lm_head:
        raise ValueError("model does not expose its embeddings as model.embed_tokens / lm_head")


def sharded_parameter_report(model: nn.Module) -> dict:
    """What actually ended up split, and how the bytes landed per card.

    Reported rather than assumed: a module that silently failed to shard still runs,
    it just uses twice the memory and none of the parallelism.
    """
    sharded_modules = (
        TensorParallelMLP, TensorParallelAttention, TensorParallelGatedDeltaNet,
        VocabParallelEmbedding,
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

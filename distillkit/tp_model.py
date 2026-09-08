"""Apply tensor parallelism to a loaded Qwen3.5 student.

The sharded modules are drop-in: the stock decoder layer calls ``self_attn``,
``linear_attn`` and ``mlp`` by keyword with signatures the replacements match, so
swapping the submodules is enough and the layer's own forward is untouched.

**Where things live.** Each sharded module replicates its input to both cards and
reduces back to the home card, so the residual stream -- both layer norms, the final
norm, rotary and the sidecar -- stays on card 0 throughout, and the outer model is
unmodified: gradient checkpointing, the anchor tap and the folded head all continue to
work untouched.

The tied embedding/head is the one whole parameter big enough to matter: 0.636B
parameters, 16.4% of the model, and with its gradient and 8-bit optimizer moments
3.5 GiB that would otherwise all sit on the home card on top of its half of every
layer. It cannot be split across cards without becoming two parameters, but it can be
*moved*: it lives on the last card, and :class:`RemoteEmbedding` / :class:`RemoteLinear`
run it where the weight is and hand the result back to the caller's card, so the
residual stream never notices. What crosses NVLink is the embedding output and the
post-norm state -- 20 MiB each at sequence 4096, against the ~100 reductions of the
same size the layers already do per microbatch. With the folded head, the loss's
logits chunks and their gradients live on that card too.

Vocab-parallel embeddings would instead split the parameter each way; the loss would
then need a distributed log-sum-exp, cheap here but not built, and after the move
there is no imbalance left for it to fix.

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

__all__ = [
    "RemoteEmbedding",
    "RemoteLinear",
    "place_tied_embeddings",
    "shard_model",
    "sharded_parameter_report",
    "sync_replicated_gradients",
]


class RemoteEmbedding(nn.Embedding):
    """An embedding whose weight lives on another card than its callers.

    Swapped onto the stock module's class the way ``torch.nn.utils.parametrize`` does,
    so the module object, its parameter and its state_dict keys are unchanged and
    checkpoints stay stock. The lookup runs where the weight is; the result lands on
    the caller's card, and autograd routes the gradient back the same way.
    """

    def forward(self, input_ids):
        return super().forward(input_ids.to(self.weight.device)).to(input_ids.device)


class RemoteLinear(nn.Linear):
    """``lm_head`` counterpart of :class:`RemoteEmbedding`.

    The output lands where the input came from, so a caller already on the weight's
    card pays no copy. The folded head relies on that: it moves the post-norm state
    over once and keeps every logits chunk, and its gradient, on the weight's card.
    """

    def forward(self, x):
        return super().forward(x.to(self.weight.device)).to(x.device)


def place_tied_embeddings(model: nn.Module, device) -> None:
    """Move the input embedding and output head to ``device``, keeping them tied."""
    embed, head = model.get_input_embeddings(), model.get_output_embeddings()
    if type(embed) is not nn.Embedding or type(head) is not nn.Linear:
        raise ValueError("embedding placement expects a stock nn.Embedding and nn.Linear head")
    tied = head.weight is embed.weight
    embed.to(device)
    if tied:
        # Module.to may or may not keep the Parameter object; make the tie explicit.
        head.weight = embed.weight
    else:
        head.to(device)
    embed.__class__, head.__class__ = RemoteEmbedding, RemoteLinear


def shard_model(
    model: nn.Module, devices, home: str | int | None = None, embedding_device=None,
) -> nn.Module:
    """Replace every shardable submodule in place. Returns the same model.

    ``devices`` are the cards to split across; the first is home, where the residual
    stream stays. ``embedding_device`` is where the tied embedding/head parameter goes,
    the last device unless told otherwise (see the module docstring for why).
    """
    resolved = [torch.device(d) for d in devices]
    embedding_home = torch.device(embedding_device) if embedding_device is not None else resolved[-1]
    if embedding_home not in resolved:
        raise ValueError("embedding_device must be one of the tensor-parallel devices")
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
    # The outer model stays on the home card -- both layer norms per layer, the final
    # norm, rotary -- except the tied embedding/head, which is parked where it balances.
    model.to(home_device)
    place_tied_embeddings(model, embedding_home)

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
    # remove_duplicate=False lists lm_head.weight beside the embedding it is tied to,
    # so check_tied_embeddings_colocated can see both entries agree.
    model.hf_device_map = {
        "": str(home_device),
        **{n: str(p.device) for n, p in model.named_parameters(remove_duplicate=False)},
    }

    report = sharded_parameter_report(model)
    LOG.info(
        "Tensor-parallel across %s: sharded %d MLPs, %d attention, %d gated-delta "
        "blocks; %.1f%% of parameters split, %.2f/%.2f GiB per card",
        [str(d) for d in resolved], counts["mlp"], counts["full_attention"],
        counts["linear_attention"], 100 * report["sharded_fraction"],
        *(report["gib_per_device"] + [0.0])[:2],
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

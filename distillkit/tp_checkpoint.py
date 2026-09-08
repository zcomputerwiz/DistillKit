"""Portable HF weights from TP shards, with a separate optimizer-layout marker.

Training parameters stay on their GPUs. CPU tensors here are temporary checkpoint
serialization buffers. The exported model has stock names/shapes and can be loaded
without TP; resuming its optimizer requires the same TP parameter layout.
"""
from dataclasses import dataclass
import json
from pathlib import Path

import torch

from distillkit.tp_blocks import TensorParallelAttention
from distillkit.tp_gated_delta_module import TensorParallelGatedDeltaNet
from distillkit.tp_linear import ColumnParallelLinear, RowParallelLinear
from distillkit.tp_vocab import VocabParallelEmbedding, VocabParallelHead

MARKER = "distillkit_tp.json"


@dataclass
class TensorSpec:
    name: str
    keys: tuple[str, ...]
    dim: int = 0
    indices: tuple | None = None
    replicated: bool = False


def tensor_specs(model):
    """Describe each canonical tensor exactly once, including untouched tensors."""
    specs = []
    for prefix, module in model.named_modules():
        if isinstance(module, (ColumnParallelLinear, RowParallelLinear)):
            specs.append(TensorSpec(prefix + ".weight", tuple(f"{prefix}.shards.{i}" for i in range(len(module.shards))),
                                    dim=0 if isinstance(module, ColumnParallelLinear) else 1))
            if isinstance(module, ColumnParallelLinear) and module.biases is not None:
                specs.append(TensorSpec(prefix + ".bias", tuple(f"{prefix}.biases.{i}" for i in range(len(module.biases)))))
        if isinstance(module, (VocabParallelEmbedding, VocabParallelHead)):
            # Both roles hold the same shards; each reconstructs under its own name and
            # the tie de-duplicates them on save, as it does for the stock model.
            specs.append(TensorSpec(prefix + ".weight",
                                    tuple(f"{prefix}.shards.{i}" for i in range(len(module.shards))), dim=0))
        if isinstance(module, TensorParallelAttention):
            for name in ("q_norm", "k_norm"):
                replicas = getattr(module, name + "s")
                for key in replicas[0].state_dict():
                    specs.append(TensorSpec(f"{prefix}.{name}.{key}",
                        tuple(f"{prefix}.{name}s.{i}.{key}" for i in range(len(replicas))), replicated=True))
        if isinstance(module, TensorParallelGatedDeltaNet):
            for name in ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "conv1d", "norm"):
                replicas = getattr(module, name)
                for key in replicas[0].state_dict():
                    specs.append(TensorSpec(f"{prefix}.{name}.{key}",
                        tuple(f"{prefix}.{name}.{i}.{key}" for i in range(len(replicas))),
                        indices=module.plan.channels if name in ("in_proj_qkv", "conv1d") else None,
                        replicated=name == "norm"))
            for name in ("A_log", "dt_bias"):
                specs.append(TensorSpec(f"{prefix}.{name}", tuple(f"{prefix}.{name}.{i}" for i in range(len(module.devices)))))
    handled = {key for spec in specs for key in spec.keys}
    specs.extend(TensorSpec(key, (key,)) for key in model.state_dict() if key not in handled)
    return specs


@torch.no_grad()
def consolidated_state_dict(model):
    native = model.state_dict()
    result = {}
    for spec in tensor_specs(model):
        pieces = [native[key].detach().to("cpu", copy=True) for key in spec.keys]
        if spec.replicated:
            if any(not torch.equal(pieces[0], p) for p in pieces[1:]):
                raise ValueError(f"Diverged TP replicas for {spec.name}; refusing a lossy checkpoint")
            value = pieces[0]
        elif len(pieces) == 1:
            value = pieces[0]
        else:
            value = torch.cat(pieces, dim=spec.dim)
            if spec.indices is not None:
                # Restore [all Q | all K | all V], not [rank-0 QKV | rank-1 QKV].
                for indices, piece in zip(spec.indices, pieces):
                    value.index_copy_(spec.dim, indices.cpu(), piece)
        result[spec.name] = value.contiguous()
    if model.config.tie_word_embeddings:
        result["lm_head.weight"] = result["model.embed_tokens.weight"]
    return result


@torch.no_grad()
def load_consolidated_state_dict(model, state):
    native = model.state_dict()
    specs = tensor_specs(model)
    state = dict(state)
    if model.config.tie_word_embeddings:
        embed = state.get("model.embed_tokens.weight", state.get("lm_head.weight"))
        if embed is not None:
            if "lm_head.weight" in state and not torch.equal(embed, state["lm_head.weight"]):
                raise ValueError("Checkpoint contains conflicting tied embedding/head weights")
            state.setdefault("model.embed_tokens.weight", embed)
            state.setdefault("lm_head.weight", embed)
    expected = {spec.name for spec in specs}
    if set(state) != expected:
        raise ValueError(f"TP checkpoint keys differ: missing={sorted(expected-set(state))}, unexpected={sorted(set(state)-expected)}")
    # Check every shape before mutating parameters; never silently initialize shards.
    for spec in specs:
        shape = list(native[spec.keys[0]].shape)
        if not spec.replicated and len(spec.keys) > 1:
            shape[spec.dim] = sum(native[key].shape[spec.dim] for key in spec.keys)
        if tuple(state[spec.name].shape) != tuple(shape):
            raise ValueError(f"TP checkpoint shape mismatch for {spec.name}")
    for spec in specs:
        value, offset = state[spec.name], 0
        for rank, key in enumerate(spec.keys):
            target = native[key]
            if spec.replicated or len(spec.keys) == 1:
                piece = value
            elif spec.indices is not None:
                piece = value.index_select(spec.dim, spec.indices[rank].to(value.device))
            else:
                length = target.shape[spec.dim]
                piece = value.narrow(spec.dim, offset, length)
                offset += length
            target.copy_(piece)


def training_layout(model):
    return {"format_version": 1, "parts": len(model._distillkit_tp_devices),
            "parameters": [[n, list(p.shape), str(p.dtype), p.requires_grad] for n, p in model.named_parameters()]}


def write_layout(model, directory):
    (Path(directory) / MARKER).write_text(json.dumps(training_layout(model), indent=2), encoding="utf-8")


def load_checkpoint(model, directory):
    from safetensors.torch import load_file

    directory = Path(directory)
    marker = directory / MARKER
    if not marker.is_file() or json.loads(marker.read_text(encoding="utf-8")) != training_layout(model):
        raise ValueError("TP resume requires a checkpoint with the same shard, dtype and frozen-parameter layout; use model: to start from portable weights")
    index = directory / "model.safetensors.index.json"
    files = sorted(set(json.loads(index.read_text())["weight_map"].values())) if index.is_file() else ["model.safetensors"]
    state = {}
    for filename in files:
        path = (directory / filename).resolve()
        if path.parent != directory.resolve():
            raise ValueError("Checkpoint shard must be inside its checkpoint directory")
        shard = load_file(str(path), device="cpu")
        if state.keys() & shard.keys():
            raise ValueError("Duplicate tensors in checkpoint shards")
        state.update(shard)
    load_consolidated_state_dict(model, state)

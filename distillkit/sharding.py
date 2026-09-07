"""Device placement helpers for a student split across several GPUs.

A single-process layer split needs no process group: ``Tensor.to(device)`` is
differentiable, so autograd copies activations forward and gradients back on its
own, over NVLink where the pair supports peer access. What it does *not* do is
tell the rest of the training code where anything ended up. These helpers answer
that question from ``model.hf_device_map`` so distillation projections are
constructed on the right card and the loss functions can pull their teacher-side
tensors across.

Everything here degrades to the single-device answer when no device map exists,
so the same call sites work unsharded.
"""

from __future__ import annotations

import torch


def as_device(value: str | int | torch.device) -> torch.device:
    """Normalize an ``hf_device_map`` value.

    accelerate stores bare ints for CUDA ordinals, and strings for everything
    else ("cpu", "disk", "cuda:1").
    """
    if isinstance(value, torch.device):
        return value
    if isinstance(value, int):
        return torch.device("cuda", value)
    return torch.device(value)


def _lookup(device_map: dict, name: str) -> str | int | None:
    """Resolve a module path against a device map keyed by ancestor prefixes."""
    while True:
        if name in device_map:
            return device_map[name]
        if "." not in name:
            break
        name = name.rsplit(".", 1)[0]
    return device_map.get("", None)


def is_sharded(model) -> bool:
    device_map = getattr(model, "hf_device_map", None)
    if not device_map:
        return False
    devices = {str(as_device(v)) for v in device_map.values()}
    return len(devices) > 1


def module_device(model, name: str) -> torch.device:
    """Device of a named submodule, falling back to the input embeddings."""
    device_map = getattr(model, "hf_device_map", None)
    if device_map:
        found = _lookup(device_map, name)
        if found is not None:
            return as_device(found)
    return model.get_input_embeddings().weight.device


def hidden_state_device(model, index: int) -> torch.device:
    """Device of ``outputs.hidden_states[index]``.

    The tuple has ``num_hidden_layers + 1`` entries: entry 0 is the embedding
    output, entries 1..n-1 are decoder layer outputs, and the last entry is taken
    *after* ``model.norm`` rather than from the final decoder layer.
    """
    num_layers = model.config.num_hidden_layers
    if index <= 0:
        name = "model.embed_tokens"
    elif index >= num_layers:
        name = "model.norm"
    else:
        name = f"model.layers.{index - 1}"
    return module_device(model, name)


def _module_name(model, target) -> str | None:
    for name, module in model.named_modules():
        if module is target:
            return name
    return None


def returns_outputs_on_input_device(model) -> bool:
    """True when accelerate gathers the whole forward output back to one device.

    ``dispatch_model`` puts an ``AlignDevicesHook(io_same_device=True)`` on the root,
    which sends *everything* the forward returns -- logits and the full hidden-state
    tuple alike -- to the device the inputs came from. So a device map says where a
    tensor was produced, not where it will be observed.
    """
    hook = getattr(model, "_hf_hook", None)
    return bool(getattr(hook, "io_same_device", False))


def anchor_device(model, index: int) -> torch.device:
    """Device ``outputs.hidden_states[index]`` is actually observed on.

    This is what a distillation projection has to be built on. Under accelerate's
    dispatch that is the input device regardless of which card produced the state;
    without it, the producing module's own device.
    """
    if returns_outputs_on_input_device(model):
        # Inputs are prepared for the embeddings, and the output hook returns to
        # wherever they came from.
        return model.get_input_embeddings().weight.device
    return hidden_state_device(model, index)


def check_tied_embeddings_colocated(model) -> None:
    """Tied input embeddings and head must be assigned the same device.

    ``tie_word_embeddings`` makes one parameter serve both. Splitting them would
    either fail outright or, worse, quietly leave two tensors that train apart and
    reconstruct into a checkpoint that matches neither arm. This reads the device
    *map* rather than the materialized tensors so it still fires when the split is
    only planned.
    """
    if not getattr(model.config, "tie_word_embeddings", False):
        return
    device_map = getattr(model, "hf_device_map", None)
    if not device_map:
        return
    head = model.get_output_embeddings()
    if head is None:
        return
    embed_name = _module_name(model, model.get_input_embeddings())
    head_name = _module_name(model, head)
    if embed_name is None or head_name is None:
        return
    embed_entry = _lookup(device_map, embed_name)
    head_entry = _lookup(device_map, head_name)
    if embed_entry is None or head_entry is None:
        return
    embed_device = as_device(embed_entry)
    head_device = as_device(head_entry)
    if embed_device != head_device:
        raise ValueError(
            f"tie_word_embeddings is set but the device map puts {embed_name} on "
            f"{embed_device} and {head_name} on {head_device}; pin both to one "
            f"device in model_kwargs.device_map"
        )

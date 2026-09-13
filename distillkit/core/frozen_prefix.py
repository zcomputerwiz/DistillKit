"""Keep the frozen prefix out of autograd entirely.

A native-PLE run trains a table injected after decoder block 0 with everything before it
frozen. Autograd still builds a graph through that prefix -- the embedding lookup and
block 0 -- and then throws it away, because nothing upstream of the injection has a
parameter that can move. Running the prefix under ``no_grad`` instead is the same
arithmetic with none of the bookkeeping::

    with torch.no_grad():
        h = embeddings_and_blocks_before_the_sidecar(input_ids)
    h = sidecar_and_the_rest(h)          # still fully differentiable

The suffix stays inside autograd even though its parameters are frozen: the table learns
through the suffix's Jacobian, so wrapping the *whole* backbone in ``no_grad`` would
silently train nothing. Only the prefix is graph-free, and only when every parameter in it
is already frozen -- otherwise this would quietly drop the gradients those parameters were
supposed to receive, so it refuses instead.

The prefix modules are left in place and their bound ``forward`` is wrapped, rather than
substituting a wrapper module. Nothing about the state dict, the parameter names or the
checkpoint layout changes, which matters because the model this is applied to is saved and
reloaded by the ordinary trainer.
"""

from __future__ import annotations

import functools

import torch
from torch import nn

__all__ = ["no_grad_prefix"]


def _wrap(module: nn.Module) -> None:
    if getattr(module, "_no_grad_prefix_original", None) is not None:
        return
    original = module.forward

    @functools.wraps(original)
    def forward(*args, **kwargs):
        with torch.no_grad():
            return original(*args, **kwargs)

    module._no_grad_prefix_original = original
    module.forward = forward


def _unwrap(module: nn.Module) -> None:
    original = getattr(module, "_no_grad_prefix_original", None)
    if original is not None:
        module.forward = original
        module._no_grad_prefix_original = None


def no_grad_prefix(model: nn.Module, upto_layer: int):
    """Run the embeddings and decoder layers ``[0, upto_layer)`` outside autograd.

    Returns a callable that restores the ordinary forwards. Raises if anything in the
    prefix still requires grad, because then the prefix is not actually frozen and
    skipping its graph would lose real gradients.
    """
    inner = getattr(model, "model", model)
    if not hasattr(inner, "layers"):
        raise ValueError("expected a decoder model with a `layers` list")
    if not 0 <= upto_layer <= len(inner.layers):
        raise ValueError("upto_layer %d outside the model's %d layers"
                         % (upto_layer, len(inner.layers)))

    modules = [inner.embed_tokens, *list(inner.layers)[:upto_layer]]
    trainable = [name for module in modules
                 for name, parameter in module.named_parameters()
                 if parameter.requires_grad]
    if trainable:
        raise ValueError(
            "the prefix still has trainable parameters, so running it without a graph "
            "would discard their gradients: %s" % ", ".join(trainable[:4]))

    for module in modules:
        _wrap(module)

    def restore():
        for module in modules:
            _unwrap(module)

    return restore

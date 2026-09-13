"""Bypass the MLP residual at chosen token positions, and count what that would save.

The question this exists to answer is whether an early representation could tell us which
tokens do not need the rest of the network's compute. Before building a router worth
believing, the opportunity has to be measured with an oracle -- and an oracle intervention
is only meaningful if it is exactly the intervention a router would eventually perform.

So: attention and the GatedDeltaNet state update always run. Only the MLP's contribution
to the residual stream is zeroed, and only at requested positions::

    h = h + attention(norm(h))          # always
    h = h + (0 if skipping else mlp(norm(h)))

Whole-block skipping is deliberately not offered. Attention and GDN own state that later
tokens read, so bypassing a block leaves the sequence's own history incomplete; an FFN is
position-local and owns none, which is what makes it safe to drop for one token without
corrupting another.

The altered hidden state flows onward through the ordinary model. Nothing is patched back
in: the point is to measure the real sequence-level consequence of having skipped the
compute, including its effect on every later token.

The saving is currently counted rather than taken -- the MLP still runs and its output is
masked. That keeps the arithmetic exactly "the residual was zero" with no gather/scatter
to get wrong, which for an oracle study is the right trade; `estimate_savings` reports the
FLOPs a real implementation would avoid.
"""

from __future__ import annotations

import contextlib

import torch
from torch import nn

__all__ = ["skip_ffn", "FFNSkip", "substitute_ffn", "FFNSubstitute", "capture_ffn",
           "attenuate_ffn", "FFNAttenuate", "mlp_flops_per_token",
           "model_flops_per_token", "estimate_savings"]


class FFNSkip:
    """Handle for an active intervention. Set ``mask`` before each forward."""

    def __init__(self, layers: dict[int, nn.Module]):
        self._layers = layers
        self.mask: torch.Tensor | None = None      # [batch, sequence], True = skip
        self.skipped_calls = 0
        self.total_calls = 0

    def reset_counts(self) -> None:
        self.skipped_calls = 0
        self.total_calls = 0

    @property
    def layer_indices(self) -> tuple[int, ...]:
        return tuple(sorted(self._layers))

    def _apply(self, output: torch.Tensor) -> torch.Tensor:
        self.total_calls += int(output.shape[0] * output.shape[1])
        if self.mask is None:
            return output
        mask = self.mask
        if mask.shape != output.shape[:2]:
            raise ValueError("skip mask %s does not match hidden states %s"
                             % (tuple(mask.shape), tuple(output.shape[:2])))
        self.skipped_calls += int(mask.sum())
        return output.masked_fill(mask.unsqueeze(-1).to(output.device), 0)


@contextlib.contextmanager
def skip_ffn(model: nn.Module, layers):
    """Zero the MLP residual in ``layers`` at the positions of the handle's mask.

    ``layers`` are decoder indices. An empty selection, or a handle left with no mask, is
    the stock model exactly -- which is what the correctness tests pin.
    """
    inner = getattr(model, "model", model)
    if not hasattr(inner, "layers"):
        raise ValueError("expected a decoder model with a `layers` list")
    chosen = {}
    for index in layers:
        if not 0 <= index < len(inner.layers):
            raise ValueError("layer %d outside the model's %d layers"
                             % (index, len(inner.layers)))
        chosen[index] = inner.layers[index]

    handle = FFNSkip(chosen)
    originals = {}
    for index, layer in chosen.items():
        originals[index] = layer.mlp.forward

        def wrapped(hidden_states, _original=originals[index], _handle=handle, **kwargs):
            return _handle._apply(_original(hidden_states, **kwargs))

        layer.mlp.forward = wrapped
    try:
        yield handle
    finally:
        for index, layer in chosen.items():
            layer.mlp.forward = originals[index]


# --- what a real implementation would save ------------------------------------


def mlp_flops_per_token(config) -> int:
    """SwiGLU: gate and up project hidden->intermediate, down projects back.

    Counted as two FLOPs per multiply-accumulate, which is the usual convention.
    """
    hidden = config.hidden_size
    intermediate = config.intermediate_size
    return 2 * (2 * hidden * intermediate + intermediate * hidden)


def model_flops_per_token(model, config) -> dict:
    """Linear-projection FLOPs for one token, split into MLP and everything else.

    Attention's score and context matmuls are excluded: they scale with sequence length
    rather than per token, and including them would flatter the FFN share on long
    sequences. Everything reported here is therefore a conservative account of what
    FFN skipping is worth against the rest of the model.
    """
    inner = getattr(model, "model", model)
    per_layer_mlp = mlp_flops_per_token(config)
    mlp_total = 0
    other_total = 0
    for layer in inner.layers:
        mlp_parameters = sum(p.numel() for p in layer.mlp.parameters())
        layer_parameters = sum(p.numel() for p in layer.parameters())
        mlp_total += per_layer_mlp
        other_total += 2 * (layer_parameters - mlp_parameters)
    head = 2 * config.hidden_size * config.vocab_size
    return {"mlp": mlp_total, "other": other_total, "head": head,
            "total": mlp_total + other_total + head,
            "mlp_per_layer": per_layer_mlp, "layers": len(inner.layers)}


def estimate_savings(flops: dict, skipped_calls: int, scored_tokens: int) -> dict:
    """What the counted skips would be worth if the MLP were genuinely not run."""
    if scored_tokens <= 0:
        raise ValueError("no scored tokens to attribute savings to")
    saved = skipped_calls * flops["mlp_per_layer"]
    full_mlp = scored_tokens * flops["mlp"]
    full_total = scored_tokens * flops["total"]
    return {
        "ffn_calls_skipped": int(skipped_calls),
        "ffn_calls_possible": int(scored_tokens * flops["layers"]),
        "ffn_flops_saved": int(saved),
        "ffn_flops_fraction": saved / full_mlp if full_mlp else 0.0,
        "total_flops_saved": int(saved),
        "total_flops_fraction": saved / full_total if full_total else 0.0,
    }


class FFNSubstitute:
    """Handle for residual substitution. Set a mask and a replacement per layer.

    Where the mask is true the MLP's output is replaced by the supplied residual; where it
    is false the real MLP output stands. Zeroing is the special case of substituting a
    zero residual, which the previous study found too damaging -- the question now is
    whether a *retrieved* residual does better.
    """

    def __init__(self, layers: dict[int, nn.Module]):
        self._layers = layers
        self.masks: dict[int, torch.Tensor] = {}
        self.replacements: dict[int, torch.Tensor] = {}
        self.replaced_calls = 0
        self.total_calls = 0

    def set(self, layer: int, mask: torch.Tensor, replacement: torch.Tensor) -> None:
        """``mask`` is [batch, sequence]; ``replacement`` is [batch, sequence, hidden]."""
        if layer not in self._layers:
            raise ValueError("layer %d is not intervened on" % layer)
        self.masks[layer] = mask
        self.replacements[layer] = replacement

    def clear(self) -> None:
        self.masks.clear()
        self.replacements.clear()

    def reset_counts(self) -> None:
        self.replaced_calls = 0
        self.total_calls = 0

    def _apply(self, layer: int, output: torch.Tensor) -> torch.Tensor:
        self.total_calls += int(output.shape[0] * output.shape[1])
        mask = self.masks.get(layer)
        if mask is None:
            return output
        if mask.shape != output.shape[:2]:
            raise ValueError("mask %s does not match hidden states %s"
                             % (tuple(mask.shape), tuple(output.shape[:2])))
        replacement = self.replacements[layer]
        if replacement.shape != output.shape:
            raise ValueError("replacement %s does not match hidden states %s"
                             % (tuple(replacement.shape), tuple(output.shape)))
        self.replaced_calls += int(mask.sum())
        return torch.where(mask.unsqueeze(-1).to(output.device),
                           replacement.to(output.dtype).to(output.device), output)


@contextlib.contextmanager
def substitute_ffn(model: nn.Module, layers):
    """Replace the MLP residual in ``layers`` wherever the handle's mask says to.

    Substituting a residual equal to the one the MLP would have produced reproduces the
    stock model exactly. That equivalence is what makes a cache hit meaningful, so it is
    pinned by the tests rather than assumed.
    """
    inner = getattr(model, "model", model)
    if not hasattr(inner, "layers"):
        raise ValueError("expected a decoder model with a `layers` list")
    chosen = {}
    for index in layers:
        if not 0 <= index < len(inner.layers):
            raise ValueError("layer %d outside the model's %d layers"
                             % (index, len(inner.layers)))
        chosen[index] = inner.layers[index]

    handle = FFNSubstitute(chosen)
    originals = {}
    for index, layer in chosen.items():
        originals[index] = layer.mlp.forward

        def wrapped(hidden_states, _index=index, _original=originals[index],
                    _handle=handle, **kwargs):
            return _handle._apply(_index, _original(hidden_states, **kwargs))

        layer.mlp.forward = wrapped
    try:
        yield handle
    finally:
        for index, layer in chosen.items():
            layer.mlp.forward = originals[index]


@contextlib.contextmanager
def capture_ffn(model: nn.Module, layers):
    """Record each layer's MLP input and output without changing anything.

    Used to build the cache. The forward is untouched -- the tensors are copied out on the
    way past -- so a capture run is the stock model by construction.
    """
    inner = getattr(model, "model", model)
    captured: dict[int, list] = {index: [] for index in layers}
    handles = []
    for index in layers:
        layer = inner.layers[index]

        def hook(module, inputs, output, _index=index):
            captured[_index].append((inputs[0].detach(), output.detach()))

        handles.append(layer.mlp.register_forward_hook(hook))
    try:
        yield captured
    finally:
        for handle in handles:
            handle.remove()


class FFNAttenuate:
    """Admit only ``alpha`` of the proposed FFN update at selected positions.

    The decoder's ordinary update is ``h = h + mlp(norm(h))``, which admits every
    sublayer's proposal into the residual stream at unit strength whether or not that is
    the right amount. This scales one layer's proposal at chosen positions::

        h = h + alpha * mlp(norm(h))

    ``alpha = 1`` is the stock model, ``alpha = 0`` is the zeroing intervention, and the
    values in between are what distinguish a coherent over-admission effect from an
    artefact that only appears when the update is removed entirely.
    """

    def __init__(self, layers: dict[int, nn.Module]):
        self._layers = layers
        self.masks: dict[int, torch.Tensor] = {}
        self.alphas: dict[int, float] = {}
        self.scaled_calls = 0
        self.total_calls = 0

    def set(self, layer: int, mask: torch.Tensor, alpha: float) -> None:
        if layer not in self._layers:
            raise ValueError("layer %d is not intervened on" % layer)
        self.masks[layer] = mask
        self.alphas[layer] = float(alpha)

    def clear(self) -> None:
        self.masks.clear()
        self.alphas.clear()

    def reset_counts(self) -> None:
        self.scaled_calls = 0
        self.total_calls = 0

    def _apply(self, layer: int, output: torch.Tensor) -> torch.Tensor:
        self.total_calls += int(output.shape[0] * output.shape[1])
        mask = self.masks.get(layer)
        if mask is None:
            return output
        if mask.shape != output.shape[:2]:
            raise ValueError("mask %s does not match hidden states %s"
                             % (tuple(mask.shape), tuple(output.shape[:2])))
        alpha = self.alphas[layer]
        if alpha == 1.0:
            return output
        self.scaled_calls += int(mask.sum())
        scale = torch.where(mask.unsqueeze(-1).to(output.device),
                            torch.full_like(output, alpha),
                            torch.ones_like(output))
        return output * scale


@contextlib.contextmanager
def attenuate_ffn(model: nn.Module, layers):
    """Scale the MLP update in ``layers`` by the handle's alpha where its mask is set."""
    inner = getattr(model, "model", model)
    if not hasattr(inner, "layers"):
        raise ValueError("expected a decoder model with a `layers` list")
    chosen = {}
    for index in layers:
        if not 0 <= index < len(inner.layers):
            raise ValueError("layer %d outside the model's %d layers"
                             % (index, len(inner.layers)))
        chosen[index] = inner.layers[index]

    handle = FFNAttenuate(chosen)
    originals = {}
    for index, layer in chosen.items():
        originals[index] = layer.mlp.forward

        def wrapped(hidden_states, _index=index, _original=originals[index],
                    _handle=handle, **kwargs):
            return _handle._apply(_index, _original(hidden_states, **kwargs))

        layer.mlp.forward = wrapped
    try:
        yield handle
    finally:
        for index, layer in chosen.items():
            layer.mlp.forward = originals[index]

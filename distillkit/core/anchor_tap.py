"""Capture only the hidden states a run actually reads.

``output_hidden_states=True`` retains all ``num_hidden_layers + 1`` states, and on a
device-mapped model accelerate's output hook then copies *every one of them* to the
input device. For the 4B student at sequence 4096 that is 33 tensors of
``[1, 4096, 2560]`` bf16 -- about 0.7 GiB retained, the same again copied, and gradients
for the copies -- to serve the **two** anchors the hidden-state loss reads.

A forward hook on each anchor's own module gets the same tensors without the other 31,
and without the copy: the captured state stays on the card that produced it, so the
distillation projection built there consumes it in place.

The capture script solved this shape first (``sample_transformers._AnchorTap``); this is
the training-side version, which additionally has to survive gradient checkpointing.
"""

from __future__ import annotations

import threading

import torch


def anchor_module(model, index: int):
    """Module whose output is ``outputs.hidden_states[index]``.

    The tuple has ``num_hidden_layers + 1`` entries: 0 is the embedding output,
    1..n-1 are decoder layer outputs, and the last is taken *after* ``model.norm``
    rather than from the final decoder layer.
    """
    base = getattr(model, "model", model)
    num_layers = model.config.num_hidden_layers
    if index < 0 or index > num_layers:
        raise ValueError(
            f"hidden-state index {index} outside 0..{num_layers} for this model"
        )
    if index == 0:
        return base.embed_tokens
    if index == num_layers:
        return base.norm
    return base.layers[index - 1]


def _first_tensor(value):
    """Decoder layers return a bare tensor in some versions and a tuple in others."""
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            if isinstance(item, torch.Tensor):
                return item
    raise TypeError(f"anchor module returned no tensor, got {type(value)!r}")


class CapturedStates:
    """Sparse stand-in for ``outputs.hidden_states``, indexable by anchor index only.

    Indexing an anchor that was not requested raises rather than returning the wrong
    tensor -- a silently mismatched anchor trains against the wrong depth and shows up
    as nothing worse than a slightly worse loss.
    """

    def __init__(self, states: dict[int, torch.Tensor]):
        self._states = states

    def __getitem__(self, index: int) -> torch.Tensor:
        try:
            return self._states[index]
        except KeyError:
            raise KeyError(
                f"hidden state {index} was not tapped; requested anchors are "
                f"{sorted(self._states)}"
            ) from None

    def __contains__(self, index: object) -> bool:
        return index in self._states

    def __len__(self) -> int:
        return len(self._states)

    def keys(self):
        return self._states.keys()


class AnchorTap:
    """Forward hooks capturing the states at the given ``hidden_states`` indices.

    Use as a context manager around one forward; ``states()`` is valid afterwards for
    as long as the graph is alive.
    """

    def __init__(self, model, indices):
        self.indices = sorted(set(int(i) for i in indices))
        self._model = model
        self._captured: dict[int, torch.Tensor] = {}
        self._owner: int | None = None
        self._handles = []

    def _hook(self, index: int):
        def capture(_module, _args, output):
            # Under gradient checkpointing with use_reentrant=False the hook fires
            # again during recompute, in backward. The first capture is the tensor
            # that is actually in the autograd graph and the one the loss already
            # consumed, so later firings must not replace it.
            if threading.get_ident() != self._owner:
                return
            if index not in self._captured:
                value = _first_tensor(output)
                readout = getattr(_module, "distillation_hidden_state", None)
                self._captured[index] = readout(value) if callable(readout) else value

        return capture

    def __enter__(self) -> AnchorTap:
        if self._handles:
            raise RuntimeError("AnchorTap cannot be entered twice")
        self._owner = threading.get_ident()
        self._captured = {}
        for index in self.indices:
            module = anchor_module(self._model, index)
            self._handles.append(module.register_forward_hook(self._hook(index)))
        return self

    def __exit__(self, *exc_info) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def states(self) -> CapturedStates:
        missing = [index for index in self.indices if index not in self._captured]
        if missing:
            raise RuntimeError(
                f"anchors {missing} were never produced; the forward did not reach "
                f"their modules"
            )
        return CapturedStates(dict(self._captured))

    def clear(self) -> None:
        """Drop references so the captured activations can be freed."""
        self._captured.clear()

"""Route Qwen3.5's linear-attention ops to fla on CUDA and to plain torch on CPU.

Why this exists
---------------
24 of Qwen3.5-4B's 32 layers are ``linear_attention``. Without a fused kernel they
run ``torch_chunk_gated_delta_rule``, whose two Python loops (63 iterations for the
intra-chunk triangular solve, plus one per 64-token chunk) issue thousands of tiny
kernels per layer. Measured on this box: 47.7 ms per call, ~1146 ms per forward
across 24 layers, doubled again by gradient-checkpoint recompute. That was most of a
3.2 s training step, and it is why step time barely moved between batch 1 and batch 2
-- the step was launch-bound, not compute-bound.

Installing ``flash-linear-attention`` drops that call to 1.3 ms (37x) and roughly
doubles end-to-end training throughput.

The catch: transformers' ``use_kernel_func_from_hub_with_fallback`` binds fla at
*import* time and never checks the device. fla's kernels are Triton, hence CUDA-only,
so once fla is installed every CPU call raises::

    ValueError: Pointer argument cannot be accessed from Triton (cpu tensor?)

That breaks the CPU test suite and the CPU-only verification scripts this project
relies on (the dev box's GPUs are usually occupied). This module restores a device
check by dispatching per call: fla on CUDA, the original torch implementation on CPU.

``functools.wraps`` in the transformers decorator leaves the pure-torch function on
``__wrapped__``, so the fallback is recovered rather than reimplemented.

Idempotent, and a no-op when fla is not installed.
"""

from __future__ import annotations

import functools
import logging

LOG = logging.getLogger(__name__)

__all__ = ["install_device_aware_linear_attention", "fused_linear_attention_available"]

# Ops that gained a CUDA-only implementation once fla / causal_conv1d is installed.
_PATCHED_OPS = (
    "torch_chunk_gated_delta_rule",
    "torch_recurrent_gated_delta_rule",
    "causal_conv1d_fn",
    "causal_conv1d_update",
)

_MARKER = "_distillkit_device_aware"


def fused_linear_attention_available() -> bool:
    """True when a fused CUDA path is actually bound (not just importable)."""
    import importlib.util

    return importlib.util.find_spec("fla") is not None


def _is_cuda(*args) -> bool:
    for value in args:
        if hasattr(value, "is_cuda"):
            return bool(value.is_cuda)
    return False


def install_device_aware_linear_attention() -> list[str]:
    """Patch qwen3_5's linear-attention ops to pick an implementation per call.

    Returns the names actually patched. Safe to call repeatedly.
    """
    from transformers.models.qwen3_5 import modeling_qwen3_5 as module

    patched: list[str] = []
    for name in _PATCHED_OPS:
        fused = getattr(module, name, None)
        if fused is None or getattr(fused, _MARKER, False):
            continue
        torch_impl = getattr(fused, "__wrapped__", None)
        if torch_impl is None or torch_impl is fused:
            # No wrapper means no fused implementation was bound; nothing to guard.
            continue

        def make(fused=fused, torch_impl=torch_impl, name=name):
            @functools.wraps(torch_impl)
            def dispatch(*args, **kwargs):
                # The fused kernels are Triton, so CUDA-only. Sending CPU tensors
                # there raises deep inside the Triton launcher rather than falling
                # back, so the check has to happen here.
                return (fused if _is_cuda(*args) else torch_impl)(*args, **kwargs)

            setattr(dispatch, _MARKER, True)
            dispatch._fused_impl = fused
            dispatch._torch_impl = torch_impl
            return dispatch

        setattr(module, name, make())
        patched.append(name)

    if patched:
        LOG.info("device-aware linear attention installed for: %s", ", ".join(patched))
    return patched

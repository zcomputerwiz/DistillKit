"""Inductor compilation for the elementwise chains this fork spends its memory on.

Triton 3.8 and nvcc are present on this machine and `torch.compile` does generate
working CUDA kernels here, which is worth stating because most of this project's
performance work has been shaped by things that are *not* available on Windows --
no NCCL, no `expandable_segments`, no flash-attention wheel.

What compilation buys is not arithmetic. The widened residual stream is bound by
elementwise traffic over `[batch, tokens, branches, hidden]` tensors: a sigmoid, a
broadcast add, a broadcast multiply and a reduction, each reading and writing a full
84 MB at batch 2 x 4096 and each one a separate kernel. Fused, the intermediates are
never written at all.

Everything here is optional. `DISTILLKIT_COMPILE=0` disables it, a machine without a
working Triton disables it, and a function that fails to compile falls back to eager
permanently rather than raising -- the point is a faster path to the same answer, so
losing it must never lose the answer.

Compiled and eager results are close but not bitwise equal: inductor reassociates and
keeps intermediates in registers at higher precision than the eager chain rounds to.
The identity the widening depends on survives that, because it rests on exact zeros
and ones rather than on a rounding pattern, and `tests/test_fused.py` checks it.
"""

from __future__ import annotations

import functools
import logging
import os
import threading

import torch

LOG = logging.getLogger(__name__)

_DISABLED_REASON: str | None = None


def compile_available() -> tuple[bool, str]:
    """Whether to compile at all, and why not when not."""
    if os.environ.get("DISTILLKIT_COMPILE", "1") == "0":
        return False, "DISTILLKIT_COMPILE=0"
    if not torch.cuda.is_available():
        return False, "no CUDA device"
    try:
        import triton  # noqa: F401
    except Exception as error:  # a Windows wheel that imports but cannot load
        return False, f"triton unavailable ({type(error).__name__})"
    return True, ""


def _on_cuda(args, kwargs) -> bool:
    """Compile for CUDA only.

    Inductor's CPU backend needs a C++ compiler, and there is none on the PATH here,
    so a CPU call would pay a failed compilation to arrive back at the function it
    started from. Tests run on CPU; training does not.
    """
    for value in list(args) + list(kwargs.values()):
        if isinstance(value, torch.Tensor):
            return value.is_cuda
    return False


def _on_main_thread() -> bool:
    """Compile on the main thread only.

    A compiled region inside a non-reentrant checkpoint frame breaks that frame's
    early-stop machinery when it is entered from a worker thread -- the same class of
    problem `AllReduce._save_recompute_barrier` already works around. The concurrent
    microbatch path is opt-in and measured slower, so it simply runs eager.
    """
    return threading.current_thread() is threading.main_thread()


def fused(fn):
    """Compile ``fn``, falling back to it permanently if that ever fails.

    ``dynamic=True`` on purpose: sortish batching gives almost every step a different
    sequence length, and specialising would recompile for each one. ``fullgraph`` is
    left off so an unexpected construct becomes a graph break rather than an error.
    """
    state = {"compiled": None, "failed": False}

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if state["failed"] or not _on_cuda(args, kwargs) or not _on_main_thread():
            return fn(*args, **kwargs)
        if state["compiled"] is None:
            usable, reason = compile_available()
            if not usable:
                _note_disabled(reason)
                state["failed"] = True
                return fn(*args, **kwargs)
            state["compiled"] = torch.compile(fn, dynamic=True)
        try:
            return state["compiled"](*args, **kwargs)
        except Exception as error:
            LOG.warning("compilation of %s failed (%s: %s); continuing eager",
                        fn.__qualname__, type(error).__name__, error)
            state["failed"] = True
            return fn(*args, **kwargs)

    wrapper.eager = fn
    return wrapper


def _note_disabled(reason: str) -> None:
    global _DISABLED_REASON
    if reason and _DISABLED_REASON != reason:
        _DISABLED_REASON = reason
        LOG.info("running eager: %s", reason)

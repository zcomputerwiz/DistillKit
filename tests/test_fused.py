"""Compilation is an optimisation, so losing it must never lose the answer.

Triton 3.8 and nvcc are present here and inductor does generate working CUDA kernels,
which is worth pinning because almost everything else this project reached for on
Windows was not available. What these check is the part that has to hold whether or
not that stays true: the wrapper falls back rather than raising, it does not pay for a
compile on CPU where there is no C++ compiler to use, and the widened read's identity
at initialisation survives either path.
"""

import torch

from distillkit.fused import compile_available, fused
from distillkit.widened_residual import WidenedResidual, _combine, collapse_residual


def test_cpu_calls_never_attempt_a_compile():
    """Inductor's CPU backend needs a compiler that is not on the PATH here, so a CPU
    call must run the function rather than pay for a failed compilation."""
    calls = []

    @fused
    def add(a, b):
        calls.append(torch.compiler.is_compiling())
        return a + b

    result = add(torch.ones(3), torch.ones(3))
    assert torch.equal(result, torch.full((3,), 2.0))
    assert calls == [False]


def test_a_function_that_cannot_compile_still_returns_the_right_answer(monkeypatch):
    """The fallback is the whole safety argument, so exercise it directly."""
    monkeypatch.setattr("distillkit.fused.compile_available", lambda: (True, ""))
    monkeypatch.setattr("distillkit.fused._on_cuda", lambda *_: True)

    def explode(_fn, **_kwargs):
        def raiser(*args, **kwargs):
            raise RuntimeError("no backend")
        return raiser

    monkeypatch.setattr(torch, "compile", explode)

    @fused
    def double(x):
        return x * 2

    x = torch.arange(4.0)
    assert torch.equal(double(x), x * 2)   # first call: compiles, raises, falls back
    assert torch.equal(double(x), x * 2)   # second: already known to have failed


def test_the_switch_is_honoured(monkeypatch):
    monkeypatch.setenv("DISTILLKIT_COMPILE", "0")
    usable, reason = compile_available()
    assert not usable and reason == "DISTILLKIT_COMPILE=0"


def test_the_original_stays_reachable():
    """`.eager` is how a test or a bisect gets at the uncompiled version."""
    assert collapse_residual.eager.__name__ == "collapse_residual"


def test_combine_is_the_identity_at_initialisation():
    """At initialisation both lambdas are zero, so the correction is exactly zero and
    the read is exactly its branch -- the property the whole widening rests on. It has
    to hold structurally, not by rounding, or compiling could break it."""
    torch.manual_seed(0)
    branches, width, index = 3, 8, 1
    route = WidenedResidual(width, num_branches=branches, lowrank=4, layer_idx=index)
    normalized = torch.randn(2, 5, branches, width)
    gate_logits = torch.randn(2, 5, branches * width)
    write_logits = torch.randn(2, 5, branches)

    value, weights = _combine(normalized, gate_logits, write_logits, route.read_offset,
                              route.lambda_read, route.write_offset, route.lambda_write,
                              route.read_index)

    assert torch.equal(value, normalized[..., index, :])
    assert torch.equal(weights, torch.ones_like(weights))


def test_combine_matches_the_arithmetic_it_replaced():
    """Once the routes have trained, against the expression written out longhand."""
    torch.manual_seed(1)
    branches, width = 2, 6
    normalized = torch.randn(3, 4, branches, width)
    gate_logits = torch.randn(3, 4, branches * width)
    write_logits = torch.randn(3, 4, branches)
    read_offset = torch.randn(branches)
    write_offset = torch.randn(branches)
    lambda_read, lambda_write = torch.tensor(0.7), torch.tensor(-0.3)

    value, weights = _combine(normalized, gate_logits, write_logits, read_offset,
                              lambda_read, write_offset, lambda_write, 1)

    read_gate = torch.sigmoid(gate_logits).unflatten(-1, (branches, width))
    correction = read_offset.unsqueeze(-1) + lambda_read * read_gate
    expected = normalized[..., 1, :] + (correction * normalized).sum(-2)
    torch.testing.assert_close(value, expected)
    torch.testing.assert_close(
        weights, (1 + write_offset) + lambda_write * torch.sigmoid(write_logits))


def test_collapse_is_exact_when_the_branches_agree():
    """Expressed around branch zero so equal branches collapse to themselves exactly,
    which is what makes the widened logits bit-identical at initialisation."""
    branch = torch.randn(2, 3, 7, dtype=torch.bfloat16)
    states = branch.unsqueeze(-2).expand(2, 3, 4, 7)
    assert torch.equal(collapse_residual(states), branch)

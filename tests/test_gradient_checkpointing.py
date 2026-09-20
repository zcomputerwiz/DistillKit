"""Gradient checkpointing has to be on, and has to be exact.

The flag existed on the model and the forward loop called every layer directly, so a run
that asked for checkpointing paid none of its cost and got none of its saving -- and
nothing failed, which is why it went unnoticed. These pin both halves: that the recompute
happens, and that it does not change the gradients it recomputes.
"""

import sys

import pytest
import torch

sys.path.insert(0, "scratch/dense_gr")

from distillkit.models import Qwen35WidenedForCausalLM  # noqa: E402
from tests.test_csa2_routing import BLOCK, csa2_config, tiny_config  # noqa: E402


def _model(**kwargs):
    torch.manual_seed(0)
    config = tiny_config(mla_enabled=False, **kwargs)
    config._attn_implementation = "eager"
    model = Qwen35WidenedForCausalLM(config).double()
    model.train()
    return model


def _gradients(model, tokens):
    model.zero_grad(set_to_none=True)
    out = model.model(input_ids=tokens, attention_mask=torch.ones_like(tokens),
                      use_cache=False).last_hidden_state
    out.pow(2).mean().backward()
    return {name: p.grad.clone() for name, p in model.named_parameters()
            if p.grad is not None}


@pytest.mark.parametrize("blend", [0.0, 1.0])
def test_checkpointing_does_not_change_the_gradients(blend):
    """Recomputed activations must reproduce the ones they replaced."""
    tokens = torch.randint(1, 64, (2, 16))

    model = _model(residual_stream_blend=blend)
    plain = _gradients(model, tokens)
    model.model.gradient_checkpointing = True
    recomputed = _gradients(model, tokens)

    assert plain, "no parameter took a gradient at all"
    assert set(plain) == set(recomputed)
    for name in plain:
        assert torch.allclose(plain[name], recomputed[name], atol=1e-10, rtol=1e-10), name


def test_checkpointing_actually_recomputes():
    """Count the forwards: without the wiring a layer runs once, not twice."""
    model = _model(residual_stream_blend=1.0)
    model.model.gradient_checkpointing = True
    calls = []
    layer = model.model.layers[0]
    inner = layer.forward
    layer.forward = lambda *a, **k: (calls.append(1), inner(*a, **k))[1]

    tokens = torch.randint(1, 64, (2, 16))
    out = model.model(input_ids=tokens, attention_mask=torch.ones_like(tokens),
                      use_cache=False).last_hidden_state
    assert len(calls) == 1, "forward pass should run the layer once"
    out.pow(2).mean().backward()
    assert len(calls) == 2, "backward should recompute the layer"


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="FlexAttention has no CPU backward")
def test_csa2_checkpoints_to_the_same_gradient_it_computes_without():
    """What the bus being keyed by publisher is for.

    It used to be one set of slots that each Full layer overwrote, so a Reuse layer took
    whichever had run most recently. A recompute during backward runs layers out of
    order, so the borrower read the wrong publisher and the gradients were quietly wrong,
    and the model refused the combination. Every reader names its donor now, which it
    knows from the mode sequence, so order carries nothing left to break -- and that
    matters because checkpointing is what buys the sequence length this architecture
    exists to serve.
    """
    torch.manual_seed(0)
    tokens = torch.randint(1, 64, (2, 2 * BLOCK)).cuda()
    gradients = {}
    for checkpointing in (False, True):
        torch.manual_seed(0)
        model = Qwen35WidenedForCausalLM(csa2_config()).cuda()
        model.train()
        model.model.gradient_checkpointing = checkpointing
        model(input_ids=tokens, labels=tokens, use_cache=False).loss.backward()
        gradients[checkpointing] = {n: p.grad.clone()
                                    for n, p in model.named_parameters()
                                    if p.grad is not None}

    assert set(gradients[True]) == set(gradients[False])
    assert gradients[False], "nothing took a gradient; the comparison would be vacuous"
    worst, where = 0.0, None
    for name, plain in gradients[False].items():
        gap = (plain - gradients[True][name]).abs().max().item()
        if gap > worst:
            worst, where = gap, name
    assert worst < 1e-4, "%s differs by %.3g under checkpointing" % (where, worst)

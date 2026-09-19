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
from tests.test_csa2_routing import csa2_config, tiny_config  # noqa: E402


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


def test_csa2_still_refuses_to_be_checkpointed():
    """The bus is written in forward order, which a reverse recompute breaks."""
    model = Qwen35WidenedForCausalLM(csa2_config())
    model.train()
    model.model.gradient_checkpointing = True
    tokens = torch.randint(1, 64, (2, 64))
    with pytest.raises(RuntimeError, match="gradient checkpointing"):
        model.model(input_ids=tokens, attention_mask=torch.ones_like(tokens),
                    use_cache=False)

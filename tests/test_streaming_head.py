# Assisted-by: Codex
"""Streaming head through the real accumulated training objective."""
import sys
from pathlib import Path
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scratch" / "dense_gr"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_shared_head import TinyLM, records
from training_step import backward_step


@pytest.mark.parametrize("chunk", [1, 3, 512])
@pytest.mark.parametrize("teacher_weight", [0., .5, 1.])
def test_streaming_accumulation_matches_shared(chunk, teacher_weight):
    torch.manual_seed(15)
    model = TinyLM()
    batches = records()
    # Context-only KL rows, padding and weighted CE must coexist correctly.
    context = dict(batches[1])
    context["context_kl"] = torch.zeros_like(context["weight"])
    context["context_kl"][:, :2] = .01
    batches.append(context)
    options = dict(shared_head=True, head_chunk=chunk, teacher_weight=teacher_weight,
                   unlikelihood_weight=.7)
    expected = backward_step(model, batches, **options)
    grads = {n: p.grad.clone() for n, p in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    got = backward_step(model, batches, streaming_head=True, **options)
    for key, value in expected.items():
        assert got[key] == pytest.approx(value, rel=1e-5, abs=1e-7), key
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter.grad, grads[name], rtol=1e-5, atol=1e-7, msg=name)


def test_unsupported_training_flags_are_refused_before_loading():
    from smoke_train import main
    with pytest.raises(SystemExit, match="streaming-head-loss needs shared-head-loss"):
        main(["--streaming-head-loss"])
    with pytest.raises(SystemExit, match="checkpoint-selection-cache needs outer checkpoints"):
        main(["--checkpoint-selection-cache"])


def test_streaming_head_with_frozen_body():
    torch.manual_seed(35)
    model = TinyLM()
    for parameter in model.model.parameters():
        parameter.requires_grad_(False)
    options = dict(shared_head=True, head_chunk=3, teacher_weight=.5)
    expected = backward_step(model, records(), **options)
    gradient = model.lm_head.weight.grad.clone()
    model.zero_grad(set_to_none=True)
    got = backward_step(model, records(), streaming_head=True, **options)
    assert got == pytest.approx(expected, rel=1e-5, abs=1e-7)
    torch.testing.assert_close(model.lm_head.weight.grad, gradient, rtol=1e-5, atol=1e-7)
    assert all(p.grad is None for p in model.model.parameters())

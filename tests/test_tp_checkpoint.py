"""Canonical reconstruction must preserve every tensor, not just the logits."""
import copy

import pytest
import torch

from test_tp_model import _model
from distillkit.tp_model import shard_model, sync_replicated_gradients
from distillkit.tp_checkpoint import consolidated_state_dict, load_consolidated_state_dict
from distillkit.tp_gated_delta_module import clip_grad_norm, replicated_parameter_groups


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_checkpoint_reconstructs_every_tensor_and_preserves_freezes(dtype):
    original = _model().to(dtype=dtype)
    original.model.layers[0].requires_grad_(False)
    rng = torch.random.get_rng_state()
    tp = shard_model(copy.deepcopy(original), ["cpu", "cpu"])
    assert torch.equal(torch.random.get_rng_state(), rng), "sharding changed initialization RNG"
    assert not any(p.requires_grad for p in tp.model.layers[0].parameters())
    assert {p.dtype for p in tp.parameters()} == {dtype}
    state = consolidated_state_dict(tp)
    assert set(state) == set(original.state_dict())
    for name, expected in original.state_dict().items():
        torch.testing.assert_close(state[name], expected, rtol=0, atol=0, msg=name)
    with torch.no_grad():
        for p in tp.parameters():
            p.zero_()
    # In real safetensors checkpoints the tied head has been deduplicated.
    state.pop("lm_head.weight")
    load_consolidated_state_dict(tp, state)
    rebuilt = consolidated_state_dict(tp)
    for name, expected in original.state_dict().items():
        torch.testing.assert_close(rebuilt[name], expected, rtol=0, atol=0, msg=name)
    assert tp.lm_head.weight is tp.model.embed_tokens.weight


def test_replicated_norms_and_clipping_match_unsharded_update():
    original = _model().train()
    tp = shard_model(copy.deepcopy(original), ["cpu", "cpu"]).train()
    ids = torch.arange(16).view(2, 8)
    losses = []
    for model in (original, tp):
        # Uneven microbatch window; synchronization only after all backwards.
        for row in ids:
            loss = model(input_ids=row[None]).logits.square().mean()/2
            loss.backward()
            losses.append(loss.item())
    assert losses[:2] == pytest.approx(losses[2:], rel=2e-4)
    assert sync_replicated_gradients(tp) == 6  # two gated norms + two q/k pairs
    actual_norm = clip_grad_norm(tp, 0.01)
    expected_norm = torch.nn.utils.clip_grad_norm_(original.parameters(), 0.01)
    torch.testing.assert_close(actual_norm, expected_norm, rtol=2e-4, atol=2e-5)
    for group in replicated_parameter_groups(tp):
        for p in group[1:]:
            torch.testing.assert_close(p.grad, group[0].grad, rtol=0, atol=0)
    # SGD exposes a missing gradient directly; Adam's first-step normalization can hide it.
    for model in (original, tp):
        torch.optim.SGD(model.parameters(), lr=0.1).step()
    state = consolidated_state_dict(tp)
    for name, expected in original.state_dict().items():
        torch.testing.assert_close(state[name], expected, rtol=2e-4, atol=2e-5, msg=name)


def test_export_rejects_diverged_replicas_and_load_rejects_partial_weights():
    tp = shard_model(_model(), ["cpu", "cpu"])
    state = consolidated_state_dict(tp)
    state.pop("model.norm.weight")
    with pytest.raises(ValueError, match="keys differ"):
        load_consolidated_state_dict(tp, state)
    with torch.no_grad():
        tp.model.layers[1].self_attn.q_norms[1].weight.add_(1)
    with pytest.raises(ValueError, match="Diverged TP replicas"):
        consolidated_state_dict(tp)

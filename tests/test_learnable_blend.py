"""A blend the run decides for itself, rather than one it is told.

The scheduled blend answers "what happens at 0.10". A learned one answers "where does
this sublayer want to sit", which is the question the blend sweep could only sample.
Three things have to hold for that answer to mean anything: the parameter has to be in
the graph, it has to be able to move far enough inside the step budget to say something,
and it must not be quietly pulled toward zero by machinery that has nothing to do with
the objective.
"""

import pytest
import torch

from distillkit.configuration import ResidualStreamConfig
from distillkit.hyper_connection import HyperConnection
from distillkit.optimizers import _auxiliary_parameter_ids
from distillkit.trainer import _apply_blend_lr


HIDDEN, BRANCHES, LOWRANK = 32, 4, 8


def _module(blend=0.10, learnable=True, seed=0):
    torch.manual_seed(seed)
    module = HyperConnection(HIDDEN, BRANCHES, LOWRANK, 0,
                             blend=blend, learnable_blend=learnable)
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            if name != "blend":
                parameter.normal_(std=0.05)
    return module


def _step(module, seed=1):
    torch.manual_seed(seed)
    norm = torch.nn.RMSNorm(HIDDEN, eps=1e-6)
    states = torch.randn(2, 5, BRANCHES, HIDDEN)
    output = torch.randn(2, 5, HIDDEN)
    value, weights = module.read(states, norm)
    module.write(states, output, weights).float().pow(2).mean().backward()


def test_a_learned_blend_is_a_parameter_and_a_scheduled_one_is_a_host_buffer():
    learned = _module(learnable=True)
    assert isinstance(learned.blend, torch.nn.Parameter) and learned.blend.requires_grad
    assert learned.blend.dtype is torch.float32

    scheduled = _module(learnable=False)
    assert not isinstance(scheduled.blend, torch.nn.Parameter)
    # Host-side, so reading it in `read` costs no device synchronisation.
    assert scheduled.blend.device.type == "cpu"


def test_the_blend_receives_gradient_even_at_exactly_zero():
    """Otherwise "it learned to want nothing" and "it could never move" look alike."""
    for start in (0.0, 0.10, 1.0):
        module = _module(blend=start)
        _step(module)
        assert module.blend.grad is not None
        assert float(module.blend.grad) != 0.0, start


def test_a_scheduled_blend_at_zero_stays_inert_by_design():
    """The fast path is what makes a zero blend cost nothing; it is off when learning."""
    module = _module(blend=0.0, learnable=False)
    torch.manual_seed(1)
    states = torch.randn(2, 5, BRANCHES, HIDDEN, requires_grad=True)
    value, weights = module.read(states, torch.nn.RMSNorm(HIDDEN, eps=1e-6))
    assert weights is None, "the donor write is skipped entirely at a scheduled zero"
    value.float().pow(2).mean().backward()
    assert module.W_down.weight.grad is None


def test_one_optimizer_step_moves_the_blend_by_about_the_learning_rate():
    """AdamW's step is ~lr per element, which is the whole reason blend_lr exists."""
    module = _module(blend=0.10)
    optimizer = torch.optim.AdamW([module.blend], lr=2e-3, weight_decay=0.0)
    _step(module)
    optimizer.step()
    moved = abs(float(module.blend) - 0.10)
    assert 1e-3 < moved < 3e-3, moved


def test_blend_lr_gets_its_own_group_with_no_weight_decay():
    model = torch.nn.Module()
    model.routing = _module(blend=0.10)
    other = torch.nn.Linear(4, 4)
    optimizer = torch.optim.AdamW(
        [{"params": list(other.parameters()) + [model.routing.blend],
          "lr": 1e-4, "weight_decay": 0.01}])
    _apply_blend_lr(optimizer, model, 2e-3)
    groups = {id(p): g for g in optimizer.param_groups for p in g["params"]}
    blend_group = groups[id(model.routing.blend)]
    assert blend_group["lr"] == 2e-3
    # Decay on an interpolation coefficient pulls it toward "no donor" for reasons that
    # have nothing to do with the objective, which is the question being asked.
    assert blend_group["weight_decay"] == 0.0
    assert groups[id(other.weight)]["lr"] == 1e-4
    assert sum(len(g["params"]) for g in optimizer.param_groups) == 3


def test_blend_lr_without_a_learnable_blend_is_refused():
    model = torch.nn.Module()
    model.routing = _module(learnable=False)
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=1e-4)
    with pytest.raises(ValueError, match="no learnable blend"):
        _apply_blend_lr(optimizer, model, 2e-3)


def test_the_blend_is_auxiliary_so_stage_one_freezing_keeps_it_trainable():
    model = torch.nn.Module()
    model.routing = _module(blend=0.10)
    assert id(model.routing.blend) in _auxiliary_parameter_ids(model)


def test_config_refuses_a_learned_blend_that_a_schedule_would_overwrite():
    with pytest.raises(ValueError, match="both write the blend"):
        ResidualStreamConfig(num_branches=4, lowrank=320, routing="flash_next",
                             learnable_blend=True, blend_warmup_steps=20)


def test_config_refuses_blend_settings_without_the_donor_routing():
    with pytest.raises(ValueError, match="require routing=flash_next"):
        ResidualStreamConfig(num_branches=4, lowrank=320, learnable_blend=True)
    with pytest.raises(ValueError, match="blend_lr only applies"):
        ResidualStreamConfig(num_branches=4, lowrank=320, routing="flash_next",
                             blend_lr=2e-3)


def test_a_learned_blend_survives_a_checkpoint_round_trip():
    module = _module(blend=0.10)
    with torch.no_grad():
        module.blend.fill_(0.37)
    restored = _module(blend=0.10)
    restored.load_state_dict(module.state_dict())
    assert float(restored.blend) == pytest.approx(0.37)
    # And the same state loads into the scheduled form, so the modes interchange.
    scheduled = _module(blend=0.10, learnable=False)
    scheduled.load_state_dict(module.state_dict())
    assert float(scheduled.blend) == pytest.approx(0.37)

"""The sidecar needs its own learning rate, and the split must not lose parameters.

Stage 2 runs the backbone at 1e-5 and the sidecar barely moves at all -- the chained
run's `W_side_proj` went 2.1055 -> 2.1077 across an entire epoch, and the PLE module's
weights did not change to five significant figures over fifty logged steps. So the
comparison stage 2 was supposed to make is really "which frozen sidecar can a backbone
adapt around best". These pin the fix: every parameter still gets exactly one group, the
auxiliary ones get the requested rate, and the backbone keeps the run's.
"""

import pytest
import torch
from torch import nn

from distillkit.configuration import OptimizerConfig
from distillkit.gated_residual import GatedResidual
from distillkit.trainer import _apply_sidecar_lr

BACKBONE_LR, SIDECAR_LR = 1e-5, 1e-4


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(8, 8)
        self.layer_norm = nn.LayerNorm(8)
        self.sidecar = nn.Linear(8, 8, bias=False)
        self.gated_residual = GatedResidual(8, num_branches=2)
        self.distillation_projections = nn.ModuleList([nn.Linear(8, 8)])


def _optimizer(model):
    """Two groups, the way HF builds them: decayed matrices and undecayed vectors."""
    decay = [p for n, p in model.named_parameters() if p.ndim >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.ndim < 2]
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.01},
         {"params": no_decay, "weight_decay": 0.0}], lr=BACKBONE_LR,
    )


def _by_id(optimizer):
    return {id(p): group for group in optimizer.param_groups for p in group["params"]}


def test_every_parameter_still_belongs_to_exactly_one_group():
    """The split removes parameters from one group and re-adds them to another; losing
    or duplicating one would silently stop training it, or train it twice."""
    model = _Model()
    optimizer = _optimizer(model)
    before = sorted(id(p) for g in optimizer.param_groups for p in g["params"])

    _apply_sidecar_lr(optimizer, model, SIDECAR_LR)
    after = [id(p) for g in optimizer.param_groups for p in g["params"]]

    assert sorted(after) == before, "the split lost or duplicated a parameter"
    assert len(after) == len(set(after)), "a parameter ended up in two groups"


def test_the_sidecar_gets_its_rate_and_the_backbone_keeps_the_run_s():
    from distillkit.optimizers import architecture_parameter_ids

    model = _Model()
    optimizer = _optimizer(model)
    auxiliary = architecture_parameter_ids(model)
    _apply_sidecar_lr(optimizer, model, SIDECAR_LR)

    groups = _by_id(optimizer)
    for name, parameter in model.named_parameters():
        expected = SIDECAR_LR if id(parameter) in auxiliary else BACKBONE_LR
        assert groups[id(parameter)]["lr"] == expected, name
    # The sidecar is the point; if it were empty the whole exercise would be a no-op.
    assert any(id(p) in auxiliary for p in model.parameters())


def test_weight_decay_survives_the_move():
    """Auxiliary matrices are decayed and auxiliary vectors are not, exactly as before
    the split -- otherwise raising the rate would quietly change regularisation too."""
    model = _Model()
    optimizer = _optimizer(model)
    _apply_sidecar_lr(optimizer, model, SIDECAR_LR)

    groups = _by_id(optimizer)
    for name, parameter in model.named_parameters():
        expected = 0.01 if parameter.ndim >= 2 else 0.0
        assert groups[id(parameter)]["weight_decay"] == expected, name


def test_none_leaves_the_optimizer_untouched():
    model = _Model()
    optimizer = _optimizer(model)
    before = [list(g["params"]) for g in optimizer.param_groups]
    _apply_sidecar_lr(optimizer, model, None)
    assert [list(g["params"]) for g in optimizer.param_groups] == before


def test_the_sidecar_actually_moves_further():
    """The behaviour the flag exists for, end to end on real steps."""
    torch.manual_seed(0)
    results = {}
    for sidecar_lr in (None, SIDECAR_LR):
        torch.manual_seed(0)
        model = _Model()
        optimizer = _optimizer(model)
        _apply_sidecar_lr(optimizer, model, sidecar_lr)
        start = model.sidecar.weight.detach().clone()
        for _ in range(5):
            loss = (model.sidecar(torch.randn(2, 8)) ** 2).mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        results[sidecar_lr] = (model.sidecar.weight - start).abs().max().item()
    assert results[SIDECAR_LR] > 5 * results[None], results


def test_both_strategies_accept_a_sidecar_rate():
    """Hybrid used to be refused, because the rate was applied by adding a parameter
    group and MixedMuonAdamW disallows that. It is taken at construction now, so the
    refusal is gone and the sidecar gets its own bucket either way."""
    OptimizerConfig(strategy="adamw", sidecar_lr=1e-4)
    OptimizerConfig(strategy="hybrid", sidecar_lr=1e-4)
    OptimizerConfig(strategy="hybrid")


def test_the_hybrid_rate_covers_the_sidecar_and_not_the_widening_beside_it():
    """architecture_parameter_ids also names attn_residual/mlp_residual on every layer.
    A depth or scale comparison that moved those too would change two things at once."""
    import torch
    from distillkit.optimizers import mixed_parameter_groups, sidecar_module_parameter_ids
    from distillkit.models.qwen35_widened import Qwen35WidenedForCausalLM
    from tests.test_ple_gated_sidecar import _widened_config

    model = Qwen35WidenedForCausalLM(_widened_config(layers=3))
    sidecar = sidecar_module_parameter_ids(model)
    assert sidecar, "the fixture has a sidecar"
    widening = {id(p) for layer in model.model.layers
                for p in list(layer.attn_residual.parameters())
                + list(layer.mlp_residual.parameters())}
    assert not (sidecar & widening)

    groups = mixed_parameter_groups(model, sidecar_lr=1e-5)
    rated = {id(p) for g in groups if g.get("lr") == 1e-5 for p in g["params"]}
    assert rated == sidecar
    assert not (rated & widening)


def test_the_loss_scaffolding_keeps_the_backbone_rate():
    """The distillation projections exist only to compute the hidden-state term. Raising
    their rate lets them fit their own objective -- measured: their norms went 58.4/58.4
    at the base rate to 48.2/63.3 at 1e-3 while eval_loss improved monotonically and
    independent cross-entropy did not. They must not follow the sidecar's rate."""
    from distillkit.optimizers import _auxiliary_parameter_ids, architecture_parameter_ids

    model = _Model()
    optimizer = _optimizer(model)
    projections = {id(p) for p in model.distillation_projections.parameters()}
    assert projections <= _auxiliary_parameter_ids(model), "fixture must be auxiliary"
    assert not (projections & architecture_parameter_ids(model)), (
        "the projections are still counted as architecture; sidecar_lr would raise them"
    )

    _apply_sidecar_lr(optimizer, model, SIDECAR_LR)
    groups = _by_id(optimizer)
    for name, parameter in model.named_parameters():
        if "distillation_projections" in name:
            assert groups[id(parameter)]["lr"] == BACKBONE_LR, name
    # And the architecture still gets the override, or the fix removed the feature.
    assert groups[id(model.sidecar.weight)]["lr"] == SIDECAR_LR

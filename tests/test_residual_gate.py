"""A learned gate is only evidence if the model it starts from is the model we measured.

Every claim this experiment can make rests on three mechanical facts: a fresh gate is
exactly the identity, only the gate moves during the frozen stage, and forcing ``g = 1``
recovers the stock residual. If any of those slips, a content-NLL difference between the
gated run and the stock run is a difference between two unknown models.
"""

import numpy as np
import pytest
import torch

from distillkit.optimizers import architecture_parameter_ids, freeze_backbone_for_stage1
from distillkit.experimental.residual_gate import (
    FAMILIES, ResidualAdmissionGate, ResidualGateCheckpointCallback,
    TrigramFamiliarity, calibrate_gates, family_features, gate_parameter_count,
    install_residual_gates, load_gate_checkpoint, remove_residual_gates,
    residual_gates)
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

from tests.test_sidecar_model import tiny_config

GATED = (1, 2)


def build(seed=0):
    config = tiny_config()
    torch.manual_seed(seed)
    model = Qwen3_5ForCausalLM(config).eval()
    model.config.use_cache = False
    return model


def batch(model, batch_size=2, length=12, seed=3):
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, model.config.vocab_size, (batch_size, length),
                         generator=generator)


def logits_of(model, ids):
    with torch.no_grad():
        return model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits


def familiarity_for(model, tmp_path, keys=(7, 11, 13), counts=(4, 40, 400)):
    """A cache in the shape the memoisation study wrote, small enough to reason about."""
    path = tmp_path / "cache.npz"
    np.savez(path, keys=np.array(keys, dtype=np.int64),
             counts=np.array(counts, dtype=np.int64),
             mean=np.zeros((len(keys), 4), dtype=np.float16),
             variance=np.linspace(0.1, 0.9, len(keys)).astype(np.float64),
             global_mean=np.zeros(4, dtype=np.float32))
    return TrigramFamiliarity(path, model.config.vocab_size)


def calibrated(model, ids, family="geometry", tmp_path=None, **kwargs):
    """Install and calibrate on one batch; returns the handle, gates live."""
    if "log_count" in family_features(family):
        kwargs["familiarity"] = familiarity_for(model, tmp_path)
    handle = install_residual_gates(model, GATED, family=family, **kwargs)
    calibrate_gates(model, handle,
                    [{"input_ids": ids, "attention_mask": torch.ones_like(ids)}])
    return handle


# --- identity -----------------------------------------------------------------


def test_a_fresh_gate_returns_exactly_one():
    gate = ResidualAdmissionGate(4)
    gate.observe(torch.randn(32, 4))
    gate.finalize()
    values = gate(torch.randn(5, 7, 4))
    assert torch.equal(values, torch.ones(5, 7))


def test_a_calibrated_gate_leaves_the_model_bitwise_unchanged():
    model = build()
    ids = batch(model)
    reference = logits_of(model, ids)
    handle = calibrated(model, ids)
    try:
        assert torch.equal(reference, logits_of(model, ids))
        assert handle.gated_calls == ids.numel() * len(GATED)
    finally:
        remove_residual_gates(model)


def test_the_calibration_pass_itself_is_the_stock_model():
    model = build()
    ids = batch(model)
    reference = logits_of(model, ids)
    handle = install_residual_gates(model, GATED, family="geometry")
    try:
        handle.calibrating = True
        assert torch.equal(reference, logits_of(model, ids))
        handle.calibrating = False
    finally:
        remove_residual_gates(model)


def test_an_uncalibrated_gate_refuses_to_run():
    model = build()
    ids = batch(model)
    with residual_gates(model, GATED, family="geometry"):
        with pytest.raises(ValueError, match="no feature normalizer"):
            logits_of(model, ids)


def test_removing_the_gates_restores_the_original_forwards():
    model = build()
    ids = batch(model)
    reference = logits_of(model, ids)
    handle = calibrated(model, ids)
    handle.gate(GATED[0]).output.weight.data.fill_(1.0)
    assert not torch.equal(reference, logits_of(model, ids))
    remove_residual_gates(model)
    assert torch.equal(reference, logits_of(model, ids))
    assert gate_parameter_count(model) == 0


# --- range and scalar semantics -----------------------------------------------


@pytest.mark.parametrize("span", [0.5, 1.0])
def test_the_gate_stays_inside_its_declared_range(span):
    gate = ResidualAdmissionGate(4, span=span)
    gate.observe(torch.randn(32, 4))
    gate.finalize()
    # Drive the output layer hard in both directions; tanh has to hold the bound.
    for sign in (+1.0, -1.0):
        gate.output.weight.data.fill_(sign * 50.0)
        gate.output.bias.data.fill_(sign * 50.0)
        values = gate(torch.randn(64, 4) * 10)
        assert float(values.min()) > 1.0 - span - 1e-6
        assert float(values.max()) < 1.0 + span + 1e-6


def test_the_gate_is_one_scalar_per_token():
    gate = ResidualAdmissionGate(4)
    values = gate(torch.randn(3, 5, 4))
    assert values.shape == (3, 5)


def test_forcing_the_gate_to_one_reproduces_the_stock_residual():
    """The ablation the co-adaptation stage needs: same weights, admission forced off."""
    model = build()
    ids = batch(model)
    reference = logits_of(model, ids)
    handle = calibrated(model, ids)
    try:
        for index in GATED:
            handle.gate(index).output.weight.data.normal_(0, 1.0)
            handle.gate(index).output.bias.data.normal_(0, 1.0)
        assert not torch.equal(reference, logits_of(model, ids))
        handle.force_identity = True
        assert torch.equal(reference, logits_of(model, ids))
    finally:
        remove_residual_gates(model)


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_every_family_is_tiny_and_identity(family, tmp_path):
    model = build()
    ids = batch(model)
    handle = calibrated(model, ids, family=family, tmp_path=tmp_path)
    try:
        assert gate_parameter_count(model) < 100_000
        assert torch.equal(logits_of(model, ids), logits_of(build(), ids))
    finally:
        remove_residual_gates(model)


# --- familiarity features -----------------------------------------------------


def test_the_same_context_gives_the_same_familiarity_features(tmp_path):
    model = build()
    vocab = model.config.vocab_size
    statistics = familiarity_for(model, tmp_path, keys=(0,), counts=(9,))
    ids = torch.randint(0, vocab, (2, 16))
    first = statistics.features(ids)
    second = statistics.features(ids.clone())
    assert torch.equal(first, second)


def test_an_unseen_context_reads_as_unfamiliar(tmp_path):
    model = build()
    vocab = model.config.vocab_size
    # Key for position 2 of [0, 0, 1] is (0 * V + 0) * V + 1 == 1.
    statistics = familiarity_for(model, tmp_path, keys=(1,), counts=(99,))
    ids = torch.tensor([[0, 0, 1, 2]])
    features = statistics.features(ids)
    assert features[0, 2, 0] == pytest.approx(float(np.log1p(99)))
    assert features[0, 3, 0] == 0.0
    assert features[0, 3, 1] == 1.0
    # No trigram exists at the first two positions.
    assert features[0, 0].tolist() == [0.0, 1.0]
    assert features[0, 1].tolist() == [0.0, 1.0]


def test_a_familiarity_gate_refuses_to_be_built_without_statistics():
    model = build()
    with pytest.raises(ValueError, match="needs trigram familiarity"):
        install_residual_gates(model, GATED, family="familiarity")


# --- gradients ----------------------------------------------------------------


def loss_of(model, ids):
    out = model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=ids)
    return out.loss


def test_the_frozen_stage_trains_the_gate_and_nothing_else(tmp_path):
    model = build()
    ids = batch(model)
    handle = calibrated(model, ids, family="combined", tmp_path=tmp_path)
    try:
        frozen = freeze_backbone_for_stage1(model)
        assert frozen, "the backbone was already frozen, so this pins nothing"
        loss_of(model, ids).backward()
        for index in handle.layer_indices:
            gate = handle.gate(index)
            assert gate.output.weight.grad is not None
            assert torch.any(gate.output.weight.grad != 0)
        backbone = [(name, p) for name, p in model.named_parameters()
                    if not name.startswith("residual_gates.")]
        assert all(p.grad is None for _, p in backbone)
    finally:
        remove_residual_gates(model)


def test_co_adaptation_moves_both_the_backbone_and_the_gate():
    model = build()
    ids = batch(model)
    handle = calibrated(model, ids)
    try:
        loss_of(model, ids).backward()
        gate_grad = handle.gate(GATED[0]).output.weight.grad
        assert gate_grad is not None and torch.any(gate_grad != 0)
        embedding = model.get_input_embeddings().weight
        assert embedding.grad is not None and torch.any(embedding.grad != 0)
    finally:
        remove_residual_gates(model)


def test_the_control_arm_has_no_gate_parameters():
    model = build()
    assert gate_parameter_count(model) == 0
    assert getattr(model, "residual_gates", None) is None
    ids = batch(model)
    loss_of(model, ids).backward()
    assert not any(name.startswith("residual_gates.")
                   for name, _ in model.named_parameters())


def test_the_gate_counts_as_architecture_not_loss_scaffolding():
    """`freeze_backbone_for_stage1` finds the gate through this set."""
    model = build()
    ids = batch(model)
    handle = calibrated(model, ids)
    try:
        ids_of_gate = {id(p) for p in model.residual_gates.parameters()}
        assert ids_of_gate <= architecture_parameter_ids(model)
    finally:
        remove_residual_gates(model)


# --- installation guards ------------------------------------------------------


def test_gating_the_same_model_twice_is_refused():
    model = build()
    install_residual_gates(model, GATED, family="geometry")
    try:
        with pytest.raises(ValueError, match="already has residual gates"):
            install_residual_gates(model, GATED, family="geometry")
    finally:
        remove_residual_gates(model)


def test_a_layer_outside_the_stack_is_refused():
    model = build()
    with pytest.raises(ValueError, match="outside the model"):
        install_residual_gates(model, [model.config.num_hidden_layers],
                               family="geometry")


def test_an_unknown_family_is_refused():
    with pytest.raises(ValueError, match="unknown gate family"):
        family_features("vibes")


def test_calibration_runs_on_cpu():
    """Nothing in the gate path assumes CUDA; the whole of this file is the proof."""
    model = build()
    ids = batch(model)
    handle = calibrated(model, ids)
    try:
        for index in handle.layer_indices:
            gate = handle.gate(index)
            assert gate.is_calibrated
            assert float(gate.feature_std.min()) > 0
    finally:
        remove_residual_gates(model)


# --- warm start ---------------------------------------------------------------
#
# The co-adaptation curriculum starts the gate from a policy fitted to the frozen
# pretrained backbone rather than from the identity. That is only a different
# initialization if three things hold exactly: the backbone is untouched, the gate
# weights are the ones that were fitted, and the feature normalizer travels with them.
# A warm start that silently recalibrates is a different function wearing the same
# weights, and would look like a result.


def trained_gate(tmp_path, model, ids, family="familiarity", step=284):
    """A gate that has moved off identity, saved the way a run would save it."""
    handle = calibrated(model, ids, family=family, tmp_path=tmp_path)
    torch.manual_seed(11)
    for index in handle.layer_indices:
        gate = handle.gate(index)
        gate.output.weight.data.normal_(0, 0.4)
        gate.output.bias.data.normal_(0, 0.4)
        gate.project.weight.data.normal_(0, 0.4)
    callback = ResidualGateCheckpointCallback(str(tmp_path))
    callback._write(model, step)
    return handle, callback.written[-1]


def test_a_warm_start_loads_the_saved_weights_exactly(tmp_path):
    source = build()
    ids = batch(source)
    handle, path = trained_gate(tmp_path, source, ids)
    saved = {key: value.clone() for key, value in source.residual_gates.state_dict().items()}
    remove_residual_gates(source)

    target = build()
    statistics = familiarity_for(target, tmp_path)
    fresh = install_residual_gates(target, GATED, family="familiarity",
                                   familiarity=statistics)
    try:
        digest = load_gate_checkpoint(target, fresh, path)
        assert len(digest) == 64
        loaded = target.residual_gates.state_dict()
        assert set(loaded) == set(saved)
        for key, value in saved.items():
            assert torch.equal(loaded[key], value), key
    finally:
        remove_residual_gates(target)


def test_a_warm_start_carries_the_frozen_normalizer(tmp_path):
    """Recalibrating would change the function the loaded weights were fitted for."""
    source = build()
    ids = batch(source)
    handle, path = trained_gate(tmp_path, source, ids)
    means = {index: handle.gate(index).feature_mean.clone()
             for index in handle.layer_indices}
    deviations = {index: handle.gate(index).feature_std.clone()
                  for index in handle.layer_indices}
    remove_residual_gates(source)

    target = build()
    fresh = install_residual_gates(target, GATED, family="familiarity",
                                   familiarity=familiarity_for(target, tmp_path))
    try:
        assert not fresh.gate(GATED[0]).is_calibrated
        load_gate_checkpoint(target, fresh, path)
        for index in fresh.layer_indices:
            gate = fresh.gate(index)
            assert gate.is_calibrated
            assert torch.equal(gate.feature_mean, means[index])
            assert torch.equal(gate.feature_std, deviations[index])
    finally:
        remove_residual_gates(target)


def test_a_warm_started_model_reproduces_the_model_it_was_fitted_on(tmp_path):
    """Step zero of co-adaptation has to be the frozen-stage model, bit for bit."""
    source = build()
    ids = batch(source)
    handle, path = trained_gate(tmp_path, source, ids)
    reference = logits_of(source, ids)
    remove_residual_gates(source)

    target = build()
    fresh = install_residual_gates(target, GATED, family="familiarity",
                                   familiarity=familiarity_for(target, tmp_path))
    try:
        load_gate_checkpoint(target, fresh, path)
        assert torch.equal(reference, logits_of(target, ids))
    finally:
        remove_residual_gates(target)


def test_installing_a_gate_leaves_every_backbone_tensor_untouched(tmp_path):
    """Arms A, B and C must start from the same backbone; only the gate differs."""
    model = build()
    ids = batch(model)
    before = {name: parameter.clone()
              for name, parameter in model.named_parameters()}
    handle, path = trained_gate(tmp_path, model, ids)
    remove_residual_gates(model)

    fresh = install_residual_gates(model, GATED, family="familiarity",
                                   familiarity=familiarity_for(model, tmp_path))
    try:
        load_gate_checkpoint(model, fresh, path)
        for name, parameter in model.named_parameters():
            if name.startswith("residual_gates."):
                continue
            assert torch.equal(parameter, before[name]), name
    finally:
        remove_residual_gates(model)


def test_a_warm_start_refuses_a_checkpoint_from_other_layers(tmp_path):
    """A policy moved to depths it was never fitted for would not raise on shape."""
    source = build()
    ids = batch(source)
    _, path = trained_gate(tmp_path, source, ids)
    remove_residual_gates(source)

    target = build()
    other = (0, 2)
    fresh = install_residual_gates(target, other, family="familiarity",
                                   familiarity=familiarity_for(target, tmp_path))
    try:
        with pytest.raises(ValueError, match="covers layers"):
            load_gate_checkpoint(target, fresh, path)
    finally:
        remove_residual_gates(target)


def test_a_warm_start_refuses_a_checkpoint_from_another_family(tmp_path):
    source = build()
    ids = batch(source)
    _, path = trained_gate(tmp_path, source, ids, family="geometry")
    remove_residual_gates(source)

    target = build()
    fresh = install_residual_gates(target, GATED, family="familiarity",
                                   familiarity=familiarity_for(target, tmp_path))
    try:
        with pytest.raises(ValueError, match="family"):
            load_gate_checkpoint(target, fresh, path)
    finally:
        remove_residual_gates(target)


def test_a_warm_started_gate_still_trains_with_the_backbone(tmp_path):
    """Warm start changes where training begins, not what is trainable."""
    source = build()
    ids = batch(source)
    _, path = trained_gate(tmp_path, source, ids)
    remove_residual_gates(source)

    model = build()
    fresh = install_residual_gates(model, GATED, family="familiarity",
                                   familiarity=familiarity_for(model, tmp_path))
    try:
        load_gate_checkpoint(model, fresh, path)
        loss_of(model, ids).backward()
        for index in fresh.layer_indices:
            grad = fresh.gate(index).output.weight.grad
            assert grad is not None and torch.any(grad != 0)
        embedding = model.get_input_embeddings().weight
        assert embedding.grad is not None and torch.any(embedding.grad != 0)
    finally:
        remove_residual_gates(model)

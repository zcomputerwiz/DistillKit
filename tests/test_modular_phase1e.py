"""Focused correctness tests for Phase 1e category admission."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from experiments.modular_phase1.language import Vocabulary
from experiments.modular_phase1.specialist import EVENTS
from experiments.modular_phase1c.models import BIT_IDS
from experiments.modular_phase1e.models import (
    FEATURE_NAMES,
    ConstantAdmission,
    ContextualAdmission,
    apply_category_admission,
    build_contextual_features,
)


ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "experiments" / "modular_phase1e" / "configs" / "diagnostic.json"


def test_zero_initialization_and_inactive_output_are_exact():
    base = torch.log_softmax(torch.randn(9, len(Vocabulary.TOKENS)), -1)
    features = torch.randn(9, len(FEATURE_NAMES))
    model = ContextualAdmission()
    active = torch.tensor([True, False, True, True, False, True, False, True, True])
    output = apply_category_admission(base, model(features), active)
    assert torch.equal(output, base)
    inactive = torch.zeros_like(active)
    assert apply_category_admission(base, model(features), inactive) is base


def test_admission_preserves_both_within_category_distributions():
    base = torch.log_softmax(torch.randn(11, len(Vocabulary.TOKENS)), -1)
    delta = torch.linspace(-2, 3, 11)
    active = torch.ones(11, dtype=torch.bool)
    output = apply_category_admission(base, delta, active)
    nonbits = [index for index in range(len(Vocabulary.TOKENS)) if index not in BIT_IDS]
    assert torch.allclose(
        F.softmax(output[:, list(BIT_IDS)], -1),
        F.softmax(base[:, list(BIT_IDS)], -1), atol=1e-6, rtol=0,
    )
    assert torch.allclose(
        F.softmax(output[:, nonbits], -1),
        F.softmax(base[:, nonbits], -1), atol=1e-6, rtol=0,
    )
    assert not torch.allclose(
        output[:, list(BIT_IDS)].exp().sum(-1),
        base[:, list(BIT_IDS)].exp().sum(-1),
    )


def test_zero_correction_has_a_working_category_gradient():
    base = torch.log_softmax(torch.randn(12, len(Vocabulary.TOKENS)), -1)
    targets = torch.tensor([BIT_IDS[0]] * 6 + [Vocabulary.TO_ID["<sp1>"]] * 6)
    features = torch.randn(12, len(FEATURE_NAMES))
    model = ContextualAdmission()
    output = apply_category_admission(
        base, model(features), torch.ones(12, dtype=torch.bool)
    )
    loss = F.nll_loss(output, targets)
    loss.backward()
    assert model.affine.weight.grad is not None
    assert float(model.affine.weight.grad.norm()) > 0


def test_context_features_exclude_retrieved_literal_identity():
    batch, length = 2, 7
    ids = torch.full((batch, length), Vocabulary.TO_ID["a"], dtype=torch.long)
    ids[0, 2] = Vocabulary.TO_ID["0"]
    ids[1, 2] = Vocabulary.TO_ID["1"]
    backbone = torch.log_softmax(
        torch.randn(1, length, len(Vocabulary.TOKENS)).expand(batch, -1, -1), -1
    )
    events = torch.randn(1, length, len(EVENTS)).expand(batch, -1, -1)
    validity = torch.randn(1, length, 2).expand(batch, -1, -1)
    depths = torch.ones(batch, length, dtype=torch.long)
    features = build_contextual_features(ids, backbone, events, validity, depths)
    assert features.shape[-1] == len(FEATURE_NAMES) == 26
    assert torch.equal(features[0], features[1])
    signature = inspect.signature(build_contextual_features)
    assert "retrieved_value" not in signature.parameters
    assert "answer_position" not in signature.parameters


def test_arms_and_frozen_config_obey_bounds():
    constant = ConstantAdmission()
    contextual = ContextualAdmission()
    assert sum(parameter.numel() for parameter in constant.parameters()) == 1
    assert sum(parameter.numel() for parameter in contextual.parameters()) == 27
    assert sum(parameter.numel() for parameter in contextual.parameters()) <= 33
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert tuple(config["features"]) == FEATURE_NAMES
    assert config["training"]["max_updates"] <= 500
    assert config["training"]["max_target_exposures_per_arm"] <= 1_048_576
    assert config["budget"]["whole_run_accelerator_seconds"] <= 300

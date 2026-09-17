"""Focused invariants for the bounded Phase 1c output-facing lookup module."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import torch

from experiments.modular_phase1.language import Vocabulary, tokenize
from experiments.modular_phase1c.models import (
    BIT_IDS,
    DirectLookupReadout,
    LookupOutputIntegrator,
    compose_bit_distribution,
    deterministic_copy_log_probs,
    literal_representation,
    trace_lookup_prefix,
)


ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "experiments" / "modular_phase1c" / "configs" / "diagnostic.json"


def _score_position(ids: list[int]) -> int:
    return len(ids) - 3  # token before the answer in [..., context, answer, <eos>]


def test_lookup_activation_is_causal_and_includes_post_arrow_whitespace():
    ids = tokenize("{let a=1;?a;}=> 1")
    score = _score_position(ids)
    full = trace_lookup_prefix(ids)
    assert Vocabulary.TOKENS[ids[score]] in Vocabulary.WHITESPACE
    assert full.active[score]
    assert full.retrieved_values[score] == 1
    assert full.value_addresses[score] < score
    for length in range(1, len(ids) + 1):
        prefix = trace_lookup_prefix(ids[:length])
        assert prefix.active == full.active[:length]
        assert prefix.binding_addresses == full.binding_addresses[:length]
        assert prefix.value_addresses == full.value_addresses[:length]
        assert prefix.retrieved_values == full.retrieved_values[:length]
    assert "answer_position" not in inspect.signature(trace_lookup_prefix).parameters


def test_xor_is_inactive_and_answer_suffix_cannot_change_retrieval():
    xor = tokenize("{let a=1;let b=0;?a^b;}=>1")
    assert not any(trace_lookup_prefix(xor).active)
    zero = tokenize("{let a=1;?a;}=> 0")
    one = tokenize("{let a=1;?a;}=> 1")
    score = _score_position(zero)
    assert zero[: score + 1] == one[: score + 1]
    assert trace_lookup_prefix(zero).retrieved_values[: score + 1] == (
        trace_lookup_prefix(one).retrieved_values[: score + 1]
    )


def test_pointer_override_changes_the_retrieved_observed_literal():
    ids = tokenize("{let a=0;let b=1;?a;}=>0")
    base = trace_lookup_prefix(ids)
    use = max(index for index, pointer in enumerate(base.use_pointers) if pointer >= 0)
    declaration_b = ids.index(Vocabulary.TO_ID["b"])
    changed = trace_lookup_prefix(ids, pointer_override={use: declaration_b})
    score = _score_position(ids)
    assert base.retrieved_values[score] == 0
    assert changed.retrieved_values[score] == 1
    assert changed.value_addresses[score] < score


def test_composition_preserves_bit_mass_nonbits_and_inactive_identity():
    generator = torch.Generator().manual_seed(11)
    base = torch.log_softmax(torch.randn(7, len(Vocabulary.TOKENS), generator=generator), -1)
    reader = torch.log_softmax(torch.randn(7, 2, generator=generator), -1)
    active = torch.tensor([True, False, True, False, True, True, False])
    output = compose_bit_distribution(base, reader, active)
    nonbits = [index for index in range(len(Vocabulary.TOKENS)) if index not in BIT_IDS]
    assert torch.equal(output[~active], base[~active])
    assert torch.equal(output[:, nonbits], base[:, nonbits])
    assert torch.allclose(
        output[:, list(BIT_IDS)].exp().sum(-1),
        base[:, list(BIT_IDS)].exp().sum(-1),
        atol=1e-7,
        rtol=0,
    )
    inactive = torch.zeros_like(active)
    assert compose_bit_distribution(base, reader, inactive) is base


def test_copy_and_readout_are_width_independent_and_integrator_needs_no_labels():
    values = torch.tensor([0, 1])
    assert deterministic_copy_log_probs(values).argmax(-1).tolist() == [0, 1]
    assert literal_representation(values).shape == (2, 2)
    readout = DirectLookupReadout()
    assert sum(parameter.numel() for parameter in readout.parameters()) == 6
    signature = inspect.signature(LookupOutputIntegrator.forward)
    assert "answer_position" not in signature.parameters
    assert "target" not in signature.parameters


def test_inactive_integrator_is_exactly_the_backbone_distribution():
    ids = torch.tensor([tokenize("{let a=1;?a^a;}=>0")])
    mask = torch.ones_like(ids, dtype=torch.bool)
    logits = torch.randn(1, ids.shape[1], len(Vocabulary.TOKENS))
    module = LookupOutputIntegrator(DirectLookupReadout())
    output, traces = module(ids, mask, logits)
    expected = torch.log_softmax(logits.float(), -1)
    assert not any(traces[0].active)
    assert torch.equal(output, expected)


def test_config_is_one_seed_and_within_requested_caps():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["seed"] == 11
    assert config["budget"]["max_updates"] <= 500
    assert config["budget"]["max_supervised_answers"] <= 32_768
    assert config["budget"]["max_accelerator_seconds"] <= 300
    assert config["base_checkpoint"].endswith(
        "backbones/l6_w256/plain/seed_11/checkpoint.pt"
    )

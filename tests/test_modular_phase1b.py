"""Correctness tests for the isolated Phase 1b binding-reader diagnostic."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from experiments.modular_phase1.data import collate_documents
from experiments.modular_phase1.language import Vocabulary, generate_document
from experiments.modular_phase1.reference import ReferenceInterpreter
from experiments.modular_phase1.training import _state_hash
from experiments.modular_phase1b.models import build_paired_model
from experiments.modular_phase1b.answer_analysis import (
    decompose_answer_logits,
    verify_answer_positions,
)
from experiments.modular_phase1b.run import _declaration_value_position


ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "experiments" / "modular_phase1b" / "configs" / "diagnostic.json"


def _lookup_document():
    for index in range(200):
        document = generate_document(
            91, index, split="phase1b-test", force_literal=False
        )
        if document.metadata["query_kind"] == "lookup":
            return document
    raise AssertionError("failed to generate lookup document")


def test_reader_arms_have_identical_parameters_and_initialization():
    current = build_paired_model(seed=11, reader_mode="current_window")
    exact = build_paired_model(seed=11, reader_mode="exact_value")
    assert _state_hash(current.state_dict()) == _state_hash(exact.state_dict())
    assert sum(parameter.numel() for parameter in current.parameters()) == sum(
        parameter.numel() for parameter in exact.parameters()
    )


def test_exact_reader_selects_the_observed_value_inside_current_window():
    document = _lookup_document()
    binding = ReferenceInterpreter().interpret(document.text).bindings[0]
    value_position = _declaration_value_position(document, binding.declaration_position)
    batch = collate_documents([document])
    exact = build_paired_model(seed=11, reader_mode="exact_value").eval()
    consumed = exact.reader_inputs(batch.input_ids, batch.pointers)
    use = binding.use_position
    selected = consumed["indices"][0, use][consumed["weights"][0, use].bool()]
    assert selected.tolist() == [value_position]
    assert value_position < use


def test_fixed_format_bit_flip_changes_only_value_candidate_embedding():
    document = _lookup_document()
    binding = ReferenceInterpreter().interpret(document.text).bindings[0]
    value_position = _declaration_value_position(document, binding.declaration_position)
    batch = collate_documents([document])
    flipped = batch.input_ids.clone()
    flipped[0, value_position] = (
        Vocabulary.TO_ID["1"]
        if int(flipped[0, value_position]) == Vocabulary.TO_ID["0"]
        else Vocabulary.TO_ID["0"]
    )
    model = build_paired_model(seed=11, reader_mode="current_window").eval()
    before = model.reader_inputs(batch.input_ids, batch.pointers)
    after = model.reader_inputs(flipped, batch.pointers)
    use = binding.use_position
    changed = (
        after["candidates"][0, use] - before["candidates"][0, use]
    ).norm(dim=-1) > 0
    selected_positions = before["indices"][0, use][changed]
    assert selected_positions.tolist() == [value_position]
    assert torch.equal(before["indices"], after["indices"])


def test_phase1b_config_is_single_size_seed_and_bounded():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["seed"] == 11
    assert (config["model"]["layers"], config["model"]["width"]) == (2, 64)
    assert config["budget"]["targets_per_run"] == 1_048_576
    assert config["budget"]["whole_diagnostic_accelerator_seconds"] == 600
    assert config["model"]["arms"] == ["current_window", "exact_value"]


def test_answer_nll_decomposes_into_category_and_selection():
    logits = torch.linspace(-2.0, 2.0, len(Vocabulary.TOKENS))
    logits[Vocabulary.TO_ID["0"]] = 1.5
    logits[Vocabulary.TO_ID["1"]] = 0.5
    result = decompose_answer_logits(logits, Vocabulary.TO_ID["0"])
    assert abs(
        result["answer_nll"]
        - result["bit_category_nll"]
        - result["correct_bit_given_category_nll"]
    ) < 1e-6
    assert result["conditional_bit_correct"] == 1


def test_answer_position_examples_verify_shifted_causal_target():
    documents = []
    for index in range(100):
        document = generate_document(
            97, index, split="phase1b-answer-test", force_literal=False
        )
        if document.metadata["query_kind"] == "lookup":
            documents.append(document)
        if len(documents) == 8:
            break
    examples = verify_answer_positions(documents, count=8)
    assert examples
    assert all(example["shift_identity_verified"] for example in examples)
    assert all(example["scored_logits_position"] + 1 == example["answer_position"]
               for example in examples)

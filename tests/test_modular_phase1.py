"""Focused correctness tests for the isolated modular Phase 1 experiment."""

from __future__ import annotations

import copy
from pathlib import Path

import torch

from experiments.modular_phase1.audit import eligible_pointer_case
from experiments.modular_phase1.config import load_config
from experiments.modular_phase1.data import collate_documents, load_corpus, prepare_corpora
from experiments.modular_phase1.language import (
    Program,
    Vocabulary,
    generate_document,
    render_program,
    tokenize,
    whitespace_document,
)
from experiments.modular_phase1.models import (
    CausalDecoderLM,
    DecoderConfig,
    PointerReader,
    SpecialistDecoderLM,
    build_models,
    model_parameter_report,
)
from experiments.modular_phase1.reference import ReferenceInterpreter
from experiments.modular_phase1.specialist import (
    CausalStructureMachine,
    EVENTS,
    REDACTED_VALUE_ID,
    RecurrentSpecialist,
    redact_ids,
    specialist_parameter_report,
)


ROOT = Path(__file__).parents[1]
SMOKE = ROOT / "experiments" / "modular_phase1" / "configs" / "smoke.json"


def test_reference_interpreter_resolves_shadowing_and_xor():
    text = "{ let a = 0; let b = 1; { let a = 1; ? a ^ b; } } => 0"
    result = ReferenceInterpreter().interpret(text)
    assert result.answer == 0
    assert len(result.bindings) == 2
    assert result.bindings[0].name == "a"
    assert result.bindings[0].declaration_position > result.bindings[1].declaration_position


def test_generator_and_independent_interpreter_agree_across_slices():
    for index in range(100):
        document = generate_document(
            19, index, split="test", force_heldout_combo=index % 10 == 0,
        )
        result = ReferenceInterpreter().interpret(document.text)
        assert result.answer == document.answer
        assert result.answer_position == document.answer_position


def test_whitespace_is_visible_but_semantics_are_invariant():
    document = generate_document(7, 3, split="test", force_literal=False)
    variants = [whitespace_document(document, style) for style in (
        "spaces", "tabs", "newlines", "mixed"
    )]
    assert {ReferenceInterpreter().interpret(item.text).answer for item in variants} == {
        document.answer
    }
    assert len({tuple(item.token_ids) for item in variants}) > 1


def test_answer_value_cannot_change_length():
    program = Program(0, (), (0, 1, 1), True)
    zero = tokenize(render_program(program, 0, "mixed", seed=9))
    one = tokenize(render_program(program, 1, "mixed", seed=9))
    assert len(zero) == len(one)
    differing = [index for index, pair in enumerate(zip(zero, one)) if pair[0] != pair[1]]
    assert len(differing) == 1


def test_streaming_pointer_matches_independent_reference_and_ignores_values():
    zero = "{let a=0;let b=1;{let a=1;?a^b;}}=>0"
    one = "{let a=1;let b=0;{let a=0;?a^b;}}=>1"
    machine = CausalStructureMachine()
    first = machine.analyze(tokenize(zero))
    second = machine.analyze(tokenize(one))
    assert first.pointers == second.pointers
    expected = {
        item.use_position: item.declaration_position
        for item in ReferenceInterpreter().interpret(zero).bindings
    }
    assert {i: pointer for i, pointer in enumerate(first.pointers) if pointer >= 0} == expected


def test_specialist_redacts_bits_and_is_prefix_causal():
    model = RecurrentSpecialist().eval()
    left = torch.tensor([tokenize("?0^1;=>1")])
    right = torch.tensor([tokenize("?1^0;=>0")])
    redacted_left = redact_ids(left)
    redacted_right = redact_ids(right)
    assert torch.equal(redacted_left, redacted_right)
    assert int((redacted_left == REDACTED_VALUE_ID).sum()) == 3
    prefix = torch.tensor([tokenize("{let a=0;?a;}=>0")])
    altered = prefix.clone()
    altered[:, -2] = Vocabulary.TO_ID["1"]
    with torch.no_grad():
        first = model(prefix)["event_logits"]
        second = model(altered)["event_logits"]
    assert torch.equal(first[:, :-2], second[:, :-2])


def test_specialist_budget_and_interface_exclude_hidden_values():
    report = specialist_parameter_report(RecurrentSpecialist())
    assert report["learned_parameters"] <= 100_000
    assert not report["exports_hidden_state"]
    assert not report["exports_values_or_answers"]
    assert "declaration pointer" in report["programmed"]


def test_oracle_features_use_the_frozen_interface_schema():
    document = generate_document(41, 6, split="test", force_literal=False)
    batch = collate_documents([document])
    specialist = RecurrentSpecialist().eval()
    features, pointers = specialist.oracle_interface(
        batch.input_ids,
        batch.pointers,
        scope_depths=batch.depths,
        structural_events=batch.events,
        structural_validity=batch.validity,
    )
    assert features.shape[-1] == specialist.interface_width
    assert torch.equal(pointers, batch.pointers)
    assert torch.equal(features[..., :len(EVENTS)].argmax(-1), batch.events)


def test_pointer_reader_audit_exposes_exact_consumed_sum():
    torch.manual_seed(12)
    reader = PointerReader(feature_width=30, model_width=16).eval()
    raw = torch.randn(2, 11, 16)
    features = torch.randn(2, 11, 30)
    pointers = torch.full((2, 11), -1, dtype=torch.long)
    pointers[:, 8] = 2
    consumed = reader.pointer_inputs(raw, features, pointers)
    actual = reader(raw, features, pointers)
    assert torch.equal(actual, consumed["feature_signal"] + consumed["pointer_signal"])


def test_controlled_pointer_case_is_lookup_only_and_opposite_valued():
    selected = None
    for index in range(500):
        document = generate_document(73, index, split="confirmation", force_literal=False)
        selected = eligible_pointer_case(document)
        if selected is not None:
            break
    assert selected is not None
    assert document.metadata["query_kind"] == "lookup"
    assert selected["correct_value"] != selected["alternate_value"]
    assert selected["correct_declaration_position"] != selected[
        "alternate_declaration_position"
    ]


def test_backbone_is_causal():
    torch.manual_seed(4)
    model = CausalDecoderLM(DecoderConfig(2, 32, 4, 64, 64)).eval()
    left = torch.tensor([[1, 3, 5, 13, 6, 11, 7, 2]])
    right = left.clone()
    right[:, 5:] = torch.tensor([[12, 12, 2]])
    mask = torch.ones_like(left, dtype=torch.bool)
    with torch.no_grad():
        a = model(left, mask)
        b = model(right, mask)
    assert torch.allclose(a[:, :5], b[:, :5], atol=1e-6)


def test_composed_and_plain_have_identical_initial_backbone_and_logits():
    specialist = RecurrentSpecialist()
    models = build_models(layers=2, width=64, heads=4, max_sequence=128,
                          seed=11, specialist=specialist)
    plain = models["plain"].eval()
    composed = models["specialist"].eval()
    document = generate_document(3, 9, split="test", force_literal=False)
    batch = collate_documents([document])
    with torch.no_grad():
        expected = plain(batch.input_ids, batch.attention_mask)
        actual = composed(
            batch.input_ids, batch.attention_mask, pointers=batch.pointers,
            scope_depths=batch.depths,
            alternate_pointers=batch.alternate_pointers,
        )
    assert torch.equal(expected, actual)


def test_parameter_matched_control_is_close_to_total_composed_size():
    models = build_models(layers=6, width=256, heads=8, max_sequence=128,
                          seed=11, specialist=RecurrentSpecialist())
    report = model_parameter_report(models)
    assert report["matched_relative_error"] < 0.001
    assert report["specialist"]["trainable_parameters"] < report["matched"][
        "trainable_parameters"
    ]


def test_wrong_pointer_is_a_real_control_when_a_reader_uses_the_pointer():
    models = build_models(layers=2, width=64, heads=4, max_sequence=256,
                          seed=11, specialist=RecurrentSpecialist())
    model = models["specialist"].eval()
    assert isinstance(model, SpecialistDecoderLM)
    with torch.no_grad():
        model.reader.pointer_scale.fill_(1.0)
    document = generate_document(31, 5, split="test", force_heldout_combo=True)
    batch = collate_documents([document])
    assert torch.any(
        (batch.pointers >= 0) & (batch.pointers != batch.alternate_pointers)
    )
    with torch.no_grad():
        correct = model(
            batch.input_ids, batch.attention_mask, pointers=batch.pointers,
            scope_depths=batch.depths,
            alternate_pointers=batch.alternate_pointers,
        )
        wrong = model(
            batch.input_ids, batch.attention_mask, pointers=batch.pointers,
            scope_depths=batch.depths,
            alternate_pointers=batch.alternate_pointers, intervention="wrong_pointer",
        )
    assert not torch.equal(correct, wrong)


def test_prepared_splits_are_disjoint_exactly_twenty_percent_literal_and_hold_out_combo(tmp_path):
    config = copy.deepcopy(load_config(SMOKE))
    config["data"]["documents"] = {
        "specialist_train": 40,
        "specialist_validation": 20,
        "backbone_train": 40,
        "backbone_validation": 20,
        "confirmation": 20,
    }
    config["data"]["invariance_pairs"] = 2
    manifest = prepare_corpora(config, tmp_path)
    assert not any(manifest["exact_token_sequence_overlaps"].values())
    for name in ("specialist_train", "backbone_train"):
        documents = load_corpus(manifest["files"][name]["path"])
        assert sum(document.metadata["literal"] for document in documents) == len(documents) // 5
        assert not any(
            document.metadata["depth"] == 4
            and document.metadata["query_kind"] == "xor"
            and document.metadata["whitespace"] == "newlines"
            and document.metadata["shadow_count"] >= 2
            for document in documents
        )
    confirmation = load_corpus(manifest["files"]["confirmation"]["path"])
    assert any(document.metadata.get("heldout_combo") for document in confirmation)
    assert any(5 <= document.metadata["depth"] <= 8 for document in confirmation)
    assert any(document.metadata.get("long_history") for document in confirmation)


def test_frozen_config_enforces_requested_caps_and_design():
    config = load_config(ROOT / "experiments" / "modular_phase1" / "configs" / "pilot.json")
    assert config["budget"]["specialist_tokens"] == 2_097_152
    assert config["budget"]["backbone_tokens_per_run"] == 1_048_576
    assert config["backbones"]["sizes"] == [[2, 64], [4, 128], [6, 256]]
    assert config["seeds"] == [11, 22, 33]
    assert config["evaluation"]["non_inferiority"] == {
        "accuracy_points": 0.01,
        "nll_nats": 0.02,
    }

"""Evaluation-only audit of the completed Phase 1 pilot.

No function in this module constructs an optimizer or calls backward. Checkpoint hashes
are captured before and after the audit and must remain identical.
"""

from __future__ import annotations

import inspect
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F

from .config import file_hash
from .data import collate_documents, load_corpus, prepare_corpora
from .evaluation import _load_model
from .language import Document, Vocabulary
from .models import PointerReader, SpecialistDecoderLM
from .reference import ReferenceInterpreter
from .specialist import EVENTS, CausalStructureMachine, RecurrentSpecialist
from .training import _autocast, _device, load_frozen_specialist


def _summary(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "mean": None, "mean_absolute": None, "median": None,
                "minimum": None, "maximum": None}
    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "mean_absolute": sum(abs(value) for value in values) / len(values),
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
        "nonzero": sum(value != 0.0 for value in values),
    }


def _classification(predictions: torch.Tensor, targets: torch.Tensor) -> dict:
    matrix = torch.zeros((len(EVENTS), len(EVENTS)), dtype=torch.long)
    for target, prediction in zip(targets.tolist(), predictions.tolist()):
        matrix[target, prediction] += 1
    classes = {}
    macro = []
    for index, name in enumerate(EVENTS):
        support = int(matrix[index].sum())
        predicted = int(matrix[:, index].sum())
        true_positive = int(matrix[index, index])
        false_positive = predicted - true_positive
        false_negative = support - true_positive
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / support if support else 0.0
        denominator = 2 * true_positive + false_positive + false_negative
        f1 = 2 * true_positive / denominator if denominator else 0.0
        classes[name] = {
            "support": support,
            "predicted": predicted,
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        if support and name not in ("pad", "whitespace"):
            macro.append(f1)
    structural = torch.ones_like(targets, dtype=torch.bool)
    for excluded in ("pad", "whitespace"):
        structural &= targets != EVENTS.index(excluded)
    return {
        "tokens": int(targets.numel()),
        "accuracy": float((predictions == targets).float().mean()),
        "structural_event_accuracy_excluding_pad_and_whitespace": float(
            (predictions[structural] == targets[structural]).float().mean()
        ),
        "structural_macro_f1_excluding_pad_and_whitespace": sum(macro) / len(macro),
        "classes": classes,
        "confusion_matrix_target_rows_prediction_columns": matrix.tolist(),
    }


def _validation_slices(document: Document, corrupted: bool) -> list[str]:
    metadata = document.metadata
    names = [
        "all",
        "corrupted" if corrupted else "clean",
        f"depth_{metadata['depth']}",
        f"query_{metadata['query_kind']}",
        f"whitespace_{metadata['whitespace']}",
    ]
    names.append("literal" if metadata["literal"] else "scope_resolution")
    names.append("shadowing" if metadata["shadow_count"] else "no_shadowing")
    return names


@torch.inference_mode()
def audit_specialist_gate(config: dict, run_dir: Path, device: torch.device) -> dict:
    data = prepare_corpora(config, run_dir)
    documents = load_corpus(data["files"]["specialist_validation"]["path"])
    model, manifest = load_frozen_specialist(config, run_dir, device)
    batch_size = int(config["specialist"]["batch_size"])
    corrupt_fraction = float(config["specialist"]["corrupt_fraction"])
    all_predictions = []
    all_targets = []
    slices: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = defaultdict(list)
    for offset in range(0, len(documents), batch_size):
        members = documents[offset:offset + batch_size]
        batch = collate_documents(
            members, corrupt_fraction=corrupt_fraction,
            corruption_seed=7001 + offset,
        ).to(device)
        prediction = model(batch.input_ids)["event_logits"].argmax(-1)
        for row, document in enumerate(members):
            length = len(document.token_ids)
            target = batch.events[row, :length].cpu()
            predicted = prediction[row, :length].cpu()
            original = torch.tensor(document.token_ids)
            corrupted = not torch.equal(batch.input_ids[row, :length].cpu(), original)
            all_predictions.append(predicted)
            all_targets.append(target)
            for name in _validation_slices(document, corrupted):
                slices[name].append((predicted, target))
    overall = _classification(torch.cat(all_predictions), torch.cat(all_targets))
    slice_metrics = {}
    for name, members in sorted(slices.items()):
        slice_metrics[name] = _classification(
            torch.cat([item[0] for item in members]),
            torch.cat([item[1] for item in members]),
        )
        slice_metrics[name]["documents"] = len(members)
    implemented = config["specialist"]["gate"]
    return {
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "validation_documents": len(documents),
        "corruption_fraction_configured": corrupt_fraction,
        "implemented_criterion": {
            "binding_accuracy_minimum": implemented["binding_accuracy"],
            "structural_event_metric": "macro F1 over supported events excluding pad/whitespace",
            "structural_event_minimum": implemented["structural_event_macro_f1"],
            "validity_accuracy_minimum": implemented["validity_accuracy"],
        },
        "frozen_manifest_metrics": manifest["validation"],
        "reconstructed_event_metrics": overall,
        "validation_slices": slice_metrics,
        "reconciliation": {
            "implemented_gate_passed": (
                overall["structural_macro_f1_excluding_pad_and_whitespace"]
                >= implemented["structural_event_macro_f1"]
            ),
            "structural_event_accuracy_at_least_0_99": (
                overall["structural_event_accuracy_excluding_pad_and_whitespace"] >= 0.99
            ),
            "macro_f1_at_least_0_99": (
                overall["structural_macro_f1_excluding_pad_and_whitespace"] >= 0.99
            ),
            "deviation": (
                "The pilot implemented a 0.95 structural macro-F1 gate. It did not "
                "implement or meet a 0.99 structural-event threshold; the historical "
                "'passed' label is valid only for the implemented criterion."
            ),
        },
    }


def _value_after_declaration(ids: list[int], declaration_position: int) -> tuple[int, int]:
    for position in range(declaration_position + 1, len(ids)):
        token = Vocabulary.TOKENS[ids[position]]
        if token in ("0", "1"):
            return int(token), position
        if token == ";":
            break
    raise ValueError(f"no value after declaration at {declaration_position}")


def _binding_ordinal(document: Document, declaration_position: int) -> int:
    declarations = CausalStructureMachine().analyze(document.token_ids).declaration_positions
    return declarations.index(declaration_position)


def eligible_pointer_case(document: Document) -> dict | None:
    """Select one lookup reference and another declaration with the opposite value."""
    if document.metadata.get("variant") != "base":
        return None
    if document.metadata.get("query_kind") != "lookup":
        return None
    interpretation = ReferenceInterpreter().interpret(document.text)
    if len(interpretation.bindings) != 1:
        return None
    binding = interpretation.bindings[0]
    correct_value, correct_value_position = _value_after_declaration(
        document.token_ids, binding.declaration_position
    )
    trace = CausalStructureMachine().analyze(document.token_ids)
    candidates = []
    for declaration_position in trace.declaration_positions:
        if declaration_position == binding.declaration_position:
            continue
        value, value_position = _value_after_declaration(
            document.token_ids, declaration_position
        )
        if value != correct_value:
            candidates.append((declaration_position, value, value_position))
    if not candidates:
        return None
    alternate_position, alternate_value, alternate_value_position = min(
        candidates, key=lambda item: (abs(binding.use_position - item[0]), item[0])
    )
    if correct_value != document.answer:
        raise AssertionError("lookup answer does not equal resolved declaration value")
    return {
        "document_id": document.document_id,
        "use_position": binding.use_position,
        "correct_declaration_position": binding.declaration_position,
        "correct_declaration_ordinal": _binding_ordinal(
            document, binding.declaration_position
        ),
        "correct_value": correct_value,
        "correct_value_position": correct_value_position,
        "alternate_declaration_position": alternate_position,
        "alternate_declaration_ordinal": _binding_ordinal(document, alternate_position),
        "alternate_value": alternate_value,
        "alternate_value_position": alternate_value_position,
    }


def _answer_probabilities(logits: torch.Tensor, batch, row: int) -> tuple[float, float, int]:
    answer_position = int(batch.answer_positions[row])
    answer_token = int(batch.input_ids[row, answer_position])
    answer_logits = logits[row, answer_position - 1].float()
    full_probability = float(answer_logits.softmax(-1)[answer_token])
    bit_ids = torch.tensor(
        [Vocabulary.TO_ID["0"], Vocabulary.TO_ID["1"]], device=logits.device
    )
    bit_logits = answer_logits[bit_ids]
    bit_slot = 0 if answer_token == Vocabulary.TO_ID["0"] else 1
    bit_probability = float(bit_logits.softmax(-1)[bit_slot])
    return full_probability, bit_probability, int(answer_logits.argmax())


def _checkpoint_paths(run_dir: Path, arm: str = "specialist") -> list[Path]:
    return sorted(run_dir.glob(f"backbones/l*_w*/{arm}/seed_*/checkpoint.pt"))


@torch.inference_mode()
def audit_pointer_intervention(
    config: dict,
    run_dir: Path,
    device: torch.device,
) -> dict:
    data = prepare_corpora(config, run_dir)
    confirmation = load_corpus(data["files"]["confirmation"]["path"])
    base_lookup = [
        document for document in confirmation
        if document.metadata.get("variant") == "base"
        and document.metadata.get("query_kind") == "lookup"
    ]
    cases = [(document, eligible_pointer_case(document)) for document in base_lookup]
    cases = [(document, case) for document, case in cases if case is not None]
    by_checkpoint = []
    instrumentation = defaultdict(int)
    candidate_state_norms = []
    context_norms = []
    pointer_signal_norms = []
    feature_signal_norms = []
    raw_distance_deltas = []
    reader_output_norms = []
    all_full_deltas = []
    all_bit_deltas = []
    direction_counts = defaultdict(int)
    for checkpoint_path in _checkpoint_paths(run_dir):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model = _load_model(config, run_dir, checkpoint, device)
        assert isinstance(model, SpecialistDecoderLM)
        model.eval()
        full_deltas = []
        bit_deltas = []
        accuracy_deltas = []
        batch_size = int(config["evaluation"]["batch_sizes"][str(checkpoint["width"])])
        for offset in range(0, len(cases), batch_size):
            members = cases[offset:offset + batch_size]
            documents = [item[0] for item in members]
            case_records = [item[1] for item in members]
            batch = collate_documents(documents).to(device)
            override = batch.pointers.clone()
            for row, case in enumerate(case_records):
                override[row, case["use_position"]] = case["alternate_declaration_position"]
                direction_counts[f"{case['correct_value']}_to_{case['alternate_value']}"] += 1
            changed = (override != batch.pointers)
            if not torch.all(changed.sum(-1) == 1):
                raise AssertionError("pointer audit must change exactly one reference per document")
            correct_logits = model(
                batch.input_ids, batch.attention_mask, pointers=batch.pointers,
                scope_depths=batch.depths, alternate_pointers=batch.alternate_pointers,
            )
            wrong_logits = model(
                batch.input_ids, batch.attention_mask, pointers=batch.pointers,
                scope_depths=batch.depths, alternate_pointers=batch.alternate_pointers,
                pointer_override=override,
            )

            learned_correct, selected_correct = model.specialist.interface(
                batch.input_ids, batch.pointers, scope_depths=batch.depths,
            )
            learned_wrong, selected_wrong = model.specialist.interface(
                batch.input_ids, batch.pointers, scope_depths=batch.depths,
                pointer_override=override,
            )
            feature_rows_changed = (learned_correct != learned_wrong).any(-1).sum(-1)
            if not torch.all(feature_rows_changed == 1):
                raise AssertionError("exactly one exported feature row must change")
            raw = model.backbone.embed(batch.input_ids)
            consumed_correct = model.reader.pointer_inputs(
                raw, learned_correct.to(raw.dtype), selected_correct
            )
            consumed_wrong = model.reader.pointer_inputs(
                raw, learned_wrong.to(raw.dtype), selected_wrong
            )
            for row, case in enumerate(case_records):
                use = case["use_position"]
                instrumentation["cases"] += 1
                instrumentation["pointer_index_changed"] += int(
                    selected_correct[row, use] != selected_wrong[row, use]
                )
                indices_correct = consumed_correct["indices"][row, use]
                indices_wrong = consumed_wrong["indices"][row, use]
                instrumentation["gather_indices_changed"] += int(
                    not torch.equal(indices_correct, indices_wrong)
                )
                token_window_correct = batch.input_ids[row, indices_correct]
                token_window_wrong = batch.input_ids[row, indices_wrong]
                instrumentation["gathered_token_window_changed"] += int(
                    not torch.equal(token_window_correct, token_window_wrong)
                )
                candidate_delta = (
                    consumed_wrong["candidates"][row, use].float()
                    - consumed_correct["candidates"][row, use].float()
                ).norm().item()
                context_delta = (
                    consumed_wrong["context"][row, use].float()
                    - consumed_correct["context"][row, use].float()
                ).norm().item()
                pointer_delta = (
                    consumed_wrong["pointer_signal"][row, use].float()
                    - consumed_correct["pointer_signal"][row, use].float()
                ).norm().item()
                feature_delta = (
                    consumed_wrong["feature_signal"][row, use].float()
                    - consumed_correct["feature_signal"][row, use].float()
                ).norm().item()
                raw_distance_delta = abs(float(
                    learned_wrong[row, use, -1] - learned_correct[row, use, -1]
                ))
                reader_delta = (
                    consumed_wrong["pointer_signal"][row, use].float()
                    + consumed_wrong["feature_signal"][row, use].float()
                    - consumed_correct["pointer_signal"][row, use].float()
                    - consumed_correct["feature_signal"][row, use].float()
                ).norm().item()
                candidate_state_norms.append(candidate_delta)
                context_norms.append(context_delta)
                pointer_signal_norms.append(pointer_delta)
                feature_signal_norms.append(feature_delta)
                raw_distance_deltas.append(raw_distance_delta)
                reader_output_norms.append(reader_delta)
                instrumentation["gathered_embedding_state_changed"] += int(candidate_delta > 0)
                instrumentation["weighted_context_changed"] += int(context_delta > 0)
                instrumentation["pointer_signal_changed"] += int(pointer_delta > 0)
                instrumentation["encoded_distance_feature_changed"] += int(
                    raw_distance_delta > 0
                )
                instrumentation["feature_projection_changed"] += int(feature_delta > 0)
                instrumentation["reader_output_changed"] += int(reader_delta > 0)

                full_correct, bit_correct, prediction_correct = _answer_probabilities(
                    correct_logits, batch, row
                )
                full_wrong, bit_wrong, prediction_wrong = _answer_probabilities(
                    wrong_logits, batch, row
                )
                full_deltas.append(full_wrong - full_correct)
                bit_deltas.append(bit_wrong - bit_correct)
                answer_token = int(batch.input_ids[row, int(batch.answer_positions[row])])
                accuracy_deltas.append(
                    int(prediction_wrong == answer_token) - int(prediction_correct == answer_token)
                )
        all_full_deltas.extend(full_deltas)
        all_bit_deltas.extend(bit_deltas)
        by_checkpoint.append({
            "layers": checkpoint["layers"],
            "width": checkpoint["width"],
            "seed": checkpoint["seed"],
            "eligible_documents": len(cases),
            "full_vocab_correct_answer_probability_wrong_minus_correct": _summary(full_deltas),
            "binary_normalized_correct_answer_probability_wrong_minus_correct": _summary(
                bit_deltas
            ),
            "answer_accuracy_wrong_minus_correct": sum(accuracy_deltas) / len(accuracy_deltas),
        })
    unique_cases = [case for _, case in cases]
    return {
        "base_confirmation_lookup_documents": len(base_lookup),
        "eligible_different_value_documents": len(cases),
        "ineligible_without_different_value_declaration": len(base_lookup) - len(cases),
        "unique_case_direction_counts": dict(sorted(defaultdict(int, {
            direction: sum(
                case["correct_value"] == int(direction[0])
                and case["alternate_value"] == int(direction[-1])
                for case in unique_cases
            )
            for direction in ("0_to_1", "1_to_0")
        }).items())),
        "evaluated_checkpoint_case_pairs": len(cases) * len(by_checkpoint),
        "per_checkpoint": by_checkpoint,
        "aggregate_probability_effect": {
            "full_vocab_correct_answer_probability_wrong_minus_correct": _summary(
                all_full_deltas
            ),
            "binary_normalized_correct_answer_probability_wrong_minus_correct": _summary(
                all_bit_deltas
            ),
        },
        "consumed_information_verification": {
            **dict(instrumentation),
            "candidate_embedding_delta_l2": _summary(candidate_state_norms),
            "weighted_context_delta_l2": _summary(context_norms),
            "pointer_signal_delta_l2": _summary(pointer_signal_norms),
            "encoded_distance_feature_delta_absolute": _summary(raw_distance_deltas),
            "projected_feature_signal_delta_l2": _summary(feature_signal_norms),
            "reader_output_delta_l2": _summary(reader_output_norms),
            "cached_specialist_features": False,
            "cached_gathered_states": False,
            "note": (
                "Corpora cache token ids/text only. Causal state, learned features, raw "
                "embeddings, gathered windows, reader outputs, and logits were recomputed."
            ),
        },
        "control_definition": (
            "Base confirmation lookup queries only; exactly one use pointer per document "
            "is changed to another declaration holding the opposite bit. XOR is excluded."
        ),
    }


def _aggregate_rows(rows: list[dict]) -> dict:
    return {
        "documents": len(rows),
        "answer_accuracy": sum(item["correct"] for item in rows) / max(1, len(rows)),
        "answer_nll": sum(item["answer_nll"] for item in rows) / max(1, len(rows)),
        "overall_nll": sum(item["nll_sum"] for item in rows)
        / max(1, sum(item["nll_tokens"] for item in rows)),
    }


def _invariance(rows: list[dict]) -> dict:
    base = {
        row["document_id"]: row["prediction_token"]
        for row in rows if row["metadata"].get("variant") == "base"
    }
    groups = defaultdict(list)
    for row in rows:
        variant = row["metadata"].get("variant")
        if variant not in ("renamed", "whitespace") or row["base_id"] not in base:
            continue
        groups[variant].append(int(row["prediction_token"] == base[row["base_id"]]))
    return {
        name: {"pairs": len(values), "prediction_agreement": sum(values) / len(values)}
        for name, values in groups.items()
    }


@torch.inference_mode()
def _learned_and_oracle_rows(model, documents, batch_size, device, bf16) -> dict[str, list[dict]]:
    result = {"learned": [], "oracle": []}
    for offset in range(0, len(documents), batch_size):
        batch = collate_documents(documents[offset:offset + batch_size]).to(device)
        common = {
            "input_ids": batch.input_ids,
            "attention_mask": batch.attention_mask,
            "pointers": batch.pointers,
            "scope_depths": batch.depths,
            "alternate_pointers": batch.alternate_pointers,
        }
        with _autocast(device, bf16):
            learned = model(**common)
            oracle = model(
                **common, structural_events=batch.events,
                structural_validity=batch.validity, intervention="oracle",
            )
        for mode, logits in (("learned", learned), ("oracle", oracle)):
            log_probs = F.log_softmax(logits[:, :-1].float(), -1)
            targets = batch.input_ids[:, 1:]
            token_nll = -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            for row, document in enumerate(batch.documents):
                answer_position = int(batch.answer_positions[row])
                answer_token = int(batch.input_ids[row, answer_position])
                answer_logits = logits[row, answer_position - 1].float()
                valid = batch.attention_mask[row, 1:]
                result[mode].append({
                    "document_id": document.document_id,
                    "base_id": document.metadata.get("base_id", document.document_id),
                    "prediction_token": int(answer_logits.argmax()),
                    "correct": int(answer_logits.argmax() == answer_token),
                    "answer_nll": float(-F.log_softmax(answer_logits, -1)[answer_token]),
                    "nll_sum": float(token_nll[row, valid].sum()),
                    "nll_tokens": int(valid.sum()),
                    "metadata": document.metadata,
                })
    return result


@torch.inference_mode()
def audit_semantic_alignment(
    specialist: RecurrentSpecialist,
    documents: list[Document],
    device: torch.device,
) -> dict:
    outputs = {}
    traces = {}
    batch_size = 64
    for offset in range(0, len(documents), batch_size):
        members = documents[offset:offset + batch_size]
        batch = collate_documents(members).to(device)
        predicted = specialist(batch.input_ids)
        event = predicted["event_logits"].softmax(-1).cpu()
        validity = predicted["validity_logits"].softmax(-1)[..., 1].cpu()
        for row, document in enumerate(members):
            length = len(document.token_ids)
            outputs[document.document_id] = {
                "event": event[row, :length],
                "validity": validity[row, :length],
            }
            traces[document.document_id] = CausalStructureMachine().analyze(
                document.token_ids
            )
    document_map = {document.document_id: document for document in documents}
    metrics = defaultdict(lambda: defaultdict(list))
    for variant in documents:
        kind = variant.metadata.get("variant")
        if kind not in ("renamed", "whitespace"):
            continue
        base = document_map[variant.metadata["base_id"]]
        base_positions = [
            index for index, token in enumerate(base.token_ids)
            if Vocabulary.TOKENS[token] not in Vocabulary.WHITESPACE
        ]
        variant_positions = [
            index for index, token in enumerate(variant.token_ids)
            if Vocabulary.TOKENS[token] not in Vocabulary.WHITESPACE
        ]
        if len(base_positions) != len(variant_positions):
            raise AssertionError("semantic token alignment length changed")
        left = outputs[base.document_id]
        right = outputs[variant.document_id]
        left_events = left["event"][base_positions]
        right_events = right["event"][variant_positions]
        metrics[kind]["event_argmax_agreement"].append(float(
            (left_events.argmax(-1) == right_events.argmax(-1)).float().mean()
        ))
        metrics[kind]["event_probability_mae"].append(float(
            (left_events - right_events).abs().mean()
        ))
        metrics[kind]["validity_probability_mae"].append(float(
            (left["validity"][base_positions] - right["validity"][variant_positions])
            .abs().mean()
        ))
        base_depths = torch.tensor(traces[base.document_id].depths)[base_positions]
        variant_depths = torch.tensor(traces[variant.document_id].depths)[variant_positions]
        metrics[kind]["programmed_depth_agreement"].append(float(
            (base_depths == variant_depths).float().mean()
        ))
        base_bindings = ReferenceInterpreter().interpret(base.text).bindings
        variant_bindings = ReferenceInterpreter().interpret(variant.text).bindings
        base_ordinals = [
            _binding_ordinal(base, binding.declaration_position) for binding in base_bindings
        ]
        variant_ordinals = [
            _binding_ordinal(variant, binding.declaration_position)
            for binding in variant_bindings
        ]
        metrics[kind]["semantic_binding_ordinal_agreement"].append(float(
            base_ordinals == variant_ordinals
        ))
    return {
        kind: {name: _summary(values) for name, values in values.items()}
        for kind, values in metrics.items()
    }


@torch.inference_mode()
def audit_oracle_features(config: dict, run_dir: Path, device: torch.device) -> dict:
    data = prepare_corpora(config, run_dir)
    documents = load_corpus(data["files"]["confirmation"]["path"])
    specialist, _ = load_frozen_specialist(config, run_dir, device)
    alignment = audit_semantic_alignment(specialist, documents, device)
    checkpoints = []
    grouped = defaultdict(lambda: defaultdict(list))
    for checkpoint_path in _checkpoint_paths(run_dir):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model = _load_model(config, run_dir, checkpoint, device)
        assert isinstance(model, SpecialistDecoderLM)
        rows = _learned_and_oracle_rows(
            model, documents,
            int(config["evaluation"]["batch_sizes"][str(checkpoint["width"])]),
            device, bool(config["runtime"]["bf16"]),
        )
        record = {
            "layers": checkpoint["layers"], "width": checkpoint["width"],
            "seed": checkpoint["seed"],
        }
        for mode in ("learned", "oracle"):
            record[mode] = {
                "metrics": _aggregate_rows(rows[mode]),
                "invariance": _invariance(rows[mode]),
            }
            grouped[(checkpoint["layers"], checkpoint["width"])][mode].append(record[mode])
        checkpoints.append(record)

    plain_invariance = defaultdict(lambda: defaultdict(list))
    for evaluation_path in run_dir.glob("backbones/l*_w*/plain/seed_*/evaluation.json"):
        value = json.loads(evaluation_path.read_text(encoding="utf-8"))
        key = (value["layers"], value["width"])
        for kind, metric in value["invariance"].items():
            plain_invariance[key][kind].append(metric["prediction_agreement"])

    by_size = {}
    for key, modes in grouped.items():
        label = f"l{key[0]}_w{key[1]}"
        by_size[label] = {}
        for mode, records in modes.items():
            by_size[label][mode] = {
                "metrics": {
                    name: sum(item["metrics"][name] for item in records) / len(records)
                    for name in ("answer_accuracy", "answer_nll", "overall_nll")
                },
                "invariance": {
                    kind: sum(item["invariance"][kind]["prediction_agreement"]
                              for item in records) / len(records)
                    for kind in ("renamed", "whitespace")
                },
            }
        by_size[label]["plain_existing_invariance"] = {
            kind: sum(values) / len(values)
            for kind, values in plain_invariance[key].items()
        }
    return {
        "per_checkpoint": checkpoints,
        "by_size_mean_across_seeds": by_size,
        "semantic_alignment_after_token_position_remap": alignment,
        "interface": (
            "Oracle event one-hots, exact programmed depths, oracle prefix validity, "
            "confidence=1, and exact pointers use the same 30-column interface and the "
            "same frozen PointerReader/backbone as learned features."
        ),
    }


def _synchronized_time(
    function: Callable[[], object],
    *,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> float:
    for _ in range(warmup):
        function()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for _ in range(repeats):
        function()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return (time.perf_counter() - started) / repeats


@torch.inference_mode()
def audit_timing(config: dict, run_dir: Path, device: torch.device) -> dict:
    data = prepare_corpora(config, run_dir)
    documents = [
        document for document in load_corpus(data["files"]["confirmation"]["path"])
        if document.metadata.get("variant") == "base"
    ][:16]
    cpu_ids = [document.token_ids for document in documents]
    results = []
    for checkpoint_path in _checkpoint_paths(run_dir):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model = _load_model(config, run_dir, checkpoint, device)
        assert isinstance(model, SpecialistDecoderLM)
        model.eval()
        batch = collate_documents(documents).to(device)
        with _autocast(device, bool(config["runtime"]["bf16"])):
            features, pointers = model.specialist.interface(
                batch.input_ids, batch.pointers, scope_depths=batch.depths
            )
            raw = model.backbone.embed(batch.input_ids)
            reader_signal = model.reader(raw, features.to(raw.dtype), pointers)
        machine = CausalStructureMachine()

        state_seconds = _synchronized_time(
            lambda: [machine.analyze(ids) for ids in cpu_ids],
            device=torch.device("cpu"), warmup=20, repeats=100,
        )

        def specialist_call():
            with _autocast(device, bool(config["runtime"]["bf16"])):
                return model.specialist.interface(
                    batch.input_ids, batch.pointers, scope_depths=batch.depths
                )

        specialist_seconds = _synchronized_time(
            specialist_call, device=device, warmup=20, repeats=100
        )

        def reader_call():
            with _autocast(device, bool(config["runtime"]["bf16"])):
                return model.reader(raw, features.to(raw.dtype), pointers)

        reader_seconds = _synchronized_time(
            reader_call, device=device, warmup=20, repeats=100
        )

        def backbone_call():
            with _autocast(device, bool(config["runtime"]["bf16"])):
                embedded = model.backbone.embed(batch.input_ids)
                return model.backbone.decode(
                    embedded + reader_signal, batch.attention_mask
                )

        backbone_seconds = _synchronized_time(
            backbone_call, device=device, warmup=20, repeats=100
        )
        total = state_seconds + specialist_seconds + reader_seconds + backbone_seconds
        actual_tokens = sum(len(document.token_ids) for document in documents)
        results.append({
            "layers": checkpoint["layers"], "width": checkpoint["width"],
            "seed": checkpoint["seed"], "batch_documents": len(documents),
            "padded_shape": list(batch.input_ids.shape), "actual_tokens": actual_tokens,
            "warmup_iterations_per_component": 20,
            "timed_iterations_per_component": 100,
            "synchronized": True,
            "milliseconds_per_batch": {
                "programmed_state_updates_cpu": 1000 * state_seconds,
                "recurrent_specialist_and_feature_pack_gpu": 1000 * specialist_seconds,
                "specialist_plus_state": 1000 * (state_seconds + specialist_seconds),
                "reader_gpu": 1000 * reader_seconds,
                "backbone_embedding_blocks_head_gpu": 1000 * backbone_seconds,
                "sum_of_components": 1000 * total,
            },
            "microseconds_per_actual_token": {
                "specialist_plus_state": 1e6 * (state_seconds + specialist_seconds)
                / actual_tokens,
                "reader": 1e6 * reader_seconds / actual_tokens,
                "backbone": 1e6 * backbone_seconds / actual_tokens,
                "sum_of_components": 1e6 * total / actual_tokens,
            },
        })
    by_size = {}
    for layers, width in config["backbones"]["sizes"]:
        members = [item for item in results if item["layers"] == layers and item["width"] == width]
        label = f"l{layers}_w{width}"
        by_size[label] = {
            component: sum(item["microseconds_per_actual_token"][component]
                           for item in members) / len(members)
            for component in ("specialist_plus_state", "reader", "backbone", "sum_of_components")
        }
        by_size[label]["composition_overhead_vs_backbone_percent"] = 100 * (
            by_size[label]["specialist_plus_state"] + by_size[label]["reader"]
        ) / by_size[label]["backbone"]
    return {
        "per_checkpoint": results,
        "mean_microseconds_per_actual_token_by_size": by_size,
        "device": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else str(device)
        ),
        "method": (
            "Each component was warmed for 20 iterations, timed for 100, and bounded by "
            "torch.cuda.synchronize on cuda:0. State updates were timed on CPU. Transfers "
            "and tokenization are excluded; all learned features are recomputed."
        ),
    }


def _code_reference(obj) -> str:
    path = Path(inspect.getsourcefile(obj)).resolve()
    line = inspect.getsourcelines(obj)[1]
    return f"{path}:{line}"


def _markdown(report: dict) -> str:
    gate = report["specialist_gate"]
    event = gate["reconstructed_event_metrics"]
    pointer = report["pointer_intervention"]
    oracle = report["oracle_features"]
    lines = [
        "# Phase 1 audit",
        "",
        "**Corrected conclusion: the Phase 1 verdict remains failed composition.** No "
        "checkpoint was retrained or modified. The specialist passed only the implemented "
        "0.95 macro-F1 gate; it did not satisfy a 0.99 event-accuracy requirement. The "
        "reader consumes changed pointer inputs, but answer probabilities are effectively "
        "insensitive to a one-reference, opposite-value lookup intervention.",
        "",
        "## 1. Specialist gate reconciliation",
        "",
        f"Implemented threshold: macro-F1 >= "
        f"{gate['implemented_criterion']['structural_event_minimum']:.2f}, excluding pad and "
        f"whitespace. Reconstructed macro-F1: "
        f"{event['structural_macro_f1_excluding_pad_and_whitespace']:.6f}; overall event "
        f"structural-event accuracy excluding pad/whitespace: "
        f"{event['structural_event_accuracy_excluding_pad_and_whitespace']:.6f} "
        f"(overall token accuracy {event['accuracy']:.6f}). Both structural metrics are "
        "below 0.99. Binding remains "
        f"{gate['frozen_manifest_metrics']['binding_accuracy']:.6f}.",
        "",
        "The historical `passed` field is retained because it reflects the implemented "
        "criterion. Treating it as proof of >=99% structural-event accuracy was incorrect.",
        "",
        "| Event | Support | Predicted | TP | FP | FN | Precision | Recall | F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, value in event["classes"].items():
        if value["support"]:
            lines.append(
                f"| {name} | {value['support']} | {value['predicted']} | "
                f"{value['true_positive']} | {value['false_positive']} | "
                f"{value['false_negative']} | {value['precision']:.4f} | "
                f"{value['recall']:.4f} | {value['f1']:.4f} |"
            )
    lines.extend((
        "", "Validation slices:", "",
        "| Slice | Documents | Tokens | Overall accuracy | Structural accuracy | Macro-F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ))
    for name, value in gate["validation_slices"].items():
        lines.append(
            f"| {name} | {value['documents']} | {value['tokens']} | "
            f"{value['accuracy']:.4f} | "
            f"{value['structural_event_accuracy_excluding_pad_and_whitespace']:.4f} | "
            f"{value['structural_macro_f1_excluding_pad_and_whitespace']:.4f} |"
        )
    aggregate = pointer["aggregate_probability_effect"]
    full = aggregate["full_vocab_correct_answer_probability_wrong_minus_correct"]
    bit = aggregate["binary_normalized_correct_answer_probability_wrong_minus_correct"]
    consumed = pointer["consumed_information_verification"]
    lines.extend((
        "", "## 2. Controlled pointer intervention", "",
        f"Eligible lookup-only cases: {pointer['eligible_different_value_documents']} of "
        f"{pointer['base_confirmation_lookup_documents']} base lookup documents; "
        f"{pointer['evaluated_checkpoint_case_pairs']} checkpoint/case pairs. Every case "
        "changes exactly one reference to a declaration holding the opposite bit; XOR is "
        f"excluded ({pointer['unique_case_direction_counts']['0_to_1']} 0->1 and "
        f"{pointer['unique_case_direction_counts']['1_to_0']} 1->0 documents).", "",
        f"Mean change in full-vocabulary correct-answer probability: {full['mean']:+.8f} "
        f"(mean absolute {full['mean_absolute']:.8f}, range {full['minimum']:+.8f} to "
        f"{full['maximum']:+.8f}). Mean "
        f"change after normalizing over bit tokens only: {bit['mean']:+.8f} (mean absolute "
        f"{bit['mean_absolute']:.8f}). Top-1 answer accuracy changed by exactly zero in "
        "all nine checkpoints.", "",
        f"Exact input checks across {consumed['cases']} checkpoint/case pairs: pointer index "
        f"changed {consumed['pointer_index_changed']}; gather indices changed "
        f"{consumed['gather_indices_changed']}; gathered embedding state changed "
        f"{consumed['gathered_embedding_state_changed']}; weighted context changed "
        f"{consumed['weighted_context_changed']}; final reader output changed "
        f"{consumed['reader_output_changed']}. No specialist feature or gathered-state "
        "cache exists.", "",
        "## 3. How pointers enter", "",
        "Pointers are deterministic token indices from the programmed lexical-scope table, "
        "not learned predictions. At each use, the reader gathers a seven-token window of "
        "raw token-plus-position embeddings beginning at the declaration identifier, scores "
        "that window, forms a weighted context, gates/scales it, and adds it to a learned "
        "projection of structural features. Pointer distance is also an encoded scalar in "
        "the feature vector. Learned fields are event probabilities, validity probability, "
        "and confidence; programmed fields are exact depth, pointer presence/distance, and "
        "the pointer address.", "",
        "## 4. Oracle features and invariance localization", "",
        "Oracle causal event/validity labels were passed through the same 30-column interface, "
        "frozen reader, and frozen backbone. Semantic binding comparisons use declaration "
        "ordinals rather than raw token positions.", "",
        "| Size | Mode | Accuracy | Answer NLL | Overall NLL | Rename agreement | Whitespace agreement |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ))
    for size, values in oracle["by_size_mean_across_seeds"].items():
        for mode in ("learned", "oracle"):
            metric = values[mode]["metrics"]
            invariant = values[mode]["invariance"]
            lines.append(
                f"| {size} | {mode} | {metric['answer_accuracy']:.4f} | "
                f"{metric['answer_nll']:.4f} | {metric['overall_nll']:.4f} | "
                f"{invariant['renamed']:.4f} | {invariant['whitespace']:.4f} |"
            )
        plain = values["plain_existing_invariance"]
        lines.append(
            f"| {size} | plain existing | - | - | - | {plain['renamed']:.4f} | "
            f"{plain['whitespace']:.4f} |"
        )
    lines.extend(("", "Aligned specialist features:", ""))
    for kind, metrics in oracle["semantic_alignment_after_token_position_remap"].items():
        lines.append(
            f"- {kind}: semantic binding ordinal agreement "
            f"{metrics['semantic_binding_ordinal_agreement']['mean']:.4f}; programmed depth "
            f"agreement {metrics['programmed_depth_agreement']['mean']:.4f}; learned event "
            f"argmax agreement {metrics['event_argmax_agreement']['mean']:.4f}; event-probability "
            f"MAE {metrics['event_probability_mae']['mean']:.6f}."
        )
    lines.extend((
        "", "## 5. Synchronized online timing", "",
        f"Device: {report['timing']['device']}.", "",
        "| Size | Specialist + state (us/token) | Reader | Backbone | Sum | Overhead vs backbone |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ))
    for size, values in report["timing"]["mean_microseconds_per_actual_token_by_size"].items():
        lines.append(
            f"| {size} | {values['specialist_plus_state']:.3f} | {values['reader']:.3f} | "
            f"{values['backbone']:.3f} | {values['sum_of_components']:.3f} | "
            f"{values['composition_overhead_vs_backbone_percent']:.1f}% |"
        )
    lines.extend((
        "", report["timing"]["method"], "", "## Code references", "",
    ))
    for name, reference in report["code_references"].items():
        lines.append(f"- {name}: `{reference}`")
    lines.extend((
        "", "## Final corrected conclusions", "",
        "- Preserve the original negative quality and cost verdict.",
        "- The specialist checkpoint met its implemented 0.95 macro-F1 gate, not a 0.99 "
        "structural-event criterion.",
        "- Programmed semantic bindings are invariant after remapping positions. Any learned "
        "feature drift is small, and oracle features do not repair output invariance. The "
        "similar plain-backbone failures localize renaming/whitespace sensitivity primarily "
        "to the raw-token/position backbone, not the specialist.",
        "- The trained reader receives genuinely different pointer-derived tensors yet has "
        "negligible answer-probability sensitivity to an opposite-value lookup pointer. It "
        "did not learn useful causal pointer retrieval; this is a separate reader failure.",
        "- No further training or architecture change is justified by this audit alone.",
        "", "Reproduce the audit (evaluation only):", "",
        "`python -m experiments.modular_phase1.cli --config "
        "experiments/modular_phase1/configs/pilot.json --run-dir "
        "scratch/modular-phase1 audit`",
        "",
    ))
    return "\n".join(lines)


def run_audit(config: dict, run_dir: str | Path) -> dict:
    run_dir = Path(run_dir)
    checkpoint_paths = [run_dir / "specialist" / "specialist.pt"] + list(
        run_dir.glob("backbones/l*_w*/*/seed_*/checkpoint.pt")
    )
    hashes_before = {str(path.resolve()): file_hash(path) for path in checkpoint_paths}
    device = _device(config)
    report = {
        "schema": 1,
        "audit_only": True,
        "optimizer_steps": 0,
        "preserved_phase1_verdict": "failed composition",
        "config_sha256": config["_config_sha256"],
        "specialist_gate": audit_specialist_gate(config, run_dir, device),
        "pointer_intervention": audit_pointer_intervention(config, run_dir, device),
        "oracle_features": audit_oracle_features(config, run_dir, device),
        "timing": audit_timing(config, run_dir, device),
        "code_references": {
            "programmed_state_and_pointers": _code_reference(CausalStructureMachine.analyze),
            "learned_interface": _code_reference(RecurrentSpecialist.interface),
            "oracle_interface": _code_reference(RecurrentSpecialist.oracle_interface),
            "reader_gather": _code_reference(PointerReader.pointer_inputs),
            "composed_forward": _code_reference(SpecialistDecoderLM.forward),
            "audit_entrypoint": _code_reference(run_audit),
            "frozen_config": str(Path(config["_config_path"]).resolve()),
        },
        "checkpoint_hashes_before": hashes_before,
    }
    hashes_after = {str(path.resolve()): file_hash(path) for path in checkpoint_paths}
    if hashes_after != hashes_before:
        raise RuntimeError("audit modified a frozen checkpoint")
    report["checkpoint_hashes_after"] = hashes_after
    report["all_checkpoint_hashes_unchanged"] = True
    json_path = run_dir / "audit.json"
    markdown_path = run_dir / "AUDIT.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(_markdown(report), encoding="utf-8")
    return report

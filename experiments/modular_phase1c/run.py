"""Run the bounded Phase 1c direct lookup-readout experiment."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from experiments.modular_phase1.audit import eligible_pointer_case
from experiments.modular_phase1.config import file_hash, load_config
from experiments.modular_phase1.data import load_corpus
from experiments.modular_phase1.evaluation import _load_model
from experiments.modular_phase1.language import Document, Vocabulary
from experiments.modular_phase1.reference import ReferenceInterpreter
from experiments.modular_phase1.training import _autocast, _state_hash

from .models import (
    BIT_IDS,
    DirectLookupReadout,
    compose_bit_distribution,
    deterministic_copy_log_probs,
    literal_representation,
    trace_lookup_prefix,
)


@dataclass(frozen=True)
class LookupExample:
    example_id: str
    pair_id: str
    split: str
    token_ids: tuple[int, ...]
    answer_position: int
    target: int
    metadata: dict
    flipped: bool


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load_diagnostic_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    config["_config_path"] = str(path.resolve())
    config["_config_sha256"] = file_hash(path)
    return config


def _summary(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "mean": None, "mean_absolute": None, "median": None,
                "minimum": None, "maximum": None, "nonzero": 0}
    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "mean_absolute": sum(abs(value) for value in values) / len(values),
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
        "nonzero": sum(value != 0.0 for value in values),
    }


def _source_hashes(roots: list[Path]) -> dict[str, str]:
    paths = []
    for root in roots:
        paths.extend(
            path for path in root.rglob("*")
            if path.is_file() and path.suffix in (".pt", ".json", ".md", ".gz")
        )
    return {str(path.resolve()): file_hash(path) for path in sorted(paths)}


def _value_position(document: Document, declaration: int) -> int:
    for position in range(declaration + 1, len(document.token_ids)):
        token = document.token_ids[position]
        if token in BIT_IDS:
            return position
        if token == Vocabulary.TO_ID[";"]:
            break
    raise ValueError("bound declaration contains no literal")


def _pair_document(document: Document, split: str) -> tuple[LookupExample, LookupExample]:
    interpretation = ReferenceInterpreter().interpret(document.text)
    if len(interpretation.bindings) != 1:
        raise ValueError("Phase 1c requires one-operand lookup documents")
    binding = interpretation.bindings[0]
    value_position = _value_position(document, binding.declaration_position)
    value = 0 if document.token_ids[value_position] == BIT_IDS[0] else 1
    if value != document.answer:
        raise AssertionError("lookup answer does not match bound literal")
    pair_id = f"{split}:{document.document_id}"
    base = LookupExample(
        f"{document.document_id}:base", pair_id, split, tuple(document.token_ids),
        document.answer_position, value, dict(document.metadata), False,
    )
    flipped_ids = list(document.token_ids)
    flipped_value = 1 - value
    flipped_ids[value_position] = BIT_IDS[flipped_value]
    flipped_ids[document.answer_position] = BIT_IDS[flipped_value]
    flipped = LookupExample(
        f"{document.document_id}:flip", pair_id, split, tuple(flipped_ids),
        document.answer_position, flipped_value, dict(document.metadata), True,
    )
    return base, flipped


def _single_document(document: Document, split: str) -> LookupExample:
    trace = trace_lookup_prefix(document.token_ids)
    score_position = document.answer_position - 1
    if not trace.active[score_position]:
        raise AssertionError("valid lookup variant is inactive at its answer score")
    return LookupExample(
        document.document_id, f"{split}:{document.document_id}", split,
        tuple(document.token_ids), document.answer_position, document.answer,
        dict(document.metadata), False,
    )


def prepare_examples(source_run: Path) -> tuple[dict[str, list[LookupExample]], dict]:
    manifest = json.loads(
        (source_run / "data" / "manifest.json").read_text(encoding="utf-8")
    )
    source_names = {
        "train": "backbone_train",
        "validation": "backbone_validation",
        "heldout": "confirmation",
    }
    result = {}
    pair_ids = {}
    base_documents = {}
    for split, source in source_names.items():
        documents = load_corpus(manifest["files"][source]["path"])
        selected = [
            document for document in documents
            if document.metadata.get("variant") == "base"
            and not document.metadata["literal"]
            and document.metadata["query_kind"] == "lookup"
        ]
        base_documents[split] = selected
        examples = []
        for document in selected:
            examples.extend(_pair_document(document, split))
        result[split] = examples
        pair_ids[split] = {example.pair_id for example in examples}
    if pair_ids["train"] & pair_ids["validation"] or pair_ids["train"] & pair_ids["heldout"]:
        raise AssertionError("paired examples cross splits")
    if pair_ids["validation"] & pair_ids["heldout"]:
        raise AssertionError("paired examples cross splits")

    confirmation = load_corpus(manifest["files"]["confirmation"]["path"])
    variants = [
        document for document in confirmation
        if document.metadata.get("variant") in ("renamed", "whitespace")
        and not document.metadata["literal"]
        and document.metadata["query_kind"] == "lookup"
    ]
    result["renamed"] = [
        _single_document(document, "renamed") for document in variants
        if document.metadata["variant"] == "renamed"
    ]
    result["whitespace"] = [
        _single_document(document, "whitespace") for document in variants
        if document.metadata["variant"] == "whitespace"
    ]
    metadata = {
        "pairs": {split: len(ids) for split, ids in pair_ids.items()},
        "examples": {split: len(examples) for split, examples in result.items()},
        "pair_id_overlap": {
            "train_validation": len(pair_ids["train"] & pair_ids["validation"]),
            "train_heldout": len(pair_ids["train"] & pair_ids["heldout"]),
            "validation_heldout": len(pair_ids["validation"] & pair_ids["heldout"]),
        },
        "balanced_targets": {
            split: {
                "zero": sum(example.target == 0 for example in result[split]),
                "one": sum(example.target == 1 for example in result[split]),
            }
            for split in ("train", "validation", "heldout")
        },
        "base_documents": base_documents,
    }
    return result, metadata


def _retrieved_values(examples: list[LookupExample]) -> torch.Tensor:
    values = []
    for example in examples:
        trace = trace_lookup_prefix(list(example.token_ids))
        position = example.answer_position - 1
        if not trace.active[position]:
            raise AssertionError(f"lookup inactive for {example.example_id}")
        value = trace.retrieved_values[position]
        if value != example.target:
            raise AssertionError("retrieved literal disagrees with lookup target")
        values.append(value)
    return torch.tensor(values, dtype=torch.long)


def build_interventions(documents: list[Document]) -> list[dict]:
    records = []
    for document in documents:
        case = eligible_pointer_case(document)
        if case is None:
            continue
        trace = trace_lookup_prefix(document.token_ids)
        score = document.answer_position - 1
        override = {case["use_position"]: case["alternate_declaration_position"]}
        changed = trace_lookup_prefix(document.token_ids, pointer_override=override)
        if trace.retrieved_values[score] == changed.retrieved_values[score]:
            raise AssertionError("pointer intervention did not change retrieved literal")
        if changed.retrieved_values[score] != case["alternate_value"]:
            raise AssertionError("intervened retrieval disagrees with declaration literal")
        records.append({
            "document_id": document.document_id,
            "token_ids": tuple(document.token_ids),
            "answer_position": document.answer_position,
            "use_position": case["use_position"],
            "alternate_declaration_position": case["alternate_declaration_position"],
            "original_value": trace.retrieved_values[score],
            "alternate_value": changed.retrieved_values[score],
        })
    return records


def verify_prefix_causality(
    lookup_examples: list[LookupExample],
    xor_documents: list[Document],
) -> dict:
    comparisons = 0
    score_prefix_documents = 0
    active_with_whitespace = 0
    active_without_whitespace = 0
    for example in lookup_examples:
        ids = list(example.token_ids)
        full = trace_lookup_prefix(ids)
        lengths = range(1, len(ids) + 1) if score_prefix_documents < 32 else (
            example.answer_position,
        )
        for length in lengths:
            prefix = trace_lookup_prefix(ids[:length])
            for name in (
                "use_pointers", "active", "binding_addresses",
                "value_addresses", "retrieved_values",
            ):
                if getattr(prefix, name) != getattr(full, name)[:length]:
                    raise AssertionError(f"future suffix changed {name}")
            comparisons += 1
        score_prefix_documents += 1
        score = example.answer_position - 1
        previous = ids[score]
        if Vocabulary.TOKENS[previous] in Vocabulary.WHITESPACE:
            active_with_whitespace += int(full.active[score])
        else:
            active_without_whitespace += int(full.active[score])
    xor_inactive = 0
    for document in xor_documents[:64]:
        trace = trace_lookup_prefix(document.token_ids)
        xor_inactive += int(not any(trace.active))
    signature = inspect.signature(trace_lookup_prefix)
    return {
        "prefix_comparisons": comparisons,
        "all_score_prefix_documents_checked": score_prefix_documents,
        "all_prefix_fields_invariant_to_unseen_suffix": True,
        "active_answers_after_arrow_whitespace": active_with_whitespace,
        "active_answers_immediately_after_arrow": active_without_whitespace,
        "xor_documents_inactive": xor_inactive,
        "xor_documents_checked": min(64, len(xor_documents)),
        "answer_annotation_is_not_an_activation_input": (
            "answer_position" not in signature.parameters
        ),
    }


def _conditional_accuracy(model: DirectLookupReadout, values: torch.Tensor,
                          targets: torch.Tensor, independent: bool = False) -> float:
    representation = literal_representation(values)
    if independent:
        representation = torch.zeros_like(representation)
    return float((model(representation).argmax(-1) == targets).float().mean())


def _conditional_metrics(
    model: DirectLookupReadout,
    values: torch.Tensor,
    targets: torch.Tensor,
    *,
    independent: bool = False,
) -> dict:
    representation = literal_representation(values)
    if independent:
        representation = torch.zeros_like(representation)
    logits = model(representation)
    return {
        "accuracy": float((logits.argmax(-1) == targets).float().mean()),
        "nll": float(F.cross_entropy(logits, targets)),
        "answers": len(targets),
    }


def train_readouts(
    examples: dict[str, list[LookupExample]],
    interventions: list[dict],
    config: dict,
    device: torch.device,
) -> tuple[DirectLookupReadout, DirectLookupReadout, dict]:
    torch.manual_seed(int(config["seed"]))
    learned = DirectLookupReadout().to(device)
    control = DirectLookupReadout().to(device)
    parameters = list(learned.parameters()) + list(control.parameters())
    settings = config["training"]
    optimizer = torch.optim.AdamW(
        parameters, lr=float(settings["learning_rate"]),
        betas=tuple(settings["betas"]), eps=float(settings["epsilon"]),
        weight_decay=float(settings["weight_decay"]),
    )
    split_tensors = {}
    for split, members in examples.items():
        values = _retrieved_values(members).to(device)
        targets = torch.tensor([item.target for item in members], device=device)
        split_tensors[split] = (values, targets)
    intervention_values = torch.tensor(
        [item["alternate_value"] for item in interventions], device=device
    )
    intervention_targets = intervention_values.clone()

    train_members = examples["train"]
    if len(train_members) % 2:
        raise AssertionError("paired train examples must be even")
    pair_count = len(train_members) // 2
    for index in range(pair_count):
        if train_members[2 * index].pair_id != train_members[2 * index + 1].pair_id:
            raise AssertionError("training pair adjacency broken")
    pairs_per_batch = int(settings["pairs_per_batch"])
    max_updates = int(config["budget"]["max_updates"])
    max_answers = int(config["budget"]["max_supervised_answers"])
    rng = random.Random(int(config["seed"]))
    order = list(range(pair_count))
    offset = 0
    updates = supervised = consecutive = 0
    loss_history = []
    checks = []
    stopped = "budget"
    while updates < max_updates and supervised < max_answers:
        if offset == 0:
            rng.shuffle(order)
        remaining_answers = max_answers - supervised
        take_pairs = min(pairs_per_batch, remaining_answers // 2, pair_count - offset)
        if take_pairs == 0:
            offset = 0
            continue
        pair_indices = order[offset:offset + take_pairs]
        offset += take_pairs
        if offset == pair_count:
            offset = 0
        example_indices = [
            member for pair_index in pair_indices
            for member in (2 * pair_index, 2 * pair_index + 1)
        ]
        values, targets = split_tensors["train"]
        batch_values = values[example_indices]
        batch_targets = targets[example_indices]
        representation = literal_representation(batch_values)
        optimizer.zero_grad(set_to_none=True)
        learned_loss = F.cross_entropy(learned(representation), batch_targets)
        control_loss = F.cross_entropy(
            control(torch.zeros_like(representation)), batch_targets
        )
        loss = learned_loss + control_loss
        loss.backward()
        optimizer.step()
        updates += 1
        supervised += len(example_indices)
        loss_history.append(float(learned_loss.detach()))

        if updates % int(settings["check_every_updates"]):
            continue
        with torch.inference_mode():
            validation = _conditional_accuracy(
                learned, *split_tensors["validation"]
            )
            intervention = _conditional_accuracy(
                learned, intervention_values, intervention_targets
            )
            control_accuracy = _conditional_accuracy(
                control, *split_tensors["validation"], independent=True
            )
        passed = (
            validation >= float(config["acceptance"]["conditional_accuracy"])
            and intervention >= float(config["acceptance"]["intervention_accuracy"])
            and abs(control_accuracy - 0.5)
            <= float(config["acceptance"]["control_chance_tolerance"])
        )
        checks.append({
            "updates": updates, "supervised_answers": supervised,
            "validation_accuracy": validation,
            "intervention_accuracy": intervention,
            "control_validation_accuracy": control_accuracy,
            "passed": passed,
        })
        consecutive = consecutive + 1 if passed else 0
        if consecutive >= int(settings["required_consecutive_passes"]):
            stopped = "acceptance_gate"
            break
    learned.eval()
    control.eval()
    with torch.inference_mode():
        final_conditional = {
            split: {
                "learned_readout": _conditional_metrics(
                    learned, values, targets
                ),
                "value_independent_control": _conditional_metrics(
                    control, values, targets, independent=True
                ),
            }
            for split, (values, targets) in split_tensors.items()
        }
    return learned, control, {
        "optimizer": settings,
        "updates": updates,
        "supervised_answer_count": supervised,
        "available_unique_supervised_answers": len(train_members),
        "mean_learned_conditional_ce": sum(loss_history) / len(loss_history),
        "last_learned_conditional_ce": loss_history[-1],
        "checks": checks,
        "final_conditional_metrics": final_conditional,
        "stopped_at": stopped,
        "learned_trainable_parameters": sum(p.numel() for p in learned.parameters()),
        "control_trainable_parameters": sum(p.numel() for p in control.parameters()),
    }


def _collate(examples: list[LookupExample], device: torch.device):
    maximum = max(len(item.token_ids) for item in examples)
    ids = torch.full((len(examples), maximum), Vocabulary.PAD, dtype=torch.long)
    mask = torch.zeros_like(ids, dtype=torch.bool)
    for row, example in enumerate(examples):
        length = len(example.token_ids)
        ids[row, :length] = torch.tensor(example.token_ids)
        mask[row, :length] = True
    return ids.to(device), mask.to(device)


@torch.inference_mode()
def cache_answer_logits(
    backbone,
    examples: list[LookupExample],
    *,
    batch_size: int,
    device: torch.device,
    bf16: bool,
) -> tuple[torch.Tensor, float]:
    outputs = []
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    for offset in range(0, len(examples), batch_size):
        members = examples[offset:offset + batch_size]
        ids, mask = _collate(members, device)
        with _autocast(device, bf16):
            logits = backbone(ids, attention_mask=mask)
        for row, example in enumerate(members):
            outputs.append(logits[row, example.answer_position - 1].float().cpu())
    torch.cuda.synchronize(device)
    return torch.stack(outputs), time.perf_counter() - started


@torch.inference_mode()
def evaluate_conditions(
    answer_logits: torch.Tensor,
    examples: list[LookupExample],
    learned: DirectLookupReadout,
    control: DirectLookupReadout,
    device: torch.device,
) -> tuple[dict, dict]:
    logits = answer_logits.to(device)
    base = torch.log_softmax(logits, -1)
    values = _retrieved_values(examples).to(device)
    targets = torch.tensor([item.target for item in examples], device=device)
    active = torch.ones(len(examples), dtype=torch.bool, device=device)
    representation = literal_representation(values)
    condition_logs = {
        "baseline": base,
        "deterministic_copy": compose_bit_distribution(
            base, deterministic_copy_log_probs(values), active
        ),
        "learned_readout": compose_bit_distribution(
            base, F.log_softmax(learned(representation), -1), active
        ),
        "value_independent_control": compose_bit_distribution(
            base, F.log_softmax(control(torch.zeros_like(representation)), -1), active
        ),
    }
    metrics = {}
    preservation = {}
    nonbit = torch.tensor(
        [index for index in range(len(Vocabulary.TOKENS)) if index not in BIT_IDS],
        device=device,
    )
    base_probs = base.exp()
    base_bit_mass = base_probs[:, list(BIT_IDS)].sum(-1)
    for name, log_probs in condition_logs.items():
        probs = log_probs.exp()
        bits = log_probs[:, list(BIT_IDS)]
        conditional = F.log_softmax(bits, -1)
        metrics[name] = {
            "documents": len(examples),
            "conditional_bit_accuracy": float(
                (bits.argmax(-1) == targets).float().mean()
            ),
            "conditional_bit_nll": float(
                F.nll_loss(conditional, targets)
            ),
            "full_vocabulary_answer_accuracy": float(
                (log_probs.argmax(-1) == torch.tensor(
                    [BIT_IDS[int(value)] for value in targets.tolist()], device=device
                )).float().mean()
            ),
            "answer_nll": float(F.nll_loss(
                log_probs,
                torch.tensor(
                    [BIT_IDS[int(value)] for value in targets.tolist()], device=device
                ),
            )),
            "mean_bit_category_probability": float(
                probs[:, list(BIT_IDS)].sum(-1).mean()
            ),
        }
        preservation[name] = {
            "max_abs_bit_category_probability_change": float(
                (probs[:, list(BIT_IDS)].sum(-1) - base_bit_mass).abs().max()
            ),
            "max_abs_nonbit_probability_change": float(
                (probs[:, nonbit] - base_probs[:, nonbit]).abs().max()
            ),
        }
    inactive = torch.zeros_like(active)
    inactive_output = compose_bit_distribution(
        base, F.log_softmax(learned(representation), -1), inactive
    )
    checks = {
        "inactive_output_exactly_unchanged": inactive_output is base,
        "inactive_tensor_equal": torch.equal(inactive_output, base),
        "preservation": preservation,
    }
    return metrics, checks


@torch.inference_mode()
def evaluate_interventions(
    records: list[dict],
    base_logits: torch.Tensor,
    learned: DirectLookupReadout,
    control: DirectLookupReadout,
    device: torch.device,
) -> dict:
    base = torch.log_softmax(base_logits.to(device), -1)
    original = torch.tensor([item["original_value"] for item in records], device=device)
    alternate = torch.tensor([item["alternate_value"] for item in records], device=device)
    if not torch.all(original != alternate):
        raise AssertionError("every intervention must change the retrieved literal")
    active = torch.ones(len(records), dtype=torch.bool, device=device)
    original_rep = literal_representation(original)
    alternate_rep = literal_representation(alternate)
    conditions = {
        "baseline": (base, base, None, None),
        "deterministic_copy": (
            compose_bit_distribution(base, deterministic_copy_log_probs(original), active),
            compose_bit_distribution(base, deterministic_copy_log_probs(alternate), active),
            deterministic_copy_log_probs(original), deterministic_copy_log_probs(alternate),
        ),
        "learned_readout": (
            compose_bit_distribution(base, F.log_softmax(learned(original_rep), -1), active),
            compose_bit_distribution(base, F.log_softmax(learned(alternate_rep), -1), active),
            F.log_softmax(learned(original_rep), -1),
            F.log_softmax(learned(alternate_rep), -1),
        ),
        "value_independent_control": (
            compose_bit_distribution(
                base, F.log_softmax(control(torch.zeros_like(original_rep)), -1), active
            ),
            compose_bit_distribution(
                base, F.log_softmax(control(torch.zeros_like(alternate_rep)), -1), active
            ),
            F.log_softmax(control(torch.zeros_like(original_rep)), -1),
            F.log_softmax(control(torch.zeros_like(alternate_rep)), -1),
        ),
    }
    result = {}
    indices = torch.arange(len(records), device=device)
    for name, (before, after, reader_before, reader_after) in conditions.items():
        before_bits = F.log_softmax(before[:, list(BIT_IDS)], -1)
        after_bits = F.log_softmax(after[:, list(BIT_IDS)], -1)
        probability_delta = (
            after_bits.exp()[indices, alternate]
            - before_bits.exp()[indices, alternate]
        )
        before_prediction = before_bits.argmax(-1)
        after_prediction = after_bits.argmax(-1)
        # Clamp only for this descriptive statistic so exact deterministic copy
        # remains representable in strict JSON instead of producing infinities.
        tiny = torch.finfo(after_bits.dtype).tiny
        before_probability = before_bits.exp().clamp_min(tiny)
        after_probability = after_bits.exp().clamp_min(tiny)
        log_odds_shift = (
            after_probability[indices, alternate].log()
            - after_probability[indices, original].log()
            - before_probability[indices, alternate].log()
            + before_probability[indices, original].log()
        )
        if reader_before is None:
            reader_delta = torch.zeros(len(records), device=device)
        else:
            # Probability-space distance remains finite for deterministic one-hot copy.
            reader_delta = (reader_after.exp() - reader_before.exp()).norm(dim=-1)
        result[name] = {
            "eligible_cases": len(records),
            "retrieved_literal_changed": int((original != alternate).sum()),
            "conditional_accuracy_correct_pointer": float(
                (before_prediction == original).float().mean()
            ),
            "conditional_accuracy_intervened_pointer": float(
                (after_prediction == alternate).float().mean()
            ),
            "prediction_changed_toward_new_required": float(
                ((before_prediction == original) & (after_prediction == alternate))
                .float().mean()
            ),
            "new_required_normalized_probability_delta": _summary(
                probability_delta.cpu().tolist()
            ),
            "signed_log_odds_shift_toward_new_required": _summary(
                log_odds_shift.cpu().tolist()
            ),
            "reader_distribution_delta_l2": _summary(reader_delta.cpu().tolist()),
        }
    return result


def benchmark_overhead(
    learned: DirectLookupReadout,
    answer_logits: torch.Tensor,
    examples: list[LookupExample],
    device: torch.device,
) -> dict:
    base = torch.log_softmax(answer_logits.to(device), -1)
    values = _retrieved_values(examples).to(device)
    representation = literal_representation(values)
    active = torch.ones(len(examples), dtype=torch.bool, device=device)

    def call():
        reader = F.log_softmax(learned(representation), -1)
        return compose_bit_distribution(base, reader, active)

    for _ in range(20):
        call()
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    repeats = 200
    for _ in range(repeats):
        call()
    torch.cuda.synchronize(device)
    gpu_seconds = (time.perf_counter() - started) / repeats

    ids = [list(example.token_ids) for example in examples]
    for item in ids:
        trace_lookup_prefix(item)
    started = time.perf_counter()
    cpu_repeats = 20
    for _ in range(cpu_repeats):
        for item in ids:
            trace_lookup_prefix(item)
    prefix_seconds = (time.perf_counter() - started) / cpu_repeats
    return {
        "documents_per_measurement": len(examples),
        "warmup_iterations": 20,
        "gpu_timed_iterations": repeats,
        "prefix_cpu_microseconds_per_document": 1e6 * prefix_seconds / len(examples),
        "readout_and_logspace_composition_gpu_microseconds_per_document": (
            1e6 * gpu_seconds / len(examples)
        ),
        "combined_microseconds_per_document": (
            1e6 * (prefix_seconds + gpu_seconds) / len(examples)
        ),
        "cuda_synchronized": True,
    }


def _acceptance(
    metrics: dict,
    variants: dict,
    interventions: dict,
    checks: dict,
    causality: dict,
    training: dict,
    budget: dict,
    config: dict,
) -> dict:
    thresholds = config["acceptance"]
    learned = metrics["learned_readout"]
    control = metrics["value_independent_control"]
    preservation = checks["preservation"]["learned_readout"]
    criteria = {
        "heldout_conditional_accuracy": (
            learned["conditional_bit_accuracy"] >= thresholds["conditional_accuracy"]
        ),
        "opposite_pointer_conditional_accuracy": (
            interventions["learned_readout"]["conditional_accuracy_intervened_pointer"]
            >= thresholds["intervention_accuracy"]
        ),
        "opposite_pointer_predictions_change_toward_new": (
            interventions["learned_readout"]["prediction_changed_toward_new_required"]
            >= thresholds["intervention_accuracy"]
        ),
        "renaming_conditional_accuracy": (
            variants["renamed"]["learned_readout"]["conditional_bit_accuracy"]
            >= thresholds["variant_accuracy"]
        ),
        "whitespace_conditional_accuracy": (
            variants["whitespace"]["learned_readout"]["conditional_bit_accuracy"]
            >= thresholds["variant_accuracy"]
        ),
        "bit_category_preserved": (
            preservation["max_abs_bit_category_probability_change"]
            <= thresholds["probability_preservation_tolerance"]
        ),
        "nonbit_probabilities_preserved": (
            preservation["max_abs_nonbit_probability_change"]
            <= thresholds["probability_preservation_tolerance"]
        ),
        "inactive_output_exact": checks["inactive_output_exactly_unchanged"],
        "value_independent_control_at_chance": (
            abs(control["conditional_bit_accuracy"] - 0.5)
            <= thresholds["control_chance_tolerance"]
        ),
        "prefix_causality": causality["all_prefix_fields_invariant_to_unseen_suffix"],
        "update_budget": training["updates"] <= config["budget"]["max_updates"],
        "supervision_budget": (
            training["supervised_answer_count"]
            <= config["budget"]["max_supervised_answers"]
        ),
        "accelerator_budget": budget["within_cap"],
    }
    return {"criteria": criteria, "passed": all(criteria.values())}


def _markdown(report: dict) -> str:
    metrics = report["heldout_metrics"]
    lines = [
        "# Phase 1c direct lookup-readout",
        "",
        f"**Acceptance: {'PASS' if report['acceptance']['passed'] else 'FAIL'}.** "
        "This establishes output-facing lookup integration only. The original Phase 1 "
        "verdict remains **failed composition**.",
        "",
        "## Conditions on the frozen `(6,256)`, seed-11 plain backbone",
        "",
        "| Condition | Conditional accuracy | Conditional NLL | Full-vocab accuracy | Answer NLL | Mean bit-category probability |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in (
        "baseline", "deterministic_copy", "learned_readout",
        "value_independent_control",
    ):
        item = metrics[name]
        lines.append(
            f"| {name} | {item['conditional_bit_accuracy']:.4f} | "
            f"{item['conditional_bit_nll']:.6f} | "
            f"{item['full_vocabulary_answer_accuracy']:.4f} | "
            f"{item['answer_nll']:.6f} | "
            f"{item['mean_bit_category_probability']:.6f} |"
        )
    learned_intervention = report["pointer_interventions"]["learned_readout"]
    train_conditional = report["training"]["final_conditional_metrics"]["train"]
    lines.extend((
        "", "The output module preserves the backbone's total bit mass and replaces only "
        "the conditional split between `0` and `1`; full-vocabulary accuracy therefore "
        "remains limited by the frozen backbone's category detection.",
        "", "## Learned readout and causal controls", "",
        f"The learned readout has {report['training']['learned_trainable_parameters']} "
        f"trainable parameters and stopped at {report['training']['updates']} updates after "
        f"{report['training']['supervised_answer_count']} supervised answers. The frozen "
        "backbone has no trainable parameters in this run. Training conditional accuracy/NLL "
        f"were {train_conditional['learned_readout']['accuracy']:.4f}/"
        f"{train_conditional['learned_readout']['nll']:.6f}; the value-independent control "
        f"was {train_conditional['value_independent_control']['accuracy']:.4f}/"
        f"{train_conditional['value_independent_control']['nll']:.6f}.",
        "",
        f"Held-out opposite-value pointer cases: "
        f"{learned_intervention['eligible_cases']}; retrieved literal changed in "
        f"{learned_intervention['retrieved_literal_changed']}. Intervened conditional "
        f"accuracy: {learned_intervention['conditional_accuracy_intervened_pointer']:.4f}; "
        f"predictions changed from the old to new required bit in "
        f"{learned_intervention['prediction_changed_toward_new_required']:.4f}. Mean "
        f"normalized P(new bit) change: "
        f"{learned_intervention['new_required_normalized_probability_delta']['mean']:+.6f}; "
        f"mean signed log-odds shift: "
        f"{learned_intervention['signed_log_odds_shift_toward_new_required']['mean']:+.6f}.",
        "",
        "| Variant | Learned conditional accuracy | Control conditional accuracy |",
        "| --- | ---: | ---: |",
    ))
    for name in ("renamed", "whitespace"):
        item = report["variant_metrics"][name]
        lines.append(
            f"| {name} | {item['learned_readout']['conditional_bit_accuracy']:.4f} | "
            f"{item['value_independent_control']['conditional_bit_accuracy']:.4f} |"
        )
    preservation = report["composition_checks"]["preservation"]["learned_readout"]
    overhead = report["overhead"]
    lines.extend((
        "", "## Exactness, causality, and budget", "",
        f"Maximum bit-category probability change: "
        f"{preservation['max_abs_bit_category_probability_change']:.3e}; maximum non-bit "
        f"probability change: {preservation['max_abs_nonbit_probability_change']:.3e}. "
        f"Inactive output is exactly unchanged: "
        f"{report['composition_checks']['inactive_output_exactly_unchanged']}.",
        "",
        f"Unseen-suffix prefix comparisons: "
        f"{report['causality']['prefix_comparisons']}; all addresses, activation, and "
        "retrieved values were invariant. XOR cases inactive: "
        f"{report['causality']['xor_documents_inactive']}/"
        f"{report['causality']['xor_documents_checked']}.",
        "",
        f"Measured overhead: {overhead['prefix_cpu_microseconds_per_document']:.3f} "
        f"us/document for prefix state and "
        f"{overhead['readout_and_logspace_composition_gpu_microseconds_per_document']:.3f} "
        "us/document for synchronized readout/composition. Total accelerator time: "
        f"{report['budget']['accelerator_seconds']:.3f}s / 300s.",
        "", "## Acceptance gate", "",
    ))
    for name, passed in report["acceptance"]["criteria"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'}: {name}")
    lines.extend((
        "", "All pre-existing artifact hashes are unchanged. Passing does not establish "
        "XOR, backbone-layer use, efficiency, or learned-specialist success. Any later "
        "learned-specialist pilot still requires the original >=99% binding and "
        "structural-event accuracy gates.",
        "", "Reproduce:", "",
        "`python -m experiments.modular_phase1c --config "
        "experiments/modular_phase1c/configs/diagnostic.json --run-dir "
        "scratch/modular-phase1c --force`", "",
    ))
    return "\n".join(lines)


def run(config: dict, run_dir: Path, *, force: bool = False) -> dict:
    report_path = run_dir / "report.json"
    if report_path.exists() and not force:
        return json.loads(report_path.read_text(encoding="utf-8"))
    source_run = Path(config["source_phase1_run_dir"])
    source_phase1b = Path(config["source_phase1b_run_dir"])
    hashes_before = _source_hashes([source_run, source_phase1b])
    examples, data = prepare_examples(source_run)
    manifest = json.loads((source_run / "data" / "manifest.json").read_text(encoding="utf-8"))
    confirmation_documents = load_corpus(manifest["files"]["confirmation"]["path"])
    heldout_documents = data.pop("base_documents")["heldout"]
    xor_documents = [
        document for document in confirmation_documents
        if document.metadata.get("variant") == "base"
        and document.metadata["query_kind"] == "xor"
    ]
    causality = verify_prefix_causality(examples["heldout"], xor_documents)
    interventions = build_interventions(heldout_documents)

    device = torch.device(config["runtime"]["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Phase 1c requires the configured accelerator")
    torch.cuda.synchronize(device)
    accelerator_started = time.perf_counter()

    phase1_config = load_config(config["source_phase1_config"])
    checkpoint_path = Path(config["base_checkpoint"])
    base_checkpoint_hash_before = file_hash(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (checkpoint["arm"], checkpoint["layers"], checkpoint["width"], checkpoint["seed"]) != (
        "plain", 6, 256, 11,
    ):
        raise RuntimeError("Phase 1c base checkpoint identity mismatch")
    backbone = _load_model(phase1_config, source_run, checkpoint, device)
    backbone.requires_grad_(False).eval()
    base_state_hash = _state_hash(backbone.state_dict())

    learned, control, training = train_readouts(
        examples, interventions, config, device
    )
    cache_seconds = 0.0
    cached = {}
    for split in ("heldout", "renamed", "whitespace"):
        cached[split], elapsed = cache_answer_logits(
            backbone, examples[split],
            batch_size=int(phase1_config["evaluation"]["batch_sizes"]["256"]),
            device=device, bf16=bool(config["runtime"]["bf16"]),
        )
        cache_seconds += elapsed
    heldout_metrics, composition_checks = evaluate_conditions(
        cached["heldout"], examples["heldout"], learned, control, device
    )
    variant_metrics = {}
    for split in ("renamed", "whitespace"):
        variant_metrics[split] = evaluate_conditions(
            cached[split], examples[split], learned, control, device
        )[0]

    original_indices = list(range(0, len(examples["heldout"]), 2))
    original_logits = cached["heldout"][original_indices]
    if len(original_logits) != len(heldout_documents):
        raise AssertionError("heldout base-logit alignment failed")
    intervention_lookup = {item["document_id"]: item for item in interventions}
    intervention_indices = [
        index for index, document in enumerate(heldout_documents)
        if document.document_id in intervention_lookup
    ]
    ordered_interventions = [
        intervention_lookup[heldout_documents[index].document_id]
        for index in intervention_indices
    ]
    pointer_metrics = evaluate_interventions(
        ordered_interventions, original_logits[intervention_indices],
        learned, control, device,
    )
    overhead = benchmark_overhead(
        learned, cached["heldout"], examples["heldout"], device
    )
    torch.cuda.synchronize(device)
    accelerator_seconds = time.perf_counter() - accelerator_started
    budget = {
        "accelerator_seconds": accelerator_seconds,
        "accelerator_seconds_cap": float(config["budget"]["max_accelerator_seconds"]),
        "within_cap": accelerator_seconds <= float(config["budget"]["max_accelerator_seconds"]),
        "frozen_backbone_cache_seconds": cache_seconds,
    }

    run_dir.mkdir(parents=True, exist_ok=True)
    readout_checkpoint = {
        "schema": 1,
        "phase": config["phase"],
        "config_sha256": config["_config_sha256"],
        "base_checkpoint_sha256": file_hash(checkpoint_path),
        "base_state_sha256": base_state_hash,
        "learned_state_dict": {
            name: value.detach().cpu() for name, value in learned.state_dict().items()
        },
        "control_state_dict": {
            name: value.detach().cpu() for name, value in control.state_dict().items()
        },
        "training": training,
    }
    readout_path = run_dir / "readout.pt"
    torch.save(readout_checkpoint, readout_path)
    hashes_after = _source_hashes([source_run, source_phase1b])
    if hashes_after != hashes_before:
        raise RuntimeError("Phase 1c changed a pre-existing artifact")
    base_checkpoint_hash_after = file_hash(checkpoint_path)
    if base_checkpoint_hash_after != base_checkpoint_hash_before:
        raise RuntimeError("Phase 1c changed the frozen base checkpoint")

    report = {
        "schema": 1,
        "phase": config["phase"],
        "diagnostic_only": True,
        "preserved_phase1_verdict": "failed composition",
        "scope_limit": (
            "Output-facing lookup integration only; no XOR, backbone-layer-use, efficiency, "
            "or learned-specialist claim."
        ),
        "config_sha256": config["_config_sha256"],
        "device": torch.cuda.get_device_name(device),
        "base_checkpoint": str(checkpoint_path.resolve()),
        "base_checkpoint_sha256": base_checkpoint_hash_before,
        "base_checkpoint_sha256_before": base_checkpoint_hash_before,
        "base_checkpoint_sha256_after": base_checkpoint_hash_after,
        "base_state_sha256": base_state_hash,
        "base_frozen": not any(parameter.requires_grad for parameter in backbone.parameters()),
        "data": data,
        "causality": causality,
        "training": training,
        "heldout_metrics": heldout_metrics,
        "variant_metrics": variant_metrics,
        "pointer_interventions": pointer_metrics,
        "composition_checks": composition_checks,
        "overhead": overhead,
        "budget": budget,
        "readout_checkpoint": str(readout_path.resolve()),
        "readout_checkpoint_sha256": file_hash(readout_path),
        "source_hashes_before": hashes_before,
        "source_hashes_after": hashes_after,
        "source_artifacts_unchanged": True,
    }
    report["acceptance"] = _acceptance(
        heldout_metrics, variant_metrics, pointer_metrics, composition_checks,
        causality, training, budget, config,
    )
    _write_json(report_path, report)
    (run_dir / "REPORT.md").write_text(_markdown(report), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    report = run(_load_diagnostic_config(args.config), args.run_dir, force=args.force)
    print(json.dumps({
        "acceptance": report["acceptance"],
        "budget": report["budget"],
        "report": str((args.run_dir / "REPORT.md").resolve()),
    }, indent=2, sort_keys=True))

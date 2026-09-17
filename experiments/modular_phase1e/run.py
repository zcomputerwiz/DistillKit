"""Run the bounded Phase 1e category-admission experiment."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from experiments.modular_phase1.config import file_hash, load_config
from experiments.modular_phase1.data import collate_documents
from experiments.modular_phase1.evaluation import _load_model
from experiments.modular_phase1.language import Vocabulary
from experiments.modular_phase1.specialist import RecurrentSpecialist
from experiments.modular_phase1.training import _autocast, _state_hash
from experiments.modular_phase1c.models import (
    BIT_IDS,
    DirectLookupReadout,
    compose_bit_distribution,
    literal_representation,
)
from experiments.modular_phase1c.run import LookupExample
from experiments.modular_phase1d.deployment import deployed_lookup_trace
from experiments.modular_phase1d.run import (
    _programmed_interventions,
    benchmark_path,
    verify_causality,
)

from .data import FamilyDocument, prepare_phase1e_data
from .models import (
    FEATURE_NAMES,
    ConstantAdmission,
    ContextualAdmission,
    apply_category_admission,
    build_contextual_features,
)


@dataclass
class CachedDocument:
    family_id: str
    variant: str
    confirmation_slice: str
    document_id: str
    token_ids: tuple[int, ...]
    answer_position: int
    target_ids: torch.Tensor
    phase1d_target_log_probs: torch.Tensor
    phase1d_predictions: torch.Tensor
    phase1d_answer_log_probs: torch.Tensor
    active_indices: torch.Tensor
    phase1d_active_log_probs: torch.Tensor
    backbone_active_log_probs: torch.Tensor
    active_features: torch.Tensor
    active_targets: torch.Tensor
    answer_active_slot: int
    activated_at_answer: bool
    retrieved_at_answer: bool
    retrieved_value: int

    @property
    def exposures(self) -> int:
        return len(self.target_ids)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _config(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    value["_config_path"] = str(path.resolve())
    value["_config_sha256"] = file_hash(path)
    if tuple(value["features"]) != FEATURE_NAMES:
        raise RuntimeError("frozen contextual feature list does not match code")
    return value


def _hash_tree(roots: list[Path]) -> dict[str, str]:
    return {
        str(path.resolve()): file_hash(path)
        for root in roots
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix in (".pt", ".json", ".md", ".gz")
    }


def _load_frozen(config: dict, device: torch.device):
    source = Path(config["source_phase1_run_dir"])
    phase1_config = load_config(config["source_phase1_config"])
    base_checkpoint = torch.load(
        config["base_checkpoint"], map_location="cpu", weights_only=True
    )
    backbone = _load_model(phase1_config, source, base_checkpoint, device)
    backbone.requires_grad_(False).eval()

    specialist_checkpoint = torch.load(
        config["specialist_checkpoint"], map_location="cpu", weights_only=True
    )
    specialist = RecurrentSpecialist(**specialist_checkpoint["model"])
    specialist.load_state_dict(specialist_checkpoint["state_dict"])
    specialist.requires_grad_(False).eval().to(device)
    if not specialist_checkpoint["gate"]["passed"]:
        raise RuntimeError("Phase 1d specialist is not qualified")

    readout_checkpoint = torch.load(
        config["readout_checkpoint"], map_location="cpu", weights_only=True
    )
    readout = DirectLookupReadout()
    readout.load_state_dict(readout_checkpoint["learned_state_dict"])
    readout.requires_grad_(False).eval().to(device)
    return backbone, specialist, readout, phase1_config


@torch.inference_mode()
def cache_split(
    members: list[FamilyDocument],
    backbone,
    specialist: RecurrentSpecialist,
    readout: DirectLookupReadout,
    *,
    batch_size: int,
    device: torch.device,
    bf16: bool,
) -> tuple[list[CachedDocument], dict]:
    result = []
    forward_seconds = 0.0
    state_seconds = 0.0
    for offset in range(0, len(members), batch_size):
        group = members[offset:offset + batch_size]
        documents = [member.document for member in group]
        batch = collate_documents(documents).to(device)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        with _autocast(device, bf16):
            backbone_logits = backbone(batch.input_ids, attention_mask=batch.attention_mask)
            specialist_output = specialist(batch.input_ids)
        torch.cuda.synchronize(device)
        forward_seconds += time.perf_counter() - started
        backbone_log_probs = F.log_softmax(backbone_logits.float(), -1)
        features = build_contextual_features(
            batch.input_ids, backbone_log_probs,
            specialist_output["event_logits"], specialist_output["validity_logits"],
            batch.depths,
        )
        for row, member in enumerate(group):
            document = member.document
            length = len(document.token_ids)
            started = time.perf_counter()
            trace = deployed_lookup_trace(
                specialist, list(document.token_ids), device
            )
            state_seconds += time.perf_counter() - started
            active_full = torch.tensor(trace.active, dtype=torch.bool, device=device)
            values_full = torch.tensor(trace.retrieved_values, device=device).clamp_min(0)
            reader_log_probs = F.log_softmax(
                readout(literal_representation(values_full)), -1
            )
            phase1d_full = compose_bit_distribution(
                backbone_log_probs[row, :length], reader_log_probs, active_full
            )
            target_ids = batch.input_ids[row, 1:length]
            causal_active = active_full[: length - 1]
            active_indices = causal_active.nonzero(as_tuple=False).squeeze(-1)
            phase1d_targets = phase1d_full[: length - 1].gather(
                -1, target_ids.unsqueeze(-1)
            ).squeeze(-1)
            score = document.answer_position - 1
            matching = (active_indices == score).nonzero(as_tuple=False).squeeze(-1)
            answer_slot = int(matching[0]) if len(matching) else -1
            result.append(CachedDocument(
                family_id=member.family_id,
                variant=member.variant,
                confirmation_slice=document.metadata.get("confirmation_slice", "none"),
                document_id=document.document_id,
                token_ids=tuple(document.token_ids),
                answer_position=document.answer_position,
                target_ids=target_ids.cpu(),
                phase1d_target_log_probs=phase1d_targets.cpu(),
                phase1d_predictions=phase1d_full[: length - 1].argmax(-1).cpu(),
                phase1d_answer_log_probs=phase1d_full[score].cpu(),
                active_indices=active_indices.cpu(),
                phase1d_active_log_probs=phase1d_full[: length - 1][causal_active].cpu(),
                backbone_active_log_probs=backbone_log_probs[row, : length - 1][
                    causal_active
                ].cpu(),
                active_features=features[row, : length - 1][causal_active].cpu(),
                active_targets=target_ids[causal_active].cpu(),
                answer_active_slot=answer_slot,
                activated_at_answer=trace.activated[score],
                retrieved_at_answer=trace.active[score],
                retrieved_value=trace.retrieved_values[score],
            ))
    return result, {
        "documents": len(result),
        "causal_targets": sum(document.exposures for document in result),
        "active_targets": sum(len(document.active_targets) for document in result),
        "frozen_forward_seconds": forward_seconds,
        "deployment_state_seconds": state_seconds,
    }


def _active_batch(documents: list[CachedDocument], device: torch.device):
    return (
        torch.cat([document.phase1d_active_log_probs for document in documents]).to(device),
        torch.cat([document.active_features for document in documents]).to(device),
        torch.cat([document.active_targets for document in documents]).to(device),
    )


def verify_objective_and_initialization(
    cache: list[CachedDocument], device: torch.device
) -> dict:
    documents = cache[:32]
    log_probs, features, targets = _active_batch(documents, device)
    active = torch.ones(len(targets), dtype=torch.bool, device=device)
    model = ContextualAdmission().to(device)
    correction = model(features)
    output = apply_category_admission(log_probs, correction, active)
    zero_exact = torch.equal(output, log_probs)
    inactive = torch.zeros_like(active)
    inactive_output = apply_category_admission(log_probs, correction, inactive)
    inactive_exact = inactive_output is log_probs and torch.equal(inactive_output, log_probs)

    denominator = sum(document.exposures for document in documents)
    full_loss = F.nll_loss(output, targets, reduction="sum") / denominator
    full_gradient = torch.autograd.grad(full_loss, tuple(model.parameters()), retain_graph=True)
    active_output = apply_category_admission(log_probs, model(features), active)
    equivalent_loss = -active_output.gather(-1, targets[:, None]).sum() / denominator
    equivalent_gradient = torch.autograd.grad(equivalent_loss, tuple(model.parameters()))
    max_gradient_difference = max(
        float((left - right).abs().max())
        for left, right in zip(full_gradient, equivalent_gradient)
    )
    gradient_norm = sum(float(item.norm()) for item in full_gradient)
    return {
        "zero_initialized_agreement_exact": zero_exact,
        "inactive_identity_exact": inactive_exact,
        "gradient_path_nonzero": gradient_norm > 0,
        "gradient_norm": gradient_norm,
        "category_loss_gradient_max_abs_difference": max_gradient_difference,
        "gradient_equivalent": max_gradient_difference <= 1e-8,
        "documents_checked": len(documents),
        "causal_targets_in_denominator": denominator,
        "active_targets_in_numerator": len(targets),
    }


def train_arm(
    name: str,
    cache: list[CachedDocument],
    config: dict,
    device: torch.device,
) -> tuple[torch.nn.Module, dict]:
    torch.manual_seed(int(config["seed"]))
    model = (ConstantAdmission() if name == "constant" else ContextualAdmission()).to(device)
    settings = config["training"]
    optimizer_settings = config["optimizer"]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(optimizer_settings["learning_rate"]),
        betas=tuple(optimizer_settings["betas"]),
        eps=float(optimizer_settings["epsilon"]),
        weight_decay=float(optimizer_settings["weight_decay"]),
    )
    rng = random.Random(int(config["seed"]) + (0 if name == "constant" else 1))
    order = list(range(len(cache)))
    offset = updates = exposures = answers = 0
    losses = []
    started = time.perf_counter()
    while (
        updates < int(settings["max_updates"])
        and exposures < int(settings["max_target_exposures_per_arm"])
    ):
        if offset == 0:
            rng.shuffle(order)
        selected = []
        batch_exposures = 0
        while len(selected) < int(settings["documents_per_batch"]):
            index = order[offset]
            candidate = cache[index]
            if exposures + batch_exposures + candidate.exposures > int(
                settings["max_target_exposures_per_arm"]
            ):
                break
            selected.append(candidate)
            batch_exposures += candidate.exposures
            offset += 1
            if offset == len(order):
                offset = 0
                break
        if not selected:
            break
        log_probs, features, targets = _active_batch(selected, device)
        active = torch.ones(len(targets), dtype=torch.bool, device=device)
        optimizer.zero_grad(set_to_none=True)
        output = apply_category_admission(log_probs, model(features), active)
        loss = F.nll_loss(output, targets, reduction="sum") / batch_exposures
        loss.backward()
        optimizer.step()
        updates += 1
        exposures += batch_exposures
        answers += sum(
            int((document.active_targets == Vocabulary.TO_ID[str(
                document.retrieved_value
            )]).sum())
            for document in selected if document.retrieved_value in (0, 1)
        )
        losses.append(float(loss.detach()))
    torch.cuda.synchronize(device)
    return model.eval(), {
        "arm": name,
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "updates": updates,
        "supervised_causal_target_exposures": exposures,
        "supervised_answer_targets": answers,
        "mean_optimized_complete_document_ce_contribution": sum(losses) / len(losses),
        "last_optimized_complete_document_ce_contribution": losses[-1],
        "accelerator_seconds": time.perf_counter() - started,
    }


@torch.inference_mode()
def evaluate_cache(
    cache: list[CachedDocument],
    model: torch.nn.Module | None,
    device: torch.device,
) -> tuple[dict, list[dict]]:
    total_targets = answer_count = 0
    overall_nll = answer_nll = conditional_nll = 0.0
    overall_correct = answer_correct = conditional_correct = 0
    whitespace_count = whitespace_nll = whitespace_false_bit = 0
    other_count = other_nll = 0.0
    category_probability = 0.0
    class_stats = {}
    rows = []
    coverage = {"eligible": len(cache), "activated": 0, "retrieved": 0, "correct": 0}
    for document in cache:
        targets = document.target_ids.to(device)
        target_log_probs = document.phase1d_target_log_probs.to(device).clone()
        predictions = document.phase1d_predictions.to(device).clone()
        active_output = document.phase1d_active_log_probs.to(device)
        if model is not None and len(document.active_indices):
            features = document.active_features.to(device)
            correction = model(features)
            active = torch.ones(len(correction), dtype=torch.bool, device=device)
            active_output = apply_category_admission(active_output, correction, active)
            active_targets = document.active_targets.to(device)
            target_log_probs[document.active_indices] = active_output.gather(
                -1, active_targets[:, None]
            ).squeeze(-1)
            predictions[document.active_indices] = active_output.argmax(-1)
        nll = -target_log_probs
        total_targets += len(targets)
        overall_nll += float(nll.sum())
        overall_correct += int((predictions == targets).sum())
        for token_id in targets.unique().tolist():
            mask = targets == token_id
            record = class_stats.setdefault(Vocabulary.TOKENS[token_id], {
                "targets": 0, "nll_sum": 0.0, "correct": 0,
            })
            record["targets"] += int(mask.sum())
            record["nll_sum"] += float(nll[mask].sum())
            record["correct"] += int((predictions[mask] == targets[mask]).sum())

        score = document.answer_position - 1
        target = int(targets[score])
        if document.answer_active_slot >= 0:
            answer_output = active_output[document.answer_active_slot]
        else:
            # A missed activation remains exactly the frozen backbone/Phase 1d result.
            answer_output = document.phase1d_answer_log_probs.to(device)
        answer_count += 1
        answer_nll += float(nll[score])
        answer_correct += int(predictions[score] == target)
        bit_output = F.log_softmax(answer_output[list(BIT_IDS)], -1)
        bit_target = 0 if target == BIT_IDS[0] else 1
        conditional_nll += float(-bit_output[bit_target])
        conditional_correct += int(bit_output.argmax() == bit_target)
        category_probability += float(answer_output[list(BIT_IDS)].exp().sum())
        coverage["activated"] += int(document.activated_at_answer)
        coverage["retrieved"] += int(document.retrieved_at_answer)
        coverage["correct"] += int(
            document.retrieved_at_answer and document.retrieved_value == bit_target
        )
        rows.append({
            "family_id": document.family_id,
            "variant": document.variant,
            "confirmation_slice": document.confirmation_slice,
            "baseline_answer_correct": int(document.phase1d_predictions[score] == target),
            "answer_correct": int(predictions[score] == target),
            "answer_nll": float(nll[score]),
        })

        for slot, target_id in enumerate(document.active_targets.tolist()):
            output = active_output[slot]
            item_nll = float(-output[target_id])
            if Vocabulary.TOKENS[target_id] in Vocabulary.WHITESPACE:
                whitespace_count += 1
                whitespace_nll += item_nll
                whitespace_false_bit += int(int(output.argmax()) in BIT_IDS)
            elif target_id not in BIT_IDS:
                other_count += 1
                other_nll += item_nll
    per_class = {
        name: {
            "targets": value["targets"],
            "nll": value["nll_sum"] / value["targets"],
            "accuracy": value["correct"] / value["targets"],
        }
        for name, value in sorted(class_stats.items())
    }
    metrics = {
        "documents": len(cache),
        "causal_targets": total_targets,
        "overall_nll": overall_nll / total_targets,
        "overall_accuracy": overall_correct / total_targets,
        "answer_accuracy": answer_correct / answer_count,
        "answer_nll": answer_nll / answer_count,
        "conditional_lookup_accuracy": conditional_correct / answer_count,
        "conditional_lookup_nll": conditional_nll / answer_count,
        "mean_bit_category_probability_at_answers": category_probability / answer_count,
        "active_whitespace_continuations": {
            "targets": whitespace_count,
            "nll": whitespace_nll / max(1, whitespace_count),
            "false_bit_emission_rate": whitespace_false_bit / max(1, whitespace_count),
        },
        "other_active_nonanswer_targets": {
            "targets": other_count,
            "nll": other_nll / max(1, other_count),
        },
        "coverage": {
            **coverage,
            "activation_rate": coverage["activated"] / max(1, coverage["eligible"]),
            "retrieval_rate": coverage["retrieved"] / max(1, coverage["eligible"]),
            "correct_retrieval_rate": coverage["correct"] / max(1, coverage["eligible"]),
        },
        "per_target_class": per_class,
    }
    return metrics, rows


def compare_metrics(baseline: dict, arm: dict) -> dict:
    class_effects = {}
    for name in sorted(set(baseline["per_target_class"]) | set(arm["per_target_class"])):
        left = baseline["per_target_class"][name]
        right = arm["per_target_class"][name]
        class_effects[name] = {
            "targets": right["targets"],
            "accuracy_change": right["accuracy"] - left["accuracy"],
            "nll_change": right["nll"] - left["nll"],
        }
    return {
        "answer_accuracy_change": arm["answer_accuracy"] - baseline["answer_accuracy"],
        "answer_nll_change": arm["answer_nll"] - baseline["answer_nll"],
        "overall_nll_change": arm["overall_nll"] - baseline["overall_nll"],
        "whitespace_continuation_nll_change": (
            arm["active_whitespace_continuations"]["nll"]
            - baseline["active_whitespace_continuations"]["nll"]
        ),
        "other_active_nonanswer_nll_change": (
            arm["other_active_nonanswer_targets"]["nll"]
            - baseline["other_active_nonanswer_targets"]["nll"]
        ),
        "false_bit_emission_rate_change": (
            arm["active_whitespace_continuations"]["false_bit_emission_rate"]
            - baseline["active_whitespace_continuations"]["false_bit_emission_rate"]
        ),
        "per_target_class": class_effects,
    }


def _qualifies(baseline: dict, arm: dict, comparison: dict, thresholds: dict) -> dict:
    criteria = {
        "full_accuracy_gain": comparison["answer_accuracy_change"]
        >= float(thresholds["minimum_full_accuracy_gain"]),
        "answer_nll_improves": comparison["answer_nll_change"] < 0,
        "conditional_accuracy": arm["conditional_lookup_accuracy"]
        >= float(thresholds["minimum_conditional_accuracy"]),
        "whitespace_nll": comparison["whitespace_continuation_nll_change"]
        <= float(thresholds["maximum_whitespace_nll_degradation"]),
        "other_nonanswer_nll": comparison["other_active_nonanswer_nll_change"]
        <= float(thresholds["maximum_other_nonanswer_nll_degradation"]),
        "false_bit_emission": comparison["false_bit_emission_rate_change"]
        <= float(thresholds["maximum_false_bit_emission_increase"]),
    }
    return {"criteria": criteria, "passed": all(criteria.values())}


def paired_bootstrap(rows: list[dict], samples: int, seed: int) -> dict:
    by_family = {}
    for row in rows:
        record = by_family.setdefault(row["family_id"], [[], []])
        record[0].append(row["baseline_answer_correct"])
        record[1].append(row["answer_correct"])
    differences = [
        sum(selected) / len(selected) - sum(baseline) / len(baseline)
        for baseline, selected in by_family.values()
    ]
    rng = random.Random(seed)
    estimates = []
    for _ in range(samples):
        draw = [differences[rng.randrange(len(differences))] for _ in differences]
        estimates.append(sum(draw) / len(draw))
    estimates.sort()
    return {
        "families": len(differences),
        "samples": samples,
        "mean_paired_gain": sum(differences) / len(differences),
        "ci95_lower": estimates[int(0.025 * samples)],
        "ci95_upper": estimates[min(samples - 1, int(0.975 * samples))],
        "unit": "program_family_with_all_variants",
    }


@torch.inference_mode()
def evaluate_interventions(
    cache: list[CachedDocument],
    specialist: RecurrentSpecialist,
    readout: DirectLookupReadout,
    admission: torch.nn.Module,
    device: torch.device,
) -> dict:
    base_cache = [document for document in cache if document.variant == "base"]
    examples = [
        LookupExample(
            document.document_id, document.family_id, "confirmation",
            document.token_ids, document.answer_position,
            0 if document.token_ids[document.answer_position] == BIT_IDS[0] else 1,
            {}, False,
        )
        for document in base_cache
    ]
    indices, records = _programmed_interventions(examples)
    correct = changed = retrieved_changed = 0
    probability_deltas = []
    full_correct = 0
    for index, record in zip(indices, records):
        document = base_cache[index]
        example = examples[index]
        score_slot = document.answer_active_slot
        feature = document.active_features[score_slot:score_slot + 1].to(device)
        backbone = document.backbone_active_log_probs[score_slot:score_slot + 1].to(device)
        original_value = int(record["original_value"])
        alternate_value = int(record["alternate_value"])
        original_trace = deployed_lookup_trace(
            specialist, list(example.token_ids), device
        )
        changed_trace = deployed_lookup_trace(
            specialist, list(example.token_ids), device,
            pointer_override={record["use_position"]: record["alternate_pointer"]},
        )
        score = example.answer_position - 1
        original_active = torch.tensor([original_trace.active[score]], device=device)
        changed_active = torch.tensor([changed_trace.active[score]], device=device)
        before_phase1d = compose_bit_distribution(
            backbone,
            F.log_softmax(
                readout(literal_representation(torch.tensor([original_value], device=device))),
                -1,
            ),
            original_active,
        )
        after_phase1d = compose_bit_distribution(
            backbone,
            F.log_softmax(
                readout(literal_representation(torch.tensor([alternate_value], device=device))),
                -1,
            ),
            changed_active,
        )
        delta = admission(feature)
        before = apply_category_admission(before_phase1d, delta, original_active)
        after = apply_category_admission(after_phase1d, delta, changed_active)
        before_bit = F.softmax(before[0, list(BIT_IDS)], -1)
        after_bit = F.softmax(after[0, list(BIT_IDS)], -1)
        before_prediction = int(before_bit.argmax())
        after_prediction = int(after_bit.argmax())
        correct += int(after_prediction == alternate_value)
        changed += int(
            before_prediction == original_value and after_prediction == alternate_value
        )
        retrieved_changed += int(
            original_trace.retrieved_values[score] == original_value
            and changed_trace.retrieved_values[score] == alternate_value
        )
        probability_deltas.append(
            float(after_bit[alternate_value] - before_bit[alternate_value])
        )
        full_correct += int(int(after.argmax(-1)) == BIT_IDS[alternate_value])
    return {
        "eligible_cases": len(records),
        "retrieved_literal_changed": retrieved_changed,
        "conditional_accuracy_intervened": correct / max(1, len(records)),
        "prediction_changed_toward_new_required": changed / max(1, len(records)),
        "full_vocabulary_accuracy_intervened": full_correct / max(1, len(records)),
        "signed_normalized_new_bit_probability_delta": {
            "mean": sum(probability_deltas) / max(1, len(probability_deltas)),
            "minimum": min(probability_deltas) if probability_deltas else None,
            "maximum": max(probability_deltas) if probability_deltas else None,
        },
    }


@torch.inference_mode()
def verify_preservation(
    cache: list[CachedDocument], model: torch.nn.Module, device: torch.device
) -> dict:
    documents = cache[:128]
    base, features, _ = _active_batch(documents, device)
    active = torch.ones(len(base), dtype=torch.bool, device=device)
    output = apply_category_admission(base, model(features), active)
    bit_before = F.softmax(base[:, list(BIT_IDS)], -1)
    bit_after = F.softmax(output[:, list(BIT_IDS)], -1)
    nonbit = [index for index in range(len(Vocabulary.TOKENS)) if index not in BIT_IDS]
    nonbit_before = F.softmax(base[:, nonbit], -1)
    nonbit_after = F.softmax(output[:, nonbit], -1)
    inactive = torch.zeros_like(active)
    inactive_output = apply_category_admission(base, model(features), inactive)
    return {
        "active_rows_checked": len(base),
        "max_abs_bit_conditional_change": float((bit_after - bit_before).abs().max()),
        "max_abs_nonbit_conditional_change": float(
            (nonbit_after - nonbit_before).abs().max()
        ),
        "inactive_identity_exact": inactive_output is base and torch.equal(inactive_output, base),
    }


def benchmark_admission(
    cache: list[CachedDocument], model: torch.nn.Module, device: torch.device
) -> dict:
    base, features, _ = _active_batch(cache[:64], device)
    active = torch.ones(len(base), dtype=torch.bool, device=device)
    for _ in range(20):
        apply_category_admission(base, model(features), active)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    repeats = 500
    for _ in range(repeats):
        apply_category_admission(base, model(features), active)
    torch.cuda.synchronize(device)
    return {
        "active_prefixes_per_iteration": len(base),
        "warmup_iterations": 20,
        "timed_iterations": repeats,
        "cuda_synchronized": True,
        "admission_microseconds_per_active_prefix": (
            (time.perf_counter() - started) * 1e6 / (repeats * len(base))
        ),
    }


@torch.inference_mode()
def correction_statistics(
    cache: list[CachedDocument], model: torch.nn.Module, device: torch.device
) -> dict:
    _, features, targets = _active_batch(cache, device)
    correction = model(features)
    whitespace = torch.tensor(
        [Vocabulary.TOKENS[int(target)] in Vocabulary.WHITESPACE for target in targets],
        device=device,
    )
    bit = (targets == BIT_IDS[0]) | (targets == BIT_IDS[1])

    def summary(values: torch.Tensor) -> dict:
        return {
            "count": len(values),
            "mean": float(values.mean()) if len(values) else None,
            "minimum": float(values.min()) if len(values) else None,
            "maximum": float(values.max()) if len(values) else None,
        }

    return {
        "all_active": summary(correction),
        "bit_targets": summary(correction[bit]),
        "whitespace_targets": summary(correction[whitespace]),
    }


def _selection(
    baseline: dict,
    arms: dict,
    thresholds: dict,
) -> tuple[str | None, dict]:
    decisions = {}
    for name in ("constant", "contextual"):
        decisions[name] = _qualifies(
            baseline, arms[name]["metrics"], arms[name]["comparison"], thresholds
        )
    selected = "constant" if decisions["constant"]["passed"] else (
        "contextual" if decisions["contextual"]["passed"] else None
    )
    return selected, {
        "order": ["constant", "contextual"],
        "rule": "select the simplest calibration-qualifying arm; never use confirmation",
        "decisions": decisions,
        "selected": selected,
    }


def _markdown(report: dict) -> str:
    selection = report["selection"]
    lines = [
        "# Phase 1e category admission",
        "",
        f"**Acceptance: {'PASS' if report['acceptance']['passed'] else 'FAIL'}.** "
        "All Phase 1 through 1d artifacts and verdicts remain unchanged.",
        "",
        "## Calibration-only arm selection",
        "",
        "| Condition | Answer accuracy | Answer NLL | Overall NLL | Whitespace NLL | False-bit rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    baseline = report["calibration"]["phase1d"]
    lines.append(
        f"| unchanged_phase1d | {baseline['answer_accuracy']:.4f} | "
        f"{baseline['answer_nll']:.6f} | {baseline['overall_nll']:.6f} | "
        f"{baseline['active_whitespace_continuations']['nll']:.6f} | "
        f"{baseline['active_whitespace_continuations']['false_bit_emission_rate']:.4f} |"
    )
    for name in ("constant", "contextual"):
        item = report["calibration"][name]["metrics"]
        lines.append(
            f"| {name} | {item['answer_accuracy']:.4f} | {item['answer_nll']:.6f} | "
            f"{item['overall_nll']:.6f} | "
            f"{item['active_whitespace_continuations']['nll']:.6f} | "
            f"{item['active_whitespace_continuations']['false_bit_emission_rate']:.4f} |"
        )
    lines.extend((
        "",
        f"Selected on calibration: `{selection['selected']}`. Confirmation was evaluated "
        + (
            "once after this choice."
            if selection["selected"] is not None
            else "zero times because neither arm qualified."
        ),
    ))
    constant_training = report["calibration"]["constant"]["training"]
    contextual_training = report["calibration"]["contextual"]["training"]
    diagnostics = report["calibration_diagnostics"]
    lines.extend((
        "",
        "| Arm | Parameters | Updates | Causal-target exposures | Answer targets | Train seconds |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        f"| constant | {constant_training['trainable_parameters']} | "
        f"{constant_training['updates']} | "
        f"{constant_training['supervised_causal_target_exposures']} | "
        f"{constant_training['supervised_answer_targets']} | "
        f"{constant_training['accelerator_seconds']:.3f} |",
        f"| contextual | {contextual_training['trainable_parameters']} | "
        f"{contextual_training['updates']} | "
        f"{contextual_training['supervised_causal_target_exposures']} | "
        f"{contextual_training['supervised_answer_targets']} | "
        f"{contextual_training['accelerator_seconds']:.3f} |",
        "",
        f"Calibration interventions retained correct conditional changes in "
        f"{diagnostics['interventions']['constant']['prediction_changed_toward_new_required']:.4f} "
        f"(constant) and "
        f"{diagnostics['interventions']['contextual']['prediction_changed_toward_new_required']:.4f} "
        f"(contextual) across "
        f"{diagnostics['interventions']['contextual']['eligible_cases']} cases. Prefix "
        f"causality passed {diagnostics['causality']['prefix_comparisons']} comparisons.",
        "",
        f"Maximum within-bit/non-bit conditional changes were "
        f"{diagnostics['preservation']['contextual']['max_abs_bit_conditional_change']:.3e}/"
        f"{diagnostics['preservation']['contextual']['max_abs_nonbit_conditional_change']:.3e}; "
        f"inactive identity was exact. Contextual admission overhead was "
        f"{diagnostics['runtime_overhead']['admission']['contextual']['admission_microseconds_per_active_prefix']:.3f} "
        "us per active prefix.",
        "",
        f"Frozen backbone/specialist/readout hashes: "
        f"`{report['frozen_checkpoint_hashes_after']['backbone']}` / "
        f"`{report['frozen_checkpoint_hashes_after']['specialist']}` / "
        f"`{report['frozen_checkpoint_hashes_after']['readout']}`. Admission checkpoint "
        f"hash: `{report['admission_checkpoint_sha256']}`. All prior artifact hashes are "
        "unchanged.",
    ))
    if "confirmation" in report:
        baseline = report["confirmation"]["phase1d"]
        selected = report["confirmation"]["selected"]
        comparison = report["confirmation"]["comparison"]
        bootstrap = report["confirmation"]["paired_bootstrap"]
        lines.extend((
            "", "## Fresh confirmation", "",
            f"Full-vocabulary answer accuracy: {baseline['answer_accuracy']:.4f} -> "
            f"{selected['answer_accuracy']:.4f} "
            f"({comparison['answer_accuracy_change']:+.4f}); paired family-bootstrap 95% CI "
            f"[{bootstrap['ci95_lower']:+.4f}, {bootstrap['ci95_upper']:+.4f}].",
            "",
            f"Answer NLL: {baseline['answer_nll']:.6f} -> "
            f"{selected['answer_nll']:.6f}; conditional lookup accuracy: "
            f"{selected['conditional_lookup_accuracy']:.4f}; coverage: "
            f"{selected['coverage']['correct_retrieval_rate']:.4f}.",
            "",
            f"Whitespace-continuation NLL change: "
            f"{comparison['whitespace_continuation_nll_change']:+.6f}; other active "
            f"non-answer NLL change: {comparison['other_active_nonanswer_nll_change']:+.6f}; "
            f"false-bit emission change: "
            f"{comparison['false_bit_emission_rate_change']:+.4f}.",
            "",
            f"Opposite-value interventions: "
            f"{report['interventions']['eligible_cases']}; correct changed prediction: "
            f"{report['interventions']['prediction_changed_toward_new_required']:.4f}.",
        ))
    lines.extend((
        "", "Passing establishes useful category admission only on this frozen backbone. "
        "Cross-scale reuse, matched-quality efficiency, XOR, and backbone-layer integration "
        "remain out of scope.",
        "", "Reproduce:", "",
        "`python -m experiments.modular_phase1e --config "
        "experiments/modular_phase1e/configs/diagnostic.json --run-dir "
        "scratch/modular-phase1e --force`",
    ))
    return "\n".join(lines) + "\n"


def run(config: dict, run_dir: Path, *, force: bool = False) -> dict:
    report_path = run_dir / "report.json"
    if report_path.exists() and not force:
        return json.loads(report_path.read_text(encoding="utf-8"))
    source_roots = [
        Path(config["source_phase1_run_dir"]),
        Path(config["source_phase1b_run_dir"]),
        Path(config["source_phase1c_run_dir"]),
        Path(config["source_phase1d_run_dir"]),
    ]
    hashes_before = _hash_tree(source_roots)
    frozen_paths = {
        "backbone": Path(config["base_checkpoint"]),
        "specialist": Path(config["specialist_checkpoint"]),
        "readout": Path(config["readout_checkpoint"]),
    }
    frozen_hashes_before = {name: file_hash(path) for name, path in frozen_paths.items()}
    device = torch.device(config["runtime"]["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Phase 1e requires the configured accelerator")
    torch.cuda.synchronize(device)
    run_started = time.perf_counter()
    run_dir.mkdir(parents=True, exist_ok=True)
    splits, data_manifest = prepare_phase1e_data(config, source_roots[0], run_dir)
    backbone, specialist, readout, phase1_config = _load_frozen(config, device)
    frozen_states_before = {
        "backbone": _state_hash(backbone.state_dict()),
        "specialist": _state_hash(specialist.state_dict()),
        "readout": _state_hash(readout.state_dict()),
    }

    cache = {}
    cache_costs = {}
    batch_size = int(phase1_config["evaluation"]["batch_sizes"]["256"])
    for split in ("train", "calibration"):
        cache[split], cache_costs[split] = cache_split(
            splits[split], backbone, specialist, readout,
            batch_size=batch_size, device=device,
            bf16=bool(config["runtime"]["bf16"]),
        )
    interface_checks = verify_objective_and_initialization(cache["train"], device)

    arms = {}
    for name in ("constant", "contextual"):
        model, training = train_arm(name, cache["train"], config, device)
        metrics, _ = evaluate_cache(cache["calibration"], model, device)
        training["calibration_correction_statistics"] = correction_statistics(
            cache["calibration"], model, device
        )
        arms[name] = {"model": model, "training": training, "metrics": metrics}
    calibration_baseline, _ = evaluate_cache(cache["calibration"], None, device)
    for name in ("constant", "contextual"):
        arms[name]["comparison"] = compare_metrics(
            calibration_baseline, arms[name]["metrics"]
        )
    selected_name, selection = _selection(
        calibration_baseline, arms, config["selection"]
    )
    calibration_examples = [
        LookupExample(
            member.document.document_id, member.family_id, "calibration",
            tuple(member.document.token_ids), member.document.answer_position,
            member.document.answer, dict(member.document.metadata), False,
        )
        for member in splits["calibration"]
    ]
    calibration_diagnostics = {
        "causality": verify_causality(specialist, calibration_examples, device),
        "preservation": {
            name: verify_preservation(cache["calibration"], arms[name]["model"], device)
            for name in ("constant", "contextual")
        },
        "interventions": {
            name: evaluate_interventions(
                cache["calibration"], specialist, readout, arms[name]["model"], device
            )
            for name in ("constant", "contextual")
        },
        "runtime_overhead": {
            "frozen_path": benchmark_path(
                backbone, specialist, readout, calibration_examples, device,
                bool(config["runtime"]["bf16"]),
            ),
            "admission": {
                name: benchmark_admission(
                    cache["calibration"], arms[name]["model"], device
                )
                for name in ("constant", "contextual")
            },
        },
    }

    report = {
        "schema": 1,
        "phase": config["phase"],
        "preserved_verdicts": {
            "phase1": "failed composition",
            "phase1c": "output-facing lookup integration only",
            "phase1d": "learned-specialist delivery only",
        },
        "config_sha256": config["_config_sha256"],
        "frozen_feature_list": list(FEATURE_NAMES),
        "retrieved_bit_identity_in_features": False,
        "data": data_manifest,
        "cache_costs": cache_costs,
        "interface_checks": interface_checks,
        "calibration": {
            "phase1d": calibration_baseline,
            **{
                name: {
                    "metrics": arms[name]["metrics"],
                    "comparison": arms[name]["comparison"],
                    "training": arms[name]["training"],
                }
                for name in ("constant", "contextual")
            },
        },
        "selection": selection,
        "confirmation_evaluation_count": 0,
        "calibration_diagnostics": calibration_diagnostics,
    }
    admission_dir = run_dir / "admission"
    admission_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = admission_dir / "admission.pt"
    if selected_name is not None:
        selected = arms[selected_name]["model"]
        selected_parameters = sum(parameter.numel() for parameter in selected.parameters())
        if selected_parameters > 33:
            raise RuntimeError("selected admission arm exceeds 33 parameters")
        # Confirmation is first materialized only after the calibration-only choice.
        cache["confirmation"], cache_costs["confirmation"] = cache_split(
            splits["confirmation"], backbone, specialist, readout,
            batch_size=batch_size, device=device,
            bf16=bool(config["runtime"]["bf16"]),
        )
        confirmation_baseline, _ = evaluate_cache(cache["confirmation"], None, device)
        confirmation_selected, confirmation_rows = evaluate_cache(
            cache["confirmation"], selected, device
        )
        report["confirmation_evaluation_count"] = 1
        confirmation_comparison = compare_metrics(
            confirmation_baseline, confirmation_selected
        )
        bootstrap = paired_bootstrap(
            confirmation_rows,
            int(config["acceptance"]["bootstrap_samples"]),
            int(config["seed"]) + 99,
        )
        variant_metrics = {}
        for variant in ("base", "renamed", "whitespace"):
            subset = [document for document in cache["confirmation"] if document.variant == variant]
            variant_metrics[variant] = evaluate_cache(subset, selected, device)[0]
        slice_metrics = {}
        for slice_name in ("standard", "depth_ood", "long_history"):
            subset = [
                document for document in cache["confirmation"]
                if document.confirmation_slice == slice_name
            ]
            baseline_slice = evaluate_cache(subset, None, device)[0]
            selected_slice = evaluate_cache(subset, selected, device)[0]
            slice_metrics[slice_name] = {
                "phase1d": baseline_slice,
                "selected": selected_slice,
                "comparison": compare_metrics(baseline_slice, selected_slice),
            }
        interventions = evaluate_interventions(
            cache["confirmation"], specialist, readout, selected, device
        )
        confirmation_examples = [
            LookupExample(
                member.document.document_id, member.family_id, "confirmation",
                tuple(member.document.token_ids), member.document.answer_position,
                member.document.answer, dict(member.document.metadata), False,
            )
            for member in splits["confirmation"]
        ]
        causality = verify_causality(specialist, confirmation_examples, device)
        preservation = verify_preservation(cache["confirmation"], selected, device)
        path_overhead = benchmark_path(
            backbone, specialist, readout, confirmation_examples, device,
            bool(config["runtime"]["bf16"]),
        )
        admission_overhead = benchmark_admission(cache["confirmation"], selected, device)
        acceptance_thresholds = config["acceptance"]
        criteria = {
            "answer_accuracy_gain_at_least_five_points": (
                confirmation_comparison["answer_accuracy_change"]
                >= float(acceptance_thresholds["minimum_full_accuracy_gain"])
            ),
            "paired_ci_positive": bootstrap["ci95_lower"] > 0,
            "answer_nll_improves": confirmation_comparison["answer_nll_change"] < 0,
            "conditional_accuracy": (
                confirmation_selected["conditional_lookup_accuracy"]
                >= float(acceptance_thresholds["minimum_conditional_accuracy"])
            ),
            "renaming_conditional_accuracy": (
                variant_metrics["renamed"]["conditional_lookup_accuracy"]
                >= float(acceptance_thresholds["minimum_conditional_accuracy"])
            ),
            "whitespace_conditional_accuracy": (
                variant_metrics["whitespace"]["conditional_lookup_accuracy"]
                >= float(acceptance_thresholds["minimum_conditional_accuracy"])
            ),
            "intervention_accuracy": (
                interventions["conditional_accuracy_intervened"]
                >= float(acceptance_thresholds["minimum_intervention_accuracy"])
            ),
            "intervention_prediction_change": (
                interventions["prediction_changed_toward_new_required"]
                >= float(acceptance_thresholds["minimum_intervention_accuracy"])
            ),
            "whitespace_nll_degradation": (
                confirmation_comparison["whitespace_continuation_nll_change"]
                <= float(acceptance_thresholds["maximum_whitespace_nll_degradation"])
            ),
            "other_nonanswer_nll_degradation": (
                confirmation_comparison["other_active_nonanswer_nll_change"]
                <= float(
                    acceptance_thresholds["maximum_other_nonanswer_nll_degradation"]
                )
            ),
            "false_bit_emission": (
                confirmation_comparison["false_bit_emission_rate_change"]
                <= float(acceptance_thresholds["maximum_false_bit_emission_increase"])
            ),
            "bit_conditional_preserved": (
                preservation["max_abs_bit_conditional_change"]
                <= float(acceptance_thresholds["probability_preservation_tolerance"])
            ),
            "nonbit_conditional_preserved": (
                preservation["max_abs_nonbit_conditional_change"]
                <= float(acceptance_thresholds["probability_preservation_tolerance"])
            ),
            "inactive_identity_exact": preservation["inactive_identity_exact"],
            "prefix_causality": causality[
                "all_addresses_activation_values_and_predictions_invariant"
            ],
        }
        torch.save({
            "schema": 1,
            "phase": config["phase"],
            "config_sha256": config["_config_sha256"],
            "selected_arm": selected_name,
            "feature_names": FEATURE_NAMES,
            "parameters": selected_parameters,
            "state_dict": {
                name: value.detach().cpu() for name, value in selected.state_dict().items()
            },
            "source_hashes": frozen_hashes_before,
            "training": arms[selected_name]["training"],
            "selection": selection,
            "candidate_state_dicts": {
                arm: {
                    name: value.detach().cpu()
                    for name, value in arms[arm]["model"].state_dict().items()
                }
                for arm in ("constant", "contextual")
            },
        }, checkpoint_path)
        report.update({
            "confirmation": {
                "phase1d": confirmation_baseline,
                "selected": confirmation_selected,
                "comparison": confirmation_comparison,
                "paired_bootstrap": bootstrap,
                "variant_metrics": variant_metrics,
                "slice_metrics": slice_metrics,
            },
            "interventions": interventions,
            "causality": causality,
            "preservation": preservation,
            "runtime_overhead": {**path_overhead, **admission_overhead},
            "acceptance": {"criteria": criteria, "passed": all(criteria.values())},
        })
    else:
        torch.save({
            "schema": 1,
            "phase": config["phase"],
            "config_sha256": config["_config_sha256"],
            "selected_arm": None,
            "reason": "neither trained arm qualified on calibration",
            "selection": selection,
            "candidate_state_dicts": {
                arm: {
                    name: value.detach().cpu()
                    for name, value in arms[arm]["model"].state_dict().items()
                }
                for arm in ("constant", "contextual")
            },
            "training": {
                arm: arms[arm]["training"] for arm in ("constant", "contextual")
            },
        }, checkpoint_path)
        report["acceptance"] = {
            "criteria": {"calibration_selection_available": False}, "passed": False,
        }
    report["admission_checkpoint"] = str(checkpoint_path.resolve())
    report["admission_checkpoint_sha256"] = file_hash(checkpoint_path)
    torch.cuda.synchronize(device)
    accelerator_seconds = time.perf_counter() - run_started
    report["budget"] = {
        "accelerator_seconds": accelerator_seconds,
        "cap": float(config["budget"]["whole_run_accelerator_seconds"]),
        "within_cap": accelerator_seconds
        <= float(config["budget"]["whole_run_accelerator_seconds"]),
        "per_arm_limits": {
            name: {
                "updates": arms[name]["training"]["updates"],
                "target_exposures": arms[name]["training"][
                    "supervised_causal_target_exposures"
                ],
                "within_updates": arms[name]["training"]["updates"]
                <= int(config["training"]["max_updates"]),
                "within_exposures": arms[name]["training"][
                    "supervised_causal_target_exposures"
                ] <= int(config["training"]["max_target_exposures_per_arm"]),
            }
            for name in ("constant", "contextual")
        },
    }
    if not report["budget"]["within_cap"]:
        report["acceptance"]["criteria"]["accelerator_budget"] = False
        report["acceptance"]["passed"] = False
    else:
        report["acceptance"]["criteria"]["accelerator_budget"] = True

    frozen_states_after = {
        "backbone": _state_hash(backbone.state_dict()),
        "specialist": _state_hash(specialist.state_dict()),
        "readout": _state_hash(readout.state_dict()),
    }
    hashes_after = _hash_tree(source_roots)
    frozen_hashes_after = {name: file_hash(path) for name, path in frozen_paths.items()}
    if hashes_after != hashes_before:
        raise RuntimeError("Phase 1e changed a prior artifact")
    if frozen_hashes_after != frozen_hashes_before:
        raise RuntimeError("a frozen checkpoint changed")
    if frozen_states_after != frozen_states_before:
        raise RuntimeError("a frozen in-memory state changed")
    report.update({
        "source_hashes_before": hashes_before,
        "source_hashes_after": hashes_after,
        "source_artifacts_unchanged": True,
        "frozen_checkpoint_hashes_before": frozen_hashes_before,
        "frozen_checkpoint_hashes_after": frozen_hashes_after,
        "frozen_state_hashes_before": frozen_states_before,
        "frozen_state_hashes_after": frozen_states_after,
    })
    _write_json(report_path, report)
    (run_dir / "REPORT.md").write_text(_markdown(report), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    report = run(_config(args.config), args.run_dir, force=args.force)
    print(json.dumps({
        "selected": report["selection"]["selected"],
        "acceptance": report["acceptance"],
        "budget": report["budget"],
        "report": str((args.run_dir / "REPORT.md").resolve()),
    }, indent=2, sort_keys=True))

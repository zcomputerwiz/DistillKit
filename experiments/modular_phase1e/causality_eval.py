"""Evaluation-only causality repair audit for the frozen Phase 1d/1e path."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

from experiments.modular_phase1.config import file_hash
from experiments.modular_phase1.language import Document, Vocabulary
from experiments.modular_phase1.specialist import CausalStructureMachine, RecurrentSpecialist
from experiments.modular_phase1.training import _autocast
from experiments.modular_phase1c.models import (
    BIT_IDS,
    DirectLookupReadout,
    compose_bit_distribution,
    literal_representation,
)
from experiments.modular_phase1d.deployment import deployed_lookup_trace

from .data import FamilyDocument
from .models import (
    CAUSAL_POSITION_SCALE,
    ConstantAdmission,
    ContextualAdmission,
    apply_category_admission,
    build_contextual_features,
)
from .run import _config, _load_frozen, cache_split, evaluate_cache


OUTPUT_TOLERANCE = 1e-5
MAX_SEQUENCE = int(CAUSAL_POSITION_SCALE)


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load_members(path: Path) -> list[FamilyDocument]:
    members = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            members.append(FamilyDocument(
                family_id=record["phase1e_family_id"],
                split=record["phase1e_split"],
                variant=record["phase1e_variant"],
                document=Document.from_json(record),
            ))
    return members


def _load_candidates(
    path: Path, device: torch.device
) -> tuple[dict[str, torch.nn.Module], dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    result: dict[str, torch.nn.Module] = {
        "constant": ConstantAdmission(),
        "contextual": ContextualAdmission(),
    }
    for name, model in result.items():
        model.load_state_dict(checkpoint["candidate_state_dicts"][name])
        model.requires_grad_(False).eval().to(device)
    return result, {
        "selected_arm": checkpoint["selected_arm"],
        "trained_config_sha256": checkpoint["config_sha256"],
    }


def _pad_sequences(
    sequences: list[list[int]], device: torch.device, *, width: int | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    natural = max(len(sequence) for sequence in sequences)
    width = natural if width is None else width
    if width < natural or width > MAX_SEQUENCE:
        raise ValueError("invalid padded sequence width")
    ids = torch.full(
        (len(sequences), width), Vocabulary.PAD, dtype=torch.long, device=device
    )
    mask = torch.zeros((len(sequences), width), dtype=torch.bool, device=device)
    depths = torch.zeros((len(sequences), width), dtype=torch.long, device=device)
    machine = CausalStructureMachine()
    for row, sequence in enumerate(sequences):
        length = len(sequence)
        ids[row, :length] = torch.tensor(sequence, dtype=torch.long, device=device)
        mask[row, :length] = True
        depths[row, :length] = torch.tensor(
            machine.analyze(sequence).depths, dtype=torch.long, device=device
        )
    return ids, mask, depths


@torch.inference_mode()
def _forward_sequences(
    sequences: list[list[int]],
    backbone,
    specialist: RecurrentSpecialist,
    device: torch.device,
    bf16: bool,
    *,
    width: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    ids, mask, depths = _pad_sequences(sequences, device, width=width)
    with _autocast(device, bf16):
        backbone_logits = backbone(ids, attention_mask=mask)
        specialist_output = specialist(ids)
    backbone_log_probs = F.log_softmax(backbone_logits.float(), -1)
    features = build_contextual_features(
        ids,
        backbone_log_probs,
        specialist_output["event_logits"],
        specialist_output["validity_logits"],
        depths,
    )
    return backbone_log_probs, features


@torch.inference_mode()
def _outputs_at(
    sequences: list[list[int]],
    positions: list[int],
    backbone_log_probs: torch.Tensor,
    features: torch.Tensor,
    specialist: RecurrentSpecialist,
    readout: DirectLookupReadout,
    candidates: dict[str, torch.nn.Module],
    device: torch.device,
) -> tuple[list[torch.Tensor], dict[str, list[torch.Tensor]], list[dict]]:
    phase1d = []
    admitted = {name: [] for name in candidates}
    traces = []
    for row, (sequence, position) in enumerate(zip(sequences, positions)):
        trace = deployed_lookup_trace(specialist, sequence, device)
        active = bool(trace.active[position])
        base = backbone_log_probs[row, position]
        output = base
        if active:
            value = torch.tensor([trace.retrieved_values[position]], device=device)
            reader = F.log_softmax(readout(literal_representation(value)), -1)
            output = compose_bit_distribution(
                base.unsqueeze(0), reader, torch.ones(1, dtype=torch.bool, device=device)
            )[0]
        phase1d.append(output.cpu())
        feature = features[row, position].unsqueeze(0)
        active_tensor = torch.tensor([active], dtype=torch.bool, device=device)
        for name, model in candidates.items():
            corrected = apply_category_admission(
                output.unsqueeze(0), model(feature), active_tensor
            )[0]
            admitted[name].append(corrected.cpu())
        traces.append({
            "active": active,
            "binding": trace.binding_addresses[position],
            "value_address": trace.value_addresses[position],
            "retrieved_value": trace.retrieved_values[position],
        })
    return phase1d, admitted, traces


def _max_difference(left: list[torch.Tensor], right: list[torch.Tensor]) -> float:
    return max(float((a - b).abs().max()) for a, b in zip(left, right))


def _trace_mismatches(left: list[dict], right: list[dict]) -> int:
    return sum(a != b for a, b in zip(left, right))


@torch.inference_mode()
def evaluate_causal_invariance(
    members: list[FamilyDocument],
    backbone,
    specialist: RecurrentSpecialist,
    readout: DirectLookupReadout,
    candidates: dict[str, torch.nn.Module],
    device: torch.device,
    bf16: bool,
    *,
    max_cases: int,
) -> dict:
    selected = members[:max_cases]
    positions = [member.document.answer_position - 1 for member in selected]
    full = [list(member.document.token_ids) for member in selected]
    prefixes = [sequence[:position + 1] for sequence, position in zip(full, positions)]
    changed_suffixes = []
    for member, prefix in zip(selected, prefixes):
        opposite = BIT_IDS[1 - int(member.document.answer)]
        changed_suffixes.append(prefix + [opposite, Vocabulary.TO_ID["<nl>"], Vocabulary.TO_ID["<eos>"]])

    scenarios: dict[str, tuple[list[list[int]], int | None]] = {
        "truncated": (prefixes, None),
        "original_suffix": (full, None),
        "changed_suffix": (changed_suffixes, None),
        "different_batch_padding": (
            prefixes,
            min(MAX_SEQUENCE, max(len(item) for item in prefixes) + 97),
        ),
    }
    captured = {}
    for name, (sequences, width) in scenarios.items():
        backbone_probs, features = _forward_sequences(
            sequences, backbone, specialist, device, bf16, width=width
        )
        phase1d, admitted, traces = _outputs_at(
            sequences, positions, backbone_probs, features, specialist, readout,
            candidates, device,
        )
        captured[name] = {
            "features": [features[row, position].cpu() for row, position in enumerate(positions)],
            "phase1d": phase1d,
            "admitted": admitted,
            "traces": traces,
            "tensor_width": backbone_probs.shape[1],
        }

    reference = captured["truncated"]
    comparisons = {}
    comparison_scenarios = {
        "truncation": "original_suffix",
        "suffix_change": "changed_suffix",
        "batch_padding": "different_batch_padding",
    }
    for comparison_name, scenario_name in comparison_scenarios.items():
        other = captured[scenario_name]
        arm_differences = {
            "unchanged_phase1d": _max_difference(reference["phase1d"], other["phase1d"]),
            **{
                arm: _max_difference(reference["admitted"][arm], other["admitted"][arm])
                for arm in candidates
            },
        }
        feature_difference = _max_difference(reference["features"], other["features"])
        comparisons[comparison_name] = {
            "reference_tensor_width": reference["tensor_width"],
            "comparison_tensor_width": other["tensor_width"],
            "max_abs_feature_difference": feature_difference,
            "trace_mismatches": _trace_mismatches(reference["traces"], other["traces"]),
            "max_abs_final_log_probability_difference": arm_differences,
            "passed": (
                feature_difference <= OUTPUT_TOLERANCE
                and not _trace_mismatches(reference["traces"], other["traces"])
                and max(arm_differences.values()) <= OUTPUT_TOLERANCE
            ),
        }

    old_scale_differences = []
    reference_width = reference["tensor_width"]
    for scenario_name in comparison_scenarios.values():
        other_width = captured[scenario_name]["tensor_width"]
        for position in positions:
            left = position / max(1, reference_width - 1)
            right = position / max(1, other_width - 1)
            old_scale_differences.append(abs(left - right))
    return {
        "cases": len(selected),
        "variants": dict(Counter(member.variant for member in selected)),
        "fixed_position_scale": CAUSAL_POSITION_SCALE,
        "old_length_normalization_max_abs_difference": max(old_scale_differences),
        "repaired_position_feature_max_abs_difference": max(
            item["max_abs_feature_difference"] for item in comparisons.values()
        ),
        "tolerance": OUTPUT_TOLERANCE,
        "comparisons": comparisons,
        "all_passed": all(item["passed"] for item in comparisons.values()),
    }


def _format_timing(document) -> str:
    score = document.answer_position - 1
    token = document.token_ids[score]
    if token == Vocabulary.TO_ID["=>"]:
        return "immediately_after_marker"
    if Vocabulary.TOKENS[token] in Vocabulary.WHITESPACE:
        return "after_whitespace"
    raise AssertionError("lookup answer is not preceded by marker or whitespace")


def split_phase1d_errors(cache, device: torch.device) -> dict:
    result = {}
    for timing in ("immediately_after_marker", "after_whitespace"):
        subset = [item for item in cache if _format_timing(item) == timing]
        metrics, _ = evaluate_cache(subset, None, device)
        errors = Counter()
        for item in subset:
            score = item.answer_position - 1
            target = int(item.target_ids[score])
            predicted = int(item.phase1d_predictions[score])
            if predicted != target:
                errors[Vocabulary.TOKENS[predicted]] += 1
        result[timing] = {
            "examples": len(subset),
            "errors": sum(errors.values()),
            "error_predictions": dict(sorted(errors.items())),
            "answer_accuracy": metrics["answer_accuracy"],
            "answer_nll": metrics["answer_nll"],
            "conditional_lookup_accuracy": metrics["conditional_lookup_accuracy"],
            "mean_bit_category_probability": metrics[
                "mean_bit_category_probability_at_answers"
            ],
            "coverage": metrics["coverage"],
        }
    return result


def classify_greedy_completion(generated: list[int], expected_bit: int) -> str:
    expected = BIT_IDS[expected_bit]
    if generated and generated[0] in BIT_IDS:
        return "immediate_correct" if generated[0] == expected else "immediate_wrong"
    if generated and Vocabulary.TOKENS[generated[0]] in Vocabulary.WHITESPACE:
        if len(generated) > 1 and generated[1] in BIT_IDS:
            return "whitespace_then_correct" if generated[1] == expected else "whitespace_then_wrong"
        return "whitespace_without_bit"
    return "non_bit_non_whitespace"


@torch.inference_mode()
def _greedy_step(
    sequences: list[list[int]],
    backbone,
    specialist: RecurrentSpecialist,
    readout: DirectLookupReadout,
    device: torch.device,
    bf16: bool,
) -> tuple[list[int], list[dict]]:
    log_probs, features = _forward_sequences(
        sequences, backbone, specialist, device, bf16
    )
    positions = [len(sequence) - 1 for sequence in sequences]
    phase1d, _, traces = _outputs_at(
        sequences, positions, log_probs, features, specialist, readout, {}, device
    )
    return [int(output.argmax()) for output in phase1d], traces


@torch.inference_mode()
def greedy_phase1d_completion(
    members: list[FamilyDocument],
    backbone,
    specialist: RecurrentSpecialist,
    readout: DirectLookupReadout,
    device: torch.device,
    bf16: bool,
    *,
    max_examples: int,
) -> dict:
    selected = members[:max_examples]
    prefixes = []
    for member in selected:
        before_answer = member.document.token_ids[:member.document.answer_position]
        marker = max(
            index for index, token in enumerate(before_answer)
            if token == Vocabulary.TO_ID["=>"]
        )
        prefixes.append(list(before_answer[:marker + 1]))

    first, first_traces = _greedy_step(
        prefixes, backbone, specialist, readout, device, bf16
    )
    generated = [[token] for token in first]
    pending = [
        index for index, token in enumerate(first)
        if Vocabulary.TOKENS[token] in Vocabulary.WHITESPACE
    ]
    second_traces = []
    if pending:
        second_sequences = [prefixes[index] + [first[index]] for index in pending]
        second, second_traces = _greedy_step(
            second_sequences, backbone, specialist, readout, device, bf16
        )
        for index, token in zip(pending, second):
            generated[index].append(token)

    outcomes = Counter(
        classify_greedy_completion(tokens, int(member.document.answer))
        for tokens, member in zip(generated, selected)
    )
    correct = outcomes["immediate_correct"] + outcomes["whitespace_then_correct"]
    emitted_bits = sum(
        outcomes[name] for name in (
            "immediate_correct", "immediate_wrong",
            "whitespace_then_correct", "whitespace_then_wrong",
        )
    )
    correct_bits = correct
    return {
        "examples": len(selected),
        "maximum_new_tokens": 2,
        "optional_whitespace_tokens_allowed": 1,
        "uses_answer_position_for_activation": False,
        "valid_formatted_answer_accuracy": correct / len(selected),
        "bit_content_accuracy_when_emitted": correct_bits / max(1, emitted_bits),
        "bit_emission_rate": emitted_bits / len(selected),
        "outcomes": dict(sorted(outcomes.items())),
        "first_token_predictions": dict(sorted(Counter(
            Vocabulary.TOKENS[token] for token in first
        ).items())),
        "first_step_activation_rate": sum(item["active"] for item in first_traces)
        / len(first_traces),
        "first_step_retrieval_rate": sum(
            item["retrieved_value"] in (0, 1) for item in first_traces
        ) / len(first_traces),
        "second_step_cases": len(pending),
        "second_step_activation_rate": (
            sum(item["active"] for item in second_traces) / max(1, len(second_traces))
        ),
    }


def _markdown(report: dict) -> str:
    timing = report["phase1d_answer_timing"]
    greedy = report["greedy_completion"]
    lines = [
        "# Phase 1e causality repair audit",
        "",
        "Training remained paused. The original negative category-admission verdict is "
        "preserved.",
        "",
        "## Causality repair",
        "",
        f"The old length-normalized feature changed by as much as "
        f"{report['causal_invariance']['old_length_normalization_max_abs_difference']:.6f}. "
        f"With the fixed 512-token scale, the largest complete feature-vector change was "
        f"{report['causal_invariance']['repaired_position_feature_max_abs_difference']:.3e}. "
        f"All feature, deployment-state, and final-output checks passed: "
        f"{report['causal_invariance']['all_passed']}.",
        "The failed constant and contextual candidate weights were loaded unchanged. The "
        "contextual candidate was evaluated with the repaired feature semantics without "
        "retraining or selection.",
        "",
        "## Phase 1d teacher-forced answer position",
        "",
        "| Timing | Examples | Accuracy | NLL | Conditional bit accuracy | Bit mass | Errors |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in ("immediately_after_marker", "after_whitespace"):
        item = timing[name]
        lines.append(
            f"| {name} | {item['examples']} | {item['answer_accuracy']:.4f} | "
            f"{item['answer_nll']:.6f} | {item['conditional_lookup_accuracy']:.4f} | "
            f"{item['mean_bit_category_probability']:.4f} | {item['errors']} |"
        )
    lines.extend([
        "",
        "## Bounded ordinary greedy completion",
        "",
        f"Across {greedy['examples']} lookup examples, Phase 1d produced the correct bit "
        f"immediately or after one generated whitespace token in "
        f"{greedy['valid_formatted_answer_accuracy']:.4%}. Bit emission was "
        f"{greedy['bit_emission_rate']:.4%}, and content accuracy conditional on emitting "
        f"a bit was {greedy['bit_content_accuracy_when_emitted']:.4%}.",
        "",
        f"Conclusion: {report['conclusion']}",
        "",
        "This audit does not establish a positive admission result and performs no model "
        "selection, retraining, XOR evaluation, or efficiency claim.",
        "",
        "Reproduce:",
        "",
        "`python -m experiments.modular_phase1e.causality_eval --config "
        "experiments/modular_phase1e/configs/diagnostic.json --run-dir "
        "scratch/modular-phase1e --force`",
        "",
    ])
    return "\n".join(lines)


def run(
    config: dict,
    run_dir: Path,
    *,
    force: bool = False,
    max_invariance_cases: int = 96,
    max_greedy_examples: int = 768,
) -> dict:
    output_path = run_dir / "causality_report.json"
    if output_path.exists() and not force:
        return json.loads(output_path.read_text(encoding="utf-8"))
    original_report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    if original_report["acceptance"]["passed"] or original_report["selection"]["selected"]:
        raise RuntimeError("the preserved Phase 1e result is not the expected negative verdict")

    frozen_paths = {
        "backbone": Path(config["base_checkpoint"]),
        "specialist": Path(config["specialist_checkpoint"]),
        "readout": Path(config["readout_checkpoint"]),
        "admission_candidates": run_dir / "admission" / "admission.pt",
        "original_phase1e_report": run_dir / "report.json",
    }
    hashes_before = {name: file_hash(path) for name, path in frozen_paths.items()}
    device = torch.device(config["runtime"]["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the evaluation requires the configured accelerator")
    started = time.perf_counter()
    members = _load_members(run_dir / "data" / "calibration.jsonl.gz")
    backbone, specialist, readout, phase1_config = _load_frozen(config, device)
    candidates, candidate_metadata = _load_candidates(
        frozen_paths["admission_candidates"], device
    )

    invariance = evaluate_causal_invariance(
        members, backbone, specialist, readout, candidates, device,
        bool(config["runtime"]["bf16"]), max_cases=min(max_invariance_cases, len(members)),
    )
    cache, cache_cost = cache_split(
        members, backbone, specialist, readout,
        batch_size=int(phase1_config["evaluation"]["batch_sizes"]["256"]),
        device=device, bf16=bool(config["runtime"]["bf16"]),
    )
    timing = split_phase1d_errors(cache, device)
    greedy = greedy_phase1d_completion(
        members, backbone, specialist, readout, device,
        bool(config["runtime"]["bf16"]),
        max_examples=min(max_greedy_examples, len(members)),
    )
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    hashes_after = {name: file_hash(path) for name, path in frozen_paths.items()}
    if hashes_before != hashes_after:
        raise RuntimeError("evaluation changed a frozen artifact")

    if greedy["valid_formatted_answer_accuracy"] >= 0.99:
        conclusion = (
            "Phase 1d already answers lookup prompts reliably under the accepted optional-"
            "whitespace format; the failed admission arms are not needed for this behavior."
        )
    elif greedy["bit_content_accuracy_when_emitted"] >= 0.99:
        conclusion = (
            "Retrieved-bit selection remains reliable, but ordinary generation still fails "
            "to emit a valid bit often enough; any remaining issue is category admission."
        )
    else:
        conclusion = (
            "Ordinary generation still has both formatting/category and bit-selection "
            "failures; the remaining cause is not isolated to admission."
        )
    report = {
        "schema": 1,
        "phase": "modular_phase1e_causality_repair_evaluation",
        "training": {"performed": False, "optimizer_updates": 0},
        "candidate_weights_reused_without_retraining": True,
        "candidate_checkpoint": {
            **candidate_metadata,
            "repaired_config_sha256": config["_config_sha256"],
            "config_hash_changed_only_for_causal_feature_repair": (
                candidate_metadata["trained_config_sha256"] != config["_config_sha256"]
            ),
        },
        "preserved_verdicts": {
            **original_report["preserved_verdicts"],
            "phase1e_admission": "failed on calibration; unchanged by this audit",
        },
        "evaluation_split": "existing_phase1e_calibration",
        "documents": len(members),
        "cache_cost": cache_cost,
        "causal_invariance": invariance,
        "phase1d_answer_timing": timing,
        "greedy_completion": greedy,
        "conclusion": conclusion,
        "accelerator_seconds": elapsed,
        "frozen_hashes_before": hashes_before,
        "frozen_hashes_after": hashes_after,
        "frozen_hashes_unchanged": hashes_before == hashes_after,
        "config_sha256": config["_config_sha256"],
    }
    _write_json(output_path, report)
    (run_dir / "CAUSALITY_REPORT.md").write_text(_markdown(report), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-invariance-cases", type=int, default=96)
    parser.add_argument("--max-greedy-examples", type=int, default=768)
    args = parser.parse_args()
    report = run(
        _config(args.config), args.run_dir, force=args.force,
        max_invariance_cases=args.max_invariance_cases,
        max_greedy_examples=args.max_greedy_examples,
    )
    print(json.dumps({
        "causal_invariance_passed": report["causal_invariance"]["all_passed"],
        "greedy_valid_answer_accuracy": report["greedy_completion"][
            "valid_formatted_answer_accuracy"
        ],
        "training_performed": report["training"]["performed"],
        "report": str((args.run_dir / "causality_report.json").resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()

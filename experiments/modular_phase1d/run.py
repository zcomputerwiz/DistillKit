"""Qualify a corrected specialist and hand it to the frozen Phase 1c output path."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from experiments.modular_phase1.config import file_hash, load_config
from experiments.modular_phase1.evaluation import _load_model
from experiments.modular_phase1.language import Vocabulary
from experiments.modular_phase1.specialist import CausalStructureMachine, RecurrentSpecialist
from experiments.modular_phase1.training import _autocast, _state_hash
from experiments.modular_phase1c.models import (
    BIT_IDS,
    DirectLookupReadout,
    compose_bit_distribution,
    deterministic_copy_log_probs,
    literal_representation,
)
from experiments.modular_phase1c.run import (
    LookupExample,
    _collate,
    cache_answer_logits,
    prepare_examples,
)

from .deployment import deployed_lookup_trace
from .training import load_corrected_specialist, train_corrected_specialist


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
    return value


def _hash_tree(roots: list[Path]) -> dict[str, str]:
    return {
        str(path.resolve()): file_hash(path)
        for root in roots
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix in (".pt", ".json", ".md", ".gz")
    }


def _tensor_state_hash(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _load_readout(path: Path, device: torch.device) -> tuple[DirectLookupReadout, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = DirectLookupReadout()
    model.load_state_dict(checkpoint["learned_state_dict"])
    model.requires_grad_(False).eval().to(device)
    return model, checkpoint


def _deployed_rows(
    specialist: RecurrentSpecialist,
    examples: list[LookupExample],
    device: torch.device,
    *,
    overrides: list[dict[int, int] | None] | None = None,
) -> dict:
    activated = []
    active = []
    values = []
    bindings = []
    sources = []
    for index, example in enumerate(examples):
        trace = deployed_lookup_trace(
            specialist,
            list(example.token_ids),
            device,
            pointer_override=None if overrides is None else overrides[index],
        )
        score = example.answer_position - 1
        activated.append(trace.activated[score])
        active.append(trace.active[score])
        values.append(trace.retrieved_values[score])
        bindings.append(trace.binding_addresses[score])
        sources.append(trace.value_addresses[score])
    return {
        "activated": torch.tensor(activated, dtype=torch.bool, device=device),
        "active": torch.tensor(active, dtype=torch.bool, device=device),
        "values": torch.tensor(values, dtype=torch.long, device=device),
        "bindings": bindings,
        "sources": sources,
    }


@torch.inference_mode()
def evaluate_lookup(
    logits: torch.Tensor,
    examples: list[LookupExample],
    specialist: RecurrentSpecialist,
    readout: DirectLookupReadout,
    device: torch.device,
) -> tuple[dict, dict]:
    base = torch.log_softmax(logits.to(device), -1)
    rows = _deployed_rows(specialist, examples, device)
    targets = torch.tensor([example.target for example in examples], device=device)
    safe_values = rows["values"].clamp_min(0)
    representation = literal_representation(safe_values)
    learned = compose_bit_distribution(
        base, F.log_softmax(readout(representation), -1), rows["active"]
    )
    copied = compose_bit_distribution(
        base, deterministic_copy_log_probs(safe_values), rows["active"]
    )
    conditions = {"baseline": base, "deterministic_copy": copied, "learned_readout": learned}
    target_ids = torch.tensor([BIT_IDS[int(value)] for value in targets.tolist()], device=device)
    metrics = {}
    for name, output in conditions.items():
        conditional = F.log_softmax(output[:, list(BIT_IDS)], -1)
        metrics[name] = {
            "eligible_queries": len(examples),
            "conditional_accuracy_all_eligible": float(
                (conditional.argmax(-1) == targets).float().mean()
            ),
            "conditional_nll_all_eligible": float(F.nll_loss(conditional, targets)),
            "full_vocabulary_answer_accuracy": float(
                (output.argmax(-1) == target_ids).float().mean()
            ),
            "answer_nll": float(F.nll_loss(output, target_ids)),
            "mean_bit_category_probability": float(
                output.exp()[:, list(BIT_IDS)].sum(-1).mean()
            ),
        }
    retrieval_correct = rows["active"] & (rows["values"] == targets)
    coverage = {
        "eligible_queries": len(examples),
        "activated": int(rows["activated"].sum()),
        "retrieval_succeeded": int(rows["active"].sum()),
        "retrieval_correct": int(retrieval_correct.sum()),
        "missed_activations": int((~rows["activated"]).sum()),
        "failed_retrievals_after_activation": int(
            (rows["activated"] & ~rows["active"]).sum()
        ),
        "activation_coverage": float(rows["activated"].float().mean()),
        "retrieval_coverage": float(rows["active"].float().mean()),
        "correct_retrieval_rate_all_eligible": float(retrieval_correct.float().mean()),
    }
    return metrics, coverage


def _programmed_interventions(examples: list[LookupExample]) -> tuple[list[int], list[dict]]:
    indices = []
    records = []
    machine = CausalStructureMachine()
    for index, example in enumerate(examples):
        if example.flipped:
            continue
        trace = machine.analyze(example.token_ids)
        uses = [position for position, pointer in enumerate(trace.pointers) if pointer >= 0]
        if len(uses) != 1:
            continue
        use = uses[0]
        original = trace.pointers[use]
        original_source = next(
            (position for position in range(original + 1, len(example.token_ids))
             if example.token_ids[position] in BIT_IDS), -1
        )
        if original_source < 0:
            continue
        original_value = int(example.token_ids[original_source] == BIT_IDS[1])
        candidates = []
        for declaration in trace.declaration_positions:
            if declaration == original:
                continue
            source = next(
                (position for position in range(declaration + 1, len(example.token_ids))
                 if example.token_ids[position] in BIT_IDS), -1
            )
            if source < 0:
                continue
            value = int(example.token_ids[source] == BIT_IDS[1])
            if value != original_value:
                candidates.append((abs(use - declaration), declaration, source, value))
        if not candidates:
            continue
        _, alternate, alternate_source, alternate_value = min(candidates)
        indices.append(index)
        records.append({
            "use_position": use,
            "original_pointer": original,
            "alternate_pointer": alternate,
            "original_value": original_value,
            "alternate_value": alternate_value,
        })
    return indices, records


def _summary(values: torch.Tensor) -> dict:
    members = values.detach().cpu().tolist()
    return {
        "count": len(members),
        "mean": sum(members) / len(members),
        "median": statistics.median(members),
        "minimum": min(members),
        "maximum": max(members),
    }


@torch.inference_mode()
def evaluate_interventions(
    logits: torch.Tensor,
    examples: list[LookupExample],
    specialist: RecurrentSpecialist,
    readout: DirectLookupReadout,
    records: list[dict],
    device: torch.device,
) -> dict:
    base = torch.log_softmax(logits.to(device), -1)
    overrides = [{record["use_position"]: record["alternate_pointer"]} for record in records]
    before = _deployed_rows(specialist, examples, device)
    after = _deployed_rows(specialist, examples, device, overrides=overrides)
    original = torch.tensor([record["original_value"] for record in records], device=device)
    alternate = torch.tensor([record["alternate_value"] for record in records], device=device)
    if not torch.all(original != alternate):
        raise AssertionError("interventions must change the source literal")

    def outputs(rows, copy: bool):
        values = rows["values"].clamp_min(0)
        reader = (
            deterministic_copy_log_probs(values)
            if copy else F.log_softmax(readout(literal_representation(values)), -1)
        )
        return compose_bit_distribution(base, reader, rows["active"])

    result = {}
    for name, copy in (("deterministic_copy", True), ("learned_readout", False)):
        first = F.log_softmax(outputs(before, copy)[:, list(BIT_IDS)], -1)
        second = F.log_softmax(outputs(after, copy)[:, list(BIT_IDS)], -1)
        row = torch.arange(len(records), device=device)
        probability_delta = second.exp()[row, alternate] - first.exp()[row, alternate]
        first_prediction = first.argmax(-1)
        second_prediction = second.argmax(-1)
        result[name] = {
            "eligible_cases": len(records),
            "original_activation_coverage": float(before["active"].float().mean()),
            "intervened_activation_coverage": float(after["active"].float().mean()),
            "retrieved_literal_changed": int(
                ((before["values"] == original) & (after["values"] == alternate)).sum()
            ),
            "conditional_accuracy_intervened_all_eligible": float(
                (second_prediction == alternate).float().mean()
            ),
            "prediction_changed_toward_new_required": float(
                ((first_prediction == original) & (second_prediction == alternate))
                .float().mean()
            ),
            "signed_new_required_probability_delta": _summary(probability_delta),
        }
    return result


def verify_causality(
    specialist: RecurrentSpecialist,
    examples: list[LookupExample],
    device: torch.device,
) -> dict:
    comparisons = 0
    for example_index, example in enumerate(examples):
        ids = list(example.token_ids)
        full = deployed_lookup_trace(specialist, ids, device)
        lengths = range(1, len(ids) + 1) if example_index < 16 else (example.answer_position,)
        for length in lengths:
            prefix = deployed_lookup_trace(specialist, ids[:length], device)
            for field in (
                "predicted_events", "programmed_pointers", "activated", "active",
                "binding_addresses", "value_addresses", "retrieved_values",
            ):
                if getattr(prefix, field) != getattr(full, field)[:length]:
                    raise AssertionError(f"unseen suffix changed deployed {field}")
            comparisons += 1
    return {
        "documents": len(examples),
        "prefix_comparisons": comparisons,
        "all_addresses_activation_values_and_predictions_invariant": True,
        "answer_annotations_used_for_activation": False,
    }


def benchmark_path(
    backbone,
    specialist: RecurrentSpecialist,
    readout: DirectLookupReadout,
    examples: list[LookupExample],
    device: torch.device,
    bf16: bool,
) -> dict:
    members = examples[:64]
    ids, mask = _collate(members, device)
    for _ in range(10):
        with torch.inference_mode(), _autocast(device, bf16):
            backbone(ids, attention_mask=mask)
        specialist(ids)
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    for _ in range(30):
        with torch.inference_mode(), _autocast(device, bf16):
            backbone(ids, attention_mask=mask)
    torch.cuda.synchronize(device)
    backbone_us = (time.perf_counter() - start) * 1e6 / (30 * len(members))
    start = time.perf_counter()
    for _ in range(100):
        specialist(ids)
    torch.cuda.synchronize(device)
    specialist_us = (time.perf_counter() - start) * 1e6 / (100 * len(members))
    values = torch.tensor([member.target for member in members], device=device)
    representation = literal_representation(values)
    base = torch.log_softmax(torch.randn(len(members), len(Vocabulary.TOKENS), device=device), -1)
    active = torch.ones(len(members), dtype=torch.bool, device=device)
    start = time.perf_counter()
    for _ in range(500):
        reader = F.log_softmax(readout(representation), -1)
        compose_bit_distribution(base, reader, active)
    torch.cuda.synchronize(device)
    readout_us = (time.perf_counter() - start) * 1e6 / (500 * len(members))
    return {
        "batch_documents": len(members),
        "warmed": True,
        "cuda_synchronized": True,
        "backbone_microseconds_per_document": backbone_us,
        "specialist_recurrent_microseconds_per_document": specialist_us,
        "readout_and_composition_microseconds_per_document": readout_us,
    }


def _markdown(report: dict) -> str:
    qualification = report["specialist"]["qualification"]
    event = qualification["events"]
    heldout = report.get("heldout_metrics", {})
    lines = [
        "# Phase 1d learned-specialist handoff",
        "",
        f"**Qualification: {'PASS' if report['qualification_passed'] else 'FAIL'}.** "
        "The original Phase 1 failed-composition verdict remains unchanged.",
        "",
        f"Binding accuracy: {qualification['binding_accuracy']:.6f}; structural-event "
        f"accuracy excluding pad/whitespace: "
        f"{event['structural_event_accuracy_excluding_pad_and_whitespace']:.6f}; macro-F1: "
        f"{event['structural_macro_f1_excluding_pad_and_whitespace']:.6f}.",
        "",
        "| Event | Support | Old FP/FN | New FP/FN | Old F1 | New F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in ("use", "invalid", "output", "declaration", "close"):
        item = report["error_class_comparison"][name]
        old = item["before"]
        new = item["after"]
        lines.append(
            f"| {name} | {new['support']} | {old['false_positive']}/"
            f"{old['false_negative']} | {new['false_positive']}/"
            f"{new['false_negative']} | {old['f1']:.6f} | {new['f1']:.6f} |"
        )
    weakest = sorted(
        qualification["validation_slices"].items(),
        key=lambda item: item[1][
            "structural_event_accuracy_excluding_pad_and_whitespace"
        ],
    )[:3]
    lines.extend((
        "",
        "Weakest held-out structural slices: " + "; ".join(
            f"{name}={value['structural_event_accuracy_excluding_pad_and_whitespace']:.6f}"
            for name, value in weakest
        ) + ".",
        "",
        "Learned operations used by deployment: query/use/semicolon/answer-marker event "
        "classification. Programmed specialist operations: causal scope table and binding "
        "pointer. Deterministic prefix processing carries activation across raw whitespace "
        "and scans the addressed declaration span for its observed literal. No reference "
        "interpreter value or expected answer enters the deployed path.",
    ))
    if not report["qualification_passed"]:
        lines.extend(("", "The downstream handoff was not run because the strict gate failed."))
        return "\n".join(lines) + "\n"
    learned = heldout["learned_readout"]
    coverage = report["coverage"]["heldout"]
    intervention = report["pointer_interventions"]["learned_readout"]
    lines.extend((
        "", "## End-to-end frozen path", "",
        f"Held-out conditional accuracy/NLL over all eligible queries: "
        f"{learned['conditional_accuracy_all_eligible']:.6f}/"
        f"{learned['conditional_nll_all_eligible']:.6f}; full-vocabulary accuracy: "
        f"{learned['full_vocabulary_answer_accuracy']:.6f}.",
        "",
        f"Activation/retrieval coverage: {coverage['activation_coverage']:.6f}/"
        f"{coverage['retrieval_coverage']:.6f}; missed activations: "
        f"{coverage['missed_activations']}; failed retrievals after activation: "
        f"{coverage['failed_retrievals_after_activation']}.",
        "",
        f"Opposite-value interventions: {intervention['eligible_cases']}; intervened "
        f"accuracy: {intervention['conditional_accuracy_intervened_all_eligible']:.6f}; "
        f"changed toward the required value: "
        f"{intervention['prediction_changed_toward_new_required']:.6f}.",
        "",
        f"Specialist parameters/tokens/steps: "
        f"{report['specialist']['parameters']['learned_parameters']}/"
        f"{report['specialist']['training']['tokens']}/"
        f"{report['specialist']['training']['steps']}; training accelerator time: "
        f"{report['specialist']['training']['accelerator_seconds']:.3f}s. Warmed synchronized "
        f"timing per document: backbone {report['overhead']['backbone_microseconds_per_document']:.3f} us, "
        f"specialist recurrent path {report['overhead']['specialist_recurrent_microseconds_per_document']:.3f} us, "
        f"readout/composition {report['overhead']['readout_and_composition_microseconds_per_document']:.3f} us.",
        "",
        f"Specialist SHA-256: `{report['specialist']['checkpoint_sha256']}`. Frozen "
        f"backbone/readout SHA-256: `{report['base_checkpoint_sha256_after']}` / "
        f"`{report['readout_checkpoint_sha256_after']}`; all pre-existing source hashes "
        "are unchanged.",
        "", "Passing this phase establishes only learned-specialist delivery through the "
        "already-qualified output-facing lookup path. It does not establish XOR, "
        "backbone-layer use, efficiency, or change the Phase 1 verdict.",
        "", "Reproduce the full bounded run:", "",
        "`python -m experiments.modular_phase1d --config "
        "experiments/modular_phase1d/configs/diagnostic.json --run-dir "
        "scratch/modular-phase1d --force`",
        "", "Repeat evaluation without retraining:", "",
        "`python -m experiments.modular_phase1d --config "
        "experiments/modular_phase1d/configs/diagnostic.json --run-dir "
        "scratch/modular-phase1d --force --reuse-specialist`",
    ))
    return "\n".join(lines) + "\n"


def run(
    config: dict,
    run_dir: Path,
    *,
    force: bool = False,
    reuse_specialist: bool = False,
) -> dict:
    report_path = run_dir / "report.json"
    if report_path.exists() and not force:
        return json.loads(report_path.read_text(encoding="utf-8"))
    source_roots = [
        Path(config["source_phase1_run_dir"]),
        Path(config["source_phase1b_run_dir"]),
        Path(config["source_phase1c_run_dir"]),
    ]
    source_hashes_before = _hash_tree(source_roots)
    base_path = Path(config["base_checkpoint"])
    readout_path = Path(config["readout_checkpoint"])
    base_hash_before = file_hash(base_path)
    readout_hash_before = file_hash(readout_path)
    device = torch.device(config["runtime"]["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the bounded learned-specialist run requires CUDA")
    torch.cuda.synchronize(device)
    whole_started = time.perf_counter()
    if reuse_specialist:
        specialist, specialist_manifest = load_corrected_specialist(
            config, run_dir, device
        )
    else:
        specialist, specialist_manifest = train_corrected_specialist(
            config, source_roots[0], run_dir, device
        )
    report = {
        "schema": 1,
        "phase": config["phase"],
        "preserved_phase1_verdict": "failed composition",
        "config_sha256": config["_config_sha256"],
        "specialist": specialist_manifest,
        "qualification_passed": specialist_manifest["gate"]["passed"],
        "base_checkpoint_sha256_before": base_hash_before,
        "readout_checkpoint_sha256_before": readout_hash_before,
    }
    original_audit = json.loads(
        (source_roots[0] / "audit.json").read_text(encoding="utf-8")
    )["specialist_gate"]["reconstructed_event_metrics"]["classes"]
    corrected_classes = specialist_manifest["qualification"]["events"]["classes"]
    report["error_class_comparison"] = {
        name: {"before": original_audit[name], "after": corrected_classes[name]}
        for name in ("use", "invalid", "output", "declaration", "close")
    }
    if specialist_manifest["gate"]["passed"]:
        examples, data = prepare_examples(source_roots[0])
        data.pop("base_documents")
        phase1_config = load_config(config["source_phase1_config"])
        checkpoint = torch.load(base_path, map_location="cpu", weights_only=True)
        backbone = _load_model(phase1_config, source_roots[0], checkpoint, device)
        backbone.requires_grad_(False).eval()
        readout, readout_checkpoint = _load_readout(readout_path, device)
        readout_state_before = _tensor_state_hash(readout.state_dict())
        cached = {}
        for split in ("heldout", "renamed", "whitespace"):
            cached[split], _ = cache_answer_logits(
                backbone, examples[split],
                batch_size=int(phase1_config["evaluation"]["batch_sizes"]["256"]),
                device=device, bf16=bool(config["runtime"]["bf16"]),
            )
        metrics = {}
        coverage = {}
        for split in ("heldout", "renamed", "whitespace"):
            metrics[split], coverage[split] = evaluate_lookup(
                cached[split], examples[split], specialist, readout, device
            )
        indices, intervention_records = _programmed_interventions(examples["heldout"])
        intervention_examples = [examples["heldout"][index] for index in indices]
        interventions = evaluate_interventions(
            cached["heldout"][indices], intervention_examples, specialist, readout,
            intervention_records, device,
        )
        causality = verify_causality(specialist, examples["heldout"], device)
        overhead = benchmark_path(
            backbone, specialist, readout, examples["heldout"], device,
            bool(config["runtime"]["bf16"]),
        )
        readout_state_after = _tensor_state_hash(readout.state_dict())
        if readout_state_after != readout_state_before:
            raise RuntimeError("frozen Phase 1c readout state changed")
        tolerance = float(config["acceptance"]["probability_preservation_tolerance"])
        heldout_learned = metrics["heldout"]["learned_readout"]
        heldout_copy = metrics["heldout"]["deterministic_copy"]
        acceptance = {
            "heldout_end_to_end_conditional_accuracy": (
                heldout_learned["conditional_accuracy_all_eligible"]
                >= float(config["acceptance"]["end_to_end_conditional_accuracy"])
            ),
            "heldout_coverage": (
                coverage["heldout"]["retrieval_coverage"]
                >= float(config["acceptance"]["coverage"])
            ),
            "renaming_accuracy": (
                metrics["renamed"]["learned_readout"]["conditional_accuracy_all_eligible"]
                >= float(config["acceptance"]["variant_accuracy"])
            ),
            "whitespace_accuracy": (
                metrics["whitespace"]["learned_readout"]["conditional_accuracy_all_eligible"]
                >= float(config["acceptance"]["variant_accuracy"])
            ),
            "intervention_accuracy": (
                interventions["learned_readout"][
                    "conditional_accuracy_intervened_all_eligible"
                ] >= float(config["acceptance"]["intervention_accuracy"])
            ),
            "intervention_change": (
                interventions["learned_readout"]["prediction_changed_toward_new_required"]
                >= float(config["acceptance"]["intervention_accuracy"])
            ),
            "category_probability_preserved": abs(
                heldout_learned["mean_bit_category_probability"]
                - metrics["heldout"]["baseline"]["mean_bit_category_probability"]
            ) <= tolerance,
            "deterministic_copy_is_upper_reference": (
                heldout_copy["conditional_accuracy_all_eligible"]
                >= heldout_learned["conditional_accuracy_all_eligible"]
            ),
            "readout_state_unchanged": readout_state_after == readout_state_before,
        }
        report.update({
            "data": data,
            "heldout_metrics": metrics["heldout"],
            "variant_metrics": {key: metrics[key] for key in ("renamed", "whitespace")},
            "coverage": coverage,
            "pointer_interventions": interventions,
            "causality": causality,
            "overhead": overhead,
            "readout_source_base_checkpoint_sha256": readout_checkpoint[
                "base_checkpoint_sha256"
            ],
            "readout_state_sha256_before": readout_state_before,
            "readout_state_sha256_after": readout_state_after,
            "acceptance": {"criteria": acceptance, "passed": all(acceptance.values())},
            "backbone_state_sha256": _state_hash(backbone.state_dict()),
        })
    torch.cuda.synchronize(device)
    invocation_seconds = time.perf_counter() - whole_started
    combined_seconds = invocation_seconds
    if reuse_specialist:
        combined_seconds += float(specialist_manifest["training"]["accelerator_seconds"])
    report["budget"] = {
        "execution_mode": "reuse_qualified_specialist" if reuse_specialist else "train_then_evaluate",
        "current_invocation_accelerator_seconds": invocation_seconds,
        "specialist_training_accelerator_seconds": float(
            specialist_manifest["training"]["accelerator_seconds"]
        ),
        "whole_run_accelerator_seconds": combined_seconds,
        "cap": float(config["budget"]["whole_run_accelerator_seconds"]),
        "within_cap": (
            combined_seconds <= float(config["budget"]["whole_run_accelerator_seconds"])
        ),
    }
    source_hashes_after = _hash_tree(source_roots)
    base_hash_after = file_hash(base_path)
    readout_hash_after = file_hash(readout_path)
    if source_hashes_after != source_hashes_before:
        raise RuntimeError("a pre-existing Phase 1/1b/1c artifact changed")
    if base_hash_after != base_hash_before or readout_hash_after != readout_hash_before:
        raise RuntimeError("frozen backbone or readout checkpoint changed")
    report.update({
        "base_checkpoint_sha256_after": base_hash_after,
        "readout_checkpoint_sha256_after": readout_hash_after,
        "source_hashes_before": source_hashes_before,
        "source_hashes_after": source_hashes_after,
        "source_artifacts_unchanged": True,
    })
    if "acceptance" not in report:
        report["acceptance"] = {"criteria": {"qualification": False}, "passed": False}
    elif not report["budget"]["within_cap"]:
        report["acceptance"]["criteria"]["whole_run_budget"] = False
        report["acceptance"]["passed"] = False
    else:
        report["acceptance"]["criteria"]["whole_run_budget"] = True
    _write_json(report_path, report)
    (run_dir / "REPORT.md").write_text(_markdown(report), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--reuse-specialist", action="store_true")
    args = parser.parse_args()
    report = run(
        _config(args.config), args.run_dir,
        force=args.force, reuse_specialist=args.reuse_specialist,
    )
    print(json.dumps({
        "qualification": report["qualification_passed"],
        "acceptance": report["acceptance"],
        "budget": report["budget"],
        "report": str((args.run_dir / "REPORT.md").resolve()),
    }, indent=2, sort_keys=True))

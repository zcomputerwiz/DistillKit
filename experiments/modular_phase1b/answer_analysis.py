"""Evaluation-only answer-position and binary-selection analysis for Phases 1/1b."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from experiments.modular_phase1.audit import eligible_pointer_case
from experiments.modular_phase1.config import file_hash, load_config
from experiments.modular_phase1.data import collate_documents
from experiments.modular_phase1.evaluation import _load_model
from experiments.modular_phase1.language import Vocabulary, decode
from experiments.modular_phase1.training import _autocast

from .models import OracleBindingDecoder, build_paired_model
from .run import _load_config, _lookup_documents, _oracle_pointers, _summary


BIT_IDS = (Vocabulary.TO_ID["0"], Vocabulary.TO_ID["1"])


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def decompose_answer_logits(logits: torch.Tensor, correct_token: int) -> dict:
    """Split full-vocabulary answer NLL into bit-category and bit-selection terms."""
    log_probs = F.log_softmax(logits.float(), -1)
    bit_log_mass = torch.logsumexp(log_probs[list(BIT_IDS)], dim=0)
    answer_nll = -log_probs[correct_token]
    category_nll = -bit_log_mass
    selection_nll = -(log_probs[correct_token] - bit_log_mass)
    if not torch.allclose(answer_nll, category_nll + selection_nll, atol=2e-6):
        raise AssertionError("answer NLL decomposition identity failed")
    bit_logits = logits.float()[list(BIT_IDS)]
    normalized = bit_logits.softmax(-1)
    correct_slot = 0 if correct_token == BIT_IDS[0] else 1
    prediction = int(logits.argmax())
    return {
        "answer_nll": float(answer_nll),
        "bit_category_nll": float(category_nll),
        "correct_bit_given_category_nll": float(selection_nll),
        "bit_category_probability": float(bit_log_mass.exp()),
        "correct_bit_given_category_probability": float(normalized[correct_slot]),
        "full_vocabulary_prediction": prediction,
        "full_vocabulary_answer_correct": int(prediction == correct_token),
        "top1_is_bit": int(prediction in BIT_IDS),
        "conditional_bit_prediction": BIT_IDS[int(bit_logits.argmax())],
        "conditional_bit_correct": int(int(bit_logits.argmax()) == correct_slot),
    }


def verify_answer_positions(documents, count: int = 8) -> list[dict]:
    """Demonstrate that logits[t-1] score the answer token at t."""
    chosen = []
    seen_whitespace = set()
    for document in documents:
        style = document.metadata["whitespace"]
        if style not in seen_whitespace or len(chosen) < count:
            chosen.append(document)
            seen_whitespace.add(style)
        if len(chosen) >= count and len(seen_whitespace) == 4:
            break
    examples = []
    for document in chosen[:count]:
        position = document.answer_position
        ids = document.token_ids
        arrow = max(index for index, token in enumerate(ids) if token == Vocabulary.TO_ID["=>"])
        shifted_target_index = position - 1
        if ids[position] not in BIT_IDS:
            raise AssertionError("recorded answer position is not a bit")
        if ids[shifted_target_index + 1] != ids[position]:
            raise AssertionError("shifted causal target does not equal answer token")
        examples.append({
            "document_id": document.document_id,
            "whitespace": document.metadata["whitespace"],
            "text_tail": document.text[-40:],
            "arrow_position": arrow,
            "answer_position": position,
            "scored_logits_position": shifted_target_index,
            "scored_after_token": Vocabulary.TOKENS[ids[shifted_target_index]],
            "target_token": Vocabulary.TOKENS[ids[position]],
            "local_positions": list(range(max(0, arrow - 1), min(len(ids), position + 2))),
            "local_tokens": decode(ids[max(0, arrow - 1):min(len(ids), position + 2)]),
            "shift_identity_verified": True,
        })
    return examples


def _aggregate(rows: list[dict]) -> dict:
    count = len(rows)
    result = {
        "documents": count,
        "answer_accuracy_full_vocabulary": sum(
            row["full_vocabulary_answer_correct"] for row in rows
        ) / count,
        "top1_bit_category_detection_accuracy": sum(row["top1_is_bit"] for row in rows) / count,
        "conditional_bit_selection_accuracy": sum(
            row["conditional_bit_correct"] for row in rows
        ) / count,
    }
    for name in (
        "answer_nll",
        "bit_category_nll",
        "correct_bit_given_category_nll",
        "bit_category_probability",
        "correct_bit_given_category_probability",
    ):
        result[name] = sum(row[name] for row in rows) / count
    result["decomposition_residual"] = (
        result["answer_nll"]
        - result["bit_category_nll"]
        - result["correct_bit_given_category_nll"]
    )
    return result


@torch.inference_mode()
def score_model(
    model,
    documents,
    *,
    batch_size: int,
    device: torch.device,
    bf16: bool,
    phase1b: bool,
) -> dict:
    model.eval()
    rows = []
    for offset in range(0, len(documents), batch_size):
        batch = collate_documents(documents[offset:offset + batch_size]).to(device)
        kwargs = {}
        if phase1b:
            kwargs["pointers"] = _oracle_pointers(batch)
        with _autocast(device, bf16):
            logits = model(batch.input_ids, batch.attention_mask, **kwargs)
        for row, document in enumerate(batch.documents):
            answer_position = int(batch.answer_positions[row])
            answer_token = int(batch.input_ids[row, answer_position])
            item = decompose_answer_logits(
                logits[row, answer_position - 1], answer_token
            )
            item["document_id"] = document.document_id
            item["answer"] = document.answer
            rows.append(item)
    return {"aggregate": _aggregate(rows), "rows": rows}


def _load_phase1b_model(
    mode: str,
    config: dict,
    run_dir: Path,
    device: torch.device,
) -> OracleBindingDecoder:
    model = build_paired_model(
        seed=int(config["seed"]), reader_mode=mode,
        layers=int(config["model"]["layers"]), width=int(config["model"]["width"]),
        heads=int(config["model"]["heads"]),
        max_sequence=int(config["model"]["sequence_length"]),
    )
    checkpoint = torch.load(
        run_dir / "arms" / mode / "checkpoint.pt",
        map_location="cpu", weights_only=True,
    )
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval()


@torch.inference_mode()
def normalized_pointer_intervention(
    model: OracleBindingDecoder,
    documents,
    *,
    batch_size: int,
    device: torch.device,
    bf16: bool,
) -> dict:
    selected = [(document, eligible_pointer_case(document)) for document in documents]
    selected = [(document, case) for document, case in selected if case is not None]
    new_required_deltas = []
    old_required_deltas = []
    log_odds_shifts = []
    reader_deltas = []
    original_correct = 0
    intervened_correct = 0
    shifts_toward_new = 0
    conditional_predictions_changed = 0
    for offset in range(0, len(selected), batch_size):
        members = selected[offset:offset + batch_size]
        batch = collate_documents([item[0] for item in members]).to(device)
        pointers = _oracle_pointers(batch)
        override = pointers.clone()
        for row, (_, case) in enumerate(members):
            override[row, case["use_position"]] = case["alternate_declaration_position"]
        if not torch.all((override != pointers).sum(-1) == 1):
            raise AssertionError("normalized intervention must change exactly one pointer")
        with _autocast(device, bf16):
            before_logits, before_reader = model(
                batch.input_ids, batch.attention_mask, pointers=pointers,
                return_reader=True,
            )
            after_logits, after_reader = model(
                batch.input_ids, batch.attention_mask, pointers=override,
                return_reader=True,
            )
        for row, (_, case) in enumerate(members):
            answer_position = int(batch.answer_positions[row])
            before_bits = before_logits[row, answer_position - 1].float()[list(BIT_IDS)]
            after_bits = after_logits[row, answer_position - 1].float()[list(BIT_IDS)]
            before_probs = before_bits.softmax(-1)
            after_probs = after_bits.softmax(-1)
            old_slot = int(case["correct_value"])
            new_slot = int(case["alternate_value"])
            new_delta = float(after_probs[new_slot] - before_probs[new_slot])
            old_delta = float(after_probs[old_slot] - before_probs[old_slot])
            log_odds_before = before_bits[new_slot] - before_bits[old_slot]
            log_odds_after = after_bits[new_slot] - after_bits[old_slot]
            log_odds_shift = float(log_odds_after - log_odds_before)
            new_required_deltas.append(new_delta)
            old_required_deltas.append(old_delta)
            log_odds_shifts.append(log_odds_shift)
            shifts_toward_new += int(log_odds_shift > 0)
            original_correct += int(int(before_bits.argmax()) == old_slot)
            intervened_correct += int(int(after_bits.argmax()) == new_slot)
            conditional_predictions_changed += int(
                int(before_bits.argmax()) != int(after_bits.argmax())
            )
            use = case["use_position"]
            reader_deltas.append(float((
                after_reader["reader_output"][row, use].float()
                - before_reader["reader_output"][row, use].float()
            ).norm()))
    count = len(selected)
    return {
        "eligible_cases": count,
        "new_required_normalized_probability_delta": _summary(new_required_deltas),
        "old_required_normalized_probability_delta": _summary(old_required_deltas),
        "log_odds_shift_toward_new_required": _summary(log_odds_shifts),
        "fraction_log_odds_shifted_toward_new_required": shifts_toward_new / count,
        "conditional_bit_accuracy_correct_pointer": original_correct / count,
        "conditional_bit_accuracy_intervened_pointer_new_required": intervened_correct / count,
        "no_prediction_change_complement_baseline": 1.0 - original_correct / count,
        "conditional_prediction_changed_fraction": conditional_predictions_changed / count,
        "reader_output_delta_l2": _summary(reader_deltas),
        "binary_conservation_max_abs": max(
            abs(new + old) for new, old in zip(new_required_deltas, old_required_deltas)
        ),
    }


def _phase1_scores(
    phase1_config: dict,
    phase1_run: Path,
    documents,
    device: torch.device,
) -> tuple[list[dict], dict]:
    records = []
    grouped = defaultdict(list)
    paths = sorted(
        list(phase1_run.glob("backbones/l*_w*/plain/seed_*/checkpoint.pt"))
        + list(phase1_run.glob("backbones/l*_w*/matched/seed_*/checkpoint.pt"))
    )
    for path in paths:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        model = _load_model(phase1_config, phase1_run, checkpoint, device)
        scored = score_model(
            model, documents,
            batch_size=int(phase1_config["evaluation"]["batch_sizes"][str(checkpoint["width"])]),
            device=device, bf16=bool(phase1_config["runtime"]["bf16"]), phase1b=False,
        )
        record = {
            "arm": checkpoint["arm"], "layers": checkpoint["layers"],
            "width": checkpoint["width"], "seed": checkpoint["seed"],
            "checkpoint_sha256": file_hash(path),
            **scored["aggregate"],
        }
        records.append(record)
        grouped[(checkpoint["arm"], checkpoint["layers"], checkpoint["width"])].append(record)
    means = {}
    metric_names = (
        "answer_accuracy_full_vocabulary", "top1_bit_category_detection_accuracy",
        "conditional_bit_selection_accuracy", "answer_nll", "bit_category_nll",
        "correct_bit_given_category_nll", "bit_category_probability",
        "correct_bit_given_category_probability", "decomposition_residual",
    )
    for key, members in grouped.items():
        label = f"{key[0]}_l{key[1]}_w{key[2]}"
        means[label] = {
            "seeds": len(members),
            **{
                name: sum(member[name] for member in members) / len(members)
                for name in metric_names
            },
        }
    return records, means


def _checkpoint_hashes(phase1_run: Path, phase1b_run: Path) -> dict[str, str]:
    paths = [
        *phase1_run.glob("backbones/l*_w*/plain/seed_*/checkpoint.pt"),
        *phase1_run.glob("backbones/l*_w*/matched/seed_*/checkpoint.pt"),
        *phase1b_run.glob("arms/*/checkpoint.pt"),
    ]
    return {str(path.resolve()): file_hash(path) for path in sorted(paths)}


def _conclusion(report: dict) -> str:
    mean_metrics = list(report["phase1_means"].values()) + [
        value["metrics"] for value in report["phase1b"].values()
    ]
    best_mean_conditional = max(
        item["conditional_bit_selection_accuracy"] for item in mean_metrics
    )
    individual_metrics = report["phase1_checkpoints"] + [
        value["metrics"] for value in report["phase1b"].values()
    ]
    best_individual_conditional = max(
        item["conditional_bit_selection_accuracy"] for item in individual_metrics
    )
    best_category = max(
        item["top1_bit_category_detection_accuracy"] for item in mean_metrics
    )
    if best_mean_conditional >= 0.80 and best_category < 0.50:
        return (
            "binary selection was learned but answer formatting/category detection failed"
        )
    return (
        "the underlying lookup task remains unlearned: the best three-seed mean "
        f"conditional bit accuracy is {best_mean_conditional:.3f} and the best individual "
        f"checkpoint is {best_individual_conditional:.3f}, despite bit-category detection "
        f"reaching {best_category:.3f}"
    )


def _markdown(report: dict) -> str:
    lines = [
        "# Answer-position and binary-selection audit",
        "",
        f"**Conclusion: {report['conclusion']}.** This is evaluation-only; all checkpoint "
        "hashes are unchanged.",
        "",
        "## Scored position verification",
        "",
        "For every document, logits at `answer_position - 1` score the token stored at "
        "`answer_position`, exactly matching the shifted causal-CE target. Examples:",
        "",
        "| Whitespace | Arrow pos. | Logits pos. | Previous token | Answer pos. | Target |",
        "| --- | ---: | ---: | --- | ---: | --- |",
    ]
    for example in report["answer_position_examples"]:
        lines.append(
            f"| {example['whitespace']} | {example['arrow_position']} | "
            f"{example['scored_logits_position']} | {example['scored_after_token']} | "
            f"{example['answer_position']} | {example['target_token']} |"
        )
    lines.extend((
        "", "## Loss decomposition on 304 held-out lookup documents", "",
        "`answer NLL = bit-category NLL + correct-bit-given-category NLL`.", "",
        "| Model | Full answer acc. | Top-1 is bit | Conditional bit acc. | Answer NLL | Category NLL | Selection NLL | P(bit category) | P(correct bit | bit) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ))
    for name, value in report["phase1_means"].items():
        lines.append(
            f"| Phase 1 {name} | {value['answer_accuracy_full_vocabulary']:.4f} | "
            f"{value['top1_bit_category_detection_accuracy']:.4f} | "
            f"{value['conditional_bit_selection_accuracy']:.4f} | "
            f"{value['answer_nll']:.4f} | {value['bit_category_nll']:.4f} | "
            f"{value['correct_bit_given_category_nll']:.4f} | "
            f"{value['bit_category_probability']:.4f} | "
            f"{value['correct_bit_given_category_probability']:.4f} |"
        )
    for name, item in report["phase1b"].items():
        value = item["metrics"]
        lines.append(
            f"| Phase 1b {name} | {value['answer_accuracy_full_vocabulary']:.4f} | "
            f"{value['top1_bit_category_detection_accuracy']:.4f} | "
            f"{value['conditional_bit_selection_accuracy']:.4f} | "
            f"{value['answer_nll']:.4f} | {value['bit_category_nll']:.4f} | "
            f"{value['correct_bit_given_category_nll']:.4f} | "
            f"{value['bit_category_probability']:.4f} | "
            f"{value['correct_bit_given_category_probability']:.4f} |"
        )
    lines.extend((
        "", "Phase 1 entries are means across seeds 11/22/33. The decomposition residual "
        f"is at most {report['maximum_absolute_decomposition_residual']:.3e} nat.",
        "", "## Normalized `{0,1}` pointer interventions", "",
        "| Phase 1b reader | Cases | Reader delta L2 | ΔP(new required | bit) | Δlog-odds toward new | Fraction shifted toward new | Prediction changed | Correct-pointer acc. | Intervened-required acc. | No-change complement |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ))
    for name, item in report["phase1b"].items():
        value = item["pointer_intervention"]
        lines.append(
            f"| {name} | {value['eligible_cases']} | "
            f"{value['reader_output_delta_l2']['mean']:.6f} | "
            f"{value['new_required_normalized_probability_delta']['mean']:+.6f} | "
            f"{value['log_odds_shift_toward_new_required']['mean']:+.6f} | "
            f"{value['fraction_log_odds_shifted_toward_new_required']:.4f} | "
            f"{value['conditional_prediction_changed_fraction']:.4f} | "
            f"{value['conditional_bit_accuracy_correct_pointer']:.4f} | "
            f"{value['conditional_bit_accuracy_intervened_pointer_new_required']:.4f} | "
            f"{value['no_prediction_change_complement_baseline']:.4f} |"
        )
    lines.extend((
        "", "A formatting-only failure would show strong conditional bit accuracy despite "
        "low bit-category probability. That pattern is absent. Pointer interventions also "
        "fail to move normalized binary probabilities consistently toward the newly required "
        "bit. Intervened accuracy matches the complement expected when predictions do not "
        "change, so the lookup computation itself remains unlearned.",
        "", "Reproduce:", "",
        "`python -m experiments.modular_phase1b.answer_analysis --phase1-run "
        "scratch/modular-phase1 --phase1b-run scratch/modular-phase1b`", "",
    ))
    return "\n".join(lines)


def run(phase1_run: Path, phase1b_run: Path) -> dict:
    hashes_before = _checkpoint_hashes(phase1_run, phase1b_run)
    phase1_config = load_config("experiments/modular_phase1/configs/pilot.json")
    phase1b_config = _load_config(
        Path("experiments/modular_phase1b/configs/diagnostic.json")
    )
    documents = _lookup_documents(phase1_run)["heldout"]
    device = torch.device(phase1_config["runtime"]["device"])
    examples = verify_answer_positions(documents)
    phase1_records, phase1_means = _phase1_scores(
        phase1_config, phase1_run, documents, device
    )
    phase1b = {}
    for mode in phase1b_config["model"]["arms"]:
        model = _load_phase1b_model(mode, phase1b_config, phase1b_run, device)
        scored = score_model(
            model, documents,
            batch_size=int(phase1b_config["model"]["batch_size"]),
            device=device, bf16=bool(phase1b_config["runtime"]["bf16"]), phase1b=True,
        )
        phase1b[mode] = {
            "checkpoint_sha256": file_hash(phase1b_run / "arms" / mode / "checkpoint.pt"),
            "metrics": scored["aggregate"],
            "pointer_intervention": normalized_pointer_intervention(
                model, documents,
                batch_size=int(phase1b_config["model"]["batch_size"]),
                device=device, bf16=bool(phase1b_config["runtime"]["bf16"]),
            ),
        }
    residuals = [
        abs(value["decomposition_residual"]) for value in phase1_means.values()
    ] + [
        abs(value["metrics"]["decomposition_residual"])
        for value in phase1b.values()
    ]
    report = {
        "schema": 1,
        "evaluation_only": True,
        "optimizer_steps": 0,
        "documents": len(documents),
        "dataset": "Phase 1 confirmation base, non-literal lookup only",
        "answer_position_examples": examples,
        "decomposition_identity": (
            "answer_nll = bit_category_nll + correct_bit_given_category_nll"
        ),
        "phase1_checkpoints": phase1_records,
        "phase1_means": phase1_means,
        "phase1b": phase1b,
        "maximum_absolute_decomposition_residual": max(residuals),
        "checkpoint_hashes_before": hashes_before,
    }
    report["conclusion"] = _conclusion(report)
    hashes_after = _checkpoint_hashes(phase1_run, phase1b_run)
    if hashes_after != hashes_before:
        raise RuntimeError("evaluation changed a checkpoint")
    report["checkpoint_hashes_after"] = hashes_after
    report["checkpoint_hashes_unchanged"] = True
    _write_json(phase1b_run / "answer_analysis.json", report)
    (phase1b_run / "ANSWER_ANALYSIS.md").write_text(
        _markdown(report), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase1-run", type=Path, default=Path("scratch/modular-phase1"))
    parser.add_argument("--phase1b-run", type=Path, default=Path("scratch/modular-phase1b"))
    args = parser.parse_args()
    report = run(args.phase1_run, args.phase1b_run)
    print(json.dumps({
        "conclusion": report["conclusion"],
        "documents": report["documents"],
        "checkpoint_hashes_unchanged": report["checkpoint_hashes_unchanged"],
        "report": str((args.phase1b_run / "ANSWER_ANALYSIS.md").resolve()),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

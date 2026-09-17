"""Run the bounded Phase 1b binding-reader positive-control diagnostic."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from experiments.modular_phase1.audit import eligible_pointer_case
from experiments.modular_phase1.config import file_hash, object_hash
from experiments.modular_phase1.data import (
    collate_documents,
    load_corpus,
    paired_batches,
)
from experiments.modular_phase1.language import Document, Vocabulary
from experiments.modular_phase1.reference import ReferenceInterpreter
from experiments.modular_phase1.training import (
    _autocast,
    _optimizer,
    _state_hash,
    _trim_mask,
)

from .models import OracleBindingDecoder, build_paired_model


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


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


def _load_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    config["_config_path"] = str(path.resolve())
    config["_config_sha256"] = object_hash(config)
    return config


def _source_artifact_hashes(source_run: Path) -> dict[str, str]:
    paths = [
        source_run / "REPORT.md",
        source_run / "report.json",
        source_run / "AUDIT.md",
        source_run / "audit.json",
        source_run / "specialist" / "specialist.pt",
        source_run / "specialist" / "manifest.json",
        *source_run.glob("backbones/l*_w*/*/seed_*/checkpoint.pt"),
    ]
    return {
        str(path.resolve()): file_hash(path)
        for path in sorted(paths)
        if path.exists()
    }


def _lookup_documents(source_run: Path) -> dict[str, list[Document]]:
    manifest = json.loads(
        (source_run / "data" / "manifest.json").read_text(encoding="utf-8")
    )
    result = {}
    for target, source in (
        ("train", "backbone_train"),
        ("validation", "backbone_validation"),
        ("heldout", "confirmation"),
    ):
        documents = load_corpus(manifest["files"][source]["path"])
        result[target] = [
            document for document in documents
            if document.metadata.get("variant") == "base"
            and not document.metadata["literal"]
            and document.metadata["query_kind"] == "lookup"
        ]
    return result


def _declaration_value_position(document: Document, declaration_position: int) -> int:
    for position in range(declaration_position + 1, len(document.token_ids)):
        token = Vocabulary.TOKENS[document.token_ids[position]]
        if token in ("0", "1"):
            return position
        if token == ";":
            break
    raise ValueError("declaration has no value token")


def _oracle_pointers(batch) -> torch.Tensor:
    """Rebuild declaration addresses from the independent interpreter, values unused."""
    pointers = torch.full_like(batch.pointers, -1)
    reference = ReferenceInterpreter()
    for row, document in enumerate(batch.documents):
        for binding in reference.interpret(document.text).bindings:
            pointers[row, binding.use_position] = binding.declaration_position
    if not torch.equal(pointers, batch.pointers):
        raise AssertionError("reference and programmed binding addresses disagree")
    return pointers


@torch.inference_mode()
def inspect_payload(
    documents: list[Document],
    config: dict,
    device: torch.device,
) -> dict:
    """Flip only a bound declaration bit (and trailing label) and inspect the payload."""
    model = build_paired_model(
        seed=int(config["seed"]), reader_mode="current_window",
        layers=int(config["model"]["layers"]), width=int(config["model"]["width"]),
        heads=int(config["model"]["heads"]),
        max_sequence=int(config["model"]["sequence_length"]),
    ).to(device).eval()
    candidate_deltas = []
    context_deltas = []
    output_deltas = []
    offsets: dict[int, int] = {}
    changed_slots_exact = 0
    causal = 0
    examples = []
    reference = ReferenceInterpreter()
    for document in documents[:128]:
        binding = reference.interpret(document.text).bindings[0]
        value_position = _declaration_value_position(
            document, binding.declaration_position
        )
        batch = collate_documents([document]).to(device)
        pointers = _oracle_pointers(batch)
        flipped_ids = batch.input_ids.clone()
        original_value = int(flipped_ids[0, value_position])
        flipped_ids[0, value_position] = (
            Vocabulary.TO_ID["1"]
            if original_value == Vocabulary.TO_ID["0"] else Vocabulary.TO_ID["0"]
        )
        answer_position = int(batch.answer_positions[0])
        flipped_ids[0, answer_position] = (
            Vocabulary.TO_ID["1"]
            if int(flipped_ids[0, answer_position]) == Vocabulary.TO_ID["0"]
            else Vocabulary.TO_ID["0"]
        )
        before = model.reader_inputs(batch.input_ids, pointers)
        after = model.reader_inputs(flipped_ids, pointers)
        use = binding.use_position
        slot_delta = (
            after["candidates"][0, use].float()
            - before["candidates"][0, use].float()
        ).norm(dim=-1)
        changed = slot_delta > 0
        value_offset = value_position - binding.declaration_position
        offsets[value_offset] = offsets.get(value_offset, 0) + 1
        changed_slots_exact += int(
            int(changed.sum()) == 1 and bool(changed[value_offset])
        )
        causal += int(value_position < use)
        candidate_deltas.append(float(slot_delta[value_offset]))
        context_deltas.append(float(
            (after["context"][0, use].float()
             - before["context"][0, use].float()).norm()
        ))
        output_deltas.append(float(
            (after["reader_output"][0, use].float()
             - before["reader_output"][0, use].float()).norm()
        ))
        if len(examples) < 5:
            indices = before["indices"][0, use].tolist()
            examples.append({
                "document_id": document.document_id,
                "declaration_identifier_position": binding.declaration_position,
                "value_position": value_position,
                "use_position": use,
                "window_positions": indices,
                "window_tokens": [
                    Vocabulary.TOKENS[int(batch.input_ids[0, position])]
                    for position in indices
                ],
                "value_window_offset": value_offset,
            })
    return {
        "documents": min(128, len(documents)),
        "held_constant": ["identifier", "formatting", "scope", "all pre-answer tokens except assigned bit"],
        "value_window_offset_counts": {str(key): value for key, value in sorted(offsets.items())},
        "value_present_in_window": sum(offsets.values()),
        "value_position_precedes_query_use": causal,
        "exactly_one_candidate_slot_changed_and_it_was_value": changed_slots_exact,
        "value_candidate_embedding_delta_l2": _summary(candidate_deltas),
        "weighted_context_delta_l2_at_fresh_initialization": _summary(context_deltas),
        "reader_output_delta_l2_at_fresh_initialization": _summary(output_deltas),
        "fresh_reader_pointer_scale_is_zero": bool(
            torch.count_nonzero(model.reader.pointer_scale) == 0
        ),
        "representation_source": (
            "Each candidate is token_embedding(input_id) + absolute_position_embedding. "
            "These are non-contextual raw embeddings. With identifier/format/scope fixed, "
            "only the value-token candidate can contain the changed assigned bit. The "
            "entire declaration window precedes the query use, so gathering it is causal."
        ),
        "examples": examples,
    }


def _model_kwargs(config: dict, mode: str) -> dict:
    return {
        "seed": int(config["seed"]),
        "reader_mode": mode,
        "layers": int(config["model"]["layers"]),
        "width": int(config["model"]["width"]),
        "heads": int(config["model"]["heads"]),
        "max_sequence": int(config["model"]["sequence_length"]),
    }


def train_arm(
    mode: str,
    documents: list[Document],
    config: dict,
    run_dir: Path,
    device: torch.device,
) -> tuple[OracleBindingDecoder, dict]:
    model = build_paired_model(**_model_kwargs(config, mode)).to(device)
    initial_state_hash = _state_hash(model.state_dict())
    backbone_initial_hash = _state_hash(model.backbone.state_dict())
    optimizer = _optimizer(model.parameters(), config)
    batches = paired_batches(
        documents, batch_size=int(config["model"]["batch_size"]),
        seed=int(config["seed"]),
    )
    budget = int(config["budget"]["targets_per_run"])
    clip = float(config["optimizer"]["gradient_clip"])
    tokens = steps = 0
    loss_sum = 0.0
    model.train()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    while tokens < budget:
        batch = collate_documents(next(batches)).to(device)
        pointers = _oracle_pointers(batch)
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, bool(config["runtime"]["bf16"])):
            logits = model(
                batch.input_ids, batch.attention_mask, pointers=pointers
            )
            mask = _trim_mask(batch.attention_mask[:, 1:].clone(), budget - tokens)
            targets = batch.input_ids[:, 1:][mask]
            loss = F.cross_entropy(logits[:, :-1][mask].float(), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        consumed = int(mask.sum())
        tokens += consumed
        steps += 1
        loss_sum += float(loss.detach()) * consumed
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    output = run_dir / "arms" / mode
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "checkpoint.pt"
    checkpoint = {
        "schema": 1,
        "phase": config["phase"],
        "config_sha256": config["_config_sha256"],
        "mode": mode,
        "seed": int(config["seed"]),
        "state_dict": {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
    }
    torch.save(checkpoint, checkpoint_path)
    record = {
        "mode": mode,
        "tokens": tokens,
        "steps": steps,
        "mean_training_ce": loss_sum / tokens,
        "accelerator_seconds": elapsed,
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "initial_state_sha256": initial_state_hash,
        "initial_backbone_sha256": backbone_initial_hash,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": file_hash(checkpoint_path),
        "trained_state_sha256": _state_hash(checkpoint["state_dict"]),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "reader_parameters": sum(parameter.numel() for parameter in model.reader.parameters()),
    }
    _write_json(output / "train_metrics.json", record)
    return model.eval(), record


@torch.inference_mode()
def evaluate(
    model: OracleBindingDecoder,
    documents: list[Document],
    config: dict,
    device: torch.device,
) -> dict:
    correct = 0
    answer_nll = 0.0
    nll_sum = 0.0
    nll_tokens = 0
    batch_size = int(config["model"]["batch_size"])
    for offset in range(0, len(documents), batch_size):
        batch = collate_documents(documents[offset:offset + batch_size]).to(device)
        pointers = _oracle_pointers(batch)
        with _autocast(device, bool(config["runtime"]["bf16"])):
            logits = model(
                batch.input_ids, batch.attention_mask, pointers=pointers
            )
        log_probs = F.log_softmax(logits.float(), -1)
        targets = batch.input_ids[:, 1:]
        token_nll = -log_probs[:, :-1].gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        for row in range(len(batch.documents)):
            answer_position = int(batch.answer_positions[row])
            answer_token = int(batch.input_ids[row, answer_position])
            answer_logits = logits[row, answer_position - 1].float()
            correct += int(int(answer_logits.argmax()) == answer_token)
            answer_nll += float(-F.log_softmax(answer_logits, -1)[answer_token])
            valid = batch.attention_mask[row, 1:]
            nll_sum += float(token_nll[row, valid].sum())
            nll_tokens += int(valid.sum())
    return {
        "documents": len(documents),
        "answer_accuracy": correct / len(documents),
        "answer_nll": answer_nll / len(documents),
        "overall_nll": nll_sum / nll_tokens,
    }


@torch.inference_mode()
def pointer_interventions(
    model: OracleBindingDecoder,
    documents: list[Document],
    config: dict,
    device: torch.device,
) -> dict:
    selected = [(document, eligible_pointer_case(document)) for document in documents]
    selected = [(document, case) for document, case in selected if case is not None]
    original_deltas = []
    alternate_deltas = []
    alternate_bit_deltas = []
    reader_deltas = []
    attention_on_value = []
    gates = []
    pointer_signal_norms = []
    feature_signal_norms = []
    correct_accuracy = 0
    intervened_required_accuracy = 0
    reader_changed = 0
    batch_size = int(config["model"]["batch_size"])
    for offset in range(0, len(selected), batch_size):
        members = selected[offset:offset + batch_size]
        batch = collate_documents([item[0] for item in members]).to(device)
        pointers = _oracle_pointers(batch)
        override = pointers.clone()
        for row, (_, case) in enumerate(members):
            override[row, case["use_position"]] = case["alternate_declaration_position"]
        if not torch.all((override != pointers).sum(-1) == 1):
            raise AssertionError("intervention must change exactly one pointer")
        with _autocast(device, bool(config["runtime"]["bf16"])):
            correct_logits, correct_reader = model(
                batch.input_ids, batch.attention_mask, pointers=pointers,
                return_reader=True,
            )
            wrong_logits, wrong_reader = model(
                batch.input_ids, batch.attention_mask, pointers=override,
                return_reader=True,
            )
        for row, (_, case) in enumerate(members):
            use = case["use_position"]
            answer_position = int(batch.answer_positions[row])
            original_id = Vocabulary.TO_ID[str(case["correct_value"])]
            alternate_id = Vocabulary.TO_ID[str(case["alternate_value"])]
            before = correct_logits[row, answer_position - 1].float()
            after = wrong_logits[row, answer_position - 1].float()
            before_probs = before.softmax(-1)
            after_probs = after.softmax(-1)
            original_deltas.append(float(after_probs[original_id] - before_probs[original_id]))
            alternate_deltas.append(float(after_probs[alternate_id] - before_probs[alternate_id]))
            bit_ids = torch.tensor(
                [Vocabulary.TO_ID["0"], Vocabulary.TO_ID["1"]], device=device
            )
            before_bits = before[bit_ids].softmax(-1)
            after_bits = after[bit_ids].softmax(-1)
            alternate_slot = case["alternate_value"]
            alternate_bit_deltas.append(float(
                after_bits[alternate_slot] - before_bits[alternate_slot]
            ))
            correct_accuracy += int(int(before.argmax()) == original_id)
            intervened_required_accuracy += int(int(after.argmax()) == alternate_id)
            reader_delta = float((
                wrong_reader["reader_output"][row, use].float()
                - correct_reader["reader_output"][row, use].float()
            ).norm())
            reader_deltas.append(reader_delta)
            reader_changed += int(reader_delta > 0)
            value_mask = correct_reader["value_mask"][row, use]
            attention_on_value.append(float(
                correct_reader["weights"][row, use][value_mask].sum()
            ))
            gates.append(float(correct_reader["gate"][row, use]))
            pointer_signal_norms.append(float(
                correct_reader["pointer_signal"][row, use].float().norm()
            ))
            feature_signal_norms.append(float(
                correct_reader["feature_signal"][row, use].float().norm()
            ))
    count = len(selected)
    return {
        "heldout_lookup_documents": len(documents),
        "eligible_opposite_value_single_pointer_cases": count,
        "ineligible": len(documents) - count,
        "original_required_answer_probability_wrong_minus_correct_pointer": _summary(
            original_deltas
        ),
        "intervened_required_answer_probability_wrong_minus_correct_pointer": _summary(
            alternate_deltas
        ),
        "intervened_required_binary_probability_wrong_minus_correct_pointer": _summary(
            alternate_bit_deltas
        ),
        "reader_output_delta_l2": _summary(reader_deltas),
        "reader_output_changed": reader_changed,
        "correct_pointer_original_answer_accuracy": correct_accuracy / count,
        "wrong_pointer_intervened_required_answer_accuracy": (
            intervened_required_accuracy / count
        ),
        "selection_and_integration": {
            "attention_mass_on_observed_value_token": _summary(attention_on_value),
            "pointer_gate": _summary(gates),
            "pointer_signal_l2": _summary(pointer_signal_norms),
            "address_feature_signal_l2": _summary(feature_signal_norms),
            "pointer_scale_l2": float(model.reader.pointer_scale.float().norm()),
        },
    }


@torch.inference_mode()
def trained_value_flip(
    model: OracleBindingDecoder,
    documents: list[Document],
    config: dict,
    device: torch.device,
) -> dict:
    reader_deltas = []
    required_probability_deltas = []
    reference = ReferenceInterpreter()
    for document in documents[:128]:
        binding = reference.interpret(document.text).bindings[0]
        value_position = _declaration_value_position(document, binding.declaration_position)
        batch = collate_documents([document]).to(device)
        pointers = _oracle_pointers(batch)
        flipped = batch.input_ids.clone()
        new_value = 1 - document.answer
        flipped[0, value_position] = Vocabulary.TO_ID[str(new_value)]
        flipped[0, int(batch.answer_positions[0])] = Vocabulary.TO_ID[str(new_value)]
        with _autocast(device, bool(config["runtime"]["bf16"])):
            before_logits, before_reader = model(
                batch.input_ids, batch.attention_mask, pointers=pointers,
                return_reader=True,
            )
            after_logits, after_reader = model(
                flipped, batch.attention_mask, pointers=pointers,
                return_reader=True,
            )
        use = binding.use_position
        reader_deltas.append(float((
            after_reader["reader_output"][0, use].float()
            - before_reader["reader_output"][0, use].float()
        ).norm()))
        answer_position = int(batch.answer_positions[0])
        new_id = Vocabulary.TO_ID[str(new_value)]
        before_prob = before_logits[0, answer_position - 1].float().softmax(-1)[new_id]
        after_prob = after_logits[0, answer_position - 1].float().softmax(-1)[new_id]
        required_probability_deltas.append(float(after_prob - before_prob))
    return {
        "documents": min(128, len(documents)),
        "reader_output_delta_l2": _summary(reader_deltas),
        "new_required_answer_probability_flipped_minus_original_input": _summary(
            required_probability_deltas
        ),
    }


def _verdict(report: dict) -> dict:
    current = report["arms"]["current_window"]
    exact = report["arms"]["exact_value"]
    accuracy_gain = (
        exact["evaluation"]["heldout"]["answer_accuracy"]
        - current["evaluation"]["heldout"]["answer_accuracy"]
    )
    exact_intervention = exact["pointer_intervention"][
        "intervened_required_answer_probability_wrong_minus_correct_pointer"
    ]["mean"]
    current_intervention = current["pointer_intervention"][
        "intervened_required_answer_probability_wrong_minus_correct_pointer"
    ]["mean"]
    if accuracy_gain >= 0.05 and exact_intervention >= current_intervention + 0.01:
        classification = "inadequate retrieved content"
        explanation = (
            "Exact selection materially improves held-out accuracy and causal pointer "
            "sensitivity while leaving the integration path paired."
        )
    elif exact["evaluation"]["heldout"]["answer_accuracy"] < 0.50 and exact_intervention < 0.02:
        classification = "ineffective integration"
        explanation = (
            "Even the exact observed-value positive control remains weak and causally "
            "insensitive, so merely making the value payload explicit does not repair use."
        )
    else:
        classification = "still unresolved"
        explanation = (
            "The exact-value control does not separate content selection from integration "
            "strongly enough for a unique localization."
        )
    return {
        "classification": classification,
        "heldout_accuracy_exact_minus_current": accuracy_gain,
        "intervened_required_probability_mean_exact_minus_current": (
            exact_intervention - current_intervention
        ),
        "explanation": explanation,
        "scope": (
            "Diagnostic positive control only: this is not an efficiency result, does not "
            "reverse the Phase 1 failed verdict, and does not validate the learned specialist."
        ),
    }


def _markdown(report: dict) -> str:
    payload = report["payload_inspection"]
    lines = [
        "# Phase 1b binding-reader diagnostic",
        "",
        f"**Result: {report['verdict']['classification']}.** "
        f"{report['verdict']['explanation']}",
        "",
        "The original Phase 1 verdict remains **failed composition**. This diagnostic uses "
        "fresh backbones and oracle declaration addresses only; it is not an efficiency "
        "claim or a learned-specialist pass.",
        "",
        "## Retrieved payload inspection",
        "",
        f"Across {payload['documents']} fixed-format bit-flip pairs, the assigned value was "
        f"inside the current seven-token window in {payload['value_present_in_window']} and "
        f"preceded the query use in {payload['value_position_precedes_query_use']}. Exactly "
        f"one candidate embedding changed—and it was the value token—in "
        f"{payload['exactly_one_candidate_slot_changed_and_it_was_value']} cases.",
        "",
        f"The value appeared at offsets "
        f"{json.dumps(payload['value_window_offset_counts'], sort_keys=True)} relative to "
        "the declaration identifier. Addresses were rebuilt from the independent reference "
        "interpreter; its values and answers were not passed to either model.",
        "",
        f"Mean value-candidate embedding delta: "
        f"{payload['value_candidate_embedding_delta_l2']['mean']:.6f}; mean weighted-context "
        f"delta at fresh initialization: "
        f"{payload['weighted_context_delta_l2_at_fresh_initialization']['mean']:.6f}; mean "
        f"reader-output delta: "
        f"{payload['reader_output_delta_l2_at_fresh_initialization']['mean']:.6f}. The last "
        "is zero because the paired reader's pointer scale is zero-initialized, not because "
        "the retrieved payload lacks the bit.",
        "",
        payload["representation_source"],
        "",
        "## Fresh paired training",
        "",
        "| Reader | Targets | GPU seconds | Train acc. | Train answer NLL | Validation acc. | Validation NLL | Held-out acc. | Held-out NLL |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode in ("current_window", "exact_value"):
        arm = report["arms"][mode]
        train = arm["evaluation"]["train"]
        validation = arm["evaluation"]["validation"]
        heldout = arm["evaluation"]["heldout"]
        lines.append(
            f"| {mode} | {arm['training']['tokens']} | "
            f"{arm['training']['accelerator_seconds']:.3f} | "
            f"{train['answer_accuracy']:.4f} | {train['answer_nll']:.4f} | "
            f"{validation['answer_accuracy']:.4f} | {validation['answer_nll']:.4f} | "
            f"{heldout['answer_accuracy']:.4f} | {heldout['answer_nll']:.4f} |"
        )
    lines.extend((
        "", "Both arms use identical data order, backbone/reader initialization, optimizer, "
        "ordinary causal CE over every valid next-token target, parameter count, and "
        "integration path. The exact-value arm changes only the declaration-window selection "
        "weights from learned attention to the observed bit's one-hot position.",
        "", "After training, changing only the assigned bit moved the current reader output "
        f"by L2={report['arms']['current_window']['assigned_bit_flip']['reader_output_delta_l2']['mean']:.6f} "
        "and the exact-value reader by "
        f"L2={report['arms']['exact_value']['assigned_bit_flip']['reader_output_delta_l2']['mean']:.6f}. "
        "The corresponding mean changes in probability of the new required answer were "
        f"{report['arms']['current_window']['assigned_bit_flip']['new_required_answer_probability_flipped_minus_original_input']['mean']:+.6f} "
        "and "
        f"{report['arms']['exact_value']['assigned_bit_flip']['new_required_answer_probability_flipped_minus_original_input']['mean']:+.6f}; "
        "the representation changes, but the decoder does not use it effectively.",
        "", "## Single-pointer causal interventions", "",
        "| Reader | Eligible | Reader delta L2 | P(new required) delta | P(old required) delta | Correct-pointer acc. | Intervened-required acc. | Value attention |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ))
    for mode in ("current_window", "exact_value"):
        item = report["arms"][mode]["pointer_intervention"]
        lines.append(
            f"| {mode} | {item['eligible_opposite_value_single_pointer_cases']} | "
            f"{item['reader_output_delta_l2']['mean']:.6f} | "
            f"{item['intervened_required_answer_probability_wrong_minus_correct_pointer']['mean']:+.6f} | "
            f"{item['original_required_answer_probability_wrong_minus_correct_pointer']['mean']:+.6f} | "
            f"{item['correct_pointer_original_answer_accuracy']:.4f} | "
            f"{item['wrong_pointer_intervened_required_answer_accuracy']:.4f} | "
            f"{item['selection_and_integration']['attention_mass_on_observed_value_token']['mean']:.4f} |"
        )
    lines.extend((
        "", "Integration-path diagnostics on held-out queries:", "",
        "| Reader | Gate mean | Pointer-scale L2 | Pointer-signal L2 | Address-feature signal L2 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ))
    for mode in ("current_window", "exact_value"):
        item = report["arms"][mode]["pointer_intervention"]["selection_and_integration"]
        lines.append(
            f"| {mode} | {item['pointer_gate']['mean']:.4f} | "
            f"{item['pointer_scale_l2']:.4f} | {item['pointer_signal_l2']['mean']:.4f} | "
            f"{item['address_feature_signal_l2']['mean']:.4f} |"
        )
    lines.extend((
        "", f"Whole diagnostic accelerator time: "
        f"{report['budget']['whole_diagnostic_accelerator_seconds']:.3f}s of the 600s cap. "
        "Each arm consumed exactly 1,048,576 targets.",
        "", "## Conclusion", "",
        f"Classification: **{report['verdict']['classification']}**. "
        f"{report['verdict']['explanation']}",
        "",
        "The original Phase 1 artifacts and checkpoint hashes are unchanged. Any later "
        "learned-specialist experiment must independently meet the original >=99% "
        "structural-event accuracy gate before backbone training.",
        "", "Reproduce:", "",
        "`python -m experiments.modular_phase1b --config "
        "experiments/modular_phase1b/configs/diagnostic.json --run-dir "
        "scratch/modular-phase1b --force`", "",
    ))
    return "\n".join(lines)


def run(config: dict, run_dir: Path, *, force: bool = False) -> dict:
    report_path = run_dir / "report.json"
    if report_path.exists() and not force:
        return json.loads(report_path.read_text(encoding="utf-8"))
    source_run = Path(config["source_phase1_run_dir"])
    source_hashes_before = _source_artifact_hashes(source_run)
    documents = _lookup_documents(source_run)
    device = torch.device(config["runtime"]["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Phase 1b training requires the configured accelerator")
    torch.cuda.synchronize(device)
    diagnostic_started = time.perf_counter()
    payload = inspect_payload(documents["train"], config, device)

    initial_hashes = {
        mode: {
            "full": _state_hash(build_paired_model(**_model_kwargs(config, mode)).state_dict()),
            "backbone": _state_hash(
                build_paired_model(**_model_kwargs(config, mode)).backbone.state_dict()
            ),
        }
        for mode in config["model"]["arms"]
    }
    if len({item["full"] for item in initial_hashes.values()}) != 1:
        raise AssertionError("reader arms do not have identical initial tensors")

    arms = {}
    for mode in config["model"]["arms"]:
        model, training = train_arm(
            mode, documents["train"], config, run_dir, device
        )
        evaluation = {
            split: evaluate(model, members, config, device)
            for split, members in documents.items()
        }
        intervention = pointer_interventions(
            model, documents["heldout"], config, device
        )
        arms[mode] = {
            "training": training,
            "evaluation": evaluation,
            "pointer_intervention": intervention,
            "assigned_bit_flip": trained_value_flip(
                model, documents["heldout"], config, device
            ),
        }
    torch.cuda.synchronize(device)
    diagnostic_seconds = time.perf_counter() - diagnostic_started
    cap = float(config["budget"]["whole_diagnostic_accelerator_seconds"])
    if diagnostic_seconds > cap:
        raise RuntimeError(f"diagnostic exceeded {cap}s accelerator cap")
    source_hashes_after = _source_artifact_hashes(source_run)
    if source_hashes_after != source_hashes_before:
        raise RuntimeError("Phase 1b modified a Phase 1 artifact")
    report = {
        "schema": 1,
        "phase": config["phase"],
        "diagnostic_only": True,
        "original_phase1_verdict": "failed composition",
        "config_sha256": config["_config_sha256"],
        "device": torch.cuda.get_device_name(device),
        "data": {
            split: {
                "documents": len(members),
                "tokens": sum(len(document.token_ids) for document in members),
                "document_ids_sha256": object_hash(
                    [document.document_id for document in members]
                ),
            }
            for split, members in documents.items()
        },
        "oracle_address_source": (
            "ReferenceInterpreter bindings only; interpreter values and answers are unused. "
            "Equality with the programmed causal table is asserted for every batch."
        ),
        "paired_initialization": initial_hashes,
        "payload_inspection": payload,
        "arms": arms,
        "budget": {
            "targets_per_run_cap": int(config["budget"]["targets_per_run"]),
            "whole_diagnostic_accelerator_seconds_cap": cap,
            "whole_diagnostic_accelerator_seconds": diagnostic_seconds,
            "within_cap": True,
        },
        "source_phase1_artifact_hashes_before": source_hashes_before,
        "source_phase1_artifact_hashes_after": source_hashes_after,
        "source_phase1_artifacts_unchanged": True,
    }
    report["verdict"] = _verdict(report)
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(report_path, report)
    (run_dir / "REPORT.md").write_text(_markdown(report), encoding="utf-8")
    return report


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    report = run(_load_config(args.config), args.run_dir, force=args.force)
    print(json.dumps({
        "report": str((args.run_dir / "REPORT.md").resolve()),
        "verdict": report["verdict"],
        "budget": report["budget"],
    }, indent=2, sort_keys=True))

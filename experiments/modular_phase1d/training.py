"""One bounded specialist-only correction and strict qualification gate."""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

import torch

from experiments.modular_phase1.audit import _classification, _validation_slices
from experiments.modular_phase1.config import file_hash, object_hash
from experiments.modular_phase1.data import (
    collate_documents,
    load_corpus,
    paired_batches,
)
from experiments.modular_phase1.language import Vocabulary
from experiments.modular_phase1.reference import ReferenceInterpreter
from experiments.modular_phase1.specialist import (
    CausalStructureMachine,
    RecurrentSpecialist,
    specialist_parameter_report,
)
from experiments.modular_phase1.training import (
    _autocast,
    _balanced_cross_entropy,
    _optimizer,
    _state_hash,
    _trim_mask,
)


@torch.inference_mode()
def qualify_specialist(
    model: RecurrentSpecialist,
    documents,
    *,
    batch_size: int,
    device: torch.device,
    corrupt_fraction: float,
) -> dict:
    model.eval()
    all_predictions = []
    all_targets = []
    slices = defaultdict(list)
    depth_correct = depth_total = validity_correct = validity_total = 0
    for offset in range(0, len(documents), batch_size):
        members = documents[offset:offset + batch_size]
        batch = collate_documents(
            members,
            corrupt_fraction=corrupt_fraction,
            corruption_seed=7001 + offset,
        ).to(device)
        outputs = model(batch.input_ids)
        event = outputs["event_logits"].argmax(-1)
        depth = outputs["depth_logits"].argmax(-1)
        validity = outputs["validity_logits"].argmax(-1)
        mask = batch.attention_mask
        depth_correct += int((depth[mask] == batch.depths[mask]).sum())
        validity_correct += int((validity[mask] == batch.validity[mask]).sum())
        depth_total += int(mask.sum())
        validity_total += int(mask.sum())
        for row, document in enumerate(members):
            length = len(document.token_ids)
            predicted = event[row, :length].cpu()
            target = batch.events[row, :length].cpu()
            original = torch.tensor(document.token_ids)
            corrupted = not torch.equal(batch.input_ids[row, :length].cpu(), original)
            all_predictions.append(predicted)
            all_targets.append(target)
            for name in _validation_slices(document, corrupted):
                slices[name].append((predicted, target))
    overall = _classification(torch.cat(all_predictions), torch.cat(all_targets))
    slice_metrics = {}
    for name, members in sorted(slices.items()):
        detail = _classification(
            torch.cat([member[0] for member in members]),
            torch.cat([member[1] for member in members]),
        )
        slice_metrics[name] = {
            "documents": len(members),
            "tokens": detail["tokens"],
            "accuracy": detail["accuracy"],
            "structural_event_accuracy_excluding_pad_and_whitespace": detail[
                "structural_event_accuracy_excluding_pad_and_whitespace"
            ],
            "structural_macro_f1_excluding_pad_and_whitespace": detail[
                "structural_macro_f1_excluding_pad_and_whitespace"
            ],
        }

    reference = ReferenceInterpreter()
    machine = CausalStructureMachine()
    correct = total = 0
    for document in documents:
        expected = {
            binding.use_position: binding.declaration_position
            for binding in reference.interpret(document.text).bindings
        }
        trace = machine.analyze(document.token_ids)
        actual = {
            position: pointer
            for position, pointer in enumerate(trace.pointers)
            if pointer >= 0
        }
        positions = set(expected) | set(actual)
        correct += sum(expected.get(position) == actual.get(position) for position in positions)
        total += len(positions)
    return {
        "events": overall,
        "validation_slices": slice_metrics,
        "binding_accuracy": correct / max(1, total),
        "binding_operations": total,
        "depth_accuracy": depth_correct / max(1, depth_total),
        "validity_accuracy": validity_correct / max(1, validity_total),
    }


def train_corrected_specialist(
    config: dict,
    source_run: Path,
    run_dir: Path,
    device: torch.device,
) -> tuple[RecurrentSpecialist, dict]:
    source_manifest = json.loads(
        (source_run / "data" / "manifest.json").read_text(encoding="utf-8")
    )
    train_documents = load_corpus(source_manifest["files"]["specialist_train"]["path"])
    validation_documents = load_corpus(
        source_manifest["files"]["specialist_validation"]["path"]
    )
    settings = config["specialist"]
    torch.manual_seed(int(config["seed"]))
    model = RecurrentSpecialist(
        embedding_dim=int(settings["embedding_dim"]),
        hidden_dim=int(settings["hidden_dim"]),
    ).to(device)
    parameters = specialist_parameter_report(model)
    if parameters["learned_parameters"] > int(settings["max_parameters"]):
        raise RuntimeError("corrected specialist exceeds the fixed parameter ceiling")
    optimizer = _optimizer(model.parameters(), config)
    token_cap = int(config["budget"]["specialist_tokens"])
    time_cap = float(config["budget"]["specialist_accelerator_seconds"])
    batch_size = int(settings["batch_size"])
    corrupt_fraction = float(settings["corrupt_fraction"])
    batches = paired_batches(train_documents, batch_size=batch_size, seed=int(config["seed"]))
    tokens = steps = 0
    weighted_loss = 0.0
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    while tokens < token_cap and time.perf_counter() - started <= time_cap:
        batch = collate_documents(
            next(batches), corrupt_fraction=corrupt_fraction,
            corruption_seed=int(config["seed"]) + steps,
        ).to(device)
        mask = _trim_mask(batch.attention_mask.clone(), token_cap - tokens)
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, bool(config["runtime"]["bf16"])):
            output = model(batch.input_ids)
            event_loss = _balanced_cross_entropy(
                output["event_logits"][mask].float(), batch.events[mask]
            )
            depth_loss = _balanced_cross_entropy(
                output["depth_logits"][mask].float(), batch.depths[mask]
            )
            validity_loss = _balanced_cross_entropy(
                output["validity_logits"][mask].float(), batch.validity[mask]
            )
            loss = event_loss + depth_loss + validity_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(config["optimizer"]["gradient_clip"])
        )
        optimizer.step()
        consumed = int(mask.sum())
        tokens += consumed
        steps += 1
        weighted_loss += float(loss.detach()) * consumed
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    qualification = qualify_specialist(
        model, validation_documents, batch_size=batch_size, device=device,
        corrupt_fraction=corrupt_fraction,
    )
    thresholds = config["qualification"]
    gate = {
        "binding_accuracy": (
            qualification["binding_accuracy"] >= float(thresholds["binding_accuracy"])
        ),
        "structural_event_accuracy": (
            qualification["events"][
                "structural_event_accuracy_excluding_pad_and_whitespace"
            ] >= float(thresholds["structural_event_accuracy"])
        ),
        "token_budget_complete": tokens == token_cap,
        "time_budget_respected": elapsed <= time_cap,
    }
    gate["passed"] = all(gate.values())
    checkpoint = {
        "schema": 1,
        "phase": config["phase"],
        "config_sha256": config["_config_sha256"],
        "state_dict": {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
        "model": {
            "embedding_dim": model.embedding_dim,
            "hidden_dim": model.hidden_dim,
        },
        "vocabulary_sha256": Vocabulary.hash(),
        "interface_sha256": object_hash(parameters),
        "gate": gate,
    }
    specialist_dir = run_dir / "specialist"
    specialist_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = specialist_dir / "specialist.pt"
    torch.save(checkpoint, checkpoint_path)
    manifest = {
        "schema": 1,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": file_hash(checkpoint_path),
        "state_sha256": _state_hash(checkpoint["state_dict"]),
        "parameters": parameters,
        "single_predeclared_correction": {
            "old_embedding_dim": 24,
            "old_hidden_dim": 48,
            "new_embedding_dim": model.embedding_dim,
            "new_hidden_dim": model.hidden_dim,
            "representation_changed": False,
            "objective_changed": False,
            "reason": (
                "capacity correction for declaration/use state retention across whitespace "
                "and corrupted-prefix recovery; no hyperparameter sweep"
            ),
        },
        "training": {
            "tokens": tokens,
            "steps": steps,
            "mean_loss": weighted_loss / max(1, tokens),
            "accelerator_seconds": elapsed,
            "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        },
        "qualification": qualification,
        "gate": gate,
        "frozen": gate["passed"],
    }
    (specialist_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    model.requires_grad_(False).eval()
    return model, manifest


def load_corrected_specialist(
    config: dict,
    run_dir: Path,
    device: torch.device,
) -> tuple[RecurrentSpecialist, dict]:
    manifest_path = run_dir / "specialist" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checkpoint_path = Path(manifest["checkpoint"])
    if file_hash(checkpoint_path) != manifest["checkpoint_sha256"]:
        raise RuntimeError("corrected specialist checkpoint hash mismatch")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["config_sha256"] != config["_config_sha256"]:
        raise RuntimeError("corrected specialist/config hash mismatch")
    if not manifest["gate"]["passed"] or not checkpoint["gate"]["passed"]:
        raise RuntimeError("corrected specialist did not clear qualification")
    model = RecurrentSpecialist(**checkpoint["model"])
    model.load_state_dict(checkpoint["state_dict"])
    model.requires_grad_(False).eval().to(device)
    return model, manifest

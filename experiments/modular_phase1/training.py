"""Budget-enforced specialist and backbone training."""

from __future__ import annotations

import hashlib
import json
import os
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F

from .config import file_hash, object_hash
from .data import Batch, collate_documents, load_corpus, paired_batches, prepare_corpora
from .language import Vocabulary
from .models import SpecialistDecoderLM, build_models, model_parameter_report
from .reference import ReferenceInterpreter
from .specialist import (
    EVENTS,
    CausalStructureMachine,
    RecurrentSpecialist,
    specialist_parameter_report,
)


def _device(config: dict) -> torch.device:
    requested = config["runtime"]["device"]
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no GPU is available")
    return torch.device(requested)


def _autocast(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def _optimizer(parameters: Iterable[torch.nn.Parameter], config: dict) -> torch.optim.Optimizer:
    optimizer = config["optimizer"]
    return torch.optim.AdamW(
        parameters,
        lr=float(optimizer["learning_rate"]),
        betas=tuple(float(value) for value in optimizer["betas"]),
        eps=float(optimizer["epsilon"]),
        weight_decay=float(optimizer["weight_decay"]),
    )


def _state_hash(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _trim_mask(mask: torch.Tensor, remaining: int) -> torch.Tensor:
    available = int(mask.sum().item())
    if available <= remaining:
        return mask
    flat = mask.reshape(-1)
    active = flat.nonzero(as_tuple=False).squeeze(-1)
    flat[active[remaining:]] = False
    return flat.reshape_as(mask)


def _balanced_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Class-balanced structural CE with weights determined only by observed support.

    The structural gate is macro-averaged, so unweighted token CE would optimize mostly
    whitespace and semicolons. Inverse-frequency weights make each label present in the
    batch contribute equal total mass; there is no tuned weighting hyperparameter.
    """
    counts = torch.bincount(targets, minlength=logits.shape[-1]).to(logits.dtype)
    weights = torch.zeros_like(counts)
    present = counts > 0
    weights[present] = counts[present].sum() / (present.sum() * counts[present])
    return F.cross_entropy(logits, targets, weight=weights)


def _macro_f1(predictions: torch.Tensor, targets: torch.Tensor) -> tuple[float, dict[str, float]]:
    scores: dict[str, float] = {}
    for index, name in enumerate(EVENTS):
        support = targets == index
        if not support.any():
            continue
        predicted = predictions == index
        true_positive = int((support & predicted).sum())
        false_positive = int((~support & predicted).sum())
        false_negative = int((support & ~predicted).sum())
        denominator = 2 * true_positive + false_positive + false_negative
        scores[name] = 0.0 if denominator == 0 else 2 * true_positive / denominator
    structural = [score for name, score in scores.items() if name not in ("pad", "whitespace")]
    return (sum(structural) / len(structural) if structural else 0.0), scores


@torch.no_grad()
def validate_specialist(
    model: RecurrentSpecialist,
    documents,
    *,
    batch_size: int,
    device: torch.device,
    corrupt_fraction: float,
) -> dict:
    model.eval()
    event_predictions: list[torch.Tensor] = []
    event_targets: list[torch.Tensor] = []
    depth_correct = depth_total = validity_correct = validity_total = 0
    confidence_sum = confidence_total = 0
    for offset in range(0, len(documents), batch_size):
        batch = collate_documents(
            documents[offset:offset + batch_size],
            corrupt_fraction=corrupt_fraction,
            corruption_seed=7001 + offset,
        ).to(device)
        outputs = model(batch.input_ids)
        mask = batch.attention_mask
        event_pred = outputs["event_logits"].argmax(-1)
        depth_pred = outputs["depth_logits"].argmax(-1)
        validity_pred = outputs["validity_logits"].argmax(-1)
        event_predictions.append(event_pred[mask].cpu())
        event_targets.append(batch.events[mask].cpu())
        depth_correct += int((depth_pred[mask] == batch.depths[mask]).sum())
        depth_total += int(mask.sum())
        validity_correct += int((validity_pred[mask] == batch.validity[mask]).sum())
        validity_total += int(mask.sum())
        confidence_sum += float(outputs["event_logits"].softmax(-1).max(-1).values[mask].sum())
        confidence_total += int(mask.sum())

    predictions = torch.cat(event_predictions)
    targets = torch.cat(event_targets)
    macro_f1, per_event = _macro_f1(predictions, targets)

    machine = CausalStructureMachine()
    correct = total = 0
    reference = ReferenceInterpreter()
    for document in documents:
        expected = {
            item.use_position: item.declaration_position
            for item in reference.interpret(document.text).bindings
        }
        actual_trace = machine.analyze(document.token_ids)
        actual = {
            index: pointer for index, pointer in enumerate(actual_trace.pointers) if pointer >= 0
        }
        positions = set(expected) | set(actual)
        correct += sum(expected.get(position) == actual.get(position) for position in positions)
        total += len(positions)
    return {
        "event_accuracy": float((predictions == targets).float().mean()),
        "structural_event_macro_f1": macro_f1,
        "per_event_f1": per_event,
        "depth_accuracy": depth_correct / max(1, depth_total),
        "validity_accuracy": validity_correct / max(1, validity_total),
        "mean_event_confidence": confidence_sum / max(1, confidence_total),
        "binding_accuracy": correct / max(1, total),
        "binding_operations": total,
    }


def train_specialist(config: dict, run_dir: str | Path, *, force: bool = False) -> dict:
    run_dir = Path(run_dir)
    manifest_path = run_dir / "specialist" / "manifest.json"
    checkpoint_path = run_dir / "specialist" / "specialist.pt"
    if manifest_path.exists() and checkpoint_path.exists() and not force:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("config_sha256") == config["_config_sha256"]:
            return manifest
    data_manifest = prepare_corpora(config, run_dir)
    train_documents = load_corpus(data_manifest["files"]["specialist_train"]["path"])
    validation_documents = load_corpus(
        data_manifest["files"]["specialist_validation"]["path"]
    )
    device = _device(config)
    torch.manual_seed(int(config["specialist"]["seed"]))
    model = RecurrentSpecialist(
        embedding_dim=int(config["specialist"]["embedding_dim"]),
        hidden_dim=int(config["specialist"]["hidden_dim"]),
    ).to(device)
    parameters = specialist_parameter_report(model)
    if not parameters["under_ceiling"] or parameters["learned_parameters"] > int(
        config["specialist"]["max_parameters"]
    ):
        raise RuntimeError(f"specialist exceeds parameter budget: {parameters}")
    optimizer = _optimizer(model.parameters(), config)
    token_budget = int(config["budget"]["specialist_tokens"])
    time_budget = 60.0 * float(config["budget"]["specialist_accelerator_minutes"])
    batch_size = int(config["specialist"]["batch_size"])
    corrupt_fraction = float(config["specialist"]["corrupt_fraction"])
    batches = paired_batches(train_documents, batch_size=batch_size,
                             seed=int(config["specialist"]["seed"]))
    use_amp = bool(config["runtime"]["bf16"])
    clip = float(config["optimizer"]["gradient_clip"])
    tokens = steps = 0
    loss_sum = 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    while tokens < token_budget:
        if time.perf_counter() - started > time_budget:
            break
        documents = next(batches)
        batch = collate_documents(
            documents, corrupt_fraction=corrupt_fraction,
            corruption_seed=int(config["specialist"]["seed"]) + steps,
        ).to(device)
        mask = _trim_mask(batch.attention_mask.clone(), token_budget - tokens)
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, use_amp):
            outputs = model(batch.input_ids)
            event_loss = _balanced_cross_entropy(
                outputs["event_logits"][mask].float(), batch.events[mask]
            )
            depth_loss = _balanced_cross_entropy(
                outputs["depth_logits"][mask].float(), batch.depths[mask]
            )
            validity_loss = _balanced_cross_entropy(
                outputs["validity_logits"][mask].float(), batch.validity[mask]
            )
            loss = event_loss + depth_loss + validity_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        consumed = int(mask.sum())
        tokens += consumed
        steps += 1
        loss_sum += float(loss.detach()) * consumed
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    validation = validate_specialist(
        model, validation_documents, batch_size=batch_size, device=device,
        corrupt_fraction=corrupt_fraction,
    )
    thresholds = config["specialist"]["gate"]
    gate = {
        "binding": validation["binding_accuracy"] >= float(thresholds["binding_accuracy"]),
        "events": validation["structural_event_macro_f1"] >= float(
            thresholds["structural_event_macro_f1"]
        ),
        "depth": validation["depth_accuracy"] >= float(thresholds["depth_accuracy"]),
        "validity": validation["validity_accuracy"] >= float(thresholds["validity_accuracy"]),
        "token_budget_complete": tokens == token_budget,
        "time_budget_respected": elapsed <= time_budget,
    }
    gate["passed"] = all(gate.values())
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "schema": 1,
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "model": {
            "embedding_dim": model.embedding_dim,
            "hidden_dim": model.hidden_dim,
        },
        "config_sha256": config["_config_sha256"],
        "vocabulary_sha256": Vocabulary.hash(),
        "interface_sha256": object_hash(parameters),
        "gate": gate,
    }
    torch.save(checkpoint, checkpoint_path)
    manifest = {
        "schema": 1,
        "config_sha256": config["_config_sha256"],
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": file_hash(checkpoint_path),
        "state_sha256": _state_hash(checkpoint["state_dict"]),
        "interface_sha256": checkpoint["interface_sha256"],
        "parameters": parameters,
        "training": {
            "tokens": tokens,
            "steps": steps,
            "mean_loss": loss_sum / max(1, tokens),
            "accelerator_seconds": elapsed if device.type == "cuda" else 0.0,
            "wall_seconds": elapsed,
            "tokens_per_second": tokens / max(elapsed, 1e-9),
            "peak_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
            ),
            "device": str(device),
        },
        "validation": validation,
        "gate": gate,
        "frozen": gate["passed"],
    }
    _write_json(manifest_path, manifest)
    return manifest


def load_frozen_specialist(config: dict, run_dir: str | Path, device: torch.device) -> tuple[
    RecurrentSpecialist, dict
]:
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "specialist" / "manifest.json").read_text(encoding="utf-8"))
    path = Path(manifest["checkpoint"])
    if file_hash(path) != manifest["checkpoint_sha256"]:
        raise RuntimeError("specialist checkpoint hash mismatch")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not manifest["gate"]["passed"] or not checkpoint["gate"]["passed"]:
        raise RuntimeError("specialist did not pass the predeclared gate; backbone training blocked")
    if checkpoint["config_sha256"] != config["_config_sha256"]:
        raise RuntimeError("specialist/config hash mismatch")
    model = RecurrentSpecialist(**checkpoint["model"])
    model.load_state_dict(checkpoint["state_dict"])
    model.requires_grad_(False).eval().to(device)
    return model, manifest


def _backbone_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    if isinstance(model, SpecialistDecoderLM):
        return model.backbone.state_dict()
    return model.state_dict()


def _loss_for_batch(
    model: torch.nn.Module,
    batch: Batch,
    remaining: int,
) -> tuple[torch.Tensor, int]:
    kwargs = {}
    if isinstance(model, SpecialistDecoderLM):
        kwargs = {
            "pointers": batch.pointers,
            "scope_depths": batch.depths,
            "alternate_pointers": batch.alternate_pointers,
        }
    logits = model(batch.input_ids, attention_mask=batch.attention_mask, **kwargs)
    mask = batch.attention_mask[:, 1:].clone()
    mask = _trim_mask(mask, remaining)
    shifted_logits = logits[:, :-1, :][mask]
    targets = batch.input_ids[:, 1:][mask]
    return F.cross_entropy(shifted_logits.float(), targets), int(mask.sum())


def train_backbone_run(
    config: dict,
    run_dir: str | Path,
    *,
    layers: int,
    width: int,
    arm: str,
    seed: int,
    force: bool = False,
) -> dict:
    run_dir = Path(run_dir)
    name = f"l{layers}_w{width}/{arm}/seed_{seed}"
    output_dir = run_dir / "backbones" / name
    metrics_path = output_dir / "train_metrics.json"
    checkpoint_path = output_dir / "checkpoint.pt"
    if metrics_path.exists() and checkpoint_path.exists() and not force:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics.get("config_sha256") == config["_config_sha256"]:
            return metrics
    data_manifest = prepare_corpora(config, run_dir)
    documents = load_corpus(data_manifest["files"]["backbone_train"]["path"])
    device = _device(config)
    specialist, specialist_manifest = load_frozen_specialist(config, run_dir, device)
    heads = int(config["backbones"]["attention_heads"][str(width)])
    models = build_models(
        layers=layers, width=width, heads=heads,
        max_sequence=int(config["backbones"]["sequence_length"]), seed=seed,
        specialist=specialist,
    )
    parameter_report = model_parameter_report(models)
    paired_init_hash = _state_hash(_backbone_state(models["plain"]))
    model = models[arm].to(device)
    if arm == "specialist":
        assert _state_hash(_backbone_state(model)) == paired_init_hash
    optimizer = _optimizer(
        (parameter for parameter in model.parameters() if parameter.requires_grad), config
    )
    batch_size = int(config["backbones"]["batch_sizes"][str(width)])
    batches = paired_batches(documents, batch_size=batch_size, seed=seed)
    token_budget = int(config["budget"]["backbone_tokens_per_run"])
    use_amp = bool(config["runtime"]["bf16"])
    clip = float(config["optimizer"]["gradient_clip"])
    model.train()
    tokens = steps = 0
    loss_sum = 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    while tokens < token_budget:
        documents_batch = next(batches)
        batch = collate_documents(documents_batch).to(device)
        if batch.input_ids.shape[1] > int(config["backbones"]["sequence_length"]):
            raise RuntimeError("training document exceeds frozen sequence length")
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, use_amp):
            loss, consumed = _loss_for_batch(model, batch, token_budget - tokens)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad], clip
        )
        optimizer.step()
        tokens += consumed
        steps += 1
        loss_sum += float(loss.detach()) * consumed
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "schema": 1,
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "arm": arm,
        "layers": layers,
        "width": width,
        "heads": heads,
        "seed": seed,
        "config_sha256": config["_config_sha256"],
        "specialist_checkpoint_sha256": specialist_manifest["checkpoint_sha256"],
        "parameter_report": parameter_report,
    }
    torch.save(checkpoint, checkpoint_path)
    metrics = {
        "schema": 1,
        "config_sha256": config["_config_sha256"],
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": file_hash(checkpoint_path),
        "state_sha256": _state_hash(checkpoint["state_dict"]),
        "arm": arm,
        "layers": layers,
        "width": width,
        "seed": seed,
        "tokens": tokens,
        "steps": steps,
        "mean_training_nll": loss_sum / max(1, tokens),
        "wall_seconds": elapsed,
        "accelerator_seconds": elapsed if device.type == "cuda" else 0.0,
        "tokens_per_second": tokens / max(1e-9, elapsed),
        "peak_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "device": str(device),
        "paired_backbone_initialization_sha256": paired_init_hash,
        "specialist_checkpoint_sha256": specialist_manifest["checkpoint_sha256"],
        "parameter_report": parameter_report,
    }
    _write_json(metrics_path, metrics)
    return metrics


def train_all_backbones(config: dict, run_dir: str | Path, *, force: bool = False) -> list[dict]:
    started = time.perf_counter()
    time_budget = 3600.0 * float(config["budget"]["pilot_accelerator_hours"])
    results = []
    for layers, width in config["backbones"]["sizes"]:
        for seed in config["seeds"]:
            for arm in config["backbones"]["arms"]:
                if time.perf_counter() - started > time_budget:
                    raise RuntimeError("four-hour pilot wall/accelerator budget reached")
                results.append(train_backbone_run(
                    config, run_dir, layers=int(layers), width=int(width), arm=arm,
                    seed=int(seed), force=force,
                ))
    return results

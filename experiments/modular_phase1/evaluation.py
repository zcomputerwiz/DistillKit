"""Confirmation evaluation, interventions, timing, and document-level evidence."""

from __future__ import annotations

import gzip
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from .config import file_hash
from .data import collate_documents, load_corpus, prepare_corpora
from .language import Document, tokenize
from .models import SpecialistDecoderLM, build_models
from .specialist import CausalStructureMachine
from .training import _autocast, _device, load_frozen_specialist


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _slices(document: Document, dataset: str) -> list[str]:
    metadata = document.metadata
    result = [
        "all",
        dataset,
        f"depth_{metadata['depth']}",
        f"query_{metadata['query_kind']}",
        f"whitespace_{metadata['whitespace']}",
        f"variant_{metadata.get('variant', 'base')}",
    ]
    if 1 <= metadata["depth"] <= 4:
        result.append("depth_train_1_4")
    if 5 <= metadata["depth"] <= 8:
        result.append("depth_ood_5_8")
    if metadata.get("long_history"):
        result.append("long_history")
    if metadata.get("heldout_combo"):
        result.append("heldout_combo")
    if metadata["shadow_count"]:
        result.append("shadowing")
    if metadata["literal"]:
        result.append("literal_no_resolution")
    return result


@torch.no_grad()
def evaluate_documents(
    model: torch.nn.Module,
    documents: list[Document],
    *,
    dataset: str,
    batch_size: int,
    device: torch.device,
    bf16: bool,
    intervention: str = "none",
) -> tuple[list[dict], dict]:
    model.eval()
    rows: list[dict] = []
    changed_pointers = pointer_count = 0
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for offset in range(0, len(documents), batch_size):
        batch = collate_documents(documents[offset:offset + batch_size]).to(device)
        kwargs = {}
        if isinstance(model, SpecialistDecoderLM):
            kwargs = {
                "pointers": batch.pointers,
                "scope_depths": batch.depths,
                "alternate_pointers": batch.alternate_pointers,
                "intervention": intervention,
            }
            if intervention == "wrong_pointer":
                pointer_mask = batch.pointers >= 0
                changed_pointers += int(
                    ((batch.pointers != batch.alternate_pointers) & pointer_mask).sum()
                )
                pointer_count += int(pointer_mask.sum())
        with _autocast(device, bf16):
            logits = model(batch.input_ids, attention_mask=batch.attention_mask, **kwargs)
        log_probs = F.log_softmax(logits[:, :-1, :].float(), dim=-1)
        targets = batch.input_ids[:, 1:]
        target_nll = -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        valid = batch.attention_mask[:, 1:]
        for row_index, document in enumerate(batch.documents):
            answer_position = int(batch.answer_positions[row_index])
            answer_logits = logits[row_index, answer_position - 1].float()
            answer_target = int(batch.input_ids[row_index, answer_position])
            answer_nll = float(-F.log_softmax(answer_logits, -1)[answer_target])
            prediction = int(answer_logits.argmax())
            row_mask = valid[row_index]
            rows.append({
                "document_id": document.document_id,
                "base_id": document.metadata.get("base_id", document.document_id),
                "dataset": dataset,
                "intervention": intervention,
                "answer": document.answer,
                "answer_token": answer_target,
                "prediction_token": prediction,
                "correct": int(prediction == answer_target),
                "answer_nll": answer_nll,
                "nll_sum": float(target_nll[row_index][row_mask].sum()),
                "nll_tokens": int(row_mask.sum()),
                "slices": _slices(document, dataset),
                "metadata": document.metadata,
            })
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    return rows, {
        "seconds": elapsed,
        "tokens": sum(row["nll_tokens"] for row in rows),
        "tokens_per_second": sum(row["nll_tokens"] for row in rows) / max(1e-9, elapsed),
        "wrong_pointer_changed_fraction": changed_pointers / max(1, pointer_count),
    }


def aggregate_slices(rows: list[dict]) -> dict[str, dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        for name in row["slices"]:
            groups[name].append(row)
    result = {}
    for name, members in sorted(groups.items()):
        result[name] = {
            "documents": len(members),
            "answer_accuracy": sum(item["correct"] for item in members) / len(members),
            "answer_nll": sum(item["answer_nll"] for item in members) / len(members),
            "overall_nll": sum(item["nll_sum"] for item in members)
            / max(1, sum(item["nll_tokens"] for item in members)),
            "tokens": sum(item["nll_tokens"] for item in members),
        }
    return result


def invariance_metrics(rows: list[dict]) -> dict:
    base_predictions = {
        row["document_id"]: row["prediction_token"]
        for row in rows
        if row["metadata"].get("variant") == "base"
    }
    groups: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        variant = row["metadata"].get("variant")
        if variant not in ("renamed", "whitespace"):
            continue
        base = row["base_id"]
        if base in base_predictions:
            groups[variant].append(int(row["prediction_token"] == base_predictions[base]))
    return {
        name: {"pairs": len(values), "prediction_agreement": sum(values) / max(1, len(values))}
        for name, values in sorted(groups.items())
    }


@torch.no_grad()
def benchmark_online(
    model: torch.nn.Module,
    documents: list[Document],
    *,
    device: torch.device,
    bf16: bool,
    repeats: int,
) -> dict:
    sample = documents[: min(16, len(documents))]
    batch = collate_documents(sample).to(device)
    kwargs = {}
    if isinstance(model, SpecialistDecoderLM):
        kwargs = {
            "pointers": batch.pointers,
            "scope_depths": batch.depths,
            "alternate_pointers": batch.alternate_pointers,
        }
    model.eval()
    for _ in range(3):
        with _autocast(device, bf16):
            model(batch.input_ids, attention_mask=batch.attention_mask, **kwargs)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    preprocess_started = time.perf_counter()
    machine = CausalStructureMachine()
    for _ in range(repeats):
        for document in sample:
            ids = tokenize(document.text)
            if isinstance(model, SpecialistDecoderLM):
                machine.analyze(ids)
    preprocessing = time.perf_counter() - preprocess_started
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    forward_started = time.perf_counter()
    for _ in range(repeats):
        with _autocast(device, bf16):
            model(batch.input_ids, attention_mask=batch.attention_mask, **kwargs)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    forward = time.perf_counter() - forward_started
    tokens = sum(len(document.token_ids) for document in sample) * repeats
    total = preprocessing + forward
    return {
        "documents_per_repeat": len(sample),
        "repeats": repeats,
        "tokens": tokens,
        "preprocessing_seconds": preprocessing,
        "forward_seconds": forward,
        "total_seconds": total,
        "tokens_per_second": tokens / max(1e-9, total),
        "milliseconds_per_token": 1000.0 * total / max(1, tokens),
        "includes_tokenization": True,
        "includes_programmed_structure": isinstance(model, SpecialistDecoderLM),
        "includes_recurrent_specialist": isinstance(model, SpecialistDecoderLM),
        "includes_reader": isinstance(model, SpecialistDecoderLM),
    }


def _load_model(config: dict, run_dir: Path, checkpoint: dict, device: torch.device):
    specialist, manifest = load_frozen_specialist(config, run_dir, device)
    if checkpoint["specialist_checkpoint_sha256"] != manifest["checkpoint_sha256"]:
        raise RuntimeError("backbone was not trained with the frozen specialist checkpoint")
    models = build_models(
        layers=int(checkpoint["layers"]), width=int(checkpoint["width"]),
        heads=int(checkpoint["heads"]),
        max_sequence=int(config["backbones"]["sequence_length"]),
        seed=int(checkpoint["seed"]), specialist=specialist,
    )
    model = models[checkpoint["arm"]]
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device)


def evaluate_checkpoint(
    config: dict,
    run_dir: str | Path,
    checkpoint_path: str | Path,
    *,
    force: bool = False,
) -> dict:
    run_dir = Path(run_dir)
    checkpoint_path = Path(checkpoint_path)
    output_path = checkpoint_path.parent / "evaluation.json"
    rows_path = checkpoint_path.parent / "evaluation_rows.jsonl.gz"
    if output_path.exists() and rows_path.exists() and not force:
        result = json.loads(output_path.read_text(encoding="utf-8"))
        if result.get("config_sha256") == config["_config_sha256"]:
            return result
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    device = _device(config)
    model = _load_model(config, run_dir, checkpoint, device)
    data_manifest = prepare_corpora(config, run_dir)
    validation = load_corpus(data_manifest["files"]["backbone_validation"]["path"])
    confirmation = load_corpus(data_manifest["files"]["confirmation"]["path"])
    batch_size = int(config["evaluation"]["batch_sizes"][str(checkpoint["width"])])
    rows = []
    timings = {}
    current, timings["validation"] = evaluate_documents(
        model, validation, dataset="validation", batch_size=batch_size, device=device,
        bf16=bool(config["runtime"]["bf16"]),
    )
    rows.extend(current)
    current, timings["confirmation"] = evaluate_documents(
        model, confirmation, dataset="confirmation", batch_size=batch_size, device=device,
        bf16=bool(config["runtime"]["bf16"]),
    )
    rows.extend(current)
    interventions = {}
    if isinstance(model, SpecialistDecoderLM):
        base_confirmation = [
            document for document in confirmation if document.metadata.get("variant") == "base"
        ]
        for intervention in ("ablate", "wrong_pointer"):
            intervention_rows, intervention_timing = evaluate_documents(
                model, base_confirmation, dataset="confirmation", batch_size=batch_size,
                device=device, bf16=bool(config["runtime"]["bf16"]),
                intervention=intervention,
            )
            interventions[intervention] = {
                "slices": aggregate_slices(intervention_rows),
                "timing": intervention_timing,
            }
    online = benchmark_online(
        model, confirmation, device=device, bf16=bool(config["runtime"]["bf16"]),
        repeats=int(config["evaluation"]["benchmark_repeats"]),
    )
    with gzip.open(rows_path, "wt", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    result = {
        "schema": 1,
        "config_sha256": config["_config_sha256"],
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": file_hash(checkpoint_path),
        "rows": str(rows_path.resolve()),
        "rows_sha256": file_hash(rows_path),
        "arm": checkpoint["arm"],
        "layers": checkpoint["layers"],
        "width": checkpoint["width"],
        "seed": checkpoint["seed"],
        "slices": aggregate_slices(rows),
        "invariance": invariance_metrics(rows),
        "interventions": interventions,
        "timing": timings,
        "online_benchmark": online,
    }
    _write_json(output_path, result)
    return result


def evaluate_all(config: dict, run_dir: str | Path, *, force: bool = False) -> list[dict]:
    run_dir = Path(run_dir)
    results = []
    for layers, width in config["backbones"]["sizes"]:
        for seed in config["seeds"]:
            for arm in config["backbones"]["arms"]:
                checkpoint = run_dir / "backbones" / f"l{layers}_w{width}" / arm / f"seed_{seed}" / "checkpoint.pt"
                if not checkpoint.exists():
                    continue
                results.append(evaluate_checkpoint(config, run_dir, checkpoint, force=force))
    return results


def load_evaluation_rows(path: str | Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]

"""Disjoint corpus construction, caching, and deterministic paired batching."""

from __future__ import annotations

import gzip
import hashlib
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch

from .config import canonical_json, file_hash
from .language import (
    Document,
    Vocabulary,
    generate_document,
    rename_document,
    whitespace_document,
)
from .reference import ReferenceInterpreter
from .specialist import CausalStructureMachine


CORPUS_NAMES = (
    "specialist_train",
    "specialist_validation",
    "backbone_train",
    "backbone_validation",
    "confirmation",
)


def _sequence_hash(document: Document) -> str:
    payload = ",".join(str(value) for value in document.token_ids)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _is_heldout_combo(document: Document) -> bool:
    meta = document.metadata
    return (
        meta["depth"] == 4
        and meta["query_kind"] == "xor"
        and meta["whitespace"] == "newlines"
        and meta["shadow_count"] >= 2
    )


def _validate_document(document: Document) -> None:
    reference = ReferenceInterpreter().interpret(document.text)
    if reference.answer != document.answer or reference.answer_position != document.answer_position:
        raise AssertionError(f"reference disagreement for {document.document_id}")
    machine = CausalStructureMachine().analyze(document.token_ids)
    expected = {binding.use_position: binding.declaration_position for binding in reference.bindings}
    actual = {position: pointer for position, pointer in enumerate(machine.pointers) if pointer >= 0}
    if expected != actual:
        raise AssertionError(f"programmed binding disagreement for {document.document_id}")
    if not all(machine.validity):
        raise AssertionError(f"valid generator output rejected for {document.document_id}")


def _generate_unique(
    *,
    count: int,
    seed: int,
    split: str,
    literal_fraction: float,
    depth_range: tuple[int, int],
    history_range: tuple[int, int],
    seen: set[str],
    force_heldout_combo: bool = False,
    exclude_heldout_combo: bool = False,
) -> list[Document]:
    documents: list[Document] = []
    index = 0
    literal_target = round(count * literal_fraction)
    max_attempts = max(10_000, count * 100)
    while len(documents) < count and index < max_attempts:
        document = generate_document(
            seed,
            index,
            split=split,
            literal_fraction=literal_fraction,
            depth_range=depth_range,
            history_range=history_range,
            force_heldout_combo=force_heldout_combo,
            force_literal=(len(documents) < literal_target) if not force_heldout_combo else False,
        )
        index += 1
        fingerprint = _sequence_hash(document)
        if fingerprint in seen:
            continue
        if exclude_heldout_combo and _is_heldout_combo(document):
            continue
        _validate_document(document)
        seen.add(fingerprint)
        documents.append(document)
    if len(documents) != count:
        raise RuntimeError(f"could only generate {len(documents)} unique {split} documents")
    return documents


def _write_corpus(path: Path, documents: list[Document]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as handle:
        for document in documents:
            handle.write(canonical_json(document.to_json()))
            handle.write("\n")


def load_corpus(path: str | Path) -> list[Document]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [Document.from_json(json.loads(line)) for line in handle if line.strip()]


def prepare_corpora(config: dict, run_dir: str | Path, *, force: bool = False) -> dict:
    """Build all splits once and record exact overlap/preprocessing/cache costs."""
    started = time.perf_counter()
    run_dir = Path(run_dir)
    cache_dir = run_dir / "data"
    manifest_path = cache_dir / "manifest.json"
    if manifest_path.exists() and not force:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("config_sha256") == config["_config_sha256"]:
            return manifest

    counts = config["data"]["documents"]
    literal_fraction = float(config["data"]["literal_fraction"])
    seen: set[str] = set()
    corpora: dict[str, list[Document]] = {}
    specs = {
        "specialist_train": (101, (1, 4), (4, 12)),
        "specialist_validation": (202, (1, 4), (4, 14)),
        "backbone_train": (303, (1, 4), (4, 12)),
        "backbone_validation": (404, (1, 4), (4, 14)),
    }
    for name, (seed, depths, histories) in specs.items():
        corpora[name] = _generate_unique(
            count=int(counts[name]), seed=seed, split=name,
            literal_fraction=literal_fraction, depth_range=depths, history_range=histories,
            seen=seen, exclude_heldout_combo=True,
        )

    confirmation_count = int(counts["confirmation"])
    quarters = [confirmation_count // 4] * 4
    for index in range(confirmation_count % 4):
        quarters[index] += 1
    standard = _generate_unique(
        count=quarters[0], seed=505, split="confirmation_standard",
        literal_fraction=literal_fraction, depth_range=(1, 4), history_range=(4, 14),
        seen=seen, exclude_heldout_combo=True,
    )
    depth_ood = _generate_unique(
        count=quarters[1], seed=606, split="confirmation_depth_ood",
        literal_fraction=literal_fraction, depth_range=(5, 8), history_range=(8, 18),
        seen=seen,
    )
    long_history = _generate_unique(
        count=quarters[2], seed=707, split="confirmation_long_history",
        literal_fraction=literal_fraction, depth_range=(1, 4), history_range=(18, 32),
        seen=seen, exclude_heldout_combo=True,
    )
    heldout = _generate_unique(
        count=quarters[3], seed=808, split="confirmation_heldout_combo",
        literal_fraction=0.0, depth_range=(4, 4), history_range=(8, 16), seen=seen,
        force_heldout_combo=True,
    )
    for document in depth_ood:
        document.metadata["depth_ood"] = True
    for document in long_history:
        document.metadata["long_history"] = True
    confirmation = standard + depth_ood + long_history + heldout

    pair_count = min(int(config["data"]["invariance_pairs"]), len(confirmation))
    for document in confirmation[:pair_count]:
        if not document.program.literal:
            renamed = rename_document(document)
            _validate_document(renamed)
            confirmation.append(renamed)
        for style in ("spaces", "tabs", "newlines", "mixed"):
            if style != document.metadata["whitespace"]:
                variant = whitespace_document(document, style)
                _validate_document(variant)
                confirmation.append(variant)
    corpora["confirmation"] = confirmation

    files: dict[str, dict] = {}
    primary_hashes: dict[str, set[str]] = {}
    for name, documents in corpora.items():
        path = cache_dir / f"{name}.jsonl.gz"
        _write_corpus(path, documents)
        primary = [document for document in documents if document.metadata.get("variant") == "base"]
        hashes = {_sequence_hash(document) for document in primary}
        primary_hashes[name] = hashes
        files[name] = {
            "path": str(path.resolve()),
            "sha256": file_hash(path),
            "documents": len(documents),
            "base_documents": len(primary),
            "bytes": path.stat().st_size,
            "tokens": sum(len(document.token_ids) for document in documents),
        }
    overlaps: dict[str, int] = {}
    names = list(primary_hashes)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1:]:
            overlaps[f"{left}__{right}"] = len(primary_hashes[left] & primary_hashes[right])
    if any(overlaps.values()):
        raise AssertionError(f"split overlap detected: {overlaps}")

    manifest = {
        "schema": 1,
        "config_sha256": config["_config_sha256"],
        "vocabulary_sha256": Vocabulary.hash(),
        "files": files,
        "exact_token_sequence_overlaps": overlaps,
        "heldout_rule": "depth=4 AND xor AND newline whitespace AND shadow_count>=2",
        "training_depths": [1, 2, 3, 4],
        "confirmation_depths": [5, 6, 7, 8],
        "preprocessing_seconds": time.perf_counter() - started,
        "cache_bytes": sum(item["bytes"] for item in files.values()),
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
    return manifest


@dataclass
class Batch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    events: torch.Tensor
    depths: torch.Tensor
    validity: torch.Tensor
    pointers: torch.Tensor
    alternate_pointers: torch.Tensor
    answer_positions: torch.Tensor
    documents: list[Document]

    def to(self, device: torch.device | str) -> "Batch":
        return Batch(
            input_ids=self.input_ids.to(device),
            attention_mask=self.attention_mask.to(device),
            events=self.events.to(device),
            depths=self.depths.to(device),
            validity=self.validity.to(device),
            pointers=self.pointers.to(device),
            alternate_pointers=self.alternate_pointers.to(device),
            answer_positions=self.answer_positions.to(device),
            documents=self.documents,
        )


def collate_documents(
    documents: list[Document],
    *,
    corrupt_fraction: float = 0.0,
    corruption_seed: int = 0,
) -> Batch:
    maximum = max(len(document.token_ids) for document in documents)
    shape = (len(documents), maximum)
    input_ids = torch.full(shape, 0, dtype=torch.long)
    attention = torch.zeros(shape, dtype=torch.bool)
    events = torch.zeros(shape, dtype=torch.long)
    depths = torch.zeros(shape, dtype=torch.long)
    validity = torch.zeros(shape, dtype=torch.long)
    pointers = torch.full(shape, -1, dtype=torch.long)
    alternates = torch.full(shape, -1, dtype=torch.long)
    answers = torch.zeros(len(documents), dtype=torch.long)
    rng = random.Random(corruption_seed)
    machine = CausalStructureMachine()
    for row, document in enumerate(documents):
        ids = list(document.token_ids)
        if corrupt_fraction and rng.random() < corrupt_fraction and len(ids) > 5:
            candidates = list(range(2, max(3, document.answer_position - 1)))
            ids[rng.choice(candidates)] = Vocabulary.TO_ID["}"]
        trace = machine.analyze(ids)
        length = len(ids)
        input_ids[row, :length] = torch.tensor(ids)
        attention[row, :length] = True
        events[row, :length] = torch.tensor(trace.events)
        depths[row, :length] = torch.tensor(trace.depths)
        validity[row, :length] = torch.tensor(trace.validity)
        pointers[row, :length] = torch.tensor(trace.pointers)
        alternates[row, :length] = torch.tensor(trace.alternate_pointers)
        answers[row] = document.answer_position
    return Batch(input_ids, attention, events, depths, validity, pointers, alternates,
                 answers, documents)


def paired_batches(
    documents: list[Document],
    *,
    batch_size: int,
    seed: int,
) -> Iterator[list[Document]]:
    """Infinite deterministic schedule shared by every arm of a seed."""
    epoch = 0
    while True:
        order = list(range(len(documents)))
        random.Random(seed + epoch * 1009).shuffle(order)
        for offset in range(0, len(order), batch_size):
            yield [documents[index] for index in order[offset:offset + batch_size]]
        epoch += 1

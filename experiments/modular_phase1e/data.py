"""Fresh program-family splits for Phase 1e."""

from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from experiments.modular_phase1.config import canonical_json, file_hash
from experiments.modular_phase1.data import load_corpus
from experiments.modular_phase1.language import (
    Document,
    generate_document,
    rename_document,
    whitespace_document,
)


@dataclass(frozen=True)
class FamilyDocument:
    family_id: str
    split: str
    variant: str
    document: Document


def _canonical(document: Document) -> str:
    return hashlib.sha256(document.program.canonical().encode()).hexdigest()


def _old_programs(source_run: Path) -> set[str]:
    manifest = json.loads((source_run / "data" / "manifest.json").read_text())
    result = set()
    for record in manifest["files"].values():
        for document in load_corpus(record["path"]):
            result.add(_canonical(document))
    return result


def _families(
    count: int,
    *,
    split: str,
    seed: int,
    depth_range: tuple[int, int],
    history_range: tuple[int, int],
    seen: set[str],
) -> list[FamilyDocument]:
    output = []
    index = 0
    styles = ("spaces", "tabs", "newlines", "mixed")
    while len(output) // 3 < count:
        document = generate_document(
            seed, index, split=split, force_literal=False,
            depth_range=depth_range, history_range=history_range,
        )
        index += 1
        if document.metadata["query_kind"] != "lookup":
            continue
        canonical = _canonical(document)
        if canonical in seen:
            continue
        seen.add(canonical)
        family_id = f"{split}:{canonical[:20]}"
        renamed = rename_document(document)
        whitespace_style = next(
            style for style in styles if style != document.metadata["whitespace"]
        )
        whitespace = whitespace_document(document, whitespace_style)
        for variant, member in (
            ("base", document), ("renamed", renamed), ("whitespace", whitespace),
        ):
            output.append(FamilyDocument(family_id, split, variant, member))
    return output


def prepare_phase1e_data(config: dict, source_run: Path, run_dir: Path) -> tuple[dict, dict]:
    seen = _old_programs(source_run)
    old_count = len(seen)
    settings = config["data"]
    splits = {
        "train": _families(
            int(settings["train_families"]), split="phase1e_train", seed=18011,
            depth_range=(1, 4), history_range=(4, 12), seen=seen,
        ),
        "calibration": _families(
            int(settings["calibration_families"]), split="phase1e_calibration",
            seed=18022, depth_range=(1, 4), history_range=(4, 14), seen=seen,
        ),
    }
    confirmation_count = int(settings["confirmation_families"])
    counts = [confirmation_count // 3] * 3
    for index in range(confirmation_count % 3):
        counts[index] += 1
    confirmation = []
    for count, name, seed, depths, histories in (
        (counts[0], "standard", 18033, (1, 4), (4, 14)),
        (counts[1], "depth_ood", 18044, (5, 8), (8, 18)),
        (counts[2], "long_history", 18055, (1, 4), (18, 32)),
    ):
        members = _families(
            count, split=f"phase1e_confirmation_{name}", seed=seed,
            depth_range=depths, history_range=histories, seen=seen,
        )
        for member in members:
            member.document.metadata["confirmation_slice"] = name
        confirmation.extend(members)
    splits["confirmation"] = confirmation

    data_dir = run_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    split_families = {}
    for split, members in splits.items():
        path = data_dir / f"{split}.jsonl.gz"
        with gzip.open(path, "wt", encoding="utf-8", newline="\n") as handle:
            for member in members:
                record = member.document.to_json()
                record["phase1e_family_id"] = member.family_id
                record["phase1e_split"] = member.split
                record["phase1e_variant"] = member.variant
                handle.write(canonical_json(record) + "\n")
        families = {member.family_id for member in members}
        split_families[split] = families
        files[split] = {
            "path": str(path.resolve()),
            "sha256": file_hash(path),
            "families": len(families),
            "documents": len(members),
            "causal_targets": sum(len(member.document.token_ids) - 1 for member in members),
            "variants": {
                variant: sum(member.variant == variant for member in members)
                for variant in ("base", "renamed", "whitespace")
            },
        }
    overlap = {
        "train_calibration": len(split_families["train"] & split_families["calibration"]),
        "train_confirmation": len(split_families["train"] & split_families["confirmation"]),
        "calibration_confirmation": len(
            split_families["calibration"] & split_families["confirmation"]
        ),
    }
    if any(overlap.values()):
        raise AssertionError("program families cross Phase 1e splits")
    manifest = {
        "schema": 1,
        "files": files,
        "program_family_overlap": overlap,
        "excluded_prior_programs": old_count,
        "variants_grouped_with_family": True,
        "confirmation_evaluation_policy": "once, only after calibration selection",
    }
    (data_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return splits, manifest

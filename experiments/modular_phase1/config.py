"""Configuration loading, validation, and stable experiment identities."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


REQUIRED_SEEDS = [11, 22, 33]
REQUIRED_SIZES = [[2, 64], [4, 128], [6, 256]]
MAX_SPECIALIST_PARAMETERS = 100_000
MAX_SPECIALIST_TOKENS = 2_097_152
MAX_BACKBONE_TOKENS = 1_048_576
MAX_SPECIALIST_MINUTES = 30.0
MAX_PILOT_HOURS = 4.0


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def object_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _positive(config: dict, path: tuple[str, ...]) -> float:
    current: Any = config
    for key in path:
        current = current[key]
    if current <= 0:
        raise ValueError("%s must be positive" % ".".join(path))
    return float(current)


def validate_config(config: dict) -> None:
    if config.get("phase") != "bounded_modular_phase1":
        raise ValueError("config phase must be 'bounded_modular_phase1'")
    if config.get("seeds") != REQUIRED_SEEDS:
        raise ValueError(f"Phase 1 seeds are frozen to {REQUIRED_SEEDS}")
    if config.get("backbones", {}).get("sizes") != REQUIRED_SIZES:
        raise ValueError(f"Phase 1 backbone sizes are frozen to {REQUIRED_SIZES}")
    if config.get("backbones", {}).get("arms") != ["plain", "specialist", "matched"]:
        raise ValueError("backbone arms must be plain, specialist, matched in that order")

    specialist = config["specialist"]
    backbones = config["backbones"]
    budget = config["budget"]
    margins = config["evaluation"]["non_inferiority"]
    _positive(config, ("specialist", "embedding_dim"))
    _positive(config, ("specialist", "hidden_dim"))
    _positive(config, ("backbones", "sequence_length"))
    _positive(config, ("optimizer", "learning_rate"))
    if specialist["max_parameters"] > MAX_SPECIALIST_PARAMETERS:
        raise ValueError("specialist parameter cap exceeds the Phase 1 ceiling")
    if budget["specialist_tokens"] > MAX_SPECIALIST_TOKENS:
        raise ValueError("specialist token budget exceeds 2,097,152")
    if budget["backbone_tokens_per_run"] > MAX_BACKBONE_TOKENS:
        raise ValueError("backbone token budget exceeds 1,048,576 per run")
    if budget["specialist_accelerator_minutes"] > MAX_SPECIALIST_MINUTES:
        raise ValueError("specialist time budget exceeds 30 accelerator minutes")
    if budget["pilot_accelerator_hours"] > MAX_PILOT_HOURS:
        raise ValueError("pilot time budget exceeds four accelerator hours")
    if margins != {"accuracy_points": 0.01, "nll_nats": 0.02}:
        raise ValueError("Phase 1 non-inferiority margins are frozen at 0.01 and 0.02")
    if config["data"]["literal_fraction"] != 0.20:
        raise ValueError("literal_fraction is frozen at 20%")
    if backbones["attention_heads"] != {"64": 4, "128": 4, "256": 8}:
        raise ValueError("attention-head mapping differs from the frozen Phase 1 design")


def load_config(path: str | Path) -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    validate_config(config)
    config["_config_path"] = str(path.resolve())
    config["_config_sha256"] = file_hash(path)
    return config

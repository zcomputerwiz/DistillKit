# Copyright 2025 Arcee AI & DistillKit Contributors
"""Backward-compatibility shim for borrowed routing.

Moved to ``distillkit.experimental.borrowed_routing``.
"""

from __future__ import annotations

from distillkit.experimental.borrowed_routing import (
    SUBLAYERS,
    TENSORS,
    initialise_ple_reader,
    initialise_widened_residual,
    layer_map,
    load_manifest,
)

__all__ = [
    "SUBLAYERS",
    "TENSORS",
    "initialise_ple_reader",
    "initialise_widened_residual",
    "layer_map",
    "load_manifest",
]

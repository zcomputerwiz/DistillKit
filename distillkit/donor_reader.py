# Copyright 2025 Arcee AI & DistillKit Contributors
"""Backward-compatibility shim for donor reader transplant.

Moved to ``distillkit.experimental.donor_reader``.
"""

from __future__ import annotations

from distillkit.experimental.donor_reader import (
    DonorReaderTransplant,
    initialise_transplant_reader,
    load_reference_tensors,
)

__all__ = [
    "DonorReaderTransplant",
    "initialise_transplant_reader",
    "load_reference_tensors",
]

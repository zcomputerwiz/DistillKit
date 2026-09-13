# Copyright 2025 Arcee AI & DistillKit Contributors
"""Backward-compatibility shim for donor reader analysis.

Moved to ``distillkit.experimental.analyze_donor_reader``.
"""

from __future__ import annotations

from distillkit.experimental.analyze_donor_reader import (
    LAYOUT_TOKEN_IDS,
    TAP_LABELS,
    distribution,
    main,
)

__all__ = [
    "LAYOUT_TOKEN_IDS",
    "TAP_LABELS",
    "distribution",
    "main",
]

if __name__ == "__main__":
    main()

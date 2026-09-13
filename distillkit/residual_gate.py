# Copyright 2025 Arcee AI & DistillKit Contributors
"""Backward-compatibility shim for the residual admission gate.

Moved to ``distillkit.experimental.residual_gate``.
"""

from __future__ import annotations

from distillkit.experimental.residual_gate import (
    FAMILIES,
    ResidualAdmissionGate,
    ResidualGateCheckpointCallback,
    ResidualGateHandle,
    TrigramFamiliarity,
    attach_residual_gates,
    calibrate_gates,
    family_features,
    gate_parameter_count,
    install_residual_gates,
    load_gate_checkpoint,
    remove_residual_gates,
    residual_gates,
)

__all__ = [
    "FAMILIES",
    "ResidualAdmissionGate",
    "ResidualGateCheckpointCallback",
    "ResidualGateHandle",
    "TrigramFamiliarity",
    "attach_residual_gates",
    "calibrate_gates",
    "family_features",
    "gate_parameter_count",
    "install_residual_gates",
    "load_gate_checkpoint",
    "remove_residual_gates",
    "residual_gates",
]

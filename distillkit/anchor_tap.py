"""Backward compatibility shim for distillkit.core.anchor_tap."""

from distillkit.core.anchor_tap import (
    AnchorTap,
    CapturedStates,
    anchor_module,
)

__all__ = ["AnchorTap", "CapturedStates", "anchor_module"]

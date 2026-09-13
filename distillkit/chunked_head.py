"""Backward compatibility shim for distillkit.core.chunked_head."""

from distillkit.core.chunked_head import (
    HeadContext,
    chunked_head_loss,
    head_device,
)

__all__ = ["HeadContext", "chunked_head_loss", "head_device"]

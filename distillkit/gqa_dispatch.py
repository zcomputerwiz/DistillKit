"""Expand grouped-query KV heads when no fused kernel can broadcast them.

Qwen3.5 has 16 query heads against 4 key/value heads, and transformers hands that
asymmetry straight to ``scaled_dot_product_attention`` with ``enable_gqa=True``
rather than materializing the repeated heads. ``use_gqa_in_sdpa`` decides this on
two conditions -- no attention mask, and head_dim <= 256 -- whose stated purpose is
to keep SDPA off the math kernel.

That reasoning holds where a fused kernel implements the broadcast. This PyTorch
build does not have one: it reports "Torch was not compiled with flash attention",
and the memory-efficient kernel refuses unequal head counts outright ("both fused
kernels require query, key and value to have the same num_heads"). So the flag
intended to avoid the math kernel is exactly what selects it, and at sequence 4096
that costs, per attention call:

    enable_gqa=True (math kernel)   2728 MiB forward, 4328 MiB with backward
    KV expanded to 16 heads          96 MiB forward,  249 MiB with backward

Expanding costs about 50 MB of retained key/value per call and saves roughly 4 GiB
of transient. Qwen3.5 runs 8 full-attention layers, so this is the largest single
allocation left in the step.

The patch is conditional: it probes whether a fused kernel actually accepts a
broadcast GQA call and only forces expansion when none does, so a build that gains
flash attention keeps the upstream behaviour.
"""

from __future__ import annotations

import logging
import os

import torch

LOG = logging.getLogger(__name__)

_installed = False


def fused_kernel_supports_gqa() -> bool:
    """True if some fused SDPA backend accepts unequal query and key head counts.

    Probed rather than assumed: it depends on the build's kernels, not the model.
    """
    # On Windows, querying a busy/misbehaving driver can itself crash the process.
    # Respect an explicit CPU-only environment before touching the CUDA runtime.
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None and visible.strip().lower() in ("", "-1", "none"):
        return True
    if not torch.cuda.is_available():
        return True  # nothing to fix; leave upstream behaviour alone
    from torch.nn.attention import SDPBackend, sdpa_kernel

    query = torch.zeros(1, 4, 8, 32, device="cuda", dtype=torch.bfloat16)
    key = torch.zeros(1, 2, 8, 32, device="cuda", dtype=torch.bfloat16)
    for backend in (SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION):
        try:
            with sdpa_kernel(backend):
                torch.nn.functional.scaled_dot_product_attention(
                    query, key, key, is_causal=True, enable_gqa=True
                )
            return True
        except RuntimeError:
            continue
    return False


def install_expanded_gqa_attention(force: bool | None = None) -> bool:
    """Make transformers repeat KV heads instead of asking SDPA to broadcast them.

    Returns whether the patch was applied. Idempotent.
    """
    global _installed
    if _installed:
        return True
    try:
        from transformers.integrations import sdpa_attention
    except ImportError:  # pragma: no cover - transformers is a hard dependency
        return False
    if not hasattr(sdpa_attention, "use_gqa_in_sdpa"):
        return False

    should_patch = (not fused_kernel_supports_gqa()) if force is None else force
    if not should_patch:
        return False

    original = sdpa_attention.use_gqa_in_sdpa

    def expand_instead(attention_mask, key, value):
        # Returning False sends transformers down its repeat_kv path.
        return False

    expand_instead.__wrapped__ = original
    sdpa_attention.use_gqa_in_sdpa = expand_instead
    _installed = True
    LOG.info(
        "No fused SDPA kernel broadcasts grouped-query heads on this build; "
        "expanding KV heads instead of using enable_gqa (avoids the math kernel)."
    )
    return True

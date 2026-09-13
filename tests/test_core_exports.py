"""Verify export parity and compatibility between distillkit.core and legacy top-level shims."""

import distillkit.core as core
import distillkit.anchor_tap as legacy_anchor_tap
import distillkit.chunked_ce as legacy_chunked_ce
import distillkit.chunked_head as legacy_chunked_head
import distillkit.frozen_prefix as legacy_frozen_prefix
import distillkit.sortish_sampler as legacy_sortish_sampler

import distillkit.core.anchor_tap as core_anchor_tap
import distillkit.core.chunked_ce as core_chunked_ce
import distillkit.core.chunked_head as core_chunked_head
import distillkit.core.frozen_prefix as core_frozen_prefix
import distillkit.core.sortish_sampler as core_sortish_sampler


def test_anchor_tap_exports():
    assert core.AnchorTap is core_anchor_tap.AnchorTap is legacy_anchor_tap.AnchorTap
    assert core.CapturedStates is core_anchor_tap.CapturedStates is legacy_anchor_tap.CapturedStates
    assert core.anchor_module is core_anchor_tap.anchor_module is legacy_anchor_tap.anchor_module


def test_chunked_head_exports():
    assert core.HeadContext is core_chunked_head.HeadContext is legacy_chunked_head.HeadContext
    assert core.chunked_head_loss is core_chunked_head.chunked_head_loss is legacy_chunked_head.chunked_head_loss
    assert core.head_device is core_chunked_head.head_device is legacy_chunked_head.head_device


def test_chunked_ce_exports():
    assert core.chunked_causal_lm_loss is core_chunked_ce.chunked_causal_lm_loss is legacy_chunked_ce.chunked_causal_lm_loss
    assert core.DEFAULT_CHUNK_BYTES == core_chunked_ce.DEFAULT_CHUNK_BYTES == legacy_chunked_ce.DEFAULT_CHUNK_BYTES
    assert core.chunk_tokens_for is core_chunked_ce.chunk_tokens_for is legacy_chunked_ce.chunk_tokens_for
    assert core.keep_bf16_forward_outputs is core_chunked_ce.keep_bf16_forward_outputs is legacy_chunked_ce.keep_bf16_forward_outputs
    assert core.maybe_install_chunked_loss is core_chunked_ce.maybe_install_chunked_loss is legacy_chunked_ce.maybe_install_chunked_loss


def test_frozen_prefix_exports():
    assert core.no_grad_prefix is core_frozen_prefix.no_grad_prefix is legacy_frozen_prefix.no_grad_prefix


def test_sortish_sampler_exports():
    assert core.DEFAULT_SORT_WINDOW == core_sortish_sampler.DEFAULT_SORT_WINDOW == legacy_sortish_sampler.DEFAULT_SORT_WINDOW
    assert core.SortishSampler is core_sortish_sampler.SortishSampler is legacy_sortish_sampler.SortishSampler
    assert core.sortish_indices is core_sortish_sampler.sortish_indices is legacy_sortish_sampler.sortish_indices

"""The channel arithmetic a head-wise GatedDeltaNet split depends on.

`conv_dim` packs `[all Q | all K | all V]`, so the obvious contiguous half-split of
the channel axis is wrong: it hands one rank every query and half the keys. It also
runs, and trains nonsense, which is why the correct slices are pinned here against
values derived independently from the config.
"""

import pytest
import torch
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

from distillkit.tp_gated_delta import conv_channel_plan, head_group_plan


def _student_config():
    """The real 4B student's linear-attention geometry."""
    return Qwen3_5TextConfig(
        linear_num_key_heads=16, linear_num_value_heads=32,
        linear_key_head_dim=128, linear_value_head_dim=128,
    )


def test_channels_match_the_three_block_packing():
    """key_dim 2048, value_dim 4096, conv_dim 8192 -> three disjoint runs per rank."""
    plan = conv_channel_plan(_student_config(), 2)
    rank0, rank1 = plan.channels

    expected0 = torch.cat([
        torch.arange(0, 1024),        # queries
        torch.arange(2048, 3072),     # keys
        torch.arange(4096, 6144),     # values
    ])
    expected1 = torch.cat([
        torch.arange(1024, 2048),
        torch.arange(3072, 4096),
        torch.arange(6144, 8192),
    ])
    assert torch.equal(rank0, expected0)
    assert torch.equal(rank1, expected1)


def test_the_naive_contiguous_split_is_different():
    """Guards the whole point: a contiguous half-split is not this.

    If someone 'simplifies' conv_channel_plan into `torch.arange(conv_dim).chunk(2)`,
    this fails instead of the model silently learning the wrong function.
    """
    plan = conv_channel_plan(_student_config(), 2)
    naive = torch.arange(8192).chunk(2)
    assert not torch.equal(plan.channels[0], naive[0])
    # The naive rank 0 takes all 2048 queries; the correct one takes half.
    assert (plan.channels[0] < 2048).sum().item() == 1024
    assert (naive[0] < 2048).sum().item() == 2048


def test_every_channel_is_owned_exactly_once():
    plan = conv_channel_plan(_student_config(), 2)
    combined = torch.cat(plan.channels).sort().values
    assert torch.equal(combined, torch.arange(8192))


def test_per_rank_metadata_follows_the_split():
    """The forward splits its projection by key_dim/value_dim; a shard's must shrink."""
    plan = conv_channel_plan(_student_config(), 2)
    assert plan.key_dim == 1024 and plan.value_dim == 2048
    assert plan.num_k_heads == 8 and plan.num_v_heads == 16
    assert len(plan.channels[0]) == 2 * plan.key_dim + plan.value_dim


def test_head_groups_stay_together():
    """repeat_interleave(2) ties Q/K head j to value heads 2j and 2j+1."""
    groups = head_group_plan(_student_config(), 2)
    (keys0, values0), (keys1, values1) = groups
    assert list(keys0) == list(range(0, 8)) and list(values0) == list(range(0, 16))
    assert list(keys1) == list(range(8, 16)) and list(values1) == list(range(16, 32))
    for keys, values in groups:
        # Each owned key head's two value heads must be owned by the same rank.
        for head in keys:
            assert 2 * head in values and 2 * head + 1 in values


def test_indivisible_counts_are_rejected():
    config = _student_config()
    config.linear_num_key_heads = 3
    with pytest.raises(ValueError, match="evenly"):
        conv_channel_plan(config, 2)


def test_value_heads_must_be_a_multiple_of_key_heads():
    config = _student_config()
    config.linear_num_value_heads = 34  # not a whole multiple of 16
    with pytest.raises(ValueError, match="whole multiple"):
        conv_channel_plan(config, 2)

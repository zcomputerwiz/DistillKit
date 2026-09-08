"""Head-wise sharding for Qwen3.5's GatedDeltaNet (the 24 linear_attention layers).

Upstream's ``base_model_tp_plan`` marks every ``linear_attn`` projection
``colwise_gather_output``: shard the projection, gather the full tensor back, and run
the delta-rule recurrence redundantly on every rank. That caps tensor parallelism at
63.9% of parameters. Splitting by head instead covers ~84%.

An adversarial review (2026-09-07) confirmed the head decomposition is algebraically
exact -- the only cross-head mixing is the bias-free ``out_proj``, so splitting its
input columns gives ``Y = O_0 W_0^T + O_1 W_1^T`` and one all-reduce finishes the
layer. It also corrected the mechanism, which is what this module exists to get right.

**The channel layout is not per-head grouped.** ``conv_dim`` packs
``[all Q | all K | all V]``, so a contiguous half-slice of the channel axis takes the
wrong Q/K/V mix -- it would give one rank every query and half the keys, run without
error, and train nonsense. Each of the three blocks must be sliced separately and
reconcatenated. For the 4B student (``key_dim`` 2048, ``value_dim`` 4096,
``conv_dim`` 8192):

    rank 0:  Q [0:1024]     K [2048:3072]   V [4096:6144]
    rank 1:  Q [1024:2048]  K [3072:4096]   V [6144:8192]

The same slices apply to ``in_proj_qkv``'s output rows and to the convolution's
filters and cache channels, and each shard's ``key_dim``/``value_dim`` metadata (used
by the forward's ``torch.split``) must be updated to match.

Three further constraints from that review, each of which fails silently:

* ``RMSNormGated``'s weight is **one** ``head_v_dim`` parameter shared by every head,
  not per-head. Replicate it whole on both ranks; in training its gradient is a
  partial on each rank and must be all-reduced.
* Key and value heads are coupled by ``repeat_interleave(2, dim=2)``: Q/K head ``j``
  feeds value heads ``2j`` and ``2j+1``. A rank must own whole groups -- Q/K 0-7 with
  V 0-15, Q/K 8-15 with V 16-31 -- and after the repeat each shard hands FLA 16 Q,
  16 K and 16 V heads, not 8 and 16.
* The recurrent state cache is written per ``layer_idx``. Both ranks would write the
  same slot, so each needs its own cache object or their states alias.

``in_proj_z`` / ``in_proj_b`` / ``in_proj_a`` all read the full hidden vector, so the
input is replicated and their gradients contribute to the same input gradient --
handled by ``Replicate``'s backward, which sums across devices.

FLA's kernels take head counts from tensor shapes and impose no power-of-two
requirement; the real limit found was ``K <= 256``, satisfied by ``head_k_dim`` 128.
One scope caveat from the review: transformers can prefer a Hub kernel over the
installed ``fla`` package, and the dispatch wrapper does not certify which is bound.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from distillkit.tp_linear import split_sizes


@dataclass(frozen=True)
class ConvChannelPlan:
    """Which conv/projection channels each rank owns, and its resulting metadata."""

    channels: tuple[torch.Tensor, ...]
    key_dim: int
    value_dim: int
    num_k_heads: int
    num_v_heads: int


def conv_channel_plan(config, parts: int) -> ConvChannelPlan:
    """Per-rank channel indices into the ``[all Q | all K | all V]`` packing.

    Returns index tensors rather than slices because a rank's channels are three
    disjoint runs, not one. Feed them to ``index_select`` on ``in_proj_qkv.weight``
    rows, ``conv1d.weight``, and the convolution cache's channel axis.
    """
    key_dim = config.linear_num_key_heads * config.linear_key_head_dim
    value_dim = config.linear_num_value_heads * config.linear_value_head_dim
    # Fail rather than reshape the maths: every count here divides by two for the
    # 4B student, and an uneven split would still run.
    split_sizes(config.linear_num_key_heads, parts)
    split_sizes(config.linear_num_value_heads, parts)
    if config.linear_num_value_heads % config.linear_num_key_heads:
        raise ValueError(
            "value heads must be a whole multiple of key heads; the split has to keep "
            "each key head with the value heads repeat_interleave assigns it"
        )

    key_per_rank = key_dim // parts
    value_per_rank = value_dim // parts
    channels = []
    for rank in range(parts):
        queries = torch.arange(rank * key_per_rank, (rank + 1) * key_per_rank)
        keys = key_dim + queries
        values = torch.arange(
            2 * key_dim + rank * value_per_rank,
            2 * key_dim + (rank + 1) * value_per_rank,
        )
        channels.append(torch.cat([queries, keys, values]))
    return ConvChannelPlan(
        channels=tuple(channels),
        key_dim=key_per_rank,
        value_dim=value_per_rank,
        num_k_heads=config.linear_num_key_heads // parts,
        num_v_heads=config.linear_num_value_heads // parts,
    )


def head_group_plan(config, parts: int) -> tuple[tuple[range, range], ...]:
    """Which key heads and value heads each rank owns, kept in their groups.

    ``repeat_interleave(2, dim=2)`` ties Q/K head ``j`` to value heads ``2j``,
    ``2j+1``. Splitting a group would force replicating or communicating its Q/K head
    every step, so ranks take contiguous group-aligned ranges.
    """
    per_key = config.linear_num_key_heads // parts
    per_value = config.linear_num_value_heads // parts
    return tuple(
        (
            range(rank * per_key, (rank + 1) * per_key),
            range(rank * per_value, (rank + 1) * per_value),
        )
        for rank in range(parts)
    )

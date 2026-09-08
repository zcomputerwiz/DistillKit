"""Head-sharded GatedDeltaNet, built on the channel plan in ``tp_gated_delta``.

Covers the 24 ``linear_attention`` layers that upstream's plan leaves replicated,
taking tensor parallelism from 63.9% of parameters to ~84%.

Scope: **training only, no cache.** The recurrent and convolution states are keyed by
``layer_idx``, so two ranks would write the same slot and alias each other's state.
Training runs with ``use_cache=False``, so rather than invent a per-rank cache that
nothing exercises, this raises on any ``cache_params``. Generation keeps the stock
module.

What is sharded, and what is not:

* ``in_proj_qkv`` output rows, ``conv1d`` filters -- by the three-block channel plan,
  because the packing is ``[all Q | all K | all V]`` and a contiguous split takes the
  wrong mix.
* ``in_proj_z`` (one contiguous ``value_dim`` block), ``in_proj_b``/``in_proj_a``, and
  the per-head ``A_log``/``dt_bias`` -- by value head.
* ``out_proj`` -- row-parallel, one all-reduce finishing the layer.
* ``norm`` -- **replicated**. Its weight is a single ``head_v_dim`` parameter shared by
  every head, so it cannot be split. Each rank then computes a *partial* gradient for
  it, and those must be summed before the optimizer step; see
  ``replicated_parameters`` and ``sync_replicated_gradients``. Skipping that does not
  raise, it just trains the norm against one rank's share of its gradient.
"""

from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
from torch import nn

from distillkit.tensor_parallel import all_reduce, replicate
from distillkit.tp_gated_delta import conv_channel_plan
from distillkit.tp_linear import RowParallelLinear


def _slice_linear(source: nn.Linear, rows: torch.Tensor, device) -> nn.Linear:
    """A linear holding only the given output rows, on the given device."""
    # The plan's indices are built on CPU; index_select needs them beside the weight,
    # which shard_model has already moved to the home card.
    rows = rows.to(source.weight.device)
    shard = nn.Linear(source.in_features, len(rows), bias=source.bias is not None)
    with torch.no_grad():
        shard.weight.copy_(source.weight.data.index_select(0, rows))
        if source.bias is not None:
            shard.bias.copy_(source.bias.data.index_select(0, rows))
    return shard.to(device)


class TensorParallelGatedDeltaNet(nn.Module):
    def __init__(self, source: nn.Module, devices):
        super().__init__()
        self.devices = [torch.device(d) for d in devices]
        parts = len(self.devices)
        config = source.config if hasattr(source, "config") else None
        plan = conv_channel_plan(_geometry(source), parts)
        self.plan = plan
        self.head_k_dim = source.head_k_dim
        self.head_v_dim = source.head_v_dim
        self.conv_kernel_size = source.conv_kernel_size
        self.activation = source.activation
        self.layer_idx = source.layer_idx
        self.num_k_heads = plan.num_k_heads
        self.num_v_heads = plan.num_v_heads
        self.key_dim = plan.key_dim
        self.value_dim = plan.value_dim

        value_heads = source.num_v_heads // parts
        qkv, z_proj, b_proj, a_proj, convs, norms, a_logs, dts = ([] for _ in range(8))
        for rank, device in enumerate(self.devices):
            channels = plan.channels[rank]
            heads = torch.arange(rank * value_heads, (rank + 1) * value_heads)
            value_channels = torch.arange(
                rank * plan.value_dim, (rank + 1) * plan.value_dim
            )
            qkv.append(_slice_linear(source.in_proj_qkv, channels, device))
            z_proj.append(_slice_linear(source.in_proj_z, value_channels, device))
            b_proj.append(_slice_linear(source.in_proj_b, heads, device))
            a_proj.append(_slice_linear(source.in_proj_a, heads, device))
            convs.append(_slice_conv(source.conv1d, channels, device))
            # Replicated: one shared head_v_dim weight, not per head.
            norms.append(copy.deepcopy(source.norm).to(device))
            local_heads = heads.to(source.A_log.device)
            a_logs.append(nn.Parameter(source.A_log.data.index_select(0, local_heads).clone().to(device)))
            dts.append(nn.Parameter(source.dt_bias.data.index_select(0, local_heads).clone().to(device)))

        self.in_proj_qkv = nn.ModuleList(qkv)
        self.in_proj_z = nn.ModuleList(z_proj)
        self.in_proj_b = nn.ModuleList(b_proj)
        self.in_proj_a = nn.ModuleList(a_proj)
        self.conv1d = nn.ModuleList(convs)
        self.norm = nn.ModuleList(norms)
        self.A_log = nn.ParameterList(a_logs)
        self.dt_bias = nn.ParameterList(dts)
        self.out_proj = RowParallelLinear(source.out_proj, self.devices)

    def replicated_parameters(self):
        """Parameters held whole on every rank, whose gradients are partials.

        Only the gated norm: everything else is sharded, so its gradient is already
        complete on the rank that owns it.
        """
        return [list(norm.parameters()) for norm in self.norm]

    def forward(self, hidden_states, cache_params=None, attention_mask=None, **kwargs):
        from transformers.models.qwen3_5.modeling_qwen3_5 import (
            apply_mask_to_padding_states,
            causal_conv1d_fn,
            torch_chunk_gated_delta_rule,
        )

        if cache_params is not None:
            raise NotImplementedError(
                "sharded GatedDeltaNet is training-only: conv and recurrent states are "
                "keyed by layer_idx, so both ranks would write the same slot and alias"
            )
        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
        batch_size, seq_len, _ = hidden_states.shape
        copies = replicate(hidden_states, self.devices)

        outputs = []
        groups = self.num_v_heads // self.num_k_heads
        for rank, device in enumerate(self.devices):
            local = copies[rank]
            mixed = self.in_proj_qkv[rank](local).transpose(1, 2)
            mixed = causal_conv1d_fn(
                mixed,
                self.conv1d[rank].weight.squeeze(1),
                self.conv1d[rank].bias,
                activation=self.activation,
            ).transpose(1, 2)
            query, key, value = torch.split(
                mixed, [self.key_dim, self.key_dim, self.value_dim], dim=-1
            )
            query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
            key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
            value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

            beta = self.in_proj_b[rank](local).sigmoid()
            g = -self.A_log[rank].float().exp() * F.softplus(
                self.in_proj_a[rank](local).float() + self.dt_bias[rank]
            )
            if groups > 1:
                # After this each rank hands the kernel num_v_heads Q/K/V heads,
                # not num_k_heads -- the repeat happens per shard, as upstream does
                # it per model.
                query = query.repeat_interleave(groups, dim=2)
                key = key.repeat_interleave(groups, dim=2)

            core, _ = torch_chunk_gated_delta_rule(
                query, key, value, g=g, beta=beta,
                initial_state=None, output_final_state=False,
                use_qk_l2norm_in_kernel=True, cu_seqlens=None,
            )
            z = self.in_proj_z[rank](local).reshape(-1, self.head_v_dim)
            core = self.norm[rank](core.reshape(-1, self.head_v_dim), z)
            outputs.append(core.reshape(batch_size, seq_len, -1))
        return self.out_proj(outputs)[0]


def _slice_conv(source: nn.Conv1d, channels: torch.Tensor, device) -> nn.Conv1d:
    """Depthwise conv over a subset of channels; groups shrink with the channels."""
    channels = channels.to(source.weight.device)
    count = len(channels)
    shard = nn.Conv1d(
        count, count, kernel_size=source.kernel_size[0], groups=count,
        padding=source.padding[0], bias=source.bias is not None,
    )
    with torch.no_grad():
        shard.weight.copy_(source.weight.data.index_select(0, channels))
        if source.bias is not None:
            shard.bias.copy_(source.bias.data.index_select(0, channels))
    return shard.to(device)


def _geometry(source: nn.Module):
    """The four numbers the channel plan needs, read off the module itself."""

    class _Geometry:
        linear_num_key_heads = source.num_k_heads
        linear_num_value_heads = source.num_v_heads
        linear_key_head_dim = source.head_k_dim
        linear_value_head_dim = source.head_v_dim

    return _Geometry()


def sync_replicated_gradients(module: nn.Module) -> int:
    """All-reduce gradients of parameters replicated across ranks.

    A replicated parameter sees only its rank's share of the loss, so each holds a
    partial gradient. Summing them is what makes the replica equivalent to the
    unsharded parameter. Returns how many parameter groups were reduced.

    Call this after backward and before the optimizer step. Omitting it does not
    raise; the norm simply trains against a fraction of its gradient.
    """
    reduced = 0
    for child in module.modules():
        if not isinstance(child, TensorParallelGatedDeltaNet):
            continue
        for group in zip(*child.replicated_parameters()):
            grads = [p.grad for p in group]
            if any(g is None for g in grads):
                continue
            totals = all_reduce(grads)
            for parameter, total in zip(group, totals):
                parameter.grad = total.detach()
            reduced += 1
    return reduced

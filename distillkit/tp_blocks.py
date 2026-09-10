"""Tensor-parallel replacements for the Qwen3.5 blocks whose sharding is unambiguous.

Covers the parts transformers' own ``base_model_tp_plan`` marks ``colwise``/``rowwise``:
the MLP in every layer and the self-attention in the 8 ``full_attention`` layers. That
is 63.9% of the model's parameters (MLPs 58.5%, full attention 5.4%). The 24
``linear_attention`` layers and the tied embeddings are handled elsewhere.

Deliberately replicated rather than sharded, because reunifying them would cost more
than the work saved:

* **RMSNorms, including q_norm/k_norm.** Elementwise on a tensor both ranks already
  hold after a reduction; upstream marks these ``replicated_with_grad_allreduce`` for
  the same reason.
* **The n-gram sidecar and the distillation projections** (65.5M and 26M parameters).
  Sharding either inserts a reduction mid-layer to save a fraction of a gigabyte, and
  the projections consume tapped anchors that would then have to be gathered.

The rule this follows: shard where the parameter is large and the reunification is one
reduction of an activation; replicate where the parameter is small or the reduction
would exceed the flops saved.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from distillkit.fused import fused
from distillkit.tensor_parallel import all_reduce, replicate
from distillkit.tp_linear import ColumnParallelLinear, RowParallelLinear, split_sizes


@fused
def _swiglu(gate, up):
    """One kernel for the activation and the gate, and the activation's output is
    recomputed in backward rather than held: the intermediate is half the MLP width
    per shard, which is 80 MB a layer at batch 2 x 4096."""
    return F.silu(gate) * up


class TensorParallelMLP(nn.Module):
    """gate/up column-parallel, down row-parallel: one all-reduce, no gather.

    The intermediate never leaves its device. ``act_fn`` is applied per shard, which
    is exact because SwiGLU is elementwise over the intermediate channels that the
    column split divides.
    """

    def __init__(self, mlp: nn.Module, devices):
        super().__init__()
        self.devices = [torch.device(d) for d in devices]
        self.act_fn = mlp.act_fn
        # Qwen3.5 is SwiGLU. Anything else keeps the module's own activation rather
        # than being silently replaced by one that happens to be fusable.
        self._is_silu = isinstance(mlp.act_fn, nn.SiLU) or getattr(
            mlp.act_fn, "__name__", None) == "silu"
        self.gate_proj = ColumnParallelLinear(mlp.gate_proj, self.devices)
        self.up_proj = ColumnParallelLinear(mlp.up_proj, self.devices)
        # reduce_only: the residual stream is on the home card, so producing a
        # second output on card 1 only to discard it leaves an unused tensor in
        # the graph and costs a transfer.
        self.down_proj = RowParallelLinear(mlp.down_proj, self.devices, reduce_only=True)

    def forward(self, x: torch.Tensor):
        # One replication feeding both projections, not one each.
        copies = replicate(x, self.devices)
        gates = self.gate_proj(x, copies)
        ups = self.up_proj(x, copies)
        if self._is_silu:
            hidden = [_swiglu(g, u) for g, u in zip(gates, ups)]
        else:
            hidden = [self.act_fn(g) * u for g, u in zip(gates, ups)]
        return self.down_proj(hidden)[0]


class TensorParallelAttention(nn.Module):
    """q/k/v column-parallel by head, o_proj row-parallel.

    Heads are the unit of the split: attention is independent per head, so giving each
    rank whole heads is exact. The model's 16 query and 4 key/value heads both divide
    by two.

    The subtlety is ``q_proj``. Qwen3.5 packs an output gate into it, so it emits
    ``num_heads * head_dim * 2`` and the forward does
    ``view(..., -1, head_dim * 2)`` then ``chunk(2, dim=-1)`` -- meaning the layout is
    head-major, each head owning a contiguous ``[query | gate]`` block. A contiguous
    column split is therefore correct *only because* the boundary falls on a head. Had
    the layout been all queries then all gates, the same split would have handed one
    rank every query and the other every gate, and it would have run and trained
    nonsense. ``_assert_head_major`` pins the assumption.

    ``q_norm``/``k_norm`` are per-head RMSNorms over ``head_dim`` and are replicated:
    each rank normalizes only the heads it owns, which is the same arithmetic on a
    subset.
    """

    def __init__(self, attention: nn.Module, devices):
        super().__init__()
        self.devices = [torch.device(d) for d in devices]
        parts = len(self.devices)
        self.config = attention.config
        self.layer_idx = getattr(attention, "layer_idx", None)
        self.head_dim = attention.head_dim
        self.scaling = attention.scaling
        self.attention_dropout = getattr(attention, "attention_dropout", 0.0)
        # Reject an indivisible head count rather than reshaping attention quietly.
        split_sizes(self.config.num_attention_heads, parts)
        split_sizes(self.config.num_key_value_heads, parts)
        self.heads_per_rank = self.config.num_attention_heads // parts
        self.kv_heads_per_rank = self.config.num_key_value_heads // parts
        _assert_head_major(attention, self.config.num_attention_heads, self.head_dim)

        self.q_proj = ColumnParallelLinear(attention.q_proj, self.devices)
        self.k_proj = ColumnParallelLinear(attention.k_proj, self.devices)
        self.v_proj = ColumnParallelLinear(attention.v_proj, self.devices)
        self.o_proj = RowParallelLinear(attention.o_proj, self.devices, reduce_only=True)
        # Small, elementwise, per-head: replicated on every rank.
        self.q_norms = nn.ModuleList(
            _clone_to(attention.q_norm, device) for device in self.devices
        )
        self.k_norms = nn.ModuleList(
            _clone_to(attention.k_norm, device) for device in self.devices
        )

    def replicated_parameters(self):
        """Both per-head norms receive partial gradients from each head shard."""
        return [list(q.parameters()) + list(k.parameters()) for q, k in zip(self.q_norms, self.k_norms)]

    def forward(self, hidden_states, position_embeddings, attention_mask=None, **kwargs):
        from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb

        input_shape = hidden_states.shape[:-1]
        copies = replicate(hidden_states, self.devices)
        queries = self.q_proj(hidden_states, copies)
        keys = self.k_proj(hidden_states, copies)
        values = self.v_proj(hidden_states, copies)
        cos, sin = position_embeddings
        outputs = []
        for index, device in enumerate(self.devices):
            query, gate = torch.chunk(
                queries[index].view(*input_shape, -1, self.head_dim * 2), 2, dim=-1
            )
            gate = gate.reshape(*input_shape, -1)
            shape = (*input_shape, -1, self.head_dim)
            q = self.q_norms[index](query.reshape(shape)).transpose(1, 2)
            k = self.k_norms[index](keys[index].view(shape)).transpose(1, 2)
            v = values[index].view(shape).transpose(1, 2)
            q, k = apply_rotary_pos_emb(q, k, cos.to(device), sin.to(device))
            # repeat_kv rather than enable_gqa: no fused kernel on this build
            # broadcasts grouped-query heads, so asking for it selects the math
            # kernel. See distillkit/gqa_dispatch.py.
            groups = self.heads_per_rank // self.kv_heads_per_rank
            attention_output = torch.nn.functional.scaled_dot_product_attention(
                q, _repeat_kv(k, groups), _repeat_kv(v, groups),
                attn_mask=None if attention_mask is None else attention_mask.to(device),
                dropout_p=self.attention_dropout if self.training else 0.0,
                scale=self.scaling,
                is_causal=attention_mask is None and q.shape[2] > 1,
            )
            attention_output = attention_output.transpose(1, 2).reshape(*input_shape, -1)
            outputs.append(attention_output.contiguous() * torch.sigmoid(gate))
        return self.o_proj(outputs)[0], None


def _assert_head_major(attention: nn.Module, heads: int, head_dim: int) -> None:
    """q_proj must lay its gate out per head, not as a second contiguous block.

    A contiguous column split is only valid if each head owns a contiguous slice. If
    upstream ever repacks this as [all queries | all gates], the split silently
    separates a head from its gate.
    """
    expected = heads * head_dim * 2
    if attention.q_proj.out_features != expected:
        raise ValueError(
            f"q_proj emits {attention.q_proj.out_features}, expected {expected} "
            f"(heads x head_dim x 2 for the packed output gate); the head-major "
            f"layout this split relies on may have changed"
        )


def _repeat_kv(tensor: torch.Tensor, groups: int) -> torch.Tensor:
    if groups == 1:
        return tensor
    batch, heads, seq, dim = tensor.shape
    return (
        tensor[:, :, None]
        .expand(batch, heads, groups, seq, dim)
        .reshape(batch, heads * groups, seq, dim)
    )


def _clone_to(module: nn.Module, device: torch.device) -> nn.Module:
    import copy

    return copy.deepcopy(module).to(device)


def shard_decoder_layer(layer: nn.Module, devices) -> bool:
    """Replace a layer's MLP, and its attention when it is a full-attention layer.

    Returns whether the attention was sharded. The linear-attention layers keep their
    module and get only the MLP, until head-sharded GatedDeltaNet lands.
    """
    layer.mlp = TensorParallelMLP(layer.mlp, devices)
    if hasattr(layer, "self_attn") and layer.self_attn is not None:
        layer.self_attn = TensorParallelAttention(layer.self_attn, devices)
        return True
    return False

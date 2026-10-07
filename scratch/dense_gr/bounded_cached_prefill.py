# Assisted-by: Codex
"""Bound cached CSA2 score temporaries without changing the model source.

Query rows are independent after projection. Keep the original projections, cache
updates, selection and SDPA implementations; split only their query dimension.
Full boolean selection masks remain quadratic, but per-head float score tensors
are bounded by query_chunk * key_length. Decode and DeltaNet are untouched.
"""
from types import MethodType
import torch
from torch.overrides import TorchFunctionMode


class QueryChunkSDPA(TorchFunctionMode):
    def __init__(self, query_chunk):
        self.query_chunk = query_chunk

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func is not torch.nn.functional.scaled_dot_product_attention:
            return func(*args, **kwargs)
        query, key, value = args[:3]
        if query.shape[-2] <= self.query_chunk:
            return func(*args, **kwargs)
        # CSA2 passes these by keyword. Reject unknown call contracts rather than
        # silently changing a causal mask or dropout's random-number sequence.
        if len(args) != 3 or kwargs.get('is_causal', False) or kwargs.get('dropout_p', 0):
            raise ValueError('bounded CSA2 SDPA requires explicit masking and zero dropout')
        mask = kwargs.get('attn_mask')
        outputs = []
        for start in range(0, query.shape[-2], self.query_chunk):
            stop = start + self.query_chunk
            options = dict(kwargs)
            if mask is not None and mask.shape[-2] != 1:
                options['attn_mask'] = mask[..., start:stop, :]
            outputs.append(func(query[..., start:stop, :], key, value, **options))
        return torch.cat(outputs, dim=-2)


def install(model, query_chunk=256):
    """Install an inference-only, per-instance adapter; return affected layer ids."""
    from distillkit.models.qwen35.csa2 import Qwen35SparseLatentAttention
    if query_chunk < 1:
        raise ValueError('query_chunk must be positive')
    layers = []
    for module in model.modules():
        if not isinstance(module, Qwen35SparseLatentAttention):
            continue
        if hasattr(module, '_bounded_query_chunk'):
            raise ValueError('bounded prefill already installed')
        if module.record_attention:
            raise ValueError('attention recording is not bounded by this inference adapter')
        original_select = module._select_tokens
        original_attend = module._attend_gathered

        def select(self, queries, keys, weights, positions, kv_positions,
                   original=original_select):
            if torch.is_grad_enabled() or self.training:
                raise RuntimeError('bounded cached prefill is inference-only')
            if queries.shape[1] <= query_chunk:
                return original(queries, keys, weights, positions, kv_positions)
            allowed = torch.empty((queries.shape[0], queries.shape[1], keys.shape[1]),
                                  device=queries.device, dtype=torch.bool)
            for start in range(0, queries.shape[1], query_chunk):
                stop = start + query_chunk
                allowed[:, start:stop] = original(
                    queries[:, start:stop], keys, weights[:, start:stop],
                    positions[start:stop], kv_positions)
            return allowed

        def attend(self, *args, original=original_attend, **kwargs):
            if torch.is_grad_enabled() or self.training:
                raise RuntimeError('bounded cached prefill is inference-only')
            with QueryChunkSDPA(query_chunk):
                return original(*args, **kwargs)

        module._select_tokens = MethodType(select, module)
        module._attend_gathered = MethodType(attend, module)
        module._bounded_query_chunk = query_chunk
        layers.append(module.layer_idx)
    if not layers:
        raise ValueError('no CSA2 layers found')
    return layers

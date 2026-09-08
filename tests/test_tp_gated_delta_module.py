"""The head-sharded GatedDeltaNet must equal the stock module it replaces.

This is the layer upstream declines to shard, so there is no reference implementation
to lean on -- only the unsharded module's own output. Every check here is against
that, because each way this can be wrong (mis-sliced conv channels, a split head
group, a norm gradient left as a partial) produces a running model rather than an
error.
"""

import copy

import pytest
import torch

from distillkit.linear_attention_dispatch import install_device_aware_linear_attention

install_device_aware_linear_attention()

from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig  # noqa: E402
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM  # noqa: E402

from distillkit.tp_gated_delta_module import (  # noqa: E402
    TensorParallelGatedDeltaNet,
    sync_replicated_gradients,
)

TWO_GPUS = pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="needs two CUDA devices"
)


def _linear_attention_module():
    """A GatedDeltaNet with the student's 2:1 value-to-key head ratio, scaled down."""
    config = Qwen3_5TextConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        linear_key_head_dim=8, linear_value_head_dim=8, linear_num_key_heads=2,
        linear_num_value_heads=4, linear_conv_kernel_dim=4, full_attention_interval=2,
        tie_word_embeddings=True, max_position_embeddings=64, pad_token_id=0,
        eos_token_id=3, use_cache=False,
    )
    torch.manual_seed(0)
    model = Qwen3_5ForCausalLM(config)
    layer = next(
        l for l, kind in zip(model.model.layers, config.layer_types)
        if kind == "linear_attention"
    )
    return layer.linear_attn


def test_sharded_output_matches_the_stock_module_on_cpu():
    source = _linear_attention_module()
    x = torch.randn(2, 8, 32, requires_grad=True)
    reference = source(x)
    reference.sum().backward()
    reference_grad = x.grad.clone()

    sharded_x = x.detach().clone().requires_grad_(True)
    sharded = TensorParallelGatedDeltaNet(copy.deepcopy(source), ["cpu", "cpu"])
    out = sharded(sharded_x)
    out.sum().backward()

    torch.testing.assert_close(out, reference, rtol=2e-4, atol=2e-5)
    torch.testing.assert_close(sharded_x.grad, reference_grad, rtol=2e-4, atol=2e-5)


def test_replicated_norm_gradient_is_a_partial_until_reduced():
    """Each rank sees only its share of the loss through the shared norm weight.

    Summing them is what makes the replica equivalent to the unsharded parameter --
    and omitting it does not raise, it just trains the norm on a fraction.
    """
    source = _linear_attention_module()
    x = torch.randn(2, 8, 32)

    reference_module = copy.deepcopy(source)
    reference_module(x.clone()).sum().backward()
    reference_grad = reference_module.norm.weight.grad.clone()

    sharded = TensorParallelGatedDeltaNet(copy.deepcopy(source), ["cpu", "cpu"])
    sharded(x.clone()).sum().backward()
    partials = [norm.weight.grad.clone() for norm in sharded.norm]

    # Neither rank alone equals the whole; their sum does.
    assert not torch.allclose(partials[0], reference_grad, rtol=2e-3, atol=2e-5)
    torch.testing.assert_close(
        partials[0] + partials[1], reference_grad, rtol=2e-3, atol=2e-5
    )

    holder = torch.nn.Module()
    holder.child = sharded
    assert sync_replicated_gradients(holder) == 1
    for norm in sharded.norm:
        torch.testing.assert_close(norm.weight.grad, reference_grad, rtol=2e-3, atol=2e-5)


def test_cache_is_refused_rather_than_aliased():
    """Both ranks would write the same layer_idx slot; refuse instead."""
    sharded = TensorParallelGatedDeltaNet(_linear_attention_module(), ["cpu", "cpu"])
    with pytest.raises(NotImplementedError, match="alias"):
        sharded(torch.randn(1, 4, 32), cache_params=object())


def test_each_rank_holds_half_the_sharded_parameters():
    source = _linear_attention_module()
    sharded = TensorParallelGatedDeltaNet(copy.deepcopy(source), ["cpu", "cpu"])
    # The norm is replicated on purpose, so exclude it from the halving check.
    norm_size = sum(p.numel() for p in source.norm.parameters())
    original = sum(p.numel() for p in source.parameters()) - norm_size
    # Names are `norm.0.weight`, so match the prefix -- ".norm." would miss them.
    sharded_total = sum(
        p.numel() for name, p in sharded.named_parameters()
        if not name.startswith("norm.")
    )
    assert sharded_total == original, (
        f"sharded holds {sharded_total} of {original} non-replicated parameters"
    )


def test_channel_plan_metadata_reaches_the_split():
    """The forward splits by key_dim/value_dim; a shard's must be its own, not the model's."""
    source = _linear_attention_module()
    sharded = TensorParallelGatedDeltaNet(copy.deepcopy(source), ["cpu", "cpu"])
    assert sharded.key_dim == source.key_dim // 2
    assert sharded.value_dim == source.value_dim // 2
    assert sharded.num_v_heads == source.num_v_heads // 2


@TWO_GPUS
def test_sharded_output_matches_across_two_cards():
    source = _linear_attention_module()
    x = torch.randn(2, 8, 32)

    reference_x = x.clone().to("cuda:0").requires_grad_(True)
    reference = copy.deepcopy(source).to("cuda:0")(reference_x)
    reference.sum().backward()

    sharded_x = x.clone().to("cuda:0").requires_grad_(True)
    sharded = TensorParallelGatedDeltaNet(copy.deepcopy(source), ["cuda:0", "cuda:1"])
    out = sharded(sharded_x)
    out.sum().backward()

    torch.testing.assert_close(out, reference, rtol=2e-3, atol=2e-4)
    torch.testing.assert_close(sharded_x.grad, reference_x.grad, rtol=2e-3, atol=2e-4)

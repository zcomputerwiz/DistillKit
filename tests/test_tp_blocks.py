"""Sharded Qwen3.5 blocks must reproduce the originals exactly.

A sharded block that is subtly wrong still trains -- it just optimizes a different
function -- so each of these compares against the unmodified module on the same input,
in both the value and the gradient.
"""

import copy

import pytest
import torch

from distillkit.linear_attention_dispatch import install_device_aware_linear_attention
from distillkit.tp_blocks import TensorParallelAttention, TensorParallelMLP, shard_decoder_layer

install_device_aware_linear_attention()

from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig  # noqa: E402
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM  # noqa: E402

TWO_GPUS = pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="needs two CUDA devices"
)


def _config(layers=4):
    return Qwen3_5TextConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=layers,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        linear_key_head_dim=8, linear_value_head_dim=8, linear_num_key_heads=2,
        linear_num_value_heads=4, linear_conv_kernel_dim=4, full_attention_interval=2,
        tie_word_embeddings=True, max_position_embeddings=64, pad_token_id=0,
        eos_token_id=3, use_cache=False,
    )


def _model():
    torch.manual_seed(0)
    return Qwen3_5ForCausalLM(_config()).eval()


def test_sharded_mlp_matches_the_original_on_cpu():
    mlp = _model().model.layers[0].mlp
    x = torch.randn(2, 6, 32, requires_grad=True)
    reference = mlp(x)
    reference.sum().backward()
    reference_grad = x.grad.clone()

    sharded_x = x.detach().clone().requires_grad_(True)
    out = TensorParallelMLP(copy.deepcopy(mlp), ["cpu", "cpu"])(sharded_x)
    out.sum().backward()

    torch.testing.assert_close(out, reference, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(sharded_x.grad, reference_grad, rtol=1e-5, atol=1e-6)


def test_sharded_attention_matches_the_original_on_cpu():
    """Heads are the unit of the split, so the result must be bit-comparable."""
    model = _model()
    layer = next(
        l for l, kind in zip(model.model.layers, model.config.layer_types)
        if kind == "full_attention"
    )
    attention = layer.self_attn
    x = torch.randn(2, 6, 32, requires_grad=True)
    position_ids = torch.arange(6).unsqueeze(0)
    cos, sin = model.model.rotary_emb(x, position_ids.unsqueeze(0).expand(3, 1, -1))

    reference, _ = attention(x, position_embeddings=(cos, sin), attention_mask=None)
    reference.sum().backward()
    reference_grad = x.grad.clone()

    sharded_x = x.detach().clone().requires_grad_(True)
    sharded = TensorParallelAttention(copy.deepcopy(attention), ["cpu", "cpu"])
    out, _ = sharded(sharded_x, position_embeddings=(cos, sin), attention_mask=None)
    out.sum().backward()

    torch.testing.assert_close(out, reference, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(sharded_x.grad, reference_grad, rtol=1e-4, atol=1e-5)


def test_shard_decoder_layer_reports_which_layers_got_attention():
    """Only full_attention layers have a self_attn to shard.

    The linear_attention layers keep their GatedDeltaNet until head-sharding lands,
    so a silent no-op there would leave 24 of 32 layers unsharded without saying so.
    """
    model = _model()
    kinds = model.config.layer_types
    sharded_attention = [
        shard_decoder_layer(layer, ["cpu", "cpu"]) for layer in model.model.layers
    ]
    for layer, kind, got_attention in zip(model.model.layers, kinds, sharded_attention):
        assert isinstance(layer.mlp, TensorParallelMLP)
        assert got_attention == (kind == "full_attention")


def test_sharded_attention_rejects_an_indivisible_head_count():
    config = _config()
    config.num_key_value_heads = 3
    model = Qwen3_5ForCausalLM(config).eval()
    attention = next(
        l.self_attn for l, kind in zip(model.model.layers, config.layer_types)
        if kind == "full_attention"
    )
    with pytest.raises(ValueError, match="evenly"):
        TensorParallelAttention(attention, ["cpu", "cpu"])


@TWO_GPUS
def test_sharded_mlp_matches_across_two_cards():
    mlp = _model().model.layers[0].mlp
    x = torch.randn(2, 6, 32)

    reference_x = x.clone().to("cuda:0").requires_grad_(True)
    reference = copy.deepcopy(mlp).to("cuda:0")(reference_x)
    reference.sum().backward()

    sharded_x = x.clone().to("cuda:0").requires_grad_(True)
    out = TensorParallelMLP(copy.deepcopy(mlp), ["cuda:0", "cuda:1"])(sharded_x)
    out.sum().backward()

    torch.testing.assert_close(out, reference, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(sharded_x.grad, reference_x.grad, rtol=1e-4, atol=1e-5)


@TWO_GPUS
def test_sharded_mlp_halves_the_parameters_per_card():
    """The memory case: each card should hold half the MLP, not all of it."""
    mlp = _model().model.layers[0].mlp
    total = sum(p.numel() for p in mlp.parameters())
    sharded = TensorParallelMLP(copy.deepcopy(mlp), ["cuda:0", "cuda:1"])
    for index in range(2):
        on_card = sum(
            p.numel() for p in sharded.parameters()
            if p.device == torch.device("cuda", index)
        )
        assert on_card == total // 2, f"card {index} holds {on_card} of {total}"

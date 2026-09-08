"""A tensor-parallel student must be the same function as the one it replaced.

This is the end-to-end gate: everything below it has been verified module by module,
but a whole model can still be wrong through a module that quietly failed to shard,
or a device the residual stream was not expected to be on. Both would run.
"""

import copy

import pytest
import torch

from distillkit.linear_attention_dispatch import install_device_aware_linear_attention

install_device_aware_linear_attention()

from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig  # noqa: E402
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM  # noqa: E402

from distillkit.tp_blocks import TensorParallelAttention, TensorParallelMLP  # noqa: E402
from distillkit.tp_gated_delta_module import TensorParallelGatedDeltaNet  # noqa: E402
from distillkit.tp_model import (  # noqa: E402
    shard_model,
    sharded_parameter_report,
    sync_replicated_gradients,
)

TWO_GPUS = pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="needs two CUDA devices"
)


def _model(layers=4):
    config = Qwen3_5TextConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=layers,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        linear_key_head_dim=8, linear_value_head_dim=8, linear_num_key_heads=2,
        linear_num_value_heads=4, linear_conv_kernel_dim=4, full_attention_interval=2,
        tie_word_embeddings=True, max_position_embeddings=64, pad_token_id=0,
        eos_token_id=3, use_cache=False,
    )
    torch.manual_seed(0)
    return Qwen3_5ForCausalLM(config).eval()


def test_shard_model_needs_two_devices():
    with pytest.raises(ValueError, match="at least two"):
        shard_model(_model(), ["cpu"])


def test_home_must_be_the_first_device():
    with pytest.raises(ValueError, match="home must be"):
        shard_model(_model(), ["cpu", "cpu"], home="meta")


def test_every_shardable_module_is_replaced():
    """A module that silently fails to shard still runs, at twice the memory."""
    model = _model()
    kinds = list(model.config.layer_types)
    shard_model(model, ["cpu", "cpu"])
    for layer, kind in zip(model.model.layers, kinds):
        assert isinstance(layer.mlp, TensorParallelMLP)
        if kind == "full_attention":
            assert isinstance(layer.self_attn, TensorParallelAttention)
        else:
            assert isinstance(layer.linear_attn, TensorParallelGatedDeltaNet)


def test_report_counts_what_was_actually_split():
    model = _model()
    shard_model(model, ["cpu", "cpu"])
    report = sharded_parameter_report(model)
    # Embeddings are tied and stay whole on the home card; everything else splits.
    embedding = model.get_input_embeddings().weight.numel()
    assert report["total_parameters"] - report["sharded_parameters"] < embedding * 1.5
    assert report["sharded_fraction"] > 0.5


def test_sharded_model_matches_the_original_on_cpu():
    original = _model()
    input_ids = torch.randint(0, 64, (2, 8))
    reference = original(input_ids=input_ids, return_dict=True).logits

    sharded = shard_model(copy.deepcopy(original), ["cpu", "cpu"])
    out = sharded(input_ids=input_ids, return_dict=True).logits

    torch.testing.assert_close(out, reference, rtol=2e-4, atol=2e-5)


def test_sharded_model_gradients_match_the_original_on_cpu():
    """Includes the replicated-norm reduction, without which they would not."""
    original = _model()
    input_ids = torch.randint(0, 64, (2, 8))
    original(input_ids=input_ids, return_dict=True).logits.sum().backward()

    sharded = shard_model(copy.deepcopy(original), ["cpu", "cpu"])
    sharded(input_ids=input_ids, return_dict=True).logits.sum().backward()
    sync_replicated_gradients(sharded)

    # The embedding is untouched by sharding, so it is the cleanest end-to-end check
    # that the whole backward path reassembled correctly.
    torch.testing.assert_close(
        sharded.get_input_embeddings().weight.grad,
        original.get_input_embeddings().weight.grad,
        rtol=2e-3, atol=2e-5,
    )


@TWO_GPUS
def test_sharded_model_matches_across_two_cards():
    original = _model()
    input_ids = torch.randint(0, 64, (2, 8))
    reference = original.to("cuda:0")(
        input_ids=input_ids.to("cuda:0"), return_dict=True
    ).logits

    sharded = shard_model(copy.deepcopy(original).cpu(), ["cuda:0", "cuda:1"])
    out = sharded(input_ids=input_ids.to("cuda:0"), return_dict=True).logits

    torch.testing.assert_close(out, reference, rtol=2e-3, atol=2e-4)


@TWO_GPUS
def test_parameters_actually_land_on_both_cards():
    """The point of the exercise: card 1 must hold a real share of the model."""
    model = shard_model(_model(layers=8), ["cuda:0", "cuda:1"])
    report = sharded_parameter_report(model)
    first, second = report["gib_per_device"]
    assert second > 0, "card 1 holds nothing; nothing was sharded"
    # Card 0 additionally carries the tied embeddings, so exact balance is not
    # expected -- but card 1 should hold a substantial share of the layer weights.
    assert second > 0.25 * first, f"card 1 holds only {second:.3f} against {first:.3f} GiB"


@TWO_GPUS
def test_residual_stream_stays_on_the_home_card():
    """The outer model is untouched, so embeddings, norms and head stay home."""
    model = shard_model(_model(), ["cuda:0", "cuda:1"])
    home = torch.device("cuda", 0)
    assert model.get_input_embeddings().weight.device == home
    assert model.model.norm.weight.device == home
    for layer in model.model.layers:
        assert layer.input_layernorm.weight.device == home
        assert layer.post_attention_layernorm.weight.device == home

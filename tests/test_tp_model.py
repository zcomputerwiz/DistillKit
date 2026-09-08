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
    # The tied embedding is parked on card 1, which holds no norms, so card 1 now
    # comes out heavier by about that parameter rather than lighter by it.
    assert second > first, f"card 1 holds {second:.3g} against {first:.3g} GiB"


@TWO_GPUS
def test_residual_stream_stays_home_and_the_tied_embedding_moves():
    """Norms stay on the home card; the one big whole parameter goes to the other
    card, stays a single tied tensor under its stock names, and no caller notices."""
    model = shard_model(_model(), ["cuda:0", "cuda:1"])
    home, other = torch.device("cuda", 0), torch.device("cuda", 1)
    assert model.model.norm.weight.device == home
    for layer in model.model.layers:
        assert layer.input_layernorm.weight.device == home
        assert layer.post_attention_layernorm.weight.device == home

    embed, head = model.get_input_embeddings(), model.get_output_embeddings()
    assert embed.weight.device == other and head.weight is embed.weight
    assert model.hf_device_map["lm_head.weight"] == model.hf_device_map["model.embed_tokens.weight"] == "cuda:1"
    assert {"model.embed_tokens.weight", "lm_head.weight"} <= set(model.state_dict())

    logits = model(input_ids=torch.randint(0, 64, (1, 8), device=home)).logits
    assert logits.device == home
    logits.float().square().mean().backward()
    assert embed.weight.grad is not None and embed.weight.grad.device == other


@TWO_GPUS
def test_embedding_can_be_pinned_home():
    model = shard_model(_model(), ["cuda:0", "cuda:1"], embedding_device="cuda:0")
    assert model.get_input_embeddings().weight.device == torch.device("cuda", 0)
    with pytest.raises(ValueError, match="one of the tensor-parallel devices"):
        shard_model(_model(), ["cuda:0", "cuda:1"], embedding_device="cpu")


@TWO_GPUS
@pytest.mark.parametrize("shard_all", [False, True], ids=["mlp", "all_blocks"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_non_reentrant_checkpoint_matches_uncheckpointed_training(shard_all, dtype):
    """Check logits and every parameter gradient, including both MLP shards."""
    model = _model().to(device="cuda:0", dtype=dtype).train()
    devices = ["cuda:0", "cuda:1"]
    if shard_all:
        shard_model(model, devices)
    else:
        for layer in model.model.layers:
            layer.mlp = TensorParallelMLP(layer.mlp, devices)
    # Gated-delta constructs local Linear modules in the default dtype; apply
    # the training dtype to the finished model as well as the source weights.
    model.to(dtype=dtype)

    ids = torch.randint(0, 64, (1, 8), device="cuda:0")
    if shard_all:
        # Cold Triton autotuners share mutable state between the two GPU workers.
        # Warm their kernels serially before testing normal multithreaded backward.
        # Both measured paths below use the default autograd worker scheduling.
        with torch.autograd.set_multithreading_enabled(False):
            model(input_ids=ids).logits.float().square().mean().backward()
        model.zero_grad(set_to_none=True)
    reference = model(input_ids=ids).logits
    reference.float().square().mean().backward()
    reference_grads = {
        name: parameter.grad.clone()
        for name, parameter in model.named_parameters()
    }
    expected = reference.detach().clone()
    del reference
    model.zero_grad(set_to_none=True)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    # Multiple iterations also exercise frame lifetime and gradient reset.
    for _ in range(3):
        actual = model(input_ids=ids).logits
        actual.float().square().mean().backward()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(
                parameter.grad, reference_grads[name], rtol=0, atol=0,
                msg=lambda message: f"{name}: {message}",
            )
        del actual
        model.zero_grad(set_to_none=True)

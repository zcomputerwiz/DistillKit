"""Tiny architecture gates: loading, exact parity, trained reload, checkpoint replay."""

import copy

import pytest
import torch
from safetensors.torch import save_file
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config, Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

from distillkit.models import Qwen35SidecarForCausalLM
from distillkit.ngram_table import IQ4NL_KVALUES


def tiny_config(**kwargs):
    values = dict(
        vocab_size=64, hidden_size=64, intermediate_size=96, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        linear_key_head_dim=16, linear_value_head_dim=16, linear_num_key_heads=2,
        linear_num_value_heads=4, linear_conv_kernel_dim=4, full_attention_interval=2,
        tie_word_embeddings=True, max_position_embeddings=64, pad_token_id=0,
        eos_token_id=3, sidecar_num_heads=2, sidecar_head_dim=32, sidecar_num_branches=3,
        use_cache=False,
    )
    values.update(kwargs)
    return Qwen3_5TextConfig(**values)


def raw_batch(seed=42, batch=2, length=8, device="cpu"):
    generator = torch.Generator().manual_seed(seed)
    raw = torch.randint(0, 256, (batch, length, 2, 18), dtype=torch.uint8, generator=generator)
    scales = torch.full((batch, length, 2, 1), 0.001, dtype=torch.float16)
    raw[..., :2] = scales.view(torch.uint8)
    return raw.to(device)


@pytest.mark.parametrize("device,dtype", [("cpu", torch.float32)] + (
    [("cuda", torch.bfloat16)] if torch.cuda.is_available() else []
))
def test_pretrained_parity_and_missing_weight_identity(tmp_path, device, dtype):
    torch.manual_seed(7)
    stock = Qwen3_5ForCausalLM(tiny_config()).eval()
    stock.save_pretrained(tmp_path)
    custom, info = Qwen35SidecarForCausalLM.from_pretrained(tmp_path, output_loading_info=True)
    assert info["missing_keys"]
    assert all(".sidecar." in name for name in info["missing_keys"])
    assert not info["unexpected_keys"]
    assert set(stock.state_dict()).issubset(custom.state_dict())
    for name, value in stock.state_dict().items():
        assert torch.equal(value, custom.state_dict()[name]), name
    sidecar = custom.model.layers[1].sidecar
    assert torch.count_nonzero(sidecar.W_side_proj.weight) == 0
    assert all(torch.count_nonzero(branch.weight) == 0 for branch in sidecar.gated_residual.branches)
    assert torch.count_nonzero(sidecar.gated_residual.W_x.bias) == 0
    assert torch.equal(sidecar.dequant.kvalues, torch.tensor(IQ4NL_KVALUES, dtype=torch.float32))
    assert all("dequant" not in key and "table" not in key for key in custom.state_dict())
    stock.to(device=device, dtype=dtype)
    custom.to(device=device, dtype=dtype).eval()
    ids = torch.tensor([[5, 8, 9, 3, 10, 12, 20, 6]], device=device)
    with torch.no_grad():
        expected = stock(ids, output_hidden_states=True)
        actual = custom(ids, ngram_raw=raw_batch(batch=1, device=device), output_hidden_states=True)
    assert torch.equal(expected.logits, actual.logits)
    assert len(actual.hidden_states) == tiny_config().num_hidden_layers + 1
    assert all(torch.equal(a, b) for a, b in zip(expected.hidden_states, actual.hidden_states))


def test_explicit_text_class_loads_vlm_prefixed_checkpoint(tmp_path):
    stock = Qwen3_5ForCausalLM(tiny_config()).eval()
    config = Qwen3_5Config(text_config=stock.config.to_dict())
    config.architectures = ["Qwen3_5ForConditionalGeneration"]
    config.save_pretrained(tmp_path)
    weights = {
        name.replace("model.", "model.language_model.", 1) if name.startswith("model.") else name: value.clone()
        for name, value in stock.state_dict().items()
    }
    save_file(weights, str(tmp_path / "model.safetensors"), metadata={"format": "pt"})
    custom, info = Qwen35SidecarForCausalLM.from_pretrained(tmp_path, output_loading_info=True)
    assert isinstance(custom.config, Qwen3_5TextConfig)
    assert all(".sidecar." in name for name in info["missing_keys"])
    assert not info["unexpected_keys"]
    for name, value in stock.state_dict().items():
        assert torch.equal(value, custom.state_dict()[name]), name


@pytest.mark.parametrize("checkpointing", [None, True, False])
def test_frozen_backbone_gradients_and_replay_uses_original_batch(checkpointing):
    torch.manual_seed(12)
    reference = Qwen35SidecarForCausalLM(tiny_config()).train()
    # Exercise gate gradients as well as branches/projection, and make wrong-batch
    # replay detectable in a second outstanding forward graph.
    with torch.no_grad():
        for parameter in reference.model.layers[1].sidecar.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.01)
    model = copy.deepcopy(reference)
    reference.freeze_backbone()
    model.freeze_backbone()
    if checkpointing is not None:
        model.gradient_checkpointing_enable({"use_reentrant": checkpointing})
    ids = torch.tensor([[5, 8, 9, 3, 10, 12, 20, 6]])
    raw1, raw2 = raw_batch(batch=1), raw_batch(seed=8, batch=1)
    losses = [model(ids, labels=ids, ngram_raw=raw).loss for raw in (raw1, raw2)]
    sum(losses).backward()
    sum(reference(ids, labels=ids, ngram_raw=raw).loss for raw in (raw1, raw2)).backward()
    ref_params = dict(reference.named_parameters())
    for name, parameter in model.named_parameters():
        if name in model.stage1_parameter_names():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
            assert parameter.grad.norm() > 0, name
            torch.testing.assert_close(parameter.grad, ref_params[name].grad, rtol=1e-4, atol=1e-6)
        else:
            assert not parameter.requires_grad and parameter.grad is None, name
    model.unfreeze_backbone()
    assert all(parameter.requires_grad for parameter in model.parameters())


def test_initial_gradients_and_control_arm():
    model = Qwen35SidecarForCausalLM(tiny_config()).train()
    model.freeze_backbone()
    ids = torch.tensor([[5, 8, 9, 3, 10, 12, 20, 6]])
    model(ids, labels=ids, ngram_raw=raw_batch(batch=1)).loss.backward()
    sidecar = model.model.layers[1].sidecar
    assert sidecar.W_side_proj.weight.grad.norm() > 0
    assert all(branch.weight.grad.norm() > 0 for branch in sidecar.gated_residual.branches)
    assert sidecar.gated_residual.W_x.weight.grad.count_nonzero() == 0
    model.zero_grad()
    model(ids, labels=ids, sidecar_enabled=False).loss.backward()
    assert sidecar.W_side_proj.weight.grad is None
    assert all(branch.weight.grad.norm() > 0 for branch in sidecar.gated_residual.branches)
    assert "sidecar/gated_residual/gate_1_mean" in model.gate_report()


def test_trained_sidecar_checkpoint_round_trip(tmp_path):
    model = Qwen35SidecarForCausalLM(tiny_config()).eval()
    with torch.no_grad():
        for parameter in model.model.layers[1].sidecar.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.02)
    model.save_pretrained(tmp_path)
    reloaded, info = Qwen35SidecarForCausalLM.from_pretrained(tmp_path, output_loading_info=True)
    assert not info["missing_keys"] and not info["unexpected_keys"]
    assert reloaded.config.sidecar_num_heads == 2
    assert reloaded.config.architectures == ["Qwen35SidecarForCausalLM"]
    ids = torch.tensor([[5, 8, 9, 3, 10, 12, 20, 6]])
    with torch.no_grad():
        expected = model(ids, ngram_raw=raw_batch(batch=1)).logits
        actual = reloaded(ids, ngram_raw=raw_batch(batch=1)).logits
    assert torch.equal(expected, actual)
    assert all(torch.equal(value, reloaded.state_dict()[key]) for key, value in model.state_dict().items())


def test_raw_contract_and_config_errors():
    model = Qwen35SidecarForCausalLM(tiny_config())
    ids = torch.tensor([[5, 8, 9, 3, 10, 12, 20, 6]])
    with pytest.raises(ValueError, match="ngram_raw is required"):
        model(ids)
    with pytest.raises(ValueError, match="uint8 with shape"):
        model(ids, ngram_raw=raw_batch(batch=1).float())
    with pytest.raises(ValueError, match="uint8 with shape"):
        model(ids, ngram_raw=raw_batch(batch=1)[:, :-1])
    with pytest.raises(ValueError, match="existing decoder"):
        Qwen35SidecarForCausalLM(tiny_config(sidecar_layer_index=3))
    with pytest.raises(TypeError, match="text|Text"):
        Qwen35SidecarForCausalLM(Qwen3_5Config())


def test_disable_sidecar_projection_freezes_only_bypassed_weight():
    # Control arm (sidecar_enabled=False): W_side_proj is bypassed in forward, so
    # it must be frozen to stay out of DDP's unused-parameter reduction, while the
    # gated residual remains trainable.
    model = Qwen35SidecarForCausalLM(tiny_config())
    sidecar = model.model.layers[1].sidecar
    assert sidecar.W_side_proj.weight.requires_grad
    model.disable_sidecar_projection()
    assert not sidecar.W_side_proj.weight.requires_grad
    assert all(branch.weight.requires_grad for branch in sidecar.gated_residual.branches)

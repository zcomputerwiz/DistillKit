"""The direction-gated PLE sidecar: identity at load, a gate that can train, and
per-stream admission that actually depends on the stream."""

import math

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from distillkit.ple_gated_sidecar import DirectionGatedPLESidecar


def _sidecar(hc_count=2, gate_directions=2, hidden=32, features=48):
    torch.manual_seed(0)
    return DirectionGatedPLESidecar(hidden, features, hc_count=hc_count,
                                    gate_directions=gate_directions)


def _inputs(module, batch=2, sequence=6):
    torch.manual_seed(1)
    stream = torch.randn(batch, sequence, module.hc_count, module.hidden_size)
    return stream, torch.randn(batch, sequence, module.feature_dim)


def test_identity_at_initialisation_despite_nonzero_gate_directions():
    module = _sidecar()
    stream, features = _inputs(module)
    assert module.gate.abs().max() > 0, "zero directions never train; see the module note"
    assert torch.equal(module(stream, features), stream)


def test_zero_directions_are_refused_because_they_cannot_train():
    with pytest.raises(ValueError, match="gate_init_std"):
        DirectionGatedPLESidecar(8, 8, gate_init_std=0.0)


def test_gate_directions_receive_gradient_once_the_value_is_nonzero():
    module = _sidecar()
    stream, features = _inputs(module)
    # The value starts at zero, so on the very first step dL/dg is v^T delta = 0. That
    # is a temporary blockage, not the permanent one a zero direction causes.
    module(stream, features).sum().backward()
    assert module.gate.grad.abs().max() == 0
    module.zero_grad(set_to_none=True)
    with torch.no_grad():
        module.value_proj.weight.normal_(std=0.05)
    module(stream, features).sum().backward()
    assert module.gate.grad.abs().max() > 0


def test_admission_differs_per_stream_when_the_streams_differ():
    module = _sidecar()
    stream, _ = _inputs(module)
    gate = module._admission(stream)
    assert gate.shape == (*stream.shape[:2], module.hc_count, 1)
    assert not torch.allclose(gate[..., 0, :], gate[..., 1, :])
    # Each stream's admission reads that stream and nothing else: disturbing stream 1
    # must leave stream 0's gate untouched. This is the property that makes per-stream
    # admission mean something once the branches diverge -- and note it is *not* that
    # identical streams give identical gates, since every stream has its own directions
    # and would score a shared query differently.
    disturbed = stream.clone()
    disturbed[..., 1, :] += 3.0
    moved = module._admission(disturbed)
    assert torch.equal(moved[..., 0, :], gate[..., 0, :])
    assert not torch.allclose(moved[..., 1, :], gate[..., 1, :])


def test_admission_starts_at_one_for_any_number_of_directions():
    """2 * mean, so widening the gate does not silently rescale the value path."""
    for directions in (1, 2, 4, 8):
        module = _sidecar(gate_directions=directions)
        stream, _ = _inputs(module)
        with torch.no_grad():
            module.gate.zero_()  # only to read the combiner's neutral point
        assert torch.allclose(module._admission(stream),
                              torch.ones(1), atol=1e-6)


def test_convolution_branch_does_not_depend_on_the_gate():
    """RMS normalisation cancels the positive scalar, so the branch sees only the value."""
    module = _sidecar()
    stream, features = _inputs(module)
    with torch.no_grad():
        module.value_proj.weight.normal_(std=0.1)
        module.conv1d.weight.normal_(std=0.1)
    value = module.value_proj(features)
    reference = module._short_conv(value)
    # Scaling the value the way any gate would, and re-normalising, changes nothing.
    scaled = module._short_conv(value * 0.37)
    assert torch.allclose(reference, scaled, atol=1e-5)


def test_convolution_is_causal():
    module = _sidecar()
    with torch.no_grad():
        module.conv1d.weight.normal_(std=0.1)
    torch.manual_seed(3)
    value = torch.randn(1, 12, module.hidden_size)
    reference = module._short_conv(value)
    disturbed = value.clone()
    disturbed[:, 7:] += 5.0
    assert torch.allclose(reference[:, :7], module._short_conv(disturbed)[:, :7], atol=1e-5)


def test_wrong_stream_shape_is_refused():
    module = _sidecar(hc_count=2)
    features = torch.randn(2, 6, module.feature_dim)
    with pytest.raises(ValueError, match="expected a stream ending"):
        module(torch.randn(2, 6, 3, module.hidden_size), features)
    with pytest.raises(ValueError, match="expected a stream ending"):
        module(torch.randn(2, 6, module.hidden_size), features)


def test_parameter_count_drops_the_key_projection():
    """The whole point: no key_proj, so the layer is the value matrix plus scraps."""
    hidden, features, hc, k = 32, 48, 2, 2
    module = DirectionGatedPLESidecar(hidden, features, hc_count=hc, gate_directions=k)
    counts = {name: p.numel() for name, p in module.named_parameters()}
    assert set(counts) == {"value_proj.weight", "gate", "conv1d.weight"}
    assert counts["value_proj.weight"] == hidden * features
    assert counts["gate"] == hc * k * hidden
    assert counts["conv1d.weight"] == hc * hidden * 4


def test_gate_report_exposes_movement_and_selectivity():
    module = _sidecar()
    stream, features = _inputs(module)
    module.train()
    with torch.no_grad():
        module.value_proj.weight.normal_(std=0.1)
    module(stream, features)
    report = module.gate_report()
    assert report["ple/value_norm"] > 0
    assert report["ple/conv_norm"] == 0
    assert report["ple/gate_std"] >= 0
    assert "ple/gate_direction_norm_1" in report


def _widened_config(hidden=32, layers=3):
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    config = Qwen3_5TextConfig(
        vocab_size=64, hidden_size=hidden, intermediate_size=64, num_hidden_layers=layers,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8, linear_num_value_heads=4,
        linear_num_key_heads=2, linear_key_head_dim=8, linear_value_head_dim=8,
        linear_conv_kernel_dim=4, max_position_embeddings=64,
        layer_types=["linear_attention"] * layers,
    )
    config.residual_stream_enabled = True
    config.residual_stream_num_branches = 2
    config.residual_stream_lowrank = 4
    config.residual_stream_sidecar = True
    config.sidecar_variant = "ple_gated"
    config.sidecar_layer_index = 1
    config.sidecar_num_heads = 2
    config.sidecar_head_dim = 32
    config.sidecar_gate_directions = 2
    return config


def test_widened_model_with_the_gated_sidecar_is_the_identity_at_load():
    from distillkit.models.qwen35_widened import Qwen35WidenedForCausalLM
    from distillkit.ngram_table import IQ4NL_BLOCK, IQ4NL_TYPE_SIZE

    torch.manual_seed(0)
    config = _widened_config()
    model = Qwen35WidenedForCausalLM(config).eval()
    sidecar = model.model.layers[config.sidecar_layer_index].sidecar
    assert sidecar.reads_widened_stream, "the widened layer must hand over the whole stream"
    assert sidecar.ple.hc_count == config.residual_stream_num_branches
    assert sidecar.ple.gate.abs().max() > 0, "HF re-init must not zero the directions"

    ids = torch.arange(12).reshape(2, 6) % config.vocab_size
    bytes_per_head = config.sidecar_head_dim // IQ4NL_BLOCK * IQ4NL_TYPE_SIZE
    raw = torch.randint(0, 255, (2, 6, config.sidecar_num_heads, bytes_per_head),
                        dtype=torch.uint8)
    with torch.no_grad():
        enabled = model(input_ids=ids, ngram_raw=raw, use_cache=False).logits
        bypassed = model(input_ids=ids, sidecar_enabled=False, use_cache=False).logits
    assert torch.equal(enabled, bypassed), "a zero value and a zero convolution is the identity"


def test_the_gated_sidecar_is_called_once_not_once_per_branch():
    from distillkit.models.qwen35_widened import Qwen35WidenedForCausalLM
    from distillkit.ngram_table import IQ4NL_BLOCK, IQ4NL_TYPE_SIZE

    torch.manual_seed(0)
    config = _widened_config()
    model = Qwen35WidenedForCausalLM(config).eval()
    sidecar = model.model.layers[config.sidecar_layer_index].sidecar
    calls = []
    sidecar.ple.register_forward_pre_hook(
        lambda module, args: calls.append(tuple(args[0].shape)))
    bytes_per_head = config.sidecar_head_dim // IQ4NL_BLOCK * IQ4NL_TYPE_SIZE
    with torch.no_grad():
        model(input_ids=torch.arange(12).reshape(2, 6) % config.vocab_size,
              ngram_raw=torch.zeros(2, 6, config.sidecar_num_heads, bytes_per_head,
                                    dtype=torch.uint8), use_cache=False)
    assert len(calls) == 1, calls
    assert calls[0] == (2, 6, config.residual_stream_num_branches, config.hidden_size)


def test_the_gated_variant_refuses_a_single_stream():
    from distillkit.models.qwen35_sidecar import _set_sidecar_defaults
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    config = Qwen3_5TextConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                               num_hidden_layers=2, num_attention_heads=2,
                               num_key_value_heads=1, head_dim=8)
    config.sidecar_variant = "ple_gated"
    with pytest.raises(ValueError, match="widened residual"):
        _set_sidecar_defaults(config)

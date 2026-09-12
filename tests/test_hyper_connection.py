"""Independent donor equations, exact retrofit, and production integration gates."""
import copy
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5RMSNorm

from distillkit.borrowed_routing import initialise_widened_residual, TENSORS
from distillkit.configuration import ResidualStreamConfig
from distillkit.hyper_connection import HyperConnection, HyperConnectionWarmupCallback, _GatedMean
from distillkit.models import Qwen35WidenedForCausalLM
from distillkit.optimizers import mixed_parameter_groups
from test_widened_residual import tiny_config, sample, small_cpu_pool, run_config
from test_borrowed_routing import _donor, HIDDEN, BRANCHES, LOWRANK


def reference(route, states):
    # Qwen report equations 30--34, intentionally no production helpers.
    dtype = torch.promote_types(states.dtype, torch.float32)
    x = states.to(dtype)
    z = (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + route.norm_eps)
         * (1 + route.branch_gain_delta.to(dtype))).to(states.dtype)
    flat = z.flatten(-2)
    gate = torch.sigmoid(F.linear(F.silu(F.linear(flat, route.W_down.weight)
                                        / route.num_branches), route.W_up.weight))
    read = (gate.reshape_as(z) * z).mean(-2)
    write = 2 * torch.sigmoid(F.linear(flat, route.W_write.weight) / route.num_branches)
    return read, write


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float64])
def test_donor_matches_independent_equations_and_ignores_student_gain(dtype):
    torch.manual_seed(81)
    route = HyperConnection(16, 4, 7, blend=1).to(dtype)
    norm = Qwen3_5RMSNorm(16).to(dtype)
    with torch.no_grad():
        route.branch_gain_delta.uniform_(-.7, .8)
        norm.weight.fill_(4)  # stacking student and donor gains must fail this test
        route.W_write.weight.mul_(5)
    states = torch.randn(2, 5, 4, 16, dtype=dtype)
    expected, gates = reference(route, states)
    actual, weights = route.read(states, norm)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    assert torch.equal(weights, gates)
    assert weights.min() >= 0 and weights.max() <= 2
    assert weights.max() > 1
    output = torch.randn_like(actual)
    assert torch.equal(route.write(states, output, weights), states + weights[..., None] * output[..., None, :])


def test_streaming_gated_mean_gradcheck():
    x = torch.randn(2, 3, 4, 5, dtype=torch.double, requires_grad=True)
    z = torch.randn_like(x, requires_grad=True)
    assert torch.autograd.gradcheck(_GatedMean.apply, (x, z))


def test_routing_gradients_match_independent_reference():
    torch.manual_seed(82)
    route = HyperConnection(8, 4, 3, blend=1)
    states = torch.randn(2, 3, 4, 8, requires_grad=True)
    params = (states, *route.parameters())
    read, write = route.read(states, Qwen3_5RMSNorm(8))
    actual = torch.autograd.grad(read.square().sum() + write.square().sum(), params)
    read, write = reference(route, states)
    expected = torch.autograd.grad(read.square().sum() + write.square().sum(), params)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=2e-6, rtol=1e-5)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_zero_blend_is_bitwise_identity_even_with_poisoned_donor(dtype):
    route = HyperConnection(16, 4, 7, layer_idx=2).to(dtype)
    norm = Qwen3_5RMSNorm(16).to(dtype)
    x = torch.randn(2, 5, 16, dtype=dtype)
    states = x.unsqueeze(-2).expand(2, 5, 4, 16)
    with torch.no_grad():
        norm.weight.uniform_(-.3, .4)
        for p in route.parameters():
            p.fill_(float('nan'))
    read, weights = route.read(states, norm)
    assert torch.equal(read, norm(x))
    updated = route.write(states, read, weights)
    for branch in updated.unbind(-2):
        assert torch.equal(branch, x + norm(x))


def test_interpolation_includes_norm_and_both_routes():
    route = HyperConnection(16, 4, 7, layer_idx=1, blend=.125)
    norm = Qwen3_5RMSNorm(16)
    with torch.no_grad():
        norm.weight.fill_(.5)
        route.branch_gain_delta.fill_(.3)
    states = torch.randn(2, 5, 4, 16)
    donor, weight = reference(route, states)
    student = norm(states[..., 1, :])
    actual, weights = route.read(states, norm)
    torch.testing.assert_close(actual, student + .125 * (donor - student))
    torch.testing.assert_close(weights, 1 + .125 * (weight - 1))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_full_model_exact_identity_and_saved_nonzero_blend(tmp_path, dtype):
    torch.manual_seed(83)
    stock = Qwen3_5ForCausalLM(tiny_config()).to(dtype).eval()
    with torch.no_grad():
        for name, p in stock.named_parameters():
            if 'layernorm' in name:
                p.uniform_(-.3, .4)
    stock.save_pretrained(tmp_path / 'stock')
    cfg = copy.deepcopy(stock.config)
    cfg.residual_stream_routing = 'flash_next'
    cfg.residual_stream_num_branches = 4
    wide = Qwen35WidenedForCausalLM.from_pretrained(tmp_path / 'stock', config=cfg, dtype=dtype).eval()
    with torch.no_grad():
        a, b = stock(sample(), output_hidden_states=True), wide(sample(), output_hidden_states=True)
    assert torch.equal(a.logits, b.logits)
    assert all(torch.equal(x, y) for x, y in zip(a.hidden_states, b.hidden_states))
    for layer in wide.model.layers:
        for route in (layer.attn_residual, layer.mlp_residual):
            route.set_blend(.015)
    wide.save_pretrained(tmp_path / 'wide')
    restored = Qwen35WidenedForCausalLM.from_pretrained(tmp_path / 'wide', dtype=dtype).eval()
    assert all(m.blend.dtype == torch.float32 and float(m.blend) == float(torch.tensor(.015))
               for m in restored.modules() if isinstance(m, HyperConnection))
    with torch.no_grad():
        assert torch.equal(wide(sample()).logits, restored(sample()).logits)
    from distillkit.main import load_student_model
    with pytest.raises(ValueError, match='differs'):
        load_student_model(run_config(tmp_path, tmp_path / 'wide', residual_stream={'lowrank': 8}), 64)


def test_real_layout_loader_preserves_requested_blend(tmp_path):
    model = torch.nn.Module()
    model.model = torch.nn.Module()
    layer = torch.nn.Module()
    for name in ('attn_residual', 'mlp_residual'):
        setattr(layer, name, HyperConnection(HIDDEN, BRANCHES, LOWRANK, blend=.125))
    model.model.layers = torch.nn.ModuleList([layer])
    model.config = SimpleNamespace(sidecar_layer_index=0)
    directory = _donor(tmp_path, range(2))
    initialise_widened_residual(model, directory)
    held = torch.load(directory / 'layer-00.pt', weights_only=True)
    for sub in ('attn_residual', 'mlp_residual'):
        route = getattr(layer, sub)
        assert float(route.blend) == .125
        for name in TENSORS:
            assert torch.equal(route.get_parameter(name), held[f'{sub}.{name}'])


def test_warmup_resume_and_checkpoint_recompute_have_same_blend():
    model = Qwen35WidenedForCausalLM(tiny_config(residual_stream_routing='flash_next'))
    schedule = HyperConnectionWarmupCallback(0, .25, 10)
    state = SimpleNamespace(global_step=5)
    schedule.on_train_begin(None, state, None, model=model)
    assert float(model.model.layers[0].attn_residual.blend) == .125
    model.freeze_backbone()
    model.gradient_checkpointing_enable({'use_reentrant': False})
    loss = model(sample(), labels=sample()).loss
    loss.backward()
    assert model.model.layers[0].attn_residual.W_down.weight.grad.norm() > 0
    assert float(model.model.layers[0].attn_residual.blend) == .125
    state.global_step = 10
    schedule.on_step_end(None, state, None, model=model)
    assert float(model.model.layers[0].attn_residual.blend) == .25
    for group in mixed_parameter_groups(model):
        if any(p is model.model.layers[0].attn_residual.W_down.weight for p in group['params']):
            assert group['optimizer_kind'] == 'adamw'


def test_blend_stays_fp32_without_rounding_on_cast():
    route = HyperConnection(4, blend=.001)
    expected = route.blend.clone()
    route.bfloat16()
    assert route.blend.dtype == torch.float32
    assert torch.equal(route.blend, expected)


def test_blend_cannot_silently_configure_legacy_routing():
    assert ResidualStreamConfig().routing == 'widened'
    with pytest.raises(ValueError, match='require routing=flash_next'):
        ResidualStreamConfig(blend_warmup_steps=10)


def test_tp_export_and_reload_preserve_scheduled_buffer():
    from distillkit.tp_model import shard_model
    from distillkit.tp_checkpoint import consolidated_state_dict, load_consolidated_state_dict
    model = Qwen35WidenedForCausalLM(tiny_config(residual_stream_routing='flash_next')).eval()
    for route in model.modules():
        if isinstance(route, HyperConnection):
            route.set_blend(.015)
    parallel = shard_model(copy.deepcopy(model), ['cpu', 'cpu'])
    portable = consolidated_state_dict(parallel)
    assert set(portable) == set(model.state_dict())
    assert all(torch.equal(value, portable[name]) for name, value in model.state_dict().items())
    for route in parallel.modules():
        if isinstance(route, HyperConnection):
            route.set_blend(1)
    load_consolidated_state_dict(parallel, portable)
    assert all(torch.equal(m.blend, torch.tensor(.015)) for m in parallel.modules()
               if isinstance(m, HyperConnection))
    with torch.no_grad():
        torch.testing.assert_close(parallel(sample()).logits, model(sample()).logits, rtol=2e-4, atol=2e-5)


@pytest.mark.parametrize('value', [-.1, 1.1, float('nan')])
def test_invalid_blend_refused(value):
    with pytest.raises(ValueError):
        HyperConnection(4, blend=value)
    with pytest.raises(ValueError):
        ResidualStreamConfig(routing='flash_next', blend=value)

"""Persistent widening: exact retrofit, learning, checkpoint replay and TP export."""
import copy

import pytest
import torch
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

from distillkit.anchor_tap import AnchorTap
from distillkit.configuration import DistillationRunConfig, ResidualStreamConfig
from distillkit.models import Qwen35SidecarForCausalLM, Qwen35WidenedForCausalLM
from distillkit.optimizers import architecture_metrics, architecture_parameter_ids, mixed_parameter_groups
from distillkit.tp_checkpoint import consolidated_state_dict, load_consolidated_state_dict
from distillkit.tp_model import shard_model
from distillkit.widened_residual import WidenedResidual, collapse_residual


@pytest.fixture(autouse=True)
def small_cpu_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def tiny_config(**kwargs):
    values = dict(vocab_size=64, hidden_size=32, intermediate_size=64,
                  num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
                  head_dim=8, linear_key_head_dim=8, linear_value_head_dim=8,
                  linear_num_key_heads=2, linear_num_value_heads=4,
                  linear_conv_kernel_dim=4, full_attention_interval=2,
                  tie_word_embeddings=True, max_position_embeddings=64,
                  pad_token_id=0, eos_token_id=3, use_cache=False,
                  residual_stream_num_branches=2, residual_stream_lowrank=8,
                  sidecar_num_heads=2, sidecar_head_dim=32, sidecar_num_branches=2)
    values.update(kwargs)
    return Qwen3_5TextConfig(**values)


def sample():
    return torch.tensor([[5, 8, 9, 3, 10, 12, 20, 6]])


def raw_rows():
    raw = torch.randint(0, 256, (1, 8, 2, 18), dtype=torch.uint8)
    raw[..., :2] = torch.full((1, 8, 2, 1), .001, dtype=torch.float16).view(torch.uint8)
    return raw


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("branches", [1, 2, 4])
def test_load_pretrained_exact_logits_and_norms(tmp_path, dtype, branches):
    torch.manual_seed(41)
    stock = Qwen3_5ForCausalLM(tiny_config(residual_stream_num_branches=branches)).to(dtype).eval()
    # Trained norm gains expose an incorrect reset/mapping that random-init tests miss.
    with torch.no_grad():
        for name, p in stock.named_parameters():
            if "layernorm" in name or name == "model.norm.weight":
                p.uniform_(-.3, .4)
    stock.save_pretrained(tmp_path)
    wide, info = Qwen35WidenedForCausalLM.from_pretrained(
        tmp_path, dtype=dtype, output_loading_info=True)
    wide.eval()
    assert info["missing_keys"] and not info["unexpected_keys"]
    assert all("_residual." in key for key in info["missing_keys"])
    for name, value in stock.state_dict().items():
        assert torch.equal(value, wide.state_dict()[name]), name
    with torch.no_grad():
        expected = stock(sample(), output_hidden_states=True)
        actual = wide(sample(), output_hidden_states=True)
    assert torch.equal(expected.logits, actual.logits)
    assert len(actual.hidden_states) == 4
    assert all(torch.equal(a, b) for a, b in zip(expected.hidden_states, actual.hidden_states))
    for module in wide.modules():
        if isinstance(module, WidenedResidual):
            assert module.lambda_read.item() == module.lambda_write.item() == 0
            # Offsets, not routes: zero here is the one-hot read and the all-one write.
            assert torch.equal(module.read_offset, torch.zeros_like(module.read_offset))
            assert torch.equal(module.write_offset, torch.zeros_like(module.write_offset))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("variant", ["gated_residual", "ple"])
def test_trained_sidecar_retains_exact_behavior(tmp_path, dtype, variant):
    torch.manual_seed(54)
    stock = Qwen35SidecarForCausalLM(tiny_config(sidecar_variant=variant)).to(dtype).eval()
    with torch.no_grad():
        for p in stock.model.layers[1].sidecar.parameters():
            p.add_(torch.randn_like(p) * .03)
    stock.save_pretrained(tmp_path)
    cfg = copy.deepcopy(stock.config)
    cfg.residual_stream_sidecar = True
    wide = Qwen35WidenedForCausalLM.from_pretrained(tmp_path, config=cfg, dtype=dtype).eval()
    raw = raw_rows()
    with torch.no_grad():
        expected = stock(sample(), ngram_raw=raw).logits
        actual = wide(sample(), ngram_raw=raw).logits
    assert torch.equal(actual, expected)
    for name, value in stock.state_dict().items():
        assert torch.equal(value, wide.state_dict()[name]), name


def test_identity_routes_are_stored_where_bfloat16_is_dense():
    """BF16 spacing near 1.0 is 0.0078, so a route parameter held at 1.0 cannot record
    the ~1e-5 steps this project trains with. The first widening stored the static
    write as all ones: after three real trainer steps its deviation was still exactly
    0 while lambda_write, which starts at zero, had reached 2.7e-5. Identity has to be
    the zero of an offset, not a literal one."""
    route = WidenedResidual(16, num_branches=2, lowrank=4, layer_idx=1).to(torch.bfloat16)
    for name, parameter in route.named_parameters():
        if name.startswith("W_"):
            continue  # random projections, deliberately not identity constants
        assert torch.count_nonzero(parameter) == 0, f"{name} does not start at zero"
        stepped = parameter.detach().clone().add_(1e-5)
        assert torch.count_nonzero(stepped) == parameter.numel(), (
            f"{name} cannot represent a 1e-5 step from its initial value")


def test_routes_learn_then_branches_diverge_and_gate_gradients_wake():
    torch.manual_seed(4)
    model = Qwen35WidenedForCausalLM(tiny_config()).train()
    model.freeze_backbone()
    route = model.model.layers[0].attn_residual
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.02)
    first = model(sample(), labels=sample()).loss
    first.backward()
    for module in model.modules():
        if isinstance(module, WidenedResidual):
            assert module.lambda_read.grad.abs() > 0
            assert module.lambda_write.grad.abs() > 0
            assert module.W_down.weight.grad is not None
            assert module.W_down.weight.grad.count_nonzero() == 0
    optimizer.step()
    optimizer.zero_grad()
    captured = []
    hook = model.model.layers[1].register_forward_hook(lambda m, a, o: captured.append(o.detach()))
    second = model(sample(), labels=sample()).loss
    second.backward()
    hook.remove()
    assert torch.isfinite(second)
    assert route.W_down.weight.grad.norm() > 0
    assert route.W_up.weight.grad.norm() > 0
    assert route.W_write.weight.grad.norm() > 0
    assert not torch.equal(captured[0][..., 0, :], captured[0][..., 1, :])
    assert all(p.grad is None for name, p in model.named_parameters()
               if name not in model.stage1_parameter_names())


@pytest.mark.parametrize("checkpointing", [None, False, True])
def test_hidden_state_tap_checkpoint_replay_and_backward(checkpointing):
    torch.manual_seed(32)
    reference = Qwen35WidenedForCausalLM(tiny_config()).train()
    with torch.no_grad():
        for module in reference.modules():
            if isinstance(module, WidenedResidual):
                module.lambda_read.fill_(.13)
                module.lambda_write.fill_(.17)
    model = copy.deepcopy(reference)
    reference.freeze_backbone()
    model.freeze_backbone()
    if checkpointing is not None:
        model.gradient_checkpointing_enable({"use_reentrant": checkpointing})
    with AnchorTap(model, [0, 1, 2, 3]) as tap:
        output = model(sample(), labels=sample(), output_hidden_states=True)
    for index, state in enumerate(output.hidden_states):
        assert state.shape == (1, 8, 32)
        assert torch.equal(tap.states()[index], state)
    # Reentrant checkpointing exposes detached intermediate hook outputs, as with
    # stock models; training uses non-reentrant checkpointing for anchor losses.
    loss = output.loss + output.hidden_states[-1].square().mean()
    loss.backward()
    expected = reference(sample(), labels=sample(), output_hidden_states=True)
    (expected.loss + expected.hidden_states[-1].square().mean()).backward()
    for name, parameter in model.named_parameters():
        expected_grad = dict(reference.named_parameters())[name].grad
        if parameter.requires_grad:
            assert parameter.grad is not None, name
            torch.testing.assert_close(parameter.grad, expected_grad, rtol=1e-4, atol=2e-6)


def test_save_reload_trained_routes_and_tp_portable_export(tmp_path):
    torch.manual_seed(14)
    model = Qwen35WidenedForCausalLM(tiny_config()).eval()
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, WidenedResidual):
                for p in module.parameters():
                    p.add_(torch.randn_like(p) * .02)
    expected = model(sample()).logits
    model.save_pretrained(tmp_path / "direct")
    loaded = Qwen35WidenedForCausalLM.from_pretrained(tmp_path / "direct").eval()
    assert torch.equal(loaded(sample()).logits, expected)
    for name, p in model.state_dict().items():
        assert torch.equal(p, loaded.state_dict()[name]), name
    tp = shard_model(copy.deepcopy(model), ["cpu", "cpu"])
    torch.testing.assert_close(tp(sample()).logits, expected, rtol=2e-4, atol=2e-5)
    portable = consolidated_state_dict(tp)
    assert set(portable) == set(model.state_dict())
    assert all(torch.equal(p, portable[n]) for n, p in model.state_dict().items())
    load_consolidated_state_dict(tp, portable)
    tp.save_pretrained(tmp_path / "portable", state_dict=portable)
    restored = Qwen35WidenedForCausalLM.from_pretrained(tmp_path / "portable").eval()
    assert torch.equal(restored(sample()).logits, expected)
    assert restored.config.residual_stream_num_branches == 2
    assert restored.config.residual_stream_lowrank == 8


def test_optimizer_routes_all_widening_to_adamw():
    model = Qwen35WidenedForCausalLM(tiny_config())
    route_ids = {id(p) for m in model.modules() if isinstance(m, WidenedResidual)
                 for p in m.parameters()}
    assert route_ids <= architecture_parameter_ids(model)
    metrics = architecture_metrics(model)
    assert metrics["architecture/model.layers.0.attn_residual/lambda_read"] == 0
    groups = mixed_parameter_groups(model)
    for group in groups:
        if group["optimizer_kind"] == "muon":
            assert not route_ids.intersection(id(p) for p in group["params"])
    model.freeze_backbone()
    assert {id(p) for p in model.parameters() if p.requires_grad} == route_ids


def test_config_validation_and_centered_collapse():
    assert ResidualStreamConfig().num_branches == 2
    for invalid in ({"num_branches": 0}, {"lowrank": 0}):
        with pytest.raises(ValueError):
            ResidualStreamConfig(**invalid)
    x = torch.randn(2, 7, 32, dtype=torch.bfloat16)
    for branches in (2, 3, 4):
        assert torch.equal(collapse_residual(x.unsqueeze(-2).expand(2, 7, branches, 32)), x)
    states = torch.randn(2, 7, 3, 32)
    torch.testing.assert_close(collapse_residual(states), states.mean(-2), atol=2e-7, rtol=2e-6)


def run_config(tmp_path, source, **kwargs):
    return DistillationRunConfig.model_validate(dict(
        model=str(source), dataset={},
        teacher={"kind": "dataset", "cache_path": str(tmp_path / "cache")},
        sequence_length=8, output_path=str(tmp_path / "out"),
        use_flash_attention=False, **kwargs))


def test_run_loader_default_off_and_widened_checkpoint_guards(tmp_path):
    from distillkit.main import load_student_model

    stock_path = tmp_path / "stock"
    Qwen3_5ForCausalLM(tiny_config()).save_pretrained(stock_path)
    config = run_config(tmp_path, stock_path)
    assert config.residual_stream is None
    stock = load_student_model(config, tokenizer_vocab_size=64)
    assert type(stock) is Qwen3_5ForCausalLM
    config = run_config(tmp_path, stock_path, residual_stream={"lowrank": 8},
                        optimizer={"strategy": "adamw", "freeze_backbone": True})
    wide = load_student_model(config, tokenizer_vocab_size=60)
    assert isinstance(wide, Qwen35WidenedForCausalLM)
    assert wide.config.vocab_size == 64
    wide_path = tmp_path / "wide"
    wide.save_pretrained(wide_path)
    restored = load_student_model(run_config(tmp_path, wide_path,
        residual_stream={"lowrank": 8}), tokenizer_vocab_size=64)
    assert torch.equal(restored(sample()).logits, wide(sample()).logits)
    for options, match in [({}, "requires its matching"),
                           ({"residual_stream": {"num_branches": 3, "lowrank": 8}}, "differs"),
                           ({"residual_stream": {"lowrank": 16}}, "differs"),
                           ({"residual_stream": {"lowrank": 8},
                             "sidecar": {"enabled": False}}, "differs")]:
        with pytest.raises(ValueError, match=match):
            load_student_model(run_config(tmp_path, wide_path, **options), tokenizer_vocab_size=64)


def test_partial_checkpoint_does_not_reset_loaded_routing_siblings(tmp_path):
    from safetensors.torch import load_file, save_file

    model = Qwen35WidenedForCausalLM(tiny_config())
    with torch.no_grad():
        model.model.layers[0].attn_residual.lambda_read.fill_(.25)
        model.model.layers[0].attn_residual.write_offset.fill_(.75)
    model.save_pretrained(tmp_path)
    state = load_file(str(tmp_path / "model.safetensors"))
    del state["model.layers.0.attn_residual.lambda_write"]
    save_file(state, str(tmp_path / "model.safetensors"), metadata={"format": "pt"})
    loaded = Qwen35WidenedForCausalLM.from_pretrained(tmp_path)
    route = loaded.model.layers[0].attn_residual
    assert route.lambda_read.item() == .25
    assert route.lambda_write.item() == 0
    assert torch.equal(route.write_offset, torch.full_like(route.write_offset, .75))


def test_decode_cache_and_explicit_embeddings_preserve_identity():
    torch.manual_seed(35)
    stock = Qwen3_5ForCausalLM(tiny_config(use_cache=True)).eval()
    wide = Qwen35WidenedForCausalLM(copy.deepcopy(stock.config)).eval()
    wide.load_state_dict(stock.state_dict(), strict=False)
    with torch.no_grad():
        for mode in ("full", "embeds"):
            inputs = {"input_ids": sample()} if mode == "full" else {
                "inputs_embeds": stock.get_input_embeddings()(sample())}
            assert torch.equal(stock(**inputs).logits, wide(**inputs).logits)
        a = stock(sample()[:, :4])
        b = wide(sample()[:, :4])
        assert torch.equal(a.logits, b.logits)
        for index in range(4, 8):
            a = stock(sample()[:, index:index+1], past_key_values=a.past_key_values)
            b = wide(sample()[:, index:index+1], past_key_values=b.past_key_values)
            assert torch.equal(a.logits, b.logits)


def test_middle_anchor_loss_backpropagates_with_checkpointing():
    model = Qwen35WidenedForCausalLM(tiny_config()).train()
    model.freeze_backbone()
    model.gradient_checkpointing_enable({"use_reentrant": False})
    with AnchorTap(model, [2]) as tap:
        model(sample())
    state = tap.states()[2]
    assert state.requires_grad and state.shape[-1] == 32
    state.square().mean().backward()
    assert model.model.layers[0].attn_residual.lambda_read.grad.abs() > 0

"""CPU gates for the frozen donor-reader transplant."""

from types import SimpleNamespace

import pytest
import torch

from distillkit.donor_reader import DonorReaderTransplant, initialise_transplant_reader

HIDDEN = FEATURES = 8


def tiny_model_config(**kwargs):
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    values = {
        "vocab_size": 64,
        "hidden_size": 64,
        "intermediate_size": 96,
        "num_hidden_layers": 3,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "linear_key_head_dim": 16,
        "linear_value_head_dim": 16,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 4,
        "linear_conv_kernel_dim": 4,
        "full_attention_interval": 2,
        "tie_word_embeddings": True,
        "max_position_embeddings": 64,
        "pad_token_id": 0,
        "eos_token_id": 3,
        "sidecar_num_heads": 2,
        "sidecar_head_dim": 32,
        "sidecar_num_branches": 3,
        "use_cache": False,
    }
    values.update(kwargs)
    return Qwen3_5TextConfig(**values)


def raw_batch(batch=1, length=8):
    raw = torch.randint(0, 256, (batch, length, 2, 18), dtype=torch.uint8)
    raw[..., :2] = torch.full((batch, length, 2, 1), 0.001, dtype=torch.float16).view(
        torch.uint8
    )
    return raw


def reader(value="donor", conv="donor", collapse="mixer", **kwargs):
    module = DonorReaderTransplant(
        HIDDEN,
        FEATURES,
        value_source=value,
        conv_source=conv,
        collapse=collapse,
        ngram_size=3,
        **kwargs,
    )
    streams = 4 if conv == "donor" else 2
    module.load_reader_weights(
        value_weight=torch.randn(HIDDEN, FEATURES),
        conv_weight=torch.randn(streams * HIDDEN, 1, 4),
        conv_norm_delta=torch.randn(streams, HIDDEN) * 0.01
        if conv == "donor"
        else None,
    )
    return module


def test_zero_rho_is_exact_identity_with_nonzero_frozen_reader():
    module = reader()
    hidden = torch.randn(2, 11, HIDDEN)
    rows = torch.randn(2, 11, FEATURES)
    actual = module(hidden, rows)
    assert torch.equal(actual, hidden)
    assert module.value_proj.weight.count_nonzero()
    assert module.conv1d.weight.count_nonzero()


def test_only_the_10240_style_mixer_and_rho_train():
    module = reader()
    trainable = {
        name: p.numel() for name, p in module.named_parameters() if p.requires_grad
    }
    assert trainable == {"mixer.weight": 4 * HIDDEN, "rho.weight": 1}
    assert not module.value_proj.weight.requires_grad
    assert not module.conv1d.weight.requires_grad
    assert not module.conv_norm_delta.requires_grad
    # There is structurally no key/query gate and no direct-value admission parameter.
    assert not hasattr(module, "key_proj")
    assert not hasattr(module, "gate")


def test_trainable_adapter_stays_fp32_when_frozen_reader_becomes_bfloat16():
    module = reader().to(torch.bfloat16)
    assert module.value_proj.weight.dtype == torch.bfloat16
    assert module.conv1d.weight.dtype == torch.bfloat16
    assert module.mixer.weight.dtype == torch.float32
    assert module.rho.weight.dtype == torch.float32


def test_all_factorial_arms_have_the_expected_frozen_stream_count():
    for value in ("c1", "donor"):
        for conv in ("c1", "donor"):
            collapse = "equal_mean" if conv == "c1" else "mixer"
            module = reader(value, conv, collapse)
            streams, collapsed = module.features(torch.randn(2, 7, FEATURES))
            assert streams.shape == (2, 7, 4 if conv == "donor" else 2, HIDDEN)
            assert collapsed.shape == (2, 7, HIDDEN)


def test_causal_dilation_uses_t_tminus3_tminus6_tminus9():
    module = DonorReaderTransplant(
        1,
        1,
        value_source="donor",
        conv_source="donor",
        collapse="single",
        single_stream=0,
        ngram_size=3,
    )
    conv = torch.zeros(4, 1, 4)
    # PyTorch Conv1d's last kernel cell is the current position after left padding.
    conv[0, 0] = torch.tensor([8.0, 4.0, 2.0, 1.0])
    module.load_reader_weights(value_weight=torch.ones(1, 1), conv_weight=conv)
    value = torch.arange(1.0, 12.0).view(1, 11, 1)
    got = module.conv_streams(value)[0, :, 0, 0]
    normed = value[0, :, 0] / torch.sqrt(value[0, :, 0].square() + module.eps)
    expected_pre_silu = (
        normed
        + 2 * torch.roll(normed, 3)
        + 4 * torch.roll(normed, 6)
        + 8 * torch.roll(normed, 9)
    )
    expected_pre_silu[:3] -= 2 * torch.roll(normed, 3)[:3]
    expected_pre_silu[:6] -= 4 * torch.roll(normed, 6)[:6]
    expected_pre_silu[:9] -= 8 * torch.roll(normed, 9)[:9]
    torch.testing.assert_close(got, torch.nn.functional.silu(expected_pre_silu))


@pytest.mark.parametrize(
    "collapse,kwargs,expected",
    [
        ("equal_mean", {}, [2.5]),
        ("single", {"single_stream": 2}, [3.0]),
        ("scalar", {"collapse_weights": [1, 0, -1, 0]}, [-2.0]),
        ("pca_rank1", {"collapse_weights": [0, 0.5, 0, 0.5]}, [3.0]),
    ],
)
def test_fixed_mathematical_collapses(collapse, kwargs, expected):
    module = DonorReaderTransplant(
        1, 1, value_source="donor", conv_source="donor", collapse=collapse, **kwargs
    )
    streams = torch.tensor([[[[1.0], [2.0], [3.0], [4.0]]]])
    assert module.collapse_streams(streams).flatten().tolist() == expected
    assert not module.mixer.weight.requires_grad


def test_loader_selects_value_and_conv_from_different_references(tmp_path):
    c1 = {
        "model.layers.1.sidecar.ple.value_proj.weight": torch.full(
            (HIDDEN, FEATURES), 2.0
        ),
        "model.layers.1.sidecar.ple.conv1d.weight": torch.full((2 * HIDDEN, 1, 4), 3.0),
    }
    donor = {
        "value_proj.weight": torch.full((HIDDEN, FEATURES), 5.0),
        "conv1d.weight": torch.full((4 * HIDDEN, 1, 4), 7.0),
        "norm_conv.weight": torch.full((4 * HIDDEN,), 0.25),
    }
    c1_path, donor_path = tmp_path / "c1.pt", tmp_path / "donor.pt"
    torch.save(c1, c1_path)
    torch.save(donor, donor_path)

    module = DonorReaderTransplant(
        HIDDEN, FEATURES, value_source="c1", conv_source="donor", collapse="mixer"
    )
    sidecar = SimpleNamespace(reader=module)
    model = SimpleNamespace(
        config=SimpleNamespace(sidecar_layer_index=0),
        model=SimpleNamespace(layers=[SimpleNamespace(sidecar=sidecar)]),
    )
    report = initialise_transplant_reader(
        model, c1_reference=c1_path, donor_reference=donor_path
    )
    assert report["value_source"] == "c1" and report["conv_source"] == "donor"
    assert torch.equal(module.value_proj.weight, c1[next(iter(c1))])
    assert torch.equal(module.conv1d.weight, donor["conv1d.weight"])
    assert torch.equal(module.conv_norm_delta.flatten(), donor["norm_conv.weight"])


def test_shape_mismatch_is_refused_instead_of_reshaped():
    module = DonorReaderTransplant(
        HIDDEN, FEATURES, value_source="donor", conv_source="donor"
    )
    with pytest.raises(ValueError, match="value projection"):
        module.load_reader_weights(
            value_weight=torch.randn(HIDDEN + 1, FEATURES),
            conv_weight=torch.randn(4 * HIDDEN, 1, 4),
        )


def test_model_variant_is_exactly_dormant_and_stage1_trains_only_adapter():
    from distillkit.models.qwen35_sidecar import Qwen35SidecarForCausalLM
    from distillkit.optimizers import freeze_backbone_for_stage1

    config = tiny_model_config(
        sidecar_variant="donor_reader",
        sidecar_value_source="donor",
        sidecar_conv_source="donor",
        sidecar_reader_collapse="mixer",
    )
    model = Qwen35SidecarForCausalLM(config).eval()
    sidecar = model.model.layers[1].sidecar
    sidecar.reader.load_reader_weights(
        value_weight=torch.randn(64, 64),
        conv_weight=torch.randn(4 * 64, 1, 4),
        conv_norm_delta=torch.randn(4, 64),
    )
    hidden = torch.randn(1, 8, 64)
    with torch.no_grad():
        enabled = sidecar(hidden, raw_batch(), sidecar_enabled=True)
        bypassed = sidecar(hidden, None, sidecar_enabled=False)
    assert torch.equal(enabled, bypassed)

    freeze_backbone_for_stage1(model)
    trainable = {
        name: p.numel() for name, p in model.named_parameters() if p.requires_grad
    }
    assert trainable == {
        "model.layers.1.sidecar.reader.mixer.weight": 4 * 64,
        "model.layers.1.sidecar.reader.rho.weight": 1,
    }


def test_transplant_refuses_a_widened_residual_consumer():
    from distillkit.models.qwen35_sidecar import Qwen35SidecarForCausalLM

    config = tiny_model_config(
        sidecar_variant="donor_reader", residual_stream_enabled=True
    )
    with pytest.raises(ValueError, match="single-stream transplant"):
        Qwen35SidecarForCausalLM(config)


def test_stock_checkpoint_materializes_zero_adapter_then_round_trips(tmp_path):
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    from distillkit.models.qwen35_sidecar import Qwen35SidecarForCausalLM

    config = tiny_model_config()
    Qwen3_5ForCausalLM(config).save_pretrained(tmp_path / "stock")
    config.sidecar_variant = "donor_reader"
    config.sidecar_value_source = "donor"
    config.sidecar_conv_source = "donor"
    config.sidecar_reader_collapse = "mixer"
    custom, info = Qwen35SidecarForCausalLM.from_pretrained(
        tmp_path / "stock", config=config, output_loading_info=True
    )
    assert info["missing_keys"] and all(
        ".sidecar." in key for key in info["missing_keys"]
    )
    reader = custom.model.layers[1].sidecar.reader
    assert reader.rho.weight.item() == 0.0
    assert torch.equal(reader.mixer.weight, torch.full_like(reader.mixer.weight, 0.25))
    assert reader.conv_norm_delta.count_nonzero() == 0
    reader.load_reader_weights(
        value_weight=torch.randn(64, 64),
        conv_weight=torch.randn(4 * 64, 1, 4),
        conv_norm_delta=torch.randn(4, 64),
    )
    trainable = {
        name: parameter.numel()
        for name, parameter in reader.named_parameters()
        if parameter.requires_grad
    }
    assert trainable == {"mixer.weight": 4 * 64, "rho.weight": 1}
    with torch.no_grad():
        reader.rho.weight.fill_(0.125)
        reader.mixer.weight.add_(torch.randn_like(reader.mixer.weight) * 0.01)
    custom.save_pretrained(tmp_path / "transplant")
    reloaded, info = Qwen35SidecarForCausalLM.from_pretrained(
        tmp_path / "transplant", output_loading_info=True
    )
    assert not info["missing_keys"] and not info["unexpected_keys"]
    reloaded_reader = reloaded.model.layers[1].sidecar.reader
    reloaded_reader.enforce_trainability()
    assert {
        name: parameter.numel()
        for name, parameter in reloaded_reader.named_parameters()
        if parameter.requires_grad
    } == {"mixer.weight": 4 * 64, "rho.weight": 1}
    for name, value in custom.state_dict().items():
        assert torch.equal(value, reloaded.state_dict()[name]), name


def test_run_config_requires_exact_reader_sources_and_c1_mean_collapse():
    from pydantic import ValidationError

    from distillkit.configuration import SidecarConfig

    with pytest.raises(ValidationError, match="reader_donor_reference"):
        SidecarConfig(table_path="table", variant="donor_reader")
    with pytest.raises(ValidationError, match="equal_mean"):
        SidecarConfig(
            table_path="table",
            variant="donor_reader",
            reader_value_source="c1",
            reader_conv_source="c1",
            reader_c1_reference="c1",
            reader_collapse="mixer",
        )
    arm = SidecarConfig(
        table_path="table",
        variant="donor_reader",
        reader_value_source="c1",
        reader_conv_source="donor",
        reader_c1_reference="c1",
        reader_donor_reference="donor",
        reader_collapse="equal_mean",
    )
    assert arm.reader_value_source == "c1" and arm.reader_conv_source == "donor"


def test_independent_eval_reports_content_layout_and_structural_from_one_forward():
    from distillkit.independent_eval import TextCollator, score_sequences

    class FakeModel:
        config = SimpleNamespace()

        def __call__(self, input_ids, logits_to_keep, **kwargs):
            return SimpleNamespace(
                logits=torch.zeros(
                    len(input_ids), len(logits_to_keep), 248_070, dtype=torch.float32
                )
            )

    feature = {
        "ids": [5, 198, 7, 8],
        "roles": {"template": [[0, 1]], "assistant": [[1, 4]]},
    }
    scored = score_sequences(FakeModel(), [feature], TextCollator(0), "enabled", "cpu")[
        0
    ]
    assert scored["by_role"]["layout"]["tokens"] == 1
    assert scored["by_role"]["content"]["tokens"] == 2
    assert scored["by_role"]["structural"]["tokens"] == 1

"""The PLE port must match upstream's formula and be exactly inert at initialization.

The point of the port is that its gate is computed from the data rather than learned
from scratch, so the tests that matter are: it is the identity at load (safe to retrofit
onto a trained frozen backbone), its gate actually varies with the agreement between
stream and n-gram, its convolution is causal, and its arithmetic reproduces upstream's
`Qwen4ExpTextPLELayer` term for term.
"""

import math

import pytest
import torch
from torch.nn import functional as F

from distillkit.ple_sidecar import PLESidecar

HIDDEN, FEATURES, SEQ = 32, 32, 12


def _module(seed=0):
    torch.manual_seed(seed)
    return PLESidecar(HIDDEN, FEATURES)


def _inputs(batch=2, seed=1):
    generator = torch.Generator().manual_seed(seed)
    return (torch.randn(batch, SEQ, HIDDEN, generator=generator),
            torch.randn(batch, SEQ, FEATURES, generator=generator))


def test_identity_at_initialization():
    """value_proj is zero-initialised, so a retrofit cannot disturb the backbone."""
    module = _module()
    hidden, features = _inputs()
    torch.testing.assert_close(module(hidden, features), hidden, rtol=0, atol=0)


def test_value_projection_receives_gradient_immediately():
    """The old design's failure mode was a gate with no gradient path. Here the value
    moves first, which is what opens the gate's own gradient."""
    module = _module()
    hidden, features = _inputs()
    module(hidden, features).square().mean().backward()
    assert module.value_proj.weight.grad.abs().sum() > 0
    # The gate's parameters wait for a nonzero value, exactly as documented.
    assert module.key_proj.weight.grad.abs().sum() == 0

    with torch.no_grad():
        module.value_proj.weight.normal_(std=0.02)
    module.zero_grad(set_to_none=True)
    module(hidden, features).square().mean().backward()
    assert module.key_proj.weight.grad.abs().sum() > 0, (
        "once the value is nonzero the gate must start learning"
    )


def test_the_gate_responds_to_agreement_between_stream_and_ngram():
    """The whole reason for the port: selectivity is computed, not learned."""
    module = _module()
    with torch.no_grad():
        module.value_proj.weight.normal_(std=0.02)
    hidden, features = _inputs()

    aligned = module.gate_report(hidden, features)
    # Flip the features' sign: every dot product flips, so the gate must move the
    # opposite way. A gate that ignored its inputs would not budge.
    opposed = module.gate_report(hidden, -features)
    assert abs(aligned["ple/gate_mean"] - opposed["ple/gate_mean"]) > 1e-6
    assert aligned["ple/gate_std"] > 0, "a constant gate is the failure being fixed"


def test_the_convolution_is_causal():
    """Position t must not see t+1: this runs inside a causal LM."""
    module = _module()
    with torch.no_grad():
        module.value_proj.weight.normal_(std=0.05)
        module.conv1d.weight.normal_(std=0.5)
    hidden, features = _inputs(batch=1)

    baseline = module(hidden, features)
    perturbed_features = features.clone()
    perturbed_features[:, SEQ - 1] += 5.0          # disturb only the last position
    perturbed = module(hidden, perturbed_features)

    changed = (baseline - perturbed).abs().amax(dim=-1)[0]
    assert changed[SEQ - 1] > 0, "the disturbed position itself must change"
    assert changed[: SEQ - 1].max() == 0, "no earlier position may see a later one"


def test_matches_upstream_formula_term_for_term():
    """Recomputed straight from Qwen4ExpTextPLELayer.forward with hc_count == 1."""
    module = _module()
    with torch.no_grad():
        module.value_proj.weight.normal_(std=0.05)
        module.key_proj.weight.normal_(std=0.05)
        module.conv1d.weight.normal_(std=0.3)
    hidden, features = _inputs()

    key_normed = module.norm_key(module.key_proj(features))
    value = module.value_proj(features)
    query_normed = module.norm_query(hidden)
    gate = (key_normed * query_normed).sum(dim=-1, keepdim=True) / math.sqrt(HIDDEN)
    gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
    gated_value = torch.sigmoid(gate) * value
    normed = module.norm_conv(gated_value)
    conv_in = F.pad(normed.transpose(1, 2), (module.short_conv_state_len, 0))
    # Same association as the module: h + (gated + conv). Float addition is not
    # associative, so (h + gated) + conv differs in the last bits and would force a
    # tolerance that could hide a real discrepancy.
    expected = hidden + (gated_value + F.silu(module.conv1d(conv_in)).transpose(1, 2))

    torch.testing.assert_close(module(hidden, features), expected, rtol=0, atol=0)


def test_dilation_follows_the_ngram_size():
    """Upstream dilates by ngram_size so a position sees the same phase of adjacent
    n-grams rather than blurring across them."""
    module = PLESidecar(HIDDEN, FEATURES, conv_kernel_size=4, ngram_size=3)
    assert module.conv1d.dilation == (3,)
    assert module.conv1d.groups == HIDDEN, "the convolution must stay depthwise"
    assert module.short_conv_state_len == 9

    other = PLESidecar(HIDDEN, FEATURES, conv_kernel_size=2, ngram_size=5)
    assert other.conv1d.dilation == (5,) and other.short_conv_state_len == 5


@pytest.mark.parametrize("bad", [{"hidden_size": 0}, {"feature_dim": -1},
                                 {"ngram_size": 0}, {"conv_kernel_size": 0}])
def test_rejects_degenerate_shapes(bad):
    kwargs = {"hidden_size": HIDDEN, "feature_dim": FEATURES, **bad}
    with pytest.raises(ValueError):
        PLESidecar(kwargs.pop("hidden_size"), kwargs.pop("feature_dim"), **kwargs)


def test_right_padding_cannot_disturb_real_positions():
    """Upstream masks padding before the convolution; we do not, because our collator
    right-pads and the convolution is causal, so padding sits strictly after every real
    token. This proves that rather than assuming it -- it is the case that would break
    silently now that arms train at batch 4."""
    module = _module()
    with torch.no_grad():
        module.value_proj.weight.normal_(std=0.05)
        module.conv1d.weight.normal_(std=0.5)
    hidden, features = _inputs(batch=1)
    real = 7

    unpadded = module(hidden[:, :real], features[:, :real])
    junk_features = features.clone()
    junk_features[:, real:] = 50.0        # arbitrary garbage in the padded tail
    padded = module(hidden, junk_features)

    torch.testing.assert_close(padded[:, :real], unpadded, rtol=0, atol=0)


def test_norm_weights_are_excluded_from_weight_decay():
    """Upstream's (1 + w) parameterisation makes decay pull the scale toward 1; this
    fork decays only ndim >= 2, so these 1-D weights are untouched either way."""
    from distillkit.optimizers import mixed_parameter_groups

    model = torch.nn.Module()
    model.ple = _module()
    decayed = {
        name for group in mixed_parameter_groups(model) if group["decay"]
        for name in group["param_names"]
    }
    norm_weights = [n for n, p in model.named_parameters() if "norm_" in n]
    assert norm_weights, "fixture should contain the PLE norms"
    assert not (set(norm_weights) & decayed)


def test_norm_scale_can_still_move_in_bfloat16():
    """The reason these are not nn.RMSNorm. bf16 spacing near 1.0 is 0.0078, so a scale
    stored directly cannot represent a deviation smaller than ~0.004 -- at this fork's
    1e-4 learning rate the norms would sit frozen at exactly 1.0. Storing the deviation
    puts it near zero, where the same update survives.
    """
    module = _module().to(torch.bfloat16)
    hidden, features = _inputs()
    hidden, features = hidden.bfloat16(), features.bfloat16()
    before = module(hidden, features).clone()

    with torch.no_grad():                      # an update the size of one optimizer step
        for norm in (module.norm_key, module.norm_query, module.norm_conv):
            norm.weight += torch.full_like(norm.weight, 1e-3)
        module.value_proj.weight.normal_(std=0.02)

    assert module.norm_key.weight.float().abs().max() > 0, (
        "a 1e-3 update to the norm weight was rounded away; the scale is unlearnable"
    )
    assert not torch.equal(module(hidden, features), before)

    # And the deviation is faithfully reflected in the effective scale.
    scale = (1.0 + module.norm_key.weight.float())
    assert abs(scale.mean().item() - 1.001) < 1e-4


def _student(variant):
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    from distillkit.models.qwen35_sidecar import Qwen35SidecarForCausalLM

    config = Qwen3_5TextConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8, linear_key_head_dim=8,
        linear_value_head_dim=8, linear_num_key_heads=2, linear_num_value_heads=4,
        linear_conv_kernel_dim=4, full_attention_interval=2, tie_word_embeddings=True,
        max_position_embeddings=64, pad_token_id=0, eos_token_id=3, use_cache=False,
    )
    config.sidecar_num_heads, config.sidecar_head_dim = 4, 32
    config.sidecar_variant = variant
    torch.manual_seed(0)
    return Qwen35SidecarForCausalLM(config).eval()


def _raw(model, batch=2, seq=6):
    sidecar = model.model.layers[model.config.sidecar_layer_index].sidecar
    return torch.zeros(batch, seq, sidecar.num_heads, sidecar.bytes_per_head, dtype=torch.uint8)


def test_variant_selects_the_ple_module_and_loads_inert():
    """Retrofitting onto a trained backbone must not perturb it, so an enabled sidecar
    at initialisation has to be indistinguishable from a disabled one."""
    from distillkit.models.qwen35_sidecar import _NGramSidecar, _PLENGramSidecar

    model = _student("ple")
    sidecar = model.model.layers[model.config.sidecar_layer_index].sidecar
    assert isinstance(sidecar, _PLENGramSidecar)
    assert isinstance(_student("gated_residual").model.layers[1].sidecar, _NGramSidecar)

    ids = torch.randint(0, 64, (2, 6))
    raw = _raw(model)
    with torch.no_grad():
        enabled = model(input_ids=ids, ngram_raw=raw, sidecar_enabled=True).logits
        disabled = model(input_ids=ids, ngram_raw=raw, sidecar_enabled=False).logits
    torch.testing.assert_close(enabled, disabled, rtol=0, atol=0)


def test_stage1_freezing_leaves_exactly_the_ple_module_trainable():
    """The attribute is still named `sidecar`, so _auxiliary_parameter_ids recognises it
    and no optimizer change is needed to train this variant."""
    from distillkit.optimizers import freeze_backbone_for_stage1

    model = _student("ple")
    freeze_backbone_for_stage1(model)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert len(trainable) == 6, trainable
    assert all(".sidecar.ple." in name for name in trainable), trainable


def test_rejects_an_unknown_variant():
    with pytest.raises(ValueError, match="sidecar_variant"):
        _student("something_else")


def test_norms_survive_from_pretrained_reinitialisation(tmp_path):
    """from_pretrained re-initialises missing parameters, and its default for a
    norm-shaped weight is ones. Under upstream's (1 + w) that is a scale of 2.0, so the
    retrofit would start by doubling every normalised value. Caught on a real load, not
    by constructing the module directly."""
    from distillkit.models.qwen35_sidecar import Qwen35SidecarForCausalLM

    source = _student("gated_residual")          # a checkpoint with no ple.* keys at all
    source.save_pretrained(tmp_path)
    config = source.config
    config.sidecar_variant = "ple"
    model = Qwen35SidecarForCausalLM.from_pretrained(tmp_path, config=config)

    ple = model.model.layers[config.sidecar_layer_index].sidecar.ple
    for name in ("norm_key", "norm_query", "norm_conv"):
        weight = getattr(ple, name).weight
        assert weight.abs().max().item() == 0.0, (
            "%s came back at scale %.3f; the deviation parameterisation expects 0"
            % (name, 1.0 + weight.abs().max().item())
        )
    assert ple.value_proj.weight.abs().max().item() == 0.0

"""The frozen-checkpoint gate diagnostic.

Everything this reports is read off a reimplementation of ``_admission`` that stops
before the sigmoid, because the diagnostic needs the score itself. If that drifts from
the module by a normalisation constant the AUCs and the fitted scalars describe a
different model than the checkpoint, and nothing would say so. The script asserts
bit-equality on a real checkpoint; these pin the same thing without a GPU, and pin the
sign convention that decides which way every oracle verdict reads.
"""

import numpy as np
import pytest
import torch

from distillkit.ple_gated_sidecar import DirectionGatedPLESidecar

gate_diagnosis = pytest.importorskip("scratch.gate_diagnosis")


@pytest.fixture
def module():
    torch.manual_seed(0)
    sidecar = DirectionGatedPLESidecar(16, 32, hc_count=2, gate_directions=2)
    # Identity initialisation zeroes both write paths, so a forward comparison would
    # pass for any gate at all. Give them weight.
    torch.nn.init.normal_(sidecar.value_proj.weight, std=0.05)
    torch.nn.init.normal_(sidecar.conv1d.weight, std=0.05)
    with torch.no_grad():
        sidecar.sharpness_delta.add_(torch.randn_like(sidecar.sharpness_delta) * 0.1)
    return sidecar


@pytest.fixture
def inputs():
    torch.manual_seed(1)
    return torch.randn(2, 6, 2, 16), torch.randn(2, 6, 32)


def test_the_score_is_the_module_s_admission_before_its_sigmoid(module, inputs):
    stream, _ = inputs
    score = gate_diagnosis.admission_score(module, stream)
    rebuilt = 2.0 * torch.sigmoid(score).mean(-1, keepdim=True)
    assert torch.equal(rebuilt, module._admission(stream)), "the score is not the module's"


def test_the_calibrated_forward_at_neutral_is_the_module_s_forward(module, inputs):
    stream, features = inputs
    neutral = gate_diagnosis.Calibration()
    assert torch.equal(gate_diagnosis.calibrated_forward(module, neutral, stream, features),
                       module(stream, features))


def test_sharpness_enters_before_the_signed_square_root(module, inputs):
    """Doubling `sharpness_delta`'s multiplier only multiplies the sigmoid's logit by
    sqrt(2), because signed_sqrt(s*raw) = sqrt(s)*signed_sqrt(raw). The diagnostic's
    temperature is applied *after* the root for exactly this reason, so the two are not
    interchangeable and a fitted temperature cannot be folded back into sharpness."""
    stream, _ = inputs
    with torch.no_grad():
        module.sharpness_delta.zero_()
        base = gate_diagnosis.admission_score(module, stream)
        module.sharpness_delta.fill_(1.0)          # multiplier 1 -> 2
        doubled = gate_diagnosis.admission_score(module, stream)
    assert torch.allclose(doubled, base * np.sqrt(2.0), atol=1e-5)


def test_the_temperature_scales_the_logit_itself(module, inputs):
    stream, features = inputs
    hot = gate_diagnosis.Calibration()
    with torch.no_grad():
        hot.raw_temperature.fill_(2.0)
    score = gate_diagnosis.admission_score(module, stream)
    expected = 2.0 * torch.sigmoid(2.0 * score).mean(-1, keepdim=True)
    written = gate_diagnosis.calibrated_forward(module, hot, stream, features) - stream
    value = module.value_proj(features)
    conv = module._short_conv(value)
    assert torch.allclose(written - conv, expected * value.unsqueeze(-2), atol=1e-6)


def test_alpha_and_beta_scale_the_two_write_paths_independently(module, inputs):
    """beta ~ 0 is the verdict that the ungated convolution is hurting, so it has to
    actually be the convolution it turns off and not the value write."""
    stream, features = inputs
    no_conv = gate_diagnosis.Calibration()
    with torch.no_grad():
        no_conv.raw_beta.zero_()
    written = gate_diagnosis.calibrated_forward(module, no_conv, stream, features) - stream
    gated_value = module._admission(stream) * module.value_proj(features).unsqueeze(-2)
    assert torch.allclose(written, gated_value, atol=1e-6)

    no_value = gate_diagnosis.Calibration()
    with torch.no_grad():
        no_value.raw_alpha.zero_()
    written = gate_diagnosis.calibrated_forward(module, no_value, stream, features) - stream
    assert torch.allclose(written, module._short_conv(module.value_proj(features)), atol=1e-6)


def test_the_projection_keeps_the_scales_interpretable():
    """alpha, beta < 0 would answer a different question -- whether an inverted sidecar
    helps -- and a temperature at or below zero inverts the gate's ranking."""
    cal = gate_diagnosis.Calibration()
    with torch.no_grad():
        cal.raw_alpha.fill_(-3.0)
        cal.raw_beta.fill_(-0.5)
        cal.raw_temperature.fill_(-2.0)
    cal.project()
    assert cal.raw_alpha.item() == 0.0
    assert cal.raw_beta.item() == 0.0
    assert cal.raw_temperature.item() > 0


def test_a_positive_oracle_gradient_means_closing_would_have_helped(module, inputs):
    """The sign convention the whole report is read through. The probe is additive and
    starts at zero, so its gradient is dL/dg at the checkpoint's own operating point."""
    stream, features = inputs
    cal = gate_diagnosis.Calibration()
    gate_delta, _ = cal.arm_probes((2, 6, 2, 1), stream.device)
    # A loss that rises with the written value: more gate must cost more.
    output = gate_diagnosis.calibrated_forward(module, cal, stream, features)
    target = module.value_proj(features).unsqueeze(-2)
    loss = (output * target.sign() * target.abs()).sum()
    loss.backward()
    assert (gate_delta.grad > 0).all(), "dL/dg should be positive when the value hurts"


def test_layout_tokens_are_not_in_the_grading_set():
    from scratch.row_novelty import LAYOUT_TOKEN_IDS
    record = {"ids": [1, 2, 198, 4, 248068, 6, 7], "roles": {"assistant": [[1, 7]]}}
    targets = gate_diagnosis.content_targets(record)
    assert list(targets) == [1, 3, 5, 6]
    assert not set(record["ids"][i] for i in targets) & set(LAYOUT_TOKEN_IDS)


def test_auc_is_the_probability_a_positive_outranks_a_negative():
    assert gate_diagnosis.auc([3.0, 2.0, 1.0], [True, False, False]) == 1.0
    assert gate_diagnosis.auc([1.0, 2.0, 3.0], [True, False, False]) == 0.0
    assert gate_diagnosis.auc([1.0, 1.0], [True, False]) == 0.5, "ties count as half"
    assert np.isnan(gate_diagnosis.auc([1.0, 2.0], [True, True])), "one class is no AUC"


def test_tied_scores_share_their_mean_rank():
    """An untrained gate can return the same score for long runs of tokens; ranking
    those arbitrarily would manufacture an AUC away from 0.5 out of input order."""
    ranks = gate_diagnosis.average_ranks(np.array([5.0, 1.0, 5.0, 1.0]))
    assert list(ranks) == [3.5, 1.5, 3.5, 1.5]


def test_logistic_regression_recovers_a_separating_direction():
    rng = np.random.default_rng(0)
    features = rng.normal(size=(400, 2))
    labels = (features[:, 0] > 0).astype(np.float32)
    weight, _, _, _ = gate_diagnosis.fit_logistic(features, labels)
    assert weight[0] > 1.0 and abs(weight[1]) < abs(weight[0]) / 2


def test_each_kernel_index_reads_the_position_the_tap_map_claims(module):
    """The tap ablation's whole conclusion turns on which index is the instantaneous
    tap. The module pads (K-1)*dilation on the left, so output[t] = sum_k w[k] x[t+3k-9]
    and index 3 is `t` -- but that is a derivation, and an off-by-one would reverse the
    reading. An impulse settles it."""
    length, taps = 24, module.conv1d.weight.shape[-1]
    for index in range(taps):
        with torch.no_grad():
            module.conv1d.weight.zero_()
            module.conv1d.weight[..., index] = 1.0
        value = torch.zeros(1, length, module.hidden_size)
        value[0, 12] = 1.0                      # an impulse at t = 12
        responded = module._short_conv(value)[0, :, 0].abs().sum(-1).nonzero().ravel()
        assert responded.numel(), "tap %d produced no output at all" % index
        offset = gate_diagnosis.TAP_OFFSETS[index]
        assert responded[0].item() == 12 - offset, (
            "kernel index %d first responds at t=%d, so it reads t%+d, not t%+d"
            % (index, responded[0].item(), -(responded[0].item() - 12), offset))

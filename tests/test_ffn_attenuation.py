"""Alpha attenuation has to be exactly what it claims before its curve means anything.

The anomaly this supports -- that suppressing one layer's FFN update improves held-out
content NLL on familiar contexts -- is large enough to be suspicious. If the intervention
is not precisely `h + alpha * mlp(norm(h))` at exactly the selected positions, the whole
alpha-response curve is an artefact of the harness rather than a property of the model.
"""

import pytest
import torch

from distillkit.ffn_skip import attenuate_ffn, capture_ffn, skip_ffn
from tests.test_ffn_skip import batch, build, logits_of


def test_alpha_one_is_the_stock_model():
    model = build()
    ids = batch(model)
    reference = logits_of(model, ids)
    with attenuate_ffn(model, [1, 2]) as handle:
        for layer in (1, 2):
            handle.set(layer, torch.ones_like(ids, dtype=torch.bool), 1.0)
        assert torch.equal(reference, logits_of(model, ids))
        assert handle.scaled_calls == 0, "alpha 1 should not count as an intervention"


def test_no_mask_is_the_stock_model():
    model = build()
    ids = batch(model)
    reference = logits_of(model, ids)
    with attenuate_ffn(model, [0, 1, 2]):
        assert torch.equal(reference, logits_of(model, ids))


def test_alpha_zero_matches_the_previous_zeroing_intervention():
    """The two code paths must agree, or the anomaly cannot be compared to its origin."""
    model = build()
    ids = batch(model)
    mask = torch.zeros_like(ids, dtype=torch.bool)
    mask[:, 2:5] = True

    with skip_ffn(model, [1]) as skip:
        skip.mask = mask
        zeroed = logits_of(model, ids)
    with attenuate_ffn(model, [1]) as handle:
        handle.set(1, mask, 0.0)
        attenuated = logits_of(model, ids)
    assert torch.equal(zeroed, attenuated)


def test_alpha_admits_exactly_that_fraction_of_the_update():
    """Checked against the captured update itself, not against another intervention."""
    model = build()
    ids = batch(model, batch_size=1, length=6)
    with capture_ffn(model, [1]) as captured:
        logits_of(model, ids)
    stock_update = captured[1][0][1].clone()

    mask = torch.ones_like(ids, dtype=torch.bool)
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.25):
        with attenuate_ffn(model, [1]) as handle:
            handle.set(1, mask, alpha)
            with capture_ffn(model, [1]) as observed:
                logits_of(model, ids)
        scaled = observed[1][0][1]
        assert torch.allclose(scaled, stock_update * alpha, atol=1e-3), alpha


def test_only_selected_positions_are_touched():
    model = build()
    ids = batch(model, batch_size=1, length=8)
    with capture_ffn(model, [1]) as captured:
        logits_of(model, ids)
    stock_update = captured[1][0][1].clone()

    mask = torch.zeros_like(ids, dtype=torch.bool)
    mask[0, 3] = True
    with attenuate_ffn(model, [1]) as handle:
        handle.set(1, mask, 0.5)
        with capture_ffn(model, [1]) as observed:
            logits_of(model, ids)
    scaled = observed[1][0][1]
    assert torch.allclose(scaled[0, 3], stock_update[0, 3] * 0.5, atol=1e-3)
    for position in (0, 1, 2, 4, 5, 6, 7):
        assert torch.equal(scaled[0, position], stock_update[0, position]), position


def test_attention_is_untouched_and_the_past_is_causal():
    model = build()
    ids = batch(model, batch_size=1, length=8)
    reference = logits_of(model, ids)
    layer = model.model.layers[1]
    attention = layer.linear_attn if hasattr(layer, "linear_attn") else layer.self_attn
    seen = []
    hook = attention.register_forward_hook(
        lambda module, inputs, output: seen.append(
            (output[0] if isinstance(output, tuple) else output).detach().clone()))
    mask = torch.zeros_like(ids, dtype=torch.bool)
    mask[0, 4] = True
    try:
        logits_of(model, ids)
        with attenuate_ffn(model, [1]) as handle:
            handle.set(1, mask, 0.5)
            changed = logits_of(model, ids)
    finally:
        hook.remove()

    assert torch.equal(seen[0], seen[1]), "the FFN branch only"
    assert torch.equal(reference[0, 3], changed[0, 3]), "an earlier position"
    assert not torch.equal(reference[0, 4], changed[0, 4]), "the attenuated position"
    assert not torch.equal(reference[0, 7], changed[0, 7]), "a later position"


def test_the_weights_never_move():
    model = build()
    ids = batch(model)
    before = {name: parameter.detach().clone()
              for name, parameter in model.named_parameters()}
    with attenuate_ffn(model, [0, 1, 2]) as handle:
        for layer in (0, 1, 2):
            handle.set(layer, torch.ones_like(ids, dtype=torch.bool), 0.25)
        logits_of(model, ids)
    for name, parameter in model.named_parameters():
        assert torch.equal(before[name], parameter), name
    assert torch.equal(logits_of(model, ids), logits_of(model, ids))


def test_a_bad_mask_shape_is_refused():
    model = build()
    ids = batch(model)
    with attenuate_ffn(model, [1]) as handle:
        handle.set(1, torch.zeros(ids.shape[0], ids.shape[1] + 2, dtype=torch.bool), 0.5)
        with pytest.raises(ValueError, match="does not match"):
            logits_of(model, ids)


def test_an_unmanaged_layer_is_refused():
    model = build()
    ids = batch(model)
    with attenuate_ffn(model, [1]) as handle:
        with pytest.raises(ValueError, match="not intervened on"):
            handle.set(2, torch.zeros_like(ids, dtype=torch.bool), 0.5)

"""Borrowing Flash-Next's hyper-connection routing as an initialisation.

The property worth a test here is not that the tensors arrive -- that is a copy loop --
but that arriving *changes the forward*. ``_combine`` multiplies the routing matrices by
``lambda_read`` / ``lambda_write``, both of which initialise to zero, so a transfer that
copies the matrices and leaves the lambdas alone is inert: the run trains, reports
numbers, and nothing was borrowed. ``branch_gain_delta`` rides on the norm instead and
does take effect, so that failure would be half-silent.
"""

import json

import pytest
import torch

from distillkit.borrowed_routing import (
    initialise_widened_residual, layer_map,
)
from distillkit.widened_residual import WidenedResidual


BRANCHES, HIDDEN, LOWRANK = 4, 8, 6
TENSORS = ("W_down.weight", "W_up.weight", "W_write.weight", "branch_gain_delta")


def _donor(directory, blocks, seed=0):
    """A donor set shaped like the extractor's output."""
    generator = torch.Generator().manual_seed(seed)
    for block in blocks:
        held = {}
        for sublayer in ("attn_residual", "mlp_residual"):
            held["%s.W_down.weight" % sublayer] = torch.randn(
                LOWRANK, BRANCHES * HIDDEN, generator=generator)
            held["%s.W_up.weight" % sublayer] = torch.randn(
                BRANCHES * HIDDEN, LOWRANK, generator=generator)
            held["%s.W_write.weight" % sublayer] = torch.randn(
                BRANCHES, BRANCHES * HIDDEN, generator=generator)
            held["%s.branch_gain_delta" % sublayer] = torch.randn(
                BRANCHES, HIDDEN, generator=generator) * 0.1
        torch.save(held, directory / ("layer-%02d.pt" % block))
    (directory / "manifest.json").write_text(json.dumps(
        {"blocks": list(blocks), "branches": BRANCHES, "hidden": HIDDEN}), encoding="utf-8")
    return directory


class _Layer(torch.nn.Module):
    def __init__(self, index):
        super().__init__()
        self.attn_residual = WidenedResidual(HIDDEN, BRANCHES, LOWRANK, index)
        self.mlp_residual = WidenedResidual(HIDDEN, BRANCHES, LOWRANK, index)


class _Model(torch.nn.Module):
    def __init__(self, layers=4, sidecar_layer_index=2):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList(_Layer(i) for i in range(layers))
        self.config = type("C", (), {"sidecar_layer_index": sidecar_layer_index})()


def test_layer_map_proportional_spans_the_donor_stack():
    """Our 32 onto their 48: first to first, last to last, monotone between."""
    mapping = layer_map(32, list(range(48)), "proportional")
    assert mapping[0] == 0 and mapping[31] == 47
    assert all(mapping[i] <= mapping[i + 1] for i in range(31))
    # The layer the depth sweep chose, recorded so a change to the rule is visible.
    assert mapping[24] == 36


def test_layer_map_identity_uses_the_donor_bottom():
    mapping = layer_map(4, list(range(48)), "identity")
    assert mapping == {0: 0, 1: 1, 2: 2, 3: 3}
    with pytest.raises(ValueError, match="identity map needs"):
        layer_map(8, [0, 1, 2], "identity")


def test_unknown_map_is_refused():
    with pytest.raises(ValueError, match="unknown init_layer_map"):
        layer_map(4, [0, 1, 2, 3], "sideways")


def test_borrowing_copies_every_routing_tensor(tmp_path):
    model = _Model()
    donor = _donor(tmp_path, range(8))
    report = initialise_widened_residual(model, donor, "proportional")
    assert report["tensors_copied"] == 4 * 2 * len(TENSORS)
    held = torch.load(donor / ("layer-%02d.pt" % report["sidecar_layer_donor"]),
                      map_location="cpu", weights_only=False)
    module = model.model.layers[2].attn_residual
    for name in TENSORS:
        assert torch.equal(module.get_parameter(name), held["attn_residual.%s" % name])


def test_borrowing_turns_the_lambdas_on_or_the_matrices_are_multiplied_by_zero(tmp_path):
    """The bug this module exists to avoid. See the module docstring."""
    model = _Model()
    module = model.model.layers[0].attn_residual
    assert float(module.lambda_read) == 0.0, "a fresh widening starts on the identity route"

    torch.manual_seed(0)
    states = torch.randn(2, 3, BRANCHES, HIDDEN)
    norm = torch.nn.RMSNorm(HIDDEN)
    with torch.no_grad():
        before = module.read(states, norm)[0].clone()
        # Exactly the inert transfer: the matrices, and nothing else.
        for name in ("W_down.weight", "W_up.weight", "W_write.weight"):
            module.get_parameter(name).normal_(generator=torch.Generator().manual_seed(1))
        inert = module.read(states, norm)[0]
    assert torch.equal(before, inert), (
        "if this ever differs, _combine no longer multiplies the routing by lambda and "
        "the warning in borrowed_routing is stale")

    initialise_widened_residual(model, _donor(tmp_path, range(8)), "proportional")
    assert float(module.lambda_read) == 1.0
    assert float(module.lambda_write) == 1.0
    with torch.no_grad():
        after = module.read(states, norm)[0]
    assert not torch.allclose(before, after), "the borrow did not reach the forward"


def test_a_width_mismatch_is_refused_rather_than_reshaped(tmp_path):
    model = _Model()
    model.model.layers[0].attn_residual = WidenedResidual(HIDDEN, 2, LOWRANK, 0)
    with pytest.raises(ValueError, match="num_branches and lowrank must match"):
        initialise_widened_residual(model, _donor(tmp_path, range(8)), "proportional")


def test_a_missing_extraction_says_how_to_make_one(tmp_path):
    with pytest.raises(FileNotFoundError, match="extract_flashnext_hc"):
        initialise_widened_residual(_Model(), tmp_path, "proportional")

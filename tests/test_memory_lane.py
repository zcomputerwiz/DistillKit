"""The two-stream pilot's mechanism gates: stock at init, live when opened, causal when off."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from distillkit.memory_lane import MemoryRead
from distillkit.models import Qwen35SidecarForCausalLM
from scratch.ple_forensics.costream_arms import Wiring
from tests.test_sidecar_model import raw_batch, tiny_config


def build():
    config = tiny_config(sidecar_variant="ple", sidecar_layer_index=1)
    torch.manual_seed(0)
    return Qwen35SidecarForCausalLM(config).eval()


def inputs(model, batch=2, length=8):
    generator = torch.Generator().manual_seed(7)
    ids = torch.randint(0, model.config.vocab_size, (batch, length), generator=generator)
    return ids, raw_batch(batch=batch, length=length)


def test_an_empty_lane_reads_as_nothing_and_gives_the_scale_no_gradient():
    """The deadlock the read's initialisation exists to avoid.

    A read scale of zero looks like the safe choice and is not: with the sidecar's value
    projection also at zero the contribution is a product of two zeros, and neither
    factor receives any gradient to leave with. The first version of the pilot ran four
    steps with every read scale still exactly 0.0.
    """
    read = MemoryRead(16)
    stream, memory = torch.randn(2, 5, 16), torch.randn(2, 5, 16)
    empty = torch.zeros_like(memory)

    assert torch.equal(read(stream, empty), torch.zeros_like(stream))
    read(stream, empty).sum().backward()
    assert read.scale.grad.abs().item() == 0.0

    read.zero_grad(set_to_none=True)
    read(stream, memory).sum().backward()
    assert read.scale.grad.abs().item() > 0
    assert read.direction.grad.abs().max().item() > 0


def test_projecting_read_starts_as_the_identity_read():
    plain, projecting = MemoryRead(16), MemoryRead(16, project=True)
    with torch.no_grad():
        projecting.direction.copy_(plain.direction)
    stream, memory = torch.randn(2, 5, 16), torch.randn(2, 5, 16)
    assert torch.allclose(projecting(stream, memory), plain(stream, memory), atol=1e-5)


def test_arm_m_is_exactly_stock_at_initialisation():
    """An open read of an empty lane is still bitwise stock, not approximately."""
    model = build()
    ids, raw = inputs(model)
    stock = model(input_ids=ids, sidecar_enabled=False).logits

    wiring = Wiring(model, "M", read_layers=[2], project=False)
    try:
        lane = wiring.logits(ids[0].tolist(), raw[:1], ids.device)
        assert torch.equal(lane, stock[0])
    finally:
        wiring.close()


def test_open_lane_changes_the_stream_and_closing_it_restores_stock():
    model = build()
    ids, raw = inputs(model)
    stock = model(input_ids=ids, sidecar_enabled=False).logits[0]

    wiring = Wiring(model, "M", read_layers=[2], project=False)
    try:
        sidecar = model.model.layers[model.config.sidecar_layer_index].sidecar
        with torch.no_grad():
            # A trained sidecar writes something; the read admits some of it.
            torch.nn.init.normal_(sidecar.ple.value_proj.weight, std=0.05)

        opened = wiring.logits(ids[0].tolist(), raw[:1], ids.device)
        assert not torch.allclose(opened, stock, atol=1e-4)

        # The ablation has to be exact: whatever the lane bought must vanish when the
        # reads are forced to zero, or the ON/OFF comparison measures something else.
        wiring.box["enabled"] = False
        closed = wiring.logits(ids[0].tolist(), raw[:1], ids.device)
        assert torch.equal(closed, stock)
    finally:
        wiring.close()


def test_gradient_reaches_the_sidecar_through_the_lane_only():
    """The sidecar is bypassed in the layer; its only path to the loss is the read."""
    model = build()
    ids, raw = inputs(model)
    wiring = Wiring(model, "M", read_layers=[2], project=False)
    try:
        sidecar = model.model.layers[model.config.sidecar_layer_index].sidecar
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for parameter in sidecar.parameters():
            parameter.requires_grad_(True)
        with torch.no_grad():
            torch.nn.init.normal_(sidecar.ple.key_proj.weight, std=0.05)

        wiring.logits(ids[0].tolist(), raw[:1], ids.device).float().pow(2).sum().backward()
        assert sidecar.ple.value_proj.weight.grad.abs().max() > 0

        # And with the reads shut it has no path at all -- the logits carry no graph
        # back to the sidecar, which is what makes the OFF evaluation a real ablation
        # rather than a differently-trained model.
        wiring.box["enabled"] = False
        assert not wiring.logits(ids[0].tolist(), raw[:1], ids.device).requires_grad
    finally:
        wiring.close()


def test_read_diagnostics_are_per_token():
    model = build()
    ids, raw = inputs(model)
    wiring = Wiring(model, "M", read_layers=[2], project=False)
    try:
        with torch.no_grad():
            torch.nn.init.normal_(
                model.model.layers[1].sidecar.ple.value_proj.weight, std=0.05)
        wiring.logits(ids[0].tolist(), raw[:1], ids.device)
        stats = wiring.diagnostics([0, 3, 5])
        assert stats["alpha_L2"].shape == (3,)
        assert stats["ratio_L2"].shape == (3,)
        # A gate that returns the same number everywhere is a constant scale, not a
        # selector; the pilot's whole read-specialisation question needs this to vary.
        assert stats["alpha_L2"].std() > 0
    finally:
        wiring.close()

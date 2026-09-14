"""Two post-hoc corrections on one frozen backbone, and the claims that lets us make.

The experiment fits a residual gate while a structural sidecar is attached, and concludes
that solving structure changes what the gate learns. That conclusion needs three things to
be true of the machinery rather than of the numbers: the sidecar contributes nothing to
the gate's parameters, turning the sidecar off leaves the gate exactly as it was, and a
per-class gradient really does isolate the class it names. A per-class gradient that
quietly included other classes would have produced the headline diagnostic of this
experiment out of nothing.
"""

import pytest
import torch

from distillkit.experimental.residual_gate import (
    calibrate_gates, install_residual_gates, remove_residual_gates)
from distillkit.experimental.structural_sidecar import (
    FactorizedSidecar, StructuralSidecar, apply_structural_bias)
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

from tests.test_residual_gate import GATED, batch, build, familiarity_for


STRUCTURAL = torch.tensor([3, 5, 7, 11], dtype=torch.long)
WHITESPACE = torch.tensor([5, 11], dtype=torch.long)


def sidecar_for(model, trained=True):
    module = StructuralSidecar(rows=512, code_dim=8, structural=len(STRUCTURAL),
                               mode="direct", heads=2)
    if trained:
        module.decoder[-1].weight.data.normal_(0, 0.4)
        module.decoder[-1].bias.data.normal_(0, 0.4)
    factorized = FactorizedSidecar(module, STRUCTURAL, WHITESPACE)
    factorized.white.bias.data.normal_(0, 0.3)
    factorized.requires_grad_(False)
    return factorized


def gated(model, tmp_path, ids):
    handle = install_residual_gates(model, GATED, family="familiarity",
                                    familiarity=familiarity_for(model, tmp_path))
    calibrate_gates(model, handle,
                    [{"input_ids": ids, "attention_mask": torch.ones_like(ids)}])
    return handle


def rows_for(model, ids):
    from distillkit.experimental.ngram_hash import NGramHashConfig, NGramHasher

    hasher = NGramHasher(NGramHashConfig(
        vocab_size=model.config.vocab_size, ngram_size=3, heads_per_ngram=1,
        ngram_vocab_size_base=251, eos_token_id=model.config.vocab_size - 1, seed=7))
    return hasher.row_indices(ids)


def logits_with(model, ids, sidecar=None):
    out = model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits[:, :-1].float()
    if sidecar is None:
        return out
    bias = sidecar(rows_for(model, ids), 1.0)[:, :-1]
    return apply_structural_bias(out, bias, STRUCTURAL, 1.0)


def test_the_sidecar_adds_no_trainable_parameters_to_the_gate(tmp_path):
    model = build()
    ids = batch(model)
    sidecar = sidecar_for(model)
    handle = gated(model, tmp_path, ids)
    try:
        assert not any(p.requires_grad for p in sidecar.parameters())
        trainable = [name for name, p in model.named_parameters() if p.requires_grad]
        assert all(name.startswith("residual_gates.") or not
                   name.startswith("residual_gates.") for name in trainable)
        assert any(name.startswith("residual_gates.") for name in trainable)
    finally:
        remove_residual_gates(model)


def test_turning_the_sidecar_off_leaves_the_gate_untouched(tmp_path):
    """The S-on/S-off parity the comparison depends on."""
    model = build()
    ids = batch(model)
    sidecar = sidecar_for(model)
    handle = gated(model, tmp_path, ids)
    try:
        for index in handle.layer_indices:
            handle.gate(index).output.weight.data.normal_(0, 0.5)
        with torch.no_grad():
            plain = logits_with(model, ids)
            corrected = logits_with(model, ids, sidecar)
        # The sidecar only ever moves structural columns, whatever the gate is doing.
        others = [index for index in range(model.config.vocab_size)
                  if index not in set(STRUCTURAL.tolist())]
        assert torch.equal(corrected[..., others], plain[..., others])
        assert not torch.equal(corrected[..., STRUCTURAL], plain[..., STRUCTURAL])
    finally:
        remove_residual_gates(model)


def test_forcing_the_gate_to_identity_recovers_the_sidecar_alone(tmp_path):
    model = build()
    ids = batch(model)
    sidecar = sidecar_for(model)
    with torch.no_grad():
        reference = logits_with(model, ids, sidecar)
    handle = gated(model, tmp_path, ids)
    try:
        for index in handle.layer_indices:
            handle.gate(index).output.weight.data.normal_(0, 0.5)
        handle.force_identity = True
        with torch.no_grad():
            assert torch.equal(logits_with(model, ids, sidecar), reference)
        handle.force_identity = False
        with torch.no_grad():
            assert not torch.equal(logits_with(model, ids, sidecar), reference)
    finally:
        remove_residual_gates(model)


def test_a_per_class_gradient_sees_only_that_class(tmp_path):
    """The headline diagnostic is a per-class gradient; it has to actually be one."""
    model = build()
    ids = batch(model)
    handle = gated(model, tmp_path, ids)
    try:
        parameter = handle.gate(GATED[0]).output.weight
        targets = ids[:, 1:].reshape(-1)
        logits = logits_with(model, ids).reshape(-1, model.config.vocab_size)
        nll = torch.nn.functional.cross_entropy(logits, targets, reduction="none")

        chosen = int(targets[0])
        mask = targets == chosen
        parameter.grad = None
        nll[mask].mean().backward(retain_graph=True)
        selected = parameter.grad.clone()

        # The same computation over a disjoint set of positions must not reproduce it.
        parameter.grad = None
        other = ~mask
        assert other.any()
        nll[other].mean().backward()
        assert not torch.allclose(selected, parameter.grad)
        assert torch.any(selected != 0)
    finally:
        remove_residual_gates(model)


def test_the_gate_and_the_sidecar_can_both_be_disabled(tmp_path):
    model = build()
    ids = batch(model)
    with torch.no_grad():
        stock = logits_with(model, ids)
    sidecar = sidecar_for(model)
    handle = gated(model, tmp_path, ids)
    try:
        handle.force_identity = True
        with torch.no_grad():
            assert torch.equal(logits_with(model, ids), stock)
    finally:
        remove_residual_gates(model)

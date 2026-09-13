"""An oracle measurement is worthless if the intervention is not the one described.

Three things have to hold before any number from this harness means anything: skipping
nothing is the stock model bit for bit, skipping something changes only the MLP's
contribution, and the change propagates causally instead of being quietly patched over.
"""

import os

import pytest
import torch

from distillkit.ffn_skip import (estimate_savings, mlp_flops_per_token,
                                 model_flops_per_token, skip_ffn)
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

from tests.test_sidecar_model import tiny_config


def build(seed=0):
    # The stock backbone, which is what the oracle study actually measures.
    config = tiny_config()
    torch.manual_seed(seed)
    model = Qwen3_5ForCausalLM(config).eval()
    model.config.use_cache = False
    return model


def batch(model, batch_size=2, length=10, seed=3):
    generator = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, model.config.vocab_size, (batch_size, length), generator=generator)
    return ids


def logits_of(model, ids, **kwargs):
    with torch.no_grad():
        return model(input_ids=ids, attention_mask=torch.ones_like(ids), **kwargs).logits


def test_an_empty_selection_is_the_stock_model():
    model = build()
    ids = batch(model)
    reference = logits_of(model, ids)
    with skip_ffn(model, []):
        assert torch.equal(reference, logits_of(model, ids))


def test_a_mask_of_all_false_is_the_stock_model():
    """The wrapper runs on every layer here, so this pins the wrapper itself."""
    model = build()
    ids = batch(model)
    reference = logits_of(model, ids)
    with skip_ffn(model, range(model.config.num_hidden_layers)) as handle:
        handle.mask = torch.zeros_like(ids, dtype=torch.bool)
        assert torch.equal(reference, logits_of(model, ids))
        assert handle.skipped_calls == 0
        assert handle.total_calls == ids.numel() * model.config.num_hidden_layers


def test_skipping_changes_the_logits_and_counts_what_it_skipped():
    model = build()
    ids = batch(model)
    reference = logits_of(model, ids)
    mask = torch.zeros_like(ids, dtype=torch.bool)
    mask[:, 3] = True
    with skip_ffn(model, [1, 2]) as handle:
        handle.mask = mask
        changed = logits_of(model, ids)
        assert handle.skipped_calls == int(mask.sum()) * 2
    assert not torch.equal(reference, changed)


def test_only_the_mlp_residual_is_bypassed():
    """Attention still runs: its output must be identical, skip or no skip."""
    model = build()
    ids = batch(model)
    captured = []
    layer = model.model.layers[1]
    attention = layer.linear_attn if hasattr(layer, "linear_attn") else layer.self_attn
    handle_hook = attention.register_forward_hook(
        lambda module, inputs, output: captured.append(
            (output[0] if isinstance(output, tuple) else output).detach().clone()))
    try:
        logits_of(model, ids)
        mask = torch.ones_like(ids, dtype=torch.bool)
        with skip_ffn(model, [1]) as handle:
            handle.mask = mask
            logits_of(model, ids)
    finally:
        handle_hook.remove()
    assert len(captured) == 2
    # Layer 1's attention sees the same input either way -- nothing upstream changed --
    # so its output must match exactly. Only what happens after it differs.
    assert torch.equal(captured[0], captured[1])


def test_the_skip_propagates_to_later_positions():
    """A skip at position t must change position t+1, or the study measures nothing.

    Patching the full-compute state back in after each token would make the numbers look
    clean and answer a question nobody asked. What matters is the sequence-level cost of
    having skipped, which only shows up causally.
    """
    model = build()
    ids = batch(model, batch_size=1, length=8)
    reference = logits_of(model, ids)
    mask = torch.zeros_like(ids, dtype=torch.bool)
    mask[0, 2] = True
    with skip_ffn(model, [0, 1]) as handle:
        handle.mask = mask
        changed = logits_of(model, ids)
    assert not torch.equal(reference[0, 2], changed[0, 2]), "the skipped position"
    assert not torch.equal(reference[0, 5], changed[0, 5]), "a later position"
    # Earlier positions cannot see it: the model is causal.
    assert torch.equal(reference[0, 0], changed[0, 0])
    assert torch.equal(reference[0, 1], changed[0, 1])


def test_the_weights_are_untouched_and_restored():
    model = build()
    ids = batch(model)
    before = {name: parameter.detach().clone()
              for name, parameter in model.named_parameters()}
    with skip_ffn(model, [0, 1]) as handle:
        handle.mask = torch.ones_like(ids, dtype=torch.bool)
        logits_of(model, ids)
    for name, parameter in model.named_parameters():
        assert torch.equal(before[name], parameter), name
    # And the forwards are back, so the next measurement is not silently intervened on.
    assert torch.equal(logits_of(model, ids), logits_of(model, ids))


def test_a_layer_outside_the_model_is_refused():
    model = build()
    with pytest.raises(ValueError, match="outside the model"):
        with skip_ffn(model, [model.config.num_hidden_layers]):
            pass


def test_a_mask_of_the_wrong_shape_is_refused():
    model = build()
    ids = batch(model)
    with skip_ffn(model, [0]) as handle:
        handle.mask = torch.zeros(ids.shape[0], ids.shape[1] + 1, dtype=torch.bool)
        with pytest.raises(ValueError, match="does not match"):
            logits_of(model, ids)


# --- the arithmetic behind the savings ----------------------------------------


def test_the_flops_account_is_the_swiglu_arithmetic():
    model = build()
    config = model.config
    per_token = mlp_flops_per_token(config)
    assert per_token == 2 * 3 * config.hidden_size * config.intermediate_size

    flops = model_flops_per_token(model, config)
    assert flops["layers"] == config.num_hidden_layers
    assert flops["mlp"] == per_token * config.num_hidden_layers
    assert flops["total"] == flops["mlp"] + flops["other"] + flops["head"]

    # Skipping every MLP of every layer saves the whole MLP budget and no more.
    everything = estimate_savings(flops, flops["layers"] * 100, 100)
    assert everything["ffn_flops_fraction"] == pytest.approx(1.0)
    assert everything["total_flops_fraction"] < 1.0
    half = estimate_savings(flops, flops["layers"] * 50, 100)
    assert half["ffn_flops_fraction"] == pytest.approx(0.5)


# --- residual substitution ----------------------------------------------------


def test_substituting_the_real_residual_reproduces_the_model():
    """The load-bearing equivalence: a perfect cache hit must change nothing.

    If substituting the residual the MLP would have produced is not the stock model, then
    every later measurement of a cache is measuring the harness instead of the cache.
    """
    from distillkit.ffn_skip import capture_ffn, substitute_ffn

    model = build()
    ids = batch(model)
    reference = logits_of(model, ids)

    with capture_ffn(model, [1, 2]) as captured:
        logits_of(model, ids)
    residuals = {layer: captured[layer][0][1].clone() for layer in (1, 2)}

    with substitute_ffn(model, [1, 2]) as handle:
        for layer in (1, 2):
            handle.set(layer, torch.ones_like(ids, dtype=torch.bool), residuals[layer])
        replayed = logits_of(model, ids)
        assert handle.replaced_calls == ids.numel() * 2
    assert torch.equal(reference, replayed)


def test_no_substitution_is_the_stock_model():
    from distillkit.ffn_skip import substitute_ffn

    model = build()
    ids = batch(model)
    reference = logits_of(model, ids)
    with substitute_ffn(model, [0, 1, 2]) as handle:
        assert torch.equal(reference, logits_of(model, ids))
        assert handle.replaced_calls == 0
        handle.set(0, torch.zeros_like(ids, dtype=torch.bool),
                   torch.zeros(*ids.shape, model.config.hidden_size))
        assert torch.equal(reference, logits_of(model, ids))


def test_a_wrong_residual_changes_the_future_and_not_the_past():
    from distillkit.ffn_skip import capture_ffn, substitute_ffn

    model = build()
    ids = batch(model, batch_size=1, length=8)
    reference = logits_of(model, ids)
    with capture_ffn(model, [1]) as captured:
        logits_of(model, ids)
    residual = captured[1][0][1].clone()

    mask = torch.zeros_like(ids, dtype=torch.bool)
    mask[0, 3] = True
    wrong = residual.clone()
    wrong[0, 3] = residual[0, 0]              # a residual from a different position
    with substitute_ffn(model, [1]) as handle:
        handle.set(1, mask, wrong)
        changed = logits_of(model, ids)
    assert not torch.equal(reference[0, 3], changed[0, 3])
    assert not torch.equal(reference[0, 6], changed[0, 6])
    assert torch.equal(reference[0, 2], changed[0, 2])


def test_substitution_leaves_attention_and_the_weights_alone():
    from distillkit.ffn_skip import substitute_ffn

    model = build()
    ids = batch(model)
    before = {name: parameter.detach().clone()
              for name, parameter in model.named_parameters()}
    layer = model.model.layers[1]
    attention = layer.linear_attn if hasattr(layer, "linear_attn") else layer.self_attn
    captured = []
    hook = attention.register_forward_hook(
        lambda module, inputs, output: captured.append(
            (output[0] if isinstance(output, tuple) else output).detach().clone()))
    try:
        logits_of(model, ids)
        with substitute_ffn(model, [1]) as handle:
            handle.set(1, torch.ones_like(ids, dtype=torch.bool),
                       torch.randn(*ids.shape, model.config.hidden_size))
            logits_of(model, ids)
    finally:
        hook.remove()
    assert torch.equal(captured[0], captured[1])
    for name, parameter in model.named_parameters():
        assert torch.equal(before[name], parameter), name


def test_capture_does_not_change_the_model():
    from distillkit.ffn_skip import capture_ffn

    model = build()
    ids = batch(model)
    reference = logits_of(model, ids)
    with capture_ffn(model, [0, 1, 2]) as captured:
        assert torch.equal(reference, logits_of(model, ids))
    assert all(len(captured[layer]) == 1 for layer in (0, 1, 2))
    # And the captured output is what the layer's MLP actually produced.
    hidden_in, residual_out = captured[1][0]
    with torch.no_grad():
        assert torch.equal(model.model.layers[1].mlp(hidden_in), residual_out)


def test_the_cache_corpus_never_overlaps_the_evaluation_bundle():
    """Split hygiene, checked against the real selection logic rather than by eye."""
    import hashlib
    import json
    import tempfile

    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scratch" / "ffn_memo"))
    from repeatability import cache_documents

    texts = ["alpha beta gamma", "delta epsilon", "zeta eta theta"]
    digests = {hashlib.sha256(texts[1].encode("utf-8")).hexdigest()}

    with tempfile.TemporaryDirectory() as directory:
        source = os.path.join(directory, "corpus.jsonl")
        with open(source, "w", encoding="utf-8") as handle:
            for text in texts:
                handle.write(json.dumps({"text": text}) + "\n")

        class Tokenizer:
            def __call__(self, text):
                return {"input_ids": [ord(c) % 97 for c in text]}

        kept = cache_documents(source, Tokenizer(), digests, 10, 512)
    assert len(kept) == 2, "the excluded document was still cached"

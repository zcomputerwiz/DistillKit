"""The chunked token path computes what the dense gathered path computes.

`_forward_tokens` exists so memory grows with the sequence rather than its square; it is
only worth having if it is the same function. Checked against `_forward_gathered` with a
chunk far smaller than the sequence, so every chunk boundary is exercised.

Exact agreement needs a selection with no ties at the cut. The index score is a ReLU sum,
so a position every head scores at or below zero ties at exactly 0, and which of those
fills the rest of a budget is whatever `topk` does at that tensor shape -- the dense and
chunked paths legitimately differ there. So agreement of the logits, gradients and
indexer loss is checked with a budget that covers every position, and the chunked
selection itself is checked for being a top-k, ties allowed.
"""
import sys

import pytest
import torch

from distillkit.models import Qwen35WidenedForCausalLM
from distillkit.models.qwen35.csa2 import (Qwen35SparseLatentAttention, isolated_indexer,
                                            recorded_attention)
from test_csa2_routing import csa2_config

LENGTH, CHUNK = 150, 32


def build(top_k=LENGTH + 10, chunk=CHUNK):
    torch.manual_seed(0)
    config = csa2_config(csa2_router_bias=False, csa2_top_k=top_k, csa2_local_window=8,
                         csa2_query_chunk=chunk)
    model = Qwen35WidenedForCausalLM(config).double()
    layers = [l.self_attn for l in model.model.layers
              if isinstance(getattr(l, "self_attn", None), Qwen35SparseLatentAttention)]
    return model, layers


def force_dense(monkeypatch, dense):
    if dense:
        monkeypatch.setattr(Qwen35SparseLatentAttention, "_token_path",
                            lambda self, cache: False)
    else:
        monkeypatch.undo()


def run(model, tokens, dense, monkeypatch):
    force_dense(monkeypatch, dense)
    model.zero_grad(set_to_none=True)
    out = model(input_ids=tokens, labels=tokens, use_cache=False)
    out.loss.backward()
    grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    return out.logits.detach(), grads


def test_logits_selection_and_gradients_match_the_dense_path(monkeypatch):
    model, layers = build()
    tokens = torch.randint(1, 64, (2, LENGTH))
    dense_logits, dense_grads = run(model, tokens, True, monkeypatch)
    dense_allowed = [l.last_allowed.clone() for l in layers]
    token_logits, token_grads = run(model, tokens, False, monkeypatch)
    assert all(l.last_selection is not None for l in layers), "token path did not run"
    for layer, allowed in zip(layers, dense_allowed):
        assert torch.equal(layer.last_allowed, allowed), layer.layer_idx
    assert torch.allclose(token_logits, dense_logits, atol=1e-9)
    assert dense_grads.keys() == token_grads.keys()
    for name in dense_grads:
        assert torch.allclose(token_grads[name], dense_grads[name], atol=1e-9), name


def test_the_chunked_selection_is_a_top_k_with_its_window(monkeypatch):
    """With a small budget: every query reads its window, plus its top_k positive scores."""
    model, layers = build(top_k=24)
    first = layers[0]
    grab = {}
    original = Qwen35SparseLatentAttention._select_positions

    def spy(self, queries, keys, weights):
        if self is first:
            grab["scores"] = torch.einsum("bqhd,bkd->bhqk", queries, keys).relu().mul(
                weights.permute(0, 2, 1).unsqueeze(-1)).sum(1)
        return original(self, queries, keys, weights)

    monkeypatch.setattr(Qwen35SparseLatentAttention, "_select_positions", spy)
    tokens = torch.randint(1, 64, (2, LENGTH))
    with torch.no_grad():
        model(input_ids=tokens, use_cache=False)
    allowed, scores = first.last_allowed, grab["scores"]
    q = torch.arange(LENGTH).view(-1, 1)
    offsets = q - torch.arange(LENGTH).view(1, -1)
    window = (offsets >= 0) & (offsets < first.local_window)
    causal = offsets >= 0
    assert not bool((allowed & ~causal).any()), "reads the future"
    assert bool((allowed | ~window).all()), "window not read"
    for b in range(2):
        for row in range(LENGTH):
            reach = causal[row]
            picked = allowed[b, row] & reach
            if int(reach.sum()) <= first.top_k:
                # Everything fits: all of it that scored, and the window regardless.
                assert torch.equal(picked, reach & ((scores[b, row] > 0) | window[row]))
                continue
            # The top-k over every causal position: the kth best score is the bar, and
            # everything strictly above it is in, whether or not the window also holds it.
            bar = scores[b, row][reach].topk(first.top_k).values[-1]
            assert bool(picked[reach & (scores[b, row] > bar) & (scores[b, row] > 0)].all()), (b, row)
            # Beyond the window, nothing below the bar got in, and nothing no head scored.
            outside = picked & ~window[row]
            assert bool((scores[b, row][outside] >= bar).all()), (b, row)
            assert bool((scores[b, row][outside] > 0).all()), (b, row)
            assert int(outside.sum()) <= first.top_k


def test_statistics_match_the_expanded_selection():
    model, layers = build(top_k=24)
    tokens = torch.randint(1, 64, (1, LENGTH))
    with torch.no_grad():
        model(input_ids=tokens, use_cache=False)
    for layer in layers:
        compact = layer.routing_statistics()
        layer.last_allowed = layer.last_allowed  # expand, dropping the compact form
        expanded = layer.routing_statistics()
        for key in ("density", "selected", "entropy"):
            assert compact[key] == pytest.approx(expanded[key], abs=1e-9), (layer.layer_idx, key)


def test_the_sparse_stage_indexer_loss_matches_the_dense_path(monkeypatch):
    sys.path.insert(0, "scratch/dense_gr")
    from indexer_kl import indexer_loss, routing_layers, watch

    model, _ = build()
    tokens = torch.randint(1, 64, (1, LENGTH))
    mask = torch.ones(1, LENGTH, dtype=torch.bool)
    mask[:, -1] = False
    stage = routing_layers(model)

    def loss(dense):
        force_dense(monkeypatch, dense)
        model.zero_grad(set_to_none=True)
        seen, handles = watch(model, stage)
        with recorded_attention(model), isolated_indexer(model):
            model.model(input_ids=tokens, attention_mask=torch.ones_like(tokens),
                        use_cache=False)
            for handle in handles:
                handle.remove()
            targets = {i: a.last_attention for i, a in stage}
            chosen = {i: a.last_allowed for i, a in stage}
            borrowed = {i: a.bus.require_latent(a.latent_donor, i) for i, a in stage}
            value = indexer_loss(model, stage, seen, targets, borrowed, selected=chosen,
                                 query_mask=mask)
        value.backward()
        return float(value), {n: p.grad.clone() for n, p in model.named_parameters()
                              if p.grad is not None}

    dense_value, dense_grads = loss(True)
    token_value, token_grads = loss(False)
    assert token_value == pytest.approx(dense_value, rel=1e-9)
    assert dense_grads.keys() == token_grads.keys() and dense_grads
    for name in dense_grads:
        assert torch.allclose(token_grads[name], dense_grads[name], atol=1e-9), name

"""The optimized native-PLE execution path has to be the same arithmetic, not nearly.

Three things are being changed at once for speed -- the frozen prefix leaves autograd,
gradient checkpointing runs non-reentrant, and the chunked head folds the 248k-wide
projection -- and each of them is a place where a run could quietly train something
slightly different. Forward equivalence is not enough on its own: what the experiment
actually depends on is that the *trainable* gradients are unchanged.
"""

import pytest
import torch

from distillkit.frozen_prefix import no_grad_prefix
from distillkit.models import Qwen35SidecarForCausalLM
from distillkit.native_ple import native_hash_config
from distillkit.ngram_hash import NGramHasher
from tests.test_sidecar_model import tiny_config

TRACKED = ("rho", "table", "key_proj", "value_proj", "conv1d")


def build(seed=0, layer_index=1):
    config = tiny_config(sidecar_variant="ple", sidecar_table_mode="native",
                         sidecar_ngram_vocab_size_base=97,
                         sidecar_layer_index=layer_index)
    torch.manual_seed(seed)
    model = Qwen35SidecarForCausalLM(config)
    model.config.use_cache = False
    sidecar = model.model.layers[layer_index].sidecar
    # The regime under test: everything frozen except the sidecar, and rho already open
    # so the internals carry gradient.
    model.requires_grad_(False)
    sidecar.requires_grad_(True)
    with torch.no_grad():
        sidecar.rho.fill_(0.25)
    return model, sidecar


def batch(model, batch_size=2, length=12, seed=7):
    generator = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, model.config.vocab_size, (batch_size, length), generator=generator)
    rows = NGramHasher(native_hash_config(model.config)).row_indices(ids)
    return ids, rows


def tracked_of(sidecar):
    return {"rho": sidecar.rho, "table": sidecar.table.weight,
            "key_proj": sidecar.ple.key_proj.weight,
            "value_proj": sidecar.ple.value_proj.weight,
            "conv1d": sidecar.ple.conv1d.weight}


def step(model, sidecar, ids, rows):
    """One CE forward/backward; returns the loss and every trainable gradient."""
    model.zero_grad(set_to_none=True)
    logits = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                   ngram_ids=rows).logits[:, :-1].float()
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1))
    loss.backward()
    return float(loss), {name: parameter.grad.detach().clone()
                         for name, parameter in tracked_of(sidecar).items()}


def assert_same(left, right, tolerance=1e-5):
    loss_left, grads_left = left
    loss_right, grads_right = right
    assert loss_left == pytest.approx(loss_right, abs=tolerance)
    for name in TRACKED:
        assert torch.allclose(grads_left[name], grads_right[name], atol=tolerance), name


def test_the_no_grad_prefix_changes_nothing_but_the_bookkeeping():
    model, sidecar = build()
    ids, rows = batch(model)

    ordinary = step(model, sidecar, ids, rows)
    restore = no_grad_prefix(model, upto_layer=model.config.sidecar_layer_index)
    try:
        optimized = step(model, sidecar, ids, rows)
    finally:
        restore()
    assert_same(ordinary, optimized)


def test_the_prefix_really_leaves_the_graph():
    model, sidecar = build()
    ids, rows = batch(model)
    restore = no_grad_prefix(model, upto_layer=model.config.sidecar_layer_index)
    try:
        embeddings = []
        handle = model.model.embed_tokens.register_forward_hook(
            lambda mod, inputs, output: embeddings.append(output))
        step(model, sidecar, ids, rows)
        handle.remove()
    finally:
        restore()
    # The embedding output carries no graph at all, which is the point: with
    # `enable_input_require_grads` installed it would, and every prefix activation would
    # be retained for a backward pass that cannot use them.
    assert embeddings and not embeddings[0].requires_grad
    for name, parameter in model.model.embed_tokens.named_parameters():
        assert parameter.grad is None, name
    for name, parameter in model.model.layers[0].named_parameters():
        assert parameter.grad is None, name


def test_a_prefix_that_is_not_actually_frozen_is_refused():
    """Otherwise this silently discards the gradients it was asked to compute."""
    model, _ = build()
    model.model.layers[0].requires_grad_(True)
    with pytest.raises(ValueError, match="still has trainable parameters"):
        no_grad_prefix(model, upto_layer=1)


def test_restoring_puts_the_graph_back():
    model, sidecar = build()
    ids, rows = batch(model)
    restore = no_grad_prefix(model, upto_layer=1)
    restore()
    model.model.embed_tokens.weight.requires_grad_(True)
    step(model, sidecar, ids, rows)
    assert model.model.embed_tokens.weight.grad is not None


# --- gradient checkpointing --------------------------------------------------


def test_non_reentrant_checkpointing_matches_no_checkpointing():
    model, sidecar = build()
    ids, rows = batch(model)
    restore = no_grad_prefix(model, upto_layer=model.config.sidecar_layer_index)
    try:
        plain = step(model, sidecar, ids, rows)
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        model.train()
        checkpointed = step(model, sidecar, ids, rows)
        model.gradient_checkpointing_disable()
    finally:
        restore()
    assert_same(plain, checkpointed)


def test_non_reentrant_checkpointing_needs_no_input_grad_hack():
    """The legacy workaround forces the embedding to require grad; this path must not.

    Reentrant checkpointing drops the graph when a checkpointed block's inputs do not
    require grad, which is why `freeze_backbone` installs `enable_input_require_grads`.
    Non-reentrant checkpointing has no such requirement, and installing the hook anyway
    would defeat the no-grad prefix by making every prefix activation differentiable.
    """
    model, sidecar = build()
    ids, rows = batch(model)
    assert not hasattr(model, "_require_grads_hook")
    restore = no_grad_prefix(model, upto_layer=model.config.sidecar_layer_index)
    try:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        model.train()
        loss, grads = step(model, sidecar, ids, rows)
        model.gradient_checkpointing_disable()
    finally:
        restore()
    assert torch.isfinite(torch.tensor(loss))
    for name in TRACKED:
        assert grads[name].abs().sum() > 0, name


# --- chunked head ------------------------------------------------------------


def test_the_chunked_head_gives_the_same_native_ple_gradients():
    """The folded head is what makes a 248k-wide vocabulary affordable at seq 4096.

    `tests/test_chunked_ce.py` already pins the loss value against the stock
    implementation; what is added here is that the *native PLE* gradients that reach the
    table through that head are unchanged, since they are what the experiment measures.
    """
    from distillkit.chunked_ce import chunked_causal_lm_loss

    model, sidecar = build()
    ids, rows = batch(model)
    labels = ids.clone()

    def run(loss_function):
        model.loss_function = loss_function
        model.zero_grad(set_to_none=True)
        out = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                    ngram_ids=rows, labels=labels)
        out.loss.backward()
        return float(out.loss), {name: parameter.grad.detach().clone()
                                 for name, parameter in tracked_of(sidecar).items()}

    stock = model.loss_function
    try:
        ordinary = run(stock)
        chunked = run(chunked_causal_lm_loss)
    finally:
        model.loss_function = stock
    assert_same(ordinary, chunked, tolerance=1e-5)


def test_freeze_backbone_can_decline_the_input_grad_hook():
    """The hook belongs to reentrant checkpointing and nothing else.

    Installed alongside the no-grad prefix it would defeat it: the embedding output would
    require grad again and every prefix activation would be retained.
    """
    config = tiny_config(sidecar_variant="ple", sidecar_table_mode="native",
                         sidecar_ngram_vocab_size_base=97, sidecar_layer_index=1)
    torch.manual_seed(0)
    model = Qwen35SidecarForCausalLM(config)

    model.freeze_backbone(input_require_grads=False)
    assert not hasattr(model, "_require_grads_hook")
    no_grad_prefix(model, upto_layer=1)()          # accepts the frozen prefix

    model.freeze_backbone()                        # the default is still the old behaviour
    assert hasattr(model, "_require_grads_hook")
    model.unfreeze_backbone()

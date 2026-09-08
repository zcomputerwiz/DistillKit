"""Tapped anchors must equal the ones `output_hidden_states=True` would have returned.

The tap exists to stop retaining and copying 33 hidden states for the 2 that are read.
Its whole risk is an off-by-one: `hidden_states[i]` is decoder layer *i-1*'s output and
the last entry comes from `model.norm`, not the final layer. A tap that captures the
wrong module trains the projection against the wrong depth and never raises -- the loss
just comes out slightly worse.
"""

import pytest
import torch
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

from distillkit.anchor_tap import AnchorTap, CapturedStates, anchor_module
from distillkit.linear_attention_dispatch import install_device_aware_linear_attention

# flash-linear-attention is bound by transformers at import time with no device check
# and its Triton kernels reject CPU tensors. Other test modules get this installed
# incidentally by importing the sidecar model; state the dependency here instead.
install_device_aware_linear_attention()

NUM_LAYERS = 4


def _model(vocab_size=64):
    config = Qwen3_5TextConfig(
        vocab_size=vocab_size, hidden_size=32, intermediate_size=64,
        num_hidden_layers=NUM_LAYERS, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, linear_key_head_dim=8, linear_value_head_dim=8,
        linear_num_key_heads=2, linear_num_value_heads=4, linear_conv_kernel_dim=4,
        full_attention_interval=2, tie_word_embeddings=True,
        max_position_embeddings=64, pad_token_id=0, eos_token_id=3, use_cache=False,
    )
    torch.manual_seed(0)
    return Qwen3_5ForCausalLM(config).eval()


@pytest.mark.parametrize("indices", [[0], [1], [NUM_LAYERS], [1, NUM_LAYERS], [0, 2, NUM_LAYERS]])
def test_tapped_states_match_output_hidden_states(indices):
    model = _model()
    input_ids = torch.randint(0, 64, (2, 9))

    reference = model(input_ids=input_ids, return_dict=True, output_hidden_states=True)
    with AnchorTap(model, indices) as tap:
        model(input_ids=input_ids, return_dict=True)
        states = tap.states()

    for index in indices:
        torch.testing.assert_close(states[index], reference.hidden_states[index], rtol=0, atol=0)


def test_anchor_module_maps_indices_to_the_right_modules():
    model = _model()
    base = model.model
    assert anchor_module(model, 0) is base.embed_tokens
    assert anchor_module(model, 1) is base.layers[0]
    assert anchor_module(model, NUM_LAYERS - 1) is base.layers[NUM_LAYERS - 2]
    # The last entry is post-norm, not the final decoder layer.
    assert anchor_module(model, NUM_LAYERS) is base.norm
    assert anchor_module(model, NUM_LAYERS) is not base.layers[NUM_LAYERS - 1]


def test_out_of_range_index_is_rejected():
    model = _model()
    with pytest.raises(ValueError, match="outside"):
        anchor_module(model, NUM_LAYERS + 1)
    with pytest.raises(ValueError, match="outside"):
        anchor_module(model, -1)


def test_untapped_index_raises_rather_than_returning_the_wrong_state():
    states = CapturedStates({1: torch.zeros(1)})
    assert 1 in states
    with pytest.raises(KeyError, match="was not tapped"):
        states[2]


def test_gradients_flow_through_tapped_states():
    """The tap must hand back the graph tensor, not a detached copy."""
    model = _model()
    input_ids = torch.randint(0, 64, (1, 8))
    with AnchorTap(model, [1, NUM_LAYERS]) as tap:
        model(input_ids=input_ids, return_dict=True)
        states = tap.states()
    states[NUM_LAYERS].sum().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "no gradient reached the model through the tapped state"


def test_hooks_are_removed_on_exit():
    model = _model()
    layer = anchor_module(model, 1)
    before = len(layer._forward_hooks)
    with AnchorTap(model, [1]):
        assert len(layer._forward_hooks) == before + 1
    assert len(layer._forward_hooks) == before


def test_recompute_does_not_replace_the_captured_tensor():
    """Gradient checkpointing re-runs the forward during backward.

    With use_reentrant=False the hook fires again on recompute. If that second firing
    overwrote the capture, the loss would already have consumed the first tensor while
    anything reading `states()` later would get a tensor from a different graph.
    """
    model = _model()
    model.gradient_checkpointing_enable()
    model.train()
    input_ids = torch.randint(0, 64, (1, 8))
    with AnchorTap(model, [NUM_LAYERS]) as tap:
        model(input_ids=input_ids, return_dict=True)
        captured = tap.states()[NUM_LAYERS]
        loss = captured.sum()
        loss.backward()
        assert tap.states()[NUM_LAYERS] is captured


def test_missing_anchor_is_reported_not_silently_dropped():
    model = _model()
    tap = AnchorTap(model, [1])
    with pytest.raises(RuntimeError, match="never produced"):
        tap.states()


def test_hidden_state_loss_does_not_assume_anchor_zero_is_present():
    """The loss used to read `hidden_states[0]` just to pick an accumulator device.

    With a tapped forward that index is usually absent -- the real run's anchors are
    4 and 32 -- so the loss has to take its reference from a mapped anchor instead.
    """
    from distillkit.hsd_mapping import HiddenStateMapping
    from distillkit.lossfuncs.hidden_state import compute_hs_loss
    from distillkit.signals import SparseSignal

    model = _model()
    mapping = [(1, 0), (NUM_LAYERS, 1)]
    hsm = HiddenStateMapping(model, teacher_hidden_size=48, layer_mapping=mapping)
    input_ids = torch.randint(0, 64, (1, 8))
    with AnchorTap(model, [student for student, _ in mapping]) as tap:
        outputs = model(input_ids=input_ids, return_dict=True)
    outputs.hidden_states = tap.states()
    assert 0 not in outputs.hidden_states

    signal = SparseSignal(
        sparse_ids=torch.zeros(1, 8, 2, dtype=torch.long),
        sparse_values=torch.zeros(1, 8, 2),
        log_values=True, generation_temperature=1.0,
        hidden_states=(torch.randn(1, 8, 48), torch.randn(1, 8, 48)),
        vocab_size=64,
    )
    loss = compute_hs_loss("cosine", outputs, signal, None, hsm)
    assert torch.isfinite(loss)


def test_concurrent_forwards_do_not_capture_into_each_other():
    """A pipelined step runs two microbatches at once through the same modules.

    The hooks live on modules shared by the whole model, so without per-thread
    capture each forward would overwrite the other's anchors and both microbatches
    would train against whichever tensor landed last.
    """
    import threading

    model = _model()
    batches = {
        "a": torch.randint(0, 64, (1, 8)),
        "b": torch.randint(0, 64, (1, 8)),
    }
    captured = {}
    barrier = threading.Barrier(2)

    def run(name):
        with AnchorTap(model, [NUM_LAYERS]) as tap:
            barrier.wait()          # force the two forwards to interleave
            model(input_ids=batches[name], return_dict=True)
            captured[name] = tap.states()[NUM_LAYERS]

    threads = [threading.Thread(target=run, args=(name,)) for name in batches]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    for name, ids in batches.items():
        expected = model(input_ids=ids, return_dict=True, output_hidden_states=True)
        torch.testing.assert_close(
            captured[name], expected.hidden_states[NUM_LAYERS], rtol=0, atol=0,
            msg=lambda m, n=name: f"thread {n} captured another thread's state: {m}",
        )

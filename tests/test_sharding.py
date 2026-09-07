"""A student split across GPUs must produce the same loss and gradients as one card.

The split is a single-process layer assignment: no process group, no NCCL, just
``Tensor.to(device)``, which autograd already handles in both directions. What that
does not do is relocate everything the distillation loss touches -- the cached
teacher signal arrives on the batch's device, and the projections are parameters
whose device is fixed at construction. These gates pin both halves: the placement
helpers report the right device, and a sharded step matches an unsharded one.
"""

import pytest
import torch
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

from distillkit.hsd_mapping import HiddenStateMapping
from distillkit.lossfuncs.hidden_state import compute_hs_loss
from distillkit.lossfuncs.kl import KLDLoss
from distillkit.sharding import (
    as_device,
    check_tied_embeddings_colocated,
    hidden_state_device,
    is_sharded,
    module_device,
)
from distillkit.signals import SparseSignal

NUM_LAYERS = 4


def _config(vocab_size=64):
    return Qwen3_5TextConfig(
        vocab_size=vocab_size, hidden_size=32, intermediate_size=64,
        num_hidden_layers=NUM_LAYERS, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, linear_key_head_dim=8, linear_value_head_dim=8,
        linear_num_key_heads=2, linear_num_value_heads=4, linear_conv_kernel_dim=4,
        full_attention_interval=2, tie_word_embeddings=True,
        max_position_embeddings=64, pad_token_id=0, eos_token_id=3, use_cache=False,
    )


def _split_map(first_device_layers=2):
    return {
        "model.embed_tokens": 0,
        "lm_head": 0,
        "model.rotary_emb": 0,
        "model.norm": 1,
        **{
            f"model.layers.{i}": (0 if i < first_device_layers else 1)
            for i in range(NUM_LAYERS)
        },
    }


class _FakeModel:
    """Just the two attributes the placement helpers read."""

    def __init__(self, device_map, embed_device="cuda:0"):
        self.hf_device_map = device_map
        self.config = _config()
        self._embed_device = torch.device(embed_device)

    def get_input_embeddings(self):
        weight = type("W", (), {"device": self._embed_device})()
        return type("E", (), {"weight": weight})()


def test_as_device_normalizes_bare_cuda_ordinals():
    # accelerate writes ints for CUDA and strings for everything else.
    assert as_device(1) == torch.device("cuda", 1)
    assert as_device("cpu") == torch.device("cpu")
    assert as_device(torch.device("cuda:0")) == torch.device("cuda", 0)


def test_hidden_state_device_indexes_outputs_not_layers():
    """hidden_states[i] is layer i-1's output, and the last entry is post-norm.

    Off-by-one here silently builds a projection on the wrong card, which then
    either fails at the first matmul or moves the anchor over the bus every step.
    """
    model = _FakeModel(_split_map(first_device_layers=2))
    assert hidden_state_device(model, 0) == torch.device("cuda", 0)   # embeddings
    assert hidden_state_device(model, 1) == torch.device("cuda", 0)   # layer 0 out
    assert hidden_state_device(model, 2) == torch.device("cuda", 0)   # layer 1 out
    assert hidden_state_device(model, 3) == torch.device("cuda", 1)   # layer 2 out
    # The final entry comes from model.norm, not from the last decoder layer.
    assert hidden_state_device(model, NUM_LAYERS) == torch.device("cuda", 1)


def test_device_lookup_falls_back_to_ancestor_prefixes():
    model = _FakeModel({"model": 1, "lm_head": 0})
    assert module_device(model, "model.layers.7.self_attn") == torch.device("cuda", 1)
    assert module_device(model, "lm_head") == torch.device("cuda", 0)


def test_unsharded_model_reports_its_embedding_device():
    model = _FakeModel({}, embed_device="cpu")
    assert not is_sharded(model)
    assert hidden_state_device(model, 2) == torch.device("cpu")


def test_single_device_map_is_not_sharded():
    assert not is_sharded(_FakeModel({"model": 0, "lm_head": 0}))


def test_tied_embeddings_split_across_devices_is_rejected():
    """One parameter cannot live on two cards.

    Left unchecked this either fails deep inside the forward or, worse, ends up as
    two tensors that train apart and reconstruct into a checkpoint matching neither.
    """
    model = Qwen3_5ForCausalLM(_config()).eval()
    model.hf_device_map = {"model.embed_tokens": 0, "lm_head": 1, "model": 0}
    model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.detach().clone())
    with pytest.raises(ValueError, match="tie_word_embeddings"):
        check_tied_embeddings_colocated(model)


def test_colocated_tied_embeddings_pass():
    model = Qwen3_5ForCausalLM(_config()).eval()
    model.hf_device_map = {"model.embed_tokens": 0, "lm_head": 0, "model.layers.3": 1}
    check_tied_embeddings_colocated(model)


# --------------------------------------------------------------------------- #
# The real gate: two cards must give the same numbers as one.
# --------------------------------------------------------------------------- #


def _batch(vocab_size, hidden_size, batch=2, seq=12, top_k=5, seed=0, device="cuda:0"):
    generator = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(0, vocab_size, (batch, seq), generator=generator)
    mask = torch.ones(batch, seq, 1, dtype=torch.bool)
    device = torch.device(device)
    signal = SparseSignal(
        sparse_ids=torch.randint(
            0, vocab_size, (batch, seq, top_k), generator=generator
        ).to(device),
        sparse_values=torch.log_softmax(
            torch.randn(batch, seq, top_k, generator=generator), -1
        ).to(device),
        log_values=True,
        generation_temperature=1.0,
        # Deliberately left on the batch's device, as the offline cache delivers them.
        hidden_states=(
            torch.randn(batch, seq, hidden_size, generator=generator).to(device),
            torch.randn(batch, seq, hidden_size, generator=generator).to(device),
        ),
        vocab_size=vocab_size,
    )
    return input_ids.to(device), mask.to(device), signal


def _loss(model, hsm, input_ids, mask, signal):
    outputs = model(input_ids=input_ids, return_dict=True, output_hidden_states=True)
    kl = KLDLoss(temperature=1.0)(outputs, signal, mask=mask, hidden_state_mapping=hsm)
    hs = compute_hs_loss("cosine", outputs, signal, mask, hsm)
    return kl.to("cuda:0") + hs.to("cuda:0")


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two CUDA devices")
def test_sharded_step_matches_single_device_step(tmp_path):
    """Loss and every gradient must survive the split.

    A wrong seam here does not raise: the anchor loss is computed against a stale or
    misplaced tensor and training simply learns something else.
    """
    torch.manual_seed(0)
    config = _config()
    path = tmp_path / "tiny"
    Qwen3_5ForCausalLM(config).eval().save_pretrained(path)

    # Teacher hidden size differs from the student's, so projections are built --
    # they are the parameters whose placement the split has to get right.
    teacher_hidden = 48
    # Layer 0's output lives on card 0; the post-norm anchor lives on card 1.
    mapping = [(1, 0), (NUM_LAYERS, 1)]

    reference = Qwen3_5ForCausalLM.from_pretrained(path).to("cuda:0")
    torch.manual_seed(1)
    ref_hsm = HiddenStateMapping(reference, teacher_hidden, mapping)

    sharded = Qwen3_5ForCausalLM.from_pretrained(path, device_map=_split_map())
    assert is_sharded(sharded)
    torch.manual_seed(1)
    shard_hsm = HiddenStateMapping(sharded, teacher_hidden, mapping)
    # Same initial projection weights, each on its own anchor's card.
    for dst, src in zip(shard_hsm.projections, ref_hsm.projections):
        dst.weight.data.copy_(src.weight.data.to(dst.weight.device))
    assert shard_hsm.projections[0].weight.device == torch.device("cuda", 0)
    assert shard_hsm.projections[1].weight.device == torch.device("cuda", 1)

    input_ids, mask, signal = _batch(config.vocab_size, teacher_hidden)

    ref_loss = _loss(reference, ref_hsm, input_ids, mask, signal)
    ref_loss.backward()
    shard_loss = _loss(sharded, shard_hsm, input_ids, mask, signal)
    shard_loss.backward()

    torch.testing.assert_close(shard_loss, ref_loss, rtol=2e-4, atol=2e-5)

    ref_grads = {
        name: param.grad
        for name, param in reference.named_parameters()
        if param.grad is not None
    }
    assert ref_grads, "reference produced no gradients; the test proves nothing"
    for name, param in sharded.named_parameters():
        if name not in ref_grads:
            continue
        assert param.grad is not None, f"no gradient reached {name} across the split"
        torch.testing.assert_close(
            param.grad.to("cuda:0"),
            ref_grads[name],
            rtol=2e-3,
            atol=2e-5,
            msg=lambda text, name=name: f"{name}: {text}",
        )

    for index, (proj, ref_proj) in enumerate(
        zip(shard_hsm.projections, ref_hsm.projections)
    ):
        torch.testing.assert_close(
            proj.weight.grad.to("cuda:0"),
            ref_proj.weight.grad,
            rtol=2e-3,
            atol=2e-5,
            msg=lambda text, index=index: f"projection {index}: {text}",
        )

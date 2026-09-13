"""The paired co-adaptation screen is only a comparison if the arms differ in one thing.

Arm A trains the backbone and the native memory together; arm B trains the backbone alone
from the same checkpoint, with the memory bypassed and frozen. Everything that could make
the two runs incomparable -- a drifting training stream, a control whose table quietly
moves, a starting point reconstructed rather than shared -- is checked here rather than
discovered in the results.
"""

import hashlib

import pytest
import torch

from distillkit.models import Qwen35SidecarForCausalLM
from distillkit.native_ple import native_hash_config
from distillkit.ngram_hash import NGramHasher
from distillkit.optimizers import freeze_sidecar_parameters
from tests.test_sidecar_model import tiny_config

SIDECAR = ("rho", "table.weight", "ple.key_proj.weight", "ple.value_proj.weight",
           "ple.conv1d.weight")


def build(seed=0, layer_index=1):
    config = tiny_config(sidecar_variant="ple", sidecar_table_mode="native",
                         sidecar_ngram_vocab_size_base=97,
                         sidecar_layer_index=layer_index)
    torch.manual_seed(seed)
    model = Qwen35SidecarForCausalLM(config)
    model.config.use_cache = False
    with torch.no_grad():
        model.model.layers[layer_index].sidecar.rho.fill_(0.03)
    return model


def batch(model, batch_size=2, length=12, seed=7):
    generator = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, model.config.vocab_size, (batch_size, length), generator=generator)
    return ids, NGramHasher(native_hash_config(model.config)).row_indices(ids)


def sidecar_of(model):
    return model.model.layers[model.config.sidecar_layer_index].sidecar


def snapshot(model):
    sidecar = sidecar_of(model)
    return {name: sidecar.get_parameter(name).detach().clone() for name in SIDECAR}


def step(model, ids, rows, enabled):
    """One CE update at a realistic learning rate, returning the optimizer used."""
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3, weight_decay=0.1)
    optimizer.zero_grad(set_to_none=True)
    kwargs = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
    if enabled:
        kwargs["ngram_ids"] = rows
    else:
        kwargs["sidecar_enabled"] = False
    logits = model(**kwargs).logits[:, :-1].float()
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1))
    loss.backward()
    optimizer.step()
    return loss


# --- the common starting point ------------------------------------------------


def test_both_arms_start_from_bit_identical_weights():
    """Reconstructing either arm from the pretrained checkpoint would not be this."""
    arm_a, arm_b = build(seed=0), build(seed=0)
    freeze_sidecar_parameters(arm_b)

    left = dict(arm_a.named_parameters())
    right = dict(arm_b.named_parameters())
    assert set(left) == set(right)
    for name in left:
        assert torch.equal(left[name], right[name]), name


# --- arm A: everything learns -------------------------------------------------


def test_arm_a_trains_the_backbone_and_the_memory_together():
    model = build()
    ids, rows = batch(model)
    model.requires_grad_(True)

    step(model, ids, rows, enabled=True)
    sidecar = sidecar_of(model)
    for name in SIDECAR:
        parameter = sidecar.get_parameter(name)
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name
    # And the backbone, at both ends of the stack.
    for layer in (0, len(model.model.layers) - 1):
        weight = model.model.layers[layer].mlp.down_proj.weight
        assert weight.grad is not None and weight.grad.abs().sum() > 0, layer


# --- arm B: the memory must not move -----------------------------------------


def test_arm_b_leaves_the_memory_bitwise_unchanged_after_an_update():
    """Weight decay is the trap: a bypassed table is still a parameter.

    An optimizer with weight_decay > 0 will shrink any parameter it is given, gradient or
    no gradient, so 'the forward does not use it' is not the same as 'it does not move'.
    A control whose memory drifts is not a control.
    """
    model = build()
    ids, rows = batch(model)
    model.requires_grad_(True)
    frozen = freeze_sidecar_parameters(model)
    assert len(frozen) == 8, frozen

    before = snapshot(model)
    step(model, ids, rows, enabled=False)

    sidecar = sidecar_of(model)
    for name in SIDECAR:
        assert sidecar.get_parameter(name).grad is None, name
        assert torch.equal(before[name], sidecar.get_parameter(name).detach()), name
    # The backbone did move, or the arm trains nothing at all.
    assert model.model.layers[0].mlp.down_proj.weight.grad.abs().sum() > 0


def test_freezing_the_sidecar_refuses_a_model_without_one():
    model = build()
    freeze_sidecar_parameters(model)
    with pytest.raises(ValueError, match="no trainable sidecar parameters"):
        freeze_sidecar_parameters(model)


def test_the_two_freezes_together_are_refused_in_configuration():
    """Both set would train nothing at all, silently, for the whole run."""
    from distillkit.configuration import OptimizerConfig

    with pytest.raises(ValueError, match="nothing trainable"):
        OptimizerConfig(freeze_backbone=True, freeze_sidecar=True)
    OptimizerConfig(freeze_backbone=False, freeze_sidecar=True)


# --- data parity --------------------------------------------------------------


def test_the_training_stream_digest_distinguishes_order():
    """The arms' hashes are compared afterwards, so the hash has to be order-sensitive."""
    from distillkit.trainer import STREAM_PARITY_BATCHES, DistillationTrainer

    class Recorder:
        def __init__(self):
            self._stream_digest = hashlib.sha256()
            self._stream_batches = 0
            self._stream_sha256 = None

    def digest(batches):
        recorder = Recorder()
        for tokens in batches:
            DistillationTrainer._digest_stream(recorder, tokens)
        return recorder._stream_digest.hexdigest()

    first = [torch.full((1, 4), i, dtype=torch.long) for i in range(4)]
    assert digest(first) == digest(list(first))
    assert digest(first) != digest(list(reversed(first)))
    assert digest(first) != digest(first[:3])

    # It stops after STREAM_PARITY_BATCHES, and publishes the hash exactly then.
    recorder = Recorder()
    for index in range(STREAM_PARITY_BATCHES + 5):
        DistillationTrainer._digest_stream(
            recorder, torch.full((1, 4), index, dtype=torch.long))
    assert recorder._stream_batches == STREAM_PARITY_BATCHES
    assert recorder._stream_sha256 is not None


# --- evaluation contracts the comparison depends on ---------------------------


def test_the_content_layout_split_is_reported_for_plain_documents():
    """The held-out bundle is plain text, and the split must not silently vanish.

    It used to be emitted only for chat-formatted features carrying role spans. On this
    bundle that meant one aggregate number -- and the frozen stage showed an aggregate is
    79% layout, so reporting it alone is how a newline gain gets read as a content gain.
    """
    from distillkit.independent_eval import LAYOUT_TOKEN_IDS, score_sequences

    model = build()
    ids, rows = batch(model, batch_size=1, length=16)
    tokens = ids[0].tolist()
    # Guarantee both classes are present.
    tokens[3] = next(iter(LAYOUT_TOKEN_IDS)) % model.config.vocab_size
    feature = {"id": "plain", "ids": tokens}

    hasher = NGramHasher(native_hash_config(model.config))

    class Collator:
        def __call__(self, features):
            batched = torch.tensor([f["ids"] for f in features])
            return {"input_ids": batched, "attention_mask": torch.ones_like(batched),
                    "ngram_ids": hasher.row_indices(batched)}

    model.eval()
    scored = score_sequences(model, [feature], Collator(), "enabled", "cpu")[0]
    assert "by_class" in scored
    assert "content" in scored["by_class"]
    counted = sum(part["tokens"] for part in scored["by_class"].values())
    assert counted == scored["tokens"], "every scored target belongs to exactly one class"


def test_wrong_context_addressing_still_differs_on_a_co_adapted_checkpoint():
    """`--shuffle-context` must keep changing the rows, whatever the backbone has learned."""
    from distillkit.independent_eval import make_collator

    model = build()
    hasher = NGramHasher(native_hash_config(model.config))
    features = [{"ids": [5, 9, 12, 3, 7, 11, 4, 8, 2, 6]}]

    correct = make_collator(0, None, hasher, "native")(features)
    wrong = make_collator(0, None, hasher, "native", shuffle_context=7)(features)
    assert torch.equal(correct["input_ids"], wrong["input_ids"]), "the text must not change"
    assert not torch.equal(correct["ngram_ids"], wrong["ngram_ids"]), "the rows must"

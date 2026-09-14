"""Cached decoding must address the same rows teacher forcing does.

Every NLL number in this programme was teacher-forced, so the whole sequence reached
`forward` at once and the trigram at position t was computed from positions t-2..t. With a
KV cache the model sees one token per call, and both context mechanisms would then hash
the newest token against whatever happened to precede it in that call -- which is nothing.
The lookup still returns a row, so a generation benchmark would have quietly scored a
differently-addressed model and reported it as the same one.

These pin the fix from the only angle that matters: generated output must match what the
same model produces when the identical sequence is scored in one pass.
"""

import numpy as np
import pytest
import torch

from distillkit.experimental.ngram_hash import NGramHashConfig, NGramHasher
from distillkit.experimental.residual_gate import (
    ResidualGateHandle, TrigramFamiliarity, calibrate_gates, install_residual_gates,
    remove_residual_gates)
from torch import nn

from tests.test_residual_gate import GATED, batch, build, familiarity_for

VOCAB = 97


def statistics_for(tmp_path, keys=(0, 7, 31), counts=(9, 90, 900)):
    path = tmp_path / "cache.npz"
    np.savez(path, keys=np.array(keys, dtype=np.int64),
             counts=np.array(counts, dtype=np.int64),
             mean=np.zeros((len(keys), 4), dtype=np.float16),
             variance=np.linspace(0.1, 0.9, len(keys)),
             global_mean=np.zeros(4, dtype=np.float32))
    return TrigramFamiliarity(path, VOCAB)


def handle_for(tmp_path):
    return ResidualGateHandle(nn.ModuleDict(), "familiarity", statistics_for(tmp_path))


def test_the_hasher_needs_its_history_and_says_so_when_given_it():
    """The bug, stated as a test: without previous_context the rows are simply wrong."""
    hasher = NGramHasher(NGramHashConfig(
        vocab_size=VOCAB, ngram_size=3, heads_per_ngram=1,
        ngram_vocab_size_base=251, eos_token_id=VOCAB - 1, seed=7))
    full = torch.tensor([[10, 20, 30, 40, 50]])
    correct = hasher.row_indices(full)[0, 4]
    alone = hasher.row_indices(full[:, 4:])[0, 0]
    with_history = hasher.row_indices(full[:, 4:], previous_context=full[:, 2:4])[0, 0]
    assert not torch.equal(alone, correct)
    assert torch.equal(with_history, correct)


def test_one_token_at_a_time_reproduces_the_whole_sequence(tmp_path):
    handle = handle_for(tmp_path)
    full = torch.tensor([[3, 7, 31, 11, 13, 7, 31]])
    handle.set_context(full)
    teacher = handle.context.clone()

    handle.begin_generation()
    handle.set_context(full[:, :3])
    assert torch.equal(handle.context, teacher[:, :3])
    for position in range(3, full.shape[1]):
        handle.set_context(full[:, position:position + 1])
        assert torch.equal(handle.context[:, 0], teacher[:, position]), position
    handle.end_generation()


def test_the_prefix_and_the_continuation_agree_on_a_familiar_context(tmp_path):
    """A trigram the cache knows has to be found either way, not just missed either way."""
    statistics = statistics_for(tmp_path, keys=(0,), counts=(400,))
    handle = ResidualGateHandle(nn.ModuleDict(), "familiarity", statistics)
    # Key for position 2 of [0, 0, 0] is (0 * V + 0) * V + 0 == 0, which the cache holds.
    full = torch.tensor([[0, 0, 0, 5]])
    handle.set_context(full)
    teacher = handle.context.clone()
    assert teacher[0, 2, 0] > 0, "the fixture must exercise a cache hit"

    handle.begin_generation()
    handle.set_context(full[:, :2])
    handle.set_context(full[:, 2:3])
    assert torch.equal(handle.context[0, 0], teacher[0, 2])
    handle.end_generation()


def test_generation_mode_is_off_by_default(tmp_path):
    handle = handle_for(tmp_path)
    assert not handle.generating and handle.history is None
    handle.set_context(torch.tensor([[3, 7, 31]]))
    assert handle.history is None


def test_beginning_a_second_sequence_forgets_the_first(tmp_path):
    handle = handle_for(tmp_path)
    first = torch.tensor([[3, 7, 31, 11]])
    second = torch.tensor([[13, 5, 2, 11]])
    handle.set_context(second)
    teacher = handle.context.clone()

    handle.begin_generation()
    handle.set_context(first)
    handle.begin_generation()
    handle.set_context(second[:, :3])
    handle.set_context(second[:, 3:4])
    assert torch.equal(handle.context[0, 0], teacher[0, 3])
    handle.end_generation()


def test_a_gated_model_generates_what_it_would_have_scored(tmp_path):
    """End to end: greedy continuation under cache equals the teacher-forced argmax."""
    model = build()
    ids = batch(model, batch_size=1, length=8)
    handle = install_residual_gates(model, GATED, family="familiarity",
                                    familiarity=familiarity_for(model, tmp_path))
    try:
        calibrate_gates(model, handle,
                        [{"input_ids": ids, "attention_mask": torch.ones_like(ids)}])
        for index in handle.layer_indices:
            handle.gate(index).output.weight.data.normal_(0, 0.6)

        with torch.no_grad():
            model.config.use_cache = True
            handle.begin_generation()
            generated = model.generate(input_ids=ids,
                                       attention_mask=torch.ones_like(ids),
                                       max_new_tokens=6, do_sample=False,
                                       pad_token_id=0)
            handle.end_generation()
            model.config.use_cache = False
            # Score the identical sequence in one pass: the token the model chose at each
            # step must be the argmax the scored model assigns at that position.
            logits = model(input_ids=generated,
                           attention_mask=torch.ones_like(generated)).logits
        chosen = generated[0, ids.shape[1]:]
        predicted = logits[0, ids.shape[1] - 1:-1].argmax(-1)
        assert torch.equal(chosen, predicted)
    finally:
        remove_residual_gates(model)
        model.config.use_cache = False

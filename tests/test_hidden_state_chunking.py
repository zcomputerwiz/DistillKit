"""The hidden-state loss was the last unchunked path, and it is not a small one.

Each anchor projects the student's 2560-wide state to the teacher's 5120 and keeps
that output live for backward alongside the cosine's own temporaries, twice over
because there are two anchors. Chunking it costs a recompute and returns the memory,
the same trade the folded head already makes for the KL term.

These pin the two things that can go wrong: the chunked sum must agree with the whole
-sequence one, and the chunk boundary must not quietly drop or double-count positions
when the sequence is not a multiple of the chunk.
"""

import pytest
import torch
from torch import nn
from transformers.modeling_outputs import CausalLMOutput

from distillkit.lossfuncs.hidden_state import compute_hs_loss

STUDENT_WIDTH, TEACHER_WIDTH, ANCHOR = 16, 24, 3


class _Mapping:
    """The parts of HiddenStateMapping this loss reads."""

    def __init__(self, projections):
        self.layer_mapping = [(ANCHOR, 0)]
        self.projections = projections


def _fixture(batch=2, sequence=7, projected=True, seed=0):
    torch.manual_seed(seed)
    student = torch.randn(batch, sequence, STUDENT_WIDTH, requires_grad=True)
    width = STUDENT_WIDTH if not projected else TEACHER_WIDTH
    teacher = torch.randn(batch, sequence, width)
    projections = None
    if projected:
        projections = nn.ModuleList([nn.Linear(STUDENT_WIDTH, TEACHER_WIDTH, bias=False)])
    outputs = CausalLMOutput(logits=None, hidden_states={ANCHOR: student})

    class _Signal:
        hidden_states = (teacher,)

    mask = torch.ones(batch, sequence, dtype=torch.bool)
    mask[0, -2:] = False  # padding must not contribute, whichever chunk it lands in
    return outputs, _Signal(), mask, _Mapping(projections), student


@pytest.mark.parametrize("kind", ["cosine", "mse"])
@pytest.mark.parametrize("chunk_rows", [2, 3, 4, 8, 1024])
def test_chunking_agrees_with_the_whole_sequence(kind, chunk_rows):
    """Chunk sizes that do and do not divide the sequence, against one reduction."""
    outputs, signal, mask, mapping, _ = _fixture()
    whole = compute_hs_loss(kind, outputs, signal, mask, mapping, chunk_rows=10**6)
    chunked = compute_hs_loss(kind, outputs, signal, mask, mapping, chunk_rows=chunk_rows)
    torch.testing.assert_close(chunked, whole, rtol=1e-6, atol=1e-6)


def test_the_chunk_budget_scales_with_the_batch():
    """A chunk's tensors are [batch, rows, width], so the same setting at batch 4
    would allocate four times as much. Reading it as a row budget is what keeps the
    memory constant, and the loss must not change when the batch does."""
    outputs, signal, mask, mapping, _ = _fixture(batch=4, sequence=8)
    whole = compute_hs_loss("cosine", outputs, signal, mask, mapping, chunk_rows=10**6)
    for chunk_rows in (4, 8, 16):
        chunked = compute_hs_loss("cosine", outputs, signal, mask, mapping, chunk_rows=chunk_rows)
        torch.testing.assert_close(chunked, whole, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("kind", ["cosine", "mse"])
def test_gradients_survive_the_recompute(kind):
    """The chunks run under torch.utils.checkpoint; a broken recompute shows up as a
    missing or wrong gradient rather than a wrong loss."""
    outputs, signal, mask, mapping, student = _fixture()
    compute_hs_loss(kind, outputs, signal, mask, mapping, chunk_rows=3).backward()
    chunked = student.grad.clone()

    outputs, signal, mask, mapping, student = _fixture()
    compute_hs_loss(kind, outputs, signal, mask, mapping, chunk_rows=10**6).backward()

    assert torch.isfinite(chunked).all()
    assert chunked.abs().sum() > 0, "the recompute produced no gradient at all"
    torch.testing.assert_close(chunked, student.grad, rtol=1e-5, atol=1e-6)


def test_masked_positions_do_not_contribute_from_any_chunk():
    """Padding zeroed inside a chunk still counts in the denominator if the mask sum
    is taken over the whole sequence, so the two have to agree."""
    outputs, signal, mask, mapping, _ = _fixture(batch=1, sequence=6)
    mask = torch.ones(1, 6, dtype=torch.bool)
    full = compute_hs_loss("cosine", outputs, signal, mask, mapping, chunk_rows=2)

    mask[0, 4:] = False
    trimmed = compute_hs_loss("cosine", outputs, signal, mask, mapping, chunk_rows=2)
    # Dropping two positions changes the mean; if the mask were ignored it would not.
    assert not torch.isclose(full, trimmed)

    outputs.hidden_states[ANCHOR] = outputs.hidden_states[ANCHOR][:, :4]
    signal.hidden_states = (signal.hidden_states[0][:, :4],)
    shortened = compute_hs_loss("cosine", outputs, signal, mask[:, :4], mapping, chunk_rows=2)
    torch.testing.assert_close(trimmed, shortened, rtol=1e-6, atol=1e-6)


def test_no_grad_path_skips_the_checkpoint():
    """Evaluation runs under inference_mode, where checkpoint would raise."""
    outputs, signal, mask, mapping, _ = _fixture()
    with torch.no_grad():
        value = compute_hs_loss("cosine", outputs, signal, mask, mapping, chunk_rows=3)
    assert torch.isfinite(value)

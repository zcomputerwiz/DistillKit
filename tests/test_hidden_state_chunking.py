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


def test_bfloat16_chunks_do_not_undercount_the_sum():
    """A running BF16 scalar sum stops changing once the accumulator is large relative
    to each addend. Measured on the real module before the fix: orthogonal BF16 vectors
    whose cosine loss is exactly 1.0 came back as 0.0625 at chunk_rows=2 and 1.0 at
    chunk_rows=1024. CUDA autocast promotes this path and hid it in every configured
    run, which is why the reduction dtype is now explicit rather than ambient."""
    batch, sequence, width = 2, 4096, 2
    student = torch.zeros(batch, sequence, width, dtype=torch.bfloat16)
    teacher = torch.zeros(batch, sequence, width, dtype=torch.bfloat16)
    student[..., 0] = 1.0          # orthogonal, so every position contributes exactly 1
    teacher[..., 1] = 1.0
    outputs = CausalLMOutput(logits=None, hidden_states={ANCHOR: student})

    class _Signal:
        hidden_states = (teacher,)

    mapping = _Mapping(None)
    mask = torch.ones(batch, sequence, dtype=torch.bool)
    for chunk_rows in (2, 8, 1024, 10**6):
        value = compute_hs_loss("cosine", outputs, _Signal(), mask, mapping,
                                chunk_rows=chunk_rows)
        assert abs(value.item() - 1.0) < 1e-3, (chunk_rows, value.item())


def _saved_bytes(fn):
    seen = {}

    def pack(tensor):
        seen[id(tensor)] = tensor.numel() * tensor.element_size()
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        fn().backward()
    return sum(seen.values())


def test_a_frozen_anchor_with_a_trainable_projection_still_checkpoints():
    """Stage 1 freezes the backbone, so the anchor does not require grad while the
    projection consuming it does. Gating the checkpoint on the anchor alone skipped it
    in exactly the configuration chunking exists for, and every chunk's intermediates
    stayed alive to backward."""
    def build(freeze_anchor):
        torch.manual_seed(3)
        student = torch.randn(2, 64, STUDENT_WIDTH, requires_grad=not freeze_anchor)
        teacher = torch.randn(2, 64, TEACHER_WIDTH)
        projections = nn.ModuleList([nn.Linear(STUDENT_WIDTH, TEACHER_WIDTH, bias=False)])
        outputs = CausalLMOutput(logits=None, hidden_states={ANCHOR: student})

        class _Signal:
            hidden_states = (teacher,)

        mask = torch.ones(2, 64, dtype=torch.bool)
        return lambda: compute_hs_loss("cosine", outputs, _Signal(), mask,
                                       _Mapping(projections), chunk_rows=8)

    frozen = _saved_bytes(build(True))
    trainable = _saved_bytes(build(False))
    # Both paths checkpoint, so neither retains a chunk's projected states. A regression
    # shows up as the frozen case holding far more than the trainable one.
    assert frozen <= trainable * 1.5, (frozen, trainable)


def test_no_grad_still_skips_the_checkpoint():
    """Evaluation runs under inference_mode, where checkpoint would raise, and nothing
    is trainable there however the projection is flagged."""
    outputs, signal, mask, mapping, _ = _fixture()
    with torch.no_grad():
        assert torch.isfinite(compute_hs_loss("cosine", outputs, signal, mask, mapping,
                                              chunk_rows=3))

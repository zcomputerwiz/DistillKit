import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers.modeling_outputs import CausalLMOutput
from typing_extensions import override

from distillkit.hsd_mapping import HiddenStateMapping
from distillkit.lossfuncs.common import (
    LossFunctionBase,
)
from distillkit.signals import TeacherSignal


# Positions per chunk at batch 1, halved at batch 2 and so on: a chunk's tensors are
# [batch, rows, teacher_hidden], so the budget is rows x batch, not rows.
HS_CHUNK_ROWS = 1024


def _anchor_sum(kind, student_h, teacher_h, layer_mask, projection):
    """One chunk's unnormalised contribution, as a scalar."""
    if projection is not None:
        # The projections are built in fp32 after the student is loaded in bf16, so
        # this matmul only works when autocast happens to be active at this call
        # site. Align explicitly instead: relying on an ambient context manager for
        # dtype correctness fails at step 0 with a bare "mat1 and mat2 have
        # different dtype" and no indication of which side is wrong.
        student_h = projection(student_h.to(projection.weight.dtype))
    # The cached teacher states are upcast from fp8 and need not share the student's
    # dtype either; the arithmetic below assumes they do. They also arrive on the
    # batch's device, which is only one of several when the student is split.
    teacher_h = teacher_h.to(device=student_h.device, dtype=student_h.dtype)
    layer_mask = layer_mask.to(student_h.device)
    if kind == "mse":
        return _masked_mse(student_h, teacher_h, layer_mask)
    if kind != "cosine":
        raise RuntimeError(f"Unimplemented hidden state loss type {repr(kind)}")
    return _masked_cosine(student_h, teacher_h, layer_mask)


# Deliberately not compiled. These run inside `_accumulate_anchor`'s checkpoint, and
# a compiled region inside a non-reentrant checkpoint frame breaks its early-stop
# machinery when the frame is entered from a worker thread: the concurrent-microbatch
# path raised `_StopRecomputationError` with "target_frame.early_stop is set". The
# memory here comes from the checkpoint rather than from fusing four cheap kernels,
# so there is nothing to trade away.
def _masked_cosine(student_h, teacher_h, layer_mask):
    cosine_sim = F.cosine_similarity(student_h, teacher_h, dim=-1)
    return ((1 - cosine_sim) * layer_mask.squeeze(-1)).sum()


def _masked_mse(student_h, teacher_h, layer_mask):
    return (((student_h - teacher_h) ** 2) * layer_mask).sum()


def _accumulate_anchor(kind, student_h, teacher_h, layer_mask, projection, chunk_rows):
    """Project and reduce a slice of positions at a time.

    This was the one remaining unchunked path in the loss. The projection widens the
    student's hidden size to the teacher's -- 2560 to 5120 here -- and its output stays
    live for backward alongside the cosine's own temporaries, twice over because there
    are two anchors. None of it is needed once a chunk's scalar is accumulated, and the
    chunked head already does exactly this for the KL term.

    Summing chunk scalars is not bitwise identical to one reduction over the whole
    sequence. It is the same quantity to fp32 rounding, and the KL term has been
    computed this way since the folded head landed.
    """
    rows = max(1, chunk_rows // max(1, student_h.shape[0]))
    sequence = student_h.shape[1]
    total = None
    for start in range(0, sequence, rows):
        end = min(start + rows, sequence)
        pieces = (student_h[:, start:end], teacher_h[:, start:end], layer_mask[:, start:end])
        if pieces[0].requires_grad:
            # preserve_rng_state=False: no dropout here, and restoring global RNG from
            # a worker thread would race other in-flight recomputes.
            part = checkpoint(
                lambda a, b, c: _anchor_sum(kind, a, b, c, projection), *pieces,
                use_reentrant=False, preserve_rng_state=False,
            )
        else:
            part = _anchor_sum(kind, *pieces, projection)
        total = part if total is None else total + part
    return total


def compute_hs_loss(
    kind: str,
    student_outputs: CausalLMOutput,
    signal: TeacherSignal,
    mask: torch.Tensor | None = None,
    hidden_state_mapping: HiddenStateMapping | None = None,
    chunk_rows: int = HS_CHUNK_ROWS,
):
    assert hidden_state_mapping is not None, (
        "Hidden state losses require HiddenStateMapping"
    )
    assert len(hidden_state_mapping.layer_mapping) > 0, (
        "No layers specified in hidden state mapping"
    )
    assert student_outputs.hidden_states is not None
    assert signal.hidden_states is not None

    # Index 0 is not necessarily available: the trainer taps only the anchors this
    # mapping names, so `hidden_states` can be a sparse view whose keys are exactly
    # those anchors. Take the reference shape and device from the first one instead.
    first_anchor = hidden_state_mapping.layer_mapping[0][0]
    reference = student_outputs.hidden_states[first_anchor]

    if mask is None:
        mask = torch.ones(
            reference.shape[:-1], dtype=torch.bool, device=reference.device,
        )

    if mask is not None and mask.dim() == 2:
        mask = mask.unsqueeze(-1)

    total_loss = torch.tensor(0.0, device=reference.device)
    for i, (student_layer_idx, teacher_layer_idx) in enumerate(
        hidden_state_mapping.layer_mapping
    ):
        student_h = student_outputs.hidden_states[student_layer_idx]
        teacher_h = signal.hidden_states[teacher_layer_idx]

        projection = None
        if hidden_state_mapping.projections is not None:
            projection = hidden_state_mapping.projections[i]
            # On a sharded student each projection was constructed on its anchor's
            # device, so this is a no-op; it is not one if a caller supplied its own
            # mapping.
            student_h = student_h.to(projection.weight.device)

        layer_mask = mask.to(student_h.device)
        summed = _accumulate_anchor(kind, student_h, teacher_h, layer_mask,
                                    projection, chunk_rows)
        width = teacher_h.shape[-1] if kind == "mse" else 1
        layer_loss = summed / (layer_mask.sum().to(summed.device) * width)

        # Anchors on different cards each produce a scalar; only the scalar crosses.
        total_loss = total_loss + layer_loss.to(total_loss.device)

    return total_loss / len(hidden_state_mapping.layer_mapping)


class HiddenStateCosineLoss(LossFunctionBase):
    @override
    @classmethod
    def name(cls) -> str:
        return "hs_cosine"

    @override
    def requires_hidden_states(self) -> bool:
        return True

    @override
    def __init__(self): ...

    @override
    def __call__(
        self,
        student_outputs: CausalLMOutput,
        signal: TeacherSignal,
        mask: torch.Tensor | None = None,
        hidden_state_mapping: HiddenStateMapping | None = None,
        num_items_in_batch: int | None = None,
    ) -> torch.Tensor:
        return compute_hs_loss(
            "cosine", student_outputs, signal, mask, hidden_state_mapping
        )


class HiddenStateMSELoss(LossFunctionBase):
    @override
    @classmethod
    def name(cls) -> str:
        return "hs_mse"

    @override
    def requires_hidden_states(self) -> bool:
        return True

    @override
    def __init__(self): ...

    @override
    def __call__(
        self,
        student_outputs: CausalLMOutput,
        signal: TeacherSignal,
        mask: torch.Tensor | None = None,
        hidden_state_mapping: HiddenStateMapping | None = None,
        num_items_in_batch: int | None = None,
    ) -> torch.Tensor:
        return compute_hs_loss(
            "mse", student_outputs, signal, mask, hidden_state_mapping
        )

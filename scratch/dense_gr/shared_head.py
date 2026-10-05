"""Every head loss from one projection of the scored rows: cross entropy, teacher KL,
unlikelihood.

The step used to project hidden states through the 248,320-wide head separately for each
loss: cross entropy through Cut Cross-Entropy, then the teacher KL through its own chunked
projection of *every* position -- zero-weight prompt and padding rows included, masked only
afterwards -- and unlikelihood through a third. KL-only documents (the raw code, the
looping rollouts) also paid for a cross entropy whose gradient was thrown away.

Here the rows that carry any weight are gathered first, and each chunk of them is projected
once; its logits give all three losses -- the log-sum-exp, the target's logit (cross
entropy, and unlikelihood's p = exp(-nll), since at a looping position the next token *is*
the target), and the teacher's 64 ids (the grouped-tail KL, `sparse_kl_div_inner`'s
SYMMETRIC_UNIFORM arithmetic term for term). Each chunk is checkpointed, as the KL's chunks
were, so only its inputs are kept for backward.

Why not Cut Cross-Entropy's log-sum-exp plus 64 gathered logits, which would skip
projecting the logits at all: the KL's tail, 1 - sum of the student's top-k probabilities,
cancels down to ~1e-6 on predictable text, so the log-sum-exp and the gathered logits must
agree to that precision, and they only do when they come from the same logits. CCE's
log-sum-exp is that of bf16-rounded logits (within 6e-4 of it, up to 0.12 from exact), and
no rounding of separately gathered logits matched it closely enough: a fifth of a raw-code
batch's rows came out with a negative tail (head_parity.py, 2026-10-04).
"""
from __future__ import annotations

import torch
from torch.utils.checkpoint import checkpoint


def grouped_tail_kl_rows(student, teacher, eps=1e-8):
    """Per row, KL(teacher || student) over the teacher's top-k plus one tail bucket, from
    log-probs at the top-k ids; `sparse_kl_div_inner`'s SYMMETRIC_UNIFORM arithmetic."""
    upper = 1.0 - torch.finfo(torch.float32).eps
    p = teacher.exp()
    inner = (p * (teacher - student)).sum(-1)
    log_teacher_tail = torch.log1p(-p.sum(-1).clamp(min=eps, max=upper))
    log_student_tail = torch.log1p(-student.exp().sum(-1).clamp(min=eps, max=upper))
    return inner + log_teacher_tail.exp() * (log_teacher_tail - log_student_tail)


def _chunk_losses(h, head_weight, targets, weight, topk_ids, topk_logprobs, kl_weight, negative):
    """[weighted cross entropy, weighted KL, weighted unlikelihood] summed over one chunk of
    rows, from one projection. The logits are the head's own (bf16 operands give bf16
    logits, as `head(state)` did for the KL), widened to fp32 for everything after."""
    logits = (h @ head_weight.T).float()
    lse = torch.logsumexp(logits, -1)
    nll = lse - logits.gather(-1, targets[:, None]).squeeze(-1)
    zero = logits.new_zeros(())
    kl = zero
    if topk_ids is not None:
        keep = kl_weight > 0
        if bool(keep.any()):
            # Gather first: indexing the rows of `logits` would copy them, 485 MiB a chunk.
            student = logits.gather(-1, topk_ids)[keep] - lse[keep, None]
            kl = (grouped_tail_kl_rows(student, topk_logprobs[keep].float()) * kl_weight[keep]).sum()
    ul = zero
    if negative is not None and bool(negative.any()):
        p = (-nll[negative]).exp()
        ul = (-torch.log1p(-p.clamp(max=1 - 1e-6)) * weight[negative]).sum()
    return torch.stack([(nll * weight).sum(), kl, ul])


def head_losses(hidden, head_weight, ids, weight=None, topk_ids=None, topk_logprobs=None,
                kl_weight=None, negative=None, chunk=512, kl_beyond=False):
    """The head's losses for one record, projecting only the rows that carry weight.

    hidden [rows, length, d] and everything else on the head's device. Position t predicts
    token t + 1; `weight` [rows, length] is each position's loss weight (None: 1 for every
    position but the last), `kl_weight` the teacher KL's (None: no KL; it must be zero
    wherever `weight` is, unless `kl_beyond`: then rows with KL weight alone are projected
    too, for KL only -- the context tokens of an assistant-only document), `negative` the
    looping positions that take unlikelihood.

    Returns sums, for the caller to normalize as before: `nll` (sum of weight x cross
    entropy), `weight` (sum of weights), `kl` (sum of kl_weight x KL), `unlikelihood`
    (sum of weight x -log(1 - p) over the negatives).
    """
    rows, length = ids.shape
    w = torch.zeros(rows, length, device=hidden.device, dtype=torch.float32)
    w[:, :-1] = 1.0 if weight is None else weight[:, :-1].float()
    if kl_beyond and kl_weight is not None:
        kl_weight = kl_weight.clone()
        kl_weight[:, -1] = 0  # the last position predicts nothing
        at = ((w > 0) | (kl_weight > 0)).nonzero()
    else:
        if kl_weight is not None and bool((kl_weight[:, :-1] > 0)[w[:, :-1] <= 0].any()):
            raise ValueError("kl_weight must be zero wherever weight is")
        at = (w > 0).nonzero()
    r, t = at[:, 0], at[:, 1]
    h, targets, wr = hidden[r, t], ids[r, t + 1], w[r, t]
    kid = None if topk_ids is None else topk_ids[r, t]
    kval = None if topk_logprobs is None else topk_logprobs[r, t]
    kw = None if kl_weight is None else kl_weight[r, t].float()
    neg = None if negative is None else negative[r, t]
    grad = torch.is_grad_enabled() and (h.requires_grad or head_weight.requires_grad)
    total = hidden.new_zeros(3, dtype=torch.float32)
    for b in range(0, len(h), chunk):
        s = slice(b, b + chunk)
        args = (h[s], head_weight, targets[s], wr[s], None if kid is None else kid[s],
                None if kval is None else kval[s], None if kw is None else kw[s], None if neg is None else neg[s])
        total = total + (checkpoint(_chunk_losses, *args, use_reentrant=False, preserve_rng_state=False)
                         if grad else _chunk_losses(*args))
    return {"nll": total[0], "weight": wr.sum(), "kl": total[1], "unlikelihood": total[2]}

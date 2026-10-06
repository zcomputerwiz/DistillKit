# Copyright (C) 2024 Apple Inc. All Rights Reserved.
# Assisted-by: Codex
"""Isolated diagnostic extension of CCE's projection, not a training implementation.

The dot/rounding configuration matches installed cut-cross-entropy 25.9.3. Selected
logits and omitted-vocabulary statistics are extracted from those same projection
tiles. Optional tail statistics use sorted-ID membership, without subtracting nearly
unit selected probability mass. A CPU-prepared bucket option limits that membership
work to the IDs in each vocabulary tile. This module never modifies the installed
package. Neither forward variant implements the training loss's backward.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from cut_cross_entropy.tl_utils import tl_logaddexp


@triton.jit
def _selected_forward(E, C, Ids, SortedIds, SortedOrder, BucketCount, BucketStart,
                      Target, Pick, Correct, LSE, Locks,
                      PartialMax, PartialSum, TailMax, TailSum,
                      B: tl.constexpr, V: tl.constexpr, D: tl.constexpr, BUCKETED: tl.constexpr,
                      LOCKED: tl.constexpr,
                      K: tl.constexpr, BK: tl.constexpr, LOGK: tl.constexpr, TAIL: tl.constexpr,
                      BLOCK_B: tl.constexpr = 128, BLOCK_V: tl.constexpr = 128,
                      BLOCK_D: tl.constexpr = 32):
    # Same grouped CTA order and dot tiles as CCE's non-autotuned BF16 kernel.
    pid = tl.program_id(0)
    nb, nv = tl.cdiv(B, BLOCK_B), tl.cdiv(V, BLOCK_V)
    group = pid // (8 * nv)
    first = group * 8
    gs = tl.minimum(nb - first, 8)
    pb = first + (pid % (8 * nv)) % gs
    pv = (pid % (8 * nv)) // gs
    rb = pb * BLOCK_B + tl.arange(0, BLOCK_B)
    rv = pv * BLOCK_V + tl.arange(0, BLOCK_V)
    rd = tl.arange(0, BLOCK_D)
    ep = E + rb[:, None] * D + rd[None, :]
    cp = C + rv[None, :] * D + rd[:, None]
    accum = tl.zeros((BLOCK_B, BLOCK_V), tl.float32)
    for d in range(tl.cdiv(D, BLOCK_D)):
        e = tl.load(ep, (rb[:, None] < B) & (rd[None, :] + d * BLOCK_D < D), 0.)
        c = tl.load(cp, (rv[None, :] < V) & (rd[:, None] + d * BLOCK_D < D), 0.)
        accum = tl.dot(e, c, accum, input_precision="ieee")
        ep += BLOCK_D
        cp += BLOCK_D
    tl.debug_barrier()
    logits = accum.cast(E.dtype.element_ty, fp_downcast_rounding="rtne").cast(tl.float32)
    logits = tl.where(rv[None, :] < V, logits, -float("inf"))

    if BUCKETED:
        # Counts/starts are [vocabulary tile, row], coalesced across this CTA's rows.
        count = tl.load(BucketCount + pv * B + rb, rb < B, other=0).to(tl.int32)
        start = tl.load(BucketStart + pv * B + rb, rb < B, other=0).to(tl.int32)
        tail = logits
        # Empty CTAs run zero iterations. A touched row loads only its local IDs.
        for local_index in range(tl.max(count, 0)):
            valid_pick = (rb < B) & (local_index < count)
            offset = rb * K + start + local_index
            chosen_id = tl.load(SortedIds + offset, valid_pick, other=-1)
            original_column = tl.load(SortedOrder + offset, valid_pick, other=0).to(tl.int32)
            owned = valid_pick[:, None] & (chosen_id[:, None] == rv[None, :])
            # Reduce the unique owned column before the row-shaped output store.
            chosen = tl.sum(tl.where(owned, logits, 0.), 1)
            tl.store(Pick + rb * K + original_column, chosen, valid_pick)
            if TAIL:
                tail = tl.where(owned, -float("inf"), tail)
    else:
        rk = tl.arange(0, BK)
        ids = tl.load(Ids + rb[:, None] * K + rk[None, :],
                      (rb[:, None] < B) & (rk[None, :] < K), other=-1)
        local = ids - pv * BLOCK_V
        owned = (local >= 0) & (local < BLOCK_V) & (rk[None, :] < K) & (rb[:, None] < B)
        chosen = tl.gather(logits, tl.minimum(tl.maximum(local, 0), BLOCK_V - 1).to(tl.int32), 1)
        tl.store(Pick + rb[:, None] * K + rk[None, :], chosen, owned)
    target = tl.load(Target + rb, rb < B, other=-1)
    selected = tl.sum(tl.where(target[:, None] == rv[None, :], logits, 0.), 1)
    tl.store(Correct + rb, selected, (rb < B) & (target >= pv * BLOCK_V) &
             (target < (pv + 1) * BLOCK_V))

    mx = tl.max(logits, 1)
    sm = tl.sum(tl.exp(logits - mx[:, None]), 1)
    tl.store(PartialMax + rb * nv + pv, mx, rb < B)
    tl.store(PartialSum + rb * nv + pv, sm, rb < B)
    if LOCKED:
        this_lse = mx + tl.log(sm)
        lock = Locks + pb
        while tl.atomic_cas(lock, 0, 1) == 1:
            pass
        old = tl.load(LSE + rb, rb < B, other=-float("inf"))
        tl.store(LSE + rb, tl_logaddexp(old, this_lse), rb < B)
        tl.debug_barrier()
        tl.atomic_xchg(lock, 0)

    if TAIL:
        if not BUCKETED:
            # Retained baseline: O(log K) lookups per vocabulary entry.
            lo = tl.full((BLOCK_B, BLOCK_V), 0, tl.int32)
            hi = tl.full((BLOCK_B, BLOCK_V), K, tl.int32)
            for _ in range(LOGK):
                mid = (lo + hi) // 2
                value = tl.load(SortedIds + rb[:, None] * K + mid,
                                (rb[:, None] < B) & (mid < K), other=V + 1)
                lower = value < rv[None, :]
                lo = tl.where(lower, mid + 1, lo)
                hi = tl.where(lower, hi, mid)
            found = tl.load(SortedIds + rb[:, None] * K + lo,
                            (rb[:, None] < B) & (lo < K), other=V + 1)
            tail = tl.where(found == rv[None, :], -float("inf"), logits)
        tm = tl.max(tail, 1)
        # A tile can be completely selected when K >= BLOCK_V.
        ts = tl.sum(tl.exp(tail - tl.where(tm == -float("inf"), 0., tm)[:, None]), 1)
        tl.store(TailMax + rb * nv + pv, tm, rb < B)
        tl.store(TailSum + rb * nv + pv, ts, rb < B)


def _reduce_parts(mx, sums):
    # FP64 reduction diagnoses the error of CCE's locked, incremental FP32 logaddexp.
    maximum = mx.max(-1).values.double()
    anchor = torch.where(torch.isneginf(maximum), 0., maximum)
    total = (sums.double() * (mx.double() - anchor[:, None]).exp()).sum(-1)
    return maximum + total.log()


def selected_forward(hidden, head, targets, ids, *, tail=False, buckets=None, locked=True):
    """Forward-only diagnostics; locked=False omits stock LSE comparison work."""
    assert hidden.ndim == head.ndim == ids.ndim == 2
    assert hidden.dtype == head.dtype == torch.bfloat16
    assert hidden.is_contiguous() and head.is_contiguous() and ids.is_contiguous()
    b, d = hidden.shape
    v, dh = head.shape
    assert d == dh and targets.shape == (b,) and ids.shape[0] == b
    k = ids.shape[1]
    nv = triton.cdiv(v, 128)
    pick = torch.empty((b, k), device=hidden.device, dtype=torch.float32)
    correct = torch.empty((b,), device=hidden.device, dtype=torch.float32)
    locked_lse = torch.full((b,), -torch.inf, device=hidden.device, dtype=torch.float32) if locked else None
    locks = torch.zeros(triton.cdiv(b, 128), device=hidden.device, dtype=torch.int32) if locked else None
    mx = torch.empty((b, nv), device=hidden.device, dtype=torch.float32)
    sm = torch.empty_like(mx)
    tm = torch.empty_like(mx) if tail else None
    ts = torch.empty_like(mx) if tail else None
    if buckets is None:
        sorted_ids = ids.sort(-1).values if tail else ids
        sorted_order = counts = starts = None
    else:
        assert buckets["vocab"] == v and buckets["block_v"] == 128
        sorted_ids, sorted_order = buckets["sorted_ids"], buckets["sorted_order"]
        counts, starts = buckets["counts"], buckets["starts"]
        assert sorted_ids.shape == sorted_order.shape == ids.shape
        assert counts.shape == starts.shape == (nv, b)
        assert sorted_ids.dtype == torch.int64
        assert sorted_order.dtype == counts.dtype == starts.dtype == torch.uint8
        assert all(x.is_contiguous() and x.device == hidden.device
                   for x in (sorted_ids, sorted_order, counts, starts))
    _selected_forward[(triton.cdiv(b, 128) * nv,)](
        hidden, head, ids, sorted_ids, sorted_order, counts, starts,
        targets, pick, correct, locked_lse, locks,
        mx, sm, tm, ts, b, v, d, buckets is not None, locked,
        k, triton.next_power_of_2(k), k.bit_length(), tail,
        num_warps=4, num_stages=4)
    stable_lse = _reduce_parts(mx, sm)
    return {"pick": pick, "correct": correct, "locked_lse": locked_lse,
            "stable_lse": stable_lse,
            "tail_lse": _reduce_parts(tm, ts) if tail else None}


def prepare_buckets_cpu(ids, vocab, block_v=128):
    """Pure NumPy preparation, suitable for CPU teacher-cache data before transfer.

    uint8 counts, starts and original-column indices support this diagnostic's K<=255.
    Dense tile metadata costs 2*rows*ceil(vocab/block_v) bytes, not rows*vocab bytes.
    Every real teacher row must contain unique IDs; no padding rows belong here.
    """
    import numpy as np

    original = np.asarray(ids)
    if original.ndim != 2 or original.dtype.kind not in "iu":
        raise ValueError("teacher IDs must be a two-dimensional integer array")
    rows, k = original.shape
    if rows < 1 or not 1 <= k <= 255 or vocab < 1 or block_v < 1:
        raise ValueError("need nonempty rows, 1 <= K <= 255, and positive vocabulary/tile size")
    if np.any(original < 0) or np.any(original >= vocab):
        raise ValueError("teacher ID is outside the vocabulary")
    order = np.argsort(original, axis=1, kind="stable").astype(np.uint8)
    sorted_ids = np.take_along_axis(original, order, axis=1).astype(np.int64)
    if np.any(sorted_ids[:, 1:] == sorted_ids[:, :-1]):
        raise ValueError("real teacher top-k rows must contain unique IDs")
    tiles = (vocab + block_v - 1) // block_v
    counts = np.zeros((rows, tiles), dtype=np.uint8)
    np.add.at(counts, (np.repeat(np.arange(rows), k), (sorted_ids // block_v).ravel()), 1)
    starts = (np.cumsum(counts, axis=1, dtype=np.uint16).astype(np.uint8) - counts).astype(np.uint8)
    # Transpose once on CPU, so each GPU CTA loads contiguous row metadata.
    arrays = dict(sorted_ids=np.ascontiguousarray(sorted_ids), sorted_order=np.ascontiguousarray(order),
                  counts=np.ascontiguousarray(counts.T), starts=np.ascontiguousarray(starts.T))
    # Report the actual CTA loop work: different rows can touch different tiles.
    # Empty row buckets do not imply empty 128-row CTAs.
    padded_rows = ((rows + 127) // 128) * 128
    cta_counts = (counts if padded_rows == rows else
                  np.pad(counts, ((0, padded_rows - rows), (0, 0))))
    cta_max = cta_counts.reshape(-1, 128, tiles).max(axis=1)
    work = {"row_tile_pairs": int(np.count_nonzero(counts)),
            "total_row_tile_pairs": int(rows * tiles),
            "nonempty_ctas": int(np.count_nonzero(cta_max)),
            "total_ctas": int(cta_max.size),
            "cta_local_id_iterations": int(cta_max.sum(dtype=np.uint64)),
            "max_local_ids_in_any_row_tile": int(counts.max())}
    return {**arrays, "vocab": int(vocab), "block_v": int(block_v),
            "metadata_bytes": int(sum(a.nbytes for a in arrays.values())), "work": work}


def upload_buckets(prepared, device):
    """Explicit transfer outside the timed forward; preparation/transfer need separate timing."""
    return {name: torch.from_numpy(value).to(device) if hasattr(value, "dtype") else value
            for name, value in prepared.items()}

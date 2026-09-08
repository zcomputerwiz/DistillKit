"""Single-process tensor parallelism over two peer-accessible CUDA devices.

Deliberately not built on ``torch.distributed``. A process group exists to reach
ranks in *other* processes or on other machines; both of our devices are in this
one, and CUDA peer access already lets either read the other's memory. So an
all-reduce here is a peer copy and an add, and autograd handles the copy's backward
without help.

That removes the whole reason to depend on a collective library:

* ``torch.distributed.is_nccl_available()`` is False on this Windows wheel -- NCCL
  is gated on UNIX at build time -- and dropping a DLL beside it cannot add
  ``ProcessGroupNCCL``, which is genuinely absent from ``_distributed_c10d``.
* NCCL would buy nothing anyway at this scale. Measured on this machine: peer
  copies run at 38-48 GB/s for 64 MiB payloads against NCCL's 37.5 GB/s all-reduce.
  Its ring algorithms and multi-node transports are solving a problem we do not
  have.
* No ``Work`` objects, no process group lifecycle, no ``wait()`` ordering -- which
  is where the deadlocks in a hand-rolled backend would have lived.

llama.cpp's ``allreduce.cu`` was considered as a donor and rejected: it stages
through pinned host memory *because* it targets machines without NVLink, and it is
inference-only, so the part that is hard here -- backward through the collective --
is not there to borrow.

The sharding *specification* is borrowed, from transformers' own ``base_model_tp_plan``
for this architecture (colwise on q/k/v/gate/up, rowwise on o_proj/down_proj).
"""

from __future__ import annotations

import torch


def peer_capable(devices) -> bool:
    """True when every ordered pair of devices can access the other's memory."""
    indices = [torch.device(d).index for d in devices]
    for a in indices:
        for b in indices:
            if a != b and not torch.cuda.can_device_access_peer(a, b):
                return False
    return True


class AllReduce(torch.autograd.Function):
    """Sum matching tensors across devices; every device receives the total.

    Forward is ``y_i = sum_j x_j`` for each device i. Each input contributes to
    every output, so each input's gradient is the sum of all output gradients --
    the same reduction, which is why backward mirrors forward exactly.
    """

    @staticmethod
    def forward(ctx, *shards):
        _save_recompute_barrier(ctx, shards[0])
        return _sum_to_each(shards)

    @staticmethod
    def backward(ctx, *grads):
        _ = ctx.saved_tensors  # Recompute before releasing per-device branches.
        return _sum_to_each(grads)


def _save_recompute_barrier(ctx, tensor):
    """Make a reduction trigger checkpoint recomputation before backward forks.

    Non-reentrant checkpoint's saved-tensor unpack hook (torch 2.11) does not
    serialize recomputation across device workers. Without a saved tensor here,
    both upstream shards can unpack concurrently and recompute the same frame,
    corrupting its saved-tensor counter. Unpacking this empty sentinel in the
    reduction's single backward node finishes recomputation before either shard
    is scheduled. No activation storage or numerical operation is needed.
    """
    ctx.save_for_backward(tensor.new_empty(0))


def _sum_to_each(shards):
    """Reduce a tuple of same-shaped tensors on different devices, one copy each.

    Sums onto the first device, then sends the result out. Two transfers per extra
    device rather than the n^2 an all-pairs sum would cost.
    """
    if len(shards) == 1:
        return (shards[0],)
    home = shards[0].device
    total = shards[0]
    for shard in shards[1:]:
        total = total + shard.to(home)
    return tuple(
        total if shard.device == home else total.to(shard.device)
        for shard in shards
    )


def all_reduce(shards):
    """Autograd-aware all-reduce across devices. Returns one tensor per input."""
    return AllReduce.apply(*shards)


class Replicate(torch.autograd.Function):
    """Send one tensor to every device; gradients come back summed.

    The forward direction of a column-parallel layer: every shard needs the same
    input, and each shard's gradient with respect to that input is a separate
    contribution, so backward is an all-reduce.
    """

    @staticmethod
    def forward(ctx, source, *devices):
        ctx.source_device = source.device
        return tuple(
            source if source.device == device else source.to(device)
            for device in devices
        )

    @staticmethod
    def backward(ctx, *grads):
        home = ctx.source_device
        total = None
        for grad in grads:
            moved = grad if grad.device == home else grad.to(home)
            total = moved if total is None else total + moved
        return (total,) + (None,) * len(grads)


def replicate(source: torch.Tensor, devices):
    """Broadcast ``source`` to each device, summing gradients on the way back."""
    return Replicate.apply(source, *[torch.device(d) for d in devices])


class Reduce(torch.autograd.Function):
    """Sum shards onto one device only, rather than onto all of them.

    An all-reduce whose result is used on a single device still pays to copy the
    total back out to the others. Where the consumer is one device -- which is every
    row-parallel layer here, since the residual stream lives on the home card -- this
    halves the traffic: n-1 copies in, none back.

    Backward is the mirror: the single output gradient is broadcast to each shard's
    device, since each contributed additively.
    """

    @staticmethod
    def forward(ctx, home, *shards):
        ctx.shard_devices = [shard.device for shard in shards]
        _save_recompute_barrier(ctx, shards[0])
        total = shards[0] if shards[0].device == home else shards[0].to(home)
        for shard in shards[1:]:
            total = total + shard.to(home)
        return total

    @staticmethod
    def backward(ctx, grad):
        _ = ctx.saved_tensors  # Recompute before releasing per-device branches.
        return (None,) + tuple(
            grad if grad.device == device else grad.to(device)
            for device in ctx.shard_devices
        )


def reduce_to(shards, home) -> torch.Tensor:
    """Autograd-aware reduction onto ``home``. One tensor out, not one per device."""
    return Reduce.apply(torch.device(home), *shards)

"""The tied embedding and head split across devices along the vocabulary.

Every other block in the tensor-parallel student is split by channel or by head and
keeps the residual stream on the home card. The tied embedding/head is different: one
``[248320, 2560]`` parameter serving two roles, 16.4% of the model, and with its
gradient and 8-bit optimizer moments 3.5 GiB that has to land somewhere. Whole on the
home card it made card 0 the heavier one by about 5 GiB; whole on the other card it
flipped the imbalance (11.42 / 15.02 GiB measured at sequence 4096). Split by rows it
lands half on each, and the head's ~20 TFLOP per 4096-token microbatch -- projection,
recompute and both backward products -- splits with it.

This is Megatron-LM's ``VocabParallelEmbedding`` and column-parallel head in the
single-process form the rest of ``tp_*`` uses, plus the loss-side composition the
folded head needs:

* **Lookup.** Each rank embeds the ids that fall in its range and contributes zero rows
  for the rest; one reduction onto home sums them. Exact, since every id belongs to
  exactly one rank.
* **Head.** Each rank projects its rows of the vocabulary. The sparse divergences need
  only two things from a full row of logits -- its log-sum-exp and its values at the
  teacher's top-k ids -- and both compose from per-rank pieces: the log-sum-exp of the
  per-rank log-sum-exps, and a masked gather summed across ranks.
  :class:`VocabShardedLogits` carries the pieces and does that composition, so a
  chunk's 248,320-wide row never exists on any card; what crosses NVLink per chunk is
  ``[batch, chunk, 1]`` and ``[batch, chunk, k]``.
* **Dense projection** is still available (:meth:`VocabParallelHead.forward`) for the
  model's own ``lm_head`` call, which under the folded head is one position wide.

The tie is kept the way a shared parameter keeps it: the head holds the *same*
``ParameterList`` as the embedding, so there is one parameter per shard, one gradient
and one optimizer state, and ``tp_checkpoint`` reconstructs the stock
``model.embed_tokens.weight`` / ``lm_head.weight`` pair from it.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from distillkit.tensor_parallel import _save_recompute_barrier, reduce_to, replicate
from distillkit.tp_linear import split_sizes

__all__ = [
    "Collect",
    "VocabParallelEmbedding",
    "VocabParallelHead",
    "VocabShardedLogits",
    "collect_to",
]


class Collect(torch.autograd.Function):
    """Copy one tensor from each device onto home; the many-output sibling of Reduce.

    Carries the same recompute barrier: its single backward node unpacks the sentinel,
    finishing checkpoint recomputation before the gradients fork to the per-device
    branches that would otherwise race to recompute the same frame.
    """

    @staticmethod
    def forward(ctx, home, *parts):
        ctx.devices = [part.device for part in parts]
        _save_recompute_barrier(ctx, parts[0])
        return tuple(part if part.device == home else part.to(home) for part in parts)

    @staticmethod
    def backward(ctx, *grads):
        _ = ctx.saved_tensors  # Recompute before releasing per-device branches.
        return (None,) + tuple(
            grad if grad.device == device else grad.to(device)
            for grad, device in zip(grads, ctx.devices)
        )


def collect_to(parts, home) -> tuple[torch.Tensor, ...]:
    """Autograd-aware copy of every tensor onto ``home``, in order."""
    return Collect.apply(torch.device(home), *parts)


class VocabShardedLogits:
    """Per-rank logits over disjoint vocabulary ranges, never concatenated.

    Quacks just enough like the ``[batch, seq, vocab]`` tensor the sparse divergences
    expect (``shape``, ``dtype``) for their bookkeeping; the one operation that needs
    the full row -- log-softmax at the teacher's ids -- is :meth:`sparse_logprobs`.
    """

    def __init__(self, shards, starts, home):
        self.shards = list(shards)
        self.starts = list(starts)
        self.home = torch.device(home)

    @property
    def shape(self) -> torch.Size:
        batch, seq, _ = self.shards[0].shape
        return torch.Size((batch, seq, sum(shard.shape[-1] for shard in self.shards)))

    @property
    def dtype(self) -> torch.dtype:
        return self.shards[0].dtype

    @property
    def device(self) -> torch.device:
        return self.home

    def truncate(self, vocab_size: int) -> "VocabShardedLogits":
        """Drop columns past ``vocab_size``: the student's head padded wider than the signal."""
        shards, starts = [], []
        for shard, start in zip(self.shards, self.starts):
            if start >= vocab_size:
                continue
            keep = min(shard.shape[-1], vocab_size - start)
            shards.append(shard[..., :keep] if keep < shard.shape[-1] else shard)
            starts.append(start)
        return VocabShardedLogits(shards, starts, self.home)

    def sparse_logprobs(self, target_ids: torch.Tensor, scale: float | None = None) -> torch.Tensor:
        """``log_softmax(logits * scale)`` gathered at ``target_ids``, in fp32 on home.

        Widening bf16 to fp32 is exact and gathering commutes with it, so this matches
        ``logits.gather(-1, ids) - logsumexp(logits.float())`` over the whole row up to
        the rounding of composing the log-sum-exp from per-rank partials.
        """
        pieces = []
        for shard, start in zip(self.shards, self.starts):
            logits = shard.to(torch.float32)
            if scale is not None:
                logits = logits * scale
            local = target_ids.to(logits.device) - start
            inside = (local >= 0) & (local < logits.shape[-1])
            pieces.append(torch.logsumexp(logits, dim=-1, keepdim=True))
            pieces.append(logits.gather(-1, local.masked_fill(~inside, 0)) * inside)
        # One Collect for every piece: a single barrier node in backward.
        pieces = collect_to(pieces, self.home)
        lse = torch.logsumexp(torch.stack(pieces[0::2]), dim=0)
        picked = pieces[1]
        for piece in pieces[3::2]:
            picked = picked + piece
        return picked - lse


class VocabParallelEmbedding(nn.Module):
    """``nn.Embedding`` with its rows split across devices; the result lands on home."""

    def __init__(self, source: nn.Embedding, devices):
        super().__init__()
        self.devices = [torch.device(d) for d in devices]
        self.home = self.devices[0]
        self.num_embeddings = source.num_embeddings
        self.embedding_dim = source.embedding_dim
        self.padding_idx = source.padding_idx
        sizes = split_sizes(source.num_embeddings, len(self.devices))
        self.starts = [sum(sizes[:rank]) for rank in range(len(sizes))]
        self.shards = nn.ParameterList([
            nn.Parameter(
                source.weight.data[start:start + size].detach().clone().to(device),
                requires_grad=source.weight.requires_grad,
            )
            for start, size, device in zip(self.starts, sizes, self.devices)
        ])

    @property
    def device(self) -> torch.device:
        return self.home

    def _local_padding_idx(self, start: int, size: int) -> int | None:
        if self.padding_idx is None or not start <= self.padding_idx < start + size:
            return None
        return self.padding_idx - start

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        parts = []
        for shard, start in zip(self.shards, self.starts):
            local = input_ids.to(shard.device) - start
            inside = (local >= 0) & (local < shard.shape[0])
            rows = F.embedding(
                local.masked_fill(~inside, 0), shard,
                padding_idx=self._local_padding_idx(start, shard.shape[0]),
            )
            parts.append(rows * inside.unsqueeze(-1).to(rows.dtype))
        return reduce_to(parts, self.home)


class VocabParallelHead(nn.Module):
    """The output side of the tie: the embedding's own shards, used as ``lm_head``."""

    def __init__(self, embedding: VocabParallelEmbedding):
        super().__init__()
        self.shards = embedding.shards  # the same ParameterList: one parameter per shard
        self.starts = embedding.starts
        self.devices = embedding.devices
        self.home = embedding.home
        self.in_features = embedding.embedding_dim
        self.out_features = embedding.num_embeddings

    @property
    def device(self) -> torch.device:
        return self.home

    def sharded_logits(
        self, hidden_states: torch.Tensor, vocab_size: int | None = None
    ) -> VocabShardedLogits:
        """Project on every rank and keep the pieces apart: what the folded loss wants."""
        copies = replicate(hidden_states, self.devices)
        logits = VocabShardedLogits(
            [F.linear(copy, shard) for copy, shard in zip(copies, self.shards)],
            self.starts, self.home,
        )
        if vocab_size is not None and logits.shape[-1] > vocab_size:
            logits = logits.truncate(vocab_size)
        return logits

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Dense logits on home. Costs the full width; the loss uses ``sharded_logits``."""
        logits = self.sharded_logits(hidden_states)
        return torch.cat(collect_to(logits.shards, self.home), dim=-1)

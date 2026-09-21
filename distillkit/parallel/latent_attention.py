"""Tensor-parallel sharding for the MLA / CSA2 attention.

`TensorParallelAttention` shards q/k/v by head and reduces o_proj, which is the whole
story for stock attention. MLA has no `k_proj` or `v_proj` to shard: it compresses the
keys and values into one latent with `kv_a_proj` and expands that latent per head with
`kv_b_proj`. So the split falls in a different place, and one projection must not be
split at all.

    q_proj        hidden -> heads * head_dim * 2     split by head (query and gate are
                                                     head-major, so a contiguous cut
                                                     lands on a head boundary)
    kv_a_proj     hidden -> latent + rope_dim        REPLICATED: one latent shared by
                                                     every head, and by every layer that
                                                     borrows it off the bus
    kv_b_proj     latent -> heads * (content + head) split by head
    o_proj        heads * head_dim -> hidden         split by input channel, reduced home
    index_*       the router                         REPLICATED, and that is load-bearing

**Why the router is replicated rather than split.** It decides *which positions* every
head reads. Splitting it would let the two ranks reach different top-k sets where scores
tie at the cutoff -- which they do: on this checkpoint 5.3% to 31.7% of reachable cells
score exactly zero and up to 10.2% of queries have a tie at the budget. The heads would
then attend to different histories and the concatenation would be incoherent. It is also
small -- 5.7M parameters against the body's 1,375M -- so replicating costs nothing worth
counting.

**Why the outputs are gathered rather than kept split.** The CSA2 forward is not a
sequence of projections: it routes, borrows latents across layers through the bus,
optionally absorbs the up-projection into the query, and switches between a gathered path
and a FlexAttention block path on conditions computed inside it. Re-deriving all of that
in a sharded forward would be a second implementation of the hard part, and the two would
have to be kept in step. Sharding the parameters and gathering the result keeps one
implementation of the routing and still moves the weights, their gradients and their
optimizer state off the home card -- which is what the memory ceiling is made of. It buys
less than a split forward would: the activation on home is unchanged, and each projection
pays one peer transfer.

Measured on this model, that trade is worth taking for a different reason than it looks:
the MLA/CSA2 attention is 90M parameters, 4.7% of the model, against the MLP's 906M and
the linear-attention layers' 379M. This file exists so that sharding the other 67% does
not have to stop at a layer it cannot handle.
"""

from __future__ import annotations

import torch
from torch import nn

from distillkit.parallel.collectives import reduce_to, replicate
from distillkit.parallel.linear import split_sizes


def _devices(devices) -> list[torch.device]:
    return [torch.device(d) for d in devices]


class GatheredColumnLinear(nn.Module):
    """``nn.Linear`` whose output channels live on different devices.

    Presents the source's interface exactly -- one tensor in on home, one tensor out on
    home -- so a forward that was written against ``nn.Linear`` does not know the
    difference. Only the parameters move.
    """

    def __init__(self, source: nn.Linear, devices, head_multiple: int | None = None):
        super().__init__()
        self.devices = _devices(devices)
        self.in_features = source.in_features
        self.out_features = source.out_features
        parts = len(self.devices)
        if head_multiple is not None and (source.out_features // parts) % head_multiple:
            # A contiguous cut is only correct when it lands on a head boundary. Qwen3.5
            # packs the output gate into q_proj head-major, so a cut inside a head would
            # hand one rank a query and the other its gate, and it would train nonsense
            # rather than fail.
            raise ValueError(
                "cannot split %d output channels across %d devices without cutting "
                "inside a head of %d" % (source.out_features, parts, head_multiple))
        sizes = split_sizes(source.out_features, parts)
        weights, biases = [], []
        offset = 0
        for size, device in zip(sizes, self.devices):
            piece = slice(offset, offset + size)
            weights.append(nn.Parameter(
                source.weight.data[piece].detach().clone().to(device),
                requires_grad=source.weight.requires_grad))
            if source.bias is not None:
                biases.append(nn.Parameter(
                    source.bias.data[piece].detach().clone().to(device),
                    requires_grad=source.bias.requires_grad))
            offset += size
        self.shards = nn.ParameterList(weights)
        self.biases = nn.ParameterList(biases) if biases else None

    @property
    def weight(self) -> torch.Tensor:
        """The unsharded weight, assembled on home.

        Two callers need this. `_attend_absorbed` reads `kv_b_proj.weight` to fold the
        up-projection into the query, which is a decode path training never reaches.
        `_project` reads `q_proj.weight` on every forward, because it fuses the query,
        the latent and the index queries into one multiply to read the stream once.

        So this is a per-step cost for `q_proj`, not an exceptional one: one gather of
        the weight, 16 MiB at this width, against a step of several hundred milliseconds.
        What it does not cost is the memory the split was for -- the gather is a graph
        node, gradients scatter back through it to the shards, and weights, gradients and
        optimizer state all stay divided. Only the multiply comes home.
        """
        home = self.devices[0]
        return torch.cat([shard.to(home) for shard in self.shards], dim=0)

    @property
    def bias(self):
        """Matches `weight`, and answers None the way an unbiased `nn.Linear` does.

        `_project` tests `modules[0].bias is not None` to decide whether to concatenate
        biases at all, so answering with an empty list or raising would both be wrong.
        """
        if self.biases is None:
            return None
        home = self.devices[0]
        return torch.cat([shard.to(home) for shard in self.biases], dim=0)

    def forward(self, x):
        copies = replicate(x, self.devices)
        home = self.devices[0]
        parts = [
            nn.functional.linear(copy, weight,
                                 None if self.biases is None else self.biases[index])
            for index, (copy, weight) in enumerate(zip(copies, self.shards))
        ]
        return torch.cat([part.to(home, non_blocking=True) for part in parts], dim=-1)


class ReducedRowLinear(nn.Module):
    """``nn.Linear`` whose input channels live on different devices.

    Takes the whole input on home, splits it along the feature axis to the shards,
    and reduces the partial sums back. The bias is added once, after the reduction.
    """

    def __init__(self, source: nn.Linear, devices):
        super().__init__()
        self.devices = _devices(devices)
        self.in_features = source.in_features
        self.out_features = source.out_features
        sizes = split_sizes(source.in_features, len(self.devices))
        weights = []
        offset = 0
        for size, device in zip(sizes, self.devices):
            piece = slice(offset, offset + size)
            weights.append(nn.Parameter(
                source.weight.data[:, piece].detach().clone().to(device),
                requires_grad=source.weight.requires_grad))
            offset += size
        self.shards = nn.ParameterList(weights)
        self.bias = (nn.Parameter(source.bias.data.detach().clone().to(self.devices[0]),
                                  requires_grad=source.bias.requires_grad)
                     if source.bias is not None else None)
        self.sizes = sizes

    @property
    def weight(self) -> torch.Tensor:
        home = self.devices[0]
        return torch.cat([shard.to(home) for shard in self.shards], dim=1)

    def forward(self, x):
        pieces = torch.split(x, self.sizes, dim=-1)
        partials = [
            nn.functional.linear(piece.to(device, non_blocking=True), weight)
            for piece, weight, device in zip(pieces, self.shards, self.devices)
        ]
        total = reduce_to(partials, self.devices[0])
        return total if self.bias is None else total + self.bias


class RemoteEmbedding(nn.Module):
    """An embedding table that lives on a card other than home.

    The tied embedding and head are 508.6M parameters here -- 26.6% of the model, and
    3.05 GiB once a bf16 weight, a bf16 gradient and two 8-bit optimizer states are
    counted. They cannot be *split*, because Cut Cross-Entropy never forms the logits and
    needs the whole head to do that, which is the only reason a 248,320-wide vocabulary is
    affordable at all. They can be *moved*, and the two cards are not equally full: at
    micro-batch 6 home peaks at 20.61 GiB against the other card's 9.64.

    What crosses the link is the embedding's output rather than its table: one
    `[batch, length, hidden]` tensor each way, 25 MiB at micro-batch 6, against the
    3.05 GiB that stops crossing at all.
    """

    def __init__(self, source: nn.Embedding, device, home):
        super().__init__()
        self.away = torch.device(device)
        self.home = torch.device(home)
        self.embedding = source.to(self.away)
        self.num_embeddings = source.num_embeddings
        self.embedding_dim = source.embedding_dim

    @property
    def weight(self) -> torch.Tensor:
        return self.embedding.weight

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(ids.to(self.away)).to(self.home)


def head_device(model: nn.Module) -> torch.device:
    """Where a loss must put its hidden state to reach the output head.

    Callers hand `lm_head.weight` to Cut Cross-Entropy directly, so they need to know
    where it went. Returns home for an unmoved head, which keeps the ordinary path a
    no-op rather than a special case.
    """
    head = getattr(model, "lm_head", None)
    if head is None:
        return next(model.parameters()).device
    return head.weight.device


def is_latent_attention(module: nn.Module) -> bool:
    """A latent attention is one that compressed its keys and values away.

    Tested by what it has rather than by its class, so an MLA layer and a CSA2 layer
    and a borrowing CSA2 layer -- which has no `kv_a_proj` at all, because it reads
    another layer's latent off the bus -- all answer the same way.
    """
    return (hasattr(module, "kv_b_proj")
            and not hasattr(module, "k_proj")
            and not hasattr(module, "v_proj"))


def shard_latent_attention(attention: nn.Module, devices) -> nn.Module:
    """Split the per-head projections in place; leave the latent and the router home.

    Returns the same module. Only `q_proj`, `kv_b_proj` and `o_proj` are replaced, and
    each keeps its interface, so the attention's own forward is untouched.
    """
    resolved = _devices(devices)
    if len(resolved) < 2:
        return attention
    head_dim = attention.head_dim
    heads = attention.num_heads
    split_sizes(heads, len(resolved))

    # q_proj emits [query | gate] per head, so its head stride is 2 * head_dim.
    attention.q_proj = GatheredColumnLinear(
        attention.q_proj, resolved, head_multiple=head_dim * 2)
    # kv_b_proj emits [content key | value] per head.
    attention.kv_b_proj = GatheredColumnLinear(
        attention.kv_b_proj, resolved,
        head_multiple=attention.content_dim + head_dim)
    attention.o_proj = ReducedRowLinear(attention.o_proj, resolved)
    return attention

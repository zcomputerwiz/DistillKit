# Copyright 2025 Arcee AI & DistillKit Contributors
"""Single-process tensor parallelism and collectives for DistillKit.

Provides CUDA P2P autograd collectives, column- and row-parallel linear layers,
vocabulary sharding, and consolidated checkpoint serialization without depending on
torch.distributed or external process group backends.
"""

from distillkit.parallel.collectives import (
    AllReduce,
    Reduce,
    Replicate,
    all_reduce,
    peer_capable,
    reduce_to,
    replicate,
)
from distillkit.parallel.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
    split_sizes,
)
from distillkit.parallel.vocab import (
    Collect,
    VocabParallelEmbedding,
    VocabParallelHead,
    VocabShardedLogits,
    collect_to,
)
from distillkit.parallel.blocks import (
    TensorParallelAttention,
    TensorParallelMLP,
    shard_decoder_layer,
)
from distillkit.parallel.checkpoint import (
    MARKER,
    consolidated_state_dict,
    load_checkpoint,
    write_layout,
)
from distillkit.parallel.sync import (
    clip_grad_norm,
    replicated_parameter_groups,
    sync_replicated_gradients,
)
from distillkit.parallel.model import (
    shard_model,
    sharded_parameter_report,
)

__all__ = [
    # Collectives
    "AllReduce",
    "Reduce",
    "Replicate",
    "all_reduce",
    "peer_capable",
    "reduce_to",
    "replicate",
    # Linear
    "ColumnParallelLinear",
    "RowParallelLinear",
    "split_sizes",
    # Vocab
    "Collect",
    "collect_to",
    "VocabParallelEmbedding",
    "VocabParallelHead",
    "VocabShardedLogits",
    # Blocks
    "TensorParallelAttention",
    "TensorParallelMLP",
    "shard_decoder_layer",
    # Checkpoint
    "MARKER",
    "consolidated_state_dict",
    "load_checkpoint",
    "write_layout",
    # Sync
    "clip_grad_norm",
    "replicated_parameter_groups",
    "sync_replicated_gradients",
    # Model
    "shard_model",
    "sharded_parameter_report",
]

# Copyright 2025 Arcee AI & DistillKit Contributors
"""Experimental architectures, retrieval modules, and routing prototypes."""

from distillkit.experimental.donor_reader import (
    DonorReaderTransplant,
    initialise_transplant_reader,
    load_reference_tensors,
)
from distillkit.experimental.ffn_skip import (
    FFNSkip,
    FFNSubstitute,
    capture_ffn,
    skip_ffn,
    substitute_ffn,
)
from distillkit.experimental.gated_residual import (
    GateReport,
    GatedResidual,
)
from distillkit.experimental.hyper_connection import (
    HyperConnection,
)
from distillkit.experimental.memory_lane import (
    MemoryRead,
)
from distillkit.experimental.native_ple import (
    NativePLESidecar,
    native_hash_config,
)
from distillkit.experimental.ngram_hash import (
    FLASH_NEXT_NGRAM_CONFIG,
    NGramHashConfig,
    NGramHasher,
    build_layer_multipliers,
    find_nth_prime_after,
    splitmix64,
)
from distillkit.experimental.ngram_table import (
    FLASH_NEXT_TABLE,
    GGUFNGramTable,
    IQ4NL_BLOCK,
    IQ4NL_KVALUES,
    IQ4NL_TYPE_SIZE,
    IQ4NLDequant,
    NGramTableSpec,
    dequantize_iq4nl_rows,
    read_gguf_ple_metadata,
)
from distillkit.experimental.ple_gated_sidecar import (
    DirectionGatedPLESidecar,
)
from distillkit.experimental.ple_sidecar import (
    PLESidecar,
)
from distillkit.experimental.sidecar_collator import (
    SidecarDataCollator,
)
from distillkit.experimental.widened_residual import (
    WidenedResidual,
    branch_norm,
    collapse_residual,
    offload_stream_boundaries,
)

__all__ = [
    # Donor Reader
    "DonorReaderTransplant",
    "initialise_transplant_reader",
    "load_reference_tensors",
    # FFN Skip
    "FFNSkip",
    "FFNSubstitute",
    "capture_ffn",
    "skip_ffn",
    "substitute_ffn",
    # Gated Residual
    "GatedResidual",
    "GateReport",
    # HyperConnection
    "HyperConnection",
    # Memory Lane
    "MemoryRead",
    # Native PLE
    "NativePLESidecar",
    "native_hash_config",
    # NGram Hash
    "FLASH_NEXT_NGRAM_CONFIG",
    "NGramHashConfig",
    "NGramHasher",
    "build_layer_multipliers",
    "find_nth_prime_after",
    "splitmix64",
    # NGram Table
    "FLASH_NEXT_TABLE",
    "GGUFNGramTable",
    "IQ4NL_BLOCK",
    "IQ4NL_KVALUES",
    "IQ4NL_TYPE_SIZE",
    "IQ4NLDequant",
    "NGramTableSpec",
    "dequantize_iq4nl_rows",
    "read_gguf_ple_metadata",
    # PLE Sidecars
    "DirectionGatedPLESidecar",
    "PLESidecar",
    # Collator
    "SidecarDataCollator",
    # Widened Residual
    "WidenedResidual",
    "branch_norm",
    "collapse_residual",
    "offload_stream_boundaries",
]

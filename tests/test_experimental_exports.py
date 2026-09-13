"""Verify export parity, identity, and backward compatibility for distillkit.experimental."""

import distillkit.experimental as exp

import distillkit.donor_reader as legacy_donor_reader
import distillkit.analyze_donor_reader as legacy_analyze_donor
import distillkit.native_ple as legacy_native_ple
import distillkit.ple_sidecar as legacy_ple_sidecar
import distillkit.ple_gated_sidecar as legacy_ple_gated
import distillkit.ngram_table as legacy_ngram_table
import distillkit.ngram_hash as legacy_ngram_hash
import distillkit.sidecar_collator as legacy_sidecar_collator
import distillkit.gated_residual as legacy_gated_residual
import distillkit.widened_residual as legacy_widened_residual
import distillkit.hyper_connection as legacy_hyper_connection
import distillkit.ffn_skip as legacy_ffn_skip
import distillkit.memory_lane as legacy_memory_lane
import distillkit.borrowed_routing as legacy_borrowed_routing

import distillkit.experimental.donor_reader as exp_donor_reader
import distillkit.experimental.native_ple as exp_native_ple
import distillkit.experimental.ple_sidecar as exp_ple_sidecar
import distillkit.experimental.ple_gated_sidecar as exp_ple_gated
import distillkit.experimental.ngram_table as exp_ngram_table
import distillkit.experimental.ngram_hash as exp_ngram_hash
import distillkit.experimental.sidecar_collator as exp_sidecar_collator
import distillkit.experimental.gated_residual as exp_gated_residual
import distillkit.experimental.widened_residual as exp_widened_residual
import distillkit.experimental.hyper_connection as exp_hyper_connection
import distillkit.experimental.ffn_skip as exp_ffn_skip
import distillkit.experimental.memory_lane as exp_memory_lane
import distillkit.experimental.borrowed_routing as exp_borrowed_routing


def test_donor_reader_exports():
    assert exp.DonorReaderTransplant is exp_donor_reader.DonorReaderTransplant is legacy_donor_reader.DonorReaderTransplant
    assert exp.initialise_transplant_reader is exp_donor_reader.initialise_transplant_reader is legacy_donor_reader.initialise_transplant_reader
    assert exp.load_reference_tensors is exp_donor_reader.load_reference_tensors is legacy_donor_reader.load_reference_tensors


def test_native_ple_exports():
    assert exp.NativePLESidecar is exp_native_ple.NativePLESidecar is legacy_native_ple.NativePLESidecar
    assert exp.native_hash_config is exp_native_ple.native_hash_config is legacy_native_ple.native_hash_config


def test_ple_sidecar_exports():
    assert exp.PLESidecar is exp_ple_sidecar.PLESidecar is legacy_ple_sidecar.PLESidecar


def test_ple_gated_exports():
    assert exp.DirectionGatedPLESidecar is exp_ple_gated.DirectionGatedPLESidecar is legacy_ple_gated.DirectionGatedPLESidecar


def test_ngram_table_exports():
    assert exp.IQ4NL_BLOCK == exp_ngram_table.IQ4NL_BLOCK == legacy_ngram_table.IQ4NL_BLOCK
    assert exp.IQ4NL_KVALUES == exp_ngram_table.IQ4NL_KVALUES == legacy_ngram_table.IQ4NL_KVALUES
    assert exp.IQ4NL_TYPE_SIZE == exp_ngram_table.IQ4NL_TYPE_SIZE == legacy_ngram_table.IQ4NL_TYPE_SIZE
    assert exp.IQ4NLDequant is exp_ngram_table.IQ4NLDequant is legacy_ngram_table.IQ4NLDequant
    assert exp.NGramTableSpec is exp_ngram_table.NGramTableSpec is legacy_ngram_table.NGramTableSpec
    assert exp.FLASH_NEXT_TABLE == exp_ngram_table.FLASH_NEXT_TABLE == legacy_ngram_table.FLASH_NEXT_TABLE
    assert exp.dequantize_iq4nl_rows is exp_ngram_table.dequantize_iq4nl_rows is legacy_ngram_table.dequantize_iq4nl_rows
    assert exp.GGUFNGramTable is exp_ngram_table.GGUFNGramTable is legacy_ngram_table.GGUFNGramTable
    assert exp.read_gguf_ple_metadata is exp_ngram_table.read_gguf_ple_metadata is legacy_ngram_table.read_gguf_ple_metadata


def test_ngram_hash_exports():
    assert exp.NGramHashConfig is exp_ngram_hash.NGramHashConfig is legacy_ngram_hash.NGramHashConfig
    assert exp.NGramHasher is exp_ngram_hash.NGramHasher is legacy_ngram_hash.NGramHasher
    assert exp.FLASH_NEXT_NGRAM_CONFIG == exp_ngram_hash.FLASH_NEXT_NGRAM_CONFIG == legacy_ngram_hash.FLASH_NEXT_NGRAM_CONFIG
    assert exp.splitmix64 is exp_ngram_hash.splitmix64 is legacy_ngram_hash.splitmix64
    assert exp.build_layer_multipliers is exp_ngram_hash.build_layer_multipliers is legacy_ngram_hash.build_layer_multipliers
    assert exp.find_nth_prime_after is exp_ngram_hash.find_nth_prime_after is legacy_ngram_hash.find_nth_prime_after


def test_collator_exports():
    assert exp.SidecarDataCollator is exp_sidecar_collator.SidecarDataCollator is legacy_sidecar_collator.SidecarDataCollator


def test_residual_and_routing_exports():
    assert exp.GatedResidual is exp_gated_residual.GatedResidual is legacy_gated_residual.GatedResidual
    assert exp.GateReport is exp_gated_residual.GateReport is legacy_gated_residual.GateReport

    assert exp.WidenedResidual is exp_widened_residual.WidenedResidual is legacy_widened_residual.WidenedResidual
    assert exp_widened_residual._BranchNorm is legacy_widened_residual._BranchNorm
    assert exp.branch_norm is exp_widened_residual.branch_norm is legacy_widened_residual.branch_norm
    assert exp.collapse_residual is exp_widened_residual.collapse_residual is legacy_widened_residual.collapse_residual
    assert exp.offload_stream_boundaries is exp_widened_residual.offload_stream_boundaries is legacy_widened_residual.offload_stream_boundaries

    assert exp.HyperConnection is exp_hyper_connection.HyperConnection is legacy_hyper_connection.HyperConnection
    assert exp.FFNSkip is exp_ffn_skip.FFNSkip is legacy_ffn_skip.FFNSkip
    assert exp.skip_ffn is exp_ffn_skip.skip_ffn is legacy_ffn_skip.skip_ffn
    assert exp.MemoryRead is exp_memory_lane.MemoryRead is legacy_memory_lane.MemoryRead
    assert exp_borrowed_routing.initialise_widened_residual is legacy_borrowed_routing.initialise_widened_residual
    assert exp_borrowed_routing.initialise_ple_reader is legacy_borrowed_routing.initialise_ple_reader

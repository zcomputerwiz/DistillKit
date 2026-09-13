"""Verify export parity and compatibility for distillkit.parallel and model adapters."""

import distillkit.parallel as parallel
import distillkit.tensor_parallel as legacy_tp
import distillkit.tp_linear as legacy_linear
import distillkit.tp_vocab as legacy_vocab
import distillkit.tp_blocks as legacy_blocks
import distillkit.tp_checkpoint as legacy_checkpoint
import distillkit.tp_model as legacy_model
import distillkit.tp_gated_delta as legacy_gated_delta
import distillkit.tp_gated_delta_module as legacy_gated_delta_module

import distillkit.parallel.collectives as parallel_collectives
import distillkit.parallel.linear as parallel_linear
import distillkit.parallel.vocab as parallel_vocab
import distillkit.parallel.blocks as parallel_blocks
import distillkit.parallel.checkpoint as parallel_checkpoint
import distillkit.parallel.sync as parallel_sync
import distillkit.parallel.model as parallel_model

import distillkit.models.qwen35 as qwen35
import distillkit.models.qwen35.tp_gated_delta as qwen35_gated_delta
import distillkit.models.qwen35.tp_gated_delta_module as qwen35_gated_delta_module


def test_collectives_exports():
    assert parallel.AllReduce is parallel_collectives.AllReduce is legacy_tp.AllReduce
    assert parallel.Replicate is parallel_collectives.Replicate is legacy_tp.Replicate
    assert parallel.Reduce is parallel_collectives.Reduce is legacy_tp.Reduce
    assert parallel.all_reduce is parallel_collectives.all_reduce is legacy_tp.all_reduce
    assert parallel.replicate is parallel_collectives.replicate is legacy_tp.replicate
    assert parallel.reduce_to is parallel_collectives.reduce_to is legacy_tp.reduce_to
    assert parallel.peer_capable is parallel_collectives.peer_capable is legacy_tp.peer_capable


def test_linear_exports():
    assert parallel.ColumnParallelLinear is parallel_linear.ColumnParallelLinear is legacy_linear.ColumnParallelLinear
    assert parallel.RowParallelLinear is parallel_linear.RowParallelLinear is legacy_linear.RowParallelLinear
    assert parallel.split_sizes is parallel_linear.split_sizes is legacy_linear.split_sizes


def test_vocab_exports():
    assert parallel.Collect is parallel_vocab.Collect is legacy_vocab.Collect
    assert parallel.collect_to is parallel_vocab.collect_to is legacy_vocab.collect_to
    assert parallel.VocabParallelEmbedding is parallel_vocab.VocabParallelEmbedding is legacy_vocab.VocabParallelEmbedding
    assert parallel.VocabParallelHead is parallel_vocab.VocabParallelHead is legacy_vocab.VocabParallelHead
    assert parallel.VocabShardedLogits is parallel_vocab.VocabShardedLogits is legacy_vocab.VocabShardedLogits


def test_blocks_exports():
    assert parallel.TensorParallelMLP is parallel_blocks.TensorParallelMLP is legacy_blocks.TensorParallelMLP
    assert parallel.TensorParallelAttention is parallel_blocks.TensorParallelAttention is legacy_blocks.TensorParallelAttention
    assert parallel.shard_decoder_layer is parallel_blocks.shard_decoder_layer is legacy_blocks.shard_decoder_layer


def test_checkpoint_exports():
    assert parallel.MARKER == parallel_checkpoint.MARKER == legacy_checkpoint.MARKER
    assert parallel.consolidated_state_dict is parallel_checkpoint.consolidated_state_dict is legacy_checkpoint.consolidated_state_dict
    assert parallel.load_consolidated_state_dict is parallel_checkpoint.load_consolidated_state_dict is legacy_checkpoint.load_consolidated_state_dict
    assert parallel.training_layout is parallel_checkpoint.training_layout is legacy_checkpoint.training_layout
    assert parallel.write_layout is parallel_checkpoint.write_layout is legacy_checkpoint.write_layout
    assert parallel.load_checkpoint is parallel_checkpoint.load_checkpoint is legacy_checkpoint.load_checkpoint


def test_sync_exports():
    assert parallel.sync_replicated_gradients is parallel_sync.sync_replicated_gradients is legacy_model.sync_replicated_gradients is legacy_gated_delta_module.sync_replicated_gradients
    assert parallel.clip_grad_norm is parallel_sync.clip_grad_norm is legacy_gated_delta_module.clip_grad_norm
    assert parallel.replicated_parameter_groups is parallel_sync.replicated_parameter_groups


def test_model_exports():
    assert parallel.shard_model is parallel_model.shard_model is legacy_model.shard_model
    assert parallel.sharded_parameter_report is parallel_model.sharded_parameter_report is legacy_model.sharded_parameter_report


def test_qwen35_adapters_exports():
    assert qwen35.conv_channel_plan is qwen35_gated_delta.conv_channel_plan is legacy_gated_delta.conv_channel_plan
    assert qwen35.head_group_plan is qwen35_gated_delta.head_group_plan is legacy_gated_delta.head_group_plan
    assert qwen35.TensorParallelGatedDeltaNet is qwen35_gated_delta_module.TensorParallelGatedDeltaNet is legacy_gated_delta_module.TensorParallelGatedDeltaNet

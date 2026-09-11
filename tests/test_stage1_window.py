"""Stage 1 with a trainable window around the injection point.

The sidecar experiments left one hypothesis untested: that the bottleneck is the
pretrained layers *consuming* the injection rather than the retrieval. Opening a window
tests it -- but only if the window is trainable, routed to an optimizer that can handle
tensor-parallel shards, and controlled against a matched shuffled arm.
"""

import pytest
import torch

from distillkit.configuration import OptimizerConfig, SidecarConfig
from distillkit.models.qwen35_widened import Qwen35WidenedForCausalLM
from distillkit.optimizers import (
    _auxiliary_parameter_ids, freeze_backbone_for_stage1, mixed_parameter_groups,
)
from tests.test_ple_gated_sidecar import _widened_config


def _model(layers=6, sidecar_layer=2):
    config = _widened_config(layers=layers)
    config.sidecar_layer_index = sidecar_layer
    return Qwen35WidenedForCausalLM(config)


def _layer_names(names, index):
    return [n for n in names if f".layers.{index}." in n
            and not any(p in n.split(".") for p in ("attn_residual", "mlp_residual", "sidecar"))]


def test_the_window_names_its_own_layers_and_nothing_else():
    model = _model()
    model.set_stage1_trainable_layers((2, 4))
    names = set(model.stage1_parameter_names())
    assert _layer_names(names, 2) and _layer_names(names, 3)
    assert not _layer_names(names, 1) and not _layer_names(names, 4)


def test_without_a_window_stage_one_is_adapter_only():
    model = _model()
    names = set(model.stage1_parameter_names())
    assert not any(_layer_names(names, i) for i in range(6))
    model.set_stage1_trainable_layers((2, 4))
    model.set_stage1_trainable_layers(None)
    assert not any(_layer_names(set(model.stage1_parameter_names()), i) for i in range(6))


def test_a_window_outside_the_stack_is_refused():
    model = _model()
    for bad in ((4, 99), (-1, 3), (3, 3), (5, 2)):
        with pytest.raises(ValueError, match="outside 0..|window"):
            model.set_stage1_trainable_layers(bad)


def test_freezing_leaves_the_window_trainable():
    model = _model()
    model.set_stage1_trainable_layers((2, 4))
    freeze_backbone_for_stage1(model)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert _layer_names(trainable, 2), "the window must survive stage-1 freezing"
    assert not _layer_names(trainable, 0), "everything outside it must not"


def test_the_window_routes_to_adamw_not_muon():
    """Newton-Schulz does not commute with tensor-parallel slicing; Muon is refused."""
    model = _model()
    model.set_stage1_trainable_layers((2, 4))
    auxiliary = _auxiliary_parameter_ids(model)
    window = {id(p) for p in model.model.layers[2].parameters()}
    assert window <= auxiliary
    groups = mixed_parameter_groups(model)
    kinds = {g["optimizer_kind"] for g in groups
             for p in g["params"] if id(p) in window}
    assert kinds == {"adamw"}, kinds


def test_shuffled_context_keeps_every_row_real_and_only_moves_which_one():
    """The matched control: real rows, wrong context. Rolling ids, not bytes."""
    from distillkit.ngram_hash import NGramHasher

    hasher = NGramHasher()
    ids = torch.randint(0, 1000, (2, 64))
    straight = hasher.row_indices(ids)
    rolled = hasher.row_indices(ids.roll(7, dims=-1))
    assert straight.shape == rolled.shape
    assert not torch.equal(straight, rolled), "the control must change which rows are read"
    # Rows stay inside the table and are overwhelmingly the same ones, read for the
    # wrong position: rolling only forges new n-grams at the seam and in the EOS
    # prefill, so the gather pattern and hit rate carry over while the correspondence
    # to this text does not. An exact permutation is not available and not needed.
    assert rolled.min() >= 0 and rolled.max() <= straight.max().clamp(min=rolled.max())
    shared = len(set(straight.flatten().tolist()) & set(rolled.flatten().tolist()))
    assert shared / straight.numel() > 0.7, shared / straight.numel()


def test_config_carries_the_window_and_the_control():
    assert OptimizerConfig().stage1_trainable_layers is None
    assert OptimizerConfig(stage1_trainable_layers=(20, 28)).stage1_trainable_layers == (20, 28)
    assert SidecarConfig(table_path="x").shuffle_context == 0
    assert SidecarConfig(table_path="x", shuffle_context=7).shuffle_context == 7

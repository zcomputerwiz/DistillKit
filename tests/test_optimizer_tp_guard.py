"""Tensor parallelism may use the hybrid optimizer only where Muon receives nothing.

Stage 1 freezes the backbone, and every parameter it leaves trainable -- sidecar, gated
residual, distillation projections -- is auxiliary and routed to AdamW. So the Muon group
holds only frozen, sharded matrices and "hybrid" is exactly AdamW-on-auxiliary, which is
safe to shard. That equivalence is what the configuration guard is relaxed against, so
the first test here pins it: if Muon ever acquires a trainable parameter under
freeze_backbone, the relaxation is unsound and this fails.
"""

import pytest
import torch
from torch import nn

from distillkit.configuration import DistillationRunConfig
from distillkit.gated_residual import GatedResidual
from distillkit.optimizers import freeze_backbone_for_stage1, mixed_parameter_groups


class _Student(nn.Module):
    """The shapes that matter for routing: hidden matrices, an embedding, a tied head,
    and the auxiliary modules stage 1 actually trains."""

    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(32, 8)
        self.layers = nn.ModuleList([
            nn.ModuleDict({"q_proj": nn.Linear(8, 8, bias=False),
                           "mlp": nn.Linear(8, 16, bias=False),
                           "norm": nn.LayerNorm(8)})
            for _ in range(2)
        ])
        self.lm_head = nn.Linear(8, 32, bias=False)
        self.lm_head.weight = self.embed_tokens.weight
        self.sidecar = nn.Linear(8, 8, bias=False)
        self.gated_residual = GatedResidual(8, num_branches=2)
        self.distillation_projections = nn.ModuleList([nn.Linear(8, 8)])

    def get_output_embeddings(self):
        return self.lm_head


def _trainable_by_kind(model):
    totals = {}
    for group in mixed_parameter_groups(model):
        kind = group["optimizer_kind"]
        totals[kind] = totals.get(kind, 0) + sum(
            p.numel() for p in group["params"] if p.requires_grad
        )
    return totals


def test_stage1_leaves_muon_with_no_trainable_parameters():
    """The invariant the tensor-parallel relaxation depends on."""
    model = _Student()
    before = _trainable_by_kind(model)
    assert before.get("muon", 0) > 0, "fixture must give Muon something when unfrozen"

    freeze_backbone_for_stage1(model)
    after = _trainable_by_kind(model)
    assert after.get("muon", 0) == 0, (
        "freeze_backbone left %d trainable parameters in the Muon group; tensor "
        "parallelism with strategy=hybrid is no longer sound" % after.get("muon", 0)
    )
    assert after.get("adamw", 0) > 0, "stage 1 must still train the auxiliary modules"


def _config(**optimizer):
    return {
        "model": "some/model",
        "output_path": "out",
        "sequence_length": 8,
        "tensor_parallel": True,
        "dataset": {"seed": 42},   # cache_path supplies both splits
        "teacher": {"kind": "dataset", "cache_path": "cache", "anchor_layers": [1, 2]},
        "layer_mapping": [[1, 0], [2, 1]],
        "loss_functions": [{"function": "kl", "weight": 1.0, "sparse_chunk_length": 4}],
        "chunked_head": True,
        "optimizer": {"log_every_n_steps": 1, **optimizer},
    }


def test_tensor_parallel_accepts_adamw():
    DistillationRunConfig.model_validate(_config(strategy="adamw", freeze_backbone=False))


def _with_sidecar(config, table_path):
    """freeze_backbone requires a sidecar student -- stage 1 has nothing else to train."""
    config["sidecar"] = {"table_path": str(table_path), "enabled": True,
                         "resident": False, "prefault": False,
                         "layer_index": 1, "num_branches": 4}
    return config


def test_tensor_parallel_accepts_hybrid_when_the_backbone_stays_frozen(tmp_path):
    table = tmp_path / "table.gguf"
    table.write_bytes(b"")
    config = DistillationRunConfig.model_validate(_with_sidecar(
        _config(strategy="hybrid", freeze_backbone=True, unfreeze_at_step=None), table))
    assert config.optimizer.strategy == "hybrid"


def test_tensor_parallel_rejects_hybrid_that_unfreezes_even_with_a_sidecar(tmp_path):
    table = tmp_path / "table.gguf"
    table.write_bytes(b"")
    with pytest.raises(ValueError, match="unfreeze_at_step"):
        DistillationRunConfig.model_validate(_with_sidecar(
            _config(strategy="hybrid", freeze_backbone=True, unfreeze_at_step=10), table))


def test_tensor_parallel_rejects_hybrid_with_a_trainable_backbone():
    with pytest.raises(ValueError, match="freeze_backbone"):
        DistillationRunConfig.model_validate(
            _config(strategy="hybrid", freeze_backbone=False)
        )


def test_tensor_parallel_rejects_hybrid_that_unfreezes_mid_run():
    """The one path that re-enables gradients would hand Muon sharded matrices."""
    with pytest.raises(ValueError, match="unfreeze_at_step"):
        DistillationRunConfig.model_validate(
            _config(strategy="hybrid", freeze_backbone=True, unfreeze_at_step=10)
        )


def test_tensor_parallel_still_requires_an_offline_cache():
    config = _config(strategy="adamw", freeze_backbone=False)
    config["teacher"] = {"kind": "hf", "path": "some/teacher"}
    config["dataset"] = {"train_dataset": {"repo_id": "dummy/dummy"}}
    with pytest.raises(ValueError, match="offline cache"):
        DistillationRunConfig.model_validate(config)

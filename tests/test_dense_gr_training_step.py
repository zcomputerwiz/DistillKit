"""Exercise the actual shared step, not just its reduction coefficients."""
import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scratch" / "dense_gr"))

from training_state import (PlannedBatches, WindowBatches, read_training_state,
                            restore_training_state, save_training_state, take_step)
from training_step import check_optimizer, optimizer_step


def ce(model, hidden, ids):
    return F.cross_entropy(F.linear(hidden[:, :-1], model.lm_head.weight).flatten(0, 1),
                           ids[:, 1:].flatten())


class TokenBody(nn.Module):
    def __init__(self, hidden=8):
        super().__init__()
        self.embedding = nn.Embedding(32, hidden)
        self.proj = nn.Linear(hidden, hidden)

    def forward(self, input_ids, **kwargs):
        return SimpleNamespace(last_hidden_state=self.proj(self.embedding(input_ids)).tanh())


class TinyLM(nn.Module):
    def __init__(self, hidden=8):
        super().__init__()
        self.model = TokenBody(hidden)
        self.lm_head = nn.Linear(hidden, 32, bias=False)


def records(vocab=32):
    gen = torch.Generator().manual_seed(17)
    return [dict(input_ids=torch.randint(0, vocab, (rows, width), generator=gen),
                 topk_ids=torch.arange(4).expand(rows, width, 4).clone(),
                 topk_logprobs=torch.full((rows, width, 4), -3.0))
            for rows, width in ((2, 8), (1, 12), (3, 6))]


def split_rows(batch):
    return [{k: v[i:i + 1] for k, v in record.items()}
            for record in batch for i in range(record["input_ids"].shape[0])]


@pytest.mark.parametrize("teacher_weight", [0.0, 0.5])
def test_three_real_optimizer_steps_match_across_microbatch_boundaries(teacher_weight):
    torch.manual_seed(3)
    whole = TinyLM()
    split = copy.deepcopy(whole)
    a = torch.optim.AdamW(whole.parameters(), lr=1e-3)
    b = torch.optim.AdamW(split.parameters(), lr=1e-3)
    batch = records()
    for _ in range(3):
        ra = optimizer_step(whole, a, batch, ce=ce, teacher_weight=teacher_weight)
        rb = optimizer_step(split, b, split_rows(batch), ce=ce, teacher_weight=teacher_weight)
        assert ra == pytest.approx(rb, rel=2e-5, abs=2e-6)
        for x, y in zip(whole.parameters(), split.parameters()):
            torch.testing.assert_close(x.grad, y.grad, rtol=3e-5, atol=2e-7)
            torch.testing.assert_close(x, y, rtol=3e-5, atol=2e-7)


def test_three_steps_of_joint_sparse_objective_match():
    from test_csa2_routing import tiny_config
    from distillkit.models import Qwen35WidenedForCausalLM
    from indexer_kl import routing_layers

    torch.manual_seed(4)
    config = tiny_config(num_hidden_layers=2, layer_types=["full_attention"] * 2,
                         csa2_enabled=True, csa2_modes=["full", "full"],
                         csa2_router_bias=False, csa2_top_k=4, csa2_local_window=2,
                         csa2_index_source="latent")
    whole = Qwen35WidenedForCausalLM(config).train()
    split = copy.deepcopy(whole)
    a = torch.optim.AdamW(whole.parameters(), lr=1e-4)
    b = torch.optim.AdamW(split.parameters(), lr=1e-4)
    batch = records(64)
    for _ in range(3):
        ra = optimizer_step(whole, a, batch, ce=ce, teacher_weight=.5,
                            sparse_stage=routing_layers(whole))
        rb = optimizer_step(split, b, split_rows(batch), ce=ce, teacher_weight=.5,
                            sparse_stage=routing_layers(split))
        assert ra["indexer"] > 0
        assert ra == pytest.approx(rb, rel=3e-5, abs=3e-6)
        for (name, x), (_, y) in zip(whole.named_parameters(), split.named_parameters()):
            assert (x.grad is None) == (y.grad is None), name
            if x.grad is not None:
                torch.testing.assert_close(x.grad, y.grad, rtol=2e-3, atol=3e-6, msg=name)
            torch.testing.assert_close(x, y, rtol=2e-3, atol=3e-6, msg=name)


def test_stale_optimizer_refused_and_gradients_are_not_frozen():
    from distillkit.parallel.linear import ColumnParallelLinear

    source = nn.Linear(4, 4, bias=False)
    optimizer = torch.optim.AdamW(source.parameters())
    shard = ColumnParallelLinear(source, ["cpu", "cpu"], gather_output=True)
    shard(torch.ones(2, 4)).square().mean().backward()
    optimizer.zero_grad(set_to_none=True)
    assert all(p.grad is not None for p in shard.parameters())
    with pytest.raises(ValueError, match="stale"):
        check_optimizer(shard, optimizer)


def test_budget_does_not_overshoot_or_change_prefixes():
    class Teacher:
        def read_batch(self, ids, width):
            return {"input_ids": torch.ones(len(ids), width, dtype=torch.long), "doc_ids": ids}
    batches = PlannedBatches(Teacher(), [(["a", "b"], 8), (["c"], 12)], seed=7)
    result = take_step(batches, 3, 25)
    assert sum(r["input_ids"].numel() - len(r["input_ids"]) for r in result) == 25
    assert not take_step(batches, 2, 1)
    assert sorted((r["doc_ids"], r["input_ids"].shape[1]) for r in result) == [(["a", "b"], 8), (["c"], 12)]


def test_fresh_resume_matches_continuous_training(tmp_path):
    torch.manual_seed(4)
    model = TinyLM()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batches = WindowBatches(np.arange(64) % 32, 2, 8, 7, device="cpu")
    optimizer_step(model, optimizer, take_step(batches, 2, 100), ce=ce)
    checkpoint = tmp_path / "step1"
    save_training_state(checkpoint, model, optimizer, batches,
                        {"steps": 1, "targets": 28}, {"seed": 4})
    for _ in range(2):
        optimizer_step(model, optimizer, take_step(batches, 2, 100), ce=ce)
    resumed = TinyLM()
    opt = torch.optim.AdamW(resumed.parameters(), lr=1e-3)
    stream = WindowBatches(np.arange(64) % 32, 2, 8, 7, device="cpu")
    restore_training_state(read_training_state(checkpoint), resumed, opt, stream)
    for _ in range(2):
        optimizer_step(resumed, opt, take_step(stream, 2, 100), ce=ce)
    for x, y in zip(model.parameters(), resumed.parameters()):
        torch.testing.assert_close(x, y, rtol=0, atol=0)
    assert batches.state_dict() == stream.state_dict()
    with pytest.raises(FileExistsError):
        save_training_state(checkpoint, model, optimizer, batches, {}, {})
    (tmp_path / "partial").mkdir()
    with pytest.raises(ValueError, match="incomplete"):
        read_training_state(tmp_path / "partial")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="8-bit optimizer requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_resume_preserves_quantized_optimizer_state(tmp_path, dtype):
    import bitsandbytes as bnb

    torch.manual_seed(2)
    model = TinyLM(256).to(device="cuda", dtype=dtype)
    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=1e-4)
    batches = WindowBatches(np.arange(64) % 32, 2, 8, 8)
    optimizer_step(model, optimizer, take_step(batches, 2, 100), ce=ce)
    assert any(s.get("state1", torch.empty(0)).dtype == torch.uint8
               for s in optimizer.state.values())
    save_training_state(tmp_path / "state", model, optimizer, batches, {}, {})
    for _ in range(2):
        optimizer_step(model, optimizer, take_step(batches, 2, 100), ce=ce)
    restored = TinyLM(256).to(device="cuda", dtype=dtype)
    opt = bnb.optim.AdamW8bit(restored.parameters(), lr=1e-4)
    stream = WindowBatches(np.arange(64) % 32, 2, 8, 8)
    restore_training_state(read_training_state(tmp_path / "state"), restored, opt, stream)
    assert any(s.get("state1", torch.empty(0)).dtype == torch.uint8 for s in opt.state.values())
    for _ in range(2):
        optimizer_step(restored, opt, take_step(stream, 2, 100), ce=ce)
    for x, y in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(x, y, rtol=0, atol=0)


def test_resume_rejects_changed_plan_and_optimizer_order(tmp_path):
    torch.manual_seed(5)
    model = TinyLM()
    optimizer = torch.optim.AdamW(model.parameters())
    batches = WindowBatches(np.arange(64) % 32, 2, 8, 8, device="cpu")
    save_training_state(tmp_path / "state", model, optimizer, batches, {}, {})
    state = read_training_state(tmp_path / "state")
    changed = WindowBatches(np.arange(64) % 32, 1, 8, 8, device="cpu")
    with pytest.raises(ValueError, match="plan"):
        restore_training_state(state, model, optimizer, changed)
    reversed_optimizer = torch.optim.AdamW(list(model.parameters())[::-1])
    with pytest.raises(ValueError, match="order"):
        restore_training_state(state, model, reversed_optimizer, batches)


def test_benchmark_delegates_to_trainer_and_bounds_steps():
    import tp_bench
    import smoke_train

    assert tp_bench.train is smoke_train.main
    argv = tp_bench.trainer_arguments(["--cards", "2", "--steps", "3", "--output", "fresh.json"])
    assert "--tensor-parallel" in argv
    assert argv[argv.index("--max-steps") + 1] == "5"
    assert "--no-checkpoint" in argv
    assert tp_bench.trainer_arguments(["--help"]) == ["--help"]


def test_resume_after_export_ignores_only_export_metadata(tmp_path):
    from test_csa2_routing import tiny_config
    from distillkit.models import Qwen35WidenedForCausalLM

    config = tiny_config(num_hidden_layers=1, layer_types=["full_attention"])
    model = Qwen35WidenedForCausalLM(config)
    opt = torch.optim.AdamW(model.parameters())
    batches = WindowBatches(np.arange(64) % 32, 2, 8, 8, device="cpu")
    # HF export mutates architectures/dtype on the live config.
    model.save_pretrained(tmp_path / "export")
    save_training_state(tmp_path / "state", model, opt, batches, {}, {})
    fresh = Qwen35WidenedForCausalLM(tiny_config(num_hidden_layers=1, layer_types=["full_attention"]))
    fresh_opt = torch.optim.AdamW(fresh.parameters())
    restore_training_state(read_training_state(tmp_path / "state"), fresh, fresh_opt, batches)
    fresh.config.rms_norm_eps *= 2
    with pytest.raises(ValueError, match="configuration"):
        restore_training_state(read_training_state(tmp_path / "state"), fresh, fresh_opt, batches)


@pytest.mark.parametrize("devices", [["cpu", "cpu"], ["cuda:0", "cuda:1"]])
def test_three_steps_tensor_parallel_matches_unsharded(devices):
    if devices[0].startswith("cuda") and torch.cuda.device_count() < 2:
        pytest.skip("needs two CUDA devices")
    from test_csa2_routing import tiny_config
    from distillkit.models import Qwen35WidenedForCausalLM
    from distillkit.parallel.model import shard_model
    from distillkit.parallel.checkpoint import consolidated_state_dict
    from indexer_kl import routing_layers
    from training_step import backward_step

    torch.manual_seed(14)
    config = tiny_config(num_hidden_layers=2, layer_types=["linear_attention", "full_attention"],
                         linear_num_key_heads=2, linear_num_value_heads=2,
                         csa2_enabled=True, csa2_modes=["full"], csa2_router_bias=False,
                         csa2_top_k=4, csa2_local_window=2, csa2_index_source="latent")
    from distillkit.linear_attention_dispatch import install_device_aware_linear_attention
    install_device_aware_linear_attention()
    plain = Qwen35WidenedForCausalLM(config).to(devices[0]).train()
    sharded = copy.deepcopy(plain)
    shard_model(sharded, devices, shard_embeddings=False)
    a = torch.optim.AdamW(plain.parameters(), lr=1e-4, betas=(.9, .95))
    b = torch.optim.AdamW(sharded.parameters(), lr=1e-4, betas=(.9, .95))
    batch = [{k: v.to(devices[0]) for k, v in r.items()} for r in records(64)]
    options = dict(ce=ce, teacher_weight=.5)
    with torch.autograd.set_multithreading_enabled(False):
        for model in (plain, sharded):
            backward_step(model, batch, sparse_stage=routing_layers(model), **options)
            model.zero_grad(set_to_none=True)
    for _ in range(3):
        ra = optimizer_step(plain, a, batch, sparse_stage=routing_layers(plain), **options)
        rb = optimizer_step(sharded, b, batch, sparse_stage=routing_layers(sharded),
                            tensor_parallel=True, **options)
        assert ra == pytest.approx(rb, rel=3e-4, abs=3e-5)
        state = consolidated_state_dict(sharded)
        for name, expected in plain.state_dict().items():
            torch.testing.assert_close(state[name].cpu(), expected.cpu(),
                                       rtol=3e-3, atol=3e-5, msg=name)

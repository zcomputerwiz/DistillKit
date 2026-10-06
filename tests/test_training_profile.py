# Assisted-by: Codex
"""CPU checks for instrumentation transparency and fixed real-record replay."""
import gc
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scratch" / "dense_gr"))

from tp_bench import trainer_arguments
from training_profile import FrozenBatches, SavedTensorAccount, TrainingProfile, record_manifest


def test_aliases_charge_underlying_storage_once():
    account = SavedTensorAccount(nn.Linear(2, 2))
    x = torch.randn(8, 4)
    a, b = x[:2], x[2:4]
    ka = account.save(a, a, "test")
    kb = account.save(b, b, "test")
    assert account.report()["peak_live_unique_storage_bytes"]["cpu"]["other"] == x.untyped_storage().nbytes()
    assert sum(r["input_logical_bytes"] for r in account.report()["groups"]) == (a.numel() + b.numel()) * 4
    account.release(ka)
    assert account.report()["remaining_live_unique_storage_bytes"]["cpu"] == x.untyped_storage().nbytes()
    account.release(kb)
    assert account.report()["remaining_live_unique_storage_bytes"]["cpu"] == 0


def test_nested_checkpoint_and_custom_hook_keep_gradients_and_payloads():
    torch.manual_seed(17)
    model = nn.Linear(4, 4)
    x = torch.randn(3, 4, requires_grad=True)

    def run():
        y = checkpoint(lambda z: checkpoint(model, z, use_reentrant=False).sin(),
                       x, use_reentrant=False)
        return y.square().sum()

    reference = run()
    reference.backward()
    wanted = [p.grad.clone() for p in model.parameters()] + [x.grad.clone()]
    model.zero_grad(set_to_none=True)
    x.grad = None
    original = torch.autograd.graph.saved_tensors_hooks.__init__
    seen = []

    def pack(t):
        return "original-marker", t.detach().clone()

    def unpack(payload):
        assert payload[0] == "original-marker"
        seen.append(True)
        return payload[1]

    account = SavedTensorAccount(model)
    with account.capture():
        with torch.autograd.graph.saved_tensors_hooks(pack_hook=pack, unpack_hook=unpack):
            actual = run()
            actual.backward()
    assert torch.autograd.graph.saved_tensors_hooks.__init__ is original
    assert seen
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    for got, expected in zip([p.grad for p in model.parameters()] + [x.grad], wanted):
        torch.testing.assert_close(got, expected, rtol=0, atol=0)
    del actual
    gc.collect()
    assert all(n == 0 for n in account.report()["remaining_live_unique_storage_bytes"].values())
    assert any("_checkpoint_hook" in r["hook"] for r in account.report()["groups"])


def test_frozen_records_preserve_masks_and_manifest_detects_target_change():
    records = [{"input_ids": torch.tensor([[1, 2, 3]]), "weight": torch.tensor([[0., 8., 0.]]),
                "topk_ids": torch.tensor([[[1], [2], [3]]]),
                "context_kl": torch.tensor([[0.08, 0., 0.]]), "negative": torch.tensor([[False, True, False]])},
               {"input_ids": torch.tensor([[4, 5, 6, 7]]), "kl_only": True}]
    batches = FrozenBatches(records)
    assert batches.counts == [8, 3]
    assert batches.take(7) is None
    assert batches.take(8) is records[0]
    assert batches.take(3) is records[1]
    assert batches.take(8) is records[0]
    before = record_manifest(records, teacher_weight=0.5)
    assert before[0]["context_rows"] == before[0]["negative_rows"] == 1
    assert before[1]["original_supervised_rows"] == before[1]["kl_rows"] == 3
    assert before[1]["ce_rows"] == 0
    records[0]["topk_ids"][0, 0, 0] = 2
    assert record_manifest(records)[0]["sha256"] != before[0]["sha256"]


def test_manifest_objective_counts_follow_ce_only_kl_only_and_negative_masks():
    row = {"input_ids": torch.tensor([[1, 2, 3, 4]]),
           "weight": torch.tensor([[0., 1., 8., 99.]]),
           "context_kl": torch.tensor([[0.08, 0., 0., 0.08]]),
           "negative": torch.tensor([[False, True, False, True]])}
    ce_only = dict(row, ce_only=True)
    kl_only = dict(row, kl_only=True)
    mixed, ce, kl = record_manifest([row, ce_only, kl_only], teacher_weight=0.5)
    assert (mixed["original_supervised_rows"], mixed["targets"], mixed["ce_rows"]) == (2, 9, 2)
    assert (mixed["kl_rows"], mixed["context_rows"], mixed["negative_rows"],
            mixed["unlikelihood_objective_rows"], mixed["shared_head_rows"]) == (2, 1, 1, 0, 3)
    assert (ce["ce_rows"], ce["kl_rows"], ce["context_rows"],
            ce["negative_rows"], ce["shared_head_rows"]) == (2, 0, 0, 0, 2)
    assert (kl["ce_rows"], kl["kl_rows"], kl["unlikelihood_objective_rows"]) == (0, 2, 1)
    no_teacher = record_manifest([row], teacher_weight=0)[0]
    teacher_only = record_manifest([row], teacher_weight=1)[0]
    no_ul = record_manifest([kl_only], teacher_weight=0.5, unlikelihood_weight=0)[0]
    assert no_teacher["ce_rows"] == 2 and no_teacher["kl_rows"] == 0
    assert teacher_only["ce_rows"] == 0 and teacher_only["kl_rows"] == 2
    assert no_ul["unlikelihood_objective_rows"] == 0
    assert no_teacher["sha256"] == teacher_only["sha256"] == mixed["sha256"]
    assert no_ul["sha256"] == kl["sha256"]


def test_profile_cli_is_bounded_and_forwards_the_complete_recipe():
    args = trainer_arguments(["--cards", "2", "--profile-dir", "new-profile", "--fixed-shape", "1", "32768",
                              "--grad-streams",
                              "--streaming-head-loss", "--checkpoint-selection-cache",
                              "--shared-head-loss", "--context-kl", "0.01", "--accumulate", "2",
                              "--teacher-cache", "real-cache", "--output", "new.json"])
    assert args[args.index("--max-steps") + 1] == "5"
    assert args[args.index("--benchmark-warmup-steps") + 1] == "2"
    assert "--benchmark-fixed-records" in args and "--tensor-parallel" in args
    assert args[args.index("--benchmark-shape") + 1:args.index("--benchmark-shape") + 3] == ["1", "32768"]
    assert args[args.index("--context-kl") + 1] == "0.01" and "--shared-head-loss" in args
    assert "--benchmark-grad-streams" in args
    assert "--streaming-head-loss" in args and "--checkpoint-selection-cache" in args
    with pytest.raises(SystemExit):
        trainer_arguments(["--warmup-steps", "1", "--output", "new.json"])
    with pytest.raises(SystemExit):
        trainer_arguments(["--grad-streams", "--output", "new.json"])


def test_post_accumulation_metadata_preserves_gradients_and_removes_hooks(tmp_path, monkeypatch):
    torch.manual_seed(22)
    model = nn.Linear(4, 3)
    x = torch.randn(2, 4)
    model(x).square().sum().backward()
    wanted = {name: p.grad.clone() for name, p in model.named_parameters()}
    model.zero_grad(set_to_none=True)

    def prohibited(*args, **kwargs):
        raise AssertionError("metadata must not request gradient edges or query CUDA for CPU leaves")

    monkeypatch.setattr(torch.autograd.graph, "get_gradient_edge", prohibited)
    monkeypatch.setattr(torch.cuda, "current_stream", prohibited)
    diagnostic = TrainingProfile(tmp_path / "trace", model, grad_streams=True)
    with diagnostic.grad_metadata() as events:
        model(x).square().sum().backward()
    assert {e["parameter"] for e in events} == set(wanted)
    assert all(e["device"] == "cpu" and e["cuda_stream"] is None
               and isinstance(e["thread_id"], int) and e["time_ns"] > 0 for e in events)
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter.grad, wanted[name], rtol=0, atol=0)
        assert not parameter._post_accumulate_grad_hooks
    model.zero_grad(set_to_none=True)
    model(x).square().sum().backward()
    assert len(events) == len(wanted)


def test_completed_recipe_replay_keeps_objective_and_explicit_overrides(tmp_path):
    recipe = tmp_path / "recipe.json"
    recipe.write_text(json.dumps({"run_args": {"init_from": "base", "inherit": True,
                      "teacher_cache": ["real-a", "real-b"], "shared_head_loss": True,
                      "head_chunk": 512, "context_kl": 0.01, "context_every": 8,
                      "checkpoint_layers": True, "accumulate": 2, "tensor_parallel": True,
                      "max_steps": 100, "output": "old.json", "benchmark_shape": None}}))
    args = trainer_arguments(["--recipe-from", str(recipe), "--cards", "2", "--fixed-records",
                              "--head-chunk", "256", "--output", "new.json"])
    assert "--shared-head-loss" in args and "--checkpoint-layers" in args
    assert args.count("--head-chunk") == 1 and args[args.index("--head-chunk") + 1] == "256"
    assert args[args.index("--context-kl") + 1] == "0.01"
    assert "old.json" not in args and args[args.index("--max-steps") + 1] == "5"


def test_cpu_timeline_annotations_preserve_full_shared_objective(tmp_path):
    from training_step import optimizer_step

    class Body(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(16, 4)
            self.layers = nn.ModuleList([nn.Linear(4, 4)])

        def forward(self, input_ids, **kwargs):
            return SimpleNamespace(last_hidden_state=self.layers[0](self.embed(input_ids)).tanh())

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Body()
            self.lm_head = nn.Linear(4, 16, bias=False)

    torch.manual_seed(25)
    model = Model()
    wanted = Model()
    wanted.load_state_dict(model.state_dict())
    records = [{"input_ids": torch.tensor([[1, 2, 3, 4]]),
                "topk_ids": torch.tensor([[[1, 2]]]).expand(1, 4, 2),
                "topk_logprobs": torch.full((1, 4, 2), -3.),
                "weight": torch.tensor([[0., 1., 8., 0.]]),
                "context_kl": torch.tensor([[0.08, 0., 0., 0.]]),
                "negative": torch.tensor([[False, True, False, False]]), "kl_only": True}]
    reference = optimizer_step(wanted, torch.optim.AdamW(wanted.parameters()), records,
                               teacher_weight=0.5, shared_head=True, head_chunk=2)
    account = SavedTensorAccount(model)
    diagnostic = TrainingProfile(tmp_path / "trace", model)
    with diagnostic.annotations(), account.capture(), torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
        actual = optimizer_step(model, torch.optim.AdamW(model.parameters()), records,
                                teacher_weight=0.5, shared_head=True, head_chunk=2)
    assert actual == reference
    for got, expected in zip(model.parameters(), wanted.parameters()):
        torch.testing.assert_close(got, expected, rtol=0, atol=0)
    assert {"train/body", "train/head", "train/backward", "train/layer/00"} <= {e.key for e in prof.key_averages()}
    path = tmp_path / "trace" / "cpu.trace.json"
    prof.export_chrome_trace(str(path))
    assert json.loads(path.read_text())["traceEvents"]

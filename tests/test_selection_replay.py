# Assisted-by: Codex
import contextlib
import copy
import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scratch" / "dense_gr"))
from selection_replay import SelectionReplayCache, frame_scope


def test_frames_keep_independent_selections_and_reject_wrong_signature():
    cache = SelectionReplayCache(None)
    owner = object()
    a, b = {}, {}
    x = torch.tensor([[2.]])
    calls = []

    def select(value):
        calls.append(value.item())
        return value.long(), value > 0

    with frame_scope(a, False):
        first = cache.select(owner, select, x)
    with frame_scope(b, False):
        second = cache.select(owner, select, -x)
    with frame_scope(a, True):
        assert cache.select(owner, select, x) is first
        with pytest.raises(RuntimeError, match="matching forward"):
            cache.select(owner, select, x.expand(2, 1))
    with frame_scope(b, True):
        assert cache.select(owner, select, x) is second
    assert calls == [2., -2.]
    assert cache.report()["computed"] == cache.report()["reused"] == 2


@pytest.mark.parametrize("streaming", [False, True])
def test_actual_hybrid_shared_objective_and_all_gradients(monkeypatch, streaming):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from distillkit.models import Qwen35WidenedForCausalLM
    from training_step import backward_step

    config = json.loads((ROOT / "scratch/llama_hybrid_port/synthetic_checkpoint/config.json").read_text())
    config["vocab_size"] = 64
    torch.manual_seed(173)
    model = Qwen35WidenedForCausalLM(Qwen3_5TextConfig(**config)).train()
    model.model.gradient_checkpointing = True
    wanted = copy.deepcopy(model)
    records = []
    for i in range(2):
        ids = torch.randint(0, 64, (1, 24))
        weight = torch.ones(1, 24)
        weight[:, :4] = 0
        extra = torch.zeros_like(weight)
        extra[:, :4:2] = 0.08
        picks = torch.stack([torch.randperm(64)[:8] for _ in range(24)])[None]
        records.append(dict(input_ids=ids, weight=weight, context_kl=extra,
                            topk_ids=picks, topk_logprobs=torch.full((1, 24, 8), -3.),
                            negative=torch.arange(24)[None] == 6, kl_only=bool(i)))
    expected = backward_step(wanted, records, teacher_weight=0.5, shared_head=True, head_chunk=8)
    original = model.model.layers[3].self_attn._select_positions
    diagnostic = SelectionReplayCache(model)
    with diagnostic:
        actual = backward_step(model, records, teacher_weight=0.5, shared_head=True, head_chunk=8,
                               streaming_head=streaming)
    assert model.model.layers[3].self_attn._select_positions == original
    assert diagnostic.computed == diagnostic.reused == 2
    assert actual == pytest.approx(expected, rel=1e-6, abs=1e-7)
    for (name, parameter), (expected_name, reference) in zip(model.named_parameters(), wanted.named_parameters()):
        assert name == expected_name
        if reference.grad is None:
            assert parameter.grad is None, name
        else:
            torch.testing.assert_close(parameter.grad, reference.grad, rtol=1e-6, atol=1e-7, msg=name)


def test_scope_restores_after_exception():
    cache = SelectionReplayCache(None)
    with pytest.raises(RuntimeError, match="test failure"):
        with frame_scope({}, True):
            raise RuntimeError("test failure")
    x = torch.tensor([1.])
    assert cache.select(object(), lambda value: value, x) is x

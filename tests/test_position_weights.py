"""Per-position loss weights: answer spans, assistant-only turns and padding as one concept.

The frontier QA documents are ~30K tokens of source code with a few hundred answer tokens;
scored uniformly the answers are under 1% of their loss (Codex review of long round 2). A
position's weight scales its cross entropy and KL alike, a step is the weighted mean over
everything in it, and the planner and the token budget count the same weights.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scratch" / "dense_gr"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_merged_cache import write  # noqa: E402
from teacher_kl import CachedTeacher  # noqa: E402


def teacher(tmp_path, spans, weight=4.0, **kwargs):
    tokens = {"a": list(range(1, 13)), "b": list(range(1, 8))}
    path = write(tmp_path / "cache", list(tokens), tokens=tokens)
    return CachedTeacher(path, device="cpu", answer_spans=spans, answer_weight=weight, **kwargs)


def test_answer_spans_weigh_the_answer_and_its_close(tmp_path):
    t = teacher(tmp_path, {"a": [[5, 8]]})
    # Answer body is tokens 5..7, its close token 8: predicted from positions 4..7.
    w = t.position_weight("a", np.arange(1, 13), 12)
    assert w.tolist() == [1, 1, 1, 1, 4, 4, 4, 4, 1, 1, 1, 0]
    assert t.doc_weight("a", 12) == float(w.sum()) and t.doc_weight("b", 7) == 6.0


def test_planner_and_batch_count_the_same_weight(tmp_path):
    t = teacher(tmp_path, {"a": [[5, 8]]})
    t.pad_blocks = True
    groups = t._groups(4, block=4)
    total = 0.0
    for group, width in groups:
        batch = t.read_batch(group, width)
        charged = float(batch["weight"][:, :-1].sum()) if "weight" in batch else \
            batch["input_ids"].numel() - batch["input_ids"].shape[0]
        assert charged == pytest.approx(t.group_weight(group, width))
        total += charged
    assert t.planned_tokens(4, block=4) == pytest.approx(total)


def test_spans_set_the_answer_start(tmp_path):
    t = teacher(tmp_path, {"a": [[5, 8]]}, answer_marker=[99], min_answer_tokens=1)
    assert t.answer_start["a"] == 5


def test_weighted_step_is_the_weighted_mean():
    from test_assistant_mask import masked_ce
    from test_dense_gr_training_step import TinyLM
    from training_step import backward_step

    torch.manual_seed(0)
    model = TinyLM()
    ids = torch.randint(0, 32, (2, 10))
    weight = torch.zeros(2, 10)
    weight[0, :9] = 1.0
    weight[0, 3:5] = 5.0
    weight[1, 2:7] = 0.5
    record = dict(input_ids=ids, weight=weight, topk_ids=torch.arange(4).expand(2, 10, 4).clone(),
                  topk_logprobs=torch.full((2, 10, 4), -1.5))
    result = backward_step(model, [record], ce=masked_ce, teacher_weight=0.5)
    assert result["targets"] == pytest.approx(float(weight[:, :-1].sum()))
    with torch.no_grad():
        expected = masked_ce(model, model.model(ids).last_hidden_state, ids, weight)
    assert result["loss"] == pytest.approx(float(expected), abs=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CCE runs on CUDA")
def test_causal_ce_weights_match_plain_torch():
    import smoke_train  # noqa: F401  (the triton metadata shim CCE's import needs)
    from test_assistant_mask import masked_ce
    from training_step import causal_ce

    torch.manual_seed(0)
    head = torch.nn.Linear(64, 512, bias=False).cuda()
    model = torch.nn.Module()
    model.lm_head = head
    hidden = torch.randn(2, 33, 64, device="cuda")
    ids = torch.randint(0, 512, (2, 33), device="cuda")
    weight = torch.rand(2, 33, device="cuda")
    weight[weight < 0.3] = 0.0
    with torch.no_grad():
        got = float(causal_ce(model, hidden, ids, weight=weight))
        ref = float(masked_ce(model, hidden, ids, weight))
        assert got == pytest.approx(ref, rel=2e-3)
        plain = float(causal_ce(model, hidden, ids))
        assert plain == pytest.approx(float(masked_ce(model, hidden, ids)), rel=2e-3)

"""Agent traces score only the assistant's turns: span detection and the training step."""
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scratch" / "dense_gr"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from teacher_kl import assistant_tokens  # noqa: E402
from test_dense_gr_training_step import TinyLM  # noqa: E402


def test_assistant_tokens_cover_turn_content_through_its_close():
    #         0  1  2  3  4  5  6  7  8  9 10 11 12
    ids = [1, 2, 7, 8, 3, 4, 9, 5, 7, 8, 6, 9, 2]
    inside = assistant_tokens(ids, [7, 8], 9)
    assert np.nonzero(inside)[0].tolist() == [4, 5, 6, 10, 11]
    # An unclosed final turn runs to the end.
    assert np.nonzero(assistant_tokens([7, 8, 3, 3], [7, 8], 9))[0].tolist() == [2, 3]


def masked_ce(model, hidden, ids, supervised=None):
    logits = F.linear(hidden[:, :-1], model.lm_head.weight)
    targets = ids[:, 1:].clone()
    if supervised is not None:
        targets[~supervised[:, :-1]] = -100
    return F.cross_entropy(logits.flatten(0, 1), targets.flatten(), ignore_index=-100)


def test_step_scores_and_counts_only_supervised_positions():
    from training_step import backward_step

    torch.manual_seed(0)
    model = TinyLM()
    ids = torch.randint(0, 32, (2, 10))
    supervised = torch.zeros(2, 10, dtype=torch.bool)
    supervised[0, 3:6] = True
    supervised[1, 6:9] = True
    record = dict(input_ids=ids, supervised=supervised, topk_ids=torch.arange(4).expand(2, 10, 4).clone(),
                  topk_logprobs=torch.full((2, 10, 4), -1.5))
    result = backward_step(model, [record], ce=masked_ce, teacher_weight=0.5)
    assert result["targets"] == 6
    with torch.no_grad():
        expected = masked_ce(model, model.model(ids).last_hidden_state, ids, supervised)
    assert abs(result["loss"] - float(expected)) < 1e-6
    # All-true supervision is the unmasked step.
    model.zero_grad()
    full = backward_step(model, [dict(record, supervised=torch.ones(2, 10, dtype=torch.bool))],
                         ce=masked_ce, teacher_weight=0.5)
    model.zero_grad()
    plain = backward_step(model, [{k: v for k, v in record.items() if k != "supervised"}],
                          ce=masked_ce, teacher_weight=0.5)
    assert full["targets"] == plain["targets"] == 18
    assert abs(full["loss"] - plain["loss"]) < 1e-6 and abs(full["teacher_kl"] - plain["teacher_kl"]) < 1e-6

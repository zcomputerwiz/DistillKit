"""KL alone on the context of assistant-only documents (--context-kl): every k-th context
position at k times the weight, projected for KL only, the same through both head paths."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scratch" / "dense_gr"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from shared_head import head_losses  # noqa: E402

# Marker 3 opens the assistant's turn, 7 closes it; ids below TinyLM's 32.
TOKENS = {"a": [1, 2, 3, 4, 5, 6, 8, 9, 10, 7, 11], "b": [3, 12, 13, 14, 15, 7, 16, 17, 3, 18, 19, 20, 21, 7],
          "c": [2, 3, 22, 23, 24, 25, 26, 7]}


def cached(tmp_path, context=(0.1, 2)):
    from test_merged_cache import write
    from teacher_kl import CachedTeacher

    path = write(tmp_path / "cache", list(TOKENS), tokens=TOKENS)
    teacher = CachedTeacher(path, device="cpu", assistant_only=[path], answer_marker=[3], turn_close=7)
    teacher.pad_blocks = True
    teacher.context_kl = context
    return teacher


def test_context_positions_are_sampled_outside_the_turns(tmp_path):
    teacher = cached(tmp_path)
    groups = teacher._groups(4, block=8)
    for group, width in groups:
        batch = teacher.read_batch(group, width)
        weight, extra = batch["weight"].numpy(), batch["context_kl"].numpy()
        for row, doc in enumerate(group):
            real = len(TOKENS[doc])
            picked = extra[row] > 0
            assert (extra[row][picked] == pytest.approx(0.2))  # 0.1 at every 2nd position
            assert not (picked & (weight[row] > 0)).any()  # never on a scored position
            assert not picked[real - 1:].any()  # nor the last real position or padding
            context = (weight[row][:real - 1] == 0).sum()
            assert abs(int(picked.sum()) - context / 2) <= 1  # half of the context


def test_off_by_default_and_outside_the_digest(tmp_path):
    teacher = cached(tmp_path, context=None)
    groups = teacher._groups(4, block=8)
    plain = teacher.weight_identity()
    assert all("context_kl" not in teacher.read_batch(g, w) for g, w in groups)
    teacher.context_kl = (0.1, 2)
    assert teacher.weight_identity() != plain


def test_kl_beyond_adds_the_kl_only_rows_and_nothing_else():
    torch.manual_seed(0)
    rows, length, d, vocab, k = 2, 9, 8, 16, 4
    hidden, head = torch.randn(rows, length, d), torch.randn(vocab, d)
    ids = torch.randint(0, vocab, (rows, length))
    topk = torch.randint(0, vocab, (rows, length, k))
    logp = torch.log_softmax(torch.randn(rows, length, k), -1) - 0.5
    weight = torch.zeros(rows, length)
    weight[:, 4:8] = 1.0
    extra = torch.zeros(rows, length)
    extra[:, 0:4:2] = 0.3
    both = head_losses(hidden, head, ids, weight=weight, topk_ids=topk, topk_logprobs=logp,
                       kl_weight=weight + extra, kl_beyond=True, chunk=3)
    scored = head_losses(hidden, head, ids, weight=weight, topk_ids=topk, topk_logprobs=logp,
                         kl_weight=weight, chunk=3)
    alone = head_losses(hidden, head, ids, weight=extra, topk_ids=topk, topk_logprobs=logp,
                        kl_weight=extra, chunk=3)
    torch.testing.assert_close(both["nll"], scored["nll"])
    torch.testing.assert_close(both["weight"], scored["weight"])
    torch.testing.assert_close(both["kl"], scored["kl"] + alone["kl"])
    with pytest.raises(ValueError):
        head_losses(hidden, head, ids, weight=weight, topk_ids=topk, topk_logprobs=logp,
                    kl_weight=weight + extra, chunk=3)


def test_both_head_paths_agree_with_context(tmp_path):
    from test_dense_gr_training_step import TinyLM
    from test_shared_head import weighted_ce
    from training_step import backward_step

    teacher = cached(tmp_path, context=(0.25, 2))
    batch = [teacher.read_batch(g, w) for g, w in teacher._groups(4, block=8)]
    assert any("context_kl" in b for b in batch)
    torch.manual_seed(3)
    model = TinyLM()
    old = backward_step(model, batch, teacher_weight=0.5, ce=weighted_ce, kl_chunk=4)
    old_grads = {n: p.grad.clone() for n, p in model.named_parameters()}
    model.zero_grad()
    new = backward_step(model, batch, teacher_weight=0.5, ce=weighted_ce, kl_chunk=4, shared_head=True, head_chunk=3)
    for name in ("loss", "teacher_kl", "objective", "targets"):
        assert new[name] == pytest.approx(old[name], rel=1e-5, abs=1e-7), name
    for name, grad in old_grads.items():
        torch.testing.assert_close(dict(model.named_parameters())[name].grad, grad, msg=name)
    # And the context does move the KL: without it the step differs.
    model.zero_grad()
    plain = backward_step(model, [{k: v for k, v in b.items() if k != "context_kl"} for b in batch],
                          teacher_weight=0.5, ce=weighted_ce, kl_chunk=4, shared_head=True, head_chunk=3)
    assert plain["teacher_kl"] < new["teacher_kl"] and plain["loss"] == pytest.approx(new["loss"])

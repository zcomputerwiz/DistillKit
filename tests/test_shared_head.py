"""One vocabulary pass for every head loss (shared_head.py) against the separate ones.

On the CPU both paths compute in fp32, so the shared path must reproduce the old
cross entropy, grouped-tail KL and unlikelihood -- values and gradients -- for weighted,
unweighted, KL-only-with-loops and CE-only records, through the real backward_step.
"""
import sys
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scratch" / "dense_gr"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from shared_head import head_losses  # noqa: E402
from teacher_kl import grouped_tail_kl, unlikelihood_loss  # noqa: E402
from test_dense_gr_training_step import TinyLM  # noqa: E402
from training_step import backward_step  # noqa: E402

VOCAB, K = 32, 4


def teacher_rows(rows, width, gen):
    """Top-k ids and log-probs of a random teacher, so the tail carries real mass."""
    logits = torch.randn(rows, width, VOCAB, generator=gen) * 2
    logprobs = logits.log_softmax(-1)
    values, ids = logprobs.topk(K, dim=-1)
    return ids, values


def weighted_ce(model, hidden, ids, weight=None):
    nll = F.cross_entropy(F.linear(hidden[:, :-1], model.lm_head.weight).transpose(1, 2), ids[:, 1:],
                          reduction="none")
    w = torch.ones_like(nll) if weight is None else weight[:, :-1].float()
    return (nll * w).sum() / w.sum()


@pytest.mark.parametrize("chunk", [1, 4, 7, 512])
def test_head_losses_match_the_separate_losses(chunk):
    gen = torch.Generator().manual_seed(3)
    rows, width, d = 3, 10, 8
    head = torch.nn.Linear(d, VOCAB, bias=False)
    hidden = torch.randn(rows, width, d, generator=gen, requires_grad=True)
    ids = torch.randint(0, VOCAB, (rows, width), generator=gen)
    topk_ids, topk_values = teacher_rows(rows, width, gen)
    weight = torch.ones(rows, width)
    weight[0, :4] = 0          # a prompt
    weight[1, 5:8] = 8.0       # an answer span
    weight[:, -1] = 0
    negative = torch.zeros(rows, width, dtype=torch.bool)
    negative[2, 3:7] = True
    kl_mask = weight * ~negative

    got = head_losses(hidden, head.weight, ids, weight=weight, topk_ids=topk_ids,
                      topk_logprobs=topk_values, kl_weight=kl_mask, negative=negative, chunk=chunk)
    want_nll = (F.cross_entropy(head(hidden[:, :-1]).transpose(1, 2), ids[:, 1:], reduction="none")
                * weight[:, :-1]).sum()
    want_kl = grouped_tail_kl(hidden, head, topk_ids, topk_values, kl_mask, chunk_length=4)
    want_ul = unlikelihood_loss(hidden, head, ids, negative, chunk=2, weight=weight)
    torch.testing.assert_close(got["nll"], want_nll)
    torch.testing.assert_close(got["weight"], weight[:, :-1].sum())
    torch.testing.assert_close(got["kl"], want_kl)
    torch.testing.assert_close(got["unlikelihood"], want_ul)

    (got["nll"] + got["kl"] + 0.5 * got["unlikelihood"]).backward()
    shared = hidden.grad.clone(), head.weight.grad.clone()
    hidden.grad, head.weight.grad = None, None
    (want_nll + want_kl + 0.5 * want_ul).backward()
    torch.testing.assert_close(shared[0], hidden.grad)
    torch.testing.assert_close(shared[1], head.weight.grad)


def test_kl_weight_outside_weight_is_refused():
    head = torch.nn.Linear(4, VOCAB, bias=False)
    hidden = torch.randn(1, 5, 4)
    ids = torch.randint(0, VOCAB, (1, 5))
    topk_ids, topk_values = teacher_rows(1, 5, torch.Generator().manual_seed(0))
    weight = torch.tensor([[0.0, 1, 1, 1, 0]])
    with pytest.raises(ValueError):
        head_losses(hidden, head.weight, ids, weight=weight, topk_ids=topk_ids, topk_logprobs=topk_values,
                    kl_weight=torch.ones(1, 5))


def records():
    gen = torch.Generator().manual_seed(11)
    out = []
    for rows, width, kind in ((2, 8, "plain"), (1, 12, "weighted"), (2, 9, "loop"), (1, 7, "ce_only")):
        ids = torch.randint(0, VOCAB, (rows, width), generator=gen)
        topk_ids, topk_values = teacher_rows(rows, width, gen)
        record = dict(input_ids=ids, topk_ids=topk_ids, topk_logprobs=topk_values)
        if kind == "weighted":
            weight = torch.ones(rows, width)
            weight[:, :3] = 0
            weight[:, 6:9] = 8.0
            weight[:, -1] = 0
            record["weight"] = weight
        if kind == "loop":
            negative = torch.zeros(rows, width, dtype=torch.bool)
            negative[:, 4:8] = True
            record.update(kl_only=True, negative=negative)
        if kind == "ce_only":
            record["ce_only"] = True
        out.append(record)
    return out


@pytest.mark.parametrize("head_chunk", [1, 3, 7, 512])
@pytest.mark.parametrize("teacher_weight", [0.0, 0.5])
def test_backward_step_shared_matches_separate(teacher_weight, head_chunk):
    torch.manual_seed(5)
    model = TinyLM()
    batch = records()
    old = backward_step(model, batch, teacher_weight=teacher_weight, ce=weighted_ce, kl_chunk=4)
    old_grads = {n: p.grad.clone() for n, p in model.named_parameters()}
    model.zero_grad()
    new = backward_step(model, batch, teacher_weight=teacher_weight, ce=weighted_ce, kl_chunk=4,
                        shared_head=True, head_chunk=head_chunk)
    for name in ("loss", "teacher_kl", "unlikelihood", "objective", "targets"):
        assert new[name] == pytest.approx(old[name], rel=1e-5, abs=1e-7), name
    for name, grad in old_grads.items():
        torch.testing.assert_close(dict(model.named_parameters())[name].grad, grad, msg=name)


def test_cached_batches_through_both_paths(tmp_path):
    """Real CachedTeacher.read_batch output -- assistant-only weights, answer spans at 8,
    padding to the block multiple -- plus loop negatives, through backward_step both ways,
    with chunks that split rows across boundaries."""
    from test_merged_cache import write
    from teacher_kl import CachedTeacher

    # Marker 3 opens the assistant's turn, 7 closes it; ids below TinyLM's 32.
    tokens = {"a": [1, 2, 3, 4, 5, 6, 8, 9, 10, 7, 11], "b": [3, 12, 13, 14, 15, 7, 16, 17, 3, 18, 19, 20, 21, 7],
              "c": [2, 3, 22, 23, 24, 25, 26, 7]}
    path = write(tmp_path / "cache", list(tokens), tokens=tokens)
    teacher = CachedTeacher(path, device="cpu", assistant_only=[path], answer_marker=[3], turn_close=7,
                            answer_spans={"b": [[1, 5]]}, answer_weight=8.0)
    teacher.pad_blocks = True
    groups = teacher._groups(4, block=8)
    batch = [teacher.read_batch(group, width) for group, width in groups]
    loop = dict(batch[0])
    negative = torch.zeros_like(loop["input_ids"], dtype=torch.bool)
    negative[:, 4:7] = True
    loop.update(kl_only=True, negative=negative)
    batch.append(loop)
    assert any("weight" in b and (b["weight"] == 0).any() for b in batch)  # padding and prompts
    torch.manual_seed(2)
    model = TinyLM()
    old = backward_step(model, batch, teacher_weight=0.5, ce=weighted_ce, kl_chunk=4)
    old_grads = {n: p.grad.clone() for n, p in model.named_parameters()}
    model.zero_grad()
    new = backward_step(model, batch, teacher_weight=0.5, ce=weighted_ce, kl_chunk=4, shared_head=True, head_chunk=3)
    for name in ("loss", "teacher_kl", "unlikelihood", "objective", "targets"):
        assert new[name] == pytest.approx(old[name], rel=1e-5, abs=1e-7), name
    for name, grad in old_grads.items():
        torch.testing.assert_close(dict(model.named_parameters())[name].grad, grad, msg=name)

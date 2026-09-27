"""Looping rollouts: repeated spans found exactly, and trained away rather than toward."""
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scratch" / "dense_gr"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from teacher_kl import repeated_tokens, unlikelihood_loss  # noqa: E402
from test_dense_gr_training_step import TinyLM, ce, records  # noqa: E402


def test_repeated_tokens_marks_every_token_of_a_repeat_after_start():
    loop = [5, 6, 7]
    ids = np.array([1, 2] + loop * 4 + [9])
    marked = repeated_tokens(ids, start=2, n=3, count=2)
    # The first pass of the loop is new; from the second on, every token copies.
    assert not marked[:5].any()
    assert marked[5:14].all()
    assert not marked[14]
    # Nothing before `start` counts, even if it repeats there.
    assert not repeated_tokens(np.array(loop * 3), start=9, n=3, count=2).any()
    # The default spares a single legitimate repeat and catches the third pass.
    once = np.array(list(range(20)) + [99] + list(range(20)))
    assert not repeated_tokens(once, start=0).any()
    thrice = np.array(list(range(20)) * 3)
    assert repeated_tokens(thrice, start=0)[40:].all()


def test_loops_are_looked_for_in_the_thought_or_else_the_whole_reply():
    from teacher_kl import last_response, loop_tokens

    close, loop = 99, list(range(20))
    thought = np.array([7] + loop * 3 + [close] + loop * 3)
    marked = loop_tokens(thought, 1, close)
    assert marked[41:61].all() and not marked[61:].any()  # the answer may restate
    reply = np.array([7, 8, close, 5] + loop * 3)  # empty think block: non-thinking
    assert loop_tokens(reply, 0, close)[44:].all()
    assert last_response([1, 2, 9, 1, 2, 9], [1, 2]) == 5


def test_unlikelihood_matches_brute_force():
    torch.manual_seed(0)
    head = torch.nn.Linear(8, 32, bias=False)
    hidden = torch.randn(2, 6, 8)
    ids = torch.randint(0, 32, (2, 6))
    negative = torch.zeros(2, 6, dtype=torch.bool)
    negative[0, 1] = negative[1, 3] = negative[1, 4] = True
    want = 0.0
    for r, t in negative.nonzero().tolist():
        p = F.softmax(head(hidden[r, t]), -1)[ids[r, t + 1]]
        want = want - torch.log1p(-p)
    torch.testing.assert_close(unlikelihood_loss(hidden, head, ids, negative, chunk=2), want)


def test_looping_batch_takes_no_kl_on_repeats_and_pushes_them_down():
    from training_step import backward_step

    torch.manual_seed(5)
    model = TinyLM()
    batch = records()[:1]
    negative = torch.zeros_like(batch[0]["input_ids"], dtype=torch.bool)
    negative[:, 2:5] = True
    looping = [dict(batch[0], kl_only=True, negative=negative)]
    result = backward_step(model, looping, ce=ce, unlikelihood_weight=0.0)
    kl_masked = {n: p.grad.clone() for n, p in model.named_parameters()}
    assert result["unlikelihood"] > 0
    model.zero_grad()
    backward_step(model, [dict(batch[0], kl_only=True)], ce=ce)
    full = {n: p.grad.clone() for n, p in model.named_parameters()}
    assert any(not torch.allclose(kl_masked[n], full[n]) for n in full)

    # A step on the objective lowers the repeated tokens' probability.
    def repeat_probability():
        with torch.no_grad():
            hidden = model.model(batch[0]["input_ids"]).last_hidden_state
            p = F.softmax(model.lm_head(hidden[:, :-1]), -1).gather(
                -1, batch[0]["input_ids"][:, 1:, None]).squeeze(-1)
            return float(p[negative[:, :-1]].sum())

    before = repeat_probability()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    for _ in range(5):
        model.zero_grad()
        backward_step(model, looping, ce=ce, unlikelihood_weight=5.0)
        optimizer.step()
    assert repeat_probability() < before

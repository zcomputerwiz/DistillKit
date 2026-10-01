"""FTPO: the loop-start detector, row filters, and the clipped, tethered loss."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scratch" / "dense_gr"))

from ftpo_rows import alternatives, first_readable, flatten  # noqa: E402
from teacher_kl import loop_start  # noqa: E402
from training_step import ftpo_loss  # noqa: E402

CLOSE = 99


def test_loop_start_is_first_token_of_the_copy():
    prompt = list(range(50, 60))
    lead = [1, 2, 3]  # thought text before the loop
    cycle = list(range(10, 30))  # period 20 > n
    ids = prompt + lead + cycle * 4
    # The second copy of the cycle begins right after the first.
    assert loop_start(ids, len(prompt), CLOSE) == len(prompt) + len(lead) + len(cycle)


def test_loop_start_walks_back_to_where_copying_began():
    prompt = [50, 51]
    cycle = [7, 8, 9]  # period 3 < n: the 16-gram's own occurrences overlap
    ids = prompt + [1, 2] + cycle * 12
    assert loop_start(ids, len(prompt), CLOSE) == len(prompt) + 2 + 3


def test_loop_start_ignores_one_restatement_and_the_answer():
    prompt = [50]
    block = list(range(10, 30))
    once = prompt + block + [5] + block + [6]  # repeated once: legitimate
    assert loop_start(once, 1, CLOSE) is None
    # Loop after the thought closes: the answer restating is not trained against.
    answer = prompt + [1, 2, 3, 4, 5, CLOSE] + block * 3
    assert loop_start(answer, 1, CLOSE) is None


def test_loop_start_ignores_an_enumeration():
    prompt = [50]
    shared = list(range(10, 28))  # 18 tokens every item repeats
    ids = prompt + [1]
    for item in range(30, 36):  # the item varies, so no copy spans a whole period
        ids += [item] + shared
    assert loop_start(ids, 1, CLOSE) is None
    # The same lines, one repeated verbatim, are a loop.
    ids += ([35] + shared) * 2
    assert loop_start(ids, 1, CLOSE) is not None


def test_first_readable_skips_whitespace_tokens():
    readable = lambda t: t >= 10
    assert first_readable([1, 2, 3, 15], 1, readable) == 3
    assert first_readable([1, 2, 3], 1, readable) is None


def test_alternatives_filter():
    spellings = {0: " The", 1: "The", 2: " Wait", 3: " So", 4: "\n", 5: " Next", 6: " rare"}
    probs = np.array([0.5, 0.2, 0.15, 0.06, 0.05, 0.03, 0.01])
    out = alternatives(probs, 0, spell=spellings.get, readable=lambda t: t != 4, allowed=set(),
                       top=7, min_p=0.02, limit=3)
    # " The" respelled, the hedge, the newline and the improbable token are all excluded.
    assert out == [3, 5]
    assert alternatives(probs, 0, spell=spellings.get, readable=lambda t: t != 4, allowed={4},
                        top=7, min_p=0.02, limit=3) == [3, 4, 5]


def test_flatten_culls_common_tokens_and_never_empties_a_row():
    rows = [{"rejected": 1, "chosen": [7, 8]} for _ in range(200)]
    rows += [{"rejected": t, "chosen": [7]} for t in range(2, 12)]
    out = flatten([dict(r) for r in rows], rejected_strength=1.0, chosen_strength=1.0)
    assert sum(r["rejected"] == 1 for r in out) < 30
    assert all(r["chosen"] for r in out)


def row_for(logits, chosen, rejected, ref_ids):
    ref = logits.detach().clone()
    return {"chosen": torch.tensor(chosen), "rejected": torch.tensor(rejected),
            "ref_ids": torch.tensor(ref_ids), "ref_logits": ref[ref_ids],
            "ref_target_logits": ref[chosen + [rejected]]}


def test_ftpo_loss_at_the_reference_is_preference_only():
    logits = torch.zeros(16, requires_grad=True)
    row = row_for(logits, [1, 2], 0, list(range(16)))
    objective, margin, win = ftpo_loss(logits, row)
    assert torch.isclose(objective, torch.log(torch.tensor(2.0)))
    assert float(margin) == 0 and float(win) == 0
    objective.backward()
    assert logits.grad[0] > 0 and logits.grad[1] < 0 and logits.grad[2] < 0
    assert torch.all(logits.grad[3:] == 0)


def test_ftpo_loss_switches_off_past_the_clip():
    logits = torch.zeros(16)
    logits[1] = 3.0
    reference = row_for(logits, [1], 0, list(range(16)))
    logits.requires_grad_(True)
    objective, _, win = ftpo_loss(logits, reference, clip=2.0)
    objective.backward()
    assert float(win) == 1 and float(objective) == 0
    assert torch.all(logits.grad == 0)


def test_ftpo_tethers_pull_back_drift():
    base = torch.zeros(16)
    base[1] = 5.0  # separated: preference is off
    row = row_for(base, [1], 0, list(range(16)))
    moved = base.clone()
    moved[7] += 1.0  # non-target drift: full tether
    moved[1] += 1.0  # target drift within tau: free
    moved.requires_grad_(True)
    objective, _, _ = ftpo_loss(moved, row, tether=0.4, target_tether=0.05, tau=1.5)
    objective.backward()
    assert moved.grad[7] > 0 and moved.grad[1] == 0


def test_ftpo_record_through_the_training_step_separates_the_tokens():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_dense_gr_training_step import TinyLM, ce, records
    from training_step import backward_step

    torch.manual_seed(0)
    model = TinyLM()
    ids = torch.randint(0, 32, (1, 8))
    with torch.no_grad():
        ref = model.lm_head(model.model(ids).last_hidden_state[0, 5]).float()
    row = {"pair": True, "ftpo": True, "input_ids": ids, "position": 5, "rejected": torch.tensor(3),
           "chosen": torch.tensor([4, 5]), "ref_ids": torch.arange(32), "ref_logits": ref,
           "ref_target_logits": ref[[4, 5, 3]]}
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    for _ in range(30):
        model.zero_grad()
        result = backward_step(model, records()[:1] + [row], ce=ce, pair_weight=1.0)
        optimizer.step()
    assert result["chosen_win"] == 1.0 and result["ftpo_margin"] > 0
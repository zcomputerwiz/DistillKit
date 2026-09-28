"""DPO pairs through the shared training step: log 2 at the reference, and training
separates the chosen response from the rejected one."""
import math
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scratch" / "dense_gr"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_dense_gr_training_step import TinyLM, ce, records  # noqa: E402


def logprob(model, hidden, ids, start, end):
    logits = F.linear(hidden[:, start - 1:end - 1], model.lm_head.weight)
    return logits.log_softmax(-1).gather(-1, ids[:, start:end, None]).sum()


def pair_for(model, chosen, rejected, start=3):
    pair = {"pair": True, "chosen_ids": chosen, "rejected_ids": rejected, "chosen_start": start,
            "rejected_start": start, "chosen_end": chosen.shape[1], "rejected_end": rejected.shape[1]}
    with torch.no_grad():
        for side, ids in (("chosen", chosen), ("rejected", rejected)):
            hidden = model.model(ids).last_hidden_state
            pair["ref_" + side] = logprob(model, hidden, ids, start, ids.shape[1])
    return pair


def test_dpo_starts_at_log2_and_training_separates_the_pair():
    from training_step import backward_step

    torch.manual_seed(0)
    model = TinyLM()
    gen = torch.Generator().manual_seed(1)
    prompt = torch.randint(0, 32, (1, 3), generator=gen)
    chosen = torch.cat([prompt, torch.randint(0, 32, (1, 6), generator=gen)], 1)
    rejected = torch.cat([prompt, torch.tensor([[7, 7, 7, 7, 7, 7, 7, 7]])], 1)
    pair = pair_for(model, chosen, rejected)
    replay = records()[:1]
    result = backward_step(model, replay + [pair], ce=ce, pair_weight=1.0, logprob=logprob)
    assert abs(result["dpo"] - math.log(2)) < 1e-5 and abs(result["dpo_margin"]) < 1e-5
    # the replay record is accounted exactly as without the pair
    assert result["targets"] == backward_step(TinyLM(), replay, ce=ce)["targets"]
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    for _ in range(20):
        model.zero_grad()
        result = backward_step(model, replay + [pair], ce=ce, pair_weight=1.0, logprob=logprob)
        optimizer.step()
    assert result["dpo_margin"] > 0.1 and result["dpo"] < math.log(2)


def test_right_padding_leaves_the_pair_unchanged():
    """Padding after the response cannot change a causal model's view of it."""
    from training_step import preference_loss

    torch.manual_seed(0)
    model = TinyLM()
    chosen = torch.randint(0, 32, (1, 9))
    rejected = torch.randint(0, 32, (1, 11))
    pair = pair_for(model, chosen, rejected)
    padded = dict(pair, chosen_ids=F.pad(chosen, (0, 7), value=0), rejected_ids=F.pad(rejected, (0, 5), value=0))
    a = preference_loss(model, pair, beta=0.1, sft_weight=0.2, logprob=logprob)
    b = preference_loss(model, padded, beta=0.1, sft_weight=0.2, logprob=logprob)
    for x, y in zip(a, b):
        torch.testing.assert_close(x, y)

"""The copy probe: can the model copy a sequence it has just seen once?

`docs/standard_parts.md` specifies this as behavioral rather than a signature, and on
*random* tokens rather than natural text, so that the answer cannot be supplied by n-gram
statistics the model has by other means. A block of uniformly sampled ids is drawn, then
repeated verbatim, and only the second occurrence is scored.

Two references make the number readable and both are computed here rather than assumed:

* **Chance** is ``ln(vocab)``. Uniform ids carry no information, so nothing in the
  distribution of the corpus can beat it on the first occurrence.
* **The first occurrence** is the same model on the same tokens with no prior copy to
  match against. It should sit at chance. If it does not, the sampling is not uniform in
  the way the probe assumes and the second-occurrence number means something else.

So the quantity of interest is the *gain*, first occurrence minus second: how many nats
the model saves purely by having seen the block before. A backbone that has built an
induction circuit drives the second occurrence toward zero. One that has not leaves it at
chance and the gain at zero.

The attention-pattern signature -- per-head prefix-matching score -- is deliberately not
here. It needs eager attention to read weights back, which costs the Flash-Attention path,
and the gate this probe serves is behavioral. The signature belongs with the copy
experiment, where the two readings are meant to be compared.
"""

from __future__ import annotations

import numpy as np
import torch

import triton_shim  # noqa: F401  resolves triton-windows before CCE reads the version
from cut_cross_entropy import linear_cross_entropy


def probe_batches(vocab, half=256, windows=64, seed=1234):
    """Uniform random blocks of ``half`` ids, each repeated once.

    Sampling is uniform over the whole compact vocabulary, added tokens included. They
    are a few dozen ids out of tens of thousands, so they cannot move a mean taken over
    every position, and excluding them would mean assuming where they sit in the compact
    space -- which is ordered by original id, not grouped at either end.
    """
    generator = np.random.default_rng(seed)
    block = generator.integers(0, vocab, size=(windows, half), dtype=np.int64)
    return torch.from_numpy(np.concatenate([block, block], axis=1))


@torch.no_grad()
def copy_probe(model, vocab, half=256, windows=64, batch=32, seed=1234,
               device="cuda"):
    """Mean NLL in nats on each occurrence of a repeated random block."""
    sequences = probe_batches(vocab, half, windows, seed)
    was_training = model.training
    model.eval()
    first_total, second_total, counted = 0.0, 0.0, 0
    for start in range(0, sequences.shape[0], batch):
        chunk = sequences[start:start + batch].to(device)
        state = model.model(input_ids=chunk,
                            attention_mask=torch.ones_like(chunk),
                            use_cache=False).last_hidden_state
        losses = linear_cross_entropy(state, model.lm_head.weight, chunk, shift=1,
                                      reduction="none").float()
        # `shift=1` returns `length - 1` entries, where entry i is the loss for predicting
        # token i + 1 from position i. The second occurrence is tokens half..2*half-1, so
        # its predictions are the last `half` entries. The first occurrence is tokens
        # 0..half-1, and token 0 has no prediction, so it is the leading `half - 1`.
        second_total += float(losses[:, -half:].mean()) * chunk.shape[0]
        first_total += float(losses[:, :half - 1].mean()) * chunk.shape[0]
        counted += chunk.shape[0]
    if was_training:
        model.train()
    first = first_total / counted
    second = second_total / counted
    return {
        "chance": float(np.log(vocab)),
        "first_occurrence": first,
        "second_occurrence": second,
        "gain": first - second,
        "windows": int(sequences.shape[0]),
        "half": half,
    }


def format_probe(result):
    return ("copy probe: first %.3f  second %.3f  gain %+.3f  (chance %.3f)"
            % (result["first_occurrence"], result["second_occurrence"],
               result["gain"], result["chance"]))


def _self_check():
    """Two stubs with known behavior pin the slicing, which is what can silently be wrong.

    Needs CUDA: Cut Cross-Entropy is a Triton kernel and refuses CPU tensors.

    The interesting stub is deliberately good at *only* the second occurrence. If the two
    slices were swapped, or off by the boundary token, its numbers would come out the
    other way round -- which no amount of staring at the indexing proves on its own.
    """
    if not torch.cuda.is_available():
        raise SystemExit("copy_probe self-check needs CUDA (Cut Cross-Entropy is Triton)")
    vocab, half = 512, 8

    class Stub(torch.nn.Module):
        """`lm_head` is the identity, so the hidden state *is* the logits."""

        def __init__(self, copies):
            super().__init__()
            self.lm_head = torch.nn.Linear(vocab, vocab, bias=False)
            with torch.no_grad():
                self.lm_head.weight.copy_(torch.eye(vocab))
            self.copies = copies

        def model(self, input_ids, attention_mask=None, use_cache=False):
            batch, length = input_ids.shape
            state = torch.zeros(batch, length, vocab, device=input_ids.device)
            if self.copies:
                # Entry i of the loss predicts token i + 1, and the second occurrence is
                # the last `half` of those, so make positions length-1-half onward easy.
                target = torch.roll(input_ids, shifts=-1, dims=1)
                confident = torch.zeros(batch, length, vocab, device=input_ids.device)
                confident.scatter_(2, target.unsqueeze(2), 30.0)
                state[:, length - 1 - half:] = confident[:, length - 1 - half:]
            return type("Output", (), {"last_hidden_state": state})()

    flat = copy_probe(Stub(copies=False), vocab, half=half, windows=8, batch=4)
    assert abs(flat["first_occurrence"] - np.log(vocab)) < 1e-3, flat
    assert abs(flat["second_occurrence"] - np.log(vocab)) < 1e-3, flat
    assert abs(flat["gain"]) < 1e-3, flat

    copier = copy_probe(Stub(copies=True), vocab, half=half, windows=8, batch=4)
    assert abs(copier["first_occurrence"] - np.log(vocab)) < 1e-3, copier
    assert copier["second_occurrence"] < 1e-3, copier
    assert copier["gain"] > np.log(vocab) - 1e-2, copier

    print("copy_probe self-check: ok (chance %.4f, flat gain %+.2e, copier gain %+.4f)"
          % (flat["chance"], flat["gain"], copier["gain"]))


if __name__ == "__main__":
    _self_check()

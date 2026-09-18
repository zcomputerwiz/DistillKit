"""The copy module's connector, and the three arms of the displacement experiment.

The module itself is parameterless -- `repetition_fast.batch_repetition_index` computes it
exactly, with no learning anywhere. What has to be learned is the *connector*: the small,
per-backbone, disposable map from the module's fixed-width output into this backbone's
residual width. `docs/standard_parts.md` is explicit that the connector is not the reusable
artifact, and that the module has to be present from step 0, because Phase 1b showed a
channel introduced after the backbone has learned the task simply gets ignored.

The connector reads two things and adds their projection to the input embedding:

* the 4-wide feature code -- matched, log distance, agreement length, candidate stability
* the candidate continuation token, embedded through the backbone's *own* embedding table

Embedding the candidate through `embed_tokens` rather than a private table is deliberate.
It costs no parameters, it puts the candidate in the space the backbone already reads, and
it keeps the connector's width independent of the vocabulary -- so the same connector shape
works at any cut.

## The arms

| arm | module | what it isolates |
| --- | --- | --- |
| `plain` | absent | what the backbone builds unaided |
| `copy` | real | the intervention |
| `scrambled` | permuted across positions, every step | bandwidth, timing, capacity, parameter count |

`scrambled` is the control that makes `copy` readable. It holds everything fixed except
content, so if it converges to `plain` the connector is not injecting harmful noise and
mere presence of a signal is not doing the work.

**Train-time scrambling and evaluation-time substitution are different measurements.**
Scrambling is an arm, applied for the whole of training. Substitution is applied to the
*trained* `copy` model at the end, feeding it permuted content it never saw. The first
bounds the connector's cost; the second measures content dependence. An earlier draft of
the design document used one word for both.
"""

from __future__ import annotations

import numpy as np
import torch

from repetition_fast import FEATURES, batch_repetition_index

ARMS = ("plain", "copy", "scrambled")


class Connector(torch.nn.Module):
    """Fixed-width module output -> this backbone's residual width.

    Zero-initialized on the output side so training starts bit-identical to `plain`. A
    single zeroed factor still receives gradient -- ``dL/dW = grad_out (x) input`` is
    nonzero -- which is why this is safe and why zeroing *both* factors of a bottleneck,
    as `scratch/gr_retrofit/REPORT.md` records, is not.
    """

    def __init__(self, hidden, features=FEATURES):
        super().__init__()
        self.project = torch.nn.Linear(features + hidden, hidden, bias=False)
        torch.nn.init.zeros_(self.project.weight)

    def forward(self, features, candidate, embed_tokens):
        # A position with no match contributes nothing: its candidate embedding is zeroed
        # rather than left pointing at token 0, which would be a real token.
        matched = (candidate >= 0).unsqueeze(-1)
        safe = candidate.clamp(min=0)
        found = embed_tokens(safe) * matched.to(embed_tokens.weight.dtype)
        joined = torch.cat([features.to(found.dtype), found], dim=-1)
        return self.project(joined)


def module_output(tokens, arm, vocab, generator, order=3, max_length=8):
    """Features and candidate for a batch of token ids, under one arm's treatment."""
    if arm == "plain":
        return None, None
    features, candidate = batch_repetition_index(tokens, order, max_length, vocab=vocab)
    if arm == "scrambled":
        features, candidate = scramble(features, candidate, generator)
    return features, candidate


def scramble(features, candidate, generator):
    """Permute the module's output across positions, independently per sequence.

    Bandwidth, activation timing, connector capacity and parameter count are all preserved;
    only the correspondence between a position and its own suffix match is destroyed.
    """
    rows, n = candidate.shape
    order = np.argsort(generator.random((rows, n)), axis=1)
    take = np.take_along_axis
    return (take(features, order[:, :, None], axis=1),
            take(candidate, order, axis=1))


def build_inputs(model, tokens, features, candidate, connector, ablate=False):
    """Input embeddings for one step, with the module added unless it is ablated."""
    embed_tokens = model.model.embed_tokens
    inputs = embed_tokens(tokens)
    if connector is None or features is None or ablate:
        return inputs
    return inputs + connector(features, candidate, embed_tokens)


def _self_check():
    """The connector must start inert, stay trainable, and ignore unmatched positions."""
    torch.manual_seed(0)
    hidden, vocab, rows, n = 32, 128, 3, 16
    embed = torch.nn.Embedding(vocab, hidden)
    connector = Connector(hidden)

    features = torch.rand(rows, n, FEATURES)
    candidate = torch.randint(-1, vocab, (rows, n))

    out = connector(features, candidate, embed)
    assert out.shape == (rows, n, hidden)
    assert torch.count_nonzero(out) == 0, "zero init must start inert"

    # Gradient must still reach the weight, or the connector is stranded.
    out.sum().backward()
    assert connector.project.weight.grad is not None
    assert torch.count_nonzero(connector.project.weight.grad) > 0, "no gradient path"

    # An unmatched position must not read a real token's embedding.
    with torch.no_grad():
        connector.project.weight.normal_()
    blank = torch.full((1, 1), -1)
    zero_feature = torch.zeros(1, 1, FEATURES)
    unmatched = connector(zero_feature, blank, embed)
    assert torch.count_nonzero(unmatched) == 0, "unmatched position leaked an embedding"

    # Scrambling preserves the multiset of positions, and moves them.
    generator = np.random.default_rng(0)
    f = np.random.default_rng(1).random((4, 64, FEATURES)).astype(np.float32)
    c = np.arange(4 * 64, dtype=np.int64).reshape(4, 64)
    sf, sc = scramble(f, c, generator)
    assert sf.shape == f.shape and sc.shape == c.shape
    for row in range(4):
        assert set(sc[row].tolist()) == set(c[row].tolist()), "scramble lost content"
    assert (sc != c).mean() > 0.9, "scramble barely moved anything"

    print("copy_module self-check: ok (inert at init, gradient present, "
          "unmatched positions blank, scramble permutes %.0f%% of positions)"
          % (100 * (sc != c).mean()))


if __name__ == "__main__":
    _self_check()

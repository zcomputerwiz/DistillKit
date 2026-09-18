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


def align_to_prediction(features, candidate):
    """Shift the module's output to the position that consumes it.

    `repetition_index` defines ``candidate[t]`` as the continuation of the suffix ending
    at ``t-1`` -- that is, a prediction of ``tokens[t]``. But the loss uses the hidden
    state at ``t`` to predict ``tokens[t+1]``, so supplying ``candidate[t]`` at position
    ``t`` hands the backbone a second copy of the token it already has as input. Measured
    on a repeated random block: ``candidate[t] == tokens[t]`` at 29 of 29 positions inside
    the repeat, and ``== tokens[t+1]`` at 0 of 29.

    What position ``t`` needs is ``candidate[t+1]``, built from the suffix ending at ``t``.
    That is still causal: it reads ``tokens[t+1-order:t+1]`` and a previous occurrence
    strictly before ``t+1``, so nothing after ``t`` is touched. The last position has no
    successor and is left blank.
    """
    shifted_features = np.zeros_like(features)
    shifted_candidate = np.full_like(candidate, -1)
    shifted_features[:, :-1] = features[:, 1:]
    shifted_candidate[:, :-1] = candidate[:, 1:]
    return shifted_features, shifted_candidate


def module_output(tokens, arm, vocab, generator, order=3, max_length=8):
    """Features and candidate for a batch of token ids, under one arm's treatment."""
    if arm == "plain":
        return None, None
    features, candidate = batch_repetition_index(tokens, order, max_length, vocab=vocab)
    features, candidate = align_to_prediction(features, candidate)
    if arm == "scrambled":
        features, candidate = scramble(features, candidate, generator)
    return features, candidate


def randomize_candidates(features, candidate, generator, vocab):
    """Keep the channel's activation pattern, replace only what it says.

    `scramble` permutes the module's output across positions, which destroys
    position-correspondence but leaves every candidate a token drawn from this very
    sequence. On a repeated-random-block probe that leaks the block: knowing the answer
    lies in a 256-token set out of 32,768 is worth up to ln(128) = 4.85 nats, and the
    scrambled arm measured within 0.17 nats of the real module on the probe because of it.

    This replaces each matched position's candidate with a uniform draw from the
    vocabulary and leaves the features untouched, so which positions carry a signal, how
    far back it points and how long it agreed are all identical -- only the content is
    wrong. That is the `wrong_pointer` control from Phase 1, and it is what makes
    "content dependence" a measurement rather than a word.
    """
    matched = candidate >= 0
    replaced = candidate.copy()
    replaced[matched] = generator.integers(0, vocab, size=int(matched.sum()),
                                           dtype=np.int64)
    return features, replaced


def scramble(features, candidate, generator):
    """Permute the module's output across *sequences*, at the same position.

    An earlier version permuted across positions within a sequence, which is not a
    control -- it is future leakage. Position 5 could receive the output computed at
    position 900, which was built from tokens 900 and earlier. The evidence was in the
    endpoint matrix and went unread: the copy arm scored first-occurrence NLL 11.823
    under position-scrambling against 12.121 enabled, and the first occurrence contains
    no prior repeat, so nothing legitimate can make it easier.

    Permuting across batch rows at the same position keeps every treated output causal
    with respect to its own sequence -- row A's position ``t`` receives row B's position
    ``t``, built from row B's tokens up to ``t`` and nothing later. Bandwidth, activation
    timing, connector capacity and parameter count are preserved; the correspondence
    between a position and its own suffix match is destroyed.
    """
    rows = candidate.shape[0]
    if rows < 2:
        raise ValueError("cross-sequence scrambling needs at least two sequences")
    # A derangement, so no row keeps its own output.
    order = generator.permutation(rows)
    while np.any(order == np.arange(rows)):
        order = generator.permutation(rows)
    return features[order], candidate[order]


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

    # Alignment: the module attached at position t must supply tokens[t+1].
    generator = np.random.default_rng(0)
    half = 32
    block = generator.integers(0, 1000, size=(4, half))
    repeated = np.concatenate([block, block], axis=1)
    _, aligned = module_output(repeated, "copy", 1000, generator)
    inside = slice(half + 3, 2 * half - 1)
    want = repeated[:, 1:][:, inside]
    got = aligned[:, inside]
    assert np.array_equal(got, want), (
        "module at position t must supply tokens[t+1]; %d of %d wrong"
        % (int((got != want).sum()), want.size))

    # Causality: every treatment must be prefix-invariant. Truncating the batch's
    # sequences must not change any output that survives, or a position saw its future.
    for arm in ("copy", "scrambled"):
        full_f, full_c = module_output(repeated, arm, 1000,
                                       np.random.default_rng(5))
        cut = half + 7
        part_f, part_c = module_output(repeated[:, :cut], arm, 1000,
                                       np.random.default_rng(5))
        # The final position of any window is blank by construction, so compare before it.
        assert np.array_equal(full_c[:, :cut - 1], part_c[:, :cut - 1]), (
            "%s is not causal: a prefix disagrees with the full sequence" % arm)
        assert np.allclose(full_f[:, :cut - 1], part_f[:, :cut - 1]), (
            "%s features are not causal" % arm)

    # Scrambling must move content between sequences and keep each position's own slot.
    f = np.random.default_rng(1).random((4, 16, FEATURES)).astype(np.float32)
    c = np.arange(4 * 16, dtype=np.int64).reshape(4, 16)
    sf, sc = scramble(f, c, np.random.default_rng(0))
    assert sf.shape == f.shape and sc.shape == c.shape
    assert not np.any([np.array_equal(sc[r], c[r]) for r in range(4)]), \
        "a sequence kept its own module output"
    assert set(sc.ravel().tolist()) == set(c.ravel().tolist()), "scramble lost content"

    print("copy_module self-check: ok (inert at init, gradient present, unmatched "
          "positions blank, module at t supplies tokens[t+1], every treatment causal)")


if __name__ == "__main__":
    _self_check()

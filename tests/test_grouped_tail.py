"""`symmetric_uniform` is the grouped-tail objective, and this is what pins that.

The objective forensics said the zero tail is what turns the sidecar from a help into a
harm: `missing_probability_handling: zero` gives every token outside the cached top-k a
target probability of exactly zero, so a student correcting toward a true token the
teacher never ranked is penalised for being right. The repair is to distil over the k
cached tokens plus one aggregate bucket carrying the omitted mass:

    L = -sum_i p_i log q_i  -  p_tail log q_tail

which says only what the cache knows. The point of these tests is that this codebase
already computes it. Spreading the tail uniformly over the same `V - k` support on both
sides cancels the per-token factor:

    sum_{i not in k} (p_tail/(V-k)) log[(p_tail/(V-k)) / (q_tail/(V-k))]
        = p_tail log(p_tail / q_tail)

so `SYMMETRIC_UNIFORM` is grouped-tail under a name that describes an assumption it does
not actually make. No new loss function is needed, only a different flag -- and that is
worth a test rather than a comment, because the next person to read the enum will believe
the name.
"""

import torch

from distillkit.lossfuncs.kl import sparse_kl_div_inner
from distillkit.missing_probability import MissingProbabilityHandling


def case(seed=0, vocab=64, top_k=8, positions=5, retained=0.8):
    generator = torch.Generator().manual_seed(seed)
    logits = torch.randn(1, positions, vocab, dtype=torch.float64, generator=generator,
                         requires_grad=True)
    ids = torch.stack([torch.randperm(vocab, generator=generator)[:top_k]
                       for _ in range(positions)]).unsqueeze(0)
    values = torch.rand(1, positions, top_k, dtype=torch.float64, generator=generator)
    values = values / values.sum(-1, keepdim=True) * retained
    mask = torch.ones(1, positions, 1, dtype=torch.bool)
    return logits, ids, values, mask


def grouped_tail(logits, ids, probabilities):
    """The proposed target, written from its own definition rather than from the enum."""
    logprobs = torch.log_softmax(logits, dim=-1)
    kept = logprobs.gather(-1, ids.squeeze(0).unsqueeze(0))
    tail = torch.log1p(-logprobs.exp().gather(-1, ids.squeeze(0).unsqueeze(0))
                       .sum(-1).clamp(max=1 - 1e-12))
    return (-(probabilities * kept).sum(-1).sum()
            - ((1 - probabilities.sum(-1)) * tail).sum())


def test_symmetric_uniform_is_grouped_tail_up_to_the_teacher_entropy():
    logits, ids, values, mask = case()
    divergence = sparse_kl_div_inner(
        logits, ids, values.log(), mask,
        missing=MissingProbabilityHandling.SYMMETRIC_UNIFORM, log_target=True)
    cross = grouped_tail(logits, ids, values)

    tail = 1 - values.sum(-1)
    entropy = -((values * values.log()).sum(-1) + tail * tail.log()).sum()
    # The two differ by the coarsened teacher's entropy, which is a constant: the same
    # objective, reported on a different additive offset.
    assert torch.allclose(divergence - cross, -entropy, atol=1e-9)


def test_the_two_forms_have_the_same_gradient():
    """The offset is constant, so the student cannot tell them apart -- check, don't assume."""
    logits, ids, values, mask = case(seed=3)
    divergence = sparse_kl_div_inner(
        logits, ids, values.log(), mask,
        missing=MissingProbabilityHandling.SYMMETRIC_UNIFORM, log_target=True)
    from_enum, = torch.autograd.grad(divergence, logits, retain_graph=False)
    from_formula, = torch.autograd.grad(grouped_tail(logits, ids, values), logits)
    assert torch.allclose(from_enum, from_formula, atol=1e-7)


def student(ids, values, tail_mass, vocab=64):
    """Logits whose tail carries exactly `tail_mass`, in-list shape matching the teacher.

    Built directly in probability space so the two arms differ in one interpretable
    quantity -- how much the student believes lies outside the teacher's list.
    """
    probabilities = torch.full((1, values.shape[1], vocab), 0.0, dtype=torch.float64)
    inside = ids.squeeze(0).unsqueeze(0)
    probabilities.scatter_(-1, inside, values / values.sum(-1, keepdim=True) * (1 - tail_mass))
    outside = probabilities == 0
    probabilities = probabilities + outside * (tail_mass / outside.sum(-1, keepdim=True))
    return probabilities.log()


def test_the_zero_tail_punishes_a_student_for_matching_the_teacher_tail():
    """The mechanism the forensics measured, in four lines.

    The teacher here keeps 0.6 and omits 0.4. One student puts 0.05 outside the list, the
    other puts the teacher's own 0.40 there. Grouped-tail prefers the student that is
    right about the omitted mass; the zero tail prefers the one that is wrong, because it
    was told the omitted mass is zero. That is what penalises a sidecar for correcting
    toward a true token the teacher never ranked.
    """
    _, ids, values, mask = case(seed=7, retained=0.6)
    understated = student(ids, values, tail_mass=0.05)
    matching = student(ids, values, tail_mass=0.40)

    def score(tensor, missing):
        return float(sparse_kl_div_inner(tensor, ids, values.log(), mask, missing=missing,
                                         log_target=True))

    zero = MissingProbabilityHandling.ZERO
    grouped = MissingProbabilityHandling.SYMMETRIC_UNIFORM
    assert score(matching, grouped) < score(understated, grouped)
    assert score(matching, zero) > score(understated, zero)


def test_grouped_tail_is_minimised_at_the_teacher_tail_mass():
    """Two-sided, unlike the zero tail: too much omitted mass is penalised as well."""
    _, ids, values, mask = case(seed=7, retained=0.6)
    grouped = MissingProbabilityHandling.SYMMETRIC_UNIFORM
    scores = [float(sparse_kl_div_inner(student(ids, values, tail_mass=mass), ids,
                                        values.log(), mask, missing=grouped,
                                        log_target=True))
              for mass in (0.05, 0.40, 0.90)]
    assert scores[1] < scores[0] and scores[1] < scores[2]


def test_grouped_tail_survives_a_top_k_that_already_covers_the_mass():
    """The fp32 clamp bug, pinned.

    `1 - eps` at the default eps of 1e-8 rounds to exactly 1.0 in fp32, so the upper
    clamp did nothing and `log1p(-1.0)` returned -inf wherever the student's top-k had
    gathered the mass -- which, multiplied by a teacher tail of ~0, is NaN. Real captures
    are exactly that well covered: this corpus's median retained mass is 1.0000, so the
    first training step on real data produced NaN everywhere.
    """
    _, ids, values, mask = case(seed=11, retained=0.999999)
    confident = student(ids, values, tail_mass=1e-9).to(torch.float32)
    loss = sparse_kl_div_inner(
        confident, ids, values.log().to(torch.float32), mask,
        missing=MissingProbabilityHandling.SYMMETRIC_UNIFORM, log_target=True)
    assert torch.isfinite(loss), loss

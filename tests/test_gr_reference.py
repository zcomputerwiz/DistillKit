"""The independent float64 reference for the recipient-initialized GR conversion.

These tests are about the algebra, not the repository's implementation: the reference in
``scratch/gr_retrofit/reference.py`` is written separately and imports nothing from
``distillkit``, so agreement between them is a second opinion rather than a tautology.

What has to hold, in order of how badly it would mislead if it failed silently:

    the read collapses     with W_up = 0 and gamma_i = 2 gamma the four-branch read is
                           exactly the recipient's own normalized input
    the induction closes   every branch receives the same update, so all branches stay
                           equal and the final collapse returns the recipient's h
    the equations are real once branches differ, the reference must still be computing the
                           GR route rather than an identity special case
    the bottleneck opens   W_up must be able to move first and W_down must receive
                           gradient afterwards; zeroing both factors would strand W_down
    the asymmetry bites    unequal gains must produce unequal gradients, or the asymmetric
                           mode is symmetric with extra steps
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scratch" / "gr_retrofit"))

from reference import (EPSILON, branch_gains, collapse, converted_stack, gr_sublayer,
                       initial_parameters, original_stack, original_sublayer, rms_norm)

HIDDEN = 12
EPS = 1e-6
TOLERANCE = {"atol": 1e-10, "rtol": 1e-10}


def make_sublayer(seed, hidden=HIDDEN):
    """A nonlinear, non-identity stand-in for attention or an MLP."""
    generator = torch.Generator().manual_seed(seed)
    first = torch.randn(hidden, hidden, generator=generator, dtype=torch.float64) / 3
    second = torch.randn(hidden, hidden, generator=generator, dtype=torch.float64) / 3
    return lambda u: torch.tanh(u @ first) @ second


def make_state(seed, batch=3, hidden=HIDDEN):
    generator = torch.Generator().manual_seed(seed)
    # Deliberately not unit scale: an implementation that ignored the norm would pass on
    # already-normalized input.
    return torch.randn(batch, hidden, generator=generator, dtype=torch.float64) * 4.0


def make_gain(seed, hidden=HIDDEN):
    """A nonunit recipient gain, as a real checkpoint has."""
    generator = torch.Generator().manual_seed(seed)
    return 1.0 + 0.3 * torch.randn(hidden, generator=generator, dtype=torch.float64)


class TestInitialEquivalence:
    @pytest.mark.parametrize("asymmetric", [False, True])
    def test_one_sublayer_matches_the_original(self, asymmetric):
        h, gain, sublayer = make_state(0), make_gain(1), make_sublayer(2)
        w_down, w_up, w_write = initial_parameters(HIDDEN, seed=3)
        states = h.unsqueeze(-2).repeat(1, 4, 1)
        moved = gr_sublayer(states, branch_gains(gain, 4, asymmetric), w_down, w_up,
                            w_write, EPS, sublayer)
        assert torch.allclose(collapse(moved),
                              original_sublayer(h, gain, EPS, sublayer), **TOLERANCE)

    @pytest.mark.parametrize("asymmetric", [False, True])
    def test_consecutive_sublayers_match_through_the_collapse(self, asymmetric):
        """The induction: four alternating sublayers, nonunit gains, one final collapse."""
        h = make_state(10)
        gains = [make_gain(20 + i) for i in range(4)]
        sublayers = [make_sublayer(30 + i) for i in range(4)]
        assert torch.allclose(
            converted_stack(h, gains, sublayers, EPS, asymmetric=asymmetric),
            original_stack(h, gains, sublayers, EPS), **TOLERANCE)

    def test_every_branch_stays_equal_while_untrained(self):
        h, gain, sublayer = make_state(4), make_gain(5), make_sublayer(6)
        w_down, w_up, w_write = initial_parameters(HIDDEN, seed=7)
        states = h.unsqueeze(-2).repeat(1, 4, 1)
        moved = gr_sublayer(states, branch_gains(gain, 4), w_down, w_up, w_write, EPS,
                            sublayer)
        for index in range(1, 4):
            assert torch.allclose(moved[..., 0, :], moved[..., index, :], **TOLERANCE)

    def test_the_gate_is_exactly_one_half_and_the_write_exactly_one(self):
        """The two facts the collapse depends on, checked directly rather than inferred."""
        w_down, w_up, w_write = initial_parameters(HIDDEN, seed=8)
        flattened = torch.randn(3, 4 * HIDDEN, dtype=torch.float64)
        logits = torch.nn.functional.silu(flattened @ w_down.T / 4) @ w_up.T
        assert torch.equal(torch.sigmoid(logits), torch.full_like(logits, 0.5))
        write = 2.0 * torch.sigmoid(flattened @ w_write.T / 4)
        assert torch.equal(write, torch.ones_like(write))

    def test_the_read_bottleneck_is_not_doubly_zeroed(self):
        """W_down must be nonzero or it can never receive gradient once W_up moves."""
        w_down, w_up, _ = initial_parameters(HIDDEN, seed=9)
        assert float(w_down.abs().max()) > 0
        assert float(w_up.abs().max()) == 0

    def test_the_perturbation_is_mean_zero(self):
        assert float(EPSILON.sum().abs()) < 1e-12
        assert float(EPSILON.abs().max()) == pytest.approx(3 / 128)

    def test_asymmetric_gains_differ_but_average_to_the_symmetric_gain(self):
        gain = make_gain(11)
        asymmetric = branch_gains(gain, 4, asymmetric=True)
        assert torch.allclose(asymmetric.mean(0), 2.0 * gain, **TOLERANCE)
        assert not torch.allclose(asymmetric[0], asymmetric[1])


class TestRealEquations:
    """Once branches differ the reference must exercise the actual GR route."""

    def _diverged(self, seed=40):
        h = make_state(seed)
        states = h.unsqueeze(-2).repeat(1, 4, 1)
        generator = torch.Generator().manual_seed(seed + 1)
        states = states + 0.1 * torch.randn(states.shape, generator=generator,
                                            dtype=torch.float64)
        return states

    def test_unequal_branches_no_longer_collapse_to_the_original(self):
        """If this passed, the reference would be testing an identity, not the route."""
        states = self._diverged()
        gain, sublayer = make_gain(41), make_sublayer(42)
        w_down, w_up, w_write = initial_parameters(HIDDEN, seed=43)
        torch.manual_seed(44)
        w_up = torch.randn_like(w_up) * 0.1
        w_write = torch.randn_like(w_write) * 0.1
        moved = gr_sublayer(states, branch_gains(gain, 4), w_down, w_up, w_write, EPS,
                            sublayer)
        assert not torch.allclose(moved[..., 0, :], moved[..., 1, :])

    def test_nonzero_write_rows_give_branches_different_updates(self):
        states = self._diverged(50)
        gain, sublayer = make_gain(51), make_sublayer(52)
        w_down, w_up, w_write = initial_parameters(HIDDEN, seed=53)
        w_write = w_write.clone()
        w_write[0] += 0.05                       # one row only
        moved = gr_sublayer(states, branch_gains(gain, 4), w_down, w_up, w_write, EPS,
                            sublayer)
        delta = moved - states
        assert not torch.allclose(delta[..., 0, :], delta[..., 1, :])

    def test_the_read_depends_on_every_branch(self):
        """A route that read one branch would still pass the identity tests."""
        gain, sublayer = make_gain(61), make_sublayer(62)
        w_down, w_up, w_write = initial_parameters(HIDDEN, seed=63)
        torch.manual_seed(64)
        w_up = torch.randn_like(w_up) * 0.2
        base = self._diverged(60)
        first = gr_sublayer(base, branch_gains(gain, 4), w_down, w_up, w_write, EPS,
                            sublayer)
        for index in range(4):
            moved = base.clone()
            moved[..., index, :] += 0.5
            changed = gr_sublayer(moved, branch_gains(gain, 4), w_down, w_up, w_write,
                                  EPS, sublayer)
            other = [j for j in range(4) if j != index][0]
            assert not torch.allclose(changed[..., other, :], first[..., other, :]), (
                "branch %d does not influence the shared read" % index)


class TestGradientPath:
    """The bottleneck must open in the expected order, and asymmetry must bite."""

    def _setup(self, asymmetric, seed=70, lowrank=8):
        h, gain, sublayer = make_state(seed), make_gain(seed + 1), make_sublayer(seed + 2)
        w_down, w_up, w_write = initial_parameters(HIDDEN, lowrank=lowrank, seed=seed + 3)
        parameters = {"w_down": w_down.clone().requires_grad_(True),
                      "w_up": w_up.clone().requires_grad_(True),
                      "w_write": w_write.clone().requires_grad_(True)}
        gains = branch_gains(gain, 4, asymmetric).clone().requires_grad_(True)
        return h, gains, sublayer, parameters

    def _step(self, h, gains, sublayer, parameters, target):
        states = h.unsqueeze(-2).repeat(1, 4, 1)
        moved = gr_sublayer(states, gains, parameters["w_down"], parameters["w_up"],
                            parameters["w_write"], EPS, sublayer)
        loss = (collapse(moved) - target).pow(2).mean()
        loss.backward()
        return float(loss)

    @pytest.mark.parametrize("asymmetric", [False, True])
    def test_w_up_receives_gradient_first_and_w_down_does_not(self, asymmetric):
        """W_down sits behind a zero W_up, so its gradient is zero on the first step.

        This is the expected sequence, not a defect -- but it has to be verified, because
        a W_down that never receives gradient at all is the zero-multiplier trap.
        """
        h, gains, sublayer, parameters = self._setup(asymmetric)
        self._step(h, gains, sublayer, parameters, make_state(99))
        assert float(parameters["w_up"].grad.abs().max()) > 0
        assert float(parameters["w_down"].grad.abs().max()) == 0
        assert float(parameters["w_write"].grad.abs().max()) > 0

    @pytest.mark.parametrize("asymmetric", [False, True])
    def test_w_down_receives_gradient_after_w_up_moves(self, asymmetric):
        h, gains, sublayer, parameters = self._setup(asymmetric)
        target = make_state(99)
        optimizer = torch.optim.AdamW(list(parameters.values()) + [gains], lr=1e-2)
        self._step(h, gains, sublayer, parameters, target)
        optimizer.step()
        optimizer.zero_grad()
        self._step(h, gains, sublayer, parameters, target)
        assert float(parameters["w_down"].grad.abs().max()) > 0

    def test_several_steps_reduce_the_loss(self):
        h, gains, sublayer, parameters = self._setup(True)
        target = make_state(99)
        optimizer = torch.optim.AdamW(list(parameters.values()) + [gains], lr=1e-2)
        losses = []
        for _ in range(8):
            optimizer.zero_grad()
            losses.append(self._step(h, gains, sublayer, parameters, target))
            optimizer.step()
        assert losses[-1] < losses[0]

    @pytest.mark.parametrize("asymmetric", [False, True])
    def test_one_sublayer_cannot_break_symmetry_in_either_mode(self, asymmetric):
        """Measured, and structural rather than a defect.

        At initialization it is the gate *logits* that lose their input derivative, since
        they are produced through a zero ``W_up``; the value path still differentiates
        with respect to every branch state, with the gate held at a constant 1/2. So the
        read depends on the gains only through their *mean*, and a mean collapse sends
        identical gradient to every branch. Both facts force
        ``dL/dgamma_i`` and ``dL/dW_write[i]`` to be identical across branches, whatever
        the gains are. The asymmetric perturbation therefore buys nothing here -- a single
        sublayer read out by a mean cannot distinguish its branches at all.
        """
        h, gains, sublayer, parameters = self._setup(asymmetric)
        self._step(h, gains, sublayer, parameters, make_state(99))
        assert torch.allclose(gains.grad[0], gains.grad[1], atol=1e-12)
        rows = parameters["w_write"].grad
        assert torch.allclose(rows[0], rows[1], atol=1e-12)

    @pytest.mark.parametrize("asymmetric", [False, True])
    def test_depth_breaks_symmetry_in_both_modes(self, asymmetric):
        """Symmetry breaks once a *downstream* router can weight the branches unequally.

        ``W_up`` emits one output row per branch, so as soon as it leaves zero the
        downstream gate differs across branches, the gradient arriving back at each branch
        differs, the write rows diverge and the branch states follow. That is what
        actually separates the streams -- not the gain perturbation.
        """
        stack = _Stack(asymmetric, layers=3, seed=200)
        stack.run(steps=12, lr=5e-2)
        assert stack.write_row_gap() > 1e-4
        assert stack.branch_gap() > 1e-3

    def test_the_perturbation_is_not_what_breaks_symmetry(self):
        """The negative result the task asked to verify rather than assume.

        Both modes reach the same order of stream divergence in the same number of steps.
        Whatever the asymmetric initialization is worth, it is not the thing that lets the
        streams separate.
        """
        symmetric = _Stack(False, layers=3, seed=200)
        asymmetric = _Stack(True, layers=3, seed=200)
        symmetric.run(steps=12, lr=5e-2)
        asymmetric.run(steps=12, lr=5e-2)
        ratio = asymmetric.write_row_gap() / max(symmetric.write_row_gap(), 1e-30)
        assert 0.2 < ratio < 5.0


class _Stack:
    """Consecutive GR sublayers sharing one optimizer, for the depth diagnostics."""

    def __init__(self, asymmetric, layers=3, hidden=HIDDEN, seed=200, lowrank=8):
        self.hidden = hidden
        self.h = make_state(seed)
        self.target = make_state(seed + 1)
        self.sublayers = [make_sublayer(seed + 10 + i) for i in range(layers)]
        self.parameters, self.gains = [], []
        for index in range(layers):
            w_down, w_up, w_write = initial_parameters(hidden, lowrank=lowrank,
                                                       seed=seed + 20 + index)
            self.parameters.append({
                "w_down": w_down.clone().requires_grad_(True),
                "w_up": w_up.clone().requires_grad_(True),
                "w_write": w_write.clone().requires_grad_(True)})
            self.gains.append(branch_gains(make_gain(seed + 30 + index), 4, asymmetric)
                              .clone().requires_grad_(True))
        self.states = None

    def _all(self):
        return [p for entry in self.parameters for p in entry.values()] + self.gains

    def run(self, steps, lr):
        optimizer = torch.optim.AdamW(self._all(), lr=lr)
        for _ in range(steps):
            optimizer.zero_grad()
            states = self.h.unsqueeze(-2).repeat(1, 4, 1)
            for entry, gains, sublayer in zip(self.parameters, self.gains,
                                              self.sublayers):
                states = gr_sublayer(states, gains, entry["w_down"], entry["w_up"],
                                     entry["w_write"], EPS, sublayer)
            loss = (collapse(states) - self.target).pow(2).mean()
            loss.backward()
            optimizer.step()
            self.states = states.detach()
        return float(loss)

    def write_row_gap(self):
        rows = self.parameters[0]["w_write"].detach()
        return float((rows[0] - rows[1]).abs().max())

    def branch_gap(self):
        return float((self.states[..., 0, :] - self.states[..., 1, :]).abs().max())

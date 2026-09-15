"""An independent float64 statement of the original sublayer and the GR route.

Deliberately imports nothing from ``distillkit``. Its purpose is to be a second opinion
about the algebra, so that "the conversion is exact" is a claim two separately written
implementations agree on rather than a property of one of them. Everything here is
float64 and written for clarity, not speed.

The recipient's sublayer::

    u  = RMSNorm(h; gamma, eps)
    y  = F(u)
    h' = h + y

The four-stream route::

    Z_i  = RMSNorm(R_i; gamma_i, eps)
    G    = sigmoid(W_up SiLU(W_down vec(Z) / n))
    x    = mean_i(G_i * Z_i)
    s    = 2 sigmoid(W_write vec(Z) / n)
    R_i' = R_i + s_i F(x)

With ``W_up = 0`` and ``W_write = 0`` the gates are exactly ``1/2`` and the write weights
exactly ``1``, so with ``gamma_i = 2 gamma`` the read collapses to the recipient's own
normalized input and every branch receives the recipient's own update. ``W_down`` is
seeded nonzero on purpose: zeroing both factors of the read bottleneck would leave
``W_down`` with no gradient path once ``W_up`` starts to move.

The asymmetric variant uses ``gamma_i = 2 gamma (1 + eps_i)`` with ``sum(eps_i) = 0``.
The read is the mean over branches, so while every branch still holds ``h`` the mean gain
is still ``2 gamma`` and the read is unchanged in exact arithmetic -- but the branches
carry different gains, which is what allows their gradients to differ.
"""

from __future__ import annotations

import torch

#: One fixed deterministic perturbation. An engineering choice, not an optimum, and not
#: swept in this task.
EPSILON = torch.tensor([-3.0, -1.0, 1.0, 3.0]) / 128.0


def rms_norm(x: torch.Tensor, gain: torch.Tensor, eps: float) -> torch.Tensor:
    """``gain * x / sqrt(mean(x^2) + eps)``, in whatever precision ``x`` carries."""
    variance = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(variance + eps) * gain


def original_sublayer(h, gain, eps, sublayer):
    """``h + F(RMSNorm(h; gamma, eps))``."""
    return h + sublayer(rms_norm(h, gain, eps))


def branch_gains(gain: torch.Tensor, num_branches: int = 4, asymmetric: bool = False):
    """``2 gamma`` per branch, optionally perturbed so the branches differ.

    The perturbation is mean-zero, so ``mean_i(gamma_i) == 2 gamma`` exactly and the
    initial read is unchanged.
    """
    base = 2.0 * gain.unsqueeze(0).repeat(num_branches, 1)
    if not asymmetric:
        return base
    if num_branches != EPSILON.numel():
        raise ValueError("the fixed perturbation is defined for %d branches, got %d"
                         % (EPSILON.numel(), num_branches))
    return base * (1.0 + EPSILON.to(base.dtype).unsqueeze(-1))


def gr_sublayer(states, gains, w_down, w_up, w_write, eps, sublayer):
    """One GR sublayer. ``states`` is ``[..., branches, hidden]``."""
    num_branches = states.shape[-2]
    normalized = torch.stack(
        [rms_norm(states[..., i, :], gains[i], eps) for i in range(num_branches)], dim=-2)
    flattened = normalized.flatten(-2)
    logits = torch.nn.functional.silu(flattened @ w_down.T / num_branches) @ w_up.T
    gate = torch.sigmoid(logits.unflatten(-1, (num_branches, states.shape[-1])))
    read = (gate * normalized).mean(dim=-2)
    write = 2.0 * torch.sigmoid(flattened @ w_write.T / num_branches)
    update = sublayer(read)
    return states + write.unsqueeze(-1) * update.unsqueeze(-2)


def collapse(states: torch.Tensor) -> torch.Tensor:
    """Mean over branches, expressed around branch zero so identity is exact."""
    reference = states[..., 0, :]
    return reference + (states - reference.unsqueeze(-2)).mean(dim=-2)


def initial_parameters(hidden, num_branches=4, lowrank=16, seed=0, dtype=torch.float64):
    """``W_up = 0``, ``W_write = 0``, ``W_down`` seeded nonzero."""
    generator = torch.Generator().manual_seed(seed)
    w_down = torch.randn(lowrank, num_branches * hidden, generator=generator,
                         dtype=torch.float64).to(dtype) / hidden ** 0.5
    w_up = torch.zeros(num_branches * hidden, lowrank, dtype=dtype)
    w_write = torch.zeros(num_branches, num_branches * hidden, dtype=dtype)
    return w_down, w_up, w_write


def converted_stack(h, gains, sublayers, eps, asymmetric=False, lowrank=16, seed=0):
    """Run consecutive GR sublayers from a duplicated embedding and collapse at the end."""
    hidden = h.shape[-1]
    states = h.unsqueeze(-2).repeat(*([1] * (h.dim() - 1)), 4, 1)
    for index, (gain, sublayer) in enumerate(zip(gains, sublayers)):
        w_down, w_up, w_write = initial_parameters(hidden, 4, lowrank, seed + index,
                                                   h.dtype)
        states = gr_sublayer(states, branch_gains(gain, 4, asymmetric), w_down, w_up,
                             w_write, eps, sublayer)
    return collapse(states)


def original_stack(h, gains, sublayers, eps):
    for gain, sublayer in zip(gains, sublayers):
        h = original_sublayer(h, gain, eps, sublayer)
    return h

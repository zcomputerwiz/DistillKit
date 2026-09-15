"""When do the four streams first become distinguishable, and in what order?

The earlier account was too coarse. Saying "the read has no branch-state gradient at
zero ``W_up``" is wrong: only the *gate logits* lose their input derivative there, because
they are produced through ``W_up``. The normalized value path ``mean_i(G_i Z_i)`` still
differentiates with respect to every branch state, with ``G_i`` held at a constant one
half. So gradient does arrive at the branch inputs from step zero; what is absent is any
*difference* between branches.

This separates three events that the earlier diagnostic pooled:

    1. gradients arriving at the branch inputs become unequal
    2. the read gates, or the ``W_up`` rows that produce them, become unequal
    3. the write rows and the residual states become unequal

The hypothesis worth checking is that (1) can precede (2): ``W_down`` is a shared random
matrix whose columns differ per branch slot, so once ``W_up`` is nonzero the gradient
flowing back through the bottleneck can already differ between branch inputs even while
every branch still produces the same gate value. This is an explanation to verify, not an
established result -- one short run, and if attribution stays ambiguous the report says so.

    python scratch/gr_retrofit/attribution.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from reference import branch_gains, collapse, gr_sublayer, initial_parameters

HIDDEN = 12
EPS = 1e-6
LAYERS = 3
ZERO = 1e-12


def sublayer(seed, hidden=HIDDEN):
    generator = torch.Generator().manual_seed(seed)
    first = torch.randn(hidden, hidden, generator=generator, dtype=torch.float64) / 3
    second = torch.randn(hidden, hidden, generator=generator, dtype=torch.float64) / 3
    return lambda u: torch.tanh(u @ first) @ second


def state(seed, batch=3, hidden=HIDDEN, scale=4.0):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(batch, hidden, generator=generator, dtype=torch.float64) * scale


def gain(seed, hidden=HIDDEN):
    generator = torch.Generator().manual_seed(seed)
    return 1.0 + 0.3 * torch.randn(hidden, generator=generator, dtype=torch.float64)


def instrumented(states, gains, w_down, w_up, w_write, eps, function):
    """One GR sublayer that also returns the per-branch gate values."""
    branches = states.shape[-2]
    normalized = torch.stack([
        states[..., i, :] * torch.rsqrt(states[..., i, :].pow(2).mean(-1, keepdim=True)
                                        + eps) * gains[i]
        for i in range(branches)], dim=-2)
    flattened = normalized.flatten(-2)
    logits = torch.nn.functional.silu(flattened @ w_down.T / branches) @ w_up.T
    gate = torch.sigmoid(logits.unflatten(-1, (branches, states.shape[-1])))
    read = (gate * normalized).mean(dim=-2)
    write = 2.0 * torch.sigmoid(flattened @ w_write.T / branches)
    return states + write.unsqueeze(-1) * function(read).unsqueeze(-2), gate


def main() -> int:
    target = state(201)
    functions = [sublayer(210 + i) for i in range(LAYERS)]
    history = {}

    for asymmetric in (False, True):
        parameters, gains = [], []
        for index in range(LAYERS):
            w_down, w_up, w_write = initial_parameters(HIDDEN, lowrank=8,
                                                       seed=220 + index)
            parameters.append({"w_down": w_down.clone().requires_grad_(True),
                               "w_up": w_up.clone().requires_grad_(True),
                               "w_write": w_write.clone().requires_grad_(True)})
            gains.append(branch_gains(gain(230 + index), 4, asymmetric)
                         .clone().requires_grad_(True))
        everything = [p for entry in parameters for p in entry.values()] + gains
        optimizer = torch.optim.AdamW(everything, lr=5e-2)
        rows = []
        print("--- %s ---" % ("asymmetric" if asymmetric else "symmetric"))
        print("%4s %14s %14s %14s %14s %14s"
              % ("step", "d(branch in)", "W_up rows", "gate values", "write rows",
                 "branch states"))

        for step in range(9):
            optimizer.zero_grad()
            entry_states = state(200).unsqueeze(-2).repeat(1, 4, 1)
            entry_states.requires_grad_(True)
            states, gate = entry_states, None
            for layer, (block, layer_gains, function) in enumerate(
                    zip(parameters, gains, functions)):
                states, produced = instrumented(states, layer_gains, block["w_down"],
                                                block["w_up"], block["w_write"], EPS,
                                                function)
                if layer == 0:
                    gate = produced
            loss = (collapse(states) - target).pow(2).mean()
            loss.backward()

            # (1) gradient arriving at the branch inputs of the first sublayer
            incoming = entry_states.grad
            branch_input = float((incoming[..., 0, :] - incoming[..., 1, :]).abs().max())
            # (2) the rows that produce the gates, and the gate values themselves
            up = parameters[0]["w_up"].detach().unflatten(0, (4, HIDDEN))
            up_rows = float((up[0] - up[1]).abs().max())
            gate_gap = float((gate.detach()[..., 0, :]
                              - gate.detach()[..., 1, :]).abs().max())
            # (3) writes and states
            write = parameters[0]["w_write"].detach()
            write_rows = float((write[0] - write[1]).abs().max())
            branch_states = float((states.detach()[..., 0, :]
                                   - states.detach()[..., 1, :]).abs().max())
            rows.append({"step": step, "loss": float(loss),
                         "branch_input_grad_gap": branch_input,
                         "w_up_row_gap": up_rows, "gate_value_gap": gate_gap,
                         "write_row_gap": write_rows,
                         "branch_state_gap": branch_states})
            print("%4d %14.3e %14.3e %14.3e %14.3e %14.3e"
                  % (step, branch_input, up_rows, gate_gap, write_rows, branch_states))
            optimizer.step()
        history["asymmetric" if asymmetric else "symmetric"] = rows

    def onset(rows, key):
        for row in rows:
            if row[key] > ZERO:
                return row["step"]
        return None

    summary = {}
    for name, rows in history.items():
        summary[name] = {key: onset(rows, key) for key in
                         ("branch_input_grad_gap", "w_up_row_gap", "gate_value_gap",
                          "write_row_gap", "branch_state_gap")}
    print("\nfirst step at which each quantity becomes unequal (None = never, within %d steps):"
          % len(history["symmetric"]))
    print(json.dumps(summary, indent=2))

    output = Path("scratch/gr_retrofit/attribution.json")
    output.write_text(json.dumps({"onset": summary, "history": history}, indent=2),
                      encoding="utf-8")
    print("wrote %s" % output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

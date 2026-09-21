"""Calibrate the conversion from second moments instead of stored samples.

`source_capture` keeps every calibration token's block input, keys, values and rotary key
in float64 and concatenates them. At the conversion's 1024-token windows that is about
1.6 GiB per full-attention layer, six layers at once, which is why the conversion record
says `calibration_tokens = 32768`: the ceiling is memory, not a judgement that 32 windows
of text are enough to fit a 1.9B model's attention.

Nothing in the fit needs the samples. Every quantity it computes is a second moment:

    whiten(inputs^T inputs)                 G_ii    [hidden, hidden]
    solve(inputs, rotary)                   G_ii, C_ir
    svd((stacked^T inputs) @ whitener)      C_ti, and C_ri for each reader
    solve(latent, target)                   G_ll, C_lt
    the key and value r2                    G_ll, C_lt, and two scalars

All of those are fixed-size in the number of tokens. Accumulating them instead of the
samples makes the calibration set as large as one cares to stream, at constant memory --
33 MiB for the Gram of a 2048-wide input, against 537 MiB for 32,768 of its rows.

Why it is worth lifting. The latent ladder measured 256/384/512/768 at 1.5510/1.4817/
1.4628/1.4716 nats: it improves to 512 and then gets *worse* with more room. A bottleneck
that is genuinely too narrow does not do that. A least-squares solve with more parameters
than its sample supports does exactly that, and 32,768 rows against a 768-wide latent
fitting 4,096 outputs is that situation.
"""

from __future__ import annotations

import torch


class LayerMoments:
    """Second moments for one full-attention layer's fit.

    Every accumulator is allocated on first use, because their widths come from the
    tensors rather than from the config, and float64 throughout: these are sums over
    hundreds of thousands of rows feeding a solve whose conditioning is the whole reason
    the sample path used float64 too.
    """

    def __init__(self, device):
        self.device = torch.device(device)
        self.rows = 0
        self.gram_inputs = None      # inputs^T inputs           [hidden, hidden]
        self.cross_target = None     # target^T inputs           [target, hidden]
        self.cross_rotary = None     # inputs^T rotary           [hidden, rope]
        self.cross_readers = {}      # reader target^T inputs    [reader target, hidden]
        self.key_energy = 0.0        # ||keys||^2
        self.value_energy = 0.0      # ||values||^2
        self.rotary_energy = 0.0     # ||rotary||^2
        self.gram_latent = None      # latent^T latent           [latent, latent]
        self.cross_latent = None     # latent^T target           [latent, target]

    def _add(self, name, value):
        held = getattr(self, name)
        if held is None:
            setattr(self, name, value)
        else:
            held += value

    def observe(self, inputs, target, rotary, keys, values):
        """One window's contribution to everything the first phase needs."""
        self.rows += inputs.shape[0]
        self._add("gram_inputs", inputs.T @ inputs)
        self._add("cross_target", target.T @ inputs)
        self._add("cross_rotary", inputs.T @ rotary)
        self.key_energy += float(keys.pow(2).sum())
        self.value_energy += float(values.pow(2).sum())
        self.rotary_energy += float(rotary.pow(2).sum())

    def observe_reader(self, index, reader_target, inputs):
        """A reader's keys and values against *this* layer's inputs.

        The donor's encoder is the best rank-`latent` summary of what every layer reading
        its latent wants, so a reader's target enters the donor's SVD -- against the
        donor's own block input, because that is what the encoder is a function of.
        """
        held = self.cross_readers.get(index)
        value = reader_target.T @ inputs
        self.cross_readers[index] = value if held is None else held + value

    def observe_latent(self, latent, target):
        """The second phase, once the encoder exists and the latent can be read."""
        self._add("gram_latent", latent.T @ latent)
        self._add("cross_latent", latent.T @ target)

    def stacked_cross(self, order):
        """`stacked^T inputs` for the SVD, in the donor-then-readers order the fit uses."""
        parts = [self.cross_target] + [self.cross_readers[i] for i in order]
        return torch.cat(parts, dim=0) if len(parts) > 1 else parts[0]


def solve_from_moments(gram, cross, ridge):
    """`solve(source, target)` written against moments rather than samples.

    `solve` forms `source^T source` and `source^T target` and does exactly this; passing
    the sums in is the same arithmetic on the same numbers.
    """
    stabilised = gram + torch.eye(gram.shape[0], dtype=gram.dtype,
                                  device=gram.device) * (
        gram.diagonal().mean().clamp_min(1e-30) * ridge)
    return torch.linalg.solve(stabilised, cross).T


def residual_share(weight, gram_latent, cross_latent, energy, columns):
    """`1 - ||latent @ W^T - target||^2 / ||target||^2` over a slice of the columns.

    The sample path scores this by materialising the fit. Expanded, the numerator is
    `tr(W G W^T) - 2 tr(W C) + ||target||^2`, and every term is a moment. `columns` picks
    the key half or the value half out of the interleaved target.
    """
    piece = weight[columns]
    cross = cross_latent[:, columns]
    fitted_energy = float((piece @ gram_latent * piece).sum())
    agreement = float((piece.T * cross).sum())
    return float(1 - (fitted_energy - 2 * agreement + energy) / energy)


def interleaved_columns(heads, content, head_dim):
    """Which columns of the interleaved target are keys and which are values.

    `kv_b_proj` is read per head -- `view(..., heads, content + head_dim)` then split --
    so the target interleaves each head's content key with its own value. Scoring the two
    halves means selecting their columns out of that interleave rather than slicing it in
    two, which is the same mistake the layout bug was.
    """
    width = content + head_dim
    keys, values = [], []
    for head in range(heads):
        base = head * width
        keys.extend(range(base, base + content))
        values.extend(range(base + content, base + width))
    return torch.tensor(keys), torch.tensor(values)

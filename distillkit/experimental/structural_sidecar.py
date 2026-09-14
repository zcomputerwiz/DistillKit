"""A post-hoc structural prior: hashed local context biasing structural logits only.

The PLE programme closed negative as semantic memory. A backbone trained beside a
268.7M-row n-gram table gained nothing in content over a backbone trained without one,
and 79% of what the table did buy was layout. What survived that is narrower and was
never properly tested on its own:

    hashed local context may be a cheap prior for *structural* prediction

This module tests exactly that claim and nothing wider. Three things keep it honest.

**It cannot touch content.** The correction is added to the logits of structural tokens
only -- newline, other whitespace, punctuation, control -- which is 6,166 of 248,320
ids. Content logits are untouched by construction, so a content change can only come
through the softmax denominator, and the loss constrains that explicitly.

**It is fitted post hoc.** The residual-gate work established that a backbone trained
beside an auxiliary correction learns to lean on it without exploiting it, and that the
same correction attached to a normally trained backbone is worth more. So the backbone is
frozen here and never sees a gradient.

**Learned rows are optional.** ``mode`` selects what the hash addresses:

``table``   a learned row per bucket, which is the PLE mechanism shrunk to a structural
            correction
``fixed``   a deterministic seeded random code per bucket, frozen, with only the decoder
            trainable -- if this matches ``table`` then local context *identity* is the
            signal and the learned rows were never the point
``direct``  the same idea with the table removed: signed features sliced straight out of
            one mixed hash word, so nothing per-row is stored or allocated at all
``none``    one learned code for every position, no addressing at all, which is the
            control that says whether hashing contributes anything over a generic
            structural bias

The hash is the historical one, ``NGramHasher``, EOS reset and all, so a result here is
directly comparable to the earlier work rather than to a convenient reimplementation.

``strength`` is a scalar with ``0`` meaning exactly the stock model. It is calibrated
after fitting, on a split that is neither the training data nor the corpora the result is
reported on.
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["StructuralSidecar", "apply_structural_bias", "structural_token_ids",
           "wrong_context_rows", "signed_hash_features", "splitmix64_tensor", "MODES"]

MODES = ("table", "fixed", "direct", "none")

# splitmix64's constants, taken from the hasher this module addresses through. Reusing
# that finalizer rather than inventing a mixer keeps one well-understood function in the
# codebase and one set of tests pinning it.
_GAMMA = 0x9E3779B97F4A7C15
_M1 = 0xBF58476D1CE4E5B9
_M2 = 0x94D049BB133111EB


def _logical_shift(value, bits: int):
    """``value >> bits`` as if the int64 were unsigned.

    torch has no unsigned 64-bit type and ``>>`` on int64 is arithmetic, so a negative
    value -- which every well-mixed hash word is, about half the time -- would shift ones
    in from the top and quietly break the mixer. Masking to the bits that should survive
    restores the unsigned meaning exactly.
    """
    return (value >> bits) & ((1 << (64 - bits)) - 1)


def splitmix64_tensor(value):
    """The scalar ``splitmix64`` from the hasher, vectorised over int64 tensors.

    int64 multiplication wraps in two's complement, which is the same bit pattern as the
    unsigned wrap the reference performs, so only the shifts need care. Bit-identity
    against the scalar reference is pinned by the tests rather than argued for here.
    """
    value = value + _GAMMA
    value = (value ^ _logical_shift(value, 30)) * _M1
    value = (value ^ _logical_shift(value, 27)) * _M2
    return value ^ _logical_shift(value, 31)


def signed_hash_features(rows, dim: int, seed: int):
    """``+-1 / sqrt(dim)`` features derived from a row id, with no table anywhere.

    One mix per row, then one bit per feature. That is the cheapest construction that
    could work, and the previous experiment is exactly why it is worth trying first: what
    mattered there was deterministic context *identity*, not the particular random
    vectors a stored basis happened to hold. If a bit-sliced hash word is a good enough
    basis, the table is machinery with nothing left to do.
    """
    if not 1 <= dim <= 32:
        raise ValueError("dim must be between 1 and 32; got %d" % dim)
    seeded = splitmix64_tensor(
        torch.tensor(seed, dtype=torch.int64, device=rows.device))
    mixed = splitmix64_tensor(rows.to(torch.int64) ^ seeded)
    shifts = torch.arange(dim, device=rows.device, dtype=torch.int64)
    # Arithmetic shift is harmless here: only bit 0 of each shifted word is read.
    bits = (mixed.unsqueeze(-1) >> shifts) & 1
    return (bits.to(torch.float32) * 2.0 - 1.0) / dim ** 0.5


def structural_token_ids(classes, device=None) -> torch.Tensor:
    """Every token id the sidecar is allowed to move, from the existing class splitter."""
    ids = [index for index, label in enumerate(classes) if label != "content"]
    if not ids:
        raise ValueError("no structural tokens; the class splitter returned content only")
    return torch.tensor(ids, dtype=torch.long, device=device)


def wrong_context_rows(rows: torch.Tensor, shift: int = 7) -> torch.Tensor:
    """The same rows, addressed from the wrong position.

    The control the native-table screens used: same text, same table, addresses rolled
    along the sequence. A gain that survives this was never about the context.
    """
    if shift == 0:
        raise ValueError("a wrong-context control has to move the addresses")
    return rows.roll(shifts=shift, dims=1)


class StructuralSidecar(nn.Module):
    """``rows -> bias`` over structural logits; exactly zero at initialization.

    The decoder's output layer starts at zero, so a freshly built sidecar adds nothing
    whatever ``strength`` says, and the model it is attached to is bitwise the stock
    model. That is the same construction the residual gate uses, for the same reason:
    identity has to be exact, or every delta is measured against an unknown baseline.
    """

    def __init__(self, rows: int, code_dim: int, structural: int, mode: str = "fixed",
                 heads: int = 2, hidden: int = 0, seed: int = 20260913) -> None:
        super().__init__()
        if mode not in MODES:
            raise ValueError("unknown mode %r; expected one of %s"
                             % (mode, ", ".join(MODES)))
        if rows < 1 or code_dim < 1 or structural < 1 or heads < 1:
            raise ValueError("rows, code_dim, structural and heads must all be positive")
        self.mode = mode
        self.rows = rows
        self.code_dim = code_dim
        self.heads = heads
        self.structural = structural

        generator = torch.Generator().manual_seed(seed)
        codes = torch.randn(rows, code_dim, generator=generator) / code_dim ** 0.5
        if mode == "table":
            # Learned rows, initialised from the same draw the fixed arm freezes, so the
            # two arms differ in what trains rather than in where they start.
            self.codes = nn.Parameter(codes)
        elif mode == "direct":
            # No table at all: the code is computed from the row id where it is used.
            self.codes = None
        elif mode == "fixed":
            # Not persistent: this is a seeded draw, reconstructed exactly by __init__
            # from (rows, code_dim, seed), and a checkpoint that stores it is 210 MB of
            # numbers the constructor already knows. Keeping it out of the state dict is
            # the claim the fixed arm exists to make.
            self.register_buffer("codes", codes, persistent=False)
        else:
            # No addressing: one code for every position. Matched decoder capacity, no
            # local context identity, which is what makes it the right control.
            self.codes = None
            self.constant = nn.Parameter(torch.zeros(code_dim * heads))

        width = code_dim * heads
        layers: list[nn.Module] = []
        if hidden:
            layers += [nn.Linear(width, hidden), nn.Tanh()]
            width = hidden
        output = nn.Linear(width, structural)
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)
        layers.append(output)
        self.decoder = nn.Sequential(*layers)
        self.seed = seed

    @property
    def is_addressed(self) -> bool:
        return self.mode != "none"

    def code(self, rows: torch.Tensor) -> torch.Tensor:
        """``[batch, seq, heads]`` of row indices to ``[batch, seq, heads * code_dim]``."""
        if not self.is_addressed:
            shape = rows.shape[:2] + (self.constant.numel(),)
            return self.constant.expand(shape)
        if rows.shape[-1] != self.heads:
            raise ValueError("expected %d heads of row indices, got %d"
                             % (self.heads, rows.shape[-1]))
        if self.mode == "direct":
            gathered = signed_hash_features(rows, self.code_dim, self.seed)
            return gathered.reshape(rows.shape[:2] + (self.heads * self.code_dim,))
        gathered = self.codes[rows.clamp(0, self.rows - 1)]
        return gathered.reshape(rows.shape[:2] + (self.heads * self.code_dim,))

    def forward(self, rows: torch.Tensor) -> torch.Tensor:
        """``[batch, seq, structural]`` of logit corrections."""
        return self.decoder(self.code(rows).to(self.decoder[-1].weight.dtype))

    def parameter_report(self) -> dict:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(b.numel() for b in self.buffers())
        return {"mode": self.mode, "rows": self.rows, "code_dim": self.code_dim,
                "heads": self.heads, "structural": self.structural,
                "trainable_parameters": trainable, "frozen_code_entries": frozen,
                "bf16_bytes": 2 * (trainable + frozen)}


def apply_structural_bias(logits: torch.Tensor, bias: torch.Tensor,
                          structural: torch.Tensor, strength: float) -> torch.Tensor:
    """``z' = z + strength * bias`` on structural columns, everything else untouched.

    ``strength = 0`` returns the logits unchanged -- the same tensor, not a numerically
    equal copy -- because the ablation has to be exact rather than nearly exact.
    """
    if strength == 0.0:
        return logits
    if bias.shape[:2] != logits.shape[:2]:
        raise ValueError("bias %s does not match logits %s"
                         % (tuple(bias.shape[:2]), tuple(logits.shape[:2])))
    if bias.shape[-1] != structural.numel():
        raise ValueError("bias has %d structural columns but %d ids were given"
                         % (bias.shape[-1], structural.numel()))
    out = logits.clone()
    return out.index_add_(-1, structural.to(out.device),
                          (strength * bias).to(out.dtype))

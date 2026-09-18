"""Carry a structural logit bias as extra latent dimensions on the head.

`FactorizedSidecar` produces a bias over structural columns, which means materializing
logits, which means abandoning Cut Cross-Entropy -- measured at 113k tok/s down to 59k.
But the bias is linear in a code the sidecar already computes, so it can be folded into
the head instead::

    logits = W h + S c      with S zero on every content row

That is a plain linear map from a wider hidden state, which is exactly the shape CCE
consumes, and the "cannot touch content" guarantee stops being a masked `index_add` and
becomes a property of the weight matrix's sparsity.

The latent layout is ``[code | admission | 1]``:

* **code** -- ``heads * code_dim`` dimensions, the addressed branch's hashed context
* **admission** -- one dimension, the whitespace gate's ``2 sigmoid(w . c + b)``, so the
  rank-one product ``a(x) * b_whitespace`` becomes one column of ``S``
* **1** -- a constant, carrying the decoder's bias term

Nothing here is an approximation. `_self_check` asserts the assembled head reproduces
`FactorizedSidecar.forward` to bf16 rounding on random input, because a "mathematically
equivalent" refactor of a trained module is exactly the kind of claim that is wrong in
practice.
"""

from __future__ import annotations

import torch
from torch import nn


class AugmentedHead(nn.Module):
    """``(code, S)`` assembled from a `FactorizedSidecar`, for a widened `lm_head`."""

    #: Cut Cross-Entropy's kernel wants the head's width divisible by 16, and the cost of
    #: missing it is a cliff rather than a slope. Measured at batch 64 x 1024, vocab
    #: 32,768, against width 512: 576 costs 1.11x, 592 costs 1.20x, 608 costs 1.19x --
    #: and 578 costs **6.18x**, 600 costs 6.15x. 578 and 600 are the two in that list not
    #: divisible by 16. So the latent block is padded with zero columns, which cost a few
    #: percent of head FLOPs and nothing in meaning.
    ALIGNMENT = 16

    def __init__(self, sidecar, structural: torch.Tensor, vocab: int,
                 hidden: int = 0) -> None:
        super().__init__()
        self.sidecar = sidecar
        self.vocab = vocab
        self.register_buffer("structural", structural.clone())
        self.code_width = sidecar.addressed.heads * sidecar.addressed.code_dim
        self.used = self.code_width + 2
        total = hidden + self.used
        padding = (-total) % self.ALIGNMENT
        self.width = self.used + padding

    def code_for(self, rows: torch.Tensor) -> torch.Tensor:
        """``[batch, seq, width]`` of latent dimensions to append to the hidden state.

        The gate reads the trigram half of the same code, so it is applied here rather
        than through `FactorizedSidecar.admission`, which would gather the hash a second
        time -- the gather is the expensive part of this whole branch.
        """
        code = self.sidecar.addressed.code(rows)
        admission = self.sidecar.gate(code[..., self.sidecar.addressed.code_dim:])
        ones = torch.ones_like(admission)
        parts = [code, admission.unsqueeze(-1).to(code.dtype),
                 ones.unsqueeze(-1).to(code.dtype)]
        if self.width > self.used:
            parts.append(code.new_zeros(code.shape[:-1] + (self.width - self.used,)))
        return torch.cat(parts, dim=-1)

    def s_matrix(self, dtype) -> torch.Tensor:
        """``[vocab, width]``, zero on content rows by construction."""
        decoder = self.sidecar.addressed.decoder[-1]
        keep = self.sidecar.keep.unsqueeze(-1)

        # Slot-indexed first: rows are structural slots, matching `keep` and `slots`.
        slots = torch.zeros(self.sidecar.keep.numel(), self.width,
                            device=decoder.weight.device, dtype=torch.float32)
        slots[:, :self.code_width] = decoder.weight.float() * keep.float()
        slots[:, self.code_width + 1] = decoder.bias.float() * self.sidecar.keep.float()
        slots[self.sidecar.slots, self.code_width] = self.sidecar.white().float()

        full = torch.zeros(self.vocab, self.width, device=slots.device,
                           dtype=torch.float32)
        full = full.index_copy(0, self.structural, slots)
        return full.to(dtype)

    def wide_head(self, head_weight: torch.Tensor) -> torch.Tensor:
        return torch.cat([head_weight, self.s_matrix(head_weight.dtype)], dim=1)


def _self_check():
    """The assembled head must reproduce the module it replaces, on random input."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from distillkit.experimental.structural_sidecar import (FactorizedSidecar,
                                                            StructuralSidecar)

    torch.manual_seed(0)
    vocab, structural_n, whitespace_n, code_dim, heads = 4096, 300, 40, 16, 2
    rows_n = 1 << 12

    structural = torch.randperm(vocab)[:structural_n].sort().values
    whitespace = structural[torch.randperm(structural_n)[:whitespace_n].sort().values]
    addressed = StructuralSidecar(rows=rows_n, code_dim=code_dim, structural=structural_n,
                                  mode="fixed", heads=heads, seed=7)
    sidecar = FactorizedSidecar(addressed, structural, whitespace, gated=True)

    # Train the parameters away from zero, or the check passes on a module of zeros.
    with torch.no_grad():
        addressed.decoder[-1].weight.normal_(std=0.05)
        addressed.decoder[-1].bias.normal_(std=0.05)
        sidecar.white.bias.normal_(std=0.1)
        sidecar.gate.weight.normal_(std=0.3)
        sidecar.gate.bias.normal_(std=0.1)

    batch, seq = 3, 17
    rows = torch.randint(0, rows_n, (batch, seq, heads))
    hidden = torch.randn(batch, seq, 64)
    head_weight = torch.randn(vocab, 64) / 8.0

    reference = torch.zeros(batch, seq, vocab)
    reference.index_add_(-1, structural, sidecar(rows))
    reference = reference + hidden @ head_weight.T

    augmented = AugmentedHead(sidecar, structural, vocab)
    wide = torch.cat([hidden, augmented.code_for(rows)], dim=-1)
    produced = wide @ augmented.wide_head(head_weight).T

    delta = (reference - produced).abs().max().item()
    assert delta < 1e-4, delta

    s = augmented.s_matrix(torch.float32)
    content = torch.ones(vocab, dtype=torch.bool)
    content[structural] = False
    assert s[content].abs().sum() == 0, "S is nonzero on a content row"
    assert s.abs().sum() > 0, "S is entirely zero; the check proves nothing"

    print("augmented_head self-check: ok (max |delta| %.2e over %d logits, "
          "S nonzero on %d of %d rows)"
          % (delta, reference.numel(), int((s.abs().sum(-1) > 0).sum()), vocab))


if __name__ == "__main__":
    _self_check()

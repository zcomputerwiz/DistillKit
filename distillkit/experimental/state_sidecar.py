"""Admit local-context information by compatibility with the current hidden state.

Every admission signal this programme has tried so far is a property of the *context* and
not of the model reading it. The residual gate conditioned on how familiar a trigram was;
the structural sidecar conditioned on which rows a trigram addressed. Both are answers to
"how reliable is this context in general", and both were absorbed by ordinary code
pretraining. This asks a different question -- "does this context match what the model is
currently computing" -- which is the one thing the backbone cannot answer for itself,
because it requires comparing the hidden state against a retrieved memory rather than
against the corpus.

The shape is a single-slot attention read, deliberately the smallest thing that could
express the hypothesis::

    c_t = signed hash bits of the local 3-gram        deterministic, no table
    m_t = W_m c_t                                     narrow memory representation
    q_t = W_q RMSNorm(h_t)                            what the model is currently asking
    k_t, v_t = W_k m_t, W_v m_t                       what the memory offers
    g_t = sigmoid(cos(q_t, k_t) / temperature + b)    compatibility, not frequency
    h_t = h_t + W_o (g_t v_t)                         one injection point

``W_o`` is zero-initialized, so at step zero the module is exactly the identity and the
model is bitwise stock -- the same contract every retrofit in this repository starts from,
and the thing that makes "did it learn anything" a well-posed question.

Cosine similarity rather than a raw dot product is load-bearing. An unnormalized score
lets the module raise admission by growing ``||q||`` or ``||k||``, which is a way to learn
"always admit" while looking like it learned a policy; normalizing forces the gate to be
about *direction* -- agreement between the state and the memory -- which is the hypothesis
under test.

:attr:`state_conditioned` switches the gate off, holding ``g = 1``. That is the matched
control: identical hash path, identical value path, identical data order, no compatibility
term. Setting it on a *trained* module instead gives the parameter-matched ablation, which
answers the different question of whether the gate is doing the work or the extra
parameters are.
"""

from __future__ import annotations

import contextlib
import math

import torch
import torch.nn as nn

from distillkit.experimental.structural_sidecar import signed_hash_features

__all__ = [
    "StateConditionedSidecar",
    "StateSidecarHandle",
    "install_state_sidecar",
    "remove_state_sidecar",
]


class StateConditionedSidecar(nn.Module):
    """One narrow memory slot, admitted by state compatibility."""

    def __init__(self, hidden_size: int, code_dim: int = 32, memory_dim: int = 64,
                 seed: int = 20260915, temperature: float = 1.0, heads: int = 2) -> None:
        super().__init__()
        if not 1 <= code_dim <= 32:
            raise ValueError("code_dim must be between 1 and 32; got %d" % code_dim)
        self.hidden_size = hidden_size
        self.code_dim = code_dim
        self.memory_dim = memory_dim
        self.seed = seed
        # The canonical 3-gram hasher emits ``heads`` row ids per position, and every
        # addressed module in this repository flattens them into one code rather than
        # discarding a head -- see StructuralSidecar.code. Matching that keeps this
        # module's addressing identical to the structural sidecar it is being compared
        # against, which is worth more than a code width that reads as round.
        self.heads = heads
        self.memory = nn.Linear(heads * code_dim, memory_dim, bias=False)
        self.query = nn.Linear(hidden_size, memory_dim, bias=False)
        self.key = nn.Linear(memory_dim, memory_dim, bias=False)
        self.value = nn.Linear(memory_dim, memory_dim, bias=False)
        self.out = nn.Linear(memory_dim, hidden_size, bias=False)
        self.scale = nn.Parameter(torch.ones(hidden_size))
        self.temperature = nn.Parameter(torch.tensor(float(temperature)))
        self.score_bias = nn.Parameter(torch.zeros(()))
        # Exact identity at initialization: the output projection is the only path from
        # this module into the residual stream, so zeroing it makes the model stock.
        nn.init.zeros_(self.out.weight)
        self.state_conditioned = True
        self.wrong_context = False
        #: Populated during a forward when a caller asks for diagnostics.
        self.last = {}

    @property
    def gate_parameters(self) -> int:
        """Parameters that exist only to compute the compatibility gate."""
        return (self.query.weight.numel() + self.key.weight.numel()
                + self.scale.numel() + self.temperature.numel()
                + self.score_bias.numel())

    def parameter_report(self) -> dict:
        return {
            "total": sum(p.numel() for p in self.parameters()),
            "memory": self.memory.weight.numel(),
            "query": self.query.weight.numel(),
            "key": self.key.weight.numel(),
            "value": self.value.weight.numel(),
            "out": self.out.weight.numel(),
            "norm_scale": self.scale.numel(),
            "temperature_and_bias": 2,
            "gate_only": self.gate_parameters,
            "code_dim": self.code_dim, "memory_dim": self.memory_dim,
            "heads": self.heads, "code_width": self.heads * self.code_dim,
        }

    def normalize(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """RMSNorm over the feature axis, in float32 regardless of the model's dtype."""
        values = hidden_states.to(torch.float32)
        variance = values.pow(2).mean(-1, keepdim=True)
        return values * torch.rsqrt(variance + 1e-6) * self.scale

    def forward(self, hidden_states: torch.Tensor, rows: torch.Tensor,
                collect: bool = False) -> torch.Tensor:
        """The correction to add to the residual stream, ``[batch, sequence, hidden]``."""
        if rows.shape[-1] != self.heads:
            raise ValueError("expected %d heads of row indices, got %d"
                             % (self.heads, rows.shape[-1]))
        codes = signed_hash_features(rows, self.code_dim, self.seed)
        codes = codes.reshape(rows.shape[:2] + (self.heads * self.code_dim,))
        memory = self.memory(codes.to(torch.float32))
        values = self.value(memory)
        if self.state_conditioned:
            query = self.query(self.normalize(hidden_states))
            keys = self.key(memory)
            # Cosine compatibility: the gate must be about direction, not magnitude.
            compatibility = torch.nn.functional.cosine_similarity(query, keys, dim=-1)
            score = compatibility / self.temperature.clamp(min=1e-3) + self.score_bias
            gate = torch.sigmoid(score)
        else:
            compatibility = score = None
            gate = torch.ones(hidden_states.shape[:-1], dtype=torch.float32,
                              device=hidden_states.device)
        correction = self.out(gate.unsqueeze(-1) * values)
        if collect:
            self.last = {
                "gate": gate.detach(),
                "compatibility": None if compatibility is None else compatibility.detach(),
                "score": None if score is None else score.detach(),
                "correction_norm": correction.detach().float().norm(dim=-1),
                "residual_norm": hidden_states.detach().float().norm(dim=-1),
            }
        return correction.to(hidden_states.dtype)


class StateSidecarHandle:
    """Holds the per-forward hash rows and the hook that injects the correction."""

    def __init__(self, sidecar: StateConditionedSidecar, hasher, layer: int) -> None:
        self.sidecar = sidecar
        self.hasher = hasher
        self.layer = layer
        self.rows: torch.Tensor | None = None
        self.collect = False
        self.calls = 0

    def set_context(self, input_ids: torch.Tensor) -> None:
        """Compute the hash rows once per forward, not once per hook call."""
        rows = self.hasher.row_indices(input_ids)
        if self.sidecar.wrong_context:
            # The control: real rows read from the wrong position, so the module still
            # sees a genuine context, just not the one it is predicting from.
            rows = torch.roll(rows, shifts=7, dims=1)
        self.rows = rows

    def correction(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.rows is None:
            raise ValueError("no hash rows for this forward; call set_context first")
        rows = self.rows
        if rows.shape[1] != hidden_states.shape[1]:
            # A cached decode hands one position at a time; take the matching tail.
            rows = rows[:, -hidden_states.shape[1]:]
        self.calls += 1
        return self.sidecar(hidden_states, rows, collect=self.collect)


def install_state_sidecar(model: nn.Module, sidecar: StateConditionedSidecar, hasher,
                          layer: int) -> StateSidecarHandle:
    """Add the sidecar's correction to one decoder layer's output.

    One injection point, by construction: the hook is registered on a single layer and the
    module has no other path into the model.
    """
    inner = getattr(model, "model", model)
    if not hasattr(inner, "layers"):
        raise ValueError("expected a decoder model with a `layers` list")
    if not 0 <= layer < len(inner.layers):
        raise ValueError("layer %d outside the model's %d layers"
                         % (layer, len(inner.layers)))
    handle = StateSidecarHandle(sidecar, hasher, layer)

    def hook(module, inputs, output):
        if isinstance(output, tuple):
            hidden = output[0]
            return (hidden + handle.correction(hidden),) + tuple(output[1:])
        return output + handle.correction(output)

    handle._hook = inner.layers[layer].register_forward_hook(hook)
    model.state_sidecar = sidecar
    model.state_sidecar_handle = handle
    return handle


def remove_state_sidecar(model: nn.Module) -> None:
    handle = getattr(model, "state_sidecar_handle", None)
    if handle is not None:
        handle._hook.remove()
        del model.state_sidecar_handle
    if hasattr(model, "state_sidecar"):
        del model.state_sidecar


@contextlib.contextmanager
def forced_identity(sidecar: StateConditionedSidecar):
    """Evaluate a trained module with ``g = 1``: the parameter-matched ablation."""
    previous = sidecar.state_conditioned
    sidecar.state_conditioned = False
    try:
        yield sidecar
    finally:
        sidecar.state_conditioned = previous

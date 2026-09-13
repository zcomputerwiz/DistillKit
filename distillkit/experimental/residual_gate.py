"""A scalar gate on how much of an FFN's proposal reaches the residual stream.

The decoder admits every sublayer's proposal at unit strength::

    h = h + mlp(norm(h))

The attenuation study measured that this is the wrong strength in an identifiable
regime. On familiar contexts -- token trigrams the model has seen recur across
documents -- scaling layer 12's proposal down improved held-out content NLL by 0.0514
nats, monotonically in the scale factor, replicated on a disjoint corpus, and with the
sign flipping on rare contexts. That is a causal statement about a fixed hyperparameter
nobody chose: the admission strength is 1.0 because nobody wrote a different number.

This module asks whether the model can choose it::

    h = h + g * mlp(norm(h))        g learned, one scalar per token per gated layer

What is deliberately *not* here matters as much as what is. No per-channel gates, no
multiple branches, no widened stream, no cross-stream routing. A scalar is the smallest
change consistent with the measured mechanism, and if a scalar cannot pay for itself
there is no reason to believe a larger version of the same idea would.

Identity at initialization is exact, not approximate::

    g = 1 + span * tanh(W_out tanh(W_in x))     with W_out and its bias zero

The zero lives in the output projection rather than in a saturated gate bias, so the
starting model is bitwise the stock model *and* the gate sits where its gradient is
largest. ``span = 1`` puts the reachable range at (0, 2), which spans the whole alpha
curve the intervention study measured -- including the alpha = 0 endpoint that was best
-- without the parameterization ever being asked to produce a value it cannot reach.

Two conditioning families, and their union:

``familiarity``  cross-document trigram statistics, the signal the intervention study
                 identified. Two scalars: ``log1p(count)`` and the stored relative
                 residual variance.
``geometry``     what the layer can see about its own proposal without any context
                 memory: ``||h||``, ``||r||``, ``||r||/||h||`` and ``cos(h, r)``, taken
                 from the same tensors the diagnostics probe scored.

The combined family exists to answer whether familiarity carries anything the update
geometry does not already imply. Held-out AUCs for predicting *per-position* benefit
were 0.550, 0.546 and 0.566 -- all near chance -- so this is not a routing problem and
no per-token classification target appears anywhere here. The gate is trained on
cross-entropy and is expected to find a population-level admission policy, which is the
only thing the evidence supports.

Feature scales are calibrated once, before training, by a pass with the gate inert under
``no_grad``; the resulting normalizer is a constant that ships in the state dict, so
training and evaluation standardize identically. It is a separate pass rather than a
warm-up inside training on purpose: gradient checkpointing runs each gated forward twice
per step, and a normalizer that changed between those two runs would give the backward a
different function than the forward computed.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import logging
import math
import os

import numpy as np
import torch
from torch import nn
from transformers import TrainerCallback

__all__ = ["ResidualAdmissionGate", "ResidualGateHandle", "TrigramFamiliarity",
           "install_residual_gates", "remove_residual_gates", "residual_gates",
           "calibrate_gates", "attach_residual_gates",
           "load_gate_checkpoint",
           "ResidualGateCheckpointCallback", "FAMILIES", "family_features",
           "gate_parameter_count"]

LOG = logging.getLogger(__name__)

FAMILIARITY_FEATURES = ("log_count", "variance")
GEOMETRY_FEATURES = ("norm_h", "norm_r", "ratio", "cosine")
FAMILIES = {
    "familiarity": FAMILIARITY_FEATURES,
    "geometry": GEOMETRY_FEATURES,
    "combined": FAMILIARITY_FEATURES + GEOMETRY_FEATURES,
}


def family_features(family: str) -> tuple[str, ...]:
    if family not in FAMILIES:
        raise ValueError("unknown gate family %r; expected one of %s"
                         % (family, ", ".join(sorted(FAMILIES))))
    return FAMILIES[family]


class ResidualAdmissionGate(nn.Module):
    """``features -> g`` in ``(1 - span, 1 + span)``, exactly ``1`` at initialization.

    The hidden layer is randomly initialized and the output layer is zero. On the first
    optimizer step the output layer receives gradient (its input is nonzero) and the
    hidden layer does not (its gradient flows through the zero output weight); the hidden
    layer starts learning as soon as the output leaves zero. That is the price of exact
    identity and it is one step, not a permanent condition -- pinned by the tests.
    """

    def __init__(self, num_features: int, hidden: int = 16, span: float = 1.0) -> None:
        super().__init__()
        if num_features < 1:
            raise ValueError("a gate needs at least one feature")
        if not 0.0 < span <= 1.0:
            raise ValueError("span must be in (0, 1]; got %r" % span)
        self.num_features = num_features
        self.span = float(span)
        self.project = nn.Linear(num_features, hidden)
        self.output = nn.Linear(hidden, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        # Frozen after calibration; shipped in the state dict so evaluation standardizes
        # exactly as training did.
        self.register_buffer("feature_mean", torch.zeros(num_features))
        self.register_buffer("feature_std", torch.ones(num_features))
        self.register_buffer("calibration_sum", torch.zeros(num_features,
                                                            dtype=torch.float64))
        self.register_buffer("calibration_square", torch.zeros(num_features,
                                                               dtype=torch.float64))
        self.register_buffer("calibration_count", torch.zeros((), dtype=torch.float64))
        self.register_buffer("calibrated", torch.zeros((), dtype=torch.bool))

    @property
    def is_calibrated(self) -> bool:
        return bool(self.calibrated.item())

    def observe(self, features: torch.Tensor) -> None:
        """Accumulate feature statistics during the calibration pass."""
        flat = features.detach().reshape(-1, self.num_features).to(torch.float64)
        self.calibration_sum += flat.sum(0)
        self.calibration_square += (flat * flat).sum(0)
        self.calibration_count += flat.shape[0]

    def finalize(self) -> None:
        """Freeze the normalizer from what ``observe`` accumulated."""
        if float(self.calibration_count) <= 1:
            raise ValueError("nothing observed; the calibration pass ran no tokens")
        count = self.calibration_count.clamp(min=1.0)
        mean = self.calibration_sum / count
        variance = (self.calibration_square / count - mean * mean).clamp(min=0.0)
        self.feature_mean.copy_(mean.to(self.feature_mean.dtype))
        # A feature that never varies contributes nothing; a floor keeps it from
        # exploding rather than pretending it is informative.
        self.feature_std.copy_(variance.sqrt().clamp(min=1e-3).to(self.feature_std.dtype))
        self.calibrated.fill_(True)

    def gate_report(self, prefix: str = "") -> dict[str, float]:
        """How far the gate has moved off identity, for the training log.

        ``reach`` is the largest deviation this gate could produce for any input, which
        is the honest one-number answer to whether it has learned anything at all: it is
        zero at initialization by construction, so a run that keeps reporting zero is
        inert and should be diagnosed rather than extended.
        """
        weight = self.output.weight.detach().float()
        bias = self.output.bias.detach().float()
        # tanh is monotone and the hidden layer is bounded by tanh, so the extreme sits
        # at the corner where every hidden unit agrees with the sign of its weight.
        extreme = float(weight.abs().sum() + bias.abs().sum())
        return {prefix + "/output_weight_norm": float(weight.norm()),
                prefix + "/output_bias": float(bias.mean()),
                prefix + "/reach": self.span * math.tanh(extreme)}

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """``[..., num_features]`` of float32 in, ``[...]`` of float32 out."""
        if features.shape[-1] != self.num_features:
            raise ValueError("gate expects %d features, got %d"
                             % (self.num_features, features.shape[-1]))
        standard = (features - self.feature_mean) / self.feature_std
        hidden = torch.tanh(self.project(standard))
        return 1.0 + self.span * torch.tanh(self.output(hidden)).squeeze(-1)


class TrigramFamiliarity:
    """Cross-document trigram counts and residual variance, as per-token features.

    Reads the cache the memoisation study built. The key for position ``t`` is the exact
    trigram ``(t-2, t-1, t)`` -- the context whose FFN is being computed -- packed into
    one integer, so there are no hash collisions. Lookups are a sorted-array search on
    the model's own device: no Python loop over 4096 positions per step.

    A context the cache never saw gets ``count = 0`` and ``variance = 1``, the same
    defaults the intervention harness used, so 'unfamiliar' is a value the gate can read
    rather than a missing input.
    """

    def __init__(self, cache_path, vocab_size: int, device=None,
                 minimum_count: int = 0):
        data = np.load(cache_path)
        keys = data["keys"].astype(np.int64)
        counts = data["counts"].astype(np.float64)
        mean = data["mean"].astype(np.float32)
        energy = (mean.astype(np.float64) ** 2).sum(-1)
        relative = data["variance"] / np.maximum(energy + data["variance"], 1e-9)
        if minimum_count:
            keep = counts >= minimum_count
            keys, counts, relative = keys[keep], counts[keep], relative[keep]
        order = np.argsort(keys)
        self.vocab_size = int(vocab_size)
        self.keys = torch.as_tensor(keys[order], device=device)
        self.log_count = torch.as_tensor(np.log1p(counts[order]), dtype=torch.float32,
                                         device=device)
        self.variance = torch.as_tensor(relative[order], dtype=torch.float32,
                                        device=device)

    def to(self, device):
        self.keys = self.keys.to(device)
        self.log_count = self.log_count.to(device)
        self.variance = self.variance.to(device)
        return self

    def features(self, input_ids: torch.Tensor) -> torch.Tensor:
        """``[batch, sequence]`` of ids to ``[batch, sequence, 2]`` of float32."""
        if input_ids.dim() != 2:
            raise ValueError("expected [batch, sequence] ids, got %s"
                             % (tuple(input_ids.shape),))
        device = self.keys.device
        ids = input_ids.to(device=device, dtype=torch.int64)
        batch, length = ids.shape
        out = torch.zeros(batch, length, 2, dtype=torch.float32, device=device)
        # The first two positions have no trigram; they read as maximally unfamiliar,
        # which is what they are.
        out[:, :, 1] = 1.0
        if length < 3:
            return out
        vocab = self.vocab_size
        keys = (ids[:, :-2] * vocab + ids[:, 1:-1]) * vocab + ids[:, 2:]
        flat = keys.reshape(-1)
        index = torch.searchsorted(self.keys, flat).clamp(max=self.keys.numel() - 1)
        found = self.keys[index] == flat
        gathered_count = self.log_count[index]
        gathered_variance = self.variance[index]
        log_count = torch.where(found, gathered_count,
                                torch.zeros_like(gathered_count))
        variance = torch.where(found, gathered_variance,
                               torch.ones_like(gathered_variance))
        out[:, 2:, 0] = log_count.reshape(batch, length - 2)
        out[:, 2:, 1] = variance.reshape(batch, length - 2)
        return out


class ResidualGateHandle:
    """Per-forward state for the installed gates, and their running statistics."""

    def __init__(self, gates: nn.ModuleDict, family: str,
                 familiarity: TrigramFamiliarity | None):
        self.gates = gates
        self.family = family
        self.features = family_features(family)
        self.familiarity = familiarity
        self.context: torch.Tensor | None = None   # [batch, sequence, 2]
        self.calibrating = False
        # The same-checkpoint ablation: admission forced back to unit strength without
        # touching a weight, so 'does this model depend on its gate at inference' is a
        # question about one set of parameters rather than two runs.
        self.force_identity = False
        self.record = False
        # Per-token gate values, kept only when a scorer asks: the distributions are
        # the mechanistic evidence, and a running mean cannot be split by context
        # afterwards.
        self.keep = False
        self.kept: dict[int, list] = {}
        self.stats: dict[int, dict] = {}
        self.gated_calls = 0

    @property
    def layer_indices(self) -> tuple[int, ...]:
        return tuple(sorted(int(key) for key in self.gates))

    def gate(self, layer: int) -> ResidualAdmissionGate:
        return self.gates[str(layer)]

    def set_context(self, input_ids: torch.Tensor) -> None:
        """Compute the familiarity features once per forward, not once per layer."""
        if self.familiarity is None:
            return
        self.context = self.familiarity.features(input_ids)

    def reset_stats(self) -> None:
        self.stats = {}
        self.kept = {}
        self.gated_calls = 0

    def _observe(self, layer: int, values: torch.Tensor) -> None:
        flat = values.detach().reshape(-1).to(torch.float64)
        entry = self.stats.setdefault(layer, {"sum": 0.0, "square": 0.0, "count": 0,
                                              "min": math.inf, "max": -math.inf})
        entry["sum"] += float(flat.sum())
        entry["square"] += float((flat * flat).sum())
        entry["count"] += int(flat.numel())
        entry["min"] = min(entry["min"], float(flat.min()))
        entry["max"] = max(entry["max"], float(flat.max()))

    def summary(self) -> dict:
        out = {}
        for layer, entry in sorted(self.stats.items()):
            count = max(entry["count"], 1)
            mean = entry["sum"] / count
            variance = max(entry["square"] / count - mean * mean, 0.0)
            out[str(layer)] = {"mean": mean, "std": variance ** 0.5,
                               "min": entry["min"], "max": entry["max"],
                               "tokens": entry["count"]}
        return out

    def build_features(self, layer: int, hidden_states: torch.Tensor,
                       update: torch.Tensor) -> torch.Tensor:
        """Assemble the gate's input for one layer's proposal.

        ``hidden_states`` is the MLP's input -- the normalized residual state -- and
        ``update`` its output. Those are exactly the two tensors the diagnostics probe
        measured, so a geometry gate reads what that probe scored and nothing more.
        """
        parts = []
        if "log_count" in self.features:
            if self.context is None:
                raise ValueError(
                    "a familiarity gate needs per-forward context; call set_context, "
                    "which install_residual_gates wires to the model's forward")
            context = self.context
            if context.shape[:2] != update.shape[:2]:
                raise ValueError("familiarity context %s does not match hidden states %s"
                                 % (tuple(context.shape[:2]), tuple(update.shape[:2])))
            parts.append(context.to(device=update.device, dtype=torch.float32))
        if "norm_h" in self.features:
            normalized = hidden_states.to(torch.float32)
            proposal = update.to(torch.float32)
            norm_h = normalized.norm(dim=-1)
            norm_r = proposal.norm(dim=-1)
            cosine = torch.nn.functional.cosine_similarity(normalized, proposal, dim=-1)
            parts.append(torch.stack([norm_h, norm_r, norm_r / (norm_h + 1e-6), cosine],
                                     dim=-1))
        return torch.cat(parts, dim=-1)

    def apply(self, layer: int, hidden_states: torch.Tensor,
              update: torch.Tensor) -> torch.Tensor:
        gate = self.gate(layer)
        features = self.build_features(layer, hidden_states, update)
        if self.calibrating:
            # Stock arithmetic while the normalizer is being estimated: the gate is not
            # in the graph, so it takes no gradient and the model is bitwise unchanged.
            gate.observe(features)
            return update
        if not gate.is_calibrated:
            raise ValueError(
                "the gate on layer %d has no feature normalizer; run calibrate_gates "
                "before the first training step" % layer)
        values = gate(features)
        if self.force_identity:
            values = torch.ones_like(values)
        if self.record:
            self._observe(layer, values)
        if self.keep:
            self.kept.setdefault(layer, []).append(values.detach().float().cpu())
        self.gated_calls += int(values.numel())
        return update * values.unsqueeze(-1).to(update.dtype)


def _resolve_layers(model: nn.Module, layers) -> dict[int, nn.Module]:
    inner = getattr(model, "model", model)
    if not hasattr(inner, "layers"):
        raise ValueError("expected a decoder model with a `layers` list")
    chosen = {}
    for index in layers:
        index = int(index)
        if not 0 <= index < len(inner.layers):
            raise ValueError("layer %d outside the model's %d layers"
                             % (index, len(inner.layers)))
        if index in chosen:
            raise ValueError("layer %d listed twice" % index)
        chosen[index] = inner.layers[index]
    if not chosen:
        raise ValueError("no layers to gate")
    return chosen


def gate_parameter_count(model: nn.Module) -> int:
    gates = getattr(model, "residual_gates", None)
    return 0 if gates is None else sum(p.numel() for p in gates.parameters())


@torch.no_grad()
def calibrate_gates(model: nn.Module, handle: ResidualGateHandle, batches) -> dict:
    """Run ``batches`` with the gates inert, then freeze their feature normalizers.

    The model is bitwise stock throughout: the gate is not in the graph, takes no
    gradient, and scales nothing. Returns the per-layer means and standard deviations
    that were frozen, so a run can record what its gate is standardizing against.
    """
    was_training = model.training
    model.eval()
    handle.calibrating = True
    try:
        seen = 0
        for batch in batches:
            model(**batch)
            seen += 1
        if not seen:
            raise ValueError("the calibration pass was given no batches")
    finally:
        handle.calibrating = False
        model.train(was_training)
    report = {}
    for index in handle.layer_indices:
        gate = handle.gate(index)
        gate.finalize()
        report[str(index)] = {
            "batches": seen,
            "tokens": float(gate.calibration_count),
            "mean": [float(value) for value in gate.feature_mean],
            "std": [float(value) for value in gate.feature_std],
        }
    return report


def install_residual_gates(model: nn.Module, layers, family: str = "familiarity",
                           hidden: int = 16, span: float = 1.0,
                           familiarity: TrigramFamiliarity | None = None
                           ) -> ResidualGateHandle:
    """Attach scalar admission gates to ``layers`` and keep them installed.

    Unlike the intervention helpers in ``ffn_skip`` this is not a context manager: the
    gates are parameters of the model, they appear in its state dict, and they are meant
    to survive the run. ``residual_gates`` is a context-manager wrapper for tests and
    evaluation.
    """
    features = family_features(family)
    if "log_count" in features and familiarity is None:
        raise ValueError("family %r needs trigram familiarity statistics" % family)
    if getattr(model, "residual_gates", None) is not None:
        raise ValueError("this model already has residual gates installed")
    chosen = _resolve_layers(model, layers)

    gates = nn.ModuleDict({
        str(index): ResidualAdmissionGate(len(features), hidden=hidden, span=span)
        for index in chosen
    })
    parameter = next(model.parameters(), None)
    if parameter is not None:
        # float32 on the model's device: the gate is a few hundred parameters and there
        # is nothing to gain from putting a tanh in bf16.
        gates.to(device=parameter.device, dtype=torch.float32)
        if familiarity is not None:
            familiarity.to(parameter.device)

    handle = ResidualGateHandle(gates, family, familiarity)
    model.residual_gates = gates
    model.residual_gate_handle = handle
    handle._originals = {}
    for index, layer in chosen.items():
        handle._originals[index] = layer.mlp.forward

        def wrapped(hidden_states, _index=index, _original=handle._originals[index],
                    _handle=handle, **kwargs):
            return _handle.apply(_index, hidden_states,
                                 _original(hidden_states, **kwargs))

        layer.mlp.forward = wrapped

    handle._model_forward = None
    if familiarity is not None:
        handle._model_forward = model.forward

        def forward(*args, _handle=handle, _original=model.forward, **kwargs):
            ids = kwargs.get("input_ids")
            if ids is None and args:
                ids = args[0]
            if ids is None:
                raise ValueError("a familiarity gate needs input_ids; this model was "
                                 "called with embeddings only")
            _handle.set_context(ids)
            return _original(*args, **kwargs)

        model.forward = forward
    handle._chosen = chosen
    return handle


def remove_residual_gates(model: nn.Module) -> None:
    handle = getattr(model, "residual_gate_handle", None)
    if handle is None:
        return
    for index, layer in handle._chosen.items():
        layer.mlp.forward = handle._originals[index]
    if handle._model_forward is not None:
        model.forward = handle._model_forward
    model.residual_gates = None
    model.residual_gate_handle = None


@contextlib.contextmanager
def residual_gates(model: nn.Module, layers, **kwargs):
    """``install_residual_gates`` scoped to a block, for tests and evaluation."""
    handle = install_residual_gates(model, layers, **kwargs)
    try:
        yield handle
    finally:
        remove_residual_gates(model)


class ResidualGateCheckpointCallback(TrainerCallback):
    """Write the gate, and only the gate, at every evaluation.

    In the frozen stage the backbone cannot move, so a full checkpoint at each milestone
    would write the same 1.9B parameters five times over for the sake of a few hundred
    that changed. The gate is the run. Each file also carries the feature normalizer, so
    a scorer needs nothing but this file and the base model to reproduce the checkpoint
    exactly.
    """

    def __init__(self, output_path, prefix: str = "gate"):
        self.output_path = str(output_path)
        self.prefix = prefix
        self.written: list[str] = []

    def _write(self, model, step: int) -> None:
        gates = getattr(model, "residual_gates", None)
        if gates is None:
            raise ValueError("gate checkpointing was installed on an ungated model")
        os.makedirs(self.output_path, exist_ok=True)
        path = os.path.join(self.output_path, "%s-step%d.pt" % (self.prefix, step))
        state = {key: value.detach().cpu()
                 for key, value in gates.state_dict().items()}
        handle = getattr(model, "residual_gate_handle", None)
        payload = {"step": step, "state_dict": state,
                   "family": None if handle is None else handle.family,
                   "layers": [] if handle is None else list(handle.layer_indices)}
        # Temp file then replace: a crash between truncate and write would otherwise
        # destroy the previous checkpoint at the same path.
        torch.save(payload, path + ".tmp")
        os.replace(path + ".tmp", path)
        self.written.append(path)

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        if model is not None:
            self._write(model, int(state.global_step))

    def on_train_end(self, args, state, control, model=None, **kwargs):
        if model is not None:
            self._write(model, int(state.global_step))


def attach_residual_gates(config, model, dataset):
    """Install and calibrate the gates a run configures, or return None if it has none.

    Called from ``do_distill`` before the optimizer exists, so the calibration pass runs
    against a model nothing has moved yet. A calibration batch is one document: the
    statistics are per token, so padding would only dilute them, and reading the dataset
    directly rather than the training dataloader leaves the training data order untouched.
    """
    section = getattr(config, "residual_gate", None)
    if section is None:
        return None

    text_config = getattr(model.config, "text_config", model.config)
    statistics = None
    if section.familiarity_cache:
        statistics = TrigramFamiliarity(section.familiarity_cache,
                                        text_config.vocab_size)
    handle = install_residual_gates(model, section.layers, family=section.family,
                                    hidden=section.hidden, span=section.span,
                                    familiarity=statistics)
    if section.init_from:
        digest = load_gate_checkpoint(model, handle, section.init_from)
        LOG.info("Residual gates on layers %s, family %s, %d parameters; warm-started "
                 "from %s (sha256 %s)", list(handle.layer_indices), section.family,
                 gate_parameter_count(model), section.init_from, digest[:16])
        return handle
    device = next(model.parameters()).device
    batches = []
    for index in range(min(section.calibration_batches, len(dataset))):
        ids = torch.tensor([dataset[index]["input_ids"]], device=device)
        batches.append({"input_ids": ids, "attention_mask": torch.ones_like(ids)})
    calibration = calibrate_gates(model, handle, batches)
    LOG.info("Residual gates on layers %s, family %s, %d parameters; feature normalizer "
             "frozen from %d documents", list(handle.layer_indices), section.family,
             gate_parameter_count(model), len(batches))
    LOG.debug("Gate feature normalizer: %s", calibration)
    return handle


def load_gate_checkpoint(model, handle: ResidualGateHandle, path) -> str:
    """Load trained gate weights into an installed gate, and prove they arrived.

    A warm start is only a warm start if the gate is bitwise the one that was fitted.
    The family and the gated layers have to agree with the checkpoint -- loading a
    four-feature gate into a two-feature one would be caught by shape, but loading a gate
    trained on different layers would not be, and would silently move a policy to depths
    it was never fitted for. The normalizer travels with the weights and is asserted
    present: an uncalibrated warm start would standardize against zeros and mean nothing.

    Returns the checkpoint digest, so a run records which policy it started from.
    """
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("family") != handle.family:
        raise ValueError("gate checkpoint is family %r but this run configures %r"
                         % (payload.get("family"), handle.family))
    if tuple(payload.get("layers", ())) != handle.layer_indices:
        raise ValueError("gate checkpoint covers layers %s but this run gates %s"
                         % (tuple(payload.get("layers", ())), handle.layer_indices))
    model.residual_gates.load_state_dict(payload["state_dict"], strict=True)
    for index in handle.layer_indices:
        gate = handle.gate(index)
        if not gate.is_calibrated:
            raise ValueError(
                "the gate for layer %d arrived without a feature normalizer; a warm "
                "start cannot recalibrate without changing what the weights mean" % index)
    loaded = model.residual_gates.state_dict()
    for key, value in payload["state_dict"].items():
        if not torch.equal(loaded[key].cpu(), value.cpu()):
            raise ValueError("gate parameter %s did not survive loading" % key)
    with io.open(path, "rb") as handle_in:
        return hashlib.sha256(handle_in.read()).hexdigest()

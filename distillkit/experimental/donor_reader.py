"""Frozen C1/Flash-Next table readers with a tiny single-stream adapter.

This module deliberately stops at the separable PLE feature extractor.  It does not
recreate Flash-Next hyper-connections, its MoE, the PLE key/query gate, or the direct
value residual.  A reader is exactly

    table row -> frozen value projection -> frozen causal depthwise filters

and its only learned interface to Qwen3.5-4B is a per-stream/per-channel collapse plus
one residual scale.  The scale starts at zero, so attaching a reader is an exact stock
model identity even though all frozen reader weights are already nonzero.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

__all__ = [
    "DonorReaderTransplant",
    "initialise_transplant_reader",
    "load_reference_tensors",
]


def _normalise_conv(weight: torch.Tensor) -> torch.Tensor:
    if weight.ndim == 2:
        weight = weight.unsqueeze(1)
    if weight.ndim != 3 or weight.shape[1] != 1:
        raise ValueError(
            f"depthwise convolution must be [channels, 1, taps], got {tuple(weight.shape)}"
        )
    return weight


def _matching_key(keys: Iterable[str], suffix: str) -> str:
    keys = list(keys)
    exact = [key for key in keys if key == suffix]
    matches = exact or [key for key in keys if key.endswith("." + suffix)]
    # A full C1 checkpoint has dozens of backbone conv1d tensors. Requiring the sidecar
    # path prevents a suffix-only match from silently transplanting an SSM filter.
    if not exact and suffix in (
        "value_proj.weight",
        "conv1d.weight",
        "norm_conv.weight",
    ):
        sidecar = [key for key in matches if ".sidecar.ple." in key]
        if sidecar:
            matches = sidecar
    if len(matches) != 1:
        raise KeyError(f"expected one tensor ending in {suffix!r}, found {matches}")
    return matches[0]


def load_reference_tensors(
    reference: str | Path, suffixes: Iterable[str]
) -> dict[str, torch.Tensor]:
    """Load only named reader tensors from a ``.pt`` extraction or HF checkpoint.

    Directory checkpoints may be a single safetensors file or a sharded checkpoint
    with ``model.safetensors.index.json``.  Selection is suffix-based so the compact
    donor extraction (``value_proj.weight``) and a full C1 checkpoint
    (``model.layers.1.sidecar.ple.value_proj.weight``) share one validated path.
    """
    reference = Path(reference)
    if not reference.exists():
        raise FileNotFoundError(f"reader reference does not exist: {reference}")
    suffixes = tuple(suffixes)
    if reference.is_file() and reference.suffix in (".pt", ".pth", ".bin"):
        held = torch.load(reference, map_location="cpu", weights_only=False)
        if not isinstance(held, Mapping):
            raise TypeError(
                f"reader reference must contain a tensor mapping: {reference}"
            )
        return {
            suffix: held[_matching_key(held.keys(), suffix)].detach().cpu()
            for suffix in suffixes
        }

    from safetensors import safe_open

    if reference.is_file():
        files = [reference]
    else:
        index = reference / "model.safetensors.index.json"
        if index.is_file():
            weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
            all_keys = tuple(weight_map)
            selected = {suffix: _matching_key(all_keys, suffix) for suffix in suffixes}
            result = {}
            by_file: dict[str, list[tuple[str, str]]] = {}
            for suffix, key in selected.items():
                by_file.setdefault(weight_map[key], []).append((suffix, key))
            for filename, wanted in by_file.items():
                with safe_open(
                    reference / filename, framework="pt", device="cpu"
                ) as handle:
                    result.update(
                        {suffix: handle.get_tensor(key) for suffix, key in wanted}
                    )
            return result
        files = sorted(reference.glob("model*.safetensors"))
    if not files:
        raise FileNotFoundError(
            f"no .pt or safetensors reader weights found at {reference}"
        )
    locations: dict[str, tuple[Path, str]] = {}
    all_keys = []
    for file in files:
        with safe_open(file, framework="pt", device="cpu") as handle:
            for key in handle.keys():  # noqa: SIM118 - safe_open is not iterable
                all_keys.append(key)
                locations[key] = (file, key)
    selected = {suffix: _matching_key(all_keys, suffix) for suffix in suffixes}
    result = {}
    for suffix, key in selected.items():
        file, tensor_key = locations[key]
        with safe_open(file, framework="pt", device="cpu") as handle:
            result[suffix] = handle.get_tensor(tensor_key)
    return result


class _ReaderWeight(nn.Module):
    """One independently initializable tensor for safe HF missing-key loading."""

    def __init__(self, shape, initial_values: list[float], *, trainable: bool):
        super().__init__()
        self.initial_values = list(initial_values)
        self.weight = nn.Parameter(torch.empty(shape), requires_grad=trainable)
        self._sidecar_weight_init = "reader"
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Transformers marks tensors loaded from the checkpoint before it initializes
        # missing keys. This custom copy must honor that guard explicitly (unlike
        # torch.nn.init helpers, ``copy_`` is not wrapped by Transformers).
        if getattr(self.weight, "_is_hf_initialized", False):
            return
        with torch.no_grad():
            values = torch.tensor(
                self.initial_values, device=self.weight.device, dtype=self.weight.dtype
            )
            if self.weight.ndim:
                values = values.reshape(
                    self.weight.shape[0], *([1] * (self.weight.ndim - 1))
                )
                values = values.expand_as(self.weight)
            else:
                values = values[0]
            self.weight.copy_(values)


class DonorReaderTransplant(nn.Module):
    """Frozen value/temporal reader, collapsed into one residual stream.

    ``value_source`` and ``conv_source`` identify the 2x2 factorial arm.  C1 has two
    trained convolution streams in the diagnosed widened checkpoint; they are reduced
    by their fixed equal mean.  Flash-Next has four streams and supports the requested
    4xhidden learned mixer, or one of the no-training mathematical collapses.
    """

    def __init__(
        self,
        hidden_size: int,
        feature_dim: int,
        *,
        value_source: str = "donor",
        conv_source: str = "donor",
        collapse: str = "mixer",
        single_stream: int = 0,
        collapse_weights: Iterable[float] | None = None,
        rho: float = 0.0,
        conv_kernel_size: int = 4,
        ngram_size: int = 3,
        rms_norm_eps: float = 1e-6,
    ):
        super().__init__()
        if hidden_size <= 0 or feature_dim <= 0:
            raise ValueError("hidden_size and feature_dim must be positive")
        if value_source not in ("c1", "donor") or conv_source not in ("c1", "donor"):
            raise ValueError("reader sources must be 'c1' or 'donor'")
        if collapse not in ("mixer", "equal_mean", "single", "scalar", "pca_rank1"):
            raise ValueError("unknown donor stream collapse")
        self.hidden_size = hidden_size
        self.feature_dim = feature_dim
        self.value_source = value_source
        self.conv_source = conv_source
        self.stream_count = 4 if conv_source == "donor" else 2
        self.collapse = collapse if conv_source == "donor" else "equal_mean"
        self.single_stream = int(single_stream)
        self.eps = float(rms_norm_eps)
        self.conv_dilation = int(ngram_size)
        self.short_conv_state_len = (int(conv_kernel_size) - 1) * self.conv_dilation
        if not 0 <= self.single_stream < self.stream_count:
            raise ValueError(f"single_stream must be in [0, {self.stream_count})")

        self.value_proj = nn.Linear(feature_dim, hidden_size, bias=False)
        self.conv1d = nn.Conv1d(
            self.stream_count * hidden_size,
            self.stream_count * hidden_size,
            kernel_size=conv_kernel_size,
            groups=self.stream_count * hidden_size,
            dilation=self.conv_dilation,
            bias=False,
        )
        # Donor norm_conv is part of its trained temporal filter. C1 already folded
        # this scale into the learned filters when DirectionGatedPLESidecar was built.
        self.conv_norm = _ReaderWeight(
            (self.stream_count, hidden_size), [0.0] * self.stream_count, trainable=False
        )
        self.value_proj.weight.requires_grad_(False)
        self.conv1d.weight.requires_grad_(False)
        for module in (self.value_proj, self.conv1d):
            nn.init.zeros_(module.weight)
            module._sidecar_weight_init = "zero"

        weights = None if collapse_weights is None else list(collapse_weights)
        if self.collapse in ("scalar", "pca_rank1"):
            if weights is None or len(weights) != self.stream_count:
                raise ValueError(
                    f"{self.collapse} collapse needs {self.stream_count} weights"
                )
            initial_values = [float(value) for value in weights]
        elif self.collapse == "single":
            initial_values = [0.0] * self.stream_count
            initial_values[self.single_stream] = 1.0
        else:
            initial_values = [1.0 / self.stream_count] * self.stream_count
        self._initial_mixer_values = initial_values
        self.mixer = _ReaderWeight(
            (self.stream_count, hidden_size),
            initial_values,
            trainable=self.collapse == "mixer",
        )
        self.initial_rho = float(rho)
        self.rho = _ReaderWeight((), [self.initial_rho], trainable=True)
        self.enforce_trainability()
        self._reader_loaded = False

    def enforce_trainability(self) -> None:
        """Restore the experiment's freeze contract after HF materializes tensors.

        ``from_pretrained`` may replace parameters created under its empty-weight
        context and the replacements default to ``requires_grad=True``.  Reader
        sources must stay frozen regardless of how the checkpoint was loaded.
        """
        self.value_proj.weight.requires_grad_(False)
        self.conv1d.weight.requires_grad_(False)
        self.conv_norm.weight.requires_grad_(False)
        self.mixer.weight.requires_grad_(self.collapse == "mixer")
        self.rho.weight.requires_grad_(True)

    def _apply(self, fn, *args, **kwargs):
        """Keep the tiny learned adapter in fp32 through model dtype conversions.

        At 2560 dimensions the requested mixer starts at 0.25. BF16 spacing there is
        roughly 2e-3, far larger than an AdamW step at 1e-4, so allowing ``model.to``
        to cast these 10k coefficients would make most updates round away.
        """
        module = super()._apply(fn, *args, **kwargs)
        module.pin_adapter_fp32()
        return module

    def pin_adapter_fp32(self) -> None:
        for parameter in (self.rho.weight, self.mixer.weight):
            if parameter.dtype == torch.float32:
                continue
            parameter.data = parameter.data.float()
            if parameter.grad is not None:
                parameter.grad = parameter.grad.float()

    def reset_adapter_parameters(self) -> None:
        """Restore missing adapter tensors during HF's missing-weight initialization."""
        self.conv_norm.reset_parameters()
        self.mixer.reset_parameters()
        self.rho.reset_parameters()
        self.pin_adapter_fp32()

    @property
    def conv_norm_delta(self) -> torch.Tensor:
        return self.conv_norm.weight

    def load_reader_weights(
        self,
        *,
        value_weight: torch.Tensor,
        conv_weight: torch.Tensor,
        conv_norm_delta: torch.Tensor | None = None,
    ) -> None:
        conv_weight = _normalise_conv(conv_weight)
        expected_value = tuple(self.value_proj.weight.shape)
        expected_conv = tuple(self.conv1d.weight.shape)
        if tuple(value_weight.shape) != expected_value:
            raise ValueError(
                f"value projection is {tuple(value_weight.shape)}, expected {expected_value}"
            )
        if tuple(conv_weight.shape) != expected_conv:
            raise ValueError(
                f"{self.conv_source} convolution is {tuple(conv_weight.shape)}, expected {expected_conv}"
            )
        if conv_norm_delta is None:
            conv_norm_delta = torch.zeros_like(self.conv_norm_delta)
        else:
            conv_norm_delta = conv_norm_delta.reshape(
                self.stream_count, self.hidden_size
            )
        with torch.no_grad():
            self.value_proj.weight.copy_(value_weight.to(self.value_proj.weight))
            self.conv1d.weight.copy_(conv_weight.to(self.conv1d.weight))
            self.conv_norm_delta.copy_(conv_norm_delta.to(self.conv_norm_delta))
        self.enforce_trainability()
        self._reader_loaded = True

    def conv_streams(self, value: torch.Tensor) -> torch.Tensor:
        """Return every frozen temporal feature as ``[batch, time, stream, hidden]``."""
        normed = value.float()
        normed = normed * torch.rsqrt(normed.pow(2).mean(-1, keepdim=True) + self.eps)
        wide = normed.unsqueeze(-2).expand(
            *normed.shape[:-1], self.stream_count, self.hidden_size
        )
        wide = wide * (1.0 + self.conv_norm_delta.float())
        wide = wide.to(value.dtype).flatten(-2).transpose(1, 2)
        wide = F.pad(wide, (self.short_conv_state_len, 0))
        wide = F.silu(self.conv1d(wide)).transpose(1, 2)
        return wide.unflatten(-1, (self.stream_count, self.hidden_size))

    def collapse_streams(self, streams: torch.Tensor) -> torch.Tensor:
        if streams.shape[-2:] != (self.stream_count, self.hidden_size):
            raise ValueError(
                f"expected stream features ending in ({self.stream_count}, {self.hidden_size}), "
                f"got {tuple(streams.shape)}"
            )
        # Do the 10k compatibility layer in fp32. Casting its coefficients to BF16 at
        # 0.25 would quantize away the small updates that fp32 storage preserves.
        return (streams.float() * self.mixer.weight.float()).sum(-2)

    def features(self, table_rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        value = self.value_proj(table_rows.to(dtype=self.value_proj.weight.dtype))
        streams = self.conv_streams(value)
        return streams, self.collapse_streams(streams)

    def forward(
        self, hidden_states: torch.Tensor, table_rows: torch.Tensor
    ) -> torch.Tensor:
        _, collapsed = self.features(table_rows)
        contribution = self.rho.weight.to(dtype=collapsed.dtype) * collapsed
        return hidden_states + contribution.to(dtype=hidden_states.dtype)

    @torch.no_grad()
    def reader_report(self, prefix: str = "reader") -> dict[str, float]:
        report = {
            f"{prefix}/rho": self.rho.weight.float().item(),
            f"{prefix}/value_norm": self.value_proj.weight.float().norm().item(),
            f"{prefix}/conv_norm": self.conv1d.weight.float().norm().item(),
            f"{prefix}/mixer_norm": self.mixer.weight.float().norm().item(),
            f"{prefix}/trainable_parameters": float(
                sum(p.numel() for p in self.parameters() if p.requires_grad)
            ),
        }
        for stream in range(self.stream_count):
            report[f"{prefix}/mixer_abs_mean_{stream}"] = (
                self.mixer.weight[stream].float().abs().mean().item()
            )
        return report


def initialise_transplant_reader(
    model,
    *,
    c1_reference: str | Path | None = None,
    donor_reference: str | Path | None = None,
) -> dict:
    """Populate one transplant arm from frozen C1 and/or donor checkpoints."""
    sidecar = model.model.layers[model.config.sidecar_layer_index].sidecar
    reader: DonorReaderTransplant = sidecar.reader
    references = {"c1": c1_reference, "donor": donor_reference}

    value_ref = references[reader.value_source]
    conv_ref = references[reader.conv_source]
    if value_ref is None:
        raise ValueError(
            f"{reader.value_source}_reference is required for the value arm"
        )
    if conv_ref is None:
        raise ValueError(
            f"{reader.conv_source}_reference is required for the convolution arm"
        )
    value = load_reference_tensors(value_ref, ["value_proj.weight"])[
        "value_proj.weight"
    ]
    conv_names = ["conv1d.weight"] + (
        ["norm_conv.weight"] if reader.conv_source == "donor" else []
    )
    conv = load_reference_tensors(conv_ref, conv_names)
    reader.load_reader_weights(
        value_weight=value,
        conv_weight=conv["conv1d.weight"],
        conv_norm_delta=conv.get("norm_conv.weight"),
    )
    return {
        "value_source": reader.value_source,
        "conv_source": reader.conv_source,
        "value_reference": str(Path(value_ref).resolve()),
        "conv_reference": str(Path(conv_ref).resolve()),
        "streams": reader.stream_count,
        "collapse": reader.collapse,
        "trainable_parameters": sum(
            p.numel() for p in reader.parameters() if p.requires_grad
        ),
        "rho": reader.rho.weight.item(),
    }

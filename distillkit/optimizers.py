"""Mixed Muon/AdamW and explicit two-stage training helpers.

The mixed optimizer is for ordinary, unflattened PyTorch parameters. It is not
a DeepSpeed Muon implementation: ZeRO substitutes flattened master parameters,
whereas Muon needs each original 2D matrix. DeepSpeed's native Muon integration
is separate (https://pytorch.org/blog/using-muon-optimizer-with-deepspeed/).
"""

from collections.abc import Iterable
from typing import Any

import torch
from torch import nn
from transformers import TrainerCallback

from distillkit.gated_residual import GatedResidual


def validate_optimizer_backend(
    *,
    strategy: str = "hybrid",
    deepspeed: Any = None,
    fsdp: Any = None,
    dynamic_unfreeze: bool = False,
    world_size: int = 1,
) -> None:
    """Fail before wrapping a model in an unsupported optimizer/backend pair.

    Static AdamW stages are allowed with DeepSpeed; this is configuration
    validation, not proof that DeepSpeed/CPUAdam works on the current host.
    Dynamic unfreezing requires a single unpartitioned process. Start a fresh
    stage-2 job with the backbone trainable for distributed/offloaded training.
    """
    if strategy not in {"hybrid", "adamw"}:
        raise ValueError(f"Unknown optimizer strategy: {strategy!r}")
    if strategy == "hybrid" and (deepspeed or fsdp):
        raise ValueError(
            "MixedMuonAdamW does not support DeepSpeed or FSDP parameter "
            "partitioning/flattening. Select AdamW explicitly for ZeRO-2 CPU "
            "offload, or validate a separate native DeepSpeed Muon integration. "
            "zero_allow_untested_optimizer does not make this wrapper compatible."
        )
    if dynamic_unfreeze and (deepspeed or fsdp or world_size > 1):
        raise ValueError(
            "Dynamic backbone unfreezing is supported only in a single "
            "unpartitioned process. Distributed reducers and ZeRO/FSDP partitions "
            "are built from initially trainable parameters; use separate stage "
            "jobs and rebuild the optimizer/distributed engine for stage 2."
        )


def _auxiliary_parameter_ids(model: nn.Module) -> set[int]:
    """Identify architecture/projection weights by module ownership, not shape."""
    result = set()
    stage1_names = getattr(model, "stage1_parameter_names", None)
    if callable(stage1_names):
        names = set(stage1_names())
        result.update(id(p) for name, p in model.named_parameters() if name in names)
    for name, module in model.named_modules():
        parts = set(name.split("."))
        if isinstance(module, GatedResidual) or parts.intersection(
            {"sidecar", "W_side_proj", "distillation_projections"}
        ):
            result.update(id(p) for p in module.parameters())
    return result


def mixed_parameter_groups(
    model: nn.Module, *, include_frozen: bool = True
) -> list[dict[str, Any]]:
    """Route hidden nn.Linear matrices to Muon, everything else to AdamW.

    Embeddings, tied output heads, gate/sidecar/distillation projections and all
    vectors use AdamW. Frozen parameters are included by default so a later
    single-process unfreeze cannot silently drop the backbone. Optimizer states
    remain lazy while their gradients are None.
    """
    adam_ids = _auxiliary_parameter_ids(model)
    linear_ids = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.Embedding) or name.split(".")[-1] == "lm_head":
            adam_ids.update(id(p) for p in module.parameters())
        elif isinstance(module, nn.Linear):
            linear_ids.add(id(module.weight))
    output_embeddings = getattr(model, "get_output_embeddings", None)
    if callable(output_embeddings):
        head = output_embeddings()
        if head is not None:
            adam_ids.update(id(p) for p in head.parameters())

    buckets: dict[tuple[str, bool], dict[str, Any]] = {}
    seen = set()
    for name, parameter in model.named_parameters():
        if id(parameter) in seen or (not include_frozen and not parameter.requires_grad):
            continue
        seen.add(id(parameter))
        use_muon = (
            parameter.ndim == 2
            and id(parameter) in linear_ids
            and id(parameter) not in adam_ids
        )
        kind = "muon" if use_muon else "adamw"
        decay = parameter.ndim >= 2
        group = buckets.setdefault(
            (kind, decay),
            {"optimizer_kind": kind, "decay": decay, "params": [], "param_names": []},
        )
        group["params"].append(parameter)
        group["param_names"].append(name)
    return list(buckets.values())


class MixedMuonAdamW(torch.optim.Optimizer):
    """One scheduler/checkpoint surface backed by official PyTorch optimizers.

    Child optimizers share the exact parent parameter-group dictionaries and
    state mapping. Scheduler LR mutations, GradScaler unscaling, zero_grad, and
    standard state_dict therefore all refer to the same parameters and buffers.
    This optimizer deliberately disallows adding/replacing parameter groups:
    include future-unfrozen weights at construction instead.
    """

    def __init__(
        self,
        param_groups: Iterable[dict[str, Any]],
        *,
        lr: float = 3e-4,
        muon_lr: float | None = None,
        weight_decay: float = 0.01,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        momentum: float = 0.95,
        ns_steps: int = 5,
    ):
        if not hasattr(torch.optim, "Muon"):
            raise RuntimeError("MixedMuonAdamW requires PyTorch with torch.optim.Muon.")
        groups = []
        seen = set()
        for source in param_groups:
            group = dict(source)
            group["params"] = list(source["params"])
            kind = group.get("optimizer_kind")
            if kind not in {"muon", "adamw"}:
                raise ValueError("Every group must select optimizer_kind='muon' or 'adamw'.")
            for parameter in group["params"]:
                if id(parameter) in seen:
                    raise ValueError("A parameter appears more than once in optimizer groups.")
                seen.add(id(parameter))
                if kind == "muon" and parameter.ndim != 2:
                    raise ValueError("Muon parameter groups require original 2D matrices.")
            group.setdefault("lr", muon_lr if kind == "muon" and muon_lr is not None else lr)
            group.setdefault("weight_decay", weight_decay if group.pop("decay", True) else 0.0)
            if group["params"]:
                groups.append(group)
        self._groups_locked = False
        super().__init__(groups, {"lr": lr})
        self._children: dict[str, torch.optim.Optimizer] = {}
        for kind in ("muon", "adamw"):
            selected = [g for g in self.param_groups if g["optimizer_kind"] == kind]
            if not selected:
                continue
            if kind == "muon":
                child = torch.optim.Muon(
                    selected, lr=lr if muon_lr is None else muon_lr,
                    weight_decay=weight_decay, momentum=momentum, ns_steps=ns_steps,
                    adjust_lr_fn="match_rms_adamw",
                )
            else:
                child = torch.optim.AdamW(
                    selected, lr=lr, weight_decay=weight_decay,
                    betas=betas, eps=eps, foreach=False,
                )
            self._children[kind] = child
        self._groups_locked = True
        self._parameter_layout = self._layout()
        self._bind_children()

    def _layout(self):
        return tuple(
            (g["optimizer_kind"], tuple((id(p), tuple(p.shape)) for p in g["params"]))
            for g in self.param_groups
        )

    def _bind_children(self):
        for kind, child in self._children.items():
            child.param_groups = [g for g in self.param_groups if g["optimizer_kind"] == kind]
            child.state = self.state

    def add_param_group(self, param_group):
        if self._groups_locked:
            raise RuntimeError("Include future-unfrozen parameters when constructing MixedMuonAdamW.")
        super().add_param_group(param_group)

    @torch.no_grad()
    def step(self, closure=None):
        if self._layout() != self._parameter_layout:
            raise RuntimeError("MixedMuonAdamW parameter layout changed; flattened/sharded weights are unsupported.")
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for child in self._children.values():
            child.step()
        return loss

    def load_state_dict(self, state_dict):
        saved = state_dict["param_groups"]
        if [g.get("optimizer_kind") for g in saved] != [
            g["optimizer_kind"] for g in self.param_groups
        ]:
            raise ValueError("Saved optimizer parameter routing differs from the current model.")
        for current, checkpoint in zip(self.param_groups, saved):
            if current.get("param_names") != checkpoint.get("param_names"):
                raise ValueError("Saved optimizer parameter names/order differ from the current model.")
        super().load_state_dict(state_dict)
        self._bind_children()


def build_mixed_optimizer(model: nn.Module, *, include_frozen: bool = True, **kwargs):
    groups = mixed_parameter_groups(model, include_frozen=include_frozen)
    _refuse_trainable_muon_shards(model, groups)
    return MixedMuonAdamW(groups, **kwargs)


def _refuse_trainable_muon_shards(model: nn.Module, groups) -> None:
    """Muon must not receive a tensor-parallel shard that is actually being trained.

    Newton-Schulz orthogonalization does not commute with slicing, so Muon on a shard is
    not the shard of Muon on the whole matrix; the shape-dependent learning-rate scaling
    would use the shard's dimensions too. Configuration already refuses the combination
    that could produce this, but only for runs driven by ``main.py``: a programmatic
    caller can reach ``unfreeze_backbone()`` directly. This is the same invariant checked
    where it is actually load-bearing, against the live parameters.

    Note which parameters are at risk. Column- and row-parallel weights are
    ``nn.Parameter`` inside an ``nn.ParameterList`` and never match Muon's
    ``isinstance(module, nn.Linear)`` test, but the sharded GatedDeltaNet projections are
    built by ``_slice_linear`` and *are* real ``nn.Linear`` modules, so they do route to
    Muon. Tensor parallelism therefore gives a mixture, not a clean fallback to AdamW.
    """
    if not hasattr(model, "_distillkit_tp_devices"):
        return
    offenders = [
        name
        for group in groups
        if group["optimizer_kind"] == "muon"
        for name, parameter in zip(group["param_names"], group["params"])
        if parameter.requires_grad
    ]
    if offenders:
        raise ValueError(
            "Muon would receive %d trainable tensor-parallel shard(s), starting with %s. "
            "Use optimizer.strategy=adamw for a trainable backbone under tensor "
            "parallelism." % (len(offenders), offenders[0])
        )


def freeze_backbone_for_stage1(model: nn.Module) -> tuple[str, ...]:
    """Freeze currently trainable backbone weights, retaining auxiliary training.

    Call after attaching hidden-state projections. The returned names identify
    only parameters frozen here, preserving any pre-existing permanent freezes.
    """
    auxiliary_ids = _auxiliary_parameter_ids(model)
    if not any(id(p) in auxiliary_ids and p.requires_grad for p in model.parameters()):
        raise ValueError("Stage 1 requires trainable sidecar/gate/distillation projections.")
    frozen = []
    for name, parameter in model.named_parameters():
        if id(parameter) not in auxiliary_ids and parameter.requires_grad:
            parameter.requires_grad_(False)
            parameter.grad = None
            frozen.append(name)
    return tuple(frozen)


class UnfreezeBackboneCallback(TrainerCallback):
    """Unfreeze after N completed updates; also restore stage on checkpoint resume."""

    def __init__(self, unfreeze_at_step: int, frozen_parameter_names: Iterable[str]):
        if unfreeze_at_step < 0:
            raise ValueError("unfreeze_at_step must be nonnegative.")
        self.unfreeze_at_step = unfreeze_at_step
        self.frozen_parameter_names = tuple(frozen_parameter_names)
        self.unfrozen = False

    def _maybe_unfreeze(self, args, state, model, optimizer):
        validate_optimizer_backend(
            strategy="adamw", deepspeed=getattr(args, "deepspeed", None),
            fsdp=getattr(args, "fsdp", None), dynamic_unfreeze=True,
            world_size=getattr(args, "world_size", 1),
        )
        if self.unfrozen or state.global_step < self.unfreeze_at_step:
            return
        parameters = dict(model.named_parameters())
        missing = set(self.frozen_parameter_names).difference(parameters)
        if missing:
            raise ValueError(f"Frozen parameters missing from model: {sorted(missing)}")
        if optimizer is None:
            raise ValueError("Unfreezing requires an initialized optimizer containing the backbone.")
        included = {id(p) for g in optimizer.param_groups for p in g["params"]}
        if any(id(parameters[name]) not in included for name in self.frozen_parameter_names):
            raise ValueError("Optimizer omitted frozen backbone parameters; rebuild it with include_frozen=True.")
        for name in self.frozen_parameter_names:
            parameters[name].requires_grad_(True)
        self.unfrozen = True

    def on_train_begin(self, args, state, control, model=None, optimizer=None, **kwargs):
        self._maybe_unfreeze(args, state, model, optimizer)
        return control

    def on_step_begin(self, args, state, control, model=None, optimizer=None, **kwargs):
        self._maybe_unfreeze(args, state, model, optimizer)
        return control


@torch.no_grad()
def architecture_metrics(model: nn.Module) -> dict[str, float]:
    report = {}
    for name, module in model.named_modules():
        if isinstance(module, GatedResidual):
            report.update(module.gate_report(prefix=f"architecture/{name}"))
        if name.split(".")[-1] == "W_side_proj" and isinstance(module, nn.Linear):
            report[f"architecture/{name}/weight_norm"] = module.weight.float().norm().item()
    return report


def memory_metrics(model: nn.Module) -> dict[str, float]:
    """Per-device peak allocation, plus whether gradient checkpointing is really on.

    Three OOMs in this project were diagnosed by arithmetic on a traceback and twice
    the arithmetic was wrong, because a standalone probe does not reproduce what the
    Trainer actually builds -- accelerate's prepared forward, the dataloader, the
    optimizer state, and whichever of the memory-saving flags actually took effect.
    Reporting it from inside the run costs one `torch.cuda` query per logged step and
    removes the guessing.

    ``max_memory_allocated`` is reset each time this runs, so the reported peak is
    "since the last log" rather than since process start -- a steady per-step high
    water mark is more useful than a number that only ever ratchets up.
    """
    if not torch.cuda.is_available():
        return {}
    report = {}
    gib = 1024**3
    for index in range(torch.cuda.device_count()):
        try:
            peak = torch.cuda.max_memory_allocated(index)
            reserved = torch.cuda.memory_reserved(index)
            torch.cuda.reset_peak_memory_stats(index)
        except RuntimeError:
            # A visible device the process never allocated on has no allocator state
            # and raises "Invalid device argument" on reset. Nothing to report.
            continue
        report[f"vram/cuda{index}_peak_gib"] = peak / gib
        report[f"vram/cuda{index}_reserved_gib"] = reserved / gib
    # A silently inactive flag is worth many gigabytes, and `use_cache=True is
    # incompatible with gradient checkpointing` does not warn when the config already
    # has use_cache off -- so absence of that warning proves nothing either way.
    report["vram/gradient_checkpointing"] = float(
        bool(getattr(model, "is_gradient_checkpointing", False))
    )
    return report


class ArchitectureMetricsCallback(TrainerCallback):
    """Request a normal Trainer log every N steps and enrich it with gate metrics."""

    def __init__(self, every_n_steps: int = 100):
        if every_n_steps <= 0:
            raise ValueError("every_n_steps must be positive.")
        self.every_n_steps = every_n_steps
        self._last_step = -1

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step > 0 and state.global_step % self.every_n_steps == 0:
            control.should_log = True
        return control

    def enrich_logs(self, logs, state, model) -> dict[str, float]:
        """Merge gate/projection metrics into ``logs`` at most once per step.

        Returns the metrics added (empty when not due). Call this before
        ``Trainer.log`` dispatches to callbacks: reporting integrations are
        registered ahead of user callbacks and consume the dict as they see it.
        """
        if (
            logs is None or model is None
            or state.global_step % self.every_n_steps != 0
            or self._last_step == state.global_step
        ):
            return {}
        metrics = architecture_metrics(model)
        metrics.update(memory_metrics(model))
        logs.update(metrics)
        self._last_step = state.global_step
        return metrics

    def on_log(self, args, state, control, logs=None, model=None, **kwargs):
        metrics = self.enrich_logs(logs, state, model)
        if metrics:
            # Trainer appends a copy before calling callbacks; update that copy
            # too, so metrics survive save_state even with no logging backend.
            if state.log_history and state.log_history[-1].get("step") == state.global_step:
                state.log_history[-1].update(metrics)
        return control

# Copyright 2024 Charles O. Goddard

import threading

import torch
from transformers import (
    PreTrainedModel,
)
from trl import SFTTrainer

from distillkit.anchor_tap import AnchorTap
from distillkit.chunked_head import HeadContext
from distillkit.chunked_ce import keep_bf16_forward_outputs, maybe_install_chunked_loss
from distillkit.configuration import DistillationRunConfig, LossFunctionConfig
from distillkit.hsd_mapping import HiddenStateMapping
from distillkit.lossfuncs import ALL_LOSS_CLASSES, LossFunctionBase
from distillkit.signals import OnlineSignalSource, SignalSource, TeacherSignal


def create_loss_func(cfg: LossFunctionConfig) -> LossFunctionBase:
    for cls in ALL_LOSS_CLASSES:
        if cfg.function.value == cls.name():
            return cls(
                **cfg.model_dump(exclude=["function", "weight"], exclude_none=True)
            )
    raise RuntimeError(f"Unknown loss function '{cfg.function}'")


class DistillationTrainer(SFTTrainer):
    def __init__(
        self,
        model: PreTrainedModel,
        config: DistillationRunConfig,
        signal_source: SignalSource,
        true_vocab_size: int,
        *args,
        hidden_state_mapping: HiddenStateMapping | None = None,
        **kwargs,
    ):
        super().__init__(model, *args, **kwargs)
        self.true_vocab_size = true_vocab_size
        self.config = config

        self.loss_functions = [create_loss_func(lfc) for lfc in config.loss_functions]
        self.need_hidden_states = any(
            lf.requires_hidden_states() for lf in self.loss_functions
        )
        self.need_model_loss = any(lf.requires_model_loss() for lf in self.loss_functions)

        # The stock causal-LM loss keeps a full fp32 copy of the logits alive for
        # backward. Over a 248k-wide head that measured 2.84 GB where the chunked
        # form needs 1.01 GB, for the same number to within 1e-6.
        maybe_install_chunked_loss(
            model,
            need_model_loss=self.need_model_loss,
            enabled=config.chunked_cross_entropy,
        )

        self.signal_source = signal_source
        self.hidden_state_mapping = hidden_state_mapping

        if self.need_hidden_states and not self.signal_source.supports_hidden_states():
            raise ValueError(
                "Configuration requests hidden state loss, but the provided Teacher "
                "(Offline/Dataset) does not support hidden states."
            )

        if (self.hidden_state_mapping is None) and self.need_hidden_states:
            raise ValueError(
                "Must define a hidden state mapping to use hidden state losses."
            )

        if isinstance(self.signal_source, OnlineSignalSource):
            self.signal_source.teacher_model = self.signal_source.teacher_model.to(
                self.accelerator.device
            )

        self.model_accepts_loss_kwargs = False
        self._kept_bf16_outputs = False
        self.chunked_head = bool(getattr(config, "chunked_head", False))
        self.sortish_batching = bool(getattr(config, "sortish_batching", False))
        # The head loop reuses whatever chunk length the sparse divergence was tuned
        # with; they are the same positions either way.
        self._head_chunk_length = next(
            (f.sparse_chunk_length for f in config.loss_functions
             if getattr(f, "sparse_chunk_length", None)), None,
        )
        self._loss_log_local = threading.local()
        self._concurrent_pending = []
        self._concurrent_runner = None

    def train(self, *args, **kwargs):
        try:
            return super().train(*args, **kwargs)
        finally:
            self._concurrent_pending.clear()
            self._concurrent_runner = None

    def training_step(self, model, inputs, num_items_in_batch=None):
        if self.config.tensor_parallel:
            if self.accelerator.num_processes != 1 or self.accelerator.distributed_type.name != "NO" or self.accelerator.scaler is not None:
                raise ValueError("Tensor parallelism requires native single-process bf16/fp32 training")
            from distillkit.tp_model import sync_replicated_gradients
            # Cold FLA autotuners share state across device workers. Prime the first
            # backward serially; later steps retain normal autograd scheduling.
            with torch.autograd.set_multithreading_enabled(getattr(self, "_tp_warmed", False)):
                loss = super().training_step(model, inputs, num_items_in_batch)
            self._tp_warmed = True
            if self.accelerator.sync_gradients:
                sync_replicated_gradients(model)
            return loss
        if self.config.concurrent_microbatches == 1:
            return super().training_step(model, inputs, num_items_in_batch)
        if self.accelerator.num_processes != 1 or self.accelerator.distributed_type.name != "NO":
            raise ValueError("Concurrent training requires a single native process")
        if self.accelerator.scaler is not None or self.accelerator.gradient_accumulation_steps != 1:
            raise ValueError("Concurrent training requires unscaled native bf16/fp32 gradients")
        if model is not self.model or self.compute_loss_func is not None:
            raise ValueError("Concurrent training cannot use a distributed/compiled wrapper or custom loss callback")

        # HF has already collected the full accumulation window and supplies its
        # actual length (including a short final window). Earlier calls defer work;
        # the final one returns the sum of normalized microbatch losses.
        self._concurrent_pending.append(dict(inputs))
        count = self.current_gradient_accumulation_steps
        if len(self._concurrent_pending) < count:
            return torch.zeros((), device=self.args.device)
        batches, self._concurrent_pending = self._concurrent_pending, []
        from distillkit.concurrent_training import ConcurrentMicrobatches

        model.train()
        model.config.use_cache = False
        if hasattr(self.optimizer, "train"):
            self.optimizer.train()
        if not self._kept_bf16_outputs:
            keep_bf16_forward_outputs(model)
            self._kept_bf16_outputs = True
        if self._concurrent_runner is None:
            self._concurrent_runner = ConcurrentMicrobatches(model)

        def forward_loss(batch):
            self._loss_log_local.logs = []
            try:
                with self.compute_loss_context_manager():
                    loss = self.compute_loss(model, self._prepare_inputs(batch))
                return loss, self._loss_log_local.logs
            finally:
                del self._loss_log_local.logs

        value, logs = self._concurrent_runner.run(
            batches, forward_loss, warmup_first=not getattr(self, "_concurrent_warmed", False),
        )
        self._concurrent_warmed = True
        for microbatch_logs in logs:
            for entry in microbatch_logs:
                self.log(entry)
        return torch.tensor(value, device=self.args.device)

    def _get_train_sampler(self, train_dataset=None):
        if not self.sortish_batching:
            return super()._get_train_sampler(train_dataset)
        from distillkit.sortish_sampler import SortishSampler

        dataset = self.train_dataset if train_dataset is None else train_dataset
        column = self.args.length_column_name or "length"
        if column not in getattr(dataset, "column_names", []):
            raise ValueError(
                f"sortish_batching needs a `{column}` column on the training dataset"
            )
        # args.train_batch_size, deliberately not multiplied by
        # gradient_accumulation_steps the way Trainer does: padding is decided per
        # forward pass, and grouping the whole optimizer step strips the length
        # diversity out of every update. See distillkit/sortish_sampler.py.
        return SortishSampler(
            self.args.train_batch_size, dataset[column], generator=self._sortish_generator(),
        )

    def _sortish_generator(self):
        # Seeded from args.seed so a run is reproducible, and advanced per epoch so the
        # order is not identical every epoch.
        generator = torch.Generator()
        generator.manual_seed(self.args.seed + int(self.state.epoch or 0))
        return generator

    def _clip_grad_norm(self, model):
        if not self.config.tensor_parallel:
            return super()._clip_grad_norm(model)
        from distillkit.tp_gated_delta_module import clip_grad_norm
        return clip_grad_norm(model, self.args.max_grad_norm)

    def _get_grad_norm(self, model, grad_norm=None):
        if self.config.tensor_parallel and grad_norm is None:
            from distillkit.tp_gated_delta_module import clip_grad_norm
            return clip_grad_norm(model, float("inf"))
        return super()._get_grad_norm(model, grad_norm)

    def _save(self, output_dir=None, state_dict=None):
        if not self.config.tensor_parallel:
            return super()._save(output_dir, state_dict)
        from distillkit.tp_checkpoint import consolidated_state_dict, write_layout
        if state_dict is not None:
            raise ValueError("TP saving must reconstruct weights from the live shards")
        super()._save(output_dir, consolidated_state_dict(self.model))
        write_layout(self.model, output_dir or self.args.output_dir)

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        from pathlib import Path
        from distillkit.tp_checkpoint import MARKER, load_checkpoint
        if self.config.tensor_parallel:
            return load_checkpoint(self.model if model is None else model, resume_from_checkpoint)
        if (Path(resume_from_checkpoint) / MARKER).exists():
            raise ValueError("TP optimizer checkpoints require tensor_parallel: true; use model: for weights-only continuation")
        return super()._load_from_checkpoint(resume_from_checkpoint, model)

    def _load_best_model(self):
        if not self.config.tensor_parallel:
            return super()._load_best_model()
        from distillkit.tp_checkpoint import load_checkpoint
        load_checkpoint(self.model, self.state.best_model_checkpoint)

    def compute_loss(
        self,
        model: PreTrainedModel,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        **kwargs,
    ):
        if "labels" not in inputs:
            inputs["labels"] = inputs["input_ids"].clone()
            if "attention_mask" in inputs:
                inputs["labels"].masked_fill_(inputs["attention_mask"] == 0, -100)
        if self.config.dataset.eos_label_token_ids:
            inputs["labels"] = inputs["labels"].clone()
            for tok_id in self.config.dataset.eos_label_token_ids:
                inputs["labels"][inputs["labels"] == tok_id] = (
                    self.model.config.eos_token_id
                )

        # Call the wrapper: bypassing it skips DDP/DeepSpeed forward bookkeeping.
        # Withhold labels unless a configured loss reads student_outputs.loss: the
        # model would otherwise run a full-vocabulary cross-entropy whose result is
        # discarded, paying for it in both compute and activation memory.
        forwarded = ["input_ids", "attention_mask", "position_ids", "ngram_raw"]
        if self.need_model_loss:
            forwarded.append("labels")
        # accelerate wraps the prepared forward so every bf16 tensor it returns is
        # upcast to fp32 -- 3.79 GiB for the logits alone at sequence 4096, and the
        # thing that OOM'd both the control arm and the first stage-2 attempt. The
        # model is only prepared once training starts, so strip it on first use.
        if not self._kept_bf16_outputs:
            keep_bf16_forward_outputs(model)
            self._kept_bf16_outputs = True
        model_inputs = {k: inputs[k] for k in forwarded if k in inputs}
        if self.config.sidecar is not None:
            model_inputs["sidecar_enabled"] = self.config.sidecar.enabled
        # The post-norm state is lm_head's input, so folding the head into the loss
        # needs it tapped whether or not a hidden-state loss asked for it.
        if self.chunked_head:
            model_inputs["logits_to_keep"] = 1
        if self.need_hidden_states or self.chunked_head:
            # Not output_hidden_states=True: that retains all 33 states and, on a
            # device-mapped model, accelerate's output hook copies every one of them to
            # the input device -- ~0.7 GiB retained plus the same again copied, plus
            # gradients for the copies, to serve the two anchors the loss reads. Hooks
            # on just those two modules give the same tensors, on the card that made
            # them, with no copy.
            anchors = [
                student for student, _ in
                (self.hidden_state_mapping.layer_mapping if self.need_hidden_states else [])
            ]
            if self.chunked_head:
                anchors.append(self.model.config.num_hidden_layers)
            with AnchorTap(model, anchors) as tap:
                student_outputs = model(**model_inputs, return_dict=True)
            student_outputs.hidden_states = tap.states()
        else:
            student_outputs = model(**model_inputs, return_dict=True)
        # nn.DataParallel gathers one loss per replica, so student_outputs.loss arrives
        # as [n_gpu] rather than a scalar. Every downstream consumer (the cross_entropy
        # loss function, the .item() logging, the weighted sum) assumes a scalar, and
        # batch 1 hides it because DataParallel cannot split a single example.
        if student_outputs.loss is not None and student_outputs.loss.dim() > 0:
            student_outputs.loss = student_outputs.loss.mean()
        if student_outputs.logits.shape[-1] < self.true_vocab_size:
            raise ValueError("Student vocabulary is smaller than the teacher signal vocabulary")
        if not self.chunked_head and student_outputs.logits.shape[-1] != self.true_vocab_size:
            # truncate any extra logits from padding. Under chunked_head the losses
            # truncate each chunk instead, since these logits are one position wide.
            student_outputs.logits = student_outputs.logits[..., : self.true_vocab_size]

        total_loss = self.total_distillation_loss(
            student_outputs,
            inputs,
            num_items_in_batch=None,
        )
        return (total_loss, student_outputs) if return_outputs else total_loss

    def total_distillation_loss(
        self, student_outputs, inputs, num_items_in_batch: int | None = None
    ):
        valid_mask = (inputs["labels"] >= 0).unsqueeze(-1)
        if "attention_mask" in inputs:
            valid_mask = valid_mask & inputs["attention_mask"].bool().unsqueeze(-1)
        if not valid_mask.any():
            raise ValueError("Distillation batch contains no supervised token positions")
        signal: TeacherSignal = self.signal_source.get_signal(
            inputs,
            return_hidden_states=self.need_hidden_states,
        )

        head_context = None
        if self.chunked_head:
            base = self.accelerator.unwrap_model(self.model)
            head_context = HeadContext(
                student_outputs.hidden_states[base.config.num_hidden_layers],
                base.get_output_embeddings(),
                vocab_size=self.true_vocab_size,
                chunk_length=self._head_chunk_length,
            )

        losses = []
        loss_fns = []
        weights = []
        for idx, loss_fn in enumerate(self.loss_functions):
            cfg = self.config.loss_functions[idx]
            extra = {}
            if head_context is not None and loss_fn.accepts_head_context():
                extra["head_context"] = head_context
            loss = loss_fn(
                student_outputs,
                signal,
                mask=valid_mask,
                hidden_state_mapping=self.hidden_state_mapping,
                num_items_in_batch=num_items_in_batch,
                **extra,
            )
            losses.append(loss)
            loss_fns.append(cfg.function.value)
            weights.append(cfg.weight)

        # A sharded student can produce these scalars on different cards -- the KL
        # term on the head's device, a hidden-state term on its anchor's. Reduce onto
        # the device the trainer will call backward from.
        reduce_device = losses[0].device
        total_loss = 0.0
        for loss, weight in zip(losses, weights):
            total_loss = total_loss + loss.to(reduce_device) * weight
        total_loss = total_loss / sum(weights)
        metrics = {
                f"distillation_loss/{idx + 1}_{loss_fn}": loss.item()
                for idx, (loss, loss_fn) in enumerate(zip(losses, loss_fns))
            }
        pending_logs = getattr(self._loss_log_local, "logs", None)
        if pending_logs is None:
            self.log(metrics)
        else:
            pending_logs.append(metrics)
        return total_loss


class HybridDistillationTrainer(DistillationTrainer):
    """Explicit mixed optimizer selection; backend compatibility is checked at setup."""

    def log(self, logs, start_time=None):
        # Reporting integrations (W&B, TensorBoard) are registered ahead of user
        # callbacks and consume the dict inside super().log; enrich it first so
        # they receive gate/projection metrics alongside ordinary loss metrics.
        from distillkit.optimizers import ArchitectureMetricsCallback

        for callback in self.callback_handler.callbacks:
            if isinstance(callback, ArchitectureMetricsCallback):
                callback.enrich_logs(logs, self.state, self.model)
        super().log(logs, start_time=start_time)

    def create_optimizer(self):
        from distillkit.optimizers import build_mixed_optimizer

        if self.optimizer is None and self.config.optimizer.strategy == "hybrid":
            self.optimizer = build_mixed_optimizer(
                self.model,
                lr=self.args.learning_rate,
                muon_lr=self.config.optimizer.muon_lr,
                weight_decay=self.args.weight_decay,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
                include_frozen=True,
            )
        elif self.optimizer is None and self.config.optimizer.unfreeze_at_step:
            # HF filters currently frozen parameters; a later unfreeze needs them
            # registered from the outset, even though their state stays unallocated.
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=self.args.learning_rate,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon, weight_decay=self.args.weight_decay,
            )
        else:
            self.optimizer = super().create_optimizer()
        _apply_sidecar_lr(
            self.optimizer, self.accelerator.unwrap_model(self.model),
            getattr(self.config.optimizer, "sidecar_lr", None),
        )
        return self.optimizer


def _apply_sidecar_lr(optimizer, model, sidecar_lr):
    """Give the sidecar its own learning rate, leaving the backbone on the run's.

    Stage 2 runs the backbone at 1e-5, which moves the sidecar barely at all: the
    chained run's ``W_side_proj`` went 2.1055 -> 2.1077 across an entire epoch, and the
    PLE module's weights did not change to five significant figures over fifty logged
    steps. The backbone then adapts around a sidecar that is effectively frozen, which
    is not what "unlocking both" was supposed to mean.

    Splits the auxiliary parameters out of whichever groups HF put them in, preserving
    each group's weight decay, and re-adds them at ``sidecar_lr``. Done inside
    ``create_optimizer`` so the scheduler, built afterwards, records the right
    ``initial_lr`` per group and scales them proportionally.
    """
    if sidecar_lr is None:
        return optimizer
    from distillkit.optimizers import _auxiliary_parameter_ids

    auxiliary = _auxiliary_parameter_ids(model)
    moved: dict[float, list] = {}
    for group in optimizer.param_groups:
        kept = []
        for parameter in group["params"]:
            if id(parameter) in auxiliary:
                moved.setdefault(group.get("weight_decay", 0.0), []).append(parameter)
            else:
                kept.append(parameter)
        group["params"] = kept
    for weight_decay, params in moved.items():
        optimizer.add_param_group(
            {"params": params, "lr": sidecar_lr, "weight_decay": weight_decay}
        )
    return optimizer

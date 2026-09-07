# Copyright 2024 Charles O. Goddard

import torch
from transformers import (
    PreTrainedModel,
)
from trl import SFTTrainer

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
        student_outputs = model(
            **model_inputs,
            return_dict=True,
            output_hidden_states=self.need_hidden_states,
        )
        # nn.DataParallel gathers one loss per replica, so student_outputs.loss arrives
        # as [n_gpu] rather than a scalar. Every downstream consumer (the cross_entropy
        # loss function, the .item() logging, the weighted sum) assumes a scalar, and
        # batch 1 hides it because DataParallel cannot split a single example.
        if student_outputs.loss is not None and student_outputs.loss.dim() > 0:
            student_outputs.loss = student_outputs.loss.mean()
        if student_outputs.logits.shape[-1] < self.true_vocab_size:
            raise ValueError("Student vocabulary is smaller than the teacher signal vocabulary")
        if student_outputs.logits.shape[-1] != self.true_vocab_size:
            # truncate any extra logits from padding
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

        losses = []
        loss_fns = []
        weights = []
        for idx, loss_fn in enumerate(self.loss_functions):
            cfg = self.config.loss_functions[idx]
            loss = loss_fn(
                student_outputs,
                signal,
                mask=valid_mask,
                hidden_state_mapping=self.hidden_state_mapping,
                num_items_in_batch=num_items_in_batch,
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
        self.log(
            {
                f"distillation_loss/{idx + 1}_{loss_fn}": loss.item()
                for idx, (loss, loss_fn) in enumerate(zip(losses, loss_fns))
            }
        )
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
            return super().create_optimizer()
        return self.optimizer

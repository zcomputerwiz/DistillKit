# Copyright 2025 Arcee AI
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator
from typing_extensions import TypeAlias

from distillkit.compression.config import (
    DistributionQuantizationConfig,
    LegacyLogitCompressionConfig,
)
from distillkit.missing_probability import MissingProbabilityHandling


class LossFunction(str, Enum):
    CROSS_ENTROPY = "cross_entropy"
    ASSISTANT_CROSS_ENTROPY = "assistant_cross_entropy"
    KL = "kl"
    JSD = "jsd"
    TVD = "tvd"
    HINGE = "hinge"
    LOGISTIC_RANKING = "logistic_ranking"
    HIDDEN_STATE_COSINE = "hs_cosine"
    HIDDEN_STATE_MSE = "hs_mse"


class LossFunctionConfig(BaseModel):
    function: LossFunction = Field(
        ...,
        description="Type of loss function to use.",
    )
    weight: float = Field(
        ...,
        description="Weight for the loss function.",
    )
    temperature: float | None = Field(
        default=None,
        description="Temperature for loss, if applicable.",
    )
    missing_probability_handling: MissingProbabilityHandling | None = Field(
        default=None,
        description="Missing probability handling mode for sparse divergence functions.",
    )
    sparse_chunk_length: int | None = Field(
        default=None,
        description="Chunk length for sparse divergence functions. None to disable chunking.",
    )
    margin: float | None = Field(
        default=None,
        description="Margin for hinge loss, if applicable.",
    )


class HfRepoDataset(BaseModel):
    repo_id: str = Field(
        description="Hugging Face repository ID of the dataset.",
    )
    revision: str | None = Field(
        default=None,
        description="Revision of the dataset to use.",
    )
    config_name: str | None = Field(
        default=None,
        description="Configuration name of the dataset.",
    )
    split: str | None = Field(
        default=None,
        description="Split of the dataset to use.",
    )


class LocalDataset(BaseModel):
    disk_path: str = Field(
        description="Path to the local dataset or dataset dict directory.",
    )
    split: str | None = Field(
        default=None,
        description="Split of the dataset to use.",
    )


DatasetPath: TypeAlias = HfRepoDataset | LocalDataset


class DatasetConfiguration(BaseModel):
    train_dataset: DatasetPath | None = Field(
        default=None,
        description="Dataset to use for training.",
    )
    eval_dataset: DatasetPath | None = Field(
        default=None,
        description="Dataset to use for evaluation.",
    )
    seed: int | None = Field(
        default=42,
        description="Random seed for shuffling datasets.",
    )
    num_samples: int | None = Field(
        default=None,
        description="Number of samples to use from the dataset.",
    )
    num_eval_samples: int | None = Field(
        default=None,
        description="Number of samples to use from the evaluation dataset.",
    )
    eos_label_token_ids: list[int] | None = Field(
        default=None,
        description="List of token IDs to replace with EOS token IDs in the labels.",
    )
    prepared_dataset_path: str | None = Field(
        default=None,
        description="Path to store prepared dataset.",
    )
    prepacked: bool = Field(
        default=False,
        description="Assume dataset is pretokenized and packed, skip TRL packing.",
    )


class TeacherModelConfig(BaseModel):
    kind: Literal["hf"] = "hf"

    path: str
    kwargs: dict[str, Any] | None = None

    top_k: int | None = None


class TeacherDatasetConfig(BaseModel):
    kind: Literal["dataset"] = "dataset"
    cache_path: str | None = None
    anchor_layers: list[int] | None = None
    cache_dtype: Literal["float8_e4m3fn"] = "float8_e4m3fn"
    legacy_logit_compression: LegacyLogitCompressionConfig | None = Field(
        default=None,
        description="Legacy logit compression configuration. Must match configuration used to capture logits.",
    )
    logprob_compressor: DistributionQuantizationConfig | None = Field(
        default=None,
        description="Logit compression configuration. Must match configuration used to capture logits.",
    )


class SidecarConfig(BaseModel):
    table_path: str | None = None
    enabled: bool = True
    resident: bool = False
    prefault: bool = True
    layer_index: int = Field(default=1, ge=0)
    num_branches: int = Field(default=4, ge=1)
    variant: Literal["gated_residual", "ple", "ple_gated"] = Field(
        default="gated_residual",
        description=(
            "gated_residual: the original design -- features added ungated, then a "
            "learned multi-branch gate on the sum. ple: Flash-Next's integration "
            "transcribed, where the gate is a query-key dot product between the stream "
            "and the n-gram embedding. ple_gated: the same integration with the key path "
            "replaced by learned directions, one gate per residual branch over a shared "
            "value -- it requires residual_stream and ignores num_branches, taking its "
            "stream count from there. See gate_directions."
        ),
    )
    gate_directions: int = Field(
        default=2, ge=1,
        description=(
            "ple_gated only: learned directions per residual branch, combined as "
            "2*mean so every width starts at an admission of 1.0. One direction was "
            "indistinguishable from four on a single stream; the default is 2 because "
            "that ablation could not test per-branch admission, which is the point here."
        ),
    )

    @model_validator(mode="after")
    def require_table(self):
        if self.enabled and not self.table_path:
            raise ValueError("sidecar.table_path is required when enabled")
        return self


class ResidualStreamConfig(BaseModel):
    """Persistent residual widening; omit this section for the original model."""
    num_branches: int = Field(default=2, ge=1)
    lowrank: int = Field(default=64, ge=1)
    routing: Literal["widened", "flash_next"] = Field(
        default="widened", description="Legacy identity-anchored routing or donor Gated Residual arithmetic.")
    blend: float = Field(default=0.0, ge=0, le=1, allow_inf_nan=False,
                         description="Flash-Next interpolation: 0 is exact student, 1 is donor routing.")
    blend_target: float = Field(default=1.0, ge=0, le=1, allow_inf_nan=False)
    blend_warmup_steps: int = Field(default=0, ge=0,
        description="Optimizer steps to interpolate blend to blend_target; 0 keeps blend fixed.")
    init_from: str | None = Field(
        default=None,
        description=(
            "Directory of extracted Flash-Next hyper-connection routing "
            "(scratch/extract_flashnext_hc.py). Select routing=flash_next for donor "
            "semantics; widened preserves the historical borrowed initialization. Requires num_branches and "
            "lowrank to match the extraction -- 4 and 320 -- because every tensor is "
            "sized 4*2560; a mismatch is refused rather than reshaped. Initialisation "
            "only: the routing trains from there like any other parameter."
        ),
    )
    init_layer_map: Literal["proportional", "identity"] = Field(
        default="proportional",
        description=(
            "How this model's layer i picks a donor block. Flash-Next has 48 layers to "
            "this student's 32, so 'proportional' takes block round(i * donors / ours) "
            "and 'identity' takes block i, using only the bottom 32. Neither is known "
            "to be right: the depth sweep found this student wants the sidecar at layer "
            "24 of 32 while Flash-Next puts its PLE at block 1 of 48, so position in "
            "the stack demonstrably does not transfer. This is an experiment handle."
        ),
    )

    learnable_blend: bool = Field(
        default=False,
        description=(
            "Train the blend instead of scheduling it, one scalar per sublayer, so the "
            "run reports where each of them wants to sit rather than being told. It is "
            "unbounded on purpose -- where it settles is the measurement. Mutually "
            "exclusive with blend_warmup_steps, which would overwrite it every step. "
            "Needs blend_lr: the shared 1e-4 moves a parameter about lr per step, so 72 "
            "steps is a budget of 0.0072 and a blend starting at 0.10 could reach 0.107."
        ),
    )
    blend_lr: float | None = Field(
        default=None, gt=0,
        description=(
            "Learning rate for the blend scalars alone, leaving everything else on its "
            "own rate. Reaching 0.25 from 0.10 inside 72 steps needs about 2e-3, and "
            "0.50 needs about 5.6e-3."
        ),
    )

    @model_validator(mode="after")
    def validate_blend_routing(self):
        if self.routing != "flash_next" and (
            self.blend != 0 or self.blend_target != 1 or self.blend_warmup_steps
            or self.learnable_blend or self.blend_lr is not None
        ):
            raise ValueError("blend settings require routing=flash_next")
        if self.learnable_blend and self.blend_warmup_steps:
            raise ValueError(
                "learnable_blend and blend_warmup_steps both write the blend; the "
                "warmup callback would overwrite the learned value after every step"
            )
        if self.blend_lr is not None and not self.learnable_blend:
            raise ValueError("blend_lr only applies with learnable_blend")
        return self


class OptimizerConfig(BaseModel):
    strategy: Literal["hybrid", "adamw"] = "hybrid"
    muon_lr: float | None = Field(default=None, gt=0)
    sidecar_lr: float | None = Field(
        default=None, gt=0,
        description=(
            "Learning rate for the sidecar, gate and distillation projections only, "
            "leaving the backbone on training_args.learning_rate. Stage 2's 1e-5 moves "
            "them barely at all -- the chained run's W_side_proj went 2.1055 -> 2.1077 "
            "across an entire epoch, and the PLE module's norms did not move to five "
            "significant figures -- so the backbone adapts around a sidecar that is "
            "effectively frozen. Only meaningful with strategy=adamw; under hybrid the "
            "auxiliary parameters already have their own routing and MixedMuonAdamW "
            "refuses added groups."
        ),
    )
    freeze_backbone: bool = True
    unfreeze_at_step: int | None = Field(default=None, ge=1)
    log_every_n_steps: int = Field(default=100, ge=1)


    @model_validator(mode="after")
    def sidecar_lr_needs_adamw(self):
        if self.sidecar_lr is not None and self.strategy != "adamw":
            raise ValueError(
                "optimizer.sidecar_lr requires strategy=adamw; MixedMuonAdamW refuses "
                "added parameter groups, and under hybrid the auxiliary parameters are "
                "already routed to AdamW as their own bucket"
            )
        return self


class DistillationRunConfig(BaseModel):
    tensor_parallel: bool = Field(
        default=False, description="Shard Qwen3.5 across cuda:0 and cuda:1 using single-process CUDA P2P.",
    )
    concurrent_microbatches: Literal[1, 2] = Field(
        default=1, description="Opt-in bounded two-worker forward overlap on a GPU-sharded student.",
    )
    sortish_batching: bool = Field(
        default=False,
        description=(
            "Length-group each microbatch and shuffle the batch order, instead of "
            "HF's group_by_length. Keeps the padding saving without the ordering cost: "
            "group_by_length grouped at the optimizer step and emitted descending "
            "lengths, which measured 0.0145 of eval_loss on a control where grouping "
            "changed nothing but the order."
        ),
    )
    chunked_head: bool = Field(
        default=False,
        description=(
            "Project lm_head inside the loss chunk loop instead of materializing "
            "[batch, seq, vocab] logits. Saves 1.89 GiB plus its gradient at sequence "
            "4096 over a 248,320-wide head, for identical loss and gradients."
        ),
    )
    project_name: str = Field(
        default="distillkit",
        description="Project name for logging.",
    )
    train_model: str = Field(
        description="Model to train.",
        alias="model",
    )
    dataset: DatasetConfiguration
    teacher: TeacherModelConfig | TeacherDatasetConfig = Field(
        ..., discriminator="kind"
    )
    sequence_length: int = Field(
        description="Sequence length for training.",
    )
    output_path: str = Field(
        description="Path to save the model.",
    )
    resize_embeddings_to_multiple_of: int | None = Field(
        default=None,
        description="Resize embeddings to a multiple of this value.",
    )
    use_flash_attention: bool = Field(
        default=True,
        description="Use flash attention for training.",
    )

    loss_functions: list[LossFunctionConfig] = Field(
        description="List of loss functions to use for distillation.",
        default_factory=lambda: [
            LossFunctionConfig(
                function=LossFunction.CROSS_ENTROPY,
                weight=0.5,
            ),
            LossFunctionConfig(
                function=LossFunction.KL,
                weight=0.5,
                temperature=1.0,
                missing_probability_handling=MissingProbabilityHandling.ZERO,
            ),
        ],
    )
    layer_mapping: list[tuple[int, int]] | Literal["all"] | None = Field(
        default=None,
        description='List of (student_layer_idx, teacher_layer_idx) pairs (or "all" for a complete one-to-one mapping.)',
    )
    force_hidden_state_projection: bool = Field(
        default=False,
        description="Use linear layers to project between teacher and student hidden states even if sizes are equal.",
    )
    max_vram_fraction: float | None = Field(
        default=None,
        gt=0.0,
        le=1.0,
        description=(
            "Cap PyTorch's share of each GPU. On Windows WDDM an over-budget run does "
            "not OOM: the driver spills to shared system memory and services it over "
            "PCIe, which looks like 100% GPU utilisation at ~40% power with an idle "
            "memory controller and runs ~7x slower. Capping the allocator turns that "
            "silent degradation into an ordinary OOM."
        ),
    )
    chunked_cross_entropy: bool = Field(
        default=True,
        description=(
            "Compute the model's causal-LM cross-entropy in chunks instead of "
            "upcasting the whole logits tensor to fp32. Over a 248k-wide head this "
            "measured 13.15 GB vs 18.76 GB peak for a 7% throughput cost (the "
            "checkpoint recompute). Set false when VRAM is not the constraint. Has no "
            "effect unless a configured loss reads the model's own loss."
        ),
    )
    functionary_packing: bool = Field(
        default=False,
        description="Use functionary's packing code. Requires flash attention and may not be compatible with all models.",
    )
    training_args: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional arguments for the trainer.",
    )
    model_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional arguments for the model.",
    )
    model_auto_class: str | None = Field(
        default="AutoModelForCausalLM",
        description="Auto class for the model.",
    )
    sidecar: SidecarConfig | None = None
    residual_stream: ResidualStreamConfig | None = None
    optimizer: OptimizerConfig | None = None
    trust_remote_code: bool = Field(
        default=False,
        description="Trust remote code when loading the model.",
    )
    frozen_modules: list[str] | None = Field(
        default=None,
        description="List of modules to freeze during training.",
    )
    frozen_res: list[str] | None = Field(
        default=None,
        description="List of regular expressions matching names of parameters to freeze during training.",
    )

    @model_validator(mode="after")
    def validate_offline_and_sidecar(self):
        cached = isinstance(self.teacher, TeacherDatasetConfig) and self.teacher.cache_path
        if self.tensor_parallel:
            if not cached or not self.optimizer:
                raise ValueError("Tensor parallel training requires an offline cache")
            if self.optimizer.strategy != "adamw" and not (
                self.optimizer.freeze_backbone and self.optimizer.unfreeze_at_step is None
            ):
                # Muon and tensor parallelism are incompatible two ways over, and both
                # matter only for parameters that actually receive updates.
                #
                # Routing: mixed_parameter_groups identifies Muon's matrices with
                # isinstance(module, nn.Linear). Column- and row-parallel weights are
                # nn.Parameter inside an nn.ParameterList and so fall through to AdamW
                # while the config still says "hybrid" -- but the sharded GatedDeltaNet
                # projections are built by _slice_linear and *are* nn.Linear, so they
                # still route to Muon. Tensor parallelism thus yields an inconsistent
                # mixture rather than a clean fallback, which is worse than either.
                #
                # Mathematics: even with the routing fixed, Newton-Schulz
                # orthogonalization does not commute with slicing, so Muon on a shard is
                # not the shard of Muon on the whole matrix.
                #
                # With the backbone frozen for the whole run, neither applies: every
                # Muon-routed parameter is sharded and frozen, and every trainable one
                # (sidecar, gated residual, distillation projections) is auxiliary and
                # goes to AdamW, so "hybrid" is exactly AdamW-on-auxiliary. Stage 1
                # measured 0.0M trainable parameters in the Muon group against 65.5M in
                # AdamW's; tests/test_optimizers.py pins that invariant, because this
                # relaxation is unsound the moment it stops holding.
                raise ValueError(
                    "Tensor parallel training requires optimizer.strategy=adamw, or "
                    "strategy=hybrid with freeze_backbone and no unfreeze_at_step "
                    "(where Muon receives no trainable parameter)"
                )
            if self.concurrent_microbatches != 1:
                raise ValueError("Tensor parallelism cannot be combined with concurrent_microbatches")
            if self.model_kwargs.get("device_map") is not None:
                raise ValueError("Tensor parallelism owns placement; omit model_kwargs.device_map")
            if any(self.training_args.get(k) for k in ("fp16", "deepspeed", "fsdp", "activation_offloading", "torch_compile", "use_cpu")):
                raise ValueError("Tensor parallelism requires native single-process CUDA bf16/fp32 training")
            if self.optimizer.unfreeze_at_step:
                raise ValueError("Tensor parallelism does not support mid-run unfreezing")
            if self.training_args.get("gradient_checkpointing"):
                options = self.training_args.get("gradient_checkpointing_kwargs") or {}
                if options.get("use_reentrant", True):
                    raise ValueError("Tensor parallel checkpoints require use_reentrant=False")
        if self.sortish_batching and self.training_args.get("train_sampling_strategy"):
            raise ValueError(
                "sortish_batching replaces train_sampling_strategy; set one or the other"
            )
        if any(f.function == LossFunction.ASSISTANT_CROSS_ENTROPY for f in self.loss_functions):
            if not self.chunked_head:
                raise ValueError("assistant_cross_entropy requires chunked_head")
            if (self.functionary_packing or self.dataset.prepacked
                    or self.training_args.get("packing") or self.training_args.get("padding_free")):
                raise ValueError("assistant_cross_entropy requires unpacked documents")
        if self.chunked_head:
            # The forward runs with logits_to_keep=1, so student_outputs.logits covers
            # one position. cross_entropy reads the model's own loss over the full
            # head, which cannot be computed from that.
            if any(f.function.value == "cross_entropy" for f in self.loss_functions):
                raise ValueError("chunked_head is incompatible with the cross_entropy loss")
            if not any(f.function.value in ("kl", "jsd", "tvd", "assistant_cross_entropy") for f in self.loss_functions):
                raise ValueError("chunked_head needs a sparse divergence or assistant_cross_entropy loss to fold the head into")
        if self.concurrent_microbatches == 2:
            if not cached or not self.optimizer or self.optimizer.strategy != "adamw":
                raise ValueError("Concurrent training requires an offline cache and optimizer.strategy=adamw")
            if self.optimizer.unfreeze_at_step:
                raise ValueError("Concurrent training does not support mid-run unfreezing")
            if any(self.training_args.get(k) for k in ("fp16", "deepspeed", "fsdp", "activation_offloading", "torch_compile")):
                raise ValueError("Concurrent training requires native bf16/fp32 without distributed sharding")
            if self.training_args.get("gradient_accumulation_steps", 1) < 2:
                raise ValueError("Concurrent training needs gradient_accumulation_steps >= 2")
            if self.training_args.get("gradient_checkpointing"):
                options = self.training_args.get("gradient_checkpointing_kwargs") or {}
                if options.get("use_reentrant", True) or options.get("preserve_rng_state", True):
                    raise ValueError("Concurrent checkpoints require use_reentrant=False and preserve_rng_state=False")
        if not cached and self.dataset.train_dataset is None:
            raise ValueError("dataset.train_dataset or teacher.cache_path is required")
        if cached and (self.dataset.train_dataset or self.dataset.eval_dataset):
            raise ValueError("cache_path supplies both datasets; omit separate dataset paths")
        if cached and (self.teacher.logprob_compressor or self.teacher.legacy_logit_compression):
            raise ValueError("memmap cache already contains raw top-k; omit compressor configs")
        if cached or self.sidecar:
            if self.functionary_packing or self.training_args.get("packing") or self.training_args.get("padding_free"):
                raise ValueError("packing and padding_free must be disabled for sidecar/cache alignment")
            if self.training_args.get("remove_unused_columns", False):
                raise ValueError("remove_unused_columns must be false to preserve doc_id and ngram_raw")
        if self.sidecar and self.sidecar.resident and self.training_args.get("dataloader_num_workers", 0):
            raise ValueError("resident tables require dataloader_num_workers=0 to prevent worker copies")
        if self.sidecar and self.resize_embeddings_to_multiple_of is not None:
            raise ValueError("sidecar preserves the original padded vocabulary; omit embedding resize")
        if self.optimizer and self.optimizer.freeze_backbone and not (self.sidecar or self.residual_stream):
            raise ValueError("stage-1 backbone freezing requires a sidecar or residual_stream student")
        return self

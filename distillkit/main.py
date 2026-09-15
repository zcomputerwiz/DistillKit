# Copyright 2025 Arcee AI
import collections
import hashlib
import importlib.util
import json
import logging
import os
import re
from typing import Any

import click
import datasets
import torch
import transformers
import trl
import yaml
from accelerate import Accelerator

from distillkit.compression import LogprobCompressor
from distillkit.configuration import (
    DatasetConfiguration,
    DatasetPath,
    DistillationRunConfig,
    HfRepoDataset,
    LocalDataset,
    TeacherDatasetConfig,
    TeacherModelConfig,
)
from distillkit.core.frozen_prefix import no_grad_prefix
from distillkit.experimental.residual_gate import attach_residual_gates
from distillkit.hsd_mapping import HiddenStateMapping
from distillkit.models.qwen35 import (
    install_device_aware_linear_attention,
    install_expanded_gqa_attention,
)
from distillkit.monkey_patch_packing import monkey_patch_packing_for_model
from distillkit.sharding import (
    as_device,
    check_tied_embeddings_colocated,
    is_sharded,
)
from distillkit.signals import OfflineSignalSource, OnlineSignalSource, SignalSource, OfflineHiddenStateSignalSource
from distillkit.trainer import DistillationTrainer, HybridDistillationTrainer

LOG = logging.getLogger(__name__)

# flash-linear-attention, when installed, is bound by transformers at import time with
# no device check, and its Triton kernels reject CPU tensors. Restore per-call dispatch
# so CPU capture/verification runs keep working. Idempotent; no-op without fla.
install_device_aware_linear_attention()
# Qwen3.5's 16:4 query/KV head ratio reaches SDPA as enable_gqa=True, which this
# build has no fused kernel for -- it silently picks the math kernel and 4328 MiB
# per attention call at sequence 4096 instead of 249.
install_expanded_gqa_attention()


def _format_row(
    example: dict[str, Any], tokenizer: transformers.PreTrainedTokenizer
) -> dict[str, Any]:
    if ("input_ids" in example) or ("text" in example):
        # either pretokenized or raw completion - no formatting needed
        return {}
    elif "conversations" in example:
        conversations = example["conversations"]

        messages = []
        for conversation in conversations:
            role_map = {
                "human": "user",
                "user": "user",
                "gpt": "assistant",
                "assistant": "assistant",
                "system": "system",
            }
            role = role_map.get(conversation.get("from", ""), None)
            if role:
                messages.append(
                    {"role": role, "content": conversation.get("value", "")}
                )

        # Apply chat template to create a single string. SFTTrainer will handle tokenization.
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        return {"text": text}
    elif "messages" in example:
        text = tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )
        return {"text": text}
    else:
        raise RuntimeError("Expected `text`, `messages`, or `conversations` column")


def _load_dataset(
    path: DatasetPath,
    seed: int | None,
    num_samples: int | None,
    tokenizer: transformers.PreTrainedTokenizer,
    prepared_dataset_path: str | None = None,
    keep_in_memory: bool | None = None,
    prepacked: bool = False,
) -> datasets.Dataset:
    if prepared_dataset_path:
        honk = json.dumps(
            {
                "path": path.model_dump(),
                "seed": seed,
                "num_samples": num_samples,
            }
        )
        logging.info(f"Dataset spec: {honk}")
        ds_hash = hashlib.sha256(honk.encode()).hexdigest()
        full_prepared_path = os.path.join(prepared_dataset_path, f"dataset-{ds_hash}")
        if os.path.exists(full_prepared_path):
            return datasets.load_from_disk(full_prepared_path)
    else:
        full_prepared_path = None
    if isinstance(path, HfRepoDataset):
        res = datasets.load_dataset(
            path.repo_id,
            name=path.config_name,
            revision=path.revision,
            split=path.split,
            keep_in_memory=keep_in_memory,
        )
    elif isinstance(path, LocalDataset):
        res = datasets.load_from_disk(path.disk_path, keep_in_memory=keep_in_memory)
        if path.split:
            res = res[path.split]
        elif isinstance(res, datasets.DatasetDict):
            raise ValueError(
                "Dataset dict found but no split specified. Please specify a split."
            )
    else:
        raise ValueError(
            "Unsupported dataset type. Please provide a valid Hugging Face repo ID or local dataset path."
        )

    if prepacked:
        last_idx = len(res) - 1
        while len(res) >= 2 and len(res[last_idx]["input_ids"]) != len(
            res[0]["input_ids"]
        ):
            last_idx -= 1
        if last_idx <= 0:
            raise RuntimeError("Dataset config is probs wrong")
        res = res.select(range(last_idx + 1))

    if seed:
        res = res.shuffle(seed=seed)
    if num_samples:
        res = res.select(range(num_samples))
    if (
        (not prepacked)
        and ("text" not in res.column_names)
        and ("input_ids" not in res.column_names)
    ):
        res = res.map(
            _format_row,
            remove_columns=res.column_names,
            fn_kwargs={"tokenizer": tokenizer},
        )
    if full_prepared_path:
        os.makedirs(full_prepared_path, exist_ok=True)
        logging.info(
            f"Saving prepared dataset to {full_prepared_path} (hash: {ds_hash}, path: {path}, seed: {seed}, num_samples: {num_samples})"
        )
        res.save_to_disk(full_prepared_path)
        del res
        return datasets.load_from_disk(
            full_prepared_path, keep_in_memory=keep_in_memory
        )
    return res


def load_data(
    config: DatasetConfiguration,
    tokenizer: transformers.PreTrainedTokenizer,
    keep_in_memory: bool | None = None,
) -> tuple[datasets.Dataset, datasets.Dataset | None]:
    """
    Load the train (and optionally eval) datasets as specified in the configuration.
    """

    LOG.info(
        f"Loading datasets: {config.train_dataset} (train), {config.eval_dataset} (eval)"
    )
    ds_train = _load_dataset(
        config.train_dataset,
        config.seed,
        config.num_samples,
        tokenizer=tokenizer,
        prepared_dataset_path=config.prepared_dataset_path,
        keep_in_memory=keep_in_memory,
        prepacked=config.prepacked,
    )
    ds_eval = None
    if config.eval_dataset:
        ds_eval = _load_dataset(
            config.eval_dataset,
            config.seed,
            config.num_eval_samples,
            tokenizer=tokenizer,
            prepared_dataset_path=config.prepared_dataset_path,
            keep_in_memory=keep_in_memory,
            prepacked=config.prepacked,
        )
    return ds_train, ds_eval


from distillkit.models import load_student_model



def create_signal_source(
    config: DistillationRunConfig, vocab_size: int, tokenizer=None
) -> SignalSource:
    if isinstance(config.teacher, TeacherDatasetConfig):
        if config.teacher.cache_path:
            from distillkit.offline_cache import tokenizer_vocab_hash
            return OfflineHiddenStateSignalSource(
                config.teacher.cache_path,
                expected_anchor_layers=config.teacher.anchor_layers,
                expected_cache_dtype=config.teacher.cache_dtype,
                expected_sequence_length=config.sequence_length,
                expected_tokenizer_vocab_hash=tokenizer_vocab_hash(tokenizer) if tokenizer is not None else None,
            )
        compressor = LogprobCompressor(
            config=config.teacher.logprob_compressor,
            legacy_config=config.teacher.legacy_logit_compression,
        )
        return OfflineSignalSource(compressor, vocab_size=vocab_size)
    elif isinstance(config.teacher, TeacherModelConfig):
        teacher_model = transformers.AutoModelForCausalLM.from_pretrained(
            config.teacher.path, **(config.teacher.kwargs or {})
        )
        return OnlineSignalSource(
            teacher_model, vocab_size=vocab_size, sparsify_top_k=config.teacher.top_k
        )
    else:
        raise RuntimeError("Teacher configuration invalid")


from distillkit.data import collate_packed_batch, create_data_collator



def load_tokenizer(config: DistillationRunConfig) -> transformers.PreTrainedTokenizer:
    if isinstance(config.teacher, TeacherModelConfig):
        src_path = config.teacher.path
        logging.info("Using teacher's tokenizer")
    else:
        src_path = config.train_model
        logging.info("Using student's tokenizer")
    return transformers.AutoTokenizer.from_pretrained(
        src_path,
        trust_remote_code=config.trust_remote_code,
    )


def do_distill(config: DistillationRunConfig, config_source: str | None = None,
               initialise_only: bool = False):
    os.makedirs(config.output_path, exist_ok=True)
    if config_source is None:
        config_source = yaml.safe_dump(config.model_dump(mode="json", by_alias=True))
    with open(os.path.join(config.output_path, "distillkit_config.yaml"), "w") as f:
        f.write(config_source)

    if config.project_name:
        os.environ["WANDB_PROJECT"] = config.project_name

    accelerator = Accelerator()
    with accelerator.main_process_first():
        tokenizer = load_tokenizer(config)

        tokenizer_vocab_size = max(
            len(tokenizer.get_vocab()),
            max(tokenizer.get_vocab().values()) + 1,
        )
    signal_source = create_signal_source(config, tokenizer_vocab_size, tokenizer)
    if isinstance(signal_source, OfflineHiddenStateSignalSource):
        ds_train = signal_source.cache.to_dataset("train")
        ds_eval = signal_source.cache.to_dataset("eval")
        if not len(ds_train):
            raise ValueError("Cache has no training documents")
        if not len(ds_eval):
            ds_eval = None
        signal_vocab_size = signal_source.vocab_size
    else:
        ds_train, ds_eval = load_data(config.dataset, tokenizer)
        signal_vocab_size = tokenizer_vocab_size

    if config.high_resolution_timer and os.name == "nt":
        try:
            import atexit
            import ctypes
            ctypes.windll.winmm.timeBeginPeriod(1)
            atexit.register(ctypes.windll.winmm.timeEndPeriod, 1)
            LOG.info("Windows high-resolution timer (1ms) enabled.")
        except Exception as e:
            LOG.warning("Could not enable Windows high-resolution timer: %s", e)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = bool(config.allow_tf32)
        torch.backends.cudnn.allow_tf32 = bool(config.allow_tf32)
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high" if config.allow_tf32 else "highest")
        LOG.info(
            "Configured TensorFloat-32 (TF32): matmul=%s, cudnn=%s",
            torch.backends.cuda.matmul.allow_tf32,
            torch.backends.cudnn.allow_tf32,
        )

    if config.cuda_allocator_gc_threshold is not None and torch.cuda.is_available():
        current_alloc_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
        if "garbage_collection_threshold" not in current_alloc_conf:
            gc_setting = f"garbage_collection_threshold:{config.cuda_allocator_gc_threshold}"
            new_conf = f"{current_alloc_conf},{gc_setting}" if current_alloc_conf else gc_setting
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = new_conf
            LOG.info("Configured PYTORCH_CUDA_ALLOC_CONF=%s", new_conf)

    if config.max_vram_fraction is not None and torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            torch.cuda.set_per_process_memory_fraction(config.max_vram_fraction, index)
        LOG.info(
            "Capped PyTorch to %.0f%% of VRAM; over-budget allocations now raise "
            "instead of spilling to shared system memory.",
            100 * config.max_vram_fraction,
        )

    model = load_student_model(config, tokenizer_vocab_size, signal_vocab_size)
    if initialise_only:
        # The frozen rho sweep grades an arm *before* any optimizer step, so the
        # thing it grades has to be the model training would have started from --
        # same config parsing, same reference loading, same trainability contract.
        # Saved here rather than after sharding: a sharded save is a different code
        # path, and nothing below this point can run without touching the optimizer.
        LOG.info("Saving initialised model without training to %s", config.output_path)
        model.save_pretrained(config.output_path)
        tokenizer.save_pretrained(config.output_path)
        LOG.info("Done.")
        return
    if config.tensor_parallel:
        from distillkit.parallel import shard_model
        if torch.cuda.device_count() < 2 or int(os.environ.get("WORLD_SIZE", "1")) != 1:
            raise ValueError("Tensor parallel training requires two visible GPUs in one process")
        shard_model(model, ["cuda:0", "cuda:1"])
    if is_sharded(model):
        # A single-process layer split needs no process group: Tensor.to(device) is
        # differentiable, so autograd moves activations forward and gradients back
        # itself, over NVLink where the pair supports peer access. HF Trainer sees
        # hf_device_map, sets place_model_on_device=False and forces _n_gpu to 1, so
        # it will not also try to wrap this in DataParallel.
        check_tied_embeddings_colocated(model)
        placement = collections.Counter(str(as_device(v)) for v in model.hf_device_map.values())
        LOG.info("Student sharded across %s", dict(placement))
    if model.config.vocab_size < signal_vocab_size:
        raise ValueError("Student head is smaller than the cache vocabulary")
    if config.sidecar is not None and not config.sidecar.enabled:
        # Control arm: forward runs with sidecar_enabled=False, so W_side_proj is
        # permanently bypassed. A trainable-but-unused parameter makes DDP's
        # find_unused_parameters=False raise on the next reduction; freeze it
        # before optimizer/distributed setup. The gated residual stays trainable.
        model.disable_sidecar_projection()

    config_kwargs = dict(config.training_args)
    resume_from_checkpoint = config_kwargs.pop("resume_from_checkpoint", None)
    dataset_kwargs = config_kwargs.pop("dataset_kwargs", {})
    if config.dataset.prepacked or isinstance(signal_source, OfflineHiddenStateSignalSource):
        dataset_kwargs["skip_prepare_dataset"] = True
    if config.sidecar or isinstance(signal_source, OfflineHiddenStateSignalSource):
        config_kwargs["remove_unused_columns"] = False
        config_kwargs["packing"] = False
        config_kwargs["padding_free"] = False
    if config.optimizer and config.optimizer.strategy == "adamw":
        config_kwargs.setdefault("optim", "adamw_torch")
    max_length = config_kwargs.pop("max_length", config.sequence_length)
    training_arguments = trl.SFTConfig(
        **config_kwargs,
        max_length=max_length,
        output_dir=config.output_path,
        dataset_kwargs=dataset_kwargs,
    )

    if config.layer_mapping is not None:
        if isinstance(signal_source, OfflineHiddenStateSignalSource):
            teacher_hidden_size = signal_source.hidden_size
            if config.layer_mapping == "all":
                raise ValueError("Cached anchors require explicit (student_layer, compact_anchor_index) pairs")
            mapping = config.layer_mapping
            if any(t < 0 or t >= len(signal_source.anchor_layers) for _, t in mapping):
                raise ValueError("Teacher mapping index must address the compact cached anchor tuple")
        elif isinstance(signal_source, OnlineSignalSource):
            teacher_hidden_size = signal_source.teacher_model.config.hidden_size
            mapping = ([(i, i) for i in range(model.config.num_hidden_layers)]
                       if config.layer_mapping == "all" else config.layer_mapping)
        else:
            raise RuntimeError(
                "Hidden state distillation not supported for offline teachers"
            )
        if any(s < 0 or s > model.config.num_hidden_layers for s, _ in mapping):
            raise ValueError("Student hidden-state index outside model depth")
        hsm = HiddenStateMapping(
            student=model,
            teacher_hidden_size=teacher_hidden_size,
            layer_mapping=mapping,
            force_projection=config.force_hidden_state_projection,
        )
    else:
        hsm = None
    collator = create_data_collator(
        config,
        tokenizer,
        signal_source=signal_source,
        model=model,
        world_size=training_arguments.world_size,
    )
    # Installed and calibrated before the optimizer exists, so the calibration pass runs
    # against a model nothing has moved yet. Returns None when the run has no gate.
    gate_handle = attach_residual_gates(config, model, ds_train)
    from distillkit.optimizers import ReleaseEvalCacheCallback

    # Evaluation carves the allocator's pool into its own shapes, and Windows cannot
    # defragment; this belongs to every run, not just the ones with an optimizer
    # section. See ReleaseEvalCacheCallback.
    callbacks = [ReleaseEvalCacheCallback()]
    if config.residual_stream and config.residual_stream.routing == "flash_next":
        from distillkit.hyper_connection import HyperConnectionWarmupCallback
        stream = config.residual_stream
        if stream.blend_warmup_steps:
            callbacks.append(HyperConnectionWarmupCallback(
                stream.blend, stream.blend_target, stream.blend_warmup_steps))
    if config.optimizer:
        from distillkit.optimizers import (
            validate_optimizer_backend, freeze_backbone_for_stage1,
            UnfreezeBackboneCallback, ArchitectureMetricsCallback,
        )
        validate_optimizer_backend(
            strategy=config.optimizer.strategy, deepspeed=training_arguments.deepspeed,
            fsdp=training_arguments.fsdp,
            dynamic_unfreeze=config.optimizer.unfreeze_at_step is not None,
            # Read from the training args, not the Accelerator created earlier:
            # constructing SFTConfig can reset AcceleratorState, after which the
            # earlier instance raises on attribute access. Same number, stable source.
            world_size=training_arguments.world_size,
        )
        if config.optimizer.freeze_sidecar:
            from distillkit.optimizers import freeze_sidecar_parameters
            frozen_sidecar = freeze_sidecar_parameters(model)
            # Zero is the ordinary case for a control arm: `sidecar.enabled: false`
            # already froze the block. Report the invariant, not the work done, or the
            # line reads as though nothing is held fixed.
            LOG.info("Sidecar held fixed while the backbone trains (%d frozen here, "
                     "the rest already frozen by the disabled-sidecar path)",
                     len(frozen_sidecar))
        if config.optimizer.freeze_backbone:
            window = config.optimizer.stage1_trainable_layers
            if window is not None:
                if not hasattr(model, "set_stage1_trainable_layers"):
                    raise ValueError(
                        "stage1_trainable_layers needs the widened model class")
                model.set_stage1_trainable_layers(window)
                LOG.info("Stage 1 also trains decoder layers [%d, %d)", *window)
            frozen_names = freeze_backbone_for_stage1(model)
            # Only reentrant checkpointing needs this. It drops the graph when a
            # checkpointed block's inputs do not require grad, which a frozen prefix
            # guarantees, so the hook exists to force differentiable embeddings.
            # Non-reentrant checkpointing has no such requirement, and installing the
            # hook there is actively harmful: every activation before the first
            # trainable module becomes differentiable and is retained for a backward
            # pass that cannot use it. See distillkit/frozen_prefix.py.
            checkpointing = config.training_args.get("gradient_checkpointing")
            reentrant = (config.training_args.get("gradient_checkpointing_kwargs") or {}
                         ).get("use_reentrant", True)
            if checkpointing and reentrant and hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            elif not config.optimizer.unfreeze_at_step:
                # Nothing before the injection point can move, so the graph autograd
                # builds across the embedding and the prefix layers is constructed and
                # then discarded. Running that stretch under no_grad is the same
                # arithmetic without the retained activations. Only the prefix: the
                # suffix stays differentiable because the table learns through its
                # Jacobian. Skipped when the backbone unfreezes mid-run, since the
                # prefix would then need the gradients this discards.
                index = getattr(model.config, "sidecar_layer_index", None)
                if index is None and config.residual_gate:
                    # Same argument, different injection point: nothing below the
                    # shallowest gated layer can move, so the graph autograd builds
                    # across it is constructed and discarded.
                    index = min(config.residual_gate.layers)
                if index is not None:
                    try:
                        no_grad_prefix(model, upto_layer=index)
                    except ValueError as error:
                        LOG.info("Prefix stays in autograd: %s", error)
                    else:
                        LOG.info("Embeddings and decoder layers [0, %d) run outside "
                                 "autograd", index)
            if config.optimizer.unfreeze_at_step:
                callbacks.append(UnfreezeBackboneCallback(config.optimizer.unfreeze_at_step, frozen_names))
        callbacks.append(ArchitectureMetricsCallback(config.optimizer.log_every_n_steps))
    if gate_handle is not None:
        from distillkit.experimental.residual_gate import (
            ResidualGateCheckpointCallback)
        # The backbone is frozen in stage 1, so a full checkpoint every eval would write
        # the same 2B model five times. Only the gate changes; only the gate is saved.
        callbacks.append(ResidualGateCheckpointCallback(config.output_path))
    trainer_class = HybridDistillationTrainer if config.optimizer else DistillationTrainer
    trainer = trainer_class(
        model=model,
        config=config,
        signal_source=signal_source,
        hidden_state_mapping=hsm,
        true_vocab_size=signal_vocab_size,
        train_dataset=ds_train,
        eval_dataset=ds_eval,
        args=training_arguments,
        data_collator=collator,
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    LOG.info("Starting training.")
    trainer.train(
        resume_from_checkpoint=resume_from_checkpoint,
    )
    LOG.info(f"Finished training. Saving model to {config.output_path}.")
    trainer.save_model(config.output_path)
    LOG.info("Done.")


@click.command("distillkit-offline")
@click.argument(
    "config_path",
    type=click.Path(exists=True, dir_okay=False, readable=True),
)
@click.option(
    "--verbose",
    "-v",
    "verbosity",
    count=True,
    help="Increase verbosity of logging. Use -vv for debug level.",
)
@click.option(
    "--initialise-only",
    is_flag=True,
    help="Build and save the model without training, for a frozen pre-training sweep.",
)
def main(config_path: str, verbosity: int, initialise_only: bool = False):
    log_level = logging.WARNING
    if verbosity >= 2:
        log_level = logging.DEBUG
    elif verbosity == 1:
        log_level = logging.INFO
    logging.basicConfig(level=log_level)
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    config = DistillationRunConfig.model_validate(config_dict)
    do_distill(config, initialise_only=initialise_only)


if __name__ == "__main__":
    # torch.autograd.set_detect_anomaly(True)
    main()

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
from distillkit.hsd_mapping import HiddenStateMapping
from distillkit.gqa_dispatch import install_expanded_gqa_attention
from distillkit.linear_attention_dispatch import install_device_aware_linear_attention
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


def load_student_model(
    config: DistillationRunConfig,
    tokenizer_vocab_size: int,
    signal_vocab_size: int | None = None,
) -> transformers.PreTrainedModel:
    residual_stream = getattr(config, "residual_stream", None)
    if config.functionary_packing:
        monkey_patch_packing_for_model(config.train_model)
    if residual_stream is not None:
        from distillkit.models.qwen35_widened import Qwen35WidenedForCausalLM
        auto_cls = Qwen35WidenedForCausalLM
    elif config.sidecar is not None:
        from distillkit.models.qwen35_sidecar import Qwen35SidecarForCausalLM
        auto_cls = Qwen35SidecarForCausalLM
    else:
        auto_cls = getattr(transformers, config.model_auto_class, None)
    if auto_cls is None:
        raise ValueError(
            f"Model class {config.model_auto_class} not found in transformers."
        )
    LOG.info(f"Loading model {config.train_model} with class {auto_cls}")
    extra_kwargs = {"trust_remote_code": config.trust_remote_code}
    if config.use_flash_attention:
        if importlib.util.find_spec("flash_attn") is None:
            # from_pretrained's own failure here names the package but not the two
            # consequences of turning the flag off, and it surfaces only after the
            # dataset and teacher cache have already been built.
            raise RuntimeError(
                "use_flash_attention is true but flash_attn is not installed "
                "(it has no Windows wheels). Install it, or set "
                "use_flash_attention: false and set training_args.bf16 so the "
                "student still loads in bfloat16."
            )
        extra_kwargs["attn_implementation"] = "flash_attention_2"
        extra_kwargs["torch_dtype"] = torch.bfloat16
    extra_kwargs.update(config.model_kwargs)
    if "torch_dtype" not in extra_kwargs:
        # Only the flash-attention branch ever set a dtype, so with
        # `use_flash_attention: false` the dtype came entirely from the checkpoint's
        # own config -- fine for `student-hf`, which records bfloat16, and silently
        # fp32 for any checkpoint that does not. autocast does not shrink the weights,
        # so honour the trainer's own mixed-precision flags when nothing else has
        # chosen.
        if config.training_args.get("bf16"):
            extra_kwargs["torch_dtype"] = torch.bfloat16
        elif config.training_args.get("fp16"):
            extra_kwargs["torch_dtype"] = torch.float16
    # Inspect saved architecture before choosing overrides: loading widened weights
    # with the stock class would silently discard trained routing parameters.
    config_kwargs = {key: extra_kwargs[key] for key in (
        "revision", "cache_dir", "local_files_only", "token", "subfolder",
        "trust_remote_code") if key in extra_kwargs}
    stock_config = transformers.AutoConfig.from_pretrained(config.train_model, **config_kwargs)
    text_config = getattr(stock_config, "text_config", stock_config)
    if getattr(text_config, "residual_stream_enabled", False):
        if residual_stream is None:
            raise ValueError("Widened checkpoint requires its matching residual_stream run configuration")
        expected = (text_config.residual_stream_num_branches, text_config.residual_stream_lowrank,
                    getattr(text_config, "residual_stream_sidecar", False))
        requested = (residual_stream.num_branches, residual_stream.lowrank, config.sidecar is not None)
        if expected != requested:
            raise ValueError(f"Widened checkpoint architecture {expected} differs from requested {requested}")
        if config.sidecar is not None and (
            text_config.sidecar_layer_index != config.sidecar.layer_index
            or text_config.sidecar_num_branches != config.sidecar.num_branches
            or text_config.sidecar_variant != config.sidecar.variant
            # Only ple_gated has directions. Comparing them unconditionally would
            # reject every checkpoint saved before the field existed, whose config
            # carries no such key and whose variant does not use one.
            or (config.sidecar.variant == "ple_gated"
                and getattr(text_config, "sidecar_gate_directions", None)
                != config.sidecar.gate_directions)
        ):
            raise ValueError("Widened checkpoint sidecar architecture differs from requested sidecar")
    if config.sidecar is not None or residual_stream is not None:
        if config.sidecar is not None:
            text_config.sidecar_layer_index = config.sidecar.layer_index
            text_config.sidecar_num_branches = config.sidecar.num_branches
            text_config.sidecar_variant = config.sidecar.variant
            text_config.sidecar_gate_directions = config.sidecar.gate_directions
        if residual_stream is not None:
            text_config.residual_stream_enabled = True
            text_config.residual_stream_num_branches = residual_stream.num_branches
            text_config.residual_stream_lowrank = residual_stream.lowrank
            text_config.residual_stream_sidecar = config.sidecar is not None
        extra_kwargs["config"] = text_config
    model = auto_cls.from_pretrained(
        config.train_model,
        **extra_kwargs,
    )
    LOG.info("Loaded model.")

    if residual_stream is not None and residual_stream.init_from:
        # Before sharding and before the optimizer: this writes parameters in place,
        # and a copy after either would be copying into the wrong object.
        from distillkit.borrowed_routing import initialise_widened_residual

        LOG.info("Borrowed routing: %s", initialise_widened_residual(
            model, residual_stream.init_from, residual_stream.init_layer_map))

    model_vocab_size = model.get_input_embeddings().weight.shape[0]
    required_vocab_size = max(tokenizer_vocab_size, signal_vocab_size or 0)
    if config.sidecar is not None or residual_stream is not None:
        if model_vocab_size < required_vocab_size:
            raise ValueError(
                "Student head does not cover the tokenizer/signal vocabulary"
            )
    else:
        # Only a cached signal normalized over a padded (larger-than-tokenizer)
        # head forbids growth. An online teacher's signal_vocab_size equals the
        # tokenizer size and must still reach the resize path below.
        if (
            signal_vocab_size is not None
            and signal_vocab_size > tokenizer_vocab_size
            and model_vocab_size < signal_vocab_size
        ):
            # Growing would fabricate rows for IDs the cached signals already
            # normalize over, changing the captured distribution.
            raise ValueError(
                f"Student head ({model_vocab_size}) is smaller than the cached "
                f"signal vocabulary ({signal_vocab_size}); re-capture or use a "
                f"student whose head covers the cache vocabulary"
            )
        # A cached signal normalized over a padded teacher head must not be
        # shrunk to the tokenizer size; only resize when that cannot break
        # signal coverage.
        preserves_padded_head = (
            signal_vocab_size is not None
            and signal_vocab_size > tokenizer_vocab_size
            and model_vocab_size >= signal_vocab_size
        )
        if preserves_padded_head:
            LOG.info(
                f"Preserving padded student head of {model_vocab_size} entries "
                f"(tokenizer vocab {tokenizer_vocab_size}) to cover the cached "
                f"signal vocabulary"
            )
        elif (
            model_vocab_size != tokenizer_vocab_size
            or config.resize_embeddings_to_multiple_of
        ):
            model.resize_token_embeddings(
                tokenizer_vocab_size,
                pad_to_multiple_of=config.resize_embeddings_to_multiple_of,
            )
            new_model_vocab_size = model.get_input_embeddings().weight.shape[0]
            if new_model_vocab_size != model_vocab_size:
                LOG.info(
                    f"Resized model vocab size from {model_vocab_size} to {new_model_vocab_size}"
                )

    model: transformers.PreTrainedModel
    if config.frozen_modules:
        module_set = set(config.frozen_modules)
        seen = set()
        for name, module in model.named_modules():
            if name in module_set:
                module.requires_grad_(False)
                seen.add(name)
        unseen = module_set - seen
        LOG.info(f"Froze {len(seen)} modules")
        if unseen:
            raise ValueError(f"Frozen modules not found in model: {', '.join(unseen)}")
    if config.frozen_res:
        num_frozen = 0
        frozen_res = [re.compile(s) for s in config.frozen_res]
        for name, param in model.named_parameters():
            if any(fre.search(name) for fre in frozen_res):
                param.requires_grad = False
                num_frozen += 1
        if num_frozen:
            print(f"Froze {num_frozen} tensors by regular expression")
    return model


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


def collate_packed_batch(examples):
    # all sequences in the batch already have the same length
    # so we can directly stack them
    return {
        key: torch.tensor([example[key] for example in examples])
        for key in examples[0].keys()
    }


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


def do_distill(config: DistillationRunConfig, config_source: str | None = None):
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

    if config.max_vram_fraction is not None and torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            torch.cuda.set_per_process_memory_fraction(config.max_vram_fraction, index)
        LOG.info(
            "Capped PyTorch to %.0f%% of VRAM; over-budget allocations now raise "
            "instead of spilling to shared system memory.",
            100 * config.max_vram_fraction,
        )

    model = load_student_model(config, tokenizer_vocab_size, signal_vocab_size)
    if config.tensor_parallel:
        from distillkit.tp_model import shard_model
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
    from distillkit.data import CachedBatchCollator
    if isinstance(signal_source, OfflineHiddenStateSignalSource):
        collator = CachedBatchCollator(tokenizer.pad_token_id or tokenizer.eos_token_id)
    elif config.dataset.prepacked:
        collator = collate_packed_batch
    else:
        # Leave ordinary collation to SFTTrainer so packing/padding_free/
        # completion_only_loss are honored. TRL rejects a custom collator when
        # BFD packing enables padding-free mode, so an unconditional collator
        # here breaks packing=True configurations (e.g. examples/afm_test.yml).
        collator = None
    if config.sidecar and config.sidecar.enabled:
        from trl.trainer.sft_trainer import DataCollatorForLanguageModeling
        from distillkit.ngram_table import GGUFNGramTable
        from distillkit.sidecar_collator import SidecarDataCollator
        # The sidecar wraps a concrete base collator. Packing is forced off for
        # sidecar runs above, so a plain LM collator is valid when none was set.
        base_collator = (
            collator if collator is not None else DataCollatorForLanguageModeling(
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        )
        table = GGUFNGramTable(config.sidecar.table_path)
        if config.sidecar.resident:
            # training_arguments, not the Accelerator created before SFTConfig: building
            # SFTConfig resets AcceleratorState, after which that instance raises.
            if training_arguments.world_size > 1:
                raise ValueError("Resident table duplication across distributed ranks is unsupported; use memmap")
            table.load_resident()
        elif config.sidecar.prefault:
            table.prefault()
        collator = SidecarDataCollator(base_collator, table)
    from distillkit.optimizers import ReleaseEvalCacheCallback

    # Evaluation carves the allocator's pool into its own shapes, and Windows cannot
    # defragment; this belongs to every run, not just the ones with an optimizer
    # section. See ReleaseEvalCacheCallback.
    callbacks = [ReleaseEvalCacheCallback()]
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
        if config.optimizer.freeze_backbone:
            frozen_names = freeze_backbone_for_stage1(model)
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            if config.optimizer.unfreeze_at_step:
                callbacks.append(UnfreezeBackboneCallback(config.optimizer.unfreeze_at_step, frozen_names))
        callbacks.append(ArchitectureMetricsCallback(config.optimizer.log_every_n_steps))
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
def main(config_path: str, verbosity: int):
    log_level = logging.WARNING
    if verbosity >= 2:
        log_level = logging.DEBUG
    elif verbosity == 1:
        log_level = logging.INFO
    logging.basicConfig(level=log_level)
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    config = DistillationRunConfig.model_validate(config_dict)
    do_distill(config)


if __name__ == "__main__":
    # torch.autograd.set_detect_anomaly(True)
    main()

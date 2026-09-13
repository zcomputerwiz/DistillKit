"""Padding without dropping the document identity needed for offline alignment."""
from dataclasses import dataclass
from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence


@dataclass
class CachedBatchCollator:
    pad_token_id: int

    def __call__(self, examples):
        if not examples:
            raise ValueError("Cannot collate an empty batch")
        ids = [torch.as_tensor(e["input_ids"], dtype=torch.long) for e in examples]
        masks = [torch.as_tensor(e.get("attention_mask", [1] * len(t)), dtype=torch.long)
                 for e, t in zip(examples, ids)]
        labels = [torch.as_tensor(e.get("labels", e["input_ids"]), dtype=torch.long).clone()
                  for e in examples]
        for t, m, label in zip(ids, masks, labels):
            if t.ndim != 1 or len(t) < 2 or m.shape != t.shape or label.shape != t.shape:
                raise ValueError("Each cached document needs at least two tokens and matching labels/mask")
            label[m == 0] = -100
        return {
            "doc_id": [e["doc_id"] for e in examples],
            "input_ids": pad_sequence(ids, batch_first=True, padding_value=self.pad_token_id),
            "attention_mask": pad_sequence(masks, batch_first=True, padding_value=0),
            "labels": pad_sequence(labels, batch_first=True, padding_value=-100),
        }


def collate_packed_batch(examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    """Collate examples where all sequences have already been prepacked to uniform length."""
    return {
        key: torch.tensor([example[key] for example in examples])
        for key in examples[0].keys()
    }


def create_data_collator(
    config: Any,
    tokenizer: Any,
    signal_source: Any = None,
    model: Any = None,
    world_size: int = 1,
) -> Any:
    """Factory for selecting and configuring the appropriate data collator for training."""
    from distillkit.signals import OfflineHiddenStateSignalSource

    if isinstance(signal_source, OfflineHiddenStateSignalSource):
        collator = CachedBatchCollator(tokenizer.pad_token_id or tokenizer.eos_token_id)
    elif getattr(getattr(config, "dataset", None), "prepacked", False):
        collator = collate_packed_batch
    else:
        # Leave ordinary collation to SFTTrainer so packing/padding_free/
        # completion_only_loss are honored. TRL rejects a custom collator when
        # BFD packing enables padding-free mode, so an unconditional collator
        # here breaks packing=True configurations (e.g. examples/afm_test.yml).
        collator = None

    if getattr(config, "sidecar", None) and config.sidecar.enabled:
        from trl.trainer.sft_trainer import DataCollatorForLanguageModeling
        from distillkit.experimental.ngram_table import GGUFNGramTable
        from distillkit.experimental.sidecar_collator import SidecarDataCollator

        base_collator = (
            collator
            if collator is not None
            else DataCollatorForLanguageModeling(
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
            )
        )
        table = None
        hasher = None
        if config.sidecar.table_mode == "native":
            # No GGUF at all: the rows are model parameters, so collation stops at the
            # indices. The hash geometry has to be the model's own -- a collator hashing
            # into a different address space than the table it feeds is an out-of-range
            # lookup at best and silently wrong rows at worst.
            from distillkit.experimental.native_ple import native_hash_config
            from distillkit.experimental.ngram_hash import NGramHasher

            text_config = getattr(model.config, "text_config", model.config) if model else None
            hasher = NGramHasher(native_hash_config(text_config)) if text_config else None
        else:
            table = GGUFNGramTable(config.sidecar.table_path)
            if config.sidecar.resident:
                if world_size > 1:
                    raise ValueError(
                        "Resident table duplication across distributed ranks is unsupported; use memmap"
                    )
                table.load_resident()
            elif config.sidecar.prefault:
                table.prefault()
        collator = SidecarDataCollator(
            base_collator,
            table,
            hasher=hasher,
            shuffle_context=config.sidecar.shuffle_context,
            mode=config.sidecar.table_mode,
        )

    return collator


__all__ = [
    "CachedBatchCollator",
    "collate_packed_batch",
    "create_data_collator",
]

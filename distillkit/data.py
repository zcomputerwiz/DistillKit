"""Padding without dropping the document identity needed for offline alignment."""
from dataclasses import dataclass

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

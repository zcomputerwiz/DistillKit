import torch
from transformers.modeling_outputs import CausalLMOutput
from typing_extensions import override

from distillkit.hsd_mapping import HiddenStateMapping
from distillkit.lossfuncs.common import (
    LossFunctionBase,
)
from distillkit.signals import TeacherSignal


def assistant_token_mask(input_ids, attention_mask, tokenizer):
    """Label-position mask using the independent evaluator's role-span convention.

    Decode/re-encode must preserve IDs exactly; silently applying offsets to a
    different tokenization is unsafe. Padding is removed before parsing and then
    restored. Plain text without chat role markers follows role_spans (assistant).
    Chat headers and empty think blocks are template, and turn endings belong to
    the preceding body, just as in independent evaluation.
    """
    from distillkit.independent_eval import role_spans

    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    if attention_mask.shape != input_ids.shape or input_ids.ndim != 2:
        raise ValueError("assistant CE requires unpacked [batch, sequence] tokens and mask")
    host_ids = input_ids.detach().cpu()
    host_mask = attention_mask.detach().cpu().bool()
    result = torch.zeros_like(host_mask)
    for row in range(len(host_ids)):
        positions = host_mask[row].nonzero().flatten()
        if not positions.numel():
            continue
        if not torch.equal(positions, torch.arange(positions[0], positions[-1]+1)):
            raise ValueError("assistant CE requires contiguous document tokens")
        ids = host_ids[row, positions].tolist()
        text = tokenizer.decode(ids, skip_special_tokens=False,
                                clean_up_tokenization_spaces=False)
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        if encoded["input_ids"] != ids:
            raise ValueError("assistant CE role tokenization does not match input_ids exactly")
        for start, stop in role_spans(text, encoded["offset_mapping"]).get("assistant", []):
            result[row, positions[start:stop]] = True
    return result.to(input_ids.device)


def _ground_truth_ce_sum(logits, target_ids, target_values, mask):
    """One head chunk; TP composes its normalizer without gathering vocab shards."""
    if hasattr(logits, "sparse_logprobs"):
        return -logits.sparse_logprobs(target_ids).sum()
    return torch.nn.functional.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        target_ids.to(logits.device).reshape(-1), reduction="sum",
    )


class AssistantCrossEntropyLoss(LossFunctionBase):
    """Causal ground-truth CE on assistant targets, projected through the shared head.

    Requires chunked_head so the model forward never allocates sequence-wide
    logits. Mean over selected, non-ignored next-token targets, with a graph-
    connected zero when a microbatch has no assistant targets. Uses the complete
    student head vocabulary, independently of teacher vocabulary truncation.
    """

    @classmethod
    def name(cls):
        return "assistant_cross_entropy"

    def __init__(self, sparse_chunk_length: int | None = None):
        if sparse_chunk_length is not None and sparse_chunk_length < 1:
            raise ValueError("assistant CE sparse_chunk_length must be positive")
        self.chunk_length = sparse_chunk_length

    def accepts_head_context(self):
        return True

    def requires_token_targets(self):
        return True

    def __call__(self, student_outputs, signal, mask=None, hidden_state_mapping=None,
                 num_items_in_batch=None, head_context=None, labels=None,
                 assistant_mask=None, attention_mask=None):
        from distillkit.chunked_ce import chunk_tokens_for
        from distillkit.chunked_head import chunked_head_loss

        if head_context is None or labels is None or assistant_mask is None:
            raise ValueError("assistant CE needs chunked_head, labels and assistant_mask")
        hidden = head_context.hidden_states
        if labels.shape != hidden.shape[:2] or assistant_mask.shape != labels.shape:
            raise ValueError("assistant CE labels and mask must match hidden-state positions")
        labels = labels.to(hidden.device)
        selected = assistant_mask.to(hidden.device).bool()[:, 1:] & labels[:, 1:].ge(0)
        if attention_mask is not None:
            valid = attention_mask.to(hidden.device).bool()
            selected = selected & valid[:, 1:] & valid[:, :-1]
        targets = labels[:, 1:][selected].reshape(1, -1, 1)
        if targets.numel() == 0:
            return hidden.sum() * 0.0
        head = head_context.head
        vocab_size = (sum(p.shape[0] for p in head.shards)
                      if hasattr(head, "shards") else head.weight.shape[0])
        if torch.any(targets >= vocab_size):
            raise ValueError("assistant CE target outside the student's vocabulary")
        # A fixed token chunk alone still grows with vocabulary. Bound fp32
        # logits to 128 MiB too (minimum one row); no [batch, seq, vocab] exists.
        chunk = min(self.chunk_length or head_context.chunk_length or 256,
                    chunk_tokens_for(vocab_size))
        selected_hidden = hidden[:, :-1][selected].unsqueeze(0)
        loss = chunked_head_loss(selected_hidden, head, targets, targets, None,
                                 chunk, _ground_truth_ce_sum)
        denominator = targets.numel() if num_items_in_batch is None else num_items_in_batch
        return loss / torch.as_tensor(denominator, device=loss.device).clamp_min(1)


class CrossEntropyLoss(LossFunctionBase):
    @override
    @classmethod
    def name(cls) -> str:
        return "cross_entropy"

    @override
    def requires_model_loss(self) -> bool:
        return True

    @override
    def __init__(self): ...

    @override
    def __call__(
        self,
        student_outputs: CausalLMOutput,
        signal: TeacherSignal,
        mask: torch.Tensor | None = None,
        hidden_state_mapping: HiddenStateMapping | None = None,
        num_items_in_batch: int | None = None,
    ) -> torch.Tensor:
        if student_outputs.loss is None:
            raise ValueError(
                "cross_entropy loss needs the model's own loss, but the forward ran "
                "without labels. requires_model_loss() should have kept them."
            )
        return student_outputs.loss

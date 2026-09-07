from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, TypeAlias

import torch
import transformers
from typing_extensions import override

from distillkit.compression import LogprobCompressor


@dataclass
class TeacherSignalBase:
    generation_temperature: float
    hidden_states: tuple[torch.Tensor, ...] | None
    vocab_size: int


@dataclass
class DenseSignal(TeacherSignalBase):
    logits: torch.Tensor


@dataclass
class SparseSignal(TeacherSignalBase):
    sparse_ids: torch.LongTensor
    sparse_values: torch.Tensor
    log_values: bool  # if True, values are logprobs


TeacherSignal: TypeAlias = SparseSignal | DenseSignal


class SignalSource(ABC):
    @abstractmethod
    def supports_hidden_states(self) -> bool: ...

    @abstractmethod
    def get_signal(
        self, batch: dict[str, Any], return_hidden_states: bool = False
    ) -> TeacherSignal: ...


class OfflineSignalSource(SignalSource):
    compressor: LogprobCompressor
    preapplied_temperature: float
    vocab_size: int
    log_values: bool

    def __init__(
        self,
        compressor: LogprobCompressor,
        vocab_size: int,
        preapplied_temperature: float = 1.0,
        log_values: bool = True,
    ):
        self.compressor = compressor
        self.vocab_size = vocab_size
        self.preapplied_temperature = preapplied_temperature
        self.log_values = log_values

    @override
    def supports_hidden_states(self) -> bool:
        return False

    @override
    def get_signal(
        self, batch: dict[str, Any], return_hidden_states: bool = False
    ) -> SparseSignal:
        if return_hidden_states:
            raise RuntimeError(
                "Hidden states requested but signal source is precomputed logits"
            )
        with torch.no_grad():
            sparse_ids, sparse_values = self.compressor.decompress_to_sparse(batch)
        return SparseSignal(
            sparse_ids=sparse_ids,
            sparse_values=sparse_values,
            log_values=self.log_values,
            generation_temperature=self.preapplied_temperature,
            hidden_states=None,
            vocab_size=self.vocab_size,
        )


class OfflineHiddenStateSignalSource(SignalSource):
    """Load a compact anchor tuple and top-k log probabilities by document ID.

    The collator must preserve ``doc_id`` and the exact captured tokens. Only
    contiguous left/right padding is accepted; cropped windows and packing
    require new teacher captures because they change the visible prefix.
    """

    def __init__(self, cache_path, **cache_validation):
        from distillkit.offline_cache import OfflineTeacherCache

        self.cache = OfflineTeacherCache(cache_path, **cache_validation)
        self.vocab_size = self.cache.vocab_size
        self.anchor_layers = self.cache.anchor_layers
        self.hidden_size = self.cache.hidden_size
        self.preapplied_temperature = 1.0
        self.log_values = True

    @override
    def supports_hidden_states(self) -> bool:
        return True

    @override
    def get_signal(self, batch: dict[str, Any], return_hidden_states: bool = False) -> SparseSignal:
        import numpy as np

        tokens = batch.get("input_ids")
        if not isinstance(tokens, torch.Tensor) or tokens.ndim != 2:
            raise ValueError("Offline cache requires a [batch, sequence] input_ids tensor")
        doc_ids = batch.get("doc_id")
        if not isinstance(doc_ids, (list, tuple)) or len(doc_ids) != tokens.shape[0]:
            raise ValueError("Offline cache requires one doc_id string per batch row")
        if any(not isinstance(doc_id, str) for doc_id in doc_ids):
            raise ValueError("Offline cache doc_id values must be strings")
        mask = batch.get("attention_mask", torch.ones_like(tokens))
        if not isinstance(mask, torch.Tensor) or mask.shape != tokens.shape:
            raise ValueError("Offline cache attention_mask must match input_ids")
        if not torch.all((mask == 0) | (mask == 1)):
            raise ValueError("Offline cache requires a binary attention_mask; packing is unsupported")
        host_tokens = tokens.detach().cpu().numpy()
        host_mask = mask.detach().bool().cpu().numpy()
        batch_size, seq_length = tokens.shape
        top_k = self.cache.manifest["top_k"]
        sparse_ids = torch.zeros((batch_size, seq_length, top_k), dtype=torch.long)
        sparse_values = torch.full((batch_size, seq_length, top_k), -1e4, dtype=torch.float16)
        # Assembled in the cache's own float8_e4m3fn, not bfloat16. The upcast used to
        # happen here on the host, which doubled the largest host-to-device payload of
        # the step (36.9 MB per microbatch instead of 18.4) for a conversion the GPU
        # does for free. It matters more once the student is split across cards.
        hidden_states = (
            tuple(torch.zeros((batch_size, seq_length, self.hidden_size), dtype=torch.float8_e4m3fn)
                  for _ in self.anchor_layers) if return_hidden_states else None
        )
        with torch.no_grad():
            for row, doc_id in enumerate(doc_ids):
                positions = np.flatnonzero(host_mask[row])
                if not len(positions) or not np.array_equal(positions, np.arange(positions[0], positions[-1] + 1)):
                    raise ValueError("Offline cache requires nonempty contiguous document tokens")
                cached = self.cache.read_document(doc_id, include_hidden_states=return_hidden_states)
                if not np.array_equal(host_tokens[row, positions], cached["input_ids"]):
                    raise ValueError(f"Offline cache token mismatch for document {doc_id!r}; truncation/packing is unsupported")
                start, end = int(positions[0]), int(positions[-1]) + 1
                sparse_ids[row, start:end] = torch.from_numpy(cached["topk_ids"].astype(np.int64))
                sparse_values[row, start:end] = torch.from_numpy(cached["topk_logprobs"])
                if hidden_states is not None:
                    decoded = torch.from_numpy(cached["hidden_states"]).view(torch.float8_e4m3fn)
                    for anchor, target in enumerate(hidden_states):
                        target[row, start:end] = decoded[:, anchor, :]
            return SparseSignal(
                sparse_ids=sparse_ids.to(tokens.device), sparse_values=sparse_values.to(tokens.device),
                log_values=True, generation_temperature=1.0,
                # Transfer in fp8, widen on the device. The finiteness check moves here
                # with it: float8_e4m3fn has NaN but no infinities, and a NaN survives
                # the widening, so checking after the copy catches the same corruption.
                hidden_states=tuple(_to_bfloat16_on(h, tokens.device) for h in hidden_states)
                if hidden_states is not None else None,
                vocab_size=self.vocab_size,
            )



def _to_bfloat16_on(cached_fp8: torch.Tensor, device: torch.device) -> torch.Tensor:
    widened = cached_fp8.to(device=device, non_blocking=True).to(torch.bfloat16)
    if not torch.isfinite(widened).all():
        raise ValueError("Nonfinite hidden-state cache")
    return widened


class OnlineSignalSource(SignalSource):
    teacher_model: transformers.PreTrainedModel
    vocab_size: int
    sparsify_top_k: int | None
    teacher_kwargs: dict[str, Any]

    def __init__(
        self,
        teacher_model: transformers.PreTrainedModel,
        vocab_size: int,
        sparsify_top_k: int | None = None,
        teacher_kwargs: dict[str, Any] | None = None,
    ):
        self.teacher_model = teacher_model.eval()
        self.vocab_size = vocab_size
        self.sparsify_top_k = sparsify_top_k
        self.teacher_kwargs = teacher_kwargs or {}

        for param in self.teacher_model.parameters():
            param.requires_grad_(False)

    @override
    def supports_hidden_states(self) -> bool:
        return True

    @override
    def get_signal(
        self, batch: dict[str, Any], return_hidden_states: bool = False
    ) -> TeacherSignal:
        with torch.no_grad():
            teacher_outputs = self.teacher_model(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask", None),
                output_hidden_states=return_hidden_states,
                **self.teacher_kwargs,
            )

        real_vocab_size = teacher_outputs.logits.shape[-1]
        vocab_size = min(real_vocab_size, self.vocab_size)

        if self.sparsify_top_k is not None:
            logprobs = torch.log_softmax(
                teacher_outputs.logits[..., :vocab_size], dim=-1
            )
            values, indices = torch.topk(logprobs, self.sparsify_top_k, dim=-1)
            return SparseSignal(
                sparse_ids=indices,
                sparse_values=values,
                log_values=True,
                generation_temperature=1.0,
                hidden_states=teacher_outputs.hidden_states,
                vocab_size=vocab_size,
            )

        return DenseSignal(
            logits=teacher_outputs.logits[..., :vocab_size],
            hidden_states=teacher_outputs.hidden_states,
            generation_temperature=1.0,
            vocab_size=vocab_size,
        )

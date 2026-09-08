"""Teacher-forced text capture: logits and anchor states from one forward.

Run ``python -m distillkit.sample_transformers --help``. The CLI defaults to
local model files and never instantiates a multimodal conditional-generation
class. Prepared JSONL records may supply input_ids, doc_id, and train/eval split.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable

import click
import numpy as np
import torch

from distillkit.gqa_dispatch import install_expanded_gqa_attention
from distillkit.linear_attention_dispatch import install_device_aware_linear_attention
from distillkit.offline_cache import (
    OfflineCacheWriter,
    file_sha256,
    tokenizer_vocab_hash,
)


LOG = logging.getLogger(__name__)

# flash-linear-attention, when installed, is bound by transformers at import time with
# no device check, and its Triton kernels reject CPU tensors. Restore per-call dispatch
# so CPU capture/verification runs keep working. Idempotent; no-op without fla.
install_device_aware_linear_attention()
# Qwen3.5's 16:4 query/KV head ratio reaches SDPA as enable_gqa=True, which this
# build has no fused kernel for -- it silently picks the math kernel and 4328 MiB
# per attention call at sequence 4096 instead of 249.
install_expanded_gqa_attention()



def text_causal_lm_class(config):
    """Select explicit Qwen text classes before allocating any model weights.

    Transformers' text checkpoint conversion maps model.language_model.* to
    model.* and retains lm_head. Using AutoModel on the parent VLM config would
    allocate the vision encoder and is deliberately unsupported here.
    """
    from transformers import Qwen3_5ForCausalLM, Qwen3_5MoeForCausalLM, Qwen4ExpForCausalLM

    text_config = config.get_text_config() if hasattr(config, "get_text_config") else config
    classes = {"qwen3_5_text": Qwen3_5ForCausalLM, "qwen3_5_moe_text": Qwen3_5MoeForCausalLM,
               "qwen4_exp_text": Qwen4ExpForCausalLM}
    if text_config.model_type not in classes:
        raise ValueError(f"Unsupported text decoder model_type {text_config.model_type!r}")
    return classes[text_config.model_type], text_config


def load_text_teacher(
    model_path: str,
    *,
    revision: str | None = None,
    int8: bool = True,
    device_map: str | dict = "auto",
    max_memory: dict | None = None,
    local_files_only: bool = True,
):
    from transformers import AutoConfig, BitsAndBytesConfig

    config = AutoConfig.from_pretrained(model_path, revision=revision, local_files_only=local_files_only)
    cls, text_config = text_causal_lm_class(config)
    kwargs = dict(config=text_config, revision=revision, local_files_only=local_files_only,
                  device_map=device_map, dtype=torch.bfloat16, output_loading_info=True)
    if max_memory is not None:
        kwargs["max_memory"] = max_memory
    if int8:
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    model, info = cls.from_pretrained(model_path, **kwargs)
    if info.get("missing_keys") or info.get("mismatched_keys") or info.get("error_msgs"):
        raise RuntimeError(f"Teacher text checkpoint did not load exactly: {info}")
    if any("visual" in name or "vision_tower" in name for name, _ in model.named_modules()):
        raise RuntimeError("Teacher loader unexpectedly instantiated vision modules")
    return model.eval()


def _input_device(model) -> torch.device:
    embedding = model.get_input_embeddings()
    hook = getattr(embedding, "_hf_hook", None)
    if getattr(hook, "execution_device", None) is not None:
        return torch.device(hook.execution_device)
    device = embedding.weight.device
    if device.type == "meta":
        raise ValueError("Teacher embeddings are meta tensors without an execution device")
    return device


class _AnchorTap:
    """Capture only the requested hidden states, via hooks.

    ``output_hidden_states=True`` materializes all ``num_layers + 1`` states and keeps
    them for the whole forward. On the 27B that is 65 x tokens x 5120 x 2 bytes -- 2.7 GB
    at 4096 tokens -- to extract two anchors worth 84 MB. With the teacher quantized to
    int8 there is only ~2 GiB free per card, so that difference decides whether a
    document fits at all.

    Index semantics are preserved exactly: ``hidden_states[i]`` is the *input* to decoder
    layer ``i``, so an anchor below ``num_layers`` is a forward-pre-hook on that layer.
    The final entry, ``hidden_states[num_layers]``, is the last layer's output *after*
    ``model.norm`` -- not the raw layer output -- so that anchor hooks the norm instead.
    Hooking the last decoder layer there silently yields the pre-norm state, which is a
    different tensor and would make the deepest anchor a wrong target.
    """

    def __init__(self, model, anchor_layers):
        text_model = model.model.language_model if hasattr(model.model, "language_model") else model.model
        self.layers = text_model.layers
        self.final_norm = text_model.norm
        self.n_layers = len(self.layers)
        self.anchors = list(anchor_layers)
        self.captured: dict[int, torch.Tensor] = {}
        self._handles = []

    def __enter__(self):
        for anchor in self.anchors:
            if anchor < self.n_layers:
                def pre_hook(_module, args, kwargs, _a=anchor):
                    tensor = args[0] if args else kwargs.get("hidden_states")
                    self.captured[_a] = tensor.detach()
                    return None
                self._handles.append(
                    self.layers[anchor].register_forward_pre_hook(pre_hook, with_kwargs=True)
                )
            elif anchor == self.n_layers:
                def post_hook(_module, _args, output, _a=anchor):
                    tensor = output[0] if isinstance(output, tuple) else output
                    self.captured[_a] = tensor.detach()
                    return None
                self._handles.append(self.final_norm.register_forward_hook(post_hook))
            else:
                raise ValueError(
                    f"Anchor {anchor} exceeds the teacher's hidden_states tuple "
                    f"(0..{self.n_layers})"
                )
        return self

    def __exit__(self, *exc):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self.captured.clear()
        return False


def capture_teacher(
    model,
    documents: Iterable[dict[str, Any]],
    output: str | Path,
    *,
    tokenizer_hash: str,
    anchor_layers: list[int],
    sequence_length: int = 4096,
    top_k: int = 64,
    shard_tokens: int = 65536,
    eval_every: int = 20,
    tokenizer_vocab_fingerprint: str | None = None,
    metadata: dict[str, Any] | None = None,
    logit_chunk_tokens: int = 64,
):
    """Capture one unpadded prefix per document; existing split labels win.

    Anchor indices refer to the exact ``output.hidden_states`` tuple, including
    its index-0 embedding state. Log probabilities are normalized over the
    complete model head on its device, in bounded fp32 chunks. FP8 overflow
    fails capture instead of silently emitting NaNs or adding undocumented scales.
    """
    if type(eval_every) is not int or eval_every < 2:
        raise ValueError("eval_every must be at least 2")
    if type(logit_chunk_tokens) is not int or logit_chunk_tokens < 1:
        raise ValueError("logit_chunk_tokens must be positive")
    config = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
    hidden_size, vocab_size = config.hidden_size, config.vocab_size
    if any(type(i) is not int or i < 0 or i > config.num_hidden_layers for i in anchor_layers):
        raise ValueError("Anchor indices must index the teacher's hidden_states tuple")
    provenance = {"teacher_model_type": config.model_type,
                  "teacher_config": config.to_dict(),
                  "capture": "single_teacher_forced_forward_per_document",
                  "eval_policy": {"explicit_split_takes_precedence": True, "fallback_eval_every": eval_every},
                  **(metadata or {})}
    device = _input_device(model)
    model.eval()
    with OfflineCacheWriter(
        output, tokenizer_hash=tokenizer_hash, tokenizer_vocab_fingerprint=tokenizer_vocab_fingerprint,
        anchor_layers=anchor_layers, hidden_size=hidden_size, vocab_size=vocab_size,
        sequence_length=sequence_length, top_k=top_k, shard_tokens=shard_tokens, metadata=provenance,
    ) as writer, torch.inference_mode():
        for ordinal, record in enumerate(documents):
            doc_id = record.get("doc_id", str(ordinal))
            if not isinstance(doc_id, str):
                doc_id = str(doc_id)
            raw_tokens = record["input_ids"]
            if not isinstance(raw_tokens, (list, tuple, np.ndarray)) or np.asarray(raw_tokens).ndim != 1:
                raise ValueError(f"Document {doc_id!r} requires a flat input_ids sequence")
            if not len(raw_tokens) or any(type(t) is not int and not isinstance(t, np.integer) for t in raw_tokens):
                raise ValueError(f"Document {doc_id!r} requires nonempty integer input_ids")
            tokens = np.asarray(raw_tokens[:sequence_length], dtype=np.int64)
            if np.any(tokens < 0) or np.any(tokens >= vocab_size):
                raise ValueError(f"Document {doc_id!r} contains IDs outside teacher vocabulary")
            if "attention_mask" in record and not np.all(np.asarray(record["attention_mask"]) == 1):
                raise ValueError("Capture inputs must be unpadded, unpacked documents")
            input_ids = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
            with _AnchorTap(model, anchor_layers) as tap:
                result = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                               use_cache=False, return_dict=True)
                anchor_states = dict(tap.captured)
            logits = result.logits
            if logits.shape != (1, len(tokens), vocab_size):
                raise ValueError("Teacher logits must cover every unshifted input position and the full vocabulary")
            if set(anchor_states) != set(anchor_layers):
                missing = sorted(set(anchor_layers) - set(anchor_states))
                raise ValueError(f"Teacher did not produce anchor states {missing}")
            ids = np.empty((len(tokens), top_k), dtype="<u4")
            values = np.empty((len(tokens), top_k), dtype="<f2")
            for start in range(0, len(tokens), logit_chunk_tokens):
                stop = min(start + logit_chunk_tokens, len(tokens))
                chunk = logits[0, start:stop].float()
                if not torch.isfinite(chunk).all():
                    raise ValueError(f"Nonfinite teacher logits for document {doc_id!r}")
                best, indices = torch.topk(chunk, k=top_k, dim=-1)
                logprobs = best - torch.logsumexp(chunk, dim=-1, keepdim=True)
                ids[start:stop] = indices.cpu().numpy().astype("<u4")
                values[start:stop] = logprobs.to(torch.float16).cpu().numpy()
                del chunk, best, indices, logprobs
            states = np.empty((len(tokens), len(anchor_layers), hidden_size), dtype=np.uint8)
            for compact_index, layer_index in enumerate(anchor_layers):
                hidden = anchor_states[layer_index]
                if hidden.shape != (1, len(tokens), hidden_size):
                    raise ValueError(f"Invalid anchor {layer_index} shape: {tuple(hidden.shape)}")
                if not torch.isfinite(hidden).all() or hidden.abs().max() > torch.finfo(torch.float8_e4m3fn).max:
                    raise ValueError(f"Anchor {layer_index} overflows unscaled float8_e4m3fn")
                states[:, compact_index, :] = hidden[0].to(torch.float8_e4m3fn).view(torch.uint8).cpu().numpy()
            writer.append(doc_id, tokens, ids, values, states,
                          split=record.get("split", "eval" if ordinal % eval_every == 0 else "train"),
                          original_length=len(raw_tokens))
            del result, logits, states, hidden, anchor_states
            if ordinal % 100 == 0:
                LOG.info("Captured document %d (%s)", ordinal + 1, doc_id)
    return Path(output) / "manifest.json"


def iter_jsonl(path: str | Path, tokenizer=None, *, add_special_tokens: bool = True):
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if "input_ids" not in record:
                if tokenizer is None or not isinstance(record.get("text"), str):
                    raise ValueError(f"JSONL line {line_number} requires input_ids or text and a tokenizer")
                record["input_ids"] = tokenizer(record["text"], add_special_tokens=add_special_tokens)["input_ids"]
            yield record


@click.command("sample-transformers")
@click.option("--model", required=True, help="Local checkpoint path or explicitly allowed Hugging Face repo.")
@click.option("--revision", default=None)
@click.option("--input-jsonl", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--source-metadata", type=click.Path(exists=True, dir_okay=False), default=None,
              help="JSON provenance emitted when preparing the capture input.")
@click.option("--output", type=click.Path(file_okay=False), required=True)
@click.option("--tokenizer", default=None)
@click.option("--tokenizer-json", type=click.Path(exists=True, dir_okay=False), required=True,
              help="Exact tokenizer.json used to prepare input tokens; SHA256 goes in manifest.")
@click.option("--anchor", "anchors", type=int, multiple=True, required=True)
@click.option("--sequence-length", type=click.IntRange(min=1), default=4096, show_default=True)
@click.option("--top-k", type=click.IntRange(min=1), default=64, show_default=True)
@click.option("--shard-tokens", type=click.IntRange(min=1), default=65536, show_default=True)
@click.option("--eval-every", type=click.IntRange(min=2), default=20, show_default=True)
@click.option("--int8/--no-int8", default=True, show_default=True)
@click.option("--device-map", default="auto", show_default=True, help="Accelerate device-map strategy or JSON mapping.")
@click.option("--max-memory", default=None, help='JSON memory budgets, e.g. {"0":"20GiB","1":"20GiB","cpu":"32GiB"}.')
@click.option("--local-files-only/--allow-download", default=True, show_default=True)
@click.option("--add-special-tokens/--no-add-special-tokens", default=True)
def main(model, revision, input_jsonl, source_metadata, output, tokenizer, tokenizer_json,
         anchors, sequence_length, top_k, shard_tokens, eval_every, int8, device_map,
         max_memory, local_files_only, add_special_tokens):
    """Capture aligned log probabilities and FP8 teacher anchors in one pass."""
    from transformers import AutoTokenizer

    logging.basicConfig(level=logging.INFO)
    if shard_tokens < sequence_length:
        raise click.UsageError("--shard-tokens must be at least --sequence-length")
    if Path(output).exists() and any(Path(output).iterdir()):
        raise click.UsageError("--output must be a new or empty directory")
    tok = AutoTokenizer.from_pretrained(tokenizer or model, revision=revision, local_files_only=local_files_only)
    if device_map.startswith("{"):
        device_map = json.loads(device_map)
    if max_memory:
        max_memory = {int(k) if k.isdigit() else k: v for k, v in json.loads(max_memory).items()}
    metadata = {"model": model, "revision": revision, "int8": int8,
                "input_jsonl_sha256": file_sha256(input_jsonl), "add_special_tokens": add_special_tokens}
    if source_metadata:
        metadata["source"] = json.loads(Path(source_metadata).read_text(encoding="utf-8"))
    teacher = load_text_teacher(model, revision=revision, int8=int8, device_map=device_map,
                                max_memory=max_memory, local_files_only=local_files_only)
    manifest = capture_teacher(
        teacher, iter_jsonl(input_jsonl, tok, add_special_tokens=add_special_tokens), output,
        tokenizer_hash=file_sha256(tokenizer_json), tokenizer_vocab_fingerprint=tokenizer_vocab_hash(tok),
        anchor_layers=list(anchors), sequence_length=sequence_length, top_k=top_k,
        shard_tokens=shard_tokens, eval_every=eval_every, metadata=metadata,
    )
    click.echo(str(manifest))


if __name__ == "__main__":
    main()

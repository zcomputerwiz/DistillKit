"""Teacher-forced text capture: logits and anchor states from one forward.

Run ``python -m distillkit.sample_transformers --help``. The CLI defaults to
local model files and never instantiates a multimodal conditional-generation
class. Prepared JSONL records may supply input_ids, doc_id, and train/eval split.
"""

from __future__ import annotations

import contextlib
import json
import logging
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable

import click
import numpy as np
import torch

from distillkit.models.qwen35 import (
    install_device_aware_linear_attention,
    install_expanded_gqa_attention,
)
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


def _skip_packed_check():
    """Capture documents are never packed (one unpadded document a forward), so skip
    transformers' packed-sequence test: it reads a GPU flag on the host, a wait in every
    FlashAttention call."""
    from transformers import modeling_flash_attention_utils

    modeling_flash_attention_utils._is_packed_sequence = lambda position_ids, batch_size: False


def _serialize_autotune():
    """Triton's autotuner keeps the arguments being tuned on the instance (``nargs``), so
    two capture workers tuning one kernel for a new shape clobber each other ("'NoneType'
    object is not a mapping"). One lock around it; a tuned launch holds it only briefly."""
    from triton.runtime import autotuner

    if getattr(autotuner.Autotuner.run, "serialized", False):
        return
    lock, run = threading.RLock(), autotuner.Autotuner.run

    def serialized(self, *args, **kwargs):
        with lock:
            return run(self, *args, **kwargs)

    serialized.serialized = True
    autotuner.Autotuner.run = serialized



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
    attn_implementation: str | None = None,
    weight_only_int8: bool = False,
):
    """`int8` is bitsandbytes' LLM.int8; `weight_only_int8` keeps activations in bf16 so a
    position's targets do not depend on the rest of the document (`weight_only_int8.py`)."""
    from transformers import AutoConfig, BitsAndBytesConfig

    if int8 and weight_only_int8:
        raise ValueError("choose one of int8 (LLM.int8) and weight_only_int8")
    config = AutoConfig.from_pretrained(model_path, revision=revision, local_files_only=local_files_only)
    cls, text_config = text_causal_lm_class(config)
    kwargs = dict(config=text_config, revision=revision, local_files_only=local_files_only,
                  device_map={"": "cpu"} if weight_only_int8 else device_map,
                  dtype=torch.bfloat16, output_loading_info=True)
    if max_memory is not None and not weight_only_int8:
        kwargs["max_memory"] = max_memory
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    if int8:
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    model, info = cls.from_pretrained(model_path, **kwargs)
    if info.get("missing_keys") or info.get("mismatched_keys") or info.get("error_msgs"):
        raise RuntimeError(f"Teacher text checkpoint did not load exactly: {info}")
    if any("visual" in name or "vision_tower" in name for name, _ in model.named_modules()):
        raise RuntimeError("Teacher loader unexpectedly instantiated vision modules")
    if weight_only_int8:
        # Quantized on the CPU, then spread over the cards whole decoder layers at a time.
        from accelerate import dispatch_model, infer_auto_device_map
        from accelerate.utils import get_balanced_memory

        from .weight_only_int8 import quantize_linears

        LOG.info("weight-only int8: %d linear layers quantized", quantize_linears(model))
        # Balanced, so each card keeps the same room for a long document's activations.
        budget = max_memory or get_balanced_memory(
            model, max_memory={i: "20GiB" for i in range(torch.cuda.device_count())},
            no_split_module_classes=list(model._no_split_modules))
        placement = infer_auto_device_map(model, max_memory=budget,
                                          no_split_module_classes=list(model._no_split_modules))
        if any(str(device) in ("cpu", "disk") for device in placement.values()):
            raise RuntimeError(f"weight-only int8 teacher does not fit the cards: {placement}")
        model = dispatch_model(model, device_map=placement)
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
    logit_chunk_tokens: int = 512,
    prefill_chunk: int | None = None,
    overlap: int = 1,
    batch_tokens: int | None = None,
):
    """Capture one unpadded prefix per document; existing split labels win.

    ``prefill_chunk`` feeds a longer document in segments with the cache carried
    between them; logits-only captures (no anchors) only.

    Anchor indices refer to the exact ``output.hidden_states`` tuple, including
    its index-0 embedding state. Log probabilities are normalized over the
    complete model head on its device, in bounded fp32 chunks. FP8 overflow
    fails capture instead of silently emitting NaNs or adding undocumented scales.
    """
    if type(eval_every) is not int or eval_every < 2:
        raise ValueError("eval_every must be at least 2")
    if type(logit_chunk_tokens) is not int or logit_chunk_tokens < 1:
        raise ValueError("logit_chunk_tokens must be positive")
    if type(overlap) is not int or overlap < 1:
        raise ValueError("overlap must be positive")
    if batch_tokens and anchor_layers:
        raise ValueError("batch_tokens captures logits only")
    if prefill_chunk and anchor_layers:
        raise ValueError("prefill_chunk captures logits only; anchors need one whole forward")
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
    _skip_packed_check()
    def project(hidden, store, offset):
        # The head a chunk at a time: the full [seq, vocab] logits are 32 GB in fp32 at
        # 32K positions. Results stay on the head's card until fetched: one host wait a
        # forward instead of three a chunk.
        for start in range(0, hidden.shape[1], logit_chunk_tokens):
            stop = min(start + logit_chunk_tokens, hidden.shape[1])
            chunk = model.lm_head(hidden[:, start:stop])[0].float()
            if chunk.shape[-1] != vocab_size:
                raise ValueError("Teacher logits must cover the full vocabulary")
            if "ids" not in store:
                store["ids"] = torch.empty((store["length"], top_k), dtype=torch.int32, device=chunk.device)
                store["values"] = torch.empty((store["length"], top_k), dtype=torch.float16, device=chunk.device)
                store["finite"] = []
            store["finite"].append(torch.isfinite(chunk).all())
            best, indices = torch.topk(chunk, k=top_k, dim=-1)
            store["values"][offset + start:offset + stop] = best - torch.logsumexp(chunk, dim=-1, keepdim=True)
            store["ids"][offset + start:offset + stop] = indices

    def fetch(store, what):
        # Copy into pinned memory and poll an event, sleeping: a blocking copy spins in
        # the CUDA driver while holding the GIL, which stalls the other worker's launches.
        device = store["ids"].device
        if device.type != "cuda":
            if not torch.stack(store["finite"]).all():
                raise ValueError(f"Nonfinite teacher logits for {what}")
            return store["ids"].numpy().view("<u4"), store["values"].numpy()
        host = {name: torch.empty(store[name].shape, dtype=store[name].dtype, pin_memory=True)
                for name in ("ids", "values")}
        finite = torch.empty((), dtype=torch.bool, pin_memory=True)
        finite.copy_(torch.stack(store["finite"]).all(), non_blocking=True)
        for name, tensor in host.items():
            tensor.copy_(store[name], non_blocking=True)
        done = torch.cuda.Event()
        done.record(torch.cuda.current_stream(device))
        while not done.query():
            time.sleep(0.0002)
        if not finite:
            raise ValueError(f"Nonfinite teacher logits for {what}")
        return host["ids"].numpy().view("<u4"), host["values"].numpy()

    def prepare(ordinal, record):
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
        split = record.get("split", "eval" if ordinal % eval_every == 0 else "train")
        return doc_id, tokens, split, len(raw_tokens)

    def run(ordinal, record):
        doc_id, tokens, split, original_length = prepare(ordinal, record)
        input_ids = torch.from_numpy(tokens).pin_memory().to(device, non_blocking=True).unsqueeze(0)
        store = {"length": len(tokens)}
        if prefill_chunk and len(tokens) > prefill_chunk:
            # Segments with the cache carried between them -- attention KV and the
            # DeltaNet states -- so activations are a segment's, not the document's.
            # Measured against one forward on the 2B source at 8K in 2K segments:
            # KL 1.29e-3 nats, against 1.08e-3 between FlashAttention and SDPA.
            from transformers import DynamicCache

            cache = DynamicCache(config=config)
            for start in range(0, len(tokens), prefill_chunk):
                out = model.model(input_ids=input_ids[:, start:start + prefill_chunk],
                                  past_key_values=cache, use_cache=True, return_dict=True)
                cache = out.past_key_values
                project(out.last_hidden_state, store, start)
                del out
            del cache
            hidden, anchor_states = None, {}
        else:
            with _AnchorTap(model, anchor_layers) as tap:
                hidden = model.model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                                     use_cache=False, return_dict=True).last_hidden_state
                anchor_states = dict(tap.captured)
            if hidden.shape[:2] != (1, len(tokens)):
                raise ValueError("Teacher hidden states must cover every input position")
            project(hidden, store, 0)
        ids, values = fetch(store, f"document {doc_id!r}")
        if set(anchor_states) != set(anchor_layers):
            missing = sorted(set(anchor_layers) - set(anchor_states))
            raise ValueError(f"Teacher did not produce anchor states {missing}")
        # Logits-only captures (no anchors) store no hidden states; see `_layouts`.
        states = (np.empty((len(tokens), len(anchor_layers), hidden_size), dtype=np.uint8)
                  if anchor_layers else None)
        for compact_index, layer_index in enumerate(anchor_layers):
            hidden = anchor_states[layer_index]
            if hidden.shape != (1, len(tokens), hidden_size):
                raise ValueError(f"Invalid anchor {layer_index} shape: {tuple(hidden.shape)}")
            if not torch.isfinite(hidden).all() or hidden.abs().max() > torch.finfo(torch.float8_e4m3fn).max:
                raise ValueError(f"Anchor {layer_index} overflows unscaled float8_e4m3fn")
            states[:, compact_index, :] = hidden[0].to(torch.float8_e4m3fn).view(torch.uint8).cpu().numpy()
        return ordinal, doc_id, tokens, ids, values, states, split, original_length

    def run_batch(job):
        # Short documents right-padded into one forward: causal attention, the short
        # convolutions and the DeltaNet recurrence never look ahead, so the padding after
        # a document leaves its positions as they are, and the per-layer launch cost is
        # paid once for the batch.
        docs = [(ordinal, *prepare(ordinal, record)) for ordinal, record in job]
        lengths = [len(tokens) for _, _, tokens, _, _ in docs]
        padded = np.zeros((len(docs), max(lengths)), dtype=np.int64)
        for row, (_, _, tokens, _, _) in enumerate(docs):
            padded[row, :len(tokens)] = tokens
        input_ids = torch.from_numpy(padded).pin_memory().to(device, non_blocking=True)
        hidden = model.model(input_ids=input_ids, use_cache=False, return_dict=True).last_hidden_state
        hidden = torch.cat([hidden[row, :n] for row, n in enumerate(lengths)])[None]
        store = {"length": sum(lengths)}
        project(hidden, store, 0)
        ids, values = fetch(store, "documents %s" % [doc_id for _, doc_id, *_ in docs])
        bounds = np.cumsum([0] + lengths)
        return [(ordinal, doc_id, tokens, ids[a:b], values[a:b], None, split, original_length)
                for (ordinal, doc_id, tokens, split, original_length), a, b in zip(docs, bounds, bounds[1:])]

    def run_job(job):
        return [run(*job[0])] if len(job) == 1 else run_batch(job)

    def jobs():
        # Consecutive documents share a forward while the padded batch stays within
        # batch_tokens; longer ones go alone. Input order is kept.
        job, longest = [], 0
        for ordinal, record in enumerate(documents):
            length = min(len(record["input_ids"]), sequence_length)
            if job and (len(job) + 1) * max(longest, length) > (batch_tokens or 0):
                yield job
                job, longest = [], 0
            job.append((ordinal, record))
            longest = max(longest, length)
        if job:
            yield job

    def write(result):
        ordinal, doc_id, tokens, ids, values, states, split, original_length = result
        writer.append(doc_id, tokens, ids, values, states, split=split, original_length=original_length)
        if ordinal % 100 == 0:
            LOG.info("Captured document %d (%s)", ordinal + 1, doc_id)

    streams = threading.local()

    def overlapped(job):
        # Each worker on its own CUDA streams, so one document's half of the layers on
        # one card runs while another document's half runs on the other card.
        if not hasattr(streams, "all"):
            streams.all = [torch.cuda.Stream(device=i) for i in range(torch.cuda.device_count())]
        with contextlib.ExitStack() as stack:
            stack.enter_context(torch.inference_mode())  # thread-local, like the streams
            for stream in streams.all:
                stack.enter_context(torch.cuda.stream(stream))
            return run_job(job)

    def pace(_module, _args, output):
        # A kernel launch that finds its card's queue full blocks while holding the GIL,
        # which starves the other worker's card. So a worker runs at most two layers ahead
        # of its card, and waits by polling an event while sleeping (GIL released).
        if not hasattr(streams, "all"):
            return
        hidden = output[0] if isinstance(output, tuple) else output
        done = torch.cuda.Event()
        done.record(torch.cuda.current_stream(hidden.device))
        ahead = streams.__dict__.setdefault("ahead", deque())
        ahead.append(done)
        while len(ahead) > 2:
            while not ahead[0].query():
                time.sleep(0.0005)
            ahead.popleft()

    with OfflineCacheWriter(
        output, tokenizer_hash=tokenizer_hash, tokenizer_vocab_fingerprint=tokenizer_vocab_fingerprint,
        anchor_layers=anchor_layers, hidden_size=hidden_size, vocab_size=vocab_size,
        sequence_length=sequence_length, top_k=top_k, shard_tokens=shard_tokens, metadata=provenance,
    ) as writer, torch.inference_mode():
        if overlap == 1:
            for job in jobs():
                for result in run_job(job):
                    write(result)
        else:
            # The teacher is split across the cards by layer, so one document at a time
            # leaves each card idle while the other works; written in input order.
            _serialize_autotune()
            hooks = [layer.register_forward_hook(pace) for layer in model.model.layers]
            pending = deque()
            try:
                with ThreadPoolExecutor(overlap) as pool:
                    for job in jobs():
                        pending.append(pool.submit(overlapped, job))
                        if len(pending) > overlap:
                            for result in pending.popleft().result():
                                write(result)
                    while pending:
                        for result in pending.popleft().result():
                            write(result)
            finally:
                for hook in hooks:
                    hook.remove()
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
@click.option("--anchor", "anchors", type=int, multiple=True, required=False, help="teacher hidden_states indices to store as fp8 anchors; omit for a logits-only capture, which stores only the top-k distribution distillation reads, at about 4%% of the size")
@click.option("--sequence-length", type=click.IntRange(min=1), default=4096, show_default=True)
@click.option("--top-k", type=click.IntRange(min=1), default=64, show_default=True)
@click.option("--shard-tokens", type=click.IntRange(min=1), default=65536, show_default=True)
@click.option("--eval-every", type=click.IntRange(min=2), default=20, show_default=True)
@click.option("--int8/--no-int8", default=True, show_default=True)
@click.option("--weight-only-int8", is_flag=True, default=False,
              help="int8 weights, bf16 activations, instead of LLM.int8 (pass --no-int8 with it)")
@click.option("--device-map", default="auto", show_default=True, help="Accelerate device-map strategy or JSON mapping.")
@click.option("--max-memory", default=None, help='JSON memory budgets, e.g. {"0":"20GiB","1":"20GiB","cpu":"32GiB"}.')
@click.option("--attn-implementation", default=None,
              help="e.g. flash_attention_2 for long captures; default is transformers' choice (SDPA)")
@click.option("--prefill-chunk", type=click.IntRange(min=1), default=None,
              help="feed longer documents in segments of this many tokens, cache carried (logits-only)")
@click.option("--logit-chunk-tokens", type=click.IntRange(min=1), default=512, show_default=True,
              help="positions projected to the vocabulary at once (fp32 logits: 1 GB per 1024)")
@click.option("--overlap", type=click.IntRange(min=1), default=2, show_default=True,
              help="documents in flight at once, so the cards of a layer-split teacher overlap")
@click.option("--batch-tokens", type=click.IntRange(min=0), default=8192, show_default=True,
              help="pad short consecutive documents into one forward up to this many positions (0: one at a time)")
@click.option("--local-files-only/--allow-download", default=True, show_default=True)
@click.option("--add-special-tokens/--no-add-special-tokens", default=True)
def main(model, revision, input_jsonl, source_metadata, output, tokenizer, tokenizer_json,
         anchors, sequence_length, top_k, shard_tokens, eval_every, int8, weight_only_int8, device_map,
         max_memory, attn_implementation, prefill_chunk, logit_chunk_tokens, overlap, batch_tokens, local_files_only, add_special_tokens):
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
    metadata = {"model": model, "revision": revision, "int8": int8, "attn_implementation": attn_implementation,
                "prefill_chunk": prefill_chunk, "logit_chunk_tokens": logit_chunk_tokens, "weight_only_int8": weight_only_int8,
                "input_jsonl_sha256": file_sha256(input_jsonl), "add_special_tokens": add_special_tokens}
    if source_metadata:
        metadata["source"] = json.loads(Path(source_metadata).read_text(encoding="utf-8"))
    teacher = load_text_teacher(model, revision=revision, int8=int8, device_map=device_map,
                                max_memory=max_memory, local_files_only=local_files_only,
                                attn_implementation=attn_implementation,
                                weight_only_int8=weight_only_int8)
    def capture(overlap, batch_tokens):
        return capture_teacher(
            teacher, iter_jsonl(input_jsonl, tok, add_special_tokens=add_special_tokens), output,
            tokenizer_hash=file_sha256(tokenizer_json), tokenizer_vocab_fingerprint=tokenizer_vocab_hash(tok),
            anchor_layers=list(anchors), sequence_length=sequence_length, top_k=top_k,
            shard_tokens=shard_tokens, eval_every=eval_every, metadata=metadata,
            prefill_chunk=prefill_chunk, logit_chunk_tokens=logit_chunk_tokens, overlap=overlap,
            batch_tokens=(batch_tokens or None) if not anchors else None,
        )

    manifest = None
    try:
        manifest = capture(overlap, batch_tokens)
    except Exception:
        if overlap == 1 and not batch_tokens:
            raise
        # Overlap and batching hold more activations; rather than lose a queued capture,
        # start it again one document at a time.
        LOG.exception("Overlapped or batched capture failed; retrying one document at a time")
    if manifest is None:
        if Path(output).exists():
            Path(output).rename(f"{output}.overlap-failed")
        torch.cuda.empty_cache()
        manifest = capture(1, 0)
    click.echo(str(manifest))


if __name__ == "__main__":
    main()

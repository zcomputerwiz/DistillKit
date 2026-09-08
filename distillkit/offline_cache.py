"""Versioned, bounded memmap storage for aligned offline teacher signals.

Every document stores its *input* tokens and signals from the same model
positions. Position t contains the teacher prediction after seeing token t;
neither logits nor hidden states are shifted. A completed manifest is published
only after all streams have been flushed. Interrupted captures cannot be read.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


FORMAT_VERSION = 1
CACHE_DTYPE = "float8_e4m3fn"
POSITION_ALIGNMENT = "unshifted_model_positions"


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tokenizer_vocab_hash(tokenizer: Any) -> str:
    """Hash actual token-to-ID assignments, independent of tokenizer metadata."""
    encoded = json.dumps(
        sorted(tokenizer.get_vocab().items()), ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _array_hash(array: np.ndarray) -> str:
    return hashlib.sha256(memoryview(np.ascontiguousarray(array)).cast("B")).hexdigest()


def _positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a SHA256 hex digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{name} must be a SHA256 hex digest") from exc
    return value


def _layouts(top_k: int, anchors: int, hidden_size: int):
    return {
        "input_ids": (np.dtype("<u4"), ()),
        "topk_ids": (np.dtype("<u4"), (top_k,)),
        "topk_logprobs": (np.dtype("<f2"), (top_k,)),
        "hidden_states": (np.dtype("u1"), (anchors, hidden_size)),
    }


class OfflineCacheWriter:
    """Write documents without holding more than one bounded shard per split.

    ``hidden_states`` passed to append are raw e4m3fn bytes in [tokens,
    anchors, width] order, not numerically converted uint8 values.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        tokenizer_hash: str,
        anchor_layers: list[int],
        hidden_size: int,
        vocab_size: int,
        sequence_length: int,
        top_k: int = 64,
        shard_tokens: int = 65536,
        tokenizer_vocab_fingerprint: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        if any(self.path.iterdir()):
            raise ValueError("Capture output must be an empty directory")
        _digest(tokenizer_hash, "tokenizer_hash")
        if tokenizer_vocab_fingerprint is not None:
            _digest(tokenizer_vocab_fingerprint, "tokenizer_vocab_hash")
        if not anchor_layers or len(set(anchor_layers)) != len(anchor_layers):
            raise ValueError("anchor_layers must be nonempty and unique")
        if any(type(i) is not int or i < 0 for i in anchor_layers):
            raise ValueError("anchor_layers must contain nonnegative hidden-state indices")
        for key, val in dict(hidden_size=hidden_size, vocab_size=vocab_size,
                             sequence_length=sequence_length, top_k=top_k,
                             shard_tokens=shard_tokens).items():
            _positive_int(val, key)
        if top_k > vocab_size or vocab_size > 2**32:
            raise ValueError("Invalid top_k or uint32 vocabulary size")
        if shard_tokens < sequence_length:
            raise ValueError("shard_tokens must be at least sequence_length")
        self.layouts = _layouts(top_k, len(anchor_layers), hidden_size)
        self.manifest = {
            "format_version": FORMAT_VERSION,
            "complete": True,
            "tokenizer_hash": tokenizer_hash,
            "tokenizer_vocab_hash": tokenizer_vocab_fingerprint,
            "anchor_layers": list(anchor_layers),
            "hidden_size": hidden_size,
            "hidden_dtype": CACHE_DTYPE,
            "hidden_layout": "tokens,anchors,hidden_size",
            "token_dtype": "uint32",
            "topk_index_dtype": "uint32",
            "topk_value_dtype": "float16",
            "top_k": top_k,
            "vocab_size": vocab_size,
            "log_values": True,
            "generation_temperature": 1.0,
            "normalization": "full_model_vocabulary",
            "position_alignment": POSITION_ALIGNMENT,
            "sequence_length": sequence_length,
            "truncation": "right_prefix_no_chunking",
            "shard_tokens": shard_tokens,
            "document_order": [],
            "documents": [],
            "shards": [],
            "metadata": metadata or {},
        }
        self._active: dict[str, dict[str, Any]] = {}
        self._doc_ids: set[str] = set()
        self._closed = False
        (self.path / ".incomplete").write_text("Capture has not completed.\n", encoding="utf-8")

    def _new_shard(self, split: str) -> dict[str, Any]:
        shard_id = len(self.manifest["shards"])
        entry = {"id": shard_id, "split": split, "tokens": 0, "doc_ids": [], "files": {}}
        arrays = {}
        for name, (dtype, shape) in self.layouts.items():
            filename = f"{split}-{shard_id:05d}.{name}.bin"
            entry["files"][name] = filename
            arrays[name] = np.memmap(
                self.path / filename, mode="w+", dtype=dtype,
                shape=(self.manifest["shard_tokens"], *shape),
            )
        self.manifest["shards"].append(entry)
        active = {"entry": entry, "arrays": arrays}
        self._active[split] = active
        return active

    def _finish_shard(self, split: str):
        active = self._active.pop(split)
        entry = active["entry"]
        for name, array in active["arrays"].items():
            array.flush()
            array._mmap.close()
            dtype, shape = self.layouts[name]
            size = entry["tokens"] * dtype.itemsize * int(np.prod(shape, dtype=np.int64))
            with (self.path / entry["files"][name]).open("r+b") as handle:
                handle.truncate(size)
        entry["doc_id_range"] = [entry["doc_ids"][0], entry["doc_ids"][-1]]

    def append(
        self,
        doc_id: str,
        input_ids: np.ndarray,
        topk_ids: np.ndarray,
        topk_logprobs: np.ndarray,
        hidden_states: np.ndarray,
        *,
        split: str = "train",
        original_length: int | None = None,
    ):
        if self._closed:
            raise RuntimeError("Cache writer is closed")
        if not isinstance(doc_id, str) or not doc_id or doc_id in self._doc_ids:
            raise ValueError("Document IDs must be unique nonempty strings")
        if split not in {"train", "eval"}:
            raise ValueError("split must be train or eval")
        tokens = np.asarray(input_ids)
        if tokens.ndim != 1 or not 0 < len(tokens) <= self.manifest["sequence_length"]:
            raise ValueError("Document length is outside the capture sequence_length")
        original_length = len(tokens) if original_length is None else original_length
        if type(original_length) is not int or original_length < len(tokens):
            raise ValueError("original_length cannot be smaller than stored token length")
        data = {"input_ids": tokens, "topk_ids": np.asarray(topk_ids),
                "topk_logprobs": np.asarray(topk_logprobs), "hidden_states": np.asarray(hidden_states)}
        for name, (dtype, shape) in self.layouts.items():
            arr = data[name]
            if arr.shape != (len(tokens), *shape):
                raise ValueError(f"Invalid {name} shape: {arr.shape}")
            if name in {"input_ids", "topk_ids"}:
                if arr.dtype.kind not in "iu" or np.any(arr < 0) or np.any(arr >= self.manifest["vocab_size"]):
                    raise ValueError(f"{name} contains invalid vocabulary IDs")
            elif name == "topk_logprobs":
                if not np.all(np.isfinite(arr)) or np.any(arr > 0):
                    raise ValueError("topk_logprobs must be finite normalized log probabilities")
            elif arr.dtype != np.uint8:
                raise ValueError("hidden_states must contain raw float8 uint8 bytes")
            data[name] = np.ascontiguousarray(arr, dtype=dtype)
        active = self._active.get(split)
        if active and active["entry"]["tokens"] + len(tokens) > self.manifest["shard_tokens"]:
            self._finish_shard(split)
            active = None
        active = active or self._new_shard(split)
        entry = active["entry"]
        start = entry["tokens"]
        for name, array in data.items():
            active["arrays"][name][start:start + len(tokens)] = array
        document = {"doc_id": doc_id, "split": split, "shard": entry["id"],
                    "offset": start, "length": len(tokens), "original_length": original_length,
                    "sha256": {name: _array_hash(arr) for name, arr in data.items()}}
        entry["tokens"] += len(tokens)
        entry["doc_ids"].append(doc_id)
        self.manifest["documents"].append(document)
        self.manifest["document_order"].append(doc_id)
        self._doc_ids.add(doc_id)

    def close(self):
        if self._closed:
            return
        if not self._doc_ids:
            self.abort()
            raise ValueError("Cannot finalize an empty capture")
        for split in list(self._active):
            self._finish_shard(split)
        temporary = self.path / "manifest.json.tmp"
        temporary.write_text(json.dumps(self.manifest, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.path / "manifest.json")
        (self.path / ".incomplete").unlink()
        self._closed = True

    def abort(self):
        for split in list(self._active):
            self._finish_shard(split)
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        if exc_type is None:
            self.close()
        else:
            self.abort()


class OfflineTeacherCache:
    """Read-only, per-document checksum validation with a bounded mmap LRU."""

    def __init__(
        self, path: str | Path, *, expected_tokenizer_hash: str | None = None,
        expected_tokenizer_vocab_hash: str | None = None,
        expected_anchor_layers: Iterable[int] | None = None,
        expected_cache_dtype: str = CACHE_DTYPE,
        expected_sequence_length: int | None = None,
        max_open_shards: int = 4,
    ):
        self.path = Path(path).resolve()
        if (self.path / ".incomplete").exists():
            raise ValueError("Offline cache is incomplete; recapture to a new directory")
        try:
            self.manifest = json.loads((self.path / "manifest.json").read_text(encoding="utf-8"))
            self._validate_manifest()
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid offline cache manifest: {exc}") from exc
        expected = {"tokenizer_hash": expected_tokenizer_hash,
                    "tokenizer_vocab_hash": expected_tokenizer_vocab_hash,
                    "anchor_layers": list(expected_anchor_layers) if expected_anchor_layers is not None else None,
                    "hidden_dtype": expected_cache_dtype,
                    "sequence_length": expected_sequence_length}
        for name, value in expected.items():
            if value is not None and self.manifest.get(name) != value:
                raise ValueError(f"Offline cache {name} mismatch: expected {value!r}, found {self.manifest.get(name)!r}")
        self.max_open_shards = _positive_int(max_open_shards, "max_open_shards")
        self._maps: OrderedDict[int, dict[str, np.memmap]] = OrderedDict()
        self._read_lock = threading.Lock()

    def _validate_manifest(self):
        m = self.manifest
        fixed = {"format_version": FORMAT_VERSION, "complete": True,
                 "hidden_dtype": CACHE_DTYPE, "hidden_layout": "tokens,anchors,hidden_size",
                 "token_dtype": "uint32", "topk_index_dtype": "uint32",
                 "topk_value_dtype": "float16", "log_values": True,
                 "normalization": "full_model_vocabulary", "generation_temperature": 1.0,
                 "position_alignment": POSITION_ALIGNMENT, "truncation": "right_prefix_no_chunking"}
        for key, value in fixed.items():
            if m.get(key) != value:
                raise ValueError(f"Unsupported offline cache {key}: {m.get(key)!r}")
        _digest(m["tokenizer_hash"], "tokenizer_hash")
        if m.get("tokenizer_vocab_hash") is not None:
            _digest(m["tokenizer_vocab_hash"], "tokenizer_vocab_hash")
        for key in ["hidden_size", "vocab_size", "sequence_length", "top_k", "shard_tokens"]:
            _positive_int(m[key], key)
        if m["top_k"] > m["vocab_size"] or m["vocab_size"] > 2**32 or m["shard_tokens"] < m["sequence_length"]:
            raise ValueError("Invalid vocabulary/shard geometry")
        anchors = m["anchor_layers"]
        if not isinstance(anchors, list) or not anchors or any(type(i) is not int or i < 0 for i in anchors) or len(set(anchors)) != len(anchors):
            raise ValueError("Invalid anchor_layers")
        self.anchor_layers = tuple(anchors)
        self.hidden_size, self.vocab_size = m["hidden_size"], m["vocab_size"]
        self.layouts = _layouts(m["top_k"], len(anchors), self.hidden_size)
        self.documents = {}
        self.shards = {}
        paths = set()
        for index, shard in enumerate(m["shards"]):
            if shard["id"] != index or shard["split"] not in {"train", "eval"}:
                raise ValueError("Invalid shard ID or split")
            if not 0 < _positive_int(shard["tokens"], "shard tokens") <= m["shard_tokens"]:
                raise ValueError("Shard exceeds token capacity")
            if set(shard["files"]) != set(self.layouts):
                raise ValueError("Shard streams are missing or unknown")
            for name, (dtype, shape) in self.layouts.items():
                filename = shard["files"][name]
                path = (self.path / filename).resolve()
                if path.parent != self.path or path in paths:
                    raise ValueError("Invalid or repeated shard file path")
                paths.add(path)
                size = shard["tokens"] * dtype.itemsize * int(np.prod(shape, dtype=np.int64))
                if not path.is_file() or path.stat().st_size != size:
                    raise ValueError(f"Corrupt cache shard size: {filename}")
            self.shards[index] = shard
        offsets = {i: 0 for i in self.shards}
        shard_docs = {i: [] for i in self.shards}
        for doc in m["documents"]:
            doc_id = doc["doc_id"]
            if not isinstance(doc_id, str) or not doc_id or doc_id in self.documents:
                raise ValueError("Duplicate or invalid cache doc_id")
            shard = self.shards[doc["shard"]]
            if doc["split"] != shard["split"] or doc["offset"] != offsets[shard["id"]]:
                raise ValueError("Document split/offset does not match shard")
            if type(doc["offset"]) is not int or not 0 < _positive_int(doc["length"], "document length") <= m["sequence_length"]:
                raise ValueError("Invalid document bounds")
            if _positive_int(doc["original_length"], "original_length") < doc["length"]:
                raise ValueError("Invalid original document length")
            if set(doc["sha256"]) != set(self.layouts):
                raise ValueError("Missing document stream checksums")
            for name, digest in doc["sha256"].items():
                _digest(digest, f"document {name} checksum")
            offsets[shard["id"]] += doc["length"]
            shard_docs[shard["id"]].append(doc_id)
            self.documents[doc_id] = doc
        if not self.documents or m["document_order"] != list(self.documents):
            raise ValueError("Document order is empty or inconsistent")
        for shard_id, shard in self.shards.items():
            docs = shard_docs[shard_id]
            if not docs or offsets[shard_id] != shard["tokens"] or docs != shard["doc_ids"] or shard["doc_id_range"] != [docs[0], docs[-1]]:
                raise ValueError("Document coverage does not match shard")

    def _open_shard(self, shard_id: int):
        if shard_id in self._maps:
            self._maps.move_to_end(shard_id)
            return self._maps[shard_id]
        while len(self._maps) >= self.max_open_shards:
            _, arrays = self._maps.popitem(last=False)
            for array in arrays.values():
                array._mmap.close()
        shard = self.shards[shard_id]
        arrays = {name: np.memmap(self.path / shard["files"][name], mode="r", dtype=dtype,
                                  shape=(shard["tokens"], *shape))
                  for name, (dtype, shape) in self.layouts.items()}
        self._maps[shard_id] = arrays
        return arrays

    def read_document(self, doc_id: str, *, include_hidden_states: bool = True, tokens_only: bool = False) -> dict[str, np.ndarray]:
        if doc_id not in self.documents:
            raise ValueError(f"Document ID {doc_id!r} is absent from offline cache")
        doc = self.documents[doc_id]
        result = {}
        # Hashing and NumPy copies release the GIL. A second reader must not evict
        # and close this mmap while the first still holds a view into it.
        with self._read_lock:
            maps = self._open_shard(doc["shard"])
            for name, array in maps.items():
                if (name == "hidden_states" and not include_hidden_states) or (tokens_only and name != "input_ids"):
                    continue
                view = array[doc["offset"]:doc["offset"] + doc["length"]]
                if _array_hash(view) != doc["sha256"][name]:
                    raise ValueError(f"Corrupt offline cache {name} for document {doc_id!r}: checksum mismatch")
                result[name] = np.array(view, copy=True)
        return result

    def document_ids(self, split: str | None = None) -> list[str]:
        if split not in {None, "train", "eval"}:
            raise ValueError("split must be train or eval")
        return [doc_id for doc_id, doc in self.documents.items() if split is None or doc["split"] == split]

    def iter_records(self, split: str = "train"):
        for doc_id in self.document_ids(split):
            tokens = self.read_document(doc_id, tokens_only=True)["input_ids"].tolist()
            # `length` feeds HF's LengthGroupedSampler (train_sampling_strategy=
            # "group_by_length"), which otherwise reconstructs it by materializing
            # every input_ids row. Batches pad to their longest member, and this
            # corpus is median 545 / max 4096 tokens, so grouping cuts simulated
            # padding waste at batch 4 from 49.7% to 3.5%. The collators select keys
            # explicitly, so the extra column is inert for everything else.
            yield {"doc_id": doc_id, "input_ids": tokens,
                   "attention_mask": [1] * len(tokens), "length": len(tokens)}

    def to_dataset(self, split: str = "train"):
        from datasets import Dataset, Features, List, Value

        features = Features({"doc_id": Value("string"), "input_ids": List(Value("int64")),
                             "attention_mask": List(Value("int64")), "length": Value("int64")})
        if not self.document_ids(split):
            # from_generator rejects an empty generator even when features are
            # supplied; return an explicitly-typed empty dataset instead so
            # callers can test split emptiness with len().
            return Dataset.from_dict(
                {"doc_id": [], "input_ids": [], "attention_mask": [], "length": []},
                features=features,
            )
        # A generator streams token records into Arrow; hidden arrays stay on disk.
        return Dataset.from_generator(self.iter_records, gen_kwargs={"split": split}, features=features)

    def close(self):
        with self._read_lock:
            for arrays in self._maps.values():
                for array in arrays.values():
                    array._mmap.close()
            self._maps.clear()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_maps"] = OrderedDict()
        state.pop("_read_lock", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._read_lock = threading.Lock()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()

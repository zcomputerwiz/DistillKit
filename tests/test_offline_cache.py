"""Regression gates for OfflineTeacherCache dataset construction."""

import numpy as np

from distillkit.offline_cache import OfflineCacheWriter, OfflineTeacherCache


def _write_document_cache(path, split):
    writer = OfflineCacheWriter(
        path,
        tokenizer_hash="ab" * 32,
        anchor_layers=[0],
        hidden_size=4,
        vocab_size=8,
        sequence_length=8,
        top_k=2,
    )
    tokens = np.arange(8, dtype=np.uint32)
    topk_ids = np.tile(np.array([1, 2], dtype=np.uint32), (8, 1))
    topk_logprobs = np.full((8, 2), -0.6931, dtype=np.float16)
    hidden_states = np.zeros((8, 1, 4), dtype=np.uint8)
    writer.append("doc-0", tokens, topk_ids, topk_logprobs, hidden_states, split=split)
    writer.close()


def test_train_only_cache_yields_empty_eval_dataset_with_schema(tmp_path):
    # Every record specifies split=train; to_dataset("eval") must return an
    # empty typed dataset instead of raising from Dataset.from_generator.
    _write_document_cache(tmp_path / "cache", "train")
    cache = OfflineTeacherCache(tmp_path / "cache")

    from datasets import Features, List, Value

    expected_features = Features(
        {"doc_id": Value("string"), "input_ids": List(Value("int64")),
         "attention_mask": List(Value("int64"))}
    )
    ds_eval = cache.to_dataset("eval")
    assert len(ds_eval) == 0
    assert ds_eval.column_names == ["doc_id", "input_ids", "attention_mask"]
    assert ds_eval.features == expected_features

    ds_train = cache.to_dataset("train")
    assert len(ds_train) == 1
    assert ds_train[0]["doc_id"] == "doc-0"
    assert ds_train[0]["input_ids"] == list(range(8))
    assert ds_train[0]["attention_mask"] == [1] * 8


def test_eval_only_cache_yields_empty_train_dataset(tmp_path):
    _write_document_cache(tmp_path / "cache", "eval")
    cache = OfflineTeacherCache(tmp_path / "cache")

    # do_distill raises "Cache has no training documents" on exactly this.
    assert len(cache.to_dataset("train")) == 0
    assert len(cache.to_dataset("eval")) == 1


def test_concurrent_reads_survive_single_shard_lru_eviction(tmp_path, monkeypatch):
    import pickle
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor
    import distillkit.offline_cache as module

    path = tmp_path / "cache"
    with OfflineCacheWriter(path, tokenizer_hash="ab"*32, anchor_layers=[0],
                            hidden_size=4, vocab_size=8, sequence_length=8,
                            top_k=2, shard_tokens=8) as writer:
        for i in range(2):
            writer.append(str(i), np.full(8, i, dtype=np.uint32),
                          np.tile(np.array([1, 2], dtype=np.uint32), (8, 1)),
                          np.full((8, 2), -2, dtype=np.float16),
                          np.full((8, 1, 4), i, dtype=np.uint8), split="train")
    # Locks and open mappings must not be serialized into dataset workers.
    original = OfflineTeacherCache(path, max_open_shards=1)
    original.read_document("0")
    cache = pickle.loads(pickle.dumps(original))
    original.close()
    active = threading.Event()
    hash_array, open_shard = module._array_hash, cache._open_shard

    def slow_hash(array):
        active.set()
        try:
            time.sleep(0.005)  # simulate hashing a large hidden-state array
            return hash_array(array)
        finally:
            active.clear()

    def checked_open(shard):
        # Detect an unsafe eviction before it can close a live mmap and crash.
        assert not active.is_set(), "eviction attempted while another reader holds a view"
        return open_shard(shard)

    monkeypatch.setattr(module, "_array_hash", slow_hash)
    monkeypatch.setattr(cache, "_open_shard", checked_open)
    barrier = threading.Barrier(2, timeout=5)
    def read(i):
        barrier.wait()
        for _ in range(8):
            result = cache.read_document(str(i))
            np.testing.assert_array_equal(result["input_ids"], np.full(8, i))
            np.testing.assert_array_equal(result["hidden_states"], np.full((8, 1, 4), i))

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(read, i) for i in range(2)]
        for future in futures:
            future.result()
    assert len(cache._maps) == 1
    cache.close()

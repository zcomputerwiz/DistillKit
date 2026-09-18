"""Vectorized, cached remap of a token store into a compact vocabulary.

The remap inside `smoke_train.py` looped in Python over every token below the cut. At
30.7M tokens and 2.4% missing that is 730k iterations and takes a few seconds; at 3.0B
tokens it is 81.6M, and it also builds an 81.6M-entry list of small arrays before
concatenating. This does the same transformation with one gather per chunk, and writes
the result next to the store so a vocabulary is remapped once per corpus rather than
once per training run.

The trick is to stop treating kept and unkept ids as different cases. Every original id
gets a ragged entry in one table: a kept id expands to the single compact id it maps to,
an unkept id expands to its byte tokens' compact ids. Remapping is then a gather through
that table with no branches, and the byte fallback falls out of the same code path that
handles ordinary tokens.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def build_expansion_table(tokenizer, forward, bytes_ids, counts):
    """One ragged table over every original id: the compact ids it becomes.

    Returns ``(values, offsets)`` where original id ``i`` expands to
    ``values[offsets[i]:offsets[i + 1]]``. Kept ids have length 1. Ids that no tokenizer
    entry decodes -- the padding rows between ``len(tokenizer)`` and
    ``config.vocab_size`` -- get length 0, which is only sound if they never occur, so
    that is checked here rather than assumed.
    """
    size = forward.shape[0]
    decoder = {char: value for value, char in bytes_to_unicode().items()}
    tokens = tokenizer.convert_ids_to_tokens(list(range(size)))

    lengths = np.zeros(size, dtype=np.int64)
    pieces = []
    undecodable = []
    for original in range(size):
        compact = int(forward[original])
        if compact >= 0:
            pieces.append((original, [compact]))
            lengths[original] = 1
            continue
        token = tokens[original]
        if token is None:
            undecodable.append(original)
            continue
        expanded = []
        for char in token:
            if char not in decoder:
                expanded = None
                break
            expanded.append(int(forward[bytes_ids[decoder[char]]]))
        if expanded is None:
            undecodable.append(original)
            continue
        pieces.append((original, expanded))
        lengths[original] = len(expanded)

    if undecodable:
        occurring = [i for i in undecodable if counts[i] > 0]
        if occurring:
            raise SystemExit(
                "%d ids occur in the corpus but decompose into neither a kept id nor "
                "bytes, first %r" % (len(occurring), occurring[:8]))

    offsets = np.zeros(size + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    values = np.zeros(int(offsets[-1]), dtype=np.int64)
    for original, expanded in pieces:
        values[offsets[original]:offsets[original + 1]] = expanded
    return values, offsets


def remap_chunk(chunk, values, offsets):
    """Remap one block of original ids through the ragged table.

    ``np.repeat`` gives every output slot its source token's offset; subtracting the
    slot's own destination start leaves the position within that token's expansion. So
    the whole thing is two repeats, an arange and a gather.
    """
    chunk = np.asarray(chunk, dtype=np.int64)
    lengths = offsets[chunk + 1] - offsets[chunk]
    total = int(lengths.sum())
    starts = np.zeros(chunk.shape[0], dtype=np.int64)
    np.cumsum(lengths[:-1], out=starts[1:])
    within = np.arange(total, dtype=np.int64) - np.repeat(starts, lengths)
    return values[np.repeat(offsets[chunk], lengths) + within]


def remap_store(source, destination, values, offsets, width, chunk_tokens=64_000_000,
                progress=None):
    """Stream a ``.bin`` store through the table into a new one, returning its length."""
    raw = np.memmap(source, dtype=np.uint32, mode="r")
    total = int(raw.shape[0])
    written = 0
    with open(destination, "wb") as handle:
        for start in range(0, total, chunk_tokens):
            out = remap_chunk(raw[start:start + chunk_tokens], values, offsets)
            handle.write(out.astype(width).tobytes())
            written += int(out.shape[0])
            if progress is not None:
                progress(min(start + chunk_tokens, total), total, written)
    return total, written


def cached_remap(store, split, vocab, tokenizer, forward, bytes_ids, counts, kept,
                 verbose=True):
    """Remap ``split`` once and reuse it, returning ``(memmap, original, compact)``.

    The cache is keyed on the kept set rather than on ``vocab`` alone, because two runs
    can ask for the same size and get different ids if the counts they ranked differ.
    """
    store = Path(store)
    fingerprint = _fingerprint(kept)
    target = store / ("%s-v%d.bin" % (split, vocab))
    meta = store / ("%s-v%d.json" % (split, vocab))
    width = np.uint16 if len(kept) <= 65_536 else np.uint32

    if target.exists() and meta.exists():
        record = json.loads(meta.read_text(encoding="utf-8"))
        if record.get("fingerprint") == fingerprint:
            stream = np.memmap(target, dtype=width, mode="r")
            if stream.shape[0] == record["compact_tokens"]:
                if verbose:
                    print("remap: reused %s (%d -> %d, inflation %.4f)"
                          % (target.name, record["original_tokens"],
                             record["compact_tokens"], record["inflation"]), flush=True)
                return stream, record["original_tokens"], record["compact_tokens"]

    def progress(done, total, written):
        if verbose:
            print("  remap %s: %d/%d original, %d compact" % (split, done, total, written),
                  flush=True)

    values, offsets = build_expansion_table(tokenizer, forward, bytes_ids, counts)
    original, compact = remap_store(store / ("%s.bin" % split), target, values, offsets,
                                    width, progress=progress)

    meta.write_text(json.dumps({
        "split": split, "vocab": vocab, "kept_ids": len(kept),
        "fingerprint": fingerprint, "original_tokens": original,
        "compact_tokens": compact, "inflation": compact / original,
        "dtype": np.dtype(width).name,
    }, indent=2), encoding="utf-8")
    return np.memmap(target, dtype=width, mode="r"), original, compact


def _fingerprint(kept):
    import hashlib
    digest = hashlib.blake2b(np.asarray(kept, dtype=np.int64).tobytes(), digest_size=16)
    return digest.hexdigest()


def byte_token_ids(tokenizer):
    """The id of each of the 256 single-byte tokens, in byte order.

    This is byte-level BPE, so every byte has a token. The alphabet is the map the
    tokenizer itself uses to make bytes printable; inverting it turns a token string back
    into the bytes it stands for.
    """
    encoder = bytes_to_unicode()
    vocabulary = tokenizer.get_vocab()
    ids = []
    for value in range(256):
        token = encoder[value]
        if token not in vocabulary:
            raise SystemExit("byte %d has no token; byte fallback would have holes" % value)
        ids.append(vocabulary[token])
    return ids


def build_vocabulary(counts, tokenizer, target):
    """Keep the bytes, the specials, then the most frequent ids until `target` is full."""
    specials = sorted(tokenizer.get_added_vocab().values())
    bytes_ids = byte_token_ids(tokenizer)
    forced = list(dict.fromkeys(bytes_ids + specials))
    if len(forced) > target:
        raise SystemExit("target %d cannot hold %d forced ids" % (target, len(forced)))

    order = np.argsort(counts)[::-1]
    kept = list(forced)
    seen = set(forced)
    for candidate in order:
        if len(kept) >= target:
            break
        candidate = int(candidate)
        if candidate not in seen:
            kept.append(candidate)
            seen.add(candidate)
    kept.sort()
    forward = np.full(counts.shape[0], -1, dtype=np.int32)
    forward[np.asarray(kept)] = np.arange(len(kept), dtype=np.int32)
    return kept, forward, bytes_ids


def bytes_to_unicode():
    """The byte-level BPE alphabet: each of the 256 bytes as a printable character.

    Defined here rather than imported. It is a fixed mapping that every byte-level BPE
    tokenizer shares, and transformers has moved it twice -- it is currently inside
    `convert_slow_tokenizer`, which is not an interface to depend on.
    """
    printable = (list(range(ord("!"), ord("~") + 1))
                 + list(range(ord("\xa1"), ord("\xac") + 1))
                 + list(range(ord("\xae"), ord("\xff") + 1)))
    mapped = printable[:]
    spare = 0
    for value in range(256):
        if value not in printable:
            printable.append(value)
            mapped.append(256 + spare)
            spare += 1
    return dict(zip(printable, (chr(point) for point in mapped)))


def _self_check():
    """The vectorized path must agree with the obvious loop, including byte fallback."""
    rng = np.random.default_rng(0)
    size = 64
    forward = np.full(size, -1, dtype=np.int32)
    kept = [0, 1, 2, 3, 10, 11, 12]
    forward[np.asarray(kept)] = np.arange(len(kept), dtype=np.int32)
    # id -> expansion, built by hand: kept ids are themselves, the rest are two "bytes".
    lengths = np.zeros(size, dtype=np.int64)
    table = []
    for original in range(size):
        compact = int(forward[original])
        table.append([compact] if compact >= 0 else [0, 1])
        lengths[original] = len(table[-1])
    offsets = np.zeros(size + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    values = np.concatenate([np.asarray(row, dtype=np.int64) for row in table])

    tokens = rng.integers(0, size, size=5000)
    expected = np.concatenate([np.asarray(table[int(t)], dtype=np.int64) for t in tokens])
    assert np.array_equal(remap_chunk(tokens, values, offsets), expected)
    # And chunking must not change the answer.
    halves = np.concatenate([remap_chunk(tokens[:1234], values, offsets),
                             remap_chunk(tokens[1234:], values, offsets)])
    assert np.array_equal(halves, expected)
    print("vocab_remap self-check: ok (%d -> %d)" % (tokens.shape[0], expected.shape[0]))


if __name__ == "__main__":
    _self_check()

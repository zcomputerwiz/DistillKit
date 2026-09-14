"""Deterministic Python corpus construction from The Stack v2.

The Stack v2 ships *metadata only*: a parquet row names a file, its repository, its
license and a ``blob_id``, but not its bytes. The bytes live in Software Heritage's
public object store, one gzip member per blob, addressed by that same ``blob_id``.
That indirection is the whole shape of this module -- everything here exists to turn
a metadata row into verified source text and then into a split assignment that cannot
drift between runs.

Three properties are load-bearing and each is enforced rather than assumed:

**The blob identifier is a checksum.** ``blob_id`` is the SHA-1 of the decompressed
content, so retrieval validates itself: :func:`verify_blob` recomputes it and a
mismatch is a hard failure, never a silently accepted file. The same identity makes
exact-duplicate detection free -- two files with one ``blob_id`` are one blob.

**The split is a function of the repository name, not of arrival order.** A corpus
built by filling ``train`` until it is full and sweeping the remainder into
``heldout`` would let shard ordering decide what is held out, and shard ordering in
The Stack v2 correlates with crawl date. :func:`split_for_repo` hashes the repository
name with a fixed seed instead, so a repository lands in the same split whatever
order it is seen in, whatever else is in the corpus, and whatever the targets are.
Repository-level isolation then follows for free: every file of a repository hashes
to the same bucket, so no repository can straddle two splits.

**Benchmark contamination is detected before writing, not after.** :class:`BenchmarkIndex`
shingles the reference solutions and prompts of MBPP+/HumanEval+ into 64-bit hashes
of fixed-length token runs, which catches copied code through reformatting and
renamed locals without the false positives of loose similarity scoring.

Nothing here touches the network. Retrieval is a URL and a gzip member -- see
:func:`blob_url` and :func:`decompress_blob` -- so the policy is unit-testable with
synthetic fixtures and the driver owns the I/O.
"""

from __future__ import annotations

import collections
import gzip
import hashlib
import re
import struct
import zlib
from dataclasses import dataclass, field
from typing import Iterable, Iterator, Mapping, Sequence

__all__ = [
    "SWH_CONTENT_URL",
    "DEFAULT_PROPORTIONS",
    "DEFAULT_SEED",
    "CONTAMINATION_THRESHOLD",
    "MIN_IDENTIFIERS",
    "SHINGLE",
    "BenchmarkIndex",
    "FilterLimits",
    "SplitPolicy",
    "blob_url",
    "decode_source",
    "decompress_blob",
    "filter_reason",
    "normalize_code",
    "shingles",
    "split_for_repo",
    "token_filter_reason",
    "verify_blob",
]

SWH_CONTENT_URL = "https://softwareheritage.s3.amazonaws.com/content/{blob_id}"

DEFAULT_SEED = 20260914
DEFAULT_PROPORTIONS = {"train": 0.85, "calibration": 0.05, "heldout": 0.10}

#: Token run length for contamination shingles. Short enough to survive a solution
#: being reformatted or partially rewritten; long enough, together with the identifier
#: floor below, that an incidental collision is rare.
SHINGLE = 13

#: Distinct alphabetic tokens a window must contain to be shingled at all. Punctuation
#: tokenizes one character at a time here, so without this a window can be entirely
#: commas and digits -- see :func:`shingles`.
MIN_IDENTIFIERS = 4

#: Distinct shingles a file must share with *one* benchmark problem to count as
#: contaminated. Measured rather than chosen -- ``scratch/code_corpus/calibrate.py``
#: sweeps it against the real solutions and against ordinary retrieved Python, and the
#: false-positive rate is flat at one file in 582 from 2 through 8 while recall keeps
#: climbing as the threshold falls. Four sits at the top of that flat region: 462 of
#: 542 solutions caught verbatim, 461 after reformatting, and the same single ordinary
#: file flagged as at 8.
CONTAMINATION_THRESHOLD = 4

_TOKEN = re.compile(r"[A-Za-z_][A-Za-z_0-9]*|\d+|[^\sA-Za-z_0-9]")
_BLOB_ID = re.compile(r"\A[0-9a-f]{40}\Z")


def blob_url(blob_id: str) -> str:
    """Public unsigned object URL for a Stack v2 blob.

    Software Heritage exposes its object store over plain HTTPS with no credentials
    and no request signing, which is why this is a format string rather than an S3
    client. The identifier is validated because a malformed one would otherwise be
    interpolated into a URL and come back as an opaque 404 among real ones.
    """
    if not _BLOB_ID.match(blob_id or ""):
        raise ValueError("blob_id must be 40 lowercase hex characters, got %r" % (blob_id,))
    return SWH_CONTENT_URL.format(blob_id=blob_id)


def decompress_blob(raw: bytes) -> bytes:
    """Decompress a stored blob, tolerating a bare deflate stream.

    Objects are gzip members. A handful answer as raw deflate without the gzip
    header, so that is tried second rather than counted as a corrupt object -- the
    checksum in :func:`verify_blob` is what decides whether the bytes are right, and
    it runs either way.
    """
    try:
        return gzip.decompress(raw)
    except (OSError, EOFError, zlib.error):
        return zlib.decompress(raw)


def verify_blob(content: bytes, blob_id: str) -> bool:
    """True when the decompressed bytes hash to the identifier that addressed them."""
    return hashlib.sha1(content).hexdigest() == blob_id


def decode_source(content: bytes, encoding: str | None) -> str | None:
    """Decode source bytes, or ``None`` when the file is not text we can use.

    The metadata's ``src_encoding`` is trusted first and UTF-8 is the fallback, both
    strictly: a file that only decodes with replacement characters is undecodable
    content, and quietly substituting U+FFFD would put mojibake in the corpus and
    spend tokens on it. A NUL byte means binary regardless of what decodes.
    """
    if b"\x00" in content:
        return None
    for candidate in (encoding, "utf-8"):
        if not candidate:
            continue
        try:
            return content.decode(candidate)
        except (UnicodeDecodeError, LookupError):
            continue
    return None


@dataclass(frozen=True)
class FilterLimits:
    """Bounds for the metadata-only filters, applied before any blob is fetched.

    These are deliberately loose. The goal of the first corpus is an ordinary Python
    distribution, so the only exclusions are ones that are indefensible to keep:
    machine-generated and vendored files, which the upstream metadata identifies for
    us, and sizes at which a file is either empty or not really source.
    """

    min_bytes: int = 64
    max_bytes: int = 1 << 20
    #: Applied after tokenization, because the worst offenders are invisible in bytes.
    #: A retrieved file of 1,035,623 bytes tokenized to 1,035,618 tokens -- one token
    #: per byte, obfuscated data rather than Python -- and sat comfortably under a
    #: 1 MiB byte cap while being a quarter of an entire training split on its own.
    max_tokens: int = 32_768
    #: Bytes per token below which the tokenizer is splitting nearly every character.
    #: Ordinary Python runs 3 to 4; machine-generated but genuine Python (SNMP MIB
    #: tables and the like) runs around 2; obfuscated or binary-ish content runs 1.
    min_bytes_per_token: float = 1.8
    exclude_generated: bool = True
    exclude_vendor: bool = True
    #: ``None`` keeps every license type. The Stack v2 labels a large fraction of
    #: files ``no_license``; restricting to ``{"permissive"}`` is a one-value change
    #: and is recorded in the manifest either way.
    license_types: frozenset[str] | None = None


def filter_reason(record: Mapping[str, object], limits: FilterLimits) -> str | None:
    """Why this metadata row is excluded, or ``None`` to keep it.

    Ordered cheapest-first and returning a single reason, so the manifest's filter
    counts partition the excluded rows instead of double-counting a file that is both
    vendored and oversized.
    """
    if limits.exclude_generated and record.get("is_generated"):
        return "generated"
    if limits.exclude_vendor and record.get("is_vendor"):
        return "vendor"
    length = record.get("length_bytes") or 0
    if length < limits.min_bytes:
        return "empty"
    if length > limits.max_bytes:
        return "oversized"
    if limits.license_types is not None:
        if record.get("license_type") not in limits.license_types:
            return "license"
    return None


def token_filter_reason(
    token_count: int, byte_length: int, limits: FilterLimits
) -> str | None:
    """Why this file is excluded once its token count is known, or ``None`` to keep it.

    The metadata filters run before retrieval and cannot see either of these: a file's
    token count and its token density are properties of the text, and the text is not
    available until the blob has been fetched. Paying for retrieval and then discarding
    the file is the cost of catching content that is pathological only in tokens.
    """
    if token_count <= 0:
        return "empty"
    if token_count > limits.max_tokens:
        return "token_oversized"
    if byte_length / token_count < limits.min_bytes_per_token:
        return "token_dense"
    return None


def split_for_repo(
    repo_name: str,
    seed: int = DEFAULT_SEED,
    proportions: Mapping[str, float] = DEFAULT_PROPORTIONS,
) -> str:
    """Assign a repository to a split by hashing its name with a fixed seed.

    The hash is BLAKE2b over ``"<seed>:<repo_name>"``, whose leading 8 bytes are read
    as a big-endian integer and scaled into the unit interval. Splits then tile that
    interval in sorted name order, so the assignment depends on nothing but the
    repository name, the seed and the proportion table -- not on insertion order, not
    on how many repositories have been seen, and not on which targets are still
    unmet.

    Sorting the split names matters: iterating a dict would make the tiling depend on
    how the caller happened to build it, which is exactly the kind of invisible
    ordering dependency this function exists to remove.
    """
    total = sum(proportions.values())
    if total <= 0:
        raise ValueError("proportions must sum to a positive value")
    digest = hashlib.blake2b(("%d:%s" % (seed, repo_name)).encode("utf-8"), digest_size=8)
    position = struct.unpack(">Q", digest.digest())[0] / 2.0 ** 64 * total
    edge = 0.0
    names = sorted(proportions)
    for name in names:
        edge += proportions[name]
        if position < edge:
            return name
    return names[-1]  # only reachable through float rounding at the top edge


@dataclass
class SplitPolicy:
    """Token targets per split, and the admission rule that stops each one.

    Two rules, and the distinction between them is the point:

    *A split stops admitting new repositories once it reaches its target.* Files from
    repositories it has already admitted keep flowing, up to ``overshoot``, so a
    repository is not cut off mid-way purely because a counter crossed a round number.

    *A split that is closed is skipped before retrieval, not after.* The split is
    known from metadata alone, so a file bound for a full split costs one dictionary
    lookup rather than an HTTP request and a tokenizer call. Streaming 45M rows to
    keep 25k files is only affordable because of this.

    Partial repository inclusion is inherent to stopping a 45M-file stream early and
    is not an artefact of the targets: almost every repository in the result is
    represented by some of its files. What the targets must not do -- and cannot do
    here -- is influence *which* split a repository belongs to.
    """

    targets: Mapping[str, int]
    overshoot: float = 1.10
    tokens: dict[str, int] = field(default_factory=dict)
    repos: dict[str, set[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in self.targets:
            self.tokens.setdefault(name, 0)
            self.repos.setdefault(name, set())

    def admits(self, split: str, repo_name: str) -> bool:
        """Whether a file from this repository should still be retrieved."""
        target = self.targets.get(split)
        if target is None:
            return False
        if self.tokens[split] >= target * self.overshoot:
            return False
        if self.tokens[split] < target:
            return True
        return repo_name in self.repos[split]

    def record(self, split: str, repo_name: str, tokens: int) -> None:
        self.tokens[split] += tokens
        self.repos[split].add(repo_name)

    @property
    def complete(self) -> bool:
        return all(self.tokens[name] >= target for name, target in self.targets.items())


def normalize_code(source: str) -> list[str]:
    """Reduce source to a token list for shingling.

    Identifiers, numbers and single punctuation characters are kept; whitespace and
    line structure are dropped. That is what lets a reindented or partially reflowed
    copy of a benchmark solution still match, while keeping matching exact at the
    token level so unrelated files do not collide.
    """
    return _TOKEN.findall(source)


def shingles(
    tokens: Sequence[str],
    width: int = SHINGLE,
    min_identifiers: int = MIN_IDENTIFIERS,
) -> Iterator[int]:
    """64-bit hashes of every ``width``-token run carrying enough distinct identifiers.

    The identifier floor is what stops the whole scheme from matching on literal data.
    Because punctuation tokenizes one character at a time, a thirteen-token window over
    ``[1, 2, 3, 4, 5, 6, 7]`` is eleven commas and digits and nothing else -- it appears
    in a benchmark's test list and in any unrelated file with a small integer list, and
    matching on it says nothing about either. Requiring several *distinct* alphabetic
    tokens in the window keeps structure while discarding those runs, at both index and
    query time so the two stay symmetric.
    """
    if len(tokens) < width:
        return
    alphabetic = [bool(token[:1].isalpha() or token[:1] == "_") for token in tokens]
    for start in range(len(tokens) - width + 1):
        window = tokens[start:start + width]
        if min_identifiers:
            names = {token for token, is_name
                     in zip(window, alphabetic[start:start + width]) if is_name}
            if len(names) < min_identifiers:
                continue
        yield int.from_bytes(
            hashlib.blake2b("\x00".join(window).encode("utf-8"), digest_size=8).digest(),
            "big")


class BenchmarkIndex:
    """Shingled benchmark problems, for keeping them out of training data.

    Scoring is per problem, not per shingle, and that distinction is the difference
    between a usable matcher and one that quietly deletes the domain being studied.
    A file that genuinely contains an MBPP solution shares a long consecutive run of
    shingles with *one* problem -- a forty-token solution yields around thirty of them.
    A file that merely writes ordinary Python shares one or two shingles each with a
    scatter of unrelated problems, because idioms like ``dp[i][j] = max(dp[i - 1]``
    and ``for _ in range(n + 1)`` appear in benchmark solutions and in every other
    dynamic-programming file ever written. Counting raw shingles cannot tell those
    apart; :meth:`best` can, and it is what :meth:`contaminated` thresholds.

    The asymmetry still favours over-flagging -- a false positive costs one file out
    of tens of thousands, a false negative puts a benchmark answer in the training set
    -- but "flag anything sharing thirteen tokens with any benchmark" turned out to
    flag algorithm-exercise repositories wholesale, which is a biased corpus rather
    than a clean one.
    """

    def __init__(self, width: int = SHINGLE, threshold: int = CONTAMINATION_THRESHOLD,
                 min_identifiers: int = MIN_IDENTIFIERS) -> None:
        self.width = width
        self.threshold = threshold
        self.min_identifiers = min_identifiers
        #: shingle -> the problem key that first produced it. A shingle two problems
        #: share is generic by construction, so first-writer-wins loses nothing.
        self.hashes: dict[int, str] = {}
        self.sources = 0

    def _shingles(self, text: str) -> Iterator[int]:
        return shingles(normalize_code(text or ""), self.width, self.min_identifiers)

    def add(self, text: str, key: str | None = None) -> int:
        """Index one benchmark text; returns how many new shingles it contributed."""
        before = len(self.hashes)
        label = key or "source-%d" % self.sources
        for value in self._shingles(text):
            self.hashes.setdefault(value, label)
        self.sources += 1
        return len(self.hashes) - before

    def extend(self, texts: Iterable[str], key: str | None = None) -> None:
        for text in texts:
            self.add(text, key)

    def hits(self, source: str) -> "collections.Counter[str]":
        """Distinct indexed shingles found, counted per benchmark problem."""
        found: dict[str, set[int]] = {}
        if self.hashes:
            for value in self._shingles(source):
                key = self.hashes.get(value)
                if key is not None:
                    found.setdefault(key, set()).add(value)
        return collections.Counter({key: len(values) for key, values in found.items()})

    def best(self, source: str) -> tuple[str | None, int]:
        """The single benchmark problem this file most resembles, and by how much."""
        counts = self.hits(source)
        if not counts:
            return None, 0
        key, count = counts.most_common(1)[0]
        return key, count

    def contaminated(self, source: str) -> bool:
        return self.best(source)[1] >= self.threshold

    def __len__(self) -> int:
        return len(self.hashes)

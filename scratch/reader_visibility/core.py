"""Small, independently testable pieces of the reader evaluation protocol."""
from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn


VERSION = "reader-visibility-v1"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def split_records(records, counts, seed=7):
    """Stable membership, deduplicated text; changing counts cannot move fit into test.

    A count of ``None`` takes the whole hash bucket. The buckets are 60/20/20 only in
    expectation, so asking for an exact 60/20/20 of a small pool fails on the shortfall
    rather than quietly rebalancing -- which would be the one thing this function exists
    to prevent.
    """
    if set(counts) != {"fit", "validation", "test"}:
        raise ValueError("request at least two documents in each of fit/validation/test")
    if any(n is not None and n < 2 for n in counts.values()):
        raise ValueError("request at least two documents in each of fit/validation/test")
    buckets = {key: [] for key in counts}
    seen_ids, seen_text = set(), set()
    for row in sorted(records, key=lambda r: str(r["id"])):
        key, text_hash = str(row["id"]), digest(row["text"])
        if key in seen_ids:
            raise ValueError(f"duplicate document ID: {key}")
        seen_ids.add(key)
        if text_hash in seen_text:
            continue
        seen_text.add(text_hash)
        slot = int(digest([VERSION, "split", seed, key])[:16], 16) % 10
        split = "fit" if slot < 6 else "validation" if slot < 8 else "test"
        buckets[split].append({**row, "id": key, "text_sha256": text_hash})
    result = {}
    for split, rows in buckets.items():
        rows.sort(key=lambda r: digest([VERSION, "order", seed, r["id"]]))
        wanted = len(rows) if counts[split] is None else counts[split]
        if len(rows) < wanted:
            raise ValueError(f"need {wanted} {split} documents, found {len(rows)}")
        if wanted < 2:
            raise ValueError(f"the {split} bucket holds {wanted} documents; at least two are needed")
        result[split] = rows[:wanted]
    return result


def donor_map(ids, seed=7):
    """One fixed cycle within a split: no batch-size-one self-shuffles."""
    keys = sorted(str(key) for key in ids)
    if len(keys) < 2 or len(keys) != len(set(keys)):
        raise ValueError("donors require at least two unique document IDs")
    random.Random(seed).shuffle(keys)
    return {key: keys[(index + 1) % len(keys)] for index, key in enumerate(keys)}


def prediction_positions(ids, roles, donor_length=None):
    """A target at token j is scored from hidden/features at j-1, never j."""
    length = len(ids)
    eligible = torch.zeros(max(0, length - 1), dtype=torch.bool)
    for start, stop in roles.get("assistant", []):
        if not 0 <= start <= stop <= length:
            raise ValueError("role span outside the token sequence")
        eligible[max(1, start) - 1:max(1, stop) - 1] = True
    total = int(eligible.sum())
    if donor_length is not None:
        # Use the common natural prefix. No fabricated padding rows enter a score.
        eligible[max(0, donor_length - 1):] = False
    positions = eligible.nonzero().flatten()
    return positions, torch.tensor(ids, dtype=torch.long)[positions + 1], total


@torch.no_grad()
def reader_representations(reader, query, features, upstream_value_weight=None):
    """Query remains fixed; branches remain separate; call the real reader methods.

    Both the mathematical update and the update surviving residual-add rounding are
    retained. For BF16 they need not agree. Convolution runs over the complete causal
    sequence before any assistant-token selection.
    """
    if reader.training or any(p.requires_grad for p in reader.parameters()):
        raise ValueError("reader must be eval() and completely frozen")
    if query.ndim != 4 or features.ndim != 3 or query.shape[:2] != features.shape[:2]:
        raise ValueError("expected query [B,T,branches,D], features [B,T,F]")
    cast_features = features.to(device=query.device, dtype=query.dtype)
    value = reader.value_proj(cast_features)
    admission = reader._admission(query).to(value.dtype)
    gated = admission * value.unsqueeze(-2)
    conv = reader._short_conv(value)
    output = reader(query, features)
    if not torch.equal(output, query + gated + conv):
        raise ValueError("reader decomposition no longer matches its actual forward")
    result = {
        "raw_table": features, "query": query, "ungated_value": value,
        "admission": admission, "gated_value": gated, "convolution": conv,
        "reader_update": gated.float() + conv.float(),
        "realized_update": output.float() - query.float(),
    }
    if upstream_value_weight is not None:
        if upstream_value_weight.ndim != 2 or upstream_value_weight.shape[1] != features.shape[-1]:
            raise ValueError("upstream value projection has incompatible feature dimension")
        result["upstream_value"] = features.float() @ upstream_value_weight.to(features.device).float().T
    if any(not torch.isfinite(value).all() for value in result.values()):
        raise ValueError("non-finite reader representation")
    return result


class MatchedExaminer(nn.Module):
    """Equal trainable parameter counts after frozen random projections.

    Each input slot (query and/or memory) gets the same width. There is no per-token
    normalization that would erase gate amplitude. Centers/scales must be computed
    on the fit split only. Random compression limits the grade to this decoder family;
    seeds/widths must be matched across arms and selected without examining test scores.
    """
    def __init__(self, input_dims, hidden_size, width=256, seed=7, statistics=None):
        super().__init__()
        if not input_dims or width < 1 or hidden_size < 1 or min(input_dims) < width:
            raise ValueError("positive dimensions and projection width <= each input dimension required")
        if statistics is not None and len(statistics) != len(input_dims):
            raise ValueError("one fit-split (center, scale) pair is required per input")
        self.input_dims = tuple(input_dims)
        for index, dim in enumerate(input_dims):
            generator = torch.Generator().manual_seed(seed + index)
            self.register_buffer(f"projection_{index}", torch.randn(dim, width, generator=generator) / math.sqrt(width))
            center, scale = ((torch.zeros(dim), torch.ones(dim)) if statistics is None else statistics[index])
            if center.shape != (dim,) or scale.shape != (dim,) or not torch.isfinite(center).all() or not torch.isfinite(scale).all() or (scale <= 0).any():
                raise ValueError("invalid fit-split normalization statistics")
            self.register_buffer(f"center_{index}", center.detach().float().clone())
            self.register_buffer(f"scale_{index}", scale.detach().float().clone())
        self.correction = nn.Linear(len(input_dims) * width, hidden_size, bias=False, dtype=torch.float32)
        nn.init.zeros_(self.correction.weight)

    def forward(self, baseline_hidden, *representations):
        if len(representations) != len(self.input_dims):
            raise ValueError("wrong number of examiner input slots")
        projected = []
        for index, (value, dim) in enumerate(zip(representations, self.input_dims)):
            if value.shape[:-1] != baseline_hidden.shape[:-1] or value.shape[-1] != dim:
                raise ValueError("flatten branch axes explicitly before the examiner")
            normalized = (value.detach().float() - getattr(self, f"center_{index}")) / getattr(self, f"scale_{index}")
            projected.append(normalized @ getattr(self, f"projection_{index}"))
        return baseline_hidden.detach().float() + self.correction(torch.cat(projected, dim=-1))


@torch.no_grad()
def token_nll(hidden, head, targets, chunk_size=256, vocab_size=None):
    """Full-vocabulary ground-truth NLL per token, without dense logits.

    Never materialises ``[tokens, vocab]`` for the whole selection: over a 248,320-wide
    head, 19,000 assistant positions in fp32 would be 18 GiB. ``chunk_size`` is a row
    budget rather than a convenience -- the reason the training loop folds its losses
    into the head is that this is the tensor that does not fit, so a probe that
    assembled it would be measuring a different memory regime than the one it grades.

    A vocab-parallel head owns only a slice of each row, so it is asked for the
    target's log-probability directly instead of for logits that would have to be
    gathered across cards first.
    """
    if chunk_size < 1:
        raise ValueError("positive chunk size required")
    if hidden.ndim != 2 or targets.shape != hidden.shape[:1]:
        raise ValueError("expected [tokens, hidden] and [tokens]")
    if head.training or any(p.requires_grad for p in head.parameters()):
        raise ValueError("language-model head must be eval() and frozen")
    project = getattr(head, "sharded_logits", None)
    device = getattr(head, "device", None) or head.weight.device
    hidden = hidden.to(device)
    targets = targets.to(device).reshape(1, -1, 1)
    dtype = next(head.parameters()).dtype
    pieces = []
    for start in range(0, hidden.shape[0], chunk_size):
        rows = hidden[start:start + chunk_size].unsqueeze(0).to(dtype=dtype)
        ids = targets[:, start:start + chunk_size]
        if project is not None:
            logits = project(rows, vocab_size)
        else:
            logits = head(rows)
            if vocab_size is not None and logits.shape[-1] > vocab_size:
                logits = logits[..., :vocab_size]
        if hasattr(logits, "sparse_logprobs"):
            losses = -logits.sparse_logprobs(ids).squeeze(-1).squeeze(0)
        else:
            losses = -logits.float().log_softmax(-1).gather(-1, ids).squeeze(-1).squeeze(0)
        if not torch.isfinite(losses).all():
            raise ValueError("non-finite token NLL")
        pieces.extend(losses.cpu().tolist())
    return pieces


def compare_scores(reference, candidate, resamples=10000, seed=7):
    """Paired document bootstrap of token-weighted mean NLL; positive gain is good."""
    def keyed(rows):
        out = {}
        for row in rows:
            key = str(row["id"])
            if key in out:
                raise ValueError("duplicate score ID")
            values = np.asarray(row["nll"], dtype=np.float64)
            if values.ndim != 1 or len(values) < 1 or not np.isfinite(values).all() or (values < 0).any():
                raise ValueError("scores must have finite, nonnegative per-token NLL")
            if not row.get("alignment"):
                raise ValueError("score alignment fingerprint required")
            out[key] = (row["alignment"], values)
        return out
    left, right = keyed(reference), keyed(candidate)
    if left.keys() != right.keys() or len(left) < 2 or resamples < 100:
        raise ValueError("paired scores require the same >=2 document IDs and >=100 resamples")
    counts, gains = [], []
    for key in sorted(left):
        fingerprint, base = left[key]
        other_fingerprint, other = right[key]
        if fingerprint != other_fingerprint or base.shape != other.shape:
            raise ValueError(f"token/target alignment mismatch for {key}")
        counts.append(len(base))
        gains.append(float((base - other).sum()))
    counts, gains = np.asarray(counts), np.asarray(gains)
    rng = np.random.default_rng(seed)
    draws = []
    for start in range(0, resamples, 256):
        indices = rng.integers(0, len(counts), size=(min(256, resamples - start), len(counts)))
        draws.extend((gains[indices].sum(axis=1) / counts[indices].sum(axis=1)).tolist())
    interval = np.quantile(draws, [0.025, 0.975]).tolist()
    return {"gain_nats": float(gains.sum() / counts.sum()), "ci95": interval,
            "documents": len(counts), "assistant_tokens": int(counts.sum()),
            "fraction_documents_helped": float((gains > 0).mean()),
            "status": "helpful" if interval[0] > 0 else "harmful" if interval[1] < 0 else "inconclusive",
            "sign_convention": "reference NLL minus candidate NLL; positive is helpful",
            "bootstrap": {"unit": "document", "estimator": "token-weighted mean", "resamples": resamples, "seed": seed}}

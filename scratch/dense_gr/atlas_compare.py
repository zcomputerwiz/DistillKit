"""Paired NLL retention and top-1 transitions from atlas token evidence, on CPU.

The signed retention contrast is G_arm - keep * G_reference, where a positive
gain is a reduction of NLL from the base. Bootstrap documents jointly across
all three arms; do not infer this interval from separate marginal intervals.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def pooled_interval(values, counts, draws=2000, seed=0):
    """Ratio of pooled document sums, with a paired document bootstrap."""
    values, counts = np.asarray(values), np.asarray(counts)
    if values.shape != counts.shape or values.ndim != 1 or not len(values):
        raise ValueError("one aligned value and count per document is required")
    total = counts.sum()
    if total <= 0:
        raise ValueError("the selected role has no targets")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), (draws, len(values)))
    denominator = counts[indices].sum(axis=1)
    valid = denominator > 0
    if not valid.any():
        raise ValueError("no bootstrap sample contains targets")
    samples = values[indices].sum(axis=1)[valid] / denominator[valid]
    return {"estimate": float(values.sum() / total),
            "low": float(np.percentile(samples, 2.5)),
            "high": float(np.percentile(samples, 97.5)),
            "bootstrap_draws_with_targets": int(valid.sum())}


def role_rows(archive, metadata, arm, domain, members):
    sums, hits, counts = [], [], []
    masks = []
    token_hits = []
    for index in range(metadata["documents"][domain]):
        stem = f"observations/{arm}/{domain}/{index}"
        nll = archive[stem + "/nll"]
        hit = archive[stem + "/hit"]
        roles = archive[f"documents/{domain}/{index}/roles"]
        if nll.shape != hit.shape or nll.shape != roles.shape:
            raise ValueError(f"unaligned evidence: {stem}")
        if not np.isfinite(nll).all():
            raise ValueError(f"nonfinite NLL: {stem}")
        mask = np.isin(roles, members)
        sums.append(nll[mask].astype(np.float64).sum())
        hits.append(hit[mask].sum())
        counts.append(mask.sum())
        masks.append(mask)
        token_hits.append(hit)
    return (np.asarray(sums), np.asarray(hits), np.asarray(counts), masks, token_hits)


def compare(archive, metadata, base, reference, arms, keep=0.8, draws=2000):
    available = metadata["arms"]
    if any(name not in available for name in [base, reference, *arms]):
        raise ValueError(f"available evidence arms: {available}")
    roles = metadata["roles"]
    groups = {name: [index] for index, name in enumerate(roles)}
    groups["all"] = list(range(len(roles)))
    groups["own-turns"] = [roles.index(r) for r in ("assistant", "thinking", "tool-call")]
    result = {}
    for domain in metadata["documents"]:
        result[domain] = {}
        for role, members in groups.items():
            b_loss, b_hits, counts, masks, b_token_hits = role_rows(
                archive, metadata, base, domain, members)
            if not counts.sum():
                continue
            r_loss, _, r_counts, _, _ = role_rows(archive, metadata, reference, domain, members)
            if not np.array_equal(counts, r_counts):
                raise ValueError("base and reference target counts differ")
            reference_gain = (b_loss - r_loss).sum() / counts.sum()
            by_arm = {}
            for arm in arms:
                a_loss, a_hits, a_counts, _, a_token_hits = role_rows(
                    archive, metadata, arm, domain, members)
                if not np.array_equal(counts, a_counts):
                    raise ValueError("base and arm target counts differ")
                forgotten = np.asarray([
                    (b & ~a & mask).sum()
                    for b, a, mask in zip(b_token_hits, a_token_hits, masks)])
                learned = np.asarray([
                    (~b & a & mask).sum()
                    for b, a, mask in zip(b_token_hits, a_token_hits, masks)])
                if not np.array_equal(learned - forgotten, a_hits - b_hits):
                    raise ValueError("transition accounting does not match top-1 counts")
                arm_gain = (b_loss - a_loss).sum() / counts.sum()
                by_arm[arm] = {
                    "targets": int(counts.sum()), "documents": len(counts),
                    "documents_with_targets": int((counts > 0).sum()),
                    "base_nll": float(b_loss.sum() / counts.sum()),
                    "reference_gain": float(reference_gain), "arm_gain": float(arm_gain),
                    "retention_fraction": (float(arm_gain / reference_gain)
                                           if reference_gain > 0 else None),
                    "retention_contrast": pooled_interval(
                        (b_loss - a_loss) - keep * (b_loss - r_loss), counts, draws),
                    "nll_delta": pooled_interval(a_loss - b_loss, counts, draws),
                    "forgotten_1_to_0": int(forgotten.sum()),
                    "learned_0_to_1": int(learned.sum()),
                    "forgotten_rate": pooled_interval(forgotten, counts, draws),
                    "learned_rate": pooled_interval(learned, counts, draws),
                    "base_top1": float(b_hits.sum() / counts.sum()),
                    "arm_top1": float(a_hits.sum() / counts.sum())}
            result[domain][role] = by_arm
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--base", default="base")
    parser.add_argument("--reference", default="tuned")
    parser.add_argument("--arm", action="append")
    parser.add_argument("--keep", type=float, default=0.8)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.keep <= 1 or args.draws < 100:
        parser.error("keep must be in [0,1] and draws must be at least 100")
    metadata = json.loads(args.evidence.with_suffix(".json").read_text(encoding="utf-8"))
    arms = args.arm or [a for a in metadata["arms"] if a not in (args.base, args.reference)]
    with np.load(args.evidence, allow_pickle=False) as archive:
        result = compare(archive, metadata, args.base, args.reference, arms, args.keep, args.draws)
    payload = {"evidence": str(args.evidence.resolve()), "base": args.base,
               "reference": args.reference, "keep": args.keep, "bootstrap_seed": 0,
               "domains_sha256": metadata["domains_sha256"], "results": result}
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

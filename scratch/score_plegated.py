"""Assistant-only paired comparison for the direction-gated sidecar.

The reply bundle's per-record JSON carries a top-level NLL over every role alongside the
`by_role` breakdown, and the generic paired-arm script reads the top-level total -- which
is how "the widening loses at every stage" survived as long as it did. This selects
`by_role.assistant` explicitly and refuses to run if the arms are not document-for-token
comparable.

    python scratch/score_plegated.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

BASE = Path("scratch/independent-eval")
BASELINE = "student-hf"
CONTROL = "widened-ple-stage1-1m"
# Every arm is scored against the transcription. Add a name here once its reply bundle
# exists; the file name follows from it. An arm that is not there yet is skipped.
ARMS = (BASELINE, CONTROL,
        "widened-plegated-stage1-1m",
        "widened-plegated-fp32-stage1-1m",
        "widened-plegated-L16-stage1-1m",
        "widened-plegated-L28-stage1-1m")


def load(path, role="assistant"):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    # `evaluate` writes {"complete": false} before it loads anything, so a run that
    # died leaves a file that exists and holds no scores. Say which, rather than
    # failing on a KeyError three frames down.
    if not data.get("complete", True) or "records" not in data:
        raise ValueError("%s is an incomplete evaluation; rerun or delete it" % path)
    records = data["records"]["nll"]
    ids = [r["id"] for r in records]
    modes = {}
    for mode in records[0]["modes"]:
        modes[mode] = np.array([r["modes"][mode]["by_role"][role]["sum_nll"] for r in records])
    tokens = np.array([r["modes"]["enabled"]["by_role"][role]["tokens"] for r in records])
    return ids, modes, tokens


def paired(left, right, tokens, draws=10000, seed=0):
    delta = left - right
    rng = np.random.default_rng(seed)
    index = rng.integers(0, len(delta), (draws, len(delta)))
    samples = delta[index].sum(1) / tokens[index].sum(1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return delta.sum() / tokens.sum(), low, high


def main():
    loaded = {}
    for name in ARMS:
        path = BASE / ("reply-%s.json" % name)
        if not path.exists():
            if name in (BASELINE, CONTROL):
                print("missing %s" % path)
                return 1
            print("skipping %s (no reply bundle yet)" % name)
            continue
        loaded[name] = load(path)
    treatments = [name for name in ARMS if name != BASELINE and name in loaded]

    reference_ids, _, reference_tokens = loaded[BASELINE]
    for name, (ids, _, tokens) in loaded.items():
        assert ids == reference_ids, "%s scored different documents" % name
        assert (tokens == reference_tokens).all(), "%s has different assistant spans" % name
    print("%d documents, %d assistant tokens\n" % (len(reference_ids), reference_tokens.sum()))

    base = loaded[BASELINE][1]["enabled"]
    print("against the pre-retrofit student:")
    for name in treatments:
        _, modes, tokens = loaded[name]
        for mode in ("enabled", "bypassed"):
            if mode not in modes:
                continue
            estimate, low, high = paired(modes[mode], base, tokens)
            print("  %-28s %-9s %+.6f [%+.6f, %+.6f]" % (name, mode, estimate, low, high))

    _, control, _ = loaded[CONTROL]
    for name in treatments:
        if name == CONTROL:
            continue
        print("\n%s minus the transcription it replaces:" % name)
        _, gated, tokens = loaded[name]
        for mode in ("enabled", "bypassed"):
            if mode not in gated or mode not in control:
                continue
            estimate, low, high = paired(gated[mode], control[mode], tokens)
            verdict = ("gated better" if high < 0
                       else "control better" if low > 0 else "spans zero")
            print("  %-9s %+.6f [%+.6f, %+.6f]  %s" % (mode, estimate, low, high, verdict))

    print("\nwhat the sidecar itself costs, within each arm:")
    for name in treatments:
        _, modes, tokens = loaded[name]
        if "bypassed" not in modes:
            continue
        estimate, low, high = paired(modes["enabled"], modes["bypassed"], tokens)
        print("  %-28s enabled - bypassed %+.6f [%+.6f, %+.6f]" % (name, estimate, low, high))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

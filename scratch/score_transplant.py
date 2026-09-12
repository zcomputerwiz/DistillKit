"""Grade the frozen donor-reader rho sweep on assistant content.

Every arm writes `rho * collapse(reader(rows))` into the layer-1 residual and nothing
else, so within one reply bundle the `bypassed` mode is the untouched stock student
regardless of arm or rho. That makes `enabled - bypassed` a paired within-document
contrast with the same denominator everywhere, and it makes the bypassed column a
consistency check: if two files disagree about it, they were not scored on the same
model and nothing may be compared across them.

Content is the primary grade. `\n` and the two think tags are 5.5% of assistant tokens
and carried 2.45x the whole measured win of the C1 arm they are being compared against
(see ../DistillKit/scratch/row-novelty/RESULTS.md), so an arm that only improves those
has not improved anything the sidecar was built for.

    python scratch/score_transplant.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

BASE = Path("scratch/transplant-eval")
ARMS = ("c1_c1", "donor_c1", "c1_donor", "donor_donor")
GRADES = ("content", "layout", "assistant")


def load(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not data.get("complete") or "records" not in data:
        raise ValueError("%s is incomplete; rerun or delete it" % path)
    records = data["records"]["nll"]
    ids = [r["id"] for r in records]
    out = {}
    for grade in GRADES:
        # A document with no tokens of a grade contributes zero to both sums and zero to
        # the denominator, which is what the bootstrap needs -- not a dropped document,
        # since dropping would change the denominator between arms.
        for mode in ("enabled", "bypassed"):
            out[(grade, mode)] = np.array(
                [r["modes"][mode]["by_role"].get(grade, {}).get("sum_nll", 0.0)
                 for r in records])
        out[(grade, "tokens")] = np.array(
            [r["modes"]["enabled"]["by_role"].get(grade, {}).get("tokens", 0)
             for r in records], dtype=np.float64)
    return ids, out, float(data["audit"].get("rho_override", float("nan")))


def paired(delta, tokens, draws=10000, seed=20260911):
    rng = np.random.default_rng(seed)
    index = rng.integers(0, len(delta), (draws, len(delta)))
    denominator = tokens[index].sum(1)
    samples = np.where(denominator > 0, delta[index].sum(1) / np.maximum(denominator, 1), np.nan)
    low, high = np.nanquantile(samples, [0.025, 0.975])
    return delta.sum() / tokens.sum(), low, high


def main():
    found = {}
    for path in sorted(BASE.glob("*-rho*.json")):
        arm = path.name.split("-rho")[0]
        if arm not in ARMS:
            continue
        try:
            # rho comes from the audit, not the filename: the shell tag strips the
            # decimal point, so 1.0 and 10 both write "rho10".
            ids, values, rho = load(path)
        except ValueError as problem:
            print("skipping %s" % problem)
            continue
        found.setdefault(arm, {})[rho] = (ids, values)
    if not found:
        print("no completed evaluations yet")
        return 1

    reference_ids = None
    baselines = {}
    for arm, points in found.items():
        for rho, (ids, values) in points.items():
            if reference_ids is None:
                reference_ids = ids
            elif ids != reference_ids:
                raise ValueError("%s rho=%s scored different documents" % (arm, rho))
            # The bypassed model is the stock student in every file; if it drifts, the
            # arms are not on a common baseline and the deltas are not comparable.
            baselines.setdefault(round(values[("content", "bypassed")].sum(), 3), []).append(
                "%s@%s" % (arm, rho))
    if len(baselines) != 1:
        raise ValueError("bypassed content NLL differs across files: %s" % baselines)
    total, = baselines
    tokens = next(iter(next(iter(found.values())).values()))[1][("content", "tokens")]
    print("%d documents, %d assistant content tokens, stock content NLL %.6f\n"
          % (len(reference_ids), int(tokens.sum()), total / tokens.sum()))

    for arm in ARMS:
        if arm not in found:
            continue
        print("%s" % arm)
        print("     rho    content                              layout      assistant")
        for rho in sorted(found[arm]):
            _, values = found[arm][rho]
            cells = []
            for grade in GRADES:
                delta = values[(grade, "enabled")] - values[(grade, "bypassed")]
                estimate, low, high = paired(delta, values[(grade, "tokens")])
                cells.append((estimate, low, high))
            content, layout, assistant = cells
            print("  %+.3f   %+.6f [%+.6f, %+.6f]   %+.6f   %+.6f"
                  % (rho, content[0], content[1], content[2], layout[0], assistant[0]))
        print()
    print("cost is enabled - bypassed against the stock student; negative is better.")
    print()
    frontier(found)
    return 0


def frontier(found, targets=(-0.03, -0.10)):
    """What each arm charges in content for the same amount of layout.

    rho is not comparable across arms: the readers differ in output norm, so one arm's
    rho=0.3 can write as hard as another's rho=10. Comparing at matched *effect* removes
    that. Layout benefit is the axis because it is the only thing any arm buys; content
    cost is the price. Linear interpolation between the two bracketing rho points is
    crude, but the curves are monotone in rho over the sampled range and the gaps here
    are multiples, not percentages.
    """
    print("content cost at matched layout benefit (linear between bracketing rho):")
    print("  arm            " + "".join("   layout %+.2f" % t for t in targets))
    for arm in ARMS:
        if arm not in found:
            continue
        points = []
        for rho in sorted(found[arm]):
            values = found[arm][rho][1]
            layout = ((values[("layout", "enabled")] - values[("layout", "bypassed")]).sum()
                      / values[("layout", "tokens")].sum())
            content = ((values[("content", "enabled")] - values[("content", "bypassed")]).sum()
                       / values[("content", "tokens")].sum())
            points.append((layout, content))
        points.sort()
        cells = []
        for target in targets:
            price = None
            for (low_layout, low_cost), (high_layout, high_cost) in zip(points, points[1:]):
                if low_layout <= target <= high_layout:
                    span = high_layout - low_layout
                    weight = 0.0 if span == 0 else (target - low_layout) / span
                    price = low_cost + weight * (high_cost - low_cost)
                    break
            cells.append("  never reached" if price is None else "      %+.6f" % price)
        print("  %-14s%s" % (arm, "".join(cells)))
    print()
    print("  lower is better; an arm that never reaches a target cannot produce that")
    print("  much layout benefit at any rho sampled, at any price.")


if __name__ == "__main__":
    raise SystemExit(main())

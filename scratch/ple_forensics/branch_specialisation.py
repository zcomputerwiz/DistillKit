"""Do Flash-Next's four PLE branches specialise by token class?

The hypothesis is that the donor routes predictable local/structural work into a subset
of its four residual branches while lexical content occupies others, and that collapsing
all of it into Qwen3.5's single stream is why the retrofit became a newline predictor.

Most of that hypothesis cannot be tested on this machine. Flash-Next is 360 GB in bf16
(48 layers, 512 experts, top-10); what is here is a 93.7 GB IQ4_XS GGUF and an HF
snapshot carrying config, index and tokenizer but **no weights**. Running llama.cpp is
out under a standing constraint. So the branchwise gradient matrix and the causal branch
ablation -- the two decisive pieces -- need a full forward and backward through a model
that does not fit and cannot be run.

One part *is* reachable, and it happens to be the part that can show specialisation the
gates cannot.

For the **direct** PLE write, `delta h_s = g_s v`: all four branches receive the same
2560-D value, so the branch energy fraction is

    e_s = ||g_s v||^2 / sum_j ||g_j v||^2 = g_s^2 / sum_j g_j^2

which is a pure function of the gate vector and does not depend on `v` at all. Direct
branch-energy specialisation and gate-amplitude specialisation are therefore the *same*
measurement, not two, and neither can be computed without the donor's residual stream.

The **convolution** path is different. Its input is `v = value_proj(rows)`, which depends
only on the n-gram table and the token ids -- not on any hidden state. The four filter
banks are genuinely distinct (flattened cosine -0.216 to +0.506, measured in the donor
preflight), so they are the one component that can specialise without the gate, and they
can be evaluated on held-out text with no model at all.

That makes this a falsification test rather than a confirmation: if the four banks show
no class-conditional difference, Codex's own stop rule fires and branch specialisation is
not supported. If they do differ by class, the decisive measurements still require the
donor and remain out of reach here.

    python scratch/ple_forensics/branch_specialisation.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from scratch.ple_forensics.token_classes import CLASSES, class_of, summarise

STREAMS = Path("scratch/ple_forensics/donor-streams.npz")
STUDENT = "D:/DeepThought/Projects/HybridModel/student-hf"
OUT = Path("scratch/ple_forensics/branch-specialisation.json")


def bootstrap_mean(values, draws=4000, seed=20260912):
    rng = np.random.default_rng(seed)
    if not len(values):
        return float("nan"), float("nan"), float("nan")
    index = rng.integers(0, len(values), (draws, len(values)))
    samples = values[index].mean(1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(values.mean()), float(low), float(high)


def main():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(STUDENT, local_files_only=True)
    packed = dict(np.load(STREAMS, allow_pickle=True))
    streams = packed["streams"].astype(np.float64)          # [positions, 4, hidden]
    targets = packed["token_ids"]
    if len(targets) != len(streams):
        raise SystemExit("token_ids and streams disagree: %d vs %d"
                         % (len(targets), len(streams)))
    labels = class_of(targets, tokenizer)
    print("%d positions, %d branches, %d dims\n" % (*streams.shape,))
    print(summarise(targets, tokenizer))

    # Energy fraction per branch per position. This is the quantity that would differ by
    # class if the filter banks specialised; for the direct write it would be identically
    # the gate profile, which is why only the convolution path is informative here.
    energy = (streams ** 2).sum(-1)                          # [positions, 4]
    total = energy.sum(-1, keepdims=True)
    fraction = np.divide(energy, total, out=np.zeros_like(energy), where=total > 0)

    report = {"positions": int(len(streams)), "branches": int(streams.shape[1]),
              "by_class": {}}
    print("\nbranch energy fraction e_s, by token class (mean [95%])")
    header = "  %-12s %7s " % ("class", "tokens")
    header += " ".join("%-26s" % ("branch %d" % s) for s in range(streams.shape[1]))
    print(header)
    for name in CLASSES:
        mask = labels == name
        if mask.sum() < 20:
            print("  %-12s %7d  (too few to report)" % (name, mask.sum()))
            continue
        cells, row = [], {}
        for branch in range(streams.shape[1]):
            mean, low, high = bootstrap_mean(fraction[mask, branch])
            row["branch_%d" % branch] = {"mean": mean, "low": low, "high": high}
            cells.append("%-26s" % ("%.4f [%.4f, %.4f]" % (mean, low, high)))
        report["by_class"][name] = row
        print("  %-12s %7d %s" % (name, mask.sum(), " ".join(cells)))

    # The stop rule: if whitespace and lexical have the same profile there is nothing to
    # route. Reported as the gap per branch with an interval on the difference.
    print("\nwhitespace minus lexical, per branch (the quantity the hypothesis needs)")
    whitespace, lexical = labels == "whitespace", labels == "lexical"
    rng = np.random.default_rng(20260912)
    gaps = {}
    for branch in range(streams.shape[1]):
        left, right = fraction[whitespace, branch], fraction[lexical, branch]
        draws = 4000
        a = left[rng.integers(0, len(left), (draws, len(left)))].mean(1)
        b = right[rng.integers(0, len(right), (draws, len(right)))].mean(1)
        difference = a - b
        low, high = np.quantile(difference, [0.025, 0.975])
        gap = float(left.mean() - right.mean())
        gaps["branch_%d" % branch] = {"gap": gap, "low": float(low), "high": float(high)}
        verdict = "separates" if (low > 0) == (high > 0) else "spans zero"
        print("  branch %d  %+.5f [%+.5f, %+.5f]  %s" % (branch, gap, low, high, verdict))
    report["whitespace_minus_lexical"] = gaps

    spread = max(abs(v["gap"]) for v in gaps.values())
    even = 1.0 / streams.shape[1]
    print("\n  largest class gap in any branch: %.5f, against an even split of %.4f"
          % (spread, even))
    print("  branch profiles are %s"
          % ("essentially identical -- specialisation not supported"
             if spread < 0.01 else "class-dependent -- see the caveat in the docstring"))

    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nwrote %s" % OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

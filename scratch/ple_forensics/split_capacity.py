"""Does the capacity probe's gain survive on ordinary content?

The probe is the only measurement in this project that still argues the table holds
something a model can use. Everything else -- the depth sweep, the A/B/C arms, the gate
diagnosis, the donor transplant -- converged on a device that predicts layout and
charges content for it. If the probe's gain is also layout, the premise is gone. If it
survives on content, the problem is transport or objective and there is something left
to build.

The original probe persisted only aggregates (`reduction="sum"`), so this cannot be
answered from stored artifacts; `--per-token` emits the NLL, the target token id and the
document index for the baseline and every arm, and this splits them.

Two things this is careful about.

**The layout mask is derived, not hardcoded.** The project's `LAYOUT_TOKEN_IDS` names
three ids -- `\n` and the two think tags -- and that is not the whitespace vocabulary.
This corpus's assistant targets use 35 whitespace-only token types, including `\n\n`
(271), tabs, and eight distinct runs of spaces. A mask that misses them scores `\n\n` as
content, and `\n\n` alone is over half of what such a mask would report as the content
gain. Any token whose decoded text is entirely whitespace is layout here, whatever its
id.

**The baseline is the reference, not the shuffled control.** `table - shuffled` reads as
"real signal" only if the shuffled arm is neutral. It is not: a free 2560x2560 matrix
fits something from any correlated input, and on content the shuffled arm is *worse*
than no readout at all. Against that, `table - shuffled` can be positive while the table
adds nothing, because failing to hurt is not the same as helping. Both comparisons are
reported, and `table - baseline` is the one that answers the question.

    python scratch/ple_forensics/split_capacity.py
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

BASE = Path("scratch/ple_forensics")
PER_TOKEN = BASE / "per-token"
RESULTS = BASE / "table-capacity.json"
ARMS = ("baseline", "table", "shuffled_control")
THINK_TOKEN_IDS = (248068, 248069)
STUDENT = "D:/DeepThought/Projects/HybridModel/student-hf"


def load():
    packed = {}
    for arm in ARMS:
        path = PER_TOKEN / ("%s.npz" % arm)
        if not path.exists():
            raise SystemExit("%s missing; run the probe with --per-token" % path)
        packed[arm] = dict(np.load(path, allow_pickle=True))
    # Every arm scores the same positions in the same order, so the target column must
    # agree exactly. If it does not, the arms are not paired and no difference between
    # them means anything.
    reference = packed["baseline"]["target"]
    for arm in ARMS:
        if not np.array_equal(packed[arm]["target"], reference):
            raise SystemExit("%s scored different tokens than the baseline" % arm)
    return packed, reference


def layout_mask(targets, tokenizer):
    """Whitespace-only decoded text, plus the think tags. Derived from the vocabulary."""
    whitespace = set()
    for token in np.unique(targets):
        text = tokenizer.decode([int(token)])
        if text and not text.strip():
            whitespace.add(int(token))
    return np.isin(targets, sorted(whitespace | set(THINK_TOKEN_IDS))), whitespace


def paired(delta, documents, draws=10000, seed=20260911):
    """Bootstrap over documents, the unit the rest of the project resamples."""
    groups = np.unique(documents)
    sums = np.array([delta[documents == group].sum() for group in groups])
    counts = np.array([(documents == group).sum() for group in groups], dtype=np.float64)
    rng = np.random.default_rng(seed)
    index = rng.integers(0, len(groups), (draws, len(groups)))
    denominator = counts[index].sum(1)
    samples = np.where(denominator > 0,
                       sums[index].sum(1) / np.maximum(denominator, 1), np.nan)
    low, high = np.nanquantile(samples, [0.025, 0.975])
    return delta.sum() / len(delta), low, high


def main():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(STUDENT, local_files_only=True)
    packed, targets = load()
    documents = packed["baseline"]["document"]
    is_layout, whitespace = layout_mask(targets, tokenizer)
    views = (("all", np.ones(len(targets), bool)),
             ("layout", is_layout), ("content", ~is_layout))

    print("%d assistant tokens over %d documents" % (len(targets), len(np.unique(documents))))
    print("%d whitespace-only target types; layout is %d tokens (%.1f%%)\n"
          % (len(whitespace), is_layout.sum(), 100 * is_layout.mean()))

    print("absolute NLL")
    print("  %-18s %10s %10s %10s" % ("arm", "all", "layout", "content"))
    for arm in ARMS:
        print("  %-18s %10.4f %10.4f %10.4f"
              % (arm, *[packed[arm]["nll"][mask].mean() for _, mask in views]))

    print("\ngain; negative is better")
    print("  %-22s %-34s %-34s %s" % ("comparison", "all", "layout", "content"))
    for label, left, right in (("table - baseline", "table", "baseline"),
                               ("shuffled - baseline", "shuffled_control", "baseline"),
                               ("table - shuffled", "table", "shuffled_control")):
        cells = []
        for _, mask in views:
            estimate, low, high = paired(
                packed[left]["nll"][mask] - packed[right]["nll"][mask], documents[mask])
            cells.append("%+.6f [%+.6f, %+.6f]" % (estimate, low, high))
        print("  %-22s %-34s %-34s %s" % (label, *cells))

    total_gain = (packed["table"]["nll"] - packed["baseline"]["nll"]).sum()
    layout_gain = (packed["table"]["nll"][is_layout]
                   - packed["baseline"]["nll"][is_layout]).sum()
    print("\n  of %.1f total nats gained over baseline, %.1f (%.1f%%) are layout tokens,"
          % (total_gain, layout_gain, 100 * layout_gain / total_gain))
    print("  which are %.1f%% of the scored tokens." % (100 * is_layout.mean()))

    content = ~is_layout
    delta = packed["table"]["nll"] - packed["baseline"]["nll"]
    gains, counts = Counter(), Counter()
    for token, value in zip(targets[content], delta[content]):
        gains[int(token)] += float(value)
        counts[int(token)] += 1
    total = delta[content].sum()
    print("\nwhere the table-over-baseline gain sits, content tokens only")
    print("  %+.1f nats over %d tokens, %d distinct target types"
          % (total, content.sum(), len(gains)))
    ranked = sorted(gains.items(), key=lambda kv: kv[1])
    print("\n  %-8s %-20s %8s %10s" % ("id", "token", "count", "nats"))
    for token, value in ranked[:12]:
        text = tokenizer.decode([token]).replace("\n", "\\n").replace("\r", "\\r")
        print("  %-8d %-20s %8d %10.2f" % (token, ascii(text[:18]), counts[token], value))
    print("  %-8s %-20s %8s %10s" % ("", "... worst:", "", ""))
    for token, value in ranked[-3:]:
        text = tokenizer.decode([token]).replace("\n", "\\n").replace("\r", "\\r")
        print("  %-8d %-20s %8d %10.2f" % (token, ascii(text[:18]), counts[token], value))

    print("\n  concentration (cumulative, best-first)")
    order = [value for _, value in ranked]
    for cut in (1, 10, 100, 1000):
        if cut <= len(order):
            print("    top %-6d %9.1f nats" % (cut, sum(order[:cut])))
    print("    all %-6d %9.1f nats" % (len(order), total))
    print("    -- cumulative beyond the total means the tail is net harmful")

    if RESULTS.exists():
        stored = json.loads(RESULTS.read_text(encoding="utf-8"))
        print("\nprobe aggregates, for cross-check: baseline %.4f  table %.4f  shuffled %.4f"
              % (stored["baseline_nll"], stored["table"]["eval_nll"],
                 stored["shuffled_control"]["eval_nll"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

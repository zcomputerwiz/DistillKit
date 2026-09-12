"""The two-stream pilot's read-out: does separation buy what removal did not?

Arms A (stock), S (PLE into the ordinary residual) and M (PLE into a private lane read
through learned zero-cost gates) train identically and are scored on the same held-out
assistant tokens. The comparisons that decide the branch:

    M - A on lexical      does separation beat doing nothing?
    M - S on lexical      does separation beat the known retrofit? -- the revealing one
    S - A on lexical      the retrofit's content cost, re-measured in this setup
    ON - OFF for M        is any of it causally the lane?

Layout is reported beside content every time, because the interesting intermediate
outcome is not a content win: it is M keeping S's whitespace gain *without* S's content
damage, which is what representational interference predicts and a single stream cannot
do.

    python scratch/ple_forensics/score_costream.py scratch/ple_forensics/costream/1e5
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from scratch.ple_forensics.offload_arms import STUDENT
from scratch.ple_forensics.score_arms import paired
from scratch.ple_forensics.token_classes import CLASSES, class_of

ARMS = ("A", "S", "M")


def load(directory):
    packed = {}
    for name in ("baseline", *ARMS, "M-off"):
        path = directory / ("%s.npz" % name)
        if path.exists():
            packed[name] = dict(np.load(path, allow_pickle=True))
    return packed


def compare(packed, left, right, labels, documents, title):
    print("\n%s" % title)
    for name in CLASSES:
        mask = labels == name
        if not mask.any():
            continue
        delta = packed[left]["nll"][mask] - packed[right]["nll"][mask]
        estimate, low, high = paired(delta, documents[mask])
        verdict = ("excludes zero" if high < 0 or low > 0 else "spans zero")
        print("  %-12s %+.6f [%+.6f, %+.6f]  %s" % (name, estimate, low, high, verdict))


def reads(packed, labels):
    """Per-layer read strength by token class -- the specialisation question.

    No class supervision reaches the gate, so any spread here is something joint training
    found. A gate that returns the same strength for whitespace and for lexical content
    is a constant admission, not a router, and the explicit lane has bought no
    specialisation regardless of what the NLL says.
    """
    layers = sorted({key.split("_L")[1] for key in packed["M"] if key.startswith("alpha_L")},
                    key=int)
    if not layers:
        return
    print("\nread strength by class, alpha = s * g(h), and read-to-stream norm ratio")
    print("  %-6s %s" % ("layer", "  ".join("%-22s" % name for name in CLASSES)))
    for layer in layers:
        alpha, ratio = packed["M"]["alpha_L%s" % layer], packed["M"]["ratio_L%s" % layer]
        cells = []
        for name in CLASSES:
            mask = labels == name
            cells.append("%7.4f / %-8.5f  " % (alpha[mask].mean(), ratio[mask].mean())
                         if mask.any() else "%-22s" % "-")
        print("  L%-5s %s" % (layer, "".join(cells)))
    spread = []
    for layer in layers:
        alpha = packed["M"]["alpha_L%s" % layer]
        means = [alpha[labels == name].mean() for name in CLASSES if (labels == name).any()]
        spread.append(max(means) - min(means))
    print("  largest between-class gap in alpha: %.6f (layer L%s)"
          % (max(spread), layers[int(np.argmax(spread))]))


def main():
    directory = Path(sys.argv[1] if len(sys.argv) > 1
                     else "scratch/ple_forensics/costream/1e5")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(STUDENT, local_files_only=True)
    packed = load(directory)
    missing = [name for name in ("baseline", *ARMS) if name not in packed]
    if missing:
        print("missing %s in %s" % (", ".join(missing), directory))
        return 1

    reference = packed["baseline"]["target"]
    for name, values in packed.items():
        if not np.array_equal(values["target"], reference):
            raise SystemExit("%s scored different tokens than the baseline" % name)
    documents = packed["baseline"]["document"]
    labels = class_of(reference, tokenizer)

    print("%d assistant tokens over %d documents" % (len(reference), len(np.unique(documents))))
    print("  " + "  ".join("%s %d" % (name, (labels == name).sum()) for name in CLASSES))

    print("\nabsolute NLL")
    print("  %-10s %s" % ("arm", "".join("%12s" % name for name in CLASSES)))
    for name in ("baseline", *ARMS, "M-off"):
        if name not in packed:
            continue
        row = "".join("%12.5f" % packed[name]["nll"][labels == klass].mean()
                      for klass in CLASSES)
        print("  %-10s %s" % (name, row))

    for name in ARMS:
        compare(packed, name, "baseline", labels, documents, "%s minus the untrained model" % name)

    compare(packed, "M", "A", labels, documents,
            "the primary question: M minus A (does separation beat doing nothing?)")
    compare(packed, "M", "S", labels, documents,
            "the revealing one: M minus S (does separation beat the retrofit?)")
    compare(packed, "S", "A", labels, documents,
            "for reference: S minus A (the retrofit's own cost in this setup)")

    if "M-off" in packed:
        compare(packed, "M", "M-off", labels, documents,
                "the causal ablation: M with the lane read minus the same model with it shut")

    reads(packed, labels)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

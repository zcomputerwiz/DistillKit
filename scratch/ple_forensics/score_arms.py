"""The A-versus-B stop gate: does removing whitespace selection improve content?

Arms A and B train the same window on the same documents with the same schedule and
seed, differing only in whether `-log P(w | WS)` contributes at whitespace targets. The
question is whether the backbone spends the freed budget on content.

Primary metric, and the only one that decides whether arm D gets built:

    delta = NLL(B, content) - NLL(A, content)

Negative is success. Everything else here is diagnosis of that number.

The whitespace column is reported as its exact factorisation rather than as one figure,
because B is *expected* to lose selection -- it stopped training it -- and expected to
keep detection. A B that also lost detection would be a broken arm, not a result.

    python scratch/ple_forensics/score_arms.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from scratch.ple_forensics.token_classes import class_of

ARMS = Path("scratch/ple_forensics/arms")
STUDENT = "D:/DeepThought/Projects/HybridModel/student-hf"
NAMES = ("baseline", "A", "B")


def paired(delta, documents, draws=10000, seed=20260912):
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
    packed = {}
    for name in NAMES:
        path = ARMS / ("%s.npz" % name)
        if not path.exists():
            print("missing %s" % path)
            return 1
        packed[name] = dict(np.load(path, allow_pickle=True))

    reference = packed["baseline"]["target"]
    for name in NAMES:
        if not np.array_equal(packed[name]["target"], reference):
            raise SystemExit("%s scored different tokens than the baseline" % name)
    documents = packed["baseline"]["document"]
    is_whitespace = class_of(reference, tokenizer) == "whitespace"
    content = ~is_whitespace

    print("%d assistant tokens over %d documents, %d whitespace (%.1f%%)\n"
          % (len(reference), len(np.unique(documents)), is_whitespace.sum(),
             100 * is_whitespace.mean()))

    print("absolute NLL")
    print("  %-10s %10s %10s %12s %12s"
          % ("arm", "content", "whitespace", "ws detect", "ws select"))
    for name in NAMES:
        values = packed[name]
        print("  %-10s %10.5f %10.5f %12.5f %12.5f"
              % (name, values["nll"][content].mean(), values["nll"][is_whitespace].mean(),
                 values["detect"][is_whitespace].mean(),
                 values["select"][is_whitespace].mean()))

    print("\nthe stop gate: B minus A on content")
    delta = packed["B"]["nll"][content] - packed["A"]["nll"][content]
    estimate, low, high = paired(delta, documents[content])
    verdict = ("B better, interval excludes zero" if high < 0
               else "A better, interval excludes zero" if low > 0 else "spans zero")
    print("  %+.6f [%+.6f, %+.6f]   %s" % (estimate, low, high, verdict))

    print("\nfor context, each arm against the untrained reference")
    for name in ("A", "B"):
        for label, mask in (("content", content), ("whitespace", is_whitespace)):
            values = packed[name]["nll"][mask] - packed["baseline"]["nll"][mask]
            estimate, low, high = paired(values, documents[mask])
            print("  %-2s %-11s %+.6f [%+.6f, %+.6f]" % (name, label, estimate, low, high))

    print("\nwhitespace factorisation, B minus A")
    print("  (B is expected to lose selection and keep detection; losing detection")
    print("   would mean the arm is broken rather than informative)")
    for term in ("detect", "select"):
        values = packed["B"][term][is_whitespace] - packed["A"][term][is_whitespace]
        estimate, low, high = paired(values, documents[is_whitespace])
        print("  %-8s %+.6f [%+.6f, %+.6f]" % (term, estimate, low, high))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

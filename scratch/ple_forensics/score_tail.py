"""Where the KL term's harm lives: inside the teacher's top-k, or outside it.

`missing_probability_handling: zero` renormalises the cached top-64 and gives every other
token a target probability of exactly zero. A student that raises probability on a true
token the teacher never ranked is therefore penalised by the KL term for being right.
The sidecar's whole contribution is exactly that kind of correction, so if the truncated
target is what turns it from a help into a harm, S minus A should be much worse on
positions whose true token falls outside the teacher's list -- and the same split under
CE training, which has no teacher, is the control.

    python scratch/ple_forensics/score_tail.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from scratch.ple_forensics.offload_arms import STUDENT
from scratch.ple_forensics.score_arms import paired
from scratch.ple_forensics.token_classes import class_of

DIAGNOSTICS = Path("scratch/ple_forensics/objective/diagnostics")


def main():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(STUDENT, local_files_only=True)
    available = sorted({path.stem.split("-")[1] for path in DIAGNOSTICS.glob("tail-*-A.npz")})
    packed = {}
    for objective in available:
        for arm in ("A", "S"):
            path = DIAGNOSTICS / ("tail-%s-%s.npz" % (objective, arm))
            if not path.exists():
                print("missing %s" % path)
                return 1
            packed[(objective, arm)] = dict(np.load(path))

    reference = packed[("ce", "A")]
    labels = class_of(reference["token"], tokenizer)
    inside = reference["in_topk"].astype(bool)
    documents = reference["document"]
    for key, values in packed.items():
        if not np.array_equal(values["token"], reference["token"]):
            raise SystemExit("%s scored different positions" % (key,))

    print("%d positions, %.1f%% with the true token inside the teacher's top-k\n"
          % (len(inside), 100 * inside.mean()))

    for objective in available:
        print("%s: S minus A, split by whether the true token is in the teacher's top-k"
              % objective)
        delta = packed[(objective, "S")]["nll"] - packed[(objective, "A")]["nll"]
        for name in ("lexical", "whitespace", "punctuation", "control"):
            klass = labels == name
            if not klass.any():
                continue
            row = []
            for label, mask in (("inside", klass & inside), ("outside", klass & ~inside)):
                if not mask.any():
                    row.append("%-8s -" % label)
                    continue
                estimate, low, high = paired(delta[mask], documents[mask])
                row.append("%-8s %+.5f [%+.5f, %+.5f] n=%-5d"
                           % (label, estimate, low, high, mask.sum()))
            print("  %-12s %s" % (name, "   ".join(row)))
        # How much of the mean shift the tail carries, against how much of the corpus it
        # is. A ratio far above 1 is the truncation signature.
        klass = labels == "lexical"
        share = (~inside & klass).sum() / max(klass.sum(), 1)
        outside_total = delta[klass & ~inside].sum()
        whole = delta[klass].sum()
        print("  lexical tail: %.1f%% of positions carrying %.1f%% of the S-A shift\n"
              % (100 * share, 100 * outside_total / whole if whole else float("nan")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

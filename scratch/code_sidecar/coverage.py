"""How well does the canonical structural output set cover Python targets?

The structural sidecar can only bias tokens inside its output set. That set was chosen on
general text, and before fitting anything to Python it is worth knowing what fraction of
Python's layout and punctuation targets it can actually reach -- a sidecar that addresses
the right rows but cannot write to the tokens Python uses would fail for a reason no
amount of refitting would cure.

Diagnostic only. The structural vocabulary is not redesigned in this task.

    python scratch/code_sidecar/coverage.py
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code_training"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ffn_memo"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "residual_gate"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "structural_sidecar"))

import numpy as np

from distillkit.code_classes import CODE_CLASSES
from corpus import TOKENS, TokenStore, class_tables, evaluation_subset, load_config, load_tokenizer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="heldout")
    parser.add_argument("--subset-tokens", type=int, default=1_250_000)
    parser.add_argument("--output", type=Path,
                        default=Path("scratch/code_sidecar/coverage.json"))
    args = parser.parse_args()

    from distillkit.experimental.structural_sidecar import structural_token_ids
    from distillkit.independent_eval import build_token_classes
    from evaluate import split_layout

    config = load_config()
    tokenizer = load_tokenizer()
    code, _ = class_tables(tokenizer, config.vocab_size)
    general = split_layout(build_token_classes(tokenizer, config.vocab_size), tokenizer,
                           config.vocab_size)
    structural = structural_token_ids(general).cpu().numpy()
    covered = np.zeros(config.vocab_size, dtype=bool)
    covered[structural] = True

    store = TokenStore(TOKENS, args.split)
    indices, subset = evaluation_subset(store, args.subset_tokens)
    targets = collections.Counter()
    reachable = collections.Counter()
    for index in indices:
        row = np.asarray(store.document(index), dtype=np.int64)[1:]
        labels = code[row]
        inside = covered[row]
        for value in np.unique(labels):
            chosen = labels == value
            targets[CODE_CLASSES[value]] += int(chosen.sum())
            reachable[CODE_CLASSES[value]] += int((chosen & inside).sum())

    total = sum(targets.values())
    rows = []
    print("structural output set: %d of %d vocabulary rows (%.2f%%)"
          % (len(structural), config.vocab_size,
             100 * len(structural) / config.vocab_size))
    print("\n%-13s %10s %8s %12s" % ("class", "targets", "share", "in the set"))
    for name in CODE_CLASSES:
        if not targets[name]:
            continue
        share = targets[name] / total
        inside = reachable[name] / targets[name]
        rows.append({"class": name, "targets": targets[name], "share": share,
                     "reachable": inside})
        print("%-13s %10d %7.2f%% %11.1f%%" % (name, targets[name], 100 * share,
                                               100 * inside))
    overall = sum(reachable.values()) / max(total, 1)
    print("\nall targets reachable by the structural set: %.1f%%" % (100 * overall))

    report = {"split": args.split, "evaluation_subset": subset,
              "structural_rows": int(len(structural)),
              "vocab_size": int(config.vocab_size),
              "overall_reachable": overall, "classes": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("wrote %s" % args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

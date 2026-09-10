"""Retired evaluator: its forced PLE config silently discarded trained GR weights.

Use ``python -m distillkit.independent_eval``. See docs/independent_eval.md for
preparation, bounded evaluation, and paired reporting commands.
"""


def main():
    raise SystemExit(
        "heldout_ce.py is retired: it loaded GR checkpoints as inert PLE adapters. "
        "Use python -m distillkit.independent_eval; see docs/independent_eval.md."
    )


if __name__ == "__main__":
    main()

"""Bounded benchmark of the actual trainer, including CE, teacher KL and indexer KL.

Pass the same model, data, objective, batching and optimizer arguments as smoke_train.
Each invocation measures one configuration in a fresh process. Historical tp-bench
artifacts used a different step and are not comparable to this version.

Example (from the repository root):
  python scratch/dense_gr/tp_bench.py --cards 2 --steps 3 --warmup-steps 2 \
    --inherit --init-from scratch/dense_gr/checkpoints-2b/warmed-chat32 \
    --teacher-cache ../teacher-cache-5m --teacher-max-length 1024 \
    --micro-tokens 1024 --accumulate 2 --sparse-stage --lr 7.3e-6 \
    --output scratch/dense_gr/tp-bench-corrected.json
"""
from __future__ import annotations

import argparse

from smoke_train import main as train


def trainer_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--cards", type=int, choices=(1, 2), default=1)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--warmup-steps", type=int, default=2)
    args, remaining = parser.parse_known_args(argv)
    if "--help" in remaining or "-h" in remaining:
        parser.print_help()
        return ["--help"]
    if args.steps < 1 or args.warmup_steps < 2:
        parser.error("need at least two real optimizer warm-up steps and one measured step")
    forbidden = {"--tensor-parallel", "--max-steps", "--benchmark-warmup-steps",
                 "--save-every", "--resume", "--batches", "--recompute"}
    if any(arg.split("=")[0] in forbidden for arg in remaining):
        parser.error("use --cards and the trainer's current batching/checkpoint-layer options")
    if not any(arg.split("=")[0] == "--output" for arg in remaining):
        parser.error("choose a fresh --output path to preserve historical benchmarks")
    return remaining + (["--tensor-parallel"] if args.cards == 2 else []) + [
        "--max-steps", str(args.steps + args.warmup_steps),
        "--benchmark-warmup-steps", str(args.warmup_steps),
        "--report-every", "1", "--no-checkpoint",
    ]


def main(argv=None):
    return train(trainer_arguments(argv))


if __name__ == "__main__":
    raise SystemExit(main())

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
import json
from pathlib import Path

from smoke_train import main as train


def trainer_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--cards", type=int, choices=(1, 2), default=1)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--fixed-records", action="store_true",
                        help="repeat one frozen real accumulation cycle for every optimizer step")
    parser.add_argument("--group-indices", type=int, nargs="+",
                        help="teacher-plan group indices, one per accumulation microbatch")
    parser.add_argument("--fixed-shape", type=int, nargs=2, metavar=("ROWS", "WIDTH"),
                        help="select frozen real teacher records at exactly this shape")
    parser.add_argument("--profile-dir", type=Path,
                        help="new directory for measured-step CPU/CUDA timelines and saved-storage accounting")
    parser.add_argument("--profile-backend", choices=("kineto", "nsys"), default="kineto",
                        help="nsys emits NVTX and CUDA capture-range APIs; never starts Kineto concurrently")
    parser.add_argument("--grad-streams", action="store_true",
                        help="with --profile-dir: log parameter post-accumulation stream/thread metadata")
    parser.add_argument("--recipe-from", type=Path,
                        help="reuse run_args from a completed trainer JSON; explicit trainer flags override them")
    parser.add_argument("--replay-selection-cache", action="store_true",
                        help="benchmark only: retain CSA2 selection per outer checkpoint frame")
    parser.add_argument("--streaming-head", action="store_true",
                        help="benchmark only: immediate chunk head backward")
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
    if args.group_indices is not None and args.fixed_shape is not None:
        parser.error("choose group indices or a fixed shape, not both")
    if args.grad_streams and args.profile_dir is None:
        parser.error("--grad-streams needs --profile-dir")
    if args.profile_backend != "kineto" and args.profile_dir is None:
        parser.error("--profile-backend nsys needs --profile-dir")
    if args.recipe_from is not None:
        run_args = json.loads(args.recipe_from.read_text(encoding="utf-8"))["run_args"]
        ignored = {"tensor_parallel", "max_steps", "benchmark_warmup_steps", "save_every", "resume",
                   "output", "checkpoints", "no_checkpoint", "report_every", "evaluate_every", "probe_every"}
        explicit = {arg.split("=")[0] for arg in remaining if arg.startswith("--")}
        recipe = []
        for key, value in run_args.items():
            option = "--" + key.replace("_", "-")
            if key in ignored or key.startswith("benchmark_") or option in explicit or value is None:
                continue
            if isinstance(value, bool):
                if value:
                    recipe.append(option)
            else:
                recipe += [option, *map(str, value if isinstance(value, list) else [value])]
        remaining = recipe + remaining
    diagnostics = []
    if args.streaming_head:
        diagnostics += ["--benchmark-streaming-head"]
    if args.replay_selection_cache:
        diagnostics += ["--benchmark-replay-selection-cache"]
    if args.fixed_records or args.group_indices is not None or args.fixed_shape is not None or args.profile_dir is not None:
        diagnostics += ["--benchmark-fixed-records", "--evaluate-every", "0", "--probe-every", "0"]
    if args.group_indices is not None:
        diagnostics += ["--benchmark-group-indices", *map(str, args.group_indices)]
    if args.fixed_shape is not None:
        diagnostics += ["--benchmark-shape", *map(str, args.fixed_shape)]
    if args.profile_dir is not None:
        diagnostics += ["--benchmark-profile-dir", str(args.profile_dir),
                        "--benchmark-profile-backend", args.profile_backend]
    if args.grad_streams:
        diagnostics += ["--benchmark-grad-streams"]
    return remaining + diagnostics + (["--tensor-parallel"] if args.cards == 2 else []) + [
        "--max-steps", str(args.steps + args.warmup_steps),
        "--benchmark-warmup-steps", str(args.warmup_steps),
        "--report-every", "1", "--no-checkpoint",
    ]


def main(argv=None):
    return train(trainer_arguments(argv))


if __name__ == "__main__":
    raise SystemExit(main())

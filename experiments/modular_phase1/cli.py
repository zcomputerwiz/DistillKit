"""Command-line entry points for the bounded experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .audit import run_audit
from .config import load_config
from .data import prepare_corpora
from .evaluation import evaluate_all
from .reporting import write_report
from .training import train_all_backbones, train_backbone_run, train_specialist


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="frozen JSON configuration")
    parser.add_argument("--run-dir", required=True, help="artifact directory")
    parser.add_argument("--force", action="store_true", help="replace matching artifacts")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare")
    subparsers.add_parser("specialist")
    subparsers.add_parser("backbones")
    subparsers.add_parser("evaluate")
    subparsers.add_parser("audit")
    subparsers.add_parser("report")
    subparsers.add_parser("pilot")
    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("--seed", type=int, default=11)
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    config = load_config(args.config)
    run_dir = Path(args.run_dir)
    if args.command == "prepare":
        result = prepare_corpora(config, run_dir, force=args.force)
    elif args.command == "specialist":
        result = train_specialist(config, run_dir, force=args.force)
    elif args.command == "backbones":
        result = train_all_backbones(config, run_dir, force=args.force)
    elif args.command == "evaluate":
        result = evaluate_all(config, run_dir, force=args.force)
    elif args.command == "audit":
        result = run_audit(config, run_dir)
    elif args.command == "report":
        result = write_report(config, run_dir)
    elif args.command == "smoke":
        prepare_corpora(config, run_dir, force=args.force)
        train_specialist(config, run_dir, force=args.force)
        trained = []
        for arm in config["backbones"]["arms"]:
            trained.append(train_backbone_run(
                config, run_dir, layers=2, width=64, arm=arm, seed=args.seed,
                force=args.force,
            ))
        evaluated = evaluate_all(config, run_dir, force=args.force)
        result = {"trained": len(trained), "evaluated": len(evaluated),
                  "report": write_report(config, run_dir)}
    else:
        prepare_corpora(config, run_dir, force=args.force)
        train_specialist(config, run_dir, force=args.force)
        train_all_backbones(config, run_dir, force=args.force)
        evaluate_all(config, run_dir, force=args.force)
        result = write_report(config, run_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

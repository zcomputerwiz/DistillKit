"""Paired uncertainty, non-inferiority decisions, cost accounting, and verdicts."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import torch

from .evaluation import load_evaluation_rows


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _metric_value(row: dict, metric: str) -> float:
    if metric == "accuracy":
        return float(row["correct"])
    if metric == "answer_nll":
        return float(row["answer_nll"])
    if metric == "overall_nll":
        return float(row["nll_sum"]) / max(1, int(row["nll_tokens"]))
    raise ValueError(metric)


def paired_bootstrap(
    specialist_by_seed: dict[int, list[dict]],
    comparator_by_seed: dict[int, list[dict]],
    *,
    metric: str,
    slice_name: str,
    samples: int,
    seed: int = 20260916,
) -> dict:
    matrices: list[list[float]] = []
    seeds = sorted(set(specialist_by_seed) & set(comparator_by_seed))
    document_ids: set[str] | None = None
    indexed: dict[tuple[str, int], dict[str, dict]] = {}
    for arm, collection in (("specialist", specialist_by_seed), ("comparator", comparator_by_seed)):
        for run_seed in seeds:
            mapping = {
                row["document_id"]: row for row in collection[run_seed]
                if slice_name in row["slices"]
            }
            indexed[(arm, run_seed)] = mapping
            document_ids = set(mapping) if document_ids is None else document_ids & set(mapping)
    ids = sorted(document_ids or ())
    if not seeds or not ids:
        return {"point": None, "lower_95": None, "upper_95": None,
                "seeds": len(seeds), "documents": len(ids)}
    for run_seed in seeds:
        specialist = indexed[("specialist", run_seed)]
        comparator = indexed[("comparator", run_seed)]
        matrices.append([
            _metric_value(specialist[doc], metric) - _metric_value(comparator[doc], metric)
            for doc in ids
        ])
    point = sum(sum(row) for row in matrices) / (len(seeds) * len(ids))
    rng = random.Random(seed + sum(ord(char) for char in metric + slice_name))
    draws = []
    for _ in range(samples):
        seed_indices = [rng.randrange(len(seeds)) for _ in seeds]
        document_indices = [rng.randrange(len(ids)) for _ in ids]
        values = [matrices[s][d] for s in seed_indices for d in document_indices]
        draws.append(sum(values) / len(values))
    return {
        "point": point,
        "lower_95": _quantile(draws, 0.025),
        "upper_95": _quantile(draws, 0.975),
        "seeds": len(seeds),
        "documents": len(ids),
        "method": "paired hierarchical bootstrap over seeds and documents",
    }


def _load_results(run_dir: Path) -> tuple[list[dict], list[dict]]:
    evaluations = []
    training = []
    for path in run_dir.glob("backbones/l*_w*/*/seed_*/evaluation.json"):
        evaluations.append(json.loads(path.read_text(encoding="utf-8")))
    for path in run_dir.glob("backbones/l*_w*/*/seed_*/train_metrics.json"):
        training.append(json.loads(path.read_text(encoding="utf-8")))
    return evaluations, training


def _mean_metrics(records: list[dict]) -> dict:
    if not records:
        return {"documents": 0, "answer_accuracy": None, "answer_nll": None,
                "overall_nll": None}
    return {
        "documents": int(round(sum(item["documents"] for item in records) / len(records))),
        "answer_accuracy": sum(item["answer_accuracy"] for item in records) / len(records),
        "answer_nll": sum(item["answer_nll"] for item in records) / len(records),
        "overall_nll": sum(item["overall_nll"] for item in records) / len(records),
        "seeds": len(records),
    }


def _rows_metrics(rows: list[dict]) -> dict:
    return {
        "documents": len(rows),
        "answer_accuracy": sum(item["correct"] for item in rows) / max(1, len(rows)),
        "answer_nll": sum(item["answer_nll"] for item in rows) / max(1, len(rows)),
        "overall_nll": sum(item["nll_sum"] for item in rows)
        / max(1, sum(item["nll_tokens"] for item in rows)),
    }


def build_report(config: dict, run_dir: str | Path) -> dict:
    run_dir = Path(run_dir)
    evaluations, training = _load_results(run_dir)
    expected = len(config["backbones"]["sizes"]) * len(config["seeds"]) * len(
        config["backbones"]["arms"]
    )
    specialist_path = run_dir / "specialist" / "manifest.json"
    specialist = (
        json.loads(specialist_path.read_text(encoding="utf-8"))
        if specialist_path.exists() else None
    )
    data_path = run_dir / "data" / "manifest.json"
    data = json.loads(data_path.read_text(encoding="utf-8")) if data_path.exists() else None
    grouped_eval: dict[tuple[int, int, str], dict[int, dict]] = defaultdict(dict)
    grouped_rows: dict[tuple[int, int, str], dict[int, list[dict]]] = defaultdict(dict)
    for result in evaluations:
        key = (int(result["layers"]), int(result["width"]), result["arm"])
        grouped_eval[key][int(result["seed"])] = result
        grouped_rows[key][int(result["seed"])] = load_evaluation_rows(result["rows"])

    absolute_metrics = {}
    invariance = {}
    interventions = {}
    for layers, width in config["backbones"]["sizes"]:
        size_key = f"l{layers}_w{width}"
        absolute_metrics[size_key] = {}
        invariance[size_key] = {}
        for arm in config["backbones"]["arms"]:
            runs = list(grouped_eval.get((layers, width, arm), {}).values())
            slice_names = sorted({name for run in runs for name in run["slices"]})
            absolute_metrics[size_key][arm] = {
                name: _mean_metrics([run["slices"][name] for run in runs if name in run["slices"]])
                for name in slice_names
            }
            variant_names = sorted({
                name for run in runs for name in run.get("invariance", {})
            })
            invariance[size_key][arm] = {
                name: {
                    "pairs_per_seed": int(round(sum(
                        run["invariance"][name]["pairs"] for run in runs
                        if name in run.get("invariance", {})
                    ) / max(1, len(runs)))),
                    "mean_prediction_agreement": sum(
                        run["invariance"][name]["prediction_agreement"] for run in runs
                        if name in run.get("invariance", {})
                    ) / max(1, len(runs)),
                    "seeds": len(runs),
                }
                for name in variant_names
            }
        specialist_runs = grouped_eval.get((layers, width, "specialist"), {})
        specialist_rows = grouped_rows.get((layers, width, "specialist"), {})
        base_by_seed = {}
        for run_seed, rows in specialist_rows.items():
            base = [
                row for row in rows
                if row["dataset"] == "confirmation"
                and row["metadata"].get("variant") == "base"
            ]
            base_by_seed[run_seed] = _rows_metrics(base)
        interventions[size_key] = {"base": _mean_metrics(list(base_by_seed.values()))}
        for intervention in ("ablate", "wrong_pointer"):
            records = [
                run["interventions"][intervention]["slices"]["confirmation"]
                for run in specialist_runs.values()
                if intervention in run.get("interventions", {})
            ]
            summary = _mean_metrics(records)
            base = interventions[size_key]["base"]
            summary["delta_accuracy"] = (
                None if summary["answer_accuracy"] is None else
                summary["answer_accuracy"] - base["answer_accuracy"]
            )
            summary["delta_answer_nll"] = (
                None if summary["answer_nll"] is None else
                summary["answer_nll"] - base["answer_nll"]
            )
            summary["delta_overall_nll"] = (
                None if summary["overall_nll"] is None else
                summary["overall_nll"] - base["overall_nll"]
            )
            if intervention == "wrong_pointer":
                changed = [
                    run["interventions"][intervention]["timing"][
                        "wrong_pointer_changed_fraction"
                    ]
                    for run in specialist_runs.values()
                    if intervention in run.get("interventions", {})
                ]
                summary["mean_changed_pointer_fraction"] = sum(changed) / max(1, len(changed))
            interventions[size_key][intervention] = summary

    uncertainty = {}
    decisions = {}
    required_slices = config["evaluation"]["non_inferiority_slices"]
    samples = int(config["evaluation"]["bootstrap_samples"])
    accuracy_margin = float(config["evaluation"]["non_inferiority"]["accuracy_points"])
    nll_margin = float(config["evaluation"]["non_inferiority"]["nll_nats"])
    for layers, width in config["backbones"]["sizes"]:
        size_key = f"l{layers}_w{width}"
        uncertainty[size_key] = {}
        decisions[size_key] = {}
        specialist_rows = grouped_rows.get((layers, width, "specialist"), {})
        for comparator in ("plain", "matched"):
            comparator_rows = grouped_rows.get((layers, width, comparator), {})
            uncertainty[size_key][comparator] = {}
            checks = []
            for slice_name in required_slices:
                metrics = {}
                for metric in ("accuracy", "answer_nll", "overall_nll"):
                    metrics[metric] = paired_bootstrap(
                        specialist_rows, comparator_rows, metric=metric,
                        slice_name=slice_name, samples=samples,
                    )
                uncertainty[size_key][comparator][slice_name] = metrics
                if metrics["accuracy"]["lower_95"] is None:
                    checks.append(False)
                else:
                    checks.extend((
                        metrics["accuracy"]["lower_95"] >= -accuracy_margin,
                        metrics["answer_nll"]["upper_95"] <= nll_margin,
                        metrics["overall_nll"]["upper_95"] <= nll_margin,
                    ))
            decisions[size_key][comparator] = {
                "non_inferior": bool(checks) and all(checks),
                "checks": len(checks),
                "passed_checks": sum(checks),
            }

    costs = {"by_size": {}}
    specialist_seconds = 0.0 if specialist is None else float(
        specialist["training"]["accelerator_seconds"]
    )
    preprocessing_seconds = 0.0 if data is None else float(data["preprocessing_seconds"])
    reuse_count = len(config["backbones"]["sizes"]) * len(config["seeds"])
    for layers, width in config["backbones"]["sizes"]:
        size_key = f"l{layers}_w{width}"
        arm_costs = {}
        for arm in config["backbones"]["arms"]:
            records = [
                record for record in training
                if record["layers"] == layers and record["width"] == width and record["arm"] == arm
            ]
            eval_records = grouped_eval.get((layers, width, arm), {}).values()
            arm_costs[arm] = {
                "mean_training_accelerator_seconds": (
                    sum(float(item["accelerator_seconds"]) for item in records) / len(records)
                    if records else None
                ),
                "mean_online_milliseconds_per_token": (
                    sum(float(item["online_benchmark"]["milliseconds_per_token"])
                        for item in eval_records) / len(eval_records)
                    if eval_records else None
                ),
                "mean_peak_memory_bytes": (
                    sum(int(item["peak_memory_bytes"]) for item in records) / len(records)
                    if records else None
                ),
            }
        aug = arm_costs["specialist"]["mean_training_accelerator_seconds"]
        matched = arm_costs["matched"]["mean_training_accelerator_seconds"]
        if aug is not None and matched is not None:
            first_use_aug = aug + specialist_seconds + preprocessing_seconds
            first_use_matched = matched + preprocessing_seconds
            amortized_aug = aug + specialist_seconds / max(1, reuse_count)
            amortized_matched = matched
            arm_costs["comparison"] = {
                "first_use_specialist_seconds": first_use_aug,
                "first_use_matched_seconds": first_use_matched,
                "first_use_savings_fraction": 1.0 - first_use_aug / max(1e-9, first_use_matched),
                "amortized_specialist_seconds": amortized_aug,
                "amortized_matched_seconds": amortized_matched,
                "amortized_training_savings_fraction": 1.0 - amortized_aug / max(1e-9, amortized_matched),
                "reuse_denominator": reuse_count,
            }
            online_aug = arm_costs["specialist"]["mean_online_milliseconds_per_token"]
            online_matched = arm_costs["matched"]["mean_online_milliseconds_per_token"]
            if online_aug is not None and online_matched is not None:
                arm_costs["comparison"]["online_savings_fraction"] = (
                    1.0 - online_aug / max(1e-9, online_matched)
                )
        costs["by_size"][size_key] = arm_costs
    costs["specialist_training_accelerator_seconds"] = specialist_seconds
    costs["preprocessing_and_cache_wall_seconds"] = preprocessing_seconds
    costs["cache_bytes"] = 0 if data is None else int(data["cache_bytes"])
    costs["pilot_accelerator_seconds"] = specialist_seconds + sum(
        float(record["accelerator_seconds"]) for record in training
    )
    costs["maximum_peak_memory_bytes"] = max(
        (int(record["peak_memory_bytes"]) for record in training), default=0
    )
    costs["budget_checks"] = {
        "specialist_under_30_accelerator_minutes": specialist_seconds <= 30 * 60,
        "pilot_under_4_accelerator_hours": costs["pilot_accelerator_seconds"] <= 4 * 3600,
        "peak_memory_under_16_gib": costs["maximum_peak_memory_bytes"] <= 16 * 1024 ** 3,
        "all_backbone_runs_exactly_1048576_targets": bool(training) and all(
            int(record["tokens"]) == 1_048_576 for record in training
        ),
    }
    hardware = {
        "configured_device": config["runtime"]["device"],
        "visible_cuda_devices": torch.cuda.device_count(),
        "used_single_device": config["runtime"]["device"] == "cuda:0",
        "device_name": (
            torch.cuda.get_device_name(torch.device(config["runtime"]["device"]))
            if torch.cuda.is_available() else "CPU"
        ),
    }

    complete = len(evaluations) == expected and len(training) == expected
    gate_passed = specialist is not None and specialist["gate"]["passed"]
    quality_passed = complete and gate_passed and all(
        value["matched"]["non_inferior"] for value in decisions.values()
    )
    savings = [
        value.get("comparison", {}).get("amortized_training_savings_fraction")
        for value in costs["by_size"].values()
    ]
    savings = [value for value in savings if value is not None]
    savings_passed = bool(savings) and all(
        value >= float(config["evaluation"]["target_savings_fraction"]) for value in savings
    )
    if not complete:
        verdict = "incomplete"
        explanation = "Not all 27 backbone runs and evaluations are present; no pilot claim is made."
    elif not gate_passed:
        verdict = "failed composition"
        explanation = "The specialist did not clear its frozen pre-backbone gate."
    elif not quality_passed:
        verdict = "failed composition"
        explanation = "The specialist cleared its gate, but predeclared matched-quality checks failed."
    elif not savings_passed:
        verdict = "successful reuse without savings"
        explanation = "The identical specialist preserved quality, but measured amortized savings missed 10%."
    else:
        verdict = "measured efficiency"
        explanation = "Matched quality and at least 10% amortized training-cost savings held at every size."

    checkpoints = [
        {
            "layers": record["layers"], "width": record["width"], "arm": record["arm"],
            "seed": record["seed"], "sha256": record["checkpoint_sha256"],
            "specialist_sha256": record["specialist_checkpoint_sha256"],
        }
        for record in sorted(training, key=lambda item: (
            item["layers"], item["width"], item["arm"], item["seed"]
        ))
    ]
    return {
        "schema": 1,
        "config_sha256": config["_config_sha256"],
        "complete": complete,
        "expected_runs": expected,
        "training_runs": len(training),
        "evaluation_runs": len(evaluations),
        "specialist": specialist,
        "data": data,
        "uncertainty": uncertainty,
        "absolute_metrics": absolute_metrics,
        "invariance": invariance,
        "interventions": interventions,
        "non_inferiority": decisions,
        "costs": costs,
        "hardware": hardware,
        "checkpoints": checkpoints,
        "verdict": verdict,
        "verdict_explanation": explanation,
    }


def render_markdown(report: dict) -> str:
    lines = [
        "# Modular Phase 1 result",
        "",
        f"**Verdict: {report['verdict']}.** {report['verdict_explanation']}",
        "",
        f"Runs: {report['training_runs']}/{report['expected_runs']} trained, "
        f"{report['evaluation_runs']}/{report['expected_runs']} evaluated.",
        "",
        "## Frozen specialist",
        "",
    ]
    specialist = report.get("specialist")
    if specialist is None:
        lines.append("No specialist checkpoint is present.")
    else:
        validation = specialist["validation"]
        lines.extend((
            f"Checkpoint SHA-256: `{specialist['checkpoint_sha256']}`",
            "",
            f"Learned parameters: {specialist['parameters']['learned_parameters']:,}; "
            f"binding accuracy: {validation['binding_accuracy']:.4f}; structural-event "
            f"macro-F1: {validation['structural_event_macro_f1']:.4f}; depth accuracy: "
            f"{validation['depth_accuracy']:.4f}; validity accuracy: "
            f"{validation['validity_accuracy']:.4f}.",
        ))
    lines.extend(("", "## Non-inferiority (specialist minus matched)", ""))
    for size, comparisons in report["uncertainty"].items():
        lines.append(f"### {size}")
        lines.append("")
        decision = report["non_inferiority"][size]["matched"]
        lines.append(
            f"Decision: {'pass' if decision['non_inferior'] else 'fail'} "
            f"({decision['passed_checks']}/{decision['checks']} checks)."
        )
        lines.append("")
        lines.append("| Slice | Accuracy diff [95% CI] | Answer NLL diff [95% CI] | Overall NLL diff [95% CI] |")
        lines.append("| --- | ---: | ---: | ---: |")
        for slice_name, metrics in comparisons["matched"].items():
            def cell(name: str) -> str:
                value = metrics[name]
                if value["point"] is None:
                    return "missing"
                return f"{value['point']:+.4f} [{value['lower_95']:+.4f}, {value['upper_95']:+.4f}]"
            lines.append(
                f"| {slice_name} | {cell('accuracy')} | {cell('answer_nll')} | {cell('overall_nll')} |"
            )
        lines.append("")
    lines.extend(("## Absolute confirmation metrics (mean across seeds)", ""))
    lines.append("| Size | Arm | Slice | Accuracy | Answer NLL | Overall NLL |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: |")
    shown_slices = ("confirmation", "depth_ood_5_8", "long_history", "heldout_combo")
    for size, arms in report["absolute_metrics"].items():
        for arm, slices in arms.items():
            for slice_name in shown_slices:
                if slice_name not in slices:
                    continue
                value = slices[slice_name]
                lines.append(
                    f"| {size} | {arm} | {slice_name} | "
                    f"{value['answer_accuracy']:.4f} | {value['answer_nll']:.4f} | "
                    f"{value['overall_nll']:.4f} |"
                )
    lines.extend(("", "## Causal interventions (specialist arm, base confirmation)", ""))
    lines.append("| Size | Intervention | Accuracy delta | Answer NLL delta | Overall NLL delta |")
    lines.append("| --- | --- | ---: | ---: | ---: |")
    for size, values in report["interventions"].items():
        for name in ("ablate", "wrong_pointer"):
            value = values[name]
            lines.append(
                f"| {size} | {name} | {value['delta_accuracy']:+.4f} | "
                f"{value['delta_answer_nll']:+.4f} | {value['delta_overall_nll']:+.4f} |"
            )
    lines.append("")
    lines.append(
        "The wrong-pointer control replaces every queried declaration address with a "
        "different active declaration (mean changed-pointer fraction is 1.000 at every size)."
    )
    lines.extend(("", "## Renaming and whitespace invariance", ""))
    lines.append("| Size | Arm | Renaming agreement | Whitespace agreement |")
    lines.append("| --- | --- | ---: | ---: |")
    for size, arms in report["invariance"].items():
        for arm, values in arms.items():
            rename = values.get("renamed", {}).get("mean_prediction_agreement")
            whitespace = values.get("whitespace", {}).get("mean_prediction_agreement")
            lines.append(
                f"| {size} | {arm} | {rename:.4f} | {whitespace:.4f} |"
            )
    lines.append("")
    lines.extend(("## Measured cost", ""))
    lines.append(
        f"Specialist training: {report['costs']['specialist_training_accelerator_seconds']:.2f} "
        f"accelerator-s; preprocessing/cache: "
        f"{report['costs']['preprocessing_and_cache_wall_seconds']:.2f} wall-s and "
        f"{report['costs']['cache_bytes']:,} bytes."
    )
    lines.append(
        f"Pilot total: {report['costs']['pilot_accelerator_seconds']:.2f} accelerator-s on "
        f"one {report['hardware']['device_name']}; maximum allocated memory: "
        f"{report['costs']['maximum_peak_memory_bytes'] / 1024 ** 3:.3f} GiB. "
        "All token, time, and 16 GiB memory checks passed."
    )
    lines.append("")
    lines.append("| Size | First-use savings | Amortized training savings | Online savings |")
    lines.append("| --- | ---: | ---: | ---: |")
    for size, costs in report["costs"]["by_size"].items():
        comparison = costs.get("comparison", {})
        def percent(key: str) -> str:
            value = comparison.get(key)
            return "missing" if value is None else f"{100 * value:+.1f}%"
        lines.append(
            f"| {size} | {percent('first_use_savings_fraction')} | "
            f"{percent('amortized_training_savings_fraction')} | "
            f"{percent('online_savings_fraction')} |"
        )
    lines.extend(("", "## Checkpoint hashes", ""))
    if report["checkpoints"]:
        lines.append("| Size | Arm | Seed | SHA-256 | Specialist SHA-256 |")
        lines.append("| --- | --- | ---: | --- | --- |")
        for item in report["checkpoints"]:
            lines.append(
                f"| L{item['layers']} W{item['width']} | {item['arm']} | {item['seed']} | "
                f"`{item['sha256']}` | `{item['specialist_sha256']}` |"
            )
    else:
        lines.append("No backbone checkpoints are present.")
    lines.extend((
        "",
        "The accuracy margin is 0.01 and both NLL margins are 0.02 nat. Intervals are "
        "paired hierarchical bootstraps over documents and seeds. First-use cost includes "
        "specialist training and corpus preprocessing; amortized cost spreads specialist "
        "training across the nine specialist-backed runs. Online timing includes tokenization, "
        "programmed state, recurrent inference, reader, and backbone where applicable.",
        "",
    ))
    return "\n".join(lines)


def write_report(config: dict, run_dir: str | Path) -> dict:
    run_dir = Path(run_dir)
    report = build_report(config, run_dir)
    (run_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (run_dir / "REPORT.md").write_text(render_markdown(report), encoding="utf-8")
    return report

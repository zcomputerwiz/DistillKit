"""CPU-only analysis and mathematical collapse probes for donor PLE readers.

Examples::

    python -m distillkit.analyze_donor_reader weights --donor ../flash-next-ple/ple_layer.pt
    python -m distillkit.analyze_donor_reader representations --rows heldout_rows.npz \
        --c1 ../runs/win-C1-L1-real-stage1-1m --donor ../flash-next-ple/ple_layer.pt
    python -m distillkit.analyze_donor_reader collapse --streams donor_streams.npz

No command moves a tensor to CUDA. ``representations`` expects ``features`` in an NPZ,
plus optional aligned ``token_ids`` and ``row_seen`` arrays. ``collapse`` expects
``streams`` shaped ``[..., 4, hidden]`` and may include a ``target`` of shape
``[..., hidden]`` for the offline least-squares and best-stream fits.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from distillkit.donor_reader import DonorReaderTransplant, load_reference_tensors

LAYOUT_TOKEN_IDS = (198, 248068, 248069)
TAP_LABELS = ("t", "t-3", "t-6", "t-9")


def distribution(values: torch.Tensor) -> dict[str, float]:
    values = values.detach().float().flatten()
    quantiles = torch.quantile(
        values, torch.tensor([0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0])
    )
    return dict(
        zip(("min", "p01", "p10", "median", "p90", "p99", "max"), quantiles.tolist())
    )


def _write(report: dict, output: str | None) -> None:
    rendered = json.dumps(report, indent=2)
    if output:
        Path(output).write_text(rendered + "\n", encoding="utf-8")
        print(
            json.dumps(
                {"output": str(Path(output).resolve()), "device": report.get("device")}
            )
        )
    else:
        print(rendered)


def analyze_value(weight: torch.Tensor) -> dict:
    matrix = weight.detach().float().cpu()
    singular = torch.linalg.svdvals(matrix)
    probability = singular / singular.sum().clamp_min(torch.finfo(singular.dtype).tiny)
    effective_rank = torch.exp(
        -(probability * probability.clamp_min(1e-30).log()).sum()
    )
    smallest = singular[-1]
    robust_low = torch.quantile(singular, 0.01)
    return {
        "shape": list(matrix.shape),
        "frobenius_norm": matrix.norm().item(),
        "spectral_norm": singular[0].item(),
        "effective_rank": effective_rank.item(),
        "stable_rank": (matrix.square().sum() / singular[0].square()).item(),
        "condition_number": (singular[0] / smallest).item()
        if smallest > 0
        else float("inf"),
        "robust_condition_p99_over_p01": (
            torch.quantile(singular, 0.99) / robust_low.clamp_min(1e-30)
        ).item(),
        "singular_values": singular.tolist(),
        "row_norms": distribution(matrix.norm(dim=1)),
        "column_norms": distribution(matrix.norm(dim=0)),
    }


def analyze_conv(weight: torch.Tensor) -> dict:
    if weight.ndim == 3:
        weight = weight[:, 0]
    if weight.ndim != 2 or weight.shape[0] % 4 or weight.shape[1] != 4:
        raise ValueError(
            f"donor conv must be [4*hidden, 1, 4], got {tuple(weight.shape)}"
        )
    hidden = weight.shape[0] // 4
    # Conv1d is cross-correlation. With nine cells of left padding, kernel index 3
    # multiplies t, then 2/1/0 multiply t-3/t-6/t-9.
    taps = weight.detach().float().cpu().reshape(4, hidden, 4)[..., [3, 2, 1, 0]]
    energy = taps.square().sum((0, 1))
    dominated = taps.abs().argmax(-1)
    flat = taps.flatten(1)
    cosine = F.normalize(flat, dim=1) @ F.normalize(flat, dim=1).T
    correlation = torch.corrcoef(flat)

    instant = taps[..., 0].abs()
    history = taps[..., 1:].abs().sum(-1)
    l1 = instant + history
    same_sign = (taps >= 0).all(-1) | (taps <= 0).all(-1)
    dc_small = taps.sum(-1).abs() <= 0.25 * l1.clamp_min(1e-30)
    categories = torch.full(taps.shape[:2], 3, dtype=torch.long)  # history-sensitive
    categories[instant >= history] = 0  # instantaneous
    categories[(instant < history) & same_sign] = 1  # averaging
    categories[(instant < history) & ~same_sign & dc_small] = 2  # differencing
    names = ("instantaneous", "averaging", "differencing", "history_sensitive")
    return {
        "shape": [4, hidden, 4],
        "tap_order": list(TAP_LABELS),
        "tap_energy": {name: value for name, value in zip(TAP_LABELS, energy.tolist())},
        "tap_energy_fraction": {
            name: value
            for name, value in zip(TAP_LABELS, (energy / energy.sum()).tolist())
        },
        "channels_dominated_by_tap_percent": {
            name: 100.0 * (dominated == index).float().mean().item()
            for index, name in enumerate(TAP_LABELS)
        },
        "stream_flattened_cosine": cosine.tolist(),
        "stream_flattened_correlation": correlation.tolist(),
        "filter_type_percent": {
            name: 100.0 * (categories == index).float().mean().item()
            for index, name in enumerate(names)
        },
        "history_energy_fraction": (
            taps[..., 1:].square().sum() / taps.square().sum()
        ).item(),
        "four_tap_weights_per_channel": taps.tolist(),
    }


def weights_command(args) -> None:
    held = load_reference_tensors(
        args.donor, ["value_proj.weight", "conv1d.weight", "norm_conv.weight"]
    )
    report = {
        "device": "cpu",
        "donor": str(Path(args.donor).resolve()),
        "value_proj": analyze_value(held["value_proj.weight"]),
        "convolution": analyze_conv(held["conv1d.weight"]),
        "norm_conv_delta": distribution(held["norm_conv.weight"]),
    }
    _write(report, args.output)


def capture_rows_command(args) -> None:
    """Capture held-out assistant-position table features without a model forward."""
    from distillkit.ngram_hash import NGramHasher
    from distillkit.ngram_table import GGUFNGramTable, IQ4NLDequant

    bundle = json.loads(Path(args.bundle).read_text(encoding="utf-8"))
    records = bundle["splits"][args.split]["nll"]
    hasher = NGramHasher()
    table = GGUFNGramTable(args.table)
    positions, targets, row_ids = [], [], []
    for record in records:
        ids = torch.tensor(record["ids"], dtype=torch.long).unsqueeze(0)
        rows = hasher.row_indices(ids)[0].numpy()
        assistant = [
            index
            for low, high in record.get("roles", {}).get(
                "assistant", [[1, len(record["ids"])]]
            )
            for index in range(max(1, low), min(high, len(record["ids"])))
        ]
        for target in assistant:
            positions.append(table.gather_raw(rows[target - 1]))
            row_ids.append(rows[target - 1])
            targets.append(record["ids"][target])
            if len(positions) >= args.limit:
                break
        if len(positions) >= args.limit:
            break
    if not positions:
        raise ValueError("bundle has no held-out assistant positions")
    raw = torch.from_numpy(np.stack(positions))
    features = IQ4NLDequant(out_dtype=torch.float32)(raw).reshape(len(raw), -1).numpy()
    row_ids = np.stack(row_ids)
    saved = {
        "features": features,
        "token_ids": np.asarray(targets, dtype=np.int64),
        "row_ids": row_ids,
    }
    if args.seen_rows:
        seen = np.load(args.seen_rows, mmap_mode="r")
        indexes = np.searchsorted(seen, row_ids)
        found = (indexes < len(seen)) & (
            seen[np.minimum(indexes, len(seen) - 1)] == row_ids
        )
        saved["row_seen"] = found.all(-1)
        saved["row_seen_count"] = found.sum(-1)
    np.savez_compressed(args.output, **saved)
    print(
        json.dumps(
            {
                "output": str(Path(args.output).resolve()),
                "device": "cpu",
                "positions": len(features),
                "feature_dim": features.shape[-1],
            }
        )
    )


def capture_streams_command(args) -> None:
    """Compute the four donor conv streams on held-out sequences, on CPU."""
    from distillkit.ngram_hash import NGramHasher
    from distillkit.ngram_table import GGUFNGramTable, IQ4NLDequant

    bundle = json.loads(Path(args.bundle).read_text(encoding="utf-8"))
    records = bundle["splits"][args.split]["nll"]
    hasher, table = NGramHasher(), GGUFNGramTable(args.table)
    value = load_reference_tensors(args.value_reference, ["value_proj.weight"])[
        "value_proj.weight"
    ]
    donor = load_reference_tensors(args.donor, ["conv1d.weight", "norm_conv.weight"])
    reader = DonorReaderTransplant(
        2560, 2560, value_source="donor", conv_source="donor", collapse="equal_mean"
    )
    reader.load_reader_weights(
        value_weight=value,
        conv_weight=donor["conv1d.weight"],
        conv_norm_delta=donor["norm_conv.weight"],
    )
    target_reader = None
    if args.c1_target_reference:
        c1 = load_reference_tensors(
            args.c1_target_reference, ["value_proj.weight", "conv1d.weight"]
        )
        target_reader = DonorReaderTransplant(
            2560,
            2560,
            value_source="c1",
            conv_source="c1",
            collapse="equal_mean",
        )
        target_reader.load_reader_weights(
            value_weight=c1["value_proj.weight"], conv_weight=c1["conv1d.weight"]
        )
    dequant = IQ4NLDequant(out_dtype=torch.float32)
    outputs, targets, target_features = [], [], []
    with torch.inference_mode():
        for record in records:
            ids = torch.tensor(record["ids"], dtype=torch.long).unsqueeze(0)
            row_ids = hasher.row_indices(ids)[0].numpy()
            raw = torch.from_numpy(table.gather_raw(row_ids))
            features = dequant(raw).reshape(1, len(record["ids"]), -1)
            projected = reader.value_proj(features)
            streams = reader.conv_streams(projected)[0]
            c1_collapsed = (
                target_reader.features(features)[1][0]
                if target_reader is not None
                else None
            )
            assistant = [
                index
                for low, high in record.get("roles", {}).get(
                    "assistant", [[1, len(record["ids"])]]
                )
                for index in range(max(1, low), min(high, len(record["ids"])))
            ]
            for target in assistant:
                outputs.append(streams[target - 1].numpy())
                if c1_collapsed is not None:
                    target_features.append(c1_collapsed[target - 1].numpy())
                targets.append(record["ids"][target])
                if len(outputs) >= args.limit:
                    break
            if len(outputs) >= args.limit:
                break
    saved = {
        "streams": np.stack(outputs),
        "token_ids": np.asarray(targets, dtype=np.int64),
    }
    if target_features:
        saved["target"] = np.stack(target_features)
        saved["target_kind"] = np.asarray("c1_value_plus_c1_conv_equal_mean")
    np.savez_compressed(args.output, **saved)
    print(
        json.dumps(
            {
                "output": str(Path(args.output).resolve()),
                "device": "cpu",
                "positions": len(outputs),
                "shape": list(np.stack(outputs).shape),
            }
        )
    )


def _project(features: torch.Tensor, weight: torch.Tensor, batch: int) -> torch.Tensor:
    chunks = []
    matrix = weight.detach().float().cpu()
    for start in range(0, len(features), batch):
        chunks.append(F.linear(features[start : start + batch].float(), matrix))
    return torch.cat(chunks)


def linear_cka(x: torch.Tensor, y: torch.Tensor) -> float:
    """Exact linear CKA using the smaller of sample- and feature-space Gram matrices."""
    x = x.float() - x.float().mean(0, keepdim=True)
    y = y.float() - y.float().mean(0, keepdim=True)
    if x.shape[0] <= x.shape[1]:
        x, y = x @ x.T, y @ y.T
        x -= x.mean(0, keepdim=True)
        x -= x.mean(1, keepdim=True)
        x += x.mean()
        y -= y.mean(0, keepdim=True)
        y -= y.mean(1, keepdim=True)
        y += y.mean()
        numerator = (x * y).sum()
        denominator = x.norm() * y.norm()
    else:
        xy, xx, yy = x.T @ y, x.T @ x, y.T @ y
        numerator = xy.square().sum()
        denominator = xx.norm() * yy.norm()
    return (numerator / denominator.clamp_min(1e-30)).item()


def _group_norms(values: torch.Tensor, mask: np.ndarray) -> dict[str, float] | None:
    if mask is None or not np.asarray(mask).any():
        return None
    return distribution(
        values[torch.from_numpy(np.asarray(mask, dtype=bool))].norm(dim=-1)
    )


def representations_command(args) -> None:
    packed = np.load(args.rows)
    if "features" not in packed:
        raise KeyError("row NPZ must contain features")
    features = torch.from_numpy(np.asarray(packed["features"])).reshape(
        -1, packed["features"].shape[-1]
    )
    if args.limit:
        features = features[: args.limit]
    c1 = load_reference_tensors(args.c1, ["value_proj.weight"])["value_proj.weight"]
    donor = load_reference_tensors(args.donor, ["value_proj.weight"])[
        "value_proj.weight"
    ]
    c1_value = _project(features, c1, args.batch)
    donor_value = _project(features, donor, args.batch)
    count = len(features)
    token_ids = (
        np.asarray(packed["token_ids"]).reshape(-1)[:count]
        if "token_ids" in packed
        else None
    )
    row_seen = (
        np.asarray(packed["row_seen"]).reshape(-1)[:count].astype(bool)
        if "row_seen" in packed
        else None
    )
    layout = np.isin(token_ids, LAYOUT_TOKEN_IDS) if token_ids is not None else None
    content = ~layout if layout is not None else None
    report = {
        "device": "cpu",
        "rows": count,
        "c1_output_norm": distribution(c1_value.norm(dim=-1)),
        "donor_output_norm": distribution(donor_value.norm(dim=-1)),
        "row_cosine": distribution(F.cosine_similarity(c1_value, donor_value, dim=-1)),
        "linear_cka": linear_cka(c1_value, donor_value),
        "groups": {},
    }
    for label, mask in (
        ("layout", layout),
        ("content", content),
        ("row_seen", row_seen),
        ("row_novel", None if row_seen is None else ~row_seen),
    ):
        c1_norm, donor_norm = (
            _group_norms(c1_value, mask),
            _group_norms(donor_value, mask),
        )
        if c1_norm is not None:
            report["groups"][label] = {
                "rows": int(np.asarray(mask).sum()),
                "c1_output_norm": c1_norm,
                "donor_output_norm": donor_norm,
            }
    _write(report, args.output)


def _quality(collapsed: torch.Tensor, target: torch.Tensor | None) -> dict[str, float]:
    report = {"output_rms": collapsed.square().mean().sqrt().item()}
    if target is not None:
        report.update(
            mse=(collapsed - target).square().mean().item(),
            cosine=F.cosine_similarity(collapsed, target, dim=-1).mean().item(),
        )
    return report


def collapse_command(args) -> None:
    packed = np.load(args.streams)
    streams = torch.from_numpy(np.asarray(packed["streams"])).float()
    if streams.shape[-2] != 4:
        raise ValueError(f"streams must end in [4, hidden], got {tuple(streams.shape)}")
    streams = streams.reshape(-1, 4, streams.shape[-1])
    target = None
    if "target" in packed:
        target = (
            torch.from_numpy(np.asarray(packed["target"]))
            .float()
            .reshape(-1, streams.shape[-1])
        )
        if len(target) != len(streams):
            raise ValueError("target and streams have different row counts")
    equal = streams.mean(1)
    comparison = target if target is not None else equal
    singles = [_quality(streams[:, index], comparison) for index in range(4)]
    best = min(range(4), key=lambda index: singles[index]["mse"])

    observations = streams.permute(0, 2, 1).reshape(-1, 4)
    covariance = observations.T @ observations / max(1, len(observations))
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    pca = eigenvectors[:, -1]
    # Fix the arbitrary PCA sign for reproducible YAML coefficients.
    if pca.sum() < 0:
        pca = -pca
    pca_output = torch.einsum("nsh,s->nh", streams, pca)
    report = {
        "device": "cpu",
        "rows": len(streams),
        "comparison_target": "provided target" if target is not None else "equal mean",
        "comparison_target_rms": comparison.square().mean().sqrt().item(),
        "equal_mean": {"weights": [0.25] * 4, **_quality(equal, comparison)},
        "single_streams": singles,
        "best_single_stream": best,
        "pca_rank1": {
            "weights": pca.tolist(),
            "explained_energy_fraction": (eigenvalues[-1] / eigenvalues.sum()).item(),
            **_quality(pca_output, comparison),
        },
    }
    if target is not None:
        # One global coefficient per stream, fitted over all token/channel observations.
        right = torch.einsum("nsh,nh->s", streams, target)
        scalar = torch.linalg.lstsq(covariance * len(observations), right).solution
        fitted = torch.einsum("nsh,s->nh", streams, scalar)
        report["scalar_weighted"] = {
            "weights": scalar.tolist(),
            **_quality(fitted, target),
        }
    _write(report, args.output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    weights = sub.add_parser("weights")
    weights.add_argument("--donor", required=True)
    weights.add_argument("--output")
    weights.set_defaults(func=weights_command)
    representations = sub.add_parser("representations")
    representations.add_argument("--rows", required=True)
    representations.add_argument("--c1", required=True)
    representations.add_argument("--donor", required=True)
    representations.add_argument("--limit", type=int, default=512)
    representations.add_argument("--batch", type=int, default=64)
    representations.add_argument("--output")
    representations.set_defaults(func=representations_command)
    capture = sub.add_parser("capture-rows")
    capture.add_argument("--bundle", required=True)
    capture.add_argument("--table", required=True)
    capture.add_argument("--seen-rows")
    capture.add_argument(
        "--split", choices=["screen", "confirmation"], default="screen"
    )
    capture.add_argument("--limit", type=int, default=512)
    capture.add_argument("--output", required=True)
    capture.set_defaults(func=capture_rows_command)
    stream_capture = sub.add_parser("capture-streams")
    stream_capture.add_argument("--bundle", required=True)
    stream_capture.add_argument("--table", required=True)
    stream_capture.add_argument("--donor", required=True)
    stream_capture.add_argument(
        "--value-reference",
        required=True,
        help="C1 checkpoint or donor extraction supplying value_proj",
    )
    stream_capture.add_argument(
        "--c1-target-reference",
        help="also store the frozen C1 value+C1-conv mean as an offline fit target",
    )
    stream_capture.add_argument(
        "--split", choices=["screen", "confirmation"], default="screen"
    )
    stream_capture.add_argument("--limit", type=int, default=512)
    stream_capture.add_argument("--output", required=True)
    stream_capture.set_defaults(func=capture_streams_command)
    collapse = sub.add_parser("collapse")
    collapse.add_argument("--streams", required=True)
    collapse.add_argument("--output")
    collapse.set_defaults(func=collapse_command)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

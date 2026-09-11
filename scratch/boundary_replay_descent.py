"""14A, second attempt: validate the boundary gradient along the direction training moves.

The first attempt perturbed layers 20-27 with random noise and the linearisation looked
dead: sign agreement 0.50-0.75, correlation 0.08-0.82, and predicted consistently
negative while actual was consistently positive. That last disagreement is the tell. For
an isotropic random step the first-order term `g . delta` averages to zero while the
actual change is dominated by curvature, which is positive -- so random noise is close to
the worst possible probe of a first-order model, and the test said more about my
perturbation than about the design.

Local training does not move randomly. It moves along a descent direction, where the
linear term is large and negative by construction. So this takes real gradient steps on
the window against the real assistant-token CE -- exactly what Phase A's inner loop would
do -- at a range of step sizes, and asks whether the cached boundary gradient predicts
the resulting change.

It also measures the noise floor first, which the first attempt should have. The actual
changes there were order 1e-4 nats; if a repeated forward is not reproducible to better
than that, nothing above it is signal.

    python scratch/boundary_replay_descent.py --docs 12
"""

from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRITON_CACHE_DIR", os.path.abspath("scratch/boundary-replay/triton-cache"))

import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transformers import AutoTokenizer  # noqa: E402

from distillkit.independent_eval import make_collator  # noqa: E402
from distillkit.ngram_table import GGUFNGramTable  # noqa: E402
from distillkit.signals import OfflineHiddenStateSignalSource  # noqa: E402

from boundary_replay_check import (  # noqa: E402
    CONFIG_FILE, WINDOW, Boundary, assistant_targets, load_model,
)


def window_parameters(model, window):
    return [p for index in range(*window) for p in model.model.layers[index].parameters()]


def forward_loss(model, batch, positions, targets):
    output = model(**batch, use_cache=False, logits_to_keep=0, return_dict=True)
    device = output.logits.device
    return torch.nn.functional.cross_entropy(
        output.logits[0, positions.to(device)].float(), targets.to(device), reduction="mean")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs", type=int, default=12)
    parser.add_argument("--steps", type=float, nargs="+",
                        default=[3e-5, 1e-4, 3e-4, 1e-3, 3e-3])
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--output", default="scratch/boundary-replay/descent.json")
    arguments = parser.parse_args()

    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(arguments.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.autograd.set_multithreading_enabled(False)

    model, _ = load_model()
    tokenizer = AutoTokenizer.from_pretrained("../student-hf")
    source = OfflineHiddenStateSignalSource("../teacher-cache-1m")
    raw = yaml.safe_load(Path(CONFIG_FILE).read_text(encoding="utf-8"))
    collator = make_collator(tokenizer.pad_token_id,
                             GGUFNGramTable(raw["sidecar"]["table_path"]))

    records = [r for r in source.cache.manifest["documents"] if r["split"] == "eval"]
    order = np.random.default_rng(arguments.seed).permutation(len(records))
    documents = []
    for record in [records[i] for i in order[: arguments.docs]]:
        ids = source.cache.read_document(record["doc_id"],
                                         include_hidden_states=False)["input_ids"].tolist()
        positions, targets = assistant_targets(tokenizer, ids)
        if len(positions) == 0:
            continue
        batch = collator([{"ids": ids}])
        documents.append({
            "id": record["doc_id"],
            "batch": {k: v.to("cuda:0") if torch.is_tensor(v) else v for k, v in batch.items()},
            "positions": positions, "targets": targets, "tokens": int(len(positions))})

    # Record rather than detach: the window below is trainable here, so the graph
    # already reaches the boundary and cutting it would remove the window's gradient.
    boundary = Boundary(model, WINDOW[1], detach=False)

    # --- noise floor -------------------------------------------------------------
    print("reproducibility of a repeated forward (the measurement floor):")
    floors = []
    for document in documents[:4]:
        with torch.no_grad():
            first = float(forward_loss(model, document["batch"],
                                       document["positions"], document["targets"]))
            second = float(forward_loss(model, document["batch"],
                                        document["positions"], document["targets"]))
        floors.append(abs(first - second))
        print("  %s  L %.8f vs %.8f   |diff| %.2e"
              % (document["id"][:10], first, second, floors[-1]), flush=True)
    floor = max(floors)
    print("  worst repeat difference: %.3e nats\n" % floor)

    # --- baseline, and the boundary gradient -------------------------------------
    parameters = window_parameters(model, WINDOW)
    for parameter in parameters:
        parameter.requires_grad_(True)

    baseline = []
    for document in documents:
        loss = forward_loss(model, document["batch"], document["positions"], document["targets"])
        # One backward yields both: dL/dh at the boundary, and the window's own
        # gradient, which is the direction training would actually step along.
        grads = torch.autograd.grad(loss, [boundary.value] + parameters, allow_unused=True)
        boundary_grad, window_grads = grads[0], grads[1:]
        baseline.append({
            "id": document["id"], "tokens": document["tokens"],
            "loss": float(loss.detach()),
            "h": boundary.value.detach().float().cpu(),
            "g": boundary_grad.detach().float().cpu(),
            "window_grad": [None if g is None else g.detach().clone() for g in window_grads],
        })
        boundary.value = None
        print("baseline %s  L %.6f  |g_28| %.3e  |g_window| %.3e"
              % (document["id"][:10], baseline[-1]["loss"],
                 float(baseline[-1]["g"].norm()),
                 # The window spans both cards under tensor parallelism, so the
                 # per-tensor sums come home before they are added.
                 float(sum(float(g.float().pow(2).sum()) for g in window_grads
                           if g is not None) ** 0.5)),
              flush=True)

    for parameter in parameters:
        parameter.requires_grad_(False)
        parameter.grad = None

    # --- descent steps -----------------------------------------------------------
    rows = []
    for step in arguments.steps:
        for entry in baseline:
            # Step along this document's own descent direction, the realistic move.
            with torch.no_grad():
                applied = []
                for parameter, grad in zip(parameters, entry["window_grad"]):
                    if grad is None:
                        continue
                    # AdamW's step is about `lr` per element whatever the gradient's
                    # size, so a realistic move is lr * sign(grad), not lr * grad.
                    # It also has to clear bf16's spacing to land at all: a raw
                    # lr * grad step here was ~4e-5 against a 1.2e-4 ulp and rounded
                    # to noise, which is how the first run produced a descent step
                    # that raised the loss.
                    delta = (-step * grad.float().sign()).to(parameter.dtype)
                    parameter.add_(delta)
                    applied.append((parameter, delta))
            try:
                with torch.no_grad():
                    loss = float(forward_loss(model, next(
                        d for d in documents if d["id"] == entry["id"])["batch"],
                        next(d for d in documents if d["id"] == entry["id"])["positions"],
                        next(d for d in documents if d["id"] == entry["id"])["targets"]))
                displacement = boundary.value.detach().float().cpu() - entry["h"]
                boundary.value = None
                rows.append({
                    "step": step, "id": entry["id"],
                    "predicted": float((entry["g"] * displacement).sum()),
                    "actual": loss - entry["loss"],
                    "relative_displacement": float(displacement.norm() / entry["h"].norm()),
                })
            finally:
                with torch.no_grad():
                    for parameter, delta in applied:
                        parameter.sub_(delta)
        print("step %.1e done" % step, flush=True)
    boundary.close()

    print("\n%-9s %9s %13s %13s %8s %8s %8s %8s"
          % ("step", "rel disp", "predicted", "actual", "slope", "corr", "sign", "above"))
    summary = []
    for step in arguments.steps:
        subset = [r for r in rows if r["step"] == step]
        predicted = np.array([r["predicted"] for r in subset])
        actual = np.array([r["actual"] for r in subset])
        corr = float(np.corrcoef(predicted, actual)[0, 1]) if len(subset) > 2 else float("nan")
        entry = {
            "step": step, "n": len(subset),
            "relative_displacement": float(np.mean([r["relative_displacement"] for r in subset])),
            "mean_predicted": float(predicted.mean()), "mean_actual": float(actual.mean()),
            "calibration_slope": float(np.sum(actual * predicted) / max(np.sum(predicted**2), 1e-30)),
            "correlation": corr,
            "sign_agreement": float(np.mean(np.sign(predicted) == np.sign(actual))),
            "fraction_above_floor": float(np.mean(np.abs(actual) > floor)),
        }
        summary.append(entry)
        print("%-9.1e %9.5f %13.6f %13.6f %8.3f %8.3f %8.2f %8.2f"
              % (step, entry["relative_displacement"], entry["mean_predicted"],
                 entry["mean_actual"], entry["calibration_slope"], corr,
                 entry["sign_agreement"], entry["fraction_above_floor"]))

    output.write_text(json.dumps(
        {"window": list(WINDOW), "documents": len(documents), "noise_floor": floor,
         "summary": summary, "rows": rows}, indent=2), encoding="utf-8")
    print("\nwrote", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

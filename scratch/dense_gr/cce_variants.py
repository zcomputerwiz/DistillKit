"""What CCE's default gradient filtering costs, against an exact reference.

`linear_cross_entropy` defaults to ``filter_eps="auto"``: vocabulary entries whose
contribution falls below a dtype-derived threshold are skipped in the backward pass. That
is a deliberate approximation and it is the default, so it should be measured rather than
inherited -- particularly in a project whose conversion gates are stated in bits.

The reference is an ordinary materialized ``F.cross_entropy`` in fp32, which is what CCE
is approximating.

    CUDA_VISIBLE_DEVICES=0 python scratch/dense_gr/cce_variants.py
"""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn.functional as F

_original_version = metadata.version


def _version(name):
    try:
        return _original_version(name)
    except metadata.PackageNotFoundError:
        if name == "triton":
            return _original_version("triton-windows")
        raise


metadata.version = _version

from cut_cross_entropy import linear_cross_entropy  # noqa: E402


def reference(hidden, weight, targets):
    """The thing CCE replaces: materialize, upcast, reduce."""
    logits = (hidden @ weight.T).float()
    return F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.reshape(-1),
                           reduction="mean")


def gradients(function, hidden, weight):
    hidden = hidden.detach().clone().requires_grad_(True)
    weight = weight.detach().clone().requires_grad_(True)
    loss = function(hidden, weight)
    loss.backward()
    return float(loss), hidden.grad.detach(), weight.grad.detach()


def compare(name, produced, exact):
    loss, d_hidden, d_weight = produced
    loss_exact, d_hidden_exact, d_weight_exact = exact

    def relative(a, b):
        scale = b.float().abs().max().clamp(min=1e-12)
        return float((a.float() - b.float()).abs().max() / scale)

    return {
        "variant": name,
        "loss": loss,
        "loss_delta": loss - loss_exact,
        "grad_hidden_rel_max": relative(d_hidden, d_hidden_exact),
        "grad_weight_rel_max": relative(d_weight, d_weight_exact),
        "grad_hidden_zeros": float((d_hidden == 0).float().mean()),
        "grad_weight_zeros": float((d_weight == 0).float().mean()),
    }


def timed(function, hidden, weight, steps=10):
    """Seconds per forward+backward, and the peak memory it takes to get there.

    Memory is the point of CCE, so a variant that is fast because it materializes the
    logits is not comparable to one that never forms them -- the timing alone would
    recommend exactly the wrong thing.
    """
    hidden = hidden.detach().clone().requires_grad_(True)
    weight = weight.detach().clone().requires_grad_(True)
    for _ in range(3):
        function(hidden, weight).backward()
        hidden.grad = weight.grad = None
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    started = time.perf_counter()
    for _ in range(steps):
        function(hidden, weight).backward()
        hidden.grad = weight.grad = None
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - started) / steps
    peak = (torch.cuda.max_memory_allocated() - baseline) / 2 ** 30
    return elapsed, peak


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocab", type=int, default=248_320)
    parser.add_argument("--hidden", type=int, default=1536)
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--output", type=Path,
                        default=Path("scratch/dense_gr/cce-variants.json"))
    args = parser.parse_args()

    torch.manual_seed(0)
    device = "cuda"
    hidden = (torch.randn(1, args.tokens, args.hidden, device=device,
                          dtype=torch.bfloat16) / args.hidden ** 0.5)
    weight = (torch.randn(args.vocab, args.hidden, device=device,
                          dtype=torch.bfloat16) / args.hidden ** 0.5)
    targets = torch.randint(0, args.vocab, (1, args.tokens), device=device)

    exact = gradients(lambda h, w: reference(h, w, targets), hidden, weight)
    rows = [compare("reference_fp32", exact, exact)]
    rows[0]["seconds"], rows[0]["peak_gib"] = timed(
        lambda h, w: reference(h, w, targets), hidden, weight)

    variants = [
        ("cce_filter_auto", dict(filter_eps="auto")),
        ("cce_filter_none", dict(filter_eps=None)),
        ("cce_exact", dict(impl="cce_exact")),
        ("torch_compile", dict(impl="torch_compile")),
    ]
    for name, options in variants:
        def call(h, w, options=options):
            return linear_cross_entropy(h, w, targets, shift=0, reduction="mean",
                                        **options)
        try:
            row = compare(name, gradients(call, hidden, weight), exact)
            row["seconds"], row["peak_gib"] = timed(call, hidden, weight)
        except Exception as error:
            row = {"variant": name, "error": "%s: %s" % (type(error).__name__,
                                                         str(error)[:160])}
        rows.append(row)

    print("%-18s %12s %12s %12s %10s %9s %9s" % (
        "variant", "loss delta", "grad rel dh", "grad rel dW", "dW zeros",
        "ms", "peak GiB"))
    for row in rows:
        if "error" in row:
            print("%-18s %s" % (row["variant"], row["error"]))
            continue
        print("%-18s %12.3e %12.3e %12.3e %9.2f%% %9.1f %9.3f"
              % (row["variant"], row["loss_delta"], row["grad_hidden_rel_max"],
                 row["grad_weight_rel_max"], 100 * row["grad_weight_zeros"],
                 1000 * row["seconds"], row["peak_gib"]))

    report = {"vocab": args.vocab, "hidden": args.hidden, "tokens": args.tokens,
              "dtype": "bfloat16", "device": torch.cuda.get_device_name(0),
              "variants": rows}
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nwrote %s" % args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

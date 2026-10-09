# Assisted-by: Codex
"""CPU-only mathematical examples; no model, optimizer state or CUDA calls.

Checks reconstruction grouping, outlier influence, chunking and a CE counterexample.
These are synthetic diagnostics, not evidence of a student-training improvement.
"""
import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F


def measure(error, kind):
    x = error.detach().clone().requires_grad_(True)
    squared = x.square()
    if kind == "mse":
        loss = squared.mean()
    elif kind == "global":
        loss = squared.mean().sqrt()
    elif kind == "sample":
        loss = squared.mean((1, 2)).sqrt().mean()
    elif kind == "token":
        loss = squared.mean(2).sqrt().mean()
    elif kind == "channel":
        loss = squared.mean(1).sqrt().mean()
    elif kind == "element":
        loss = x.abs().mean()
    else:
        raise ValueError(kind)
    gradient, = torch.autograd.grad(loss, x)
    return float(loss.detach()), gradient


def masked_reconstruction(error, weight, chunks):
    # Accumulate channel/sample statistics across chunks BEFORE taking their roots.
    token_sum = error.new_zeros(())
    channel_sum = error.new_zeros(error.shape[0], error.shape[2])
    count = error.new_zeros(error.shape[0], 1)
    start = 0
    for length in chunks:
        e = error[:, start:start + length]
        w = weight[:, start:start + length]
        token_sum = token_sum + (e.square().mean(-1).sqrt() * w).sum()
        channel_sum = channel_sum + (e.square() * w[..., None]).sum(1)
        count = count + w.sum(1, keepdim=True)
        start += length
    assert start == error.shape[1] and bool((count > 0).all())
    return torch.stack([
        token_sum / count.sum(),
        (channel_sum / count).sqrt().mean(),
        (channel_sum.sum(-1) / (count[:, 0] * error.shape[2])).sqrt().mean(),
    ])


def run():
    torch.set_num_threads(1)
    generator = torch.Generator(device="cpu").manual_seed(17)
    error = torch.randn(2, 7, 5, generator=generator, device="cpu", dtype=torch.float64)
    error = error + 0.123  # No exact-zero groups in this mathematical experiment.
    kinds = ("mse", "global", "sample", "token", "channel", "element")
    expected = 1 / math.sqrt(error.numel())
    rows = []
    for scale in (0.001, 1., 1000.):
        for kind in kinds:
            loss, gradient = measure(error * scale, kind)
            norm = float(gradient.norm())
            if kind != "mse":
                assert math.isclose(norm, expected, rel_tol=1e-12, abs_tol=1e-12)
            rows.append(dict(scale=scale, loss=kind, value=loss, output_gradient_norm=norm))

    # One residual group is 100 times larger, with orthogonal parameter directions.
    ordinary = torch.tensor([1., 0.], dtype=torch.float64)
    outlier = torch.tensor([0., 100.], dtype=torch.float64)
    mixed = ordinary + outlier
    clipped = mixed / mixed.norm()
    angle = float(F.cosine_similarity(clipped, outlier, dim=0))
    assert angle > 0.9999
    residual = torch.ones(1, 3, 4, dtype=torch.float64)
    residual[:, 2] *= 100.
    influence = {}
    for kind in ("mse", "global", "token"):
        _, gradient = measure(residual, kind)
        norms = gradient.norm(dim=-1)[0]
        influence[kind] = norms.tolist()
    assert math.isclose(influence["mse"][2] / influence["mse"][0], 100.)
    assert math.isclose(influence["global"][2] / influence["global"][0], 100.)
    assert math.isclose(influence["token"][2] / influence["token"][0], 1.)

    # Rooting CE is NOT a constant-output-gradient-norm construction.
    ce_rows = []
    for wrong_logit in (0., 10., 100.):
        logits = torch.tensor([[0., wrong_logit]], dtype=torch.float64, requires_grad=True)
        loss = F.cross_entropy(logits, torch.tensor([0], device="cpu"))
        ce_gradient, = torch.autograd.grad(loss, logits, retain_graph=True)
        root_gradient, = torch.autograd.grad(loss.sqrt(), logits)
        ce_rows.append(dict(wrong_logit=wrong_logit, ce=float(loss.detach()),
                            ce_gradient_norm=float(ce_gradient.norm()),
                            root_ce_gradient_norm=float(root_gradient.norm())))
    assert ce_rows[-1]["root_ce_gradient_norm"] < ce_rows[0]["root_ce_gradient_norm"]

    # Masking and fractional source weights require explicit, chunk-invariant counts.
    weight = torch.tensor([[1., 0., .25, 1., 1., 0., 1.],
                           [.5, 1., 1., 0., 1., 1., 1.]], dtype=torch.float64)
    a = error.detach().clone().requires_grad_(True)
    whole = masked_reconstruction(a, weight, [7])
    b = error.detach().clone().requires_grad_(True)
    chunked = masked_reconstruction(b, weight, [2, 2, 3])
    differences = []
    for index in range(3):
        ga, = torch.autograd.grad(whole[index], a, retain_graph=True)
        gb, = torch.autograd.grad(chunked[index], b, retain_graph=True)
        torch.testing.assert_close(whole[index], chunked[index], rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(ga, gb, rtol=1e-12, atol=1e-12)
        differences.append(dict(kind=("token", "channel", "sample")[index],
                                loss_difference=float((whole[index] - chunked[index]).detach().abs()),
                                max_gradient_difference=float((ga - gb).abs().max())))
    chunk_roots = sum(measure(error[:, start:end], "global")[0] * (end - start) / 7
                      for start, end in ((0, 2), (2, 4), (4, 7)))
    full_root, _ = measure(error, "global")
    assert not math.isclose(chunk_roots, full_root, rel_tol=1e-6)

    # Equal output-space norms can still produce unequal parameter-space norms.
    jacobian = torch.diag(torch.tensor([1., 1000.], dtype=torch.float64))
    basis = torch.eye(2, dtype=torch.float64)
    parameter_norms = (basis @ jacobian).norm(dim=-1).tolist()
    assert parameter_norms == [1., 1000.]
    return dict(
        scope="synthetic CPU math only; no student quality, GPU performance or gain measurement",
        dtype="float64", torch_version=torch.__version__, device="cpu", seed=17,
        reconstruction_scale=rows, expected_rmse_output_gradient_norm=expected,
        token_outlier_gradient_norms=influence,
        global_clip_cosine_to_outlier=angle, ce_counterexample=ce_rows,
        chunk_invariance=differences,
        incorrect_chunk_root_difference=chunk_roots - full_root,
        equal_output_norm_parameter_norms=parameter_norms,
        checks="all passed",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="ascii")
    print(json.dumps({key: result[key] for key in
                      ("checks", "expected_rmse_output_gradient_norm", "token_outlier_gradient_norms",
                       "global_clip_cosine_to_outlier", "ce_counterexample", "chunk_invariance")}, indent=2))

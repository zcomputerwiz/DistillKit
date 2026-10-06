# Assisted-by: Codex
"""Bounded CCE diagnostics called through the existing head_parity.py harness."""
from __future__ import annotations

import importlib.metadata as metadata
import json
import statistics
import time

import torch

_version = metadata.version


def _windows_version(name):
    try:
        return _version(name)
    except metadata.PackageNotFoundError:
        if name == "triton":
            return _version("triton-windows")
        raise


metadata.version = _windows_version
from cut_cross_entropy import linear_cross_entropy  # noqa: E402
from cce_selected import prepare_buckets_cpu, selected_forward, upload_buckets  # noqa: E402
from shared_head import grouped_tail_kl_rows  # noqa: E402


def error(a, b):
    delta = (a.double() - b.double()).abs()
    return {"max_abs": float(delta.max()), "mean_abs": float(delta.mean())}


def mass(logits, lse):
    q = (logits - lse[:, None]).exp().sum(-1)
    return {"oversummed_rows": int((q > 1).sum()),
            "max_oversum": float((q - 1).clamp_min(0).max()),
            "min_tail": float((1 - q).min()),
            "tails_below_1e_6": int(((1 - q) < 1e-6).sum())}


def timed(call, repeats, backward=False):
    for _ in range(1):
        call(backward)
    torch.cuda.synchronize()
    times, peak = [], 0
    for _ in range(repeats):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        baseline = torch.cuda.memory_allocated()
        start = time.perf_counter()
        call(backward)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
        peak = max(peak, torch.cuda.max_memory_allocated() - baseline)
    return {"seconds": statistics.median(times), "samples": times, "peak_gib": peak / 2**30}


def gradient_error(a, b):
    # Chunking avoids several simultaneous FP32 copies of the 508M-element head.
    a, b = a.flatten(), b.flatten()
    stats = torch.zeros(4, device=a.device, dtype=torch.float64)
    for start in range(0, a.numel(), 1 << 20):
        aa, bb = a[start:start + (1 << 20)].float(), b[start:start + (1 << 20)].float()
        stats += torch.stack([(aa * bb).sum(), (aa * aa).sum(), (bb * bb).sum(),
                              ((aa - bb) ** 2).sum()]).double()
    dot, sa, sb, delta = stats.cpu().tolist()
    return {"cosine": dot / max((sa * sb) ** 0.5, 1e-30),
            "relative_l2": (delta / max(sb, 1e-30)) ** 0.5}


def gradient_sanity(h, head, targets, rows):
    # Tests both public CCE differentiable outputs against the same BF16 projection.
    # This is not a gradient test of the forward-only selected/tail extension.
    h, targets = h[:rows], targets[:rows]
    eh, ew = h.detach().requires_grad_(True), head.detach().requires_grad_(True)
    logits = (eh @ ew.T).float()
    lse = torch.logsumexp(logits, -1)
    nll = lse - logits.gather(-1, targets[:, None]).squeeze(-1)
    reference = (0.5 * nll + 0.5 * lse).mean()
    reference.backward()
    ref_h, ref_w = eh.grad, ew.grad
    reference_loss = float(reference)
    del eh, ew, logits, lse, nll, reference
    eh, ew = h.detach().requires_grad_(True), head.detach().requires_grad_(True)
    nll, lse = linear_cross_entropy(eh, ew, targets, reduction="none", return_lse=True, filter_eps=None)
    loss = (0.5 * nll + 0.5 * lse).mean()
    loss.backward()
    out = {"rows": len(h), "reference_loss": reference_loss, "cce_loss": float(loss),
           "loss_difference": float(loss) - reference_loss,
           "hidden": gradient_error(eh.grad, ref_h), "head": gradient_error(ew.grad, ref_w)}
    del eh, ew, nll, lse, loss, ref_h, ref_w
    torch.cuda.empty_cache()
    return out


def run_case(harness, model, hidden, batch, args):
    ids = batch["input_ids"]
    w = batch.get("weight")
    weight = torch.ones_like(ids, dtype=torch.float32) if w is None else w
    weight = weight.clone()
    weight[:, -1] = 0
    negative = batch.get("negative")
    kw = weight if negative is None else weight * ~negative
    if "context_kl" in batch:
        kw = kw + batch["context_kl"]
    at = ((weight > 0) | (kw > 0)).nonzero()
    all_at = at
    if len(at) > args.limit_rows:
        positions = torch.linspace(0, len(at) - 1, args.limit_rows, device=at.device).long()
        at = at[positions]
    r, t = at[:, 0], at[:, 1]
    h = hidden[r, t].detach().contiguous()
    targets = ids[r, t + 1].contiguous()
    kid = batch["topk_ids"][r, t].contiguous()
    kval = batch["topk_logprobs"][r, t].contiguous()
    wr, kwr = weight[r, t], kw[r, t]
    neg = None if negative is None else negative[r, t]
    head = model.lm_head.weight.detach()
    print(f"diagnostic rows={len(h)}, hidden={h.shape[1]}, teacher_k={kid.shape[1]}", flush=True)
    with torch.no_grad():
        logits = (h @ head.T).float()
        reference_lse = torch.logsumexp(logits.double(), -1)
        pick = logits.gather(-1, kid)
        ref_mass = mass(pick.double(), reference_lse)
        cce_nll, cce_lse = linear_cross_entropy(h, head, targets, reduction="none", return_lse=True,
                                              filter_eps=None)
        selected = selected_forward(h, head, targets, kid, tail=True)
        row = {"rows": len(h), "shape": list(h.shape), "teacher_k": kid.shape[1],
               "reference_mass": ref_mass,
               "cce_lse_vs_cublas_fp64_reduction": error(cce_lse, reference_lse),
               "same_tile_pick_vs_cublas": error(selected["pick"], pick),
               "same_tile_locked_lse_vs_cce": error(selected["locked_lse"], cce_lse),
               "same_tile_stable_lse_vs_locked": error(selected["stable_lse"], selected["locked_lse"]),
               "separate_cublas_pick_cce_lse_mass": mass(pick, cce_lse),
               "same_tile_locked_mass": mass(selected["pick"], selected["locked_lse"]),
               "same_tile_stable_mass": mass(selected["pick"].double(), selected["stable_lse"]),
               "direct_tail_logprob": {"min": float((selected["tail_lse"] - selected["stable_lse"]).min()),
                                       "max": float((selected["tail_lse"] - selected["stable_lse"]).max())}}
        tail_logits = logits.double()
        tail_logits.scatter_(-1, kid, -torch.inf)
        reference_tail = torch.logsumexp(tail_logits, -1) - reference_lse
        row["direct_tail_vs_cublas_reference"] = error(selected["tail_lse"] - selected["stable_lse"], reference_tail)
        if args.cce_bucketed:
            prepared = prepare_buckets_cpu(kid.cpu().numpy(), head.shape[0])
            gpu_buckets = upload_buckets(prepared, h.device)
            bucketed = selected_forward(h, head, targets, kid, tail=True, buckets=gpu_buckets)
            row["bucketed_pick_vs_same_tile"] = error(bucketed["pick"], selected["pick"])
            row["bucketed_tail_vs_cublas_reference"] = error(bucketed["tail_lse"] - bucketed["stable_lse"], reference_tail)
            row["bucketed_metadata_bytes_numerical_rows"] = prepared["metadata_bytes"]
            row["bucketed_work_numerical_rows"] = prepared["work"]
            if args.cce_unlocked:
                unlocked = selected_forward(h, head, targets, kid, tail=True, buckets=gpu_buckets, locked=False)
                row["unlocked_bucketed_pick_vs_same_tile"] = error(unlocked["pick"], selected["pick"])
                row["unlocked_bucketed_lse_vs_cublas_reference"] = error(unlocked["stable_lse"], reference_lse)
                row["unlocked_bucketed_tail_vs_cublas_reference"] = error(
                    unlocked["tail_lse"] - unlocked["stable_lse"], reference_tail)
                del unlocked
            del prepared, gpu_buckets, bucketed
        student_ref = pick.double() - reference_lse[:, None]
        kl_ref = grouped_tail_kl_rows(student_ref, kval.double())
        for label, z, ls in [("separate", pick, cce_lse),
                             ("same_tile_locked", selected["pick"], selected["locked_lse"]),
                             ("same_tile_stable", selected["pick"].double(), selected["stable_lse"])]:
            row[label + "_kl_vs_fp64_reference"] = error(grouped_tail_kl_rows(z - ls[:, None], kval.to(z.dtype)), kl_ref)
        del logits, tail_logits, selected, pick, cce_nll, cce_lse, kl_ref
    torch.cuda.empty_cache()
    row["cce_nll_and_lse_gradient_sanity"] = gradient_sanity(h, head, targets, args.gradient_rows)

    # Timing rows are gathered independently from the complete real record.
    at = all_at
    if len(at) > args.timing_rows:
        at = at[torch.linspace(0, len(at) - 1, args.timing_rows, device=at.device).long()]
    r, t = at[:, 0], at[:, 1]
    h = hidden[r, t].detach().contiguous()
    targets = ids[r, t + 1].contiguous()
    kid = batch["topk_ids"][r, t].contiguous()
    kval = batch["topk_logprobs"][r, t].contiguous()
    wr, kwr = weight[r, t], kw[r, t]
    neg = None if negative is None else negative[r, t]
    row["timing_rows"] = len(h)
    print(f"timing rows={len(h)}", flush=True)

    def cce(backward):
        eh = h.detach().requires_grad_(backward)
        ew = head.detach().requires_grad_(backward)
        nll, lse = linear_cross_entropy(eh, ew, targets, reduction="none", return_lse=True, filter_eps=None)
        # A lower bound: both normalizer and NLL gradient, no selected capture/correction.
        loss = (nll * wr + lse * kwr).sum() / wr.sum().clamp_min(1)
        if backward:
            loss.backward()

    def shared(backward):
        from shared_head import _chunk_losses
        from torch.utils.checkpoint import checkpoint
        eh = h.detach().requires_grad_(backward)
        ew = head.detach().requires_grad_(backward)
        total = eh.new_zeros(3, dtype=torch.float32)
        for start in range(0, len(h), args.head_chunk):
            s = slice(start, start + args.head_chunk)
            inp = (eh[s], ew, targets[s], wr[s], kid[s], kval[s], kwr[s], None if neg is None else neg[s])
            out = checkpoint(_chunk_losses, *inp, use_reentrant=False, preserve_rng_state=False) if backward else _chunk_losses(*inp)
            total = total + out
        loss = (total[1] + total[2] if batch["kl_only"] else 0.5 * (total[0] + total[1])) / wr.sum().clamp_min(1)
        if backward:
            loss.backward()

    row["shared_forward_backward"] = timed(shared, args.repeats, True)
    row["cce_unfiltered_forward_backward_lower_bound"] = timed(cce, args.repeats, True)
    row["same_tile_selected_forward"] = timed(lambda _: selected_forward(h, head, targets, kid), args.repeats)
    row["same_tile_direct_tail_forward"] = timed(lambda _: selected_forward(h, head, targets, kid, tail=True), args.repeats)
    if args.cce_bucketed:
        before = time.perf_counter()
        cpu_ids = kid.cpu().numpy()
        downloaded = time.perf_counter()
        prepared = prepare_buckets_cpu(cpu_ids, head.shape[0])
        prepared_at = time.perf_counter()
        gpu_buckets = upload_buckets(prepared, h.device)
        torch.cuda.synchronize()
        row["bucketed_preparation"] = {"download_seconds": downloaded - before,
                                      "cpu_seconds": prepared_at - downloaded,
                                      "upload_seconds": time.perf_counter() - prepared_at,
                                      "metadata_bytes": prepared["metadata_bytes"],
                                      "work": prepared["work"]}
        row["same_tile_bucketed_direct_tail_forward"] = timed(
            lambda _: selected_forward(h, head, targets, kid, tail=True, buckets=gpu_buckets), args.repeats)
        if args.cce_unlocked:
            row["same_tile_unlocked_bucketed_direct_tail_forward"] = timed(
                lambda _: selected_forward(h, head, targets, kid, tail=True, buckets=gpu_buckets, locked=False),
                args.repeats)
    return row


def run(harness, model, tokenizer, args):
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"device": torch.cuda.get_device_name(0), "visible_devices": 1,
              "physical_gpu": 1, "checkpoint": str(args.checkpoint),
              "filter_eps": None, "head_chunk": args.head_chunk, "cases": []}
    report.update(context_weight=args.context_weight, context_every=8, bucketed_forward=args.cce_bucketed,
                  unlocked_bucketed_forward=args.cce_unlocked)
    for index, (name, cache, kind) in enumerate(harness.CASES):
        if str(index) not in args.cases.split(","):
            continue
        print(f"prepare {index}: {name}", flush=True)
        batch = harness.batch_for(tokenizer, kind, cache, args.budget, context_weight=args.context_weight)
        with torch.no_grad():
            hidden = model.model(input_ids=batch["input_ids"], attention_mask=torch.ones_like(batch["input_ids"]),
                                 use_cache=False).last_hidden_state
        row = run_case(harness, model, hidden, batch, args)
        row.update(case=name, batch_shape=list(batch["input_ids"].shape))
        report["cases"].append(row)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(row, indent=2), flush=True)
        del hidden, batch
        torch.cuda.empty_cache()
    return report

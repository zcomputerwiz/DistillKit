"""Isolated performance review; random disposable weights, no production edits.

Run from the repo root with .venv/Scripts/python.exe. Every timed step synchronizes
both GPUs. Profiling and transfer accounting are separate from steady-state timing.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scratch/dense_gr"))

from benchmark import apply_liger, build  # noqa: E402
import torch  # noqa: E402
from cut_cross_entropy import linear_cross_entropy  # noqa: E402
from distillkit.models import Qwen35WidenedForCausalLM  # noqa: E402
from distillkit.parallel import collectives, clip_grad_norm, sync_replicated_gradients  # noqa: E402
from tp_train import shard_bodies  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402


@triton.jit
def peer_sum_kernel(local, peer, output, size: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = index < size
    a = tl.load(local + index, valid, other=0).to(tl.float32)
    b = tl.load(peer + index, valid, other=0).to(tl.float32)
    tl.store(output + index, a + b, valid)


def gpm_forward(normalized, code, weight):
    compute = torch.promote_types(normalized.dtype, torch.float32)
    width = normalized.shape[-1]
    result = torch.zeros_like(normalized[..., 0, :], dtype=compute)
    for index, x in enumerate(normalized.unbind(-2)):
        rows = weight[index * width:(index + 1) * width]
        result.add_(x * torch.nn.functional.linear(code, rows).sigmoid())
    return (result / normalized.shape[-2]).to(normalized.dtype)


def gpm_backward(grad_output, normalized, code, weight):
    compute = torch.promote_types(normalized.dtype, torch.float32)
    branches, width = normalized.shape[-2:]
    grad = grad_output.to(compute) / branches
    work = code.dtype
    flat_code = code.flatten(0, -2)
    dx = torch.empty_like(normalized)
    dc = torch.zeros_like(code, dtype=compute)
    dw = torch.empty_like(weight)
    for index in range(branches):
        rows = weight[index * width:(index + 1) * width].to(work)
        gate = torch.nn.functional.linear(code, rows).sigmoid().to(compute)
        dx[..., index, :] = (grad * gate).to(normalized.dtype)
        dz = (grad * normalized[..., index, :] * gate * (1 - gate)).to(work)
        dc += torch.nn.functional.linear(dz, rows.t()).to(compute)
        dw[index * width:(index + 1) * width] = (dz.flatten(0, -2).t() @ flat_code).to(weight.dtype)
    return dx, dc.to(code.dtype), dw


def split_compiled_gpm():
    forward = torch.compile(gpm_forward, fullgraph=True, dynamic=False)
    backward = torch.compile(gpm_backward, fullgraph=True, dynamic=False)

    class SplitCompiledGPM(torch.autograd.Function):
        @staticmethod
        def forward(ctx, normalized, code, weight):
            ctx.save_for_backward(normalized, code, weight)
            return forward(normalized, code, weight)

        @staticmethod
        @torch.autograd.function.once_differentiable
        def backward(ctx, grad):
            return backward(grad, *ctx.saved_tensors)

    return SplitCompiledGPM


def sync():
    for device in range(2):
        torch.cuda.synchronize(device)


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def statistics_ms(samples):
    return {"median_ms": statistics.median(samples), "minimum_ms": min(samples),
            "maximum_ms": max(samples), "samples_ms": samples}


def timed(call, repeats):
    samples = []
    for _ in range(repeats):
        sync()
        start = time.perf_counter()
        call()
        sync()
        samples.append(1000 * (time.perf_counter() - start))
    return statistics_ms(samples)


def traffic_wrappers():
    """Count actual collective copy payloads; do not omit backward/recompute."""
    counters = defaultdict(Counter)
    saved = []

    def record(name, tensor, device):
        if tensor.device != torch.device(device):
            counters[name]["copies"] += 1
            counters[name]["bytes"] += tensor.numel() * tensor.element_size()

    for cls in (collectives.Replicate, collectives.Reduce):
        for method in ("forward", "backward"):
            original = getattr(cls, method)
            name = cls.__name__ + "." + method

            def make(original=original, cls=cls, method=method, name=name):
                def wrapper(ctx, *args):
                    if cls is collectives.Replicate:
                        if method == "forward":
                            for device in args[1:]:
                                record(name, args[0], device)
                        else:
                            for grad in args:
                                record(name, grad, ctx.source_device)
                    elif method == "forward":
                        for tensor in args[1:]:
                            record(name, tensor, args[0])
                    else:
                        for device in ctx.shard_devices:
                            record(name, args[0], device)
                    with torch.profiler.record_function(name):
                        return original(ctx, *args)
                return wrapper

            saved.append((cls, method, original))
            setattr(cls, method, staticmethod(make()))
    return counters, saved


def profile_step(step, path):
    counters, originals = traffic_wrappers()
    try:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA]) as prof:
            step()
            sync()
        prof.export_chrome_trace(str(path))
    finally:
        for cls, method, original in originals:
            setattr(cls, method, staticmethod(original))
    events = json.loads(path.read_text(encoding="utf-8"))["traceEvents"]
    gpu = defaultdict(lambda: defaultdict(float))
    kernels = defaultdict(float)
    runtime = defaultdict(lambda: [0, 0.0])
    for event in events:
        if event.get("ph") != "X":
            continue
        category = event.get("cat", "")
        duration = event.get("dur", 0.0)
        if category in ("kernel", "gpu_memcpy", "gpu_memset"):
            device = str(event.get("args", {}).get("device", event.get("pid")))
            gpu[device][category + "_ms"] += duration / 1000
            gpu[device][category + "_count"] += 1
            kernels[event["name"]] += duration / 1000
        if category == "cuda_runtime":
            runtime[event["name"]][0] += 1
            runtime[event["name"]][1] += duration / 1000
    return {"trace": str(path), "body_collective_traffic": dict(counters),
            "body_transfer_mib": sum(item["bytes"] for item in counters.values()) / 2**20,
            "gpu_activity": dict(gpu),
            "top_kernels_ms": sorted(kernels.items(), key=lambda x: -x[1])[:25],
            "cuda_runtime": dict(runtime),
            "note": "GPU event sums can overlap and are not fractions of wall time."}


def scoped_events(step):
    """Elapsed stream intervals including dependencies; these are not kernel sums."""
    from distillkit.experimental.hyper_connection import HyperConnection, _GatedProjectMean
    records = defaultdict(list)
    originals = []
    targets = [(HyperConnection, "_route", False), (HyperConnection, "write", False)]
    if hasattr(_GatedProjectMean, "forward"):
        targets += [(_GatedProjectMean, "forward", True),
                    (_GatedProjectMean, "backward", True)]
    for cls in (collectives.Replicate, collectives.Reduce):
        targets += [(cls, "forward", True), (cls, "backward", True)]
    for cls, method, is_static in targets:
        original = getattr(cls, method)
        name = cls.__name__ + "." + method
        def make(original=original, name=name):
            def wrapper(*args, **kwargs):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record(torch.cuda.current_stream(0))
                result = original(*args, **kwargs)
                end.record(torch.cuda.current_stream(0))
                records[name].append((start, end))
                return result
            return wrapper
        originals.append((cls, method, original, is_static))
        setattr(cls, method, staticmethod(make()) if is_static else make())
    try:
        step()
        sync()
    finally:
        for cls, method, original, is_static in originals:
            setattr(cls, method, staticmethod(original) if is_static else original)
    return {name: {"calls": len(pairs), "sum_gpu0_elapsed_ms":
                   sum(start.elapsed_time(end) for start, end in pairs)}
            for name, pairs in records.items()}


def step_review(args):
    torch.manual_seed(20260920)
    for device in range(2):
        torch.cuda.set_per_process_memory_fraction(0.9, device)
    config = build(args.hidden, args.layers, 16384, ratio="3:1", blend=1.,
                   attn_implementation="flash_attention_2")
    config.residual_stream_norm_mode = args.norm
    if args.compile_gated_mean or args.split_compile_gated_mean:
        # Diagnostic process-local substitution only. Source files/checkpoints do
        # not change, and the measured BF16 numerical differences are reported.
        import types
        from distillkit.experimental import hyper_connection as hc
        if args.split_compile_gated_mean:
            hc._GatedProjectMean = split_compiled_gpm()
        else:
            compiled = torch.compile(hc._GatedProjectMean.apply, fullgraph=True, dynamic=False)
            hc._GatedProjectMean = types.SimpleNamespace(apply=compiled)
    model = Qwen35WidenedForCausalLM(config).to(dtype=torch.bfloat16)
    apply_liger(model, config)
    if args.cards == 2:
        shard_bodies(model, ["cuda:0", "cuda:1"])
    else:
        model.to("cuda:0")
    model.model.gradient_checkpointing = args.checkpointing
    model.train()
    tokens = torch.randint(0, 16384, (args.batch, args.length), device="cuda:0")
    mask = None if args.no_mask else torch.ones_like(tokens)
    import bitsandbytes as bnb
    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=1e-4)

    def step():
        optimizer.zero_grad(set_to_none=True)
        with torch.profiler.record_function("review.forward"):
            hidden = model.model(input_ids=tokens, attention_mask=mask,
                                 use_cache=False).last_hidden_state
        with torch.profiler.record_function("review.loss"):
            loss = linear_cross_entropy(hidden, model.lm_head.weight, tokens,
                                        shift=1, reduction="mean")
        with torch.profiler.record_function("review.backward"):
            loss.backward()
        with torch.profiler.record_function("review.sync_clip_step"):
            if args.cards == 2:
                sync_replicated_gradients(model)
                clip_grad_norm(model, 1.0)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        return loss

    start = time.perf_counter()
    with torch.autograd.set_multithreading_enabled(False):
        step()
    sync()
    cold = time.perf_counter() - start
    for _ in range(3):
        step()
    sync()
    for device in range(2):
        torch.cuda.reset_peak_memory_stats(device)
    result = {"args": vars(args) | {"output": str(args.output)},
              "torch": torch.__version__, "cold_step_seconds": cold,
              "timing": timed(step, args.steps),
              "peak_allocated_gib": [torch.cuda.max_memory_allocated(d) / 2**30
                                     for d in range(2)]}
    result["tokens_per_second"] = args.batch * args.length * 1000 / result["timing"]["median_ms"]
    if args.profile:
        result["profile"] = profile_step(step, args.output.with_suffix(".trace.json"))
        result["scoped_cuda_events"] = scoped_events(step)
    save(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k != "profile"}, indent=2), flush=True)


def peer_review(args):
    results = []
    for mib in (16, 32, 64, 128):
        elements = mib * 2**20 // 2
        source = torch.randn(elements, dtype=torch.bfloat16, device="cuda:1")
        local = torch.randn_like(source, device="cuda:0")
        staged = source.to("cuda:0", non_blocking=True)
        sync()  # Remote producer is complete before the diagnostic peer dereference.
        def fused_peer_sum():
            output = torch.empty_like(local)
            with torch.cuda.device(0):
                peer_sum_kernel[(triton.cdiv(elements, 1024),)](
                    local, source, output, elements, 1024, num_warps=4)
            return output
        calls = {"peer_copy": lambda: source.to("cuda:0", non_blocking=True),
                 "local_add": lambda: local + staged,
                 "copy_add": lambda: local + source.to("cuda:0", non_blocking=True),
                 "triton_peer_sum": fused_peer_sum}
        row = {"mib": mib}
        actual = fused_peer_sum()
        sync()
        row["triton_bitwise_equal"] = torch.equal(actual, local + staged)
        for name, call in calls.items():
            for _ in range(10):
                call()
            row[name] = timed(call, args.steps)
        results.append(row)
    save(args.output, {"results": results})
    print(json.dumps(results, indent=2), flush=True)


def route_review(args):
    from distillkit.experimental.hyper_connection import _GatedProjectMean
    torch.manual_seed(20260920)
    x = torch.randn(args.batch, args.length, 4, args.hidden, device="cuda:0",
                    dtype=torch.bfloat16, requires_grad=True)
    code = torch.randn(args.batch, args.length, args.hidden // 8, device="cuda:0",
                       dtype=torch.bfloat16, requires_grad=True)
    weight = (torch.randn(4 * args.hidden, args.hidden // 8, device="cuda:0",
                          dtype=torch.bfloat16) / (args.hidden // 8)**.5).requires_grad_()
    grad = torch.randn_like(x[..., 0, :])
    compiled = torch.compile(_GatedProjectMean.apply, fullgraph=True, dynamic=False)
    results = {}
    outputs = {}
    for name, call in (("eager", _GatedProjectMean.apply), ("compiled", compiled),
                       ("split_compiled", split_compiled_gpm().apply)):
        def step():
            for tensor in (x, code, weight):
                tensor.grad = None
            result = call(x, code, weight)
            result.backward(grad)
            return result
        start = time.perf_counter()
        result = step()
        sync()
        cold = time.perf_counter() - start
        outputs[name] = [t.detach().clone() for t in (result, x.grad, code.grad, weight.grad)]
        for _ in range(3):
            step()
        torch.cuda.reset_peak_memory_stats(0)
        results[name] = {**timed(step, args.steps), "cold_seconds": cold,
                         "peak_allocated_gib": torch.cuda.max_memory_allocated(0) / 2**30}
        saved = []
        def pack(tensor):
            saved.append({"shape": list(tensor.shape), "dtype": str(tensor.dtype),
                          "bytes": tensor.numel() * tensor.element_size()})
            return tensor
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            held = call(x, code, weight)
        results[name]["saved_tensors"] = saved
        results[name]["saved_tensor_mib"] = sum(t["bytes"] for t in saved) / 2**20
        del held
    results["relative_l2_difference"] = {
        name: float((a.float() - b.float()).norm() / a.float().norm().clamp_min(1e-10))
        for name, a, b in zip(("output", "dx", "dcode", "dweight"),
                              outputs["eager"], outputs["compiled"])}
    results["split_relative_l2_difference"] = {
        name: float((a.float() - b.float()).norm() / a.float().norm().clamp_min(1e-10))
        for name, a, b in zip(("output", "dx", "dcode", "dweight"),
                              outputs["eager"], outputs["split_compiled"])}
    save(args.output, results)
    print(json.dumps(results, indent=2), flush=True)


def explain_review(args):
    config = build(512, 4, 16384, ratio="3:1", blend=1.,
                   attn_implementation="flash_attention_2")
    config.residual_stream_norm_mode = "compiled"
    model = Qwen35WidenedForCausalLM(config).to(dtype=torch.bfloat16)
    apply_liger(model, config)
    if args.cards == 2:
        shard_bodies(model, ["cuda:0", "cuda:1"])
    else:
        model.to("cuda:0")
    tokens = torch.randint(0, 16384, (1, 128), device="cuda:0")
    model.model(input_ids=tokens, attention_mask=None, use_cache=False)
    explanation = torch._dynamo.explain(model.model)(
        input_ids=tokens, attention_mask=None, use_cache=False)
    result = {"cards": args.cards, "graph_count": explanation.graph_count,
              "graph_break_count": explanation.graph_break_count,
              "op_count": explanation.op_count,
              "break_reasons": [{"reason": item.reason,
                                  "stack": [str(frame) for frame in item.user_stack]}
                                 for item in explanation.break_reasons]}
    save(args.output, result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("step", "peer", "route", "explain"), default="step")
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--layers", type=int, default=10)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--length", type=int, default=1024)
    parser.add_argument("--cards", type=int, default=2)
    parser.add_argument("--norm", default="compiled")
    parser.add_argument("--checkpointing", action="store_true")
    parser.add_argument("--no-mask", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--compile-gated-mean", action="store_true")
    parser.add_argument("--split-compile-gated-mean", action="store_true")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    {"step": step_review, "peer": peer_review, "route": route_review,
     "explain": explain_review}[args.mode](args)

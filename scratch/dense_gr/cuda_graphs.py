"""Do CUDA graphs recover anything, given the step is already 97% GPU-busy?

The profile says device time is 761 ms against 781 ms of wall clock, so there is at most
3% of launch gap at batch 32 and graphs should be near-pointless there. But launches did
bind at small batch -- the 38M model takes 1.36x the time for 2x the work going from batch
8 to 16 -- so the honest test is at both, not at the batch that already hid the problem.

Two routes are tried, because they fail differently. `torch.compile(mode="reduce-overhead")`
puts graphs behind dynamo, which has 58 graph breaks to contend with here. Manual capture
sidesteps dynamo entirely and instead needs the step to be sync-free and shape-static; the
optimizer and gradient clipping are left outside the captured region, so what is measured
is forward plus backward, which is where the launches are.

    CUDA_VISIBLE_DEVICES=0 python scratch/dense_gr/cuda_graphs.py
"""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

_original = metadata.version


def _version(name):
    try:
        return _original(name)
    except metadata.PackageNotFoundError:
        if name == "triton":
            return _original("triton-windows")
        raise


metadata.version = _version

import torch  # noqa: E402

from benchmark import apply_liger, build  # noqa: E402
from cut_cross_entropy import linear_cross_entropy  # noqa: E402
from distillkit.models import Qwen35WidenedForCausalLM  # noqa: E402


def make_model(hidden, layers, vocab, liger, attn="sdpa"):
    config = build(hidden, layers, vocab, attn_implementation=attn)
    torch.manual_seed(0)
    model = Qwen35WidenedForCausalLM(config).to(device="cuda", dtype=torch.bfloat16)
    model.train()
    if liger:
        apply_liger(model, config)
    return model, config


def decoder_forward_backward(model, tokens, attention):
    """Decoder only, reduced by a scalar that stands in for the loss.

    CCE cannot be captured, so a fully graphed step is not available. This measures
    the part that can be: everything up to the head.
    """
    state = model.model(input_ids=tokens, attention_mask=attention,
                        use_cache=False).last_hidden_state
    state.square().mean().backward()
    return state


def forward_backward(model, tokens, attention):
    hidden = model.model(input_ids=tokens, attention_mask=attention,
                         use_cache=False).last_hidden_state
    loss = linear_cross_entropy(hidden, model.lm_head.weight, tokens, shift=1,
                                reduction="mean")
    loss.backward()
    return loss


def timed(function, steps=10):
    for _ in range(3):
        function()
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(steps):
        function()
    torch.cuda.synchronize()
    return (time.perf_counter() - started) / steps


def eager_step(model, tokens, attention, decoder_only=False):
    body = decoder_forward_backward if decoder_only else forward_backward

    def run():
        model.zero_grad(set_to_none=False)
        body(model, tokens, attention)
    return run


def captured_step(model, tokens, attention, decoder_only=False):
    """Capture forward+backward into a graph and replay it.

    `set_to_none=False` is required: capture replays the same kernels writing the same
    addresses, so the gradient tensors have to exist and stay put. Warmup runs on a side
    stream, which is what the capture API asks for.
    """
    body = decoder_forward_backward if decoder_only else forward_backward
    model.zero_grad(set_to_none=False)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            model.zero_grad(set_to_none=False)
            body(model, tokens, attention)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()

    model.zero_grad(set_to_none=False)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        body(model, tokens, attention)

    def run():
        graph.replay()
    return run


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--vocab", type=int, default=16_384)
    parser.add_argument("--length", type=int, default=1024)
    parser.add_argument("--batches", type=int, nargs="+", default=[4, 8, 16, 32])
    parser.add_argument("--liger", action="store_true")
    parser.add_argument("--skip-compile", action="store_true",
                        help="reduce-overhead compiles for minutes per batch size here")
    parser.add_argument("--attn", default="sdpa",
                        choices=("sdpa", "flash_attention_2", "eager"),
                        help="flash_attention_2 makes the step uncapturable")
    parser.add_argument("--decoder-only", action="store_true",
                        help="graph the decoder and leave the loss eager, which is "
                             "the only split that captures at all here")
    parser.add_argument("--output", type=Path,
                        default=Path("scratch/dense_gr/cuda-graphs.json"))
    args = parser.parse_args()

    torch.cuda.set_per_process_memory_fraction(0.90, 0)
    rows = []
    print("%6s %12s %12s %12s %9s" % ("batch", "eager ms", "graph ms", "compile ms",
                                      "graph gain"))
    for batch in args.batches:
        row = {"batch": batch}
        tokens = torch.randint(0, args.vocab, (batch, args.length), device="cuda")
        attention = torch.ones_like(tokens)

        model, _ = make_model(args.hidden, args.layers, args.vocab, args.liger,
                              args.attn)
        row["eager_seconds"] = timed(eager_step(model, tokens, attention,
                                                args.decoder_only))
        del model
        torch.cuda.empty_cache()

        model, _ = make_model(args.hidden, args.layers, args.vocab, args.liger,
                              args.attn)
        try:
            row["graph_seconds"] = timed(captured_step(model, tokens, attention,
                                                       args.decoder_only))
        except Exception as error:
            row["graph_error"] = "%s: %s" % (type(error).__name__, str(error)[:120])
        del model
        torch.cuda.empty_cache()

        model, _ = make_model(args.hidden, args.layers, args.vocab, args.liger,
                              args.attn)
        try:
            if args.skip_compile:
                # With 58 graph breaks, reduce-overhead spends minutes compiling per
                # batch size and then cannot place graphs across the breaks anyway.
                raise RuntimeError("skipped by --skip-compile")
            model.model = torch.compile(model.model, mode="reduce-overhead")
            row["compile_seconds"] = timed(eager_step(model, tokens, attention,
                                                      args.decoder_only), steps=6)
        except Exception as error:
            row["compile_error"] = "%s: %s" % (type(error).__name__, str(error)[:120])
        del model
        torch.cuda.empty_cache()

        gain = (row["eager_seconds"] / row["graph_seconds"]
                if "graph_seconds" in row else float("nan"))
        row["graph_speedup"] = gain
        rows.append(row)
        print("%6d %12.2f %12s %12s %8.3fx"
              % (batch, 1000 * row["eager_seconds"],
                 "%.2f" % (1000 * row["graph_seconds"]) if "graph_seconds" in row
                 else "FAILED",
                 "%.2f" % (1000 * row["compile_seconds"]) if "compile_seconds" in row
                 else "FAILED", gain))
        for key in ("graph_error", "compile_error"):
            if key in row:
                print("       %s -> %s" % (key, row[key]))

    report = {"hidden": args.hidden, "layers": args.layers, "vocab": args.vocab,
              "length": args.length, "liger": args.liger,
              "device": torch.cuda.get_device_name(0), "rows": rows}
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nwrote %s" % args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

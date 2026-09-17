"""Which component makes the step uncapturable?

A failed capture invalidates the CUDA context, so every later operation in the process
fails too and one traceback tells you nothing about which op was at fault. Each component
therefore runs in its own process, selected by `--component`, and the caller compares exit
codes.

    python scratch/dense_gr/graph_capture_probe.py --component mlp
    python scratch/dense_gr/graph_capture_probe.py --component decoder
    python scratch/dense_gr/graph_capture_probe.py --component cce
    python scratch/dense_gr/graph_capture_probe.py --component full
"""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import sys
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


def capture(step, warmups=8):
    """Warm up on a side stream, then capture. Autotuning must finish in warmup."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(warmups):
            step()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    graph.replay()
    torch.cuda.synchronize()
    return graph


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component", required=True,
                        choices=("mlp", "decoder", "cce", "full"))
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--length", type=int, default=1024)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--vocab", type=int, default=16_384)
    parser.add_argument("--attn", default="flash_attention_2",
                        choices=("flash_attention_2", "sdpa", "eager"))
    args = parser.parse_args()

    torch.cuda.set_per_process_memory_fraction(0.90, 0)
    device, dtype = "cuda", torch.bfloat16

    if args.component == "mlp":
        # A plain SwiGLU stack: no fla, no CCE, no custom autograd Function.
        layer = torch.nn.Sequential(
            torch.nn.Linear(args.hidden, 4 * args.hidden, bias=False),
            torch.nn.SiLU(),
            torch.nn.Linear(4 * args.hidden, args.hidden, bias=False),
        ).to(device=device, dtype=dtype)
        x = torch.randn(args.batch, args.length, args.hidden, device=device, dtype=dtype)

        def step():
            layer.zero_grad(set_to_none=False)
            layer(x).square().mean().backward()

    elif args.component == "cce":
        from cut_cross_entropy import linear_cross_entropy
        hidden = torch.randn(args.batch, args.length, args.hidden, device=device,
                             dtype=dtype, requires_grad=True)
        weight = torch.randn(args.vocab, args.hidden, device=device, dtype=dtype,
                             requires_grad=True)
        targets = torch.randint(0, args.vocab, (args.batch, args.length), device=device)

        def step():
            if hidden.grad is not None:
                hidden.grad.zero_()
                weight.grad.zero_()
            linear_cross_entropy(hidden, weight, targets, shift=1,
                                 reduction="mean").backward()

    else:
        from benchmark import build
        from distillkit.models import Qwen35WidenedForCausalLM

        config = build(args.hidden, args.layers, args.vocab,
                       attn_implementation=args.attn)
        torch.manual_seed(0)
        model = Qwen35WidenedForCausalLM(config).to(device=device, dtype=dtype)
        model.train()
        tokens = torch.randint(0, args.vocab, (args.batch, args.length), device=device)
        attention = torch.ones_like(tokens)

        if args.component == "decoder":
            def step():
                model.zero_grad(set_to_none=False)
                state = model.model(input_ids=tokens, attention_mask=attention,
                                    use_cache=False).last_hidden_state
                state.square().mean().backward()
        else:
            from cut_cross_entropy import linear_cross_entropy

            def step():
                model.zero_grad(set_to_none=False)
                state = model.model(input_ids=tokens, attention_mask=attention,
                                    use_cache=False).last_hidden_state
                linear_cross_entropy(state, model.lm_head.weight, tokens, shift=1,
                                     reduction="mean").backward()

    try:
        capture(step)
    except Exception as error:
        print("CAPTURE FAILED [%s]: %s: %s"
              % (args.component, type(error).__name__, str(error).splitlines()[0][:150]))
        return 1
    print("CAPTURE OK [%s]" % args.component)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

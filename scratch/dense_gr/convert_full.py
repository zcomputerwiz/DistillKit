"""Convert a trained plain arm onto the whole stack: gated residual, MLA and CSA2.

The gated residual converts exactly and costs nothing. Latent attention does not: the
stock layer holds `num_key_value_heads` keys and values, a per-head norm over the whole
head, and one rotary key per key/value head, where MLA holds a shared latent, a norm over
the content half only, and a single rotary key. So the key and value side is *fitted*
rather than copied, by least squares against what the source layer actually produced on
real tokens. CSA2's indexer has no counterpart in the source at all and starts where it
is constructed -- whether training can pick it up from there is the question this
checkpoint exists to ask.

The fit runs in layer order because it has to. A Reindex or Reuse layer owns no down
projection; it reads the latent an earlier Full layer published, through an adapter that
starts at the identity. Its target therefore depends on the converted donor's output,
which does not exist until the donor is fitted. So: fit a Full layer, install it, run the
calibration windows again, fit whoever borrows from it.

What transfers verbatim: embeddings, every linear-attention layer, both MLPs, the norms,
`q_proj`, `q_norm` and `o_proj`. The query path is deliberately untouched, which is what
lets a comparison against the source attribute the difference to the key/value side.

Held-out loss is measured the way the arms measured it -- the calibration split, the same
fixed windows under seed 12345 -- so the number printed here is comparable to the numbers
in their milestones.

    python scratch/dense_gr/convert_full.py \\
        --source scratch/dense_gr/checkpoints-arm/smoke-r1-1-nogr \\
        --output scratch/dense_gr/checkpoints-conv/full-r1-1 \\
        --csa2-modes full reuse full reindex reuse
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import triton_shim  # noqa: F401,E402

from distillkit.models import Qwen35WidenedForCausalLM  # noqa: E402

STORE = Path("scratch/code_training/tokens-v2")
RIDGE = 1e-6
EVALUATION_SEED = 12345


def open_split(store, split, vocab):
    """The remapped split when one exists for this vocabulary, else the original."""
    compact = store / ("%s-v%d.bin" % (split, vocab))
    if compact.exists():
        return np.memmap(compact, dtype=np.uint16, mode="r")
    return np.memmap(store / ("%s.bin" % split), dtype=np.uint32, mode="r")


def calibration_windows(store, vocab, count, length, device):
    """Evenly spaced windows of the training split the arms were trained on."""
    stream = open_split(store, "train", vocab)
    stride = max(length, (len(stream) - length) // max(1, count))
    for index in range(count):
        start = index * stride
        yield torch.from_numpy(
            np.array(stream[start:start + length], dtype=np.int64).reshape(1, length)
        ).to(device)


@torch.no_grad()
def heldout(model, store, vocab, count, length, device):
    """The arms' own held-out measure: the calibration split, fixed windows, seed 12345."""
    stream = open_split(store, "calibration", vocab)
    starts = np.random.default_rng(EVALUATION_SEED).integers(
        0, stream.shape[0] - length - 1, size=count)
    total, seen = 0.0, 0
    for start in starts:
        ids = torch.from_numpy(
            np.array(stream[start:start + length], dtype=np.int64).reshape(1, length)
        ).to(device)
        logits = model(input_ids=ids, use_cache=False).logits.float()
        total += nn.functional.cross_entropy(
            logits[0, :-1], ids[0, 1:], reduction="sum").item()
        seen += length - 1
    return total / seen


def solve(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Least squares ``target ~= source @ W.T``, ridge-stabilized."""
    gram = source.T @ source
    gram = gram + torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device) * (
        gram.diagonal().mean().clamp_min(1e-30) * RIDGE)
    return torch.linalg.solve(gram, source.T @ target).T


def whiten(matrix):
    eigenvalues, vectors = torch.linalg.eigh(matrix)
    eigenvalues = eigenvalues.clamp_min(eigenvalues.max().clamp_min(1e-30) * RIDGE)
    return vectors @ torch.diag(eigenvalues.rsqrt()) @ vectors.T


def source_capture(stock, full, geometry, store, vocab, count, length, device):
    """Per-layer block inputs and the keys, values and rotary keys attention consumed."""
    rope, content, head_dim, groups = geometry
    book = {index: [] for index in full}

    def hook(index):
        def inner(module, args, kwargs):
            states = kwargs.get("hidden_states", args[0] if args else None)
            shape = (*states.shape[:-1], -1, head_dim)
            key = module.k_norm(module.k_proj(states).view(shape))
            value = module.v_proj(states).view(shape)
            book[index].append((
                states.reshape(-1, states.shape[-1]).double(),
                key[..., rope:].repeat_interleave(groups, dim=-2).flatten(-2)
                    .reshape(-1, key.shape[-2] * groups * content).double(),
                value.repeat_interleave(groups, dim=-2).flatten(-2)
                    .reshape(-1, value.shape[-2] * groups * head_dim).double(),
                # One shared rotary key against the source's one per key/value head: the
                # mean is the least-squares target when one vector must serve them all.
                key[..., :rope].mean(-2).reshape(-1, rope).double(),
            ))
        return inner

    handles = [stock.model.layers[i].self_attn.register_forward_pre_hook(
        hook(i), with_kwargs=True) for i in full]
    with torch.no_grad():
        for ids in calibration_windows(store, vocab, count, length, device):
            stock(input_ids=ids, use_cache=False)
    for handle in handles:
        handle.remove()
    return {index: tuple(torch.cat(part) for part in zip(*rows))
            for index, rows in book.items()}


def _record_latent(attention):
    """Record the post-norm, post-adapter latent this layer's up-projection receives."""
    attention._latent_log = []
    original = attention.kv_b_proj.forward

    def recording(x):
        attention._latent_log.append(x.reshape(-1, x.shape[-1]).double())
        return original(x)

    attention.kv_b_proj.forward = recording

    class Handle:
        def remove(self):
            attention.kv_b_proj.forward = original
    return Handle()


def build_target_config(source_config, args):
    config = copy.deepcopy(source_config)
    # CSA2 routes over a whole sequence and writes no cache, so a converted model
    # cannot carry a config that asks for one.
    config.use_cache = False
    # A checkpoint trained in this project already names the four-stream route, because
    # its arm was built with one. A stock one does not, and `residual_stream_routing`
    # falls back to "widened" -- which `recipient_initialize` refuses, since there is no
    # route for it to convert. Name it here so a stock source converts like any other.
    # The rank follows `benchmark.py`: hidden // 8, which is what the 2B's measured
    # +52.3M parameters were counted against.
    config.residual_stream_enabled = True
    config.residual_stream_routing = "flash_next"
    config.residual_stream_num_branches = getattr(
        source_config, "residual_stream_num_branches", 2)
    config.residual_stream_lowrank = getattr(
        source_config, "residual_stream_lowrank", max(8, config.hidden_size // 8))
    config.residual_stream_sidecar = False
    config.mla_enabled = True
    config.mla_latent_dim = args.mla_latent_dim
    if args.csa2_modes:
        config.csa2_enabled = True
        config.csa2_modes = list(args.csa2_modes)
        config.csa2_top_k = args.csa2_top_k
        config.csa2_local_window = args.csa2_local_window
        config.csa2_block_size = args.csa2_block_size
    return config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mla-latent-dim", type=int, default=128)
    parser.add_argument("--csa2-modes", nargs="*", default=None,
                        help="one per full-attention layer; omit for MLA without CSA2")
    parser.add_argument("--csa2-top-k", type=int, default=256)
    parser.add_argument("--csa2-local-window", type=int, default=128)
    parser.add_argument("--csa2-block-size", type=int, default=128)
    parser.add_argument("--calibrate", type=int, default=32)
    parser.add_argument("--evaluate", type=int, default=64)
    parser.add_argument("--length", type=int, default=1024)
    parser.add_argument("--store", type=Path, default=STORE)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--source-device", default=None,
                        help="where the teacher and its captured activations live. "
                             "Defaults to --device; a second card keeps the two models "
                             "and the calibration off each other.")
    parser.add_argument("--no-blend", action="store_true",
                        help="skip the gated residual conversion, to separate its effect")
    args = parser.parse_args()

    device = torch.device(args.device)
    source_device = torch.device(args.source_device or args.device)
    stock = Qwen35WidenedForCausalLM.from_pretrained(
        args.source, dtype=torch.bfloat16).to(source_device).eval()
    config = stock.config
    head_dim = getattr(config, "head_dim",
                       config.hidden_size // config.num_attention_heads)
    rope = int(head_dim * config.rope_parameters["partial_rotary_factor"])
    content = head_dim - rope
    heads, kv_heads = config.num_attention_heads, config.num_key_value_heads
    groups = heads // kv_heads
    full = [i for i, kind in enumerate(config.layer_types) if "linear" not in str(kind)]
    vocab = config.vocab_size

    before = heldout(stock, args.store, vocab, args.evaluate, args.length, source_device)
    print("source %s" % args.source)
    print("  hidden %d, %d layers, %d heads of %d over %d kv, full attention at %s"
          % (config.hidden_size, config.num_hidden_layers, heads, head_dim, kv_heads, full))
    print("  cache %d per token per layer, heldout %.4f"
          % (kv_heads * head_dim * 2, before), flush=True)

    target_config = build_target_config(config, args)
    torch.manual_seed(0)
    model = Qwen35WidenedForCausalLM(target_config).to(device=device, dtype=torch.bfloat16)
    # Shape mismatches are refused even under strict=False, and there is one by design:
    # the stock key norm spans the whole head where MLA's spans the content half, because
    # normalizing the assembled key would couple the shared rotary slice to it. Drop
    # anything whose shape disagrees rather than reshaping something we do not understand.
    wanted = model.state_dict()
    transferable = {key: value for key, value in stock.state_dict().items()
                    if key in wanted and wanted[key].shape == value.shape}
    dropped = [key for key in stock.state_dict()
               if key in wanted and key not in transferable]
    report = model.load_state_dict(transferable, strict=False)
    if dropped:
        print("\n%d tensors exist in both but disagree on shape and were left alone: %s"
              % (len(dropped), ", ".join(sorted({k.split(".", 3)[-1] for k in dropped}))))
    print("\n%d tensors did not transfer and were newly initialized:"
          % len(report.missing_keys))
    for key in report.missing_keys[:8]:
        print("  %s" % key)
    if len(report.missing_keys) > 8:
        print("  ... and %d more" % (len(report.missing_keys) - 8))
    print("%d source tensors have no home in the target (the replaced key/value side)"
          % len(report.unexpected_keys))

    targets = source_capture(stock, full, (rope, content, head_dim, groups),
                             args.store, vocab, args.calibrate, args.length,
                             source_device)
    tokens = next(iter(targets.values()))[0].shape[0]
    print("\ncalibrated on %d tokens per layer" % tokens, flush=True)
    del stock
    torch.cuda.empty_cache()

    # Which layers will read each donor's latent. A borrowing layer reads the most recent
    # full layer, so a donor's encoder has to summarize its readers' keys and values as
    # well as its own. Spending the whole rank budget on itself is what leaves a borrower
    # at 0.93 of its target where a joint fit reaches 0.97, measured by borrow_sweep.py.
    readers, donor = {}, None
    for index in full:
        if getattr(model.model.layers[index].self_attn, "mode", "full") == "full":
            donor, readers[index] = index, []
        elif donor is not None:
            readers[donor].append(index)
    if any(readers.values()):
        print("\ndonors and their readers: %s"
              % ", ".join("%d serves %s" % (d, r) for d, r in readers.items() if r))

    print("\n%-7s %-9s %9s %9s %9s"
          % ("layer", "mode", "key r2", "value r2", "rope r2"))
    fits = []
    for index in full:
        attention = model.model.layers[index].self_attn
        mode = getattr(attention, "mode", "full")
        # Only this layer's calibration crosses to the fitting card, and it goes back as
        # soon as the layer is done. Six layers at once is what does not fit.
        inputs, keys, values, rotary = (t.to(device) for t in targets[index])
        # `kv_b_proj` is read per head -- `view(..., num_heads, content + head_dim)` then
        # split -- so the target has to interleave each head's content with its own value.
        # Concatenating the two blocks instead puts head 1's content where head 0's value
        # is read, which fits its own target perfectly and is wrong in every forward.
        target = torch.cat([keys.view(-1, heads, content),
                            values.view(-1, heads, head_dim)],
                           dim=-1).reshape(len(keys), heads * (content + head_dim))

        if hasattr(attention, "kv_a_proj"):
            # The encoder is the best rank-`latent` linear summary of everything this
            # layer's readers want -- its own keys and values and theirs. Their halves go
            # in as captured rather than interleaved, because permuting a target's columns
            # permutes rows of `target.T @ inputs` and leaves the right singular vectors
            # alone. The rotary rows are a plain least-squares fit.
            wanted = [target] + [torch.cat(targets[reader][1:3], dim=-1).to(device)
                                 for reader in readers.get(index, ())]
            stacked = torch.cat(wanted, dim=-1) if len(wanted) > 1 else target
            whitener = whiten(inputs.T @ inputs)
            _, _, right = torch.linalg.svd((stacked.T @ inputs) @ whitener,
                                           full_matrices=False)
            del wanted, stacked
            torch.cuda.empty_cache()
            rotary_rows = solve(inputs, rotary)
            with torch.no_grad():
                attention.kv_a_proj.weight.copy_(torch.cat([
                    right[:attention.latent] @ whitener, rotary_rows,
                ]).to(attention.kv_a_proj.weight.dtype))
                attention.kv_a_norm.weight.zero_()
            # The rotary slice is the half of the key that `key_r2` never sees, and it is
            # where one shared vector stands in for the source's per-head ones. Scoring it
            # is what would have shown that the ceiling here is the collapse, not the fit.
            rotary_fit = float(1 - (inputs @ rotary_rows.T - rotary).pow(2).sum()
                               / rotary.pow(2).sum())

        # Whatever latent this layer ends up reading -- its own, or a donor's through the
        # adapter -- take it from the model as it now stands rather than from the algebra.
        handle = _record_latent(attention)
        with torch.no_grad():
            for ids in calibration_windows(args.store, vocab, args.calibrate,
                                           args.length, device):
                model(input_ids=ids, use_cache=False)
        handle.remove()
        latent = torch.cat(attention._latent_log)
        del attention._latent_log

        weight = solve(latent, target)
        fitted = latent @ weight.T
        # Undo the interleave to score the two halves against the shapes they were
        # captured in, so `key_r2` measures the keys the forward will actually assemble.
        parts = fitted.view(len(fitted), heads, content + head_dim)
        fitted_keys = parts[..., :content].reshape(len(fitted), -1)
        fitted_values = parts[..., content:].reshape(len(fitted), -1)
        with torch.no_grad():
            attention.kv_b_proj.weight.copy_(weight.to(attention.kv_b_proj.weight.dtype))
            if attention.k_norm is not None:
                # The legacy content key norm divides the fit's own scale back out, so it
                # has to be handed back as the gain. Without the norm the fit stands.
                scale = fitted_keys.reshape(-1, content).pow(2).mean(-1).sqrt().mean()
                attention.k_norm.weight.fill_(float(scale) - 1.0)

        share = lambda a, b: float(1 - (a - b).pow(2).sum() / b.pow(2).sum())  # noqa: E731
        fits.append((index, mode, share(fitted_keys, keys),
                     share(fitted_values, values), rotary_fit))
        print("%-7d %-9s %9.4f %9.4f %9.4f" % fits[-1], flush=True)
        del latent, fitted, target, inputs, keys, values, rotary
        targets[index] = None
        torch.cuda.empty_cache()

    if not args.no_blend:
        records = model.recipient_initialize()
        print("\nblend: %d sublayers converted, mode %s"
              % (len(records), records[0]["mode"]))

    model.eval()
    after = heldout(model, args.store, vocab, args.evaluate, args.length, device)
    latent_cache = args.mla_latent_dim + rope
    print("\ncache %d -> %d per token per layer (%.2fx), and a borrowing layer caches none"
          % (kv_heads * head_dim * 2, latent_cache,
             kv_heads * head_dim * 2 / latent_cache))
    print("heldout  source %.4f  converted %.4f  cost %+.4f"
          % (before, after, after - before))

    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output, safe_serialization=True)
    (args.output / "conversion.json").write_text(json.dumps({
        "source": str(args.source), "heldout_source": before,
        "heldout_converted": after, "mla_latent_dim": args.mla_latent_dim,
        "csa2_modes": args.csa2_modes, "calibration_tokens": tokens,
        "fits": [{"layer": i, "mode": m, "key_r2": k, "value_r2": v, "rope_r2": r}
                 for i, m, k, v, r in fits],
    }, indent=1), encoding="utf-8")
    print("wrote %s" % args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

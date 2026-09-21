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

from calibration import (LayerMoments, interleaved_columns,  # noqa: E402
                         residual_share, solve_from_moments)
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


def source_capture(stock, full, geometry, readers, store, vocab, count, length, device,
                   heads=None):
    """Second moments of what attention consumed, accumulated window by window.

    This used to concatenate every calibration token's block input, keys, values and
    rotary key in float64 and hand back the samples. At 1024-token windows that is about
    1.6 GiB per full-attention layer and six layers at once, which is why the conversion
    record says `calibration_tokens = 32768` -- the ceiling was memory, not a judgement
    that 32 windows of text are enough to refit a 1.9B model's attention.

    Nothing downstream needs the samples; every quantity the fit computes is a second
    moment, and those are fixed-size in the token count. A 2048-wide input's Gram is
    33 MiB whether it saw 32 thousand rows or 32 million.

    A donor's readers are observed here rather than later because their targets enter the
    donor's SVD against the *donor's* block input, and the two only exist together inside
    one forward.
    """
    rope, content, head_dim, groups = geometry
    moments = {index: LayerMoments(device) for index in full}
    window = {}

    def hook(index):
        def inner(module, args, kwargs):
            states = kwargs.get("hidden_states", args[0] if args else None)
            shape = (*states.shape[:-1], -1, head_dim)
            key = module.k_norm(module.k_proj(states).view(shape))
            value = module.v_proj(states).view(shape)
            keys = (key[..., rope:].repeat_interleave(groups, dim=-2).flatten(-2)
                    .reshape(-1, key.shape[-2] * groups * content).double())
            values = (value.repeat_interleave(groups, dim=-2).flatten(-2)
                      .reshape(-1, value.shape[-2] * groups * head_dim).double())
            window[index] = (
                states.reshape(-1, states.shape[-1]).double(),
                keys,
                values,
                # One shared rotary key against the source's one per key/value head: the
                # mean is the least-squares target when one vector must serve them all.
                key[..., :rope].mean(-2).reshape(-1, rope).double(),
            )
        return inner

    handles = [stock.model.layers[i].self_attn.register_forward_pre_hook(
        hook(i), with_kwargs=True) for i in full]
    with torch.no_grad():
        for ids in calibration_windows(store, vocab, count, length, device):
            window.clear()
            stock(input_ids=ids, use_cache=False)
            for index in full:
                inputs, keys, values, rotary = window[index]
                # `kv_b_proj` is read per head, so the target interleaves each head's
                # content key with its own value. Concatenating the two blocks instead
                # puts head 1's content where head 0's value is read.
                target = torch.cat([keys.view(-1, heads, content),
                                    values.view(-1, heads, head_dim)],
                                   dim=-1).reshape(len(keys), heads * (content + head_dim))
                moments[index].observe(inputs, target, rotary, keys, values)
                for reader in readers.get(index, ()):
                    # As captured rather than interleaved: permuting a target's columns
                    # permutes rows of `target.T @ inputs` and leaves the right singular
                    # vectors alone, so the SVD does not care and this matches what the
                    # sample path fed it.
                    _, reader_keys, reader_values, _ = window[reader]
                    moments[index].observe_reader(
                        reader, torch.cat([reader_keys, reader_values], dim=-1), inputs)
            window.clear()
    for handle in handles:
        handle.remove()
    return moments


def _target_hook(box, geometry, heads):
    """Rebuild one window's interleaved key/value target from the source model.

    The same arithmetic the capture does, kept separate because the second pass needs it
    for one layer at a time rather than for all of them.
    """
    rope, content, head_dim, groups = geometry

    def inner(module, args, kwargs):
        states = kwargs.get("hidden_states", args[0] if args else None)
        shape = (*states.shape[:-1], -1, head_dim)
        key = module.k_norm(module.k_proj(states).view(shape))
        value = module.v_proj(states).view(shape)
        keys = (key[..., rope:].repeat_interleave(groups, dim=-2).flatten(-2)
                .reshape(-1, key.shape[-2] * groups * content).double())
        values = (value.repeat_interleave(groups, dim=-2).flatten(-2)
                  .reshape(-1, value.shape[-2] * groups * head_dim).double())
        box["target"] = torch.cat([keys.view(-1, heads, content),
                                   values.view(-1, heads, head_dim)],
                                  dim=-1).reshape(len(keys), heads * (content + head_dim))
    return inner


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
    # Both stacks cache now. MLA stores the latent and the rotary slice; a CSA2 full layer
    # adds its index keys and a borrowing one stores nothing at all, reading its donor's.
    # The conversion used to force `use_cache = False` here because CSA2 refused a cache
    # outright, which is no longer true of either.
    # A checkpoint trained in this project already names the route, because its arm was
    # built with one. A stock one does not, and `residual_stream_routing` falls back to
    # "widened" -- which `recipient_initialize` refuses, since there is no route for it to
    # convert. Name it here so a stock source converts like any other.
    #
    # The width is stated rather than inherited. A stock config carries no branch count,
    # but it reaches this having been loaded through `Qwen35WidenedForCausalLM`, whose
    # constructor stamps its own defaults onto the config it is handed -- so reading the
    # source for these silently returned 2 branches where every toy arm ran 4, and the 2B
    # was converted narrower than the models its results are compared against.
    config.residual_stream_enabled = True
    config.residual_stream_routing = "flash_next"
    config.residual_stream_num_branches = args.residual_branches
    config.residual_stream_lowrank = args.residual_lowrank
    config.residual_stream_sidecar = False
    config.mla_enabled = True
    config.mla_latent_dim = args.mla_latent_dim
    if args.csa2_modes:
        config.csa2_enabled = True
        config.csa2_modes = list(args.csa2_modes)
        config.csa2_top_k = args.csa2_top_k
        config.csa2_local_window = args.csa2_local_window
        config.csa2_block_size = args.csa2_block_size
        config.csa2_router_bias = not args.no_router_bias
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
    parser.add_argument("--no-router-bias", action="store_true",
                        help="let the indexer decide the selection and nothing else, as "
                             "the reference does, instead of also adding its score to the "
                             "attention logits. The logit fold is what gives a discrete "
                             "top-k a gradient at all, so a model converted this way has "
                             "to be warmed up by indexer_kl.py or its router never moves.")
    parser.add_argument("--residual-branches", type=int, default=4,
                        help="streams in the gated residual. Every toy arm ran 4; the 2B "
                             "conversions before this flag existed ran 2, because the "
                             "value was read from a source config that had been stamped "
                             "with the library default on its way in.")
    parser.add_argument("--residual-lowrank", type=int, default=64,
                        help="rank of the route's read and write projections")
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

    # Which layers will read each donor's latent. A borrowing layer reads the most recent
    # full layer, so a donor's encoder has to summarize its readers' keys and values as
    # well as its own. Spending the whole rank budget on itself is what leaves a borrower
    # at 0.93 of its target where a joint fit reaches 0.97, measured by borrow_sweep.py.
    #
    # Established before the capture, because the capture now accumulates a donor's
    # readers against the donor's own inputs and the two only coexist inside one forward.
    readers, donor = {}, None
    for index in full:
        if getattr(model.model.layers[index].self_attn, "mode", "full") == "full":
            donor, readers[index] = index, []
        elif donor is not None:
            readers[donor].append(index)
    if any(readers.values()):
        print("\ndonors and their readers: %s"
              % ", ".join("%d serves %s" % (d, r) for d, r in readers.items() if r))

    targets = source_capture(stock, full, (rope, content, head_dim, groups), readers,
                             args.store, vocab, args.calibrate, args.length,
                             source_device, heads=heads)
    tokens = next(iter(targets.values())).rows
    print("\ncalibrated on %d tokens per layer" % tokens, flush=True)
    # The source stays loaded now. The up-projection's solve needs the target beside the
    # latent, and the latent only exists once this layer's encoder is set -- so the second
    # pass reads both, one from the converted model and one from here. Holding it costs
    # 3.8 GiB against the 1.6 GiB per layer the samples used to cost, and `--source-device`
    # puts it on the other card.
    torch.cuda.empty_cache()

    print("\n%-7s %-9s %9s %9s %9s"
          % ("layer", "mode", "key r2", "value r2", "rope r2"))
    fits = []
    for index in full:
        attention = model.model.layers[index].self_attn
        mode = getattr(attention, "mode", "full")
        # Moments rather than samples: fixed-size in the token count, so the whole layer's
        # calibration is 33 MiB of Gram instead of 1.6 GiB of rows.
        moments = targets[index]

        if hasattr(attention, "kv_a_proj"):
            # The encoder is the best rank-`latent` linear summary of everything this
            # layer's readers want -- its own keys and values and theirs. The SVD needs
            # only `stacked^T inputs`, which the capture accumulated a block at a time.
            whitener = whiten(moments.gram_inputs)
            _, _, right = torch.linalg.svd(
                moments.stacked_cross(readers.get(index, ())) @ whitener,
                full_matrices=False)
            torch.cuda.empty_cache()
            rotary_rows = solve_from_moments(moments.gram_inputs, moments.cross_rotary,
                                             RIDGE)
            with torch.no_grad():
                attention.kv_a_proj.weight.copy_(torch.cat([
                    right[:attention.latent] @ whitener, rotary_rows,
                ]).to(attention.kv_a_proj.weight.dtype))
                attention.kv_a_norm.weight.zero_()
            # The rotary slice is the half of the key that `key_r2` never sees, and it is
            # where one shared vector stands in for the source's per-head ones. Scoring it
            # is what would have shown that the ceiling here is the collapse, not the fit.
            # Expanded from the moments: tr(R G R^T) - 2 tr(R C) + ||rotary||^2.
            rotary_fit = float(1 - (
                float((rotary_rows @ moments.gram_inputs * rotary_rows).sum())
                - 2 * float((rotary_rows.T * moments.cross_rotary).sum())
                + moments.rotary_energy) / moments.rotary_energy)

        # Whatever latent this layer ends up reading -- its own, or a donor's through the
        # adapter -- take it from the model as it now stands rather than from the algebra.
        # The second pass, streamed the same way: the latent this layer will actually read
        # only exists once the encoder is set, and its moments against the target are all
        # the up-projection's solve needs.
        handle = _record_latent(attention)
        source_handle = stock.model.layers[index].self_attn.register_forward_pre_hook(
            _target_hook(box := {}, (rope, content, head_dim, groups), heads),
            with_kwargs=True)
        with torch.no_grad():
            for ids in calibration_windows(args.store, vocab, args.calibrate,
                                           args.length, device):
                attention._latent_log.clear()
                model(input_ids=ids, use_cache=False)
                stock(input_ids=ids.to(source_device), use_cache=False)
                latent = torch.cat(attention._latent_log)
                moments.observe_latent(latent, box["target"].to(latent.device))
                del latent
                box.clear()
        handle.remove()
        source_handle.remove()
        del attention._latent_log

        weight = solve_from_moments(moments.gram_latent, moments.cross_latent, RIDGE)
        with torch.no_grad():
            attention.kv_b_proj.weight.copy_(weight.to(attention.kv_b_proj.weight.dtype))
            if attention.k_norm is not None:
                # The legacy content key norm needs the fitted rows themselves, which the
                # moment path does not keep. It defaults off and no checkpoint in this
                # project sets it; refuse rather than approximate it silently.
                raise SystemExit(
                    "mla_content_key_norm needs per-row fitted keys, which the streaming "
                    "calibration does not retain. Turn the norm off, or restore the "
                    "sample path for that configuration.")

        key_columns, value_columns = interleaved_columns(heads, content, head_dim)
        fits.append((index, mode,
                     residual_share(weight, moments.gram_latent, moments.cross_latent,
                                    moments.key_energy, key_columns.to(device)),
                     residual_share(weight, moments.gram_latent, moments.cross_latent,
                                    moments.value_energy, value_columns.to(device)),
                     rotary_fit))
        print("%-7d %-9s %9.4f %9.4f %9.4f" % fits[-1], flush=True)
        targets[index] = None
        torch.cuda.empty_cache()

    if not args.no_blend:
        records = model.recipient_initialize()
        print("\nblend: %d sublayers converted, mode %s"
              % (len(records), records[0]["mode"]))

    model.eval()
    after = heldout(model, args.store, vocab, args.evaluate, args.length, device)
    # Ask the layer rather than reassembling MLA's formula here: a CSA2 full layer also has
    # to cache its index keys, so it holds `latent + rope + index_dim` and not the
    # `latent + rope` this used to print for every mode alike.
    caching = [model.model.layers[i].self_attn.cached_numbers_per_token() for i in full]
    latent_cache = max(caching)
    print("\ncache %d -> %d per token per layer (%.2fx); %d of %d layers cache anything"
          % (kv_heads * head_dim * 2, latent_cache,
             kv_heads * head_dim * 2 / latent_cache,
             sum(1 for c in caching if c), len(caching)))
    print("heldout  source %.4f  converted %.4f  cost %+.4f"
          % (before, after, after - before))

    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output, safe_serialization=True)

    # Read the model back and score it again. A conversion that reports a number its own
    # saved checkpoint does not reproduce is worse than one that fails, because every
    # measurement downstream is taken on the file rather than on the object in memory.
    del model
    torch.cuda.empty_cache()
    reloaded = Qwen35WidenedForCausalLM.from_pretrained(
        args.output, dtype=torch.bfloat16).to(device).eval()
    restored = heldout(reloaded, args.store, vocab, args.evaluate, args.length, device)
    print("reloaded %.4f  %s" % (restored,
          "matches" if abs(restored - after) < 0.01 else
          "DOES NOT MATCH the converted model, by %+.4f" % (restored - after)))

    (args.output / "conversion.json").write_text(json.dumps({
        "source": str(args.source), "heldout_source": before,
        "heldout_converted": after, "heldout_reloaded": restored,
        "mla_latent_dim": args.mla_latent_dim,
        "csa2_modes": args.csa2_modes, "calibration_tokens": tokens,
        "fits": [{"layer": i, "mode": m, "key_r2": k, "value_r2": v, "rope_r2": r}
                 for i, m, k, v, r in fits],
    }, indent=1), encoding="utf-8")
    print("wrote %s" % args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

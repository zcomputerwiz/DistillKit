"""Re-solve a trained model's latent projections against the source it was converted from.

The conversion fits each layer's `kv_a_proj` and `kv_b_proj` so that the latent reproduces
what the source model's attention consumed, and it does that with the *source* model's
hidden states as the regressor. That is the only thing available at conversion time and it
is not what the layer ends up seeing: once training moves the model, layer `i` reads its
own drifted residual stream, and the projections were solved for somebody else's.

This re-solves them in place, with two changes from the conversion:

* the regressor is the **trained** model's block input, so the fit is against the
  distribution the layer actually receives;
* only `kv_a_proj` and `kv_b_proj` are touched. `q_proj` and `o_proj` were transferred from
  the source and have been trained since; the router, the MLPs, the norms and the residual
  route keep everything training bought them.

The risk is the mirror image of the one it fixes. The layers downstream adapted to the
attention output the badly-fitted projections were producing, and replacing that output
puts them out of step with it. Whether the better fit is worth the disturbance is a
measurement, not a prediction, which is why this writes a checkpoint to screen rather than
an argument.

It is worth trying because the target matters more than anyone had checked: calibrating
the original conversion on chat instead of Python, at the same 32,768 tokens, was worth
+5.7 MMLU points. A refit that also uses the right input distribution is the same lever
applied twice.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import triton_shim  # noqa: F401,E402
import torch  # noqa: E402

from calibration import (LayerMoments, interleaved_columns,  # noqa: E402
                         residual_share, solve_from_moments)
from convert_full import (RIDGE, calibration_windows, heldout,  # noqa: E402
                          open_split, whiten)
from distillkit.models import Qwen35WidenedForCausalLM  # noqa: E402
from distillkit.models.qwen35.csa2 import Qwen35SparseLatentAttention  # noqa: E402
from transformers.models.qwen3_5.modeling_qwen3_5 import (  # noqa: E402
    Qwen3_5ForCausalLM)


def geometry_of(model):
    """Rope width, content width, head dim and the grouping the source keys expand by."""
    attention = next(layer.self_attn for layer in model.model.layers
                     if isinstance(getattr(layer, "self_attn", None),
                                   Qwen35SparseLatentAttention))
    config = model.config
    groups = config.num_attention_heads // config.num_key_value_heads
    return (attention.rope_dim, attention.content_dim, attention.head_dim, groups,
            config.num_attention_heads)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True,
                        help="the trained, already converted checkpoint to refit")
    parser.add_argument("--source", type=Path, required=True,
                        help="the dense model it was converted from, which supplies the "
                             "keys and values the fit targets")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--store", type=Path,
                        default=Path("scratch/dense_gr/chat-tokens"))
    parser.add_argument("--calibrate", type=int, default=256)
    parser.add_argument("--evaluate", type=int, default=64)
    parser.add_argument("--length", type=int, default=1024)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--source-device", default="cuda:1")
    args = parser.parse_args()

    device = torch.device(args.device)
    source_device = torch.device(args.source_device)
    model = Qwen35WidenedForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16).to(device).eval()
    stock = Qwen3_5ForCausalLM.from_pretrained(
        args.source, dtype=torch.bfloat16).to(source_device).eval()
    vocab = model.config.vocab_size
    rope, content, head_dim, groups, heads = geometry_of(model)

    full = [index for index, layer in enumerate(model.model.layers)
            if isinstance(getattr(layer, "self_attn", None), Qwen35SparseLatentAttention)]
    readers, donor = {}, None
    for index in full:
        if getattr(model.model.layers[index].self_attn, "mode", "full") == "full":
            donor, readers[index] = index, []
        elif donor is not None:
            readers[donor].append(index)

    before = heldout(model, args.store, vocab, args.evaluate, args.length, device)
    print("held-out before the refit: %.4f" % before, flush=True)
    print("refitting %d layers against %s, calibrated on %s"
          % (len(full), args.source, args.store), flush=True)

    moments = {index: LayerMoments(device) for index in full}
    student_window, source_window = {}, {}

    def student_hook(index):
        def inner(module, hook_args, kwargs):
            states = kwargs.get("hidden_states", hook_args[0] if hook_args else None)
            student_window[index] = states.reshape(-1, states.shape[-1]).double()
        return inner

    def source_hook(index):
        def inner(module, hook_args, kwargs):
            states = kwargs.get("hidden_states", hook_args[0] if hook_args else None)
            shape = (*states.shape[:-1], -1, head_dim)
            key = module.k_norm(module.k_proj(states).view(shape))
            value = module.v_proj(states).view(shape)
            source_window[index] = (
                key[..., rope:].repeat_interleave(groups, dim=-2).flatten(-2)
                    .reshape(-1, key.shape[-2] * groups * content).double(),
                value.repeat_interleave(groups, dim=-2).flatten(-2)
                    .reshape(-1, value.shape[-2] * groups * head_dim).double(),
                key[..., :rope].mean(-2).reshape(-1, rope).double(),
            )
        return inner

    def interleave(keys, values):
        return torch.cat([keys.view(-1, heads, content),
                          values.view(-1, heads, head_dim)],
                         dim=-1).reshape(len(keys), heads * (content + head_dim))

    handles = [model.model.layers[i].self_attn.register_forward_pre_hook(
        student_hook(i), with_kwargs=True) for i in full]
    handles += [stock.model.layers[i].self_attn.register_forward_pre_hook(
        source_hook(i), with_kwargs=True) for i in full]
    with torch.no_grad():
        for ids in calibration_windows(args.store, vocab, args.calibrate, args.length,
                                       device):
            student_window.clear()
            source_window.clear()
            model(input_ids=ids, use_cache=False)
            stock(input_ids=ids.to(source_device), use_cache=False)
            for index in full:
                inputs = student_window[index]
                keys, values, rotary = (t.to(device) for t in source_window[index])
                target = interleave(keys, values)
                moments[index].observe(inputs, target, rotary, keys, values)
                for reader in readers.get(index, ()):
                    reader_keys, reader_values, _ = (
                        t.to(device) for t in source_window[reader])
                    moments[index].observe_reader(
                        reader, torch.cat([reader_keys, reader_values], dim=-1), inputs)
    for handle in handles:
        handle.remove()
    print("calibrated on %d tokens per layer" % moments[full[0]].rows, flush=True)

    print("\n%-7s %-9s %9s %9s %9s"
          % ("layer", "mode", "key r2", "value r2", "rope r2"))
    records = []
    for index in full:
        attention = model.model.layers[index].self_attn
        mode = getattr(attention, "mode", "full")
        held = moments[index]
        if hasattr(attention, "kv_a_proj"):
            whitener = whiten(held.gram_inputs)
            _, _, right = torch.linalg.svd(
                held.stacked_cross(readers.get(index, ())) @ whitener,
                full_matrices=False)
            rotary_rows = solve_from_moments(held.gram_inputs, held.cross_rotary, RIDGE)
            with torch.no_grad():
                attention.kv_a_proj.weight.copy_(torch.cat([
                    right[:attention.latent] @ whitener, rotary_rows,
                ]).to(attention.kv_a_proj.weight.dtype))
                # The conversion zeroes this because it fits a fresh encoder. A trained
                # norm is part of what the latent means, so it is left where training put
                # it and the encoder is solved to suit it.
            rotary_fit = float(1 - (
                float((rotary_rows @ held.gram_inputs * rotary_rows).sum())
                - 2 * float((rotary_rows.T * held.cross_rotary).sum())
                + held.rotary_energy) / held.rotary_energy)
        else:
            rotary_fit = float("nan")

        # Second pass for the latent, which only exists now that the encoder is set.
        from convert_full import _record_latent

        handle = _record_latent(attention)
        source_handle = stock.model.layers[index].self_attn.register_forward_pre_hook(
            source_hook(index), with_kwargs=True)
        with torch.no_grad():
            for ids in calibration_windows(args.store, vocab, args.calibrate,
                                           args.length, device):
                attention._latent_log.clear()
                source_window.clear()
                model(input_ids=ids, use_cache=False)
                stock(input_ids=ids.to(source_device), use_cache=False)
                keys, values, _ = (t.to(device) for t in source_window[index])
                held.observe_latent(torch.cat(attention._latent_log),
                                    interleave(keys, values))
        handle.remove()
        source_handle.remove()
        del attention._latent_log

        weight = solve_from_moments(held.gram_latent, held.cross_latent, RIDGE)
        with torch.no_grad():
            attention.kv_b_proj.weight.copy_(weight.to(attention.kv_b_proj.weight.dtype))
        key_columns, value_columns = interleaved_columns(heads, content, head_dim)
        records.append((index, mode,
                        residual_share(weight, held.gram_latent, held.cross_latent,
                                       held.key_energy, key_columns.to(device)),
                        residual_share(weight, held.gram_latent, held.cross_latent,
                                       held.value_energy, value_columns.to(device)),
                        rotary_fit))
        print("%-7d %-9s %9.4f %9.4f %9.4f" % records[-1], flush=True)
        moments[index] = None
        torch.cuda.empty_cache()

    after = heldout(model, args.store, vocab, args.evaluate, args.length, device)
    print("\nheld-out  before %.4f  after %.4f  change %+.4f"
          % (before, after, after - before), flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output)
    (args.output / "refit.json").write_text(json.dumps(
        {"model": str(args.model), "source": str(args.source),
         "store": str(args.store), "calibration_tokens": args.calibrate * args.length,
         "heldout_before": before, "heldout_after": after,
         "fits": [{"layer": r[0], "mode": r[1], "key_r2": r[2], "value_r2": r[3],
                   "rope_r2": r[4]} for r in records]}, indent=2), encoding="utf-8")
    print("wrote %s" % args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

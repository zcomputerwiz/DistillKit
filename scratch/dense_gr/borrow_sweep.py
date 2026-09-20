"""What helps a CSA2 layer read somebody else's latent, swept on the CPU.

A Reindex or Reuse layer owns no down projection. It reads a Full layer's latent through
an adapter and produces its own keys and values from it, so how well that works is a
reduced-rank regression, not a training question -- which makes the design variables
testable without a GPU and without a training run. The toy arms are 44M parameters, so
even the activation capture runs on the CPU in a couple of minutes.

Swept here:

* **donor** -- which earlier Full layer publishes the latent.
* **encoder** -- whether the donor's down projection is fitted to its own keys and values
  (`solo`, which is what `convert_full.py` does today) or to its own *and its borrowers'*
  stacked (`joint`). A donor serving readers has one budget of directions to spend across
  all of them, and fitting it to itself alone spends the budget in the wrong place.
* **latent** -- how many directions there are to spend.
* **metric** -- plain least squares against the weighting that decides the loss: queries
  weight the keys, gated `o_proj` weights the values.

Deliberately not swept: the adapter's rank. `kv_adapt` sits between the donor's normalized
latent and the borrower's up projection with nothing non-linear in between, so adapter and
up projection compose into one linear map and the adapter adds no capacity. It exists to
be trained, not to widen anything.

Also not here: learning rate, warmup, or anything else about the optimizer. Those are not
properties of the fit and cannot be answered without running. What this bounds is how good
a starting point the conversion can hand training in the first place.

    python scratch/dense_gr/borrow_sweep.py --source scratch/dense_gr/checkpoints-arm/smoke-r1-1-nogr
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from distillkit.models import Qwen35WidenedForCausalLM  # noqa: E402

STORE = Path("scratch/code_training/tokens-v2")
RIDGE = 1e-6


def capture(model, full, geometry, store, vocab, count, length, device):
    """Block inputs, per-head content keys and values, and the metric's ingredients.

    The forward runs wherever the model is; everything captured comes straight back to the
    CPU in float64, because the algebra downstream is eigendecompositions and SVDs that
    want the precision more than the throughput, and a real checkpoint's activations do
    not want to sit on the card beside its weights.
    """
    rope, content, head_dim, heads, kv_heads, groups = geometry
    book = {index: [] for index in full}

    def hook(index):
        def inner(module, args, kwargs):
            x = kwargs.get("hidden_states", args[0] if args else None)
            shape = (*x.shape[:-1], -1, head_dim)
            key = module.k_norm(module.k_proj(x).view(shape))
            value = module.v_proj(x).view(shape)
            query, gate = torch.chunk(
                module.q_proj(x).view(*x.shape[:-1], -1, head_dim * 2), 2, dim=-1)
            book[index].append((
                x.reshape(-1, x.shape[-1]).double().cpu(),
                key[..., rope:].repeat_interleave(groups, -2).flatten(-2)
                    .reshape(-1, heads * content).double().cpu(),
                value.repeat_interleave(groups, -2).flatten(-2)
                    .reshape(-1, heads * head_dim).double().cpu(),
                module.q_norm(query.view(shape))[..., rope:]
                    .reshape(-1, heads * content).double().cpu(),
                torch.sigmoid(gate.reshape(-1, heads * head_dim)).double().cpu()))
        return inner

    handles = [model.model.layers[i].self_attn.register_forward_pre_hook(
        hook(i), with_kwargs=True) for i in full]
    from convert_full import open_split

    stream = open_split(store, "train", vocab)
    stride = (len(stream) - length) // max(1, count)
    with torch.no_grad():
        for index in range(count):
            start = index * stride
            ids = np.array(stream[start:start + length], dtype=np.int64).reshape(1, length)
            model(input_ids=torch.from_numpy(ids).to(device), use_cache=False)
    for handle in handles:
        handle.remove()
    return {i: tuple(torch.cat(part) for part in zip(*rows)) for i, rows in book.items()}


def root(matrix, inverse=False):
    values, vectors = torch.linalg.eigh(matrix)
    values = values.clamp_min(values.max().clamp_min(1e-30) * RIDGE)
    values = values.rsqrt() if inverse else values.sqrt()
    return vectors @ torch.diag(values) @ vectors.T


def metric_for(model, index, probe, gate, geometry):
    """Queries weight the keys; the gated output projection weights the values."""
    rope, content, head_dim, heads, kv_heads, groups = geometry
    blocks = []
    for h in range(heads):
        block = probe[:, h * content:(h + 1) * content]
        blocks.append(block.T @ block)
    scale = gate.pow(2).mean(0).sqrt()
    out = model.model.layers[index].self_attn.o_proj.weight.double().cpu()
    for h in range(heads):
        columns = out[:, h * head_dim:(h + 1) * head_dim] * scale[h * head_dim:(h + 1) * head_dim]
        blocks.append(columns.T @ columns)
    return torch.block_diag(*blocks)


def encoder_for(inputs, targets, latent):
    """Best rank-`latent` linear summary of ``inputs`` for reproducing ``targets``."""
    whitener = root(inputs.T @ inputs, inverse=True)
    stacked = torch.cat(targets, dim=-1)
    _, _, right = torch.linalg.svd((stacked.T @ inputs) @ whitener, full_matrices=False)
    return right[:latent] @ whitener


def normalize(raw, eps):
    return raw / raw.pow(2).mean(-1, keepdim=True).add(eps).sqrt()


def explained(latent_values, target, weight):
    """Share of the weighted target energy a least-squares read of the latent reproduces."""
    gram = latent_values.T @ latent_values
    gram = gram + torch.eye(gram.shape[0], dtype=gram.dtype) * (
        gram.diagonal().mean().clamp_min(1e-30) * RIDGE)
    fitted = latent_values @ torch.linalg.solve(gram, latent_values.T @ target)
    residual = fitted - target
    if weight is None:
        return float(1 - residual.pow(2).sum() / target.pow(2).sum())
    return float(1 - torch.trace(residual.T @ residual @ weight)
                 / torch.trace(target.T @ target @ weight))


def assignments(full, book, targets, metrics, eps, args):
    """Score whole mode assignments, not pairs, because a donor can serve several.

    The pairwise sweep fits a donor jointly against one borrower. An assignment that puts
    three borrowers behind one donor asks that donor to spend a single rank budget four
    ways, counting itself, and the pairwise numbers cannot say what that costs. So build
    the real encoder for each assignment -- fitted to the donor and every layer that will
    read it -- and report what each of them gets back.

    The donor's own share is reported beside its borrowers'. A joint fit that carries its
    readers by starving the layer that owns the latent has moved the loss, not removed it.
    """
    latent = args.latents[-1] if len(args.latents) == 1 else args.latents[len(args.latents) // 2]
    print("assignments over %d full-attention layers, latent %d, weighted metric"
          % (len(full), latent))
    print("the first layer is always full: there is nothing behind it to borrow from.\n")

    scored = []
    for bits in itertools.product((True, False), repeat=len(full) - 1):
        modes = (True,) + bits
        if sum(modes) != args.full_count:
            continue
        # Each borrower reads the most recent full layer, which is what `csa2_modes` means
        # by a mode sequence: a donor owns every layer up to the next full one.
        readers, donor = {}, None
        for layer, is_full in zip(full, modes):
            if is_full:
                donor = layer
                readers[layer] = []
            else:
                readers[donor].append(layer)

        shares, donors = {}, {}
        for owner, borrowers in readers.items():
            wanted = [targets[owner]] + [targets[b] for b in borrowers]
            encoder = encoder_for(book[owner][0], wanted, latent)
            seen = normalize(book[owner][0] @ encoder.T, eps)
            donors[owner] = explained(seen, targets[owner], metrics[owner])
            for borrower in borrowers:
                shares[borrower] = explained(seen, targets[borrower], metrics[borrower])
        if not shares:
            continue
        scored.append((min(shares.values()), sum(shares.values()) / len(shares),
                       min(donors.values()), modes, shares, donors))

    scored.sort(reverse=True)
    print("%-34s %8s %8s %8s  %s"
          % ("modes", "worst", "mean", "worst", "per borrower"))
    print("%-34s %8s %8s %8s" % ("", "borrow", "borrow", "donor"))
    for worst, mean, donor_worst, modes, shares, donors in scored:
        label = " ".join("full" if m else "reuse" for m in modes)
        detail = "  ".join("%d:%.4f" % (k, v) for k, v in sorted(shares.items()))
        print("%-34s %8.4f %8.4f %8.4f  %s"
              % (label, worst, mean, donor_worst, detail), flush=True)
    if scored:
        best = scored[0]
        print("\nbest worst-case: %s"
              % " ".join("full" if m else "reuse" for m in best[3]))
        print("  donors keep %s"
              % "  ".join("%d:%.4f" % (k, v) for k, v in sorted(best[5].items())))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=Path("scratch/dense_gr/checkpoints-arm/smoke-r1-1-nogr"))
    parser.add_argument("--windows", type=int, default=8)
    parser.add_argument("--length", type=int, default=1024)
    parser.add_argument("--latents", type=int, nargs="+", default=[64, 128, 192, 256])
    parser.add_argument("--store", type=Path, default=STORE)
    parser.add_argument("--assign", action="store_true",
                        help="score whole mode assignments instead of donor/borrower "
                             "pairs, with each donor's encoder fitted to everything that "
                             "will actually read it.")
    parser.add_argument("--full-count", type=int, default=3,
                        help="how many layers stay full under --assign. The rest borrow, "
                             "and cache nothing.")
    parser.add_argument("--device", default="cpu",
                        help="where the forward runs. The toy fits on the CPU; a real "
                             "checkpoint wants a card, and the captured activations come "
                             "back either way.")
    args = parser.parse_args()

    torch.set_num_threads(max(1, (torch.get_num_threads() or 4)))
    device = torch.device(args.device)
    # bf16 on the card, fp32 on the CPU: a 2B in fp32 is 8 GiB of weights to hold beside
    # activations, and the capture is cast to float64 on arrival regardless.
    model = Qwen35WidenedForCausalLM.from_pretrained(
        args.source,
        dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        attn_implementation="eager").to(device).eval()
    config = model.config
    head_dim = getattr(config, "head_dim",
                       config.hidden_size // config.num_attention_heads)
    rope = int(head_dim * config.rope_parameters["partial_rotary_factor"])
    content = head_dim - rope
    heads, kv_heads = config.num_attention_heads, config.num_key_value_heads
    geometry = (rope, content, head_dim, heads, kv_heads, heads // kv_heads)
    full = [i for i, k in enumerate(config.layer_types) if "linear" not in str(k)]
    print("%s on %s: hidden %d, full attention at %s"
          % (args.source.name, device, config.hidden_size, full), flush=True)

    book = capture(model, full, geometry, args.store, config.vocab_size,
                   args.windows, args.length, device)
    tokens = next(iter(book.values()))[0].shape[0]
    print("captured %d tokens per layer\n" % tokens, flush=True)

    metrics = {i: metric_for(model, i, book[i][3], book[i][4], geometry) for i in full}
    targets = {i: torch.cat([book[i][1], book[i][2]], dim=-1) for i in full}
    eps = config.rms_norm_eps

    print("share of the target a borrower reproduces from a donor's latent")
    print("solo: the donor's encoder fitted to its own keys and values.")
    print("joint: fitted to its own and this borrower's together.\n")
    header = "  ".join("%-13s" % ("latent %d" % v) for v in args.latents)
    print("%-22s %-9s %s" % ("donor -> borrower", "metric", header))
    print("%-22s %-9s %s" % ("", "", "  ".join("%6s %6s" % ("solo", "joint")
                                               for _ in args.latents)))

    if args.assign:
        return assignments(full, book, targets, metrics, eps, args)

    for position, borrower in enumerate(full):
        for donor in full[:position]:
            for label, weight in (("plain", None), ("weighted", metrics[borrower])):
                cells = []
                for latent in args.latents:
                    for mode in ("solo", "joint"):
                        wanted = ([targets[donor]] if mode == "solo"
                                  else [targets[donor], targets[borrower]])
                        encoder = encoder_for(book[donor][0], wanted, latent)
                        seen = normalize(book[donor][0] @ encoder.T, eps)
                        cells.append(explained(seen, targets[borrower], weight))
                print("%-22s %-9s %s"
                      % ("layer %d -> %d" % (donor, borrower), label,
                         "  ".join("%6.4f %6.4f" % (cells[i], cells[i + 1])
                                   for i in range(0, len(cells), 2))), flush=True)
        if position:
            print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

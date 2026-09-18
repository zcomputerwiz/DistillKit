"""Fit the structural sidecar post hoc to a frozen checkpoint, the regime that worked.

`ab68bef` fitted this module to a frozen backbone and bought newline -0.4684, punctuation
-0.2101, whitespace -0.1502, control -0.1852, aggregate -0.0811 and content -0.0053 --
content improving in every arm, with the guardrail never approached. `sidecar_train.py`
trains the same module jointly from scratch instead, and watching that run the content
delta climbs monotonically while held-out stays inside seed noise.

This is the other half of that comparison. Settings follow `fit.py`: lr 3e-3, ten times
the backbone's, because a 150K-parameter module fitted to fixed activations is a different
optimization problem from pretraining; beta 10.0 on the content hinge, which is meaningful
here precisely because the backbone is frozen and ``CE_content(z)`` is a fixed reference;
and AdamW rather than the 8-bit optimizer, since the state is negligible.

Three properties are enforced rather than assumed:

* **The backbone never takes a gradient**, asserted by digesting its parameters before and
  after and comparing. A post-hoc correction is only post hoc if the thing it corrects did
  not move.
* **Strength is calibrated on a split that is neither the fitting data nor the reported
  corpus.** train fits, calibration picks the strength, heldout is reported, and heldout is
  not looked at until the strength is fixed.
* **Per-class NLL is reported**, not just an aggregate, because the whole finding in
  `ab68bef` was that the classes behave differently and an aggregate hides it.

Because the backbone is frozen its hidden states never change, so they are computed once
and reused across every epoch. That is what makes this minutes rather than hours, and it
is also the amortization argument in miniature.

    python scratch/dense_gr/sidecar_fit.py --checkpoint scratch/dense_gr/checkpoints/sc-plain-s0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import triton_shim  # noqa: F401,E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from augmented_head import AugmentedHead  # noqa: E402
from benchmark import apply_liger  # noqa: E402
from distillkit.code_classes import HISTORICAL, code_class_of  # noqa: E402
from distillkit.experimental.ngram_hash import NGramHashConfig, NGramHasher  # noqa: E402
from distillkit.experimental.structural_sidecar import (  # noqa: E402
    FactorizedSidecar, StructuralSidecar)
from distillkit.models import Qwen35WidenedForCausalLM  # noqa: E402
from vocab_remap import build_vocabulary, cached_remap  # noqa: E402

BASE = "D:/DeepThought/Projects/HybridModel/student-2b-hf"
STORE = Path("scratch/code_training/tokens-v2")
CLASSES = ("content", "newline", "whitespace", "punctuation", "control")


def digest(model) -> str:
    hasher = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        hasher.update(name.encode("utf-8"))
        hasher.update(tensor.detach().to(torch.float32).cpu().numpy().tobytes())
    return hasher.hexdigest()


def windows_from(stream, count, length, seed):
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, stream.shape[0] - length - 1, size=count)
    return torch.from_numpy(
        np.stack([stream[s:s + length] for s in starts]).astype(np.int64))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vocab", type=int, default=32_768)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--code-dim", type=int, default=32)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--rows", type=int, default=1 << 20)
    parser.add_argument("--length", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--fit-windows", type=int, default=1024,
                        help="1.05M tokens per epoch against fit.py's 262K. A starved fit "
                             "overfits without generalizing: 33K tokens moved the training "
                             "loss from 1.003 to 0.859 and bought -0.0006 on held-out "
                             "newline, with punctuation going the wrong way")
    parser.add_argument("--calibration-windows", type=int, default=128)
    parser.add_argument("--heldout-windows", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--beta", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--store", type=Path, default=STORE)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.output is None:
        args.output = Path("scratch/dense_gr") / ("fit-%s.json" % args.checkpoint.name)

    started = time.perf_counter()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
    counts = np.load(args.store / "train-counts.npy")
    kept, forward, bytes_ids = build_vocabulary(counts, tokenizer, args.vocab)

    specials = set(tokenizer.get_added_vocab().values())
    labels = [HISTORICAL[code_class_of(tokenizer.decode([int(o)]), int(o) in specials)]
              for o in kept]
    structural = torch.tensor([i for i, c in enumerate(labels) if c != "content"],
                              dtype=torch.long, device="cuda")
    whitespace = torch.tensor([i for i, c in enumerate(labels) if c == "whitespace"],
                              dtype=torch.long, device="cuda")
    class_of = torch.tensor([CLASSES.index(c) for c in labels], dtype=torch.long,
                            device="cuda")

    model = Qwen35WidenedForCausalLM.from_pretrained(
        args.checkpoint, dtype=torch.bfloat16, local_files_only=True).to("cuda")
    apply_liger(model, model.config)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    before = digest(model)
    print("%s: backbone frozen, digest %s" % (args.checkpoint.name, before[:16]),
          flush=True)

    eos = int(forward[tokenizer.eos_token_id]) if tokenizer.eos_token_id is not None \
        and forward[tokenizer.eos_token_id] >= 0 else args.vocab - 1
    hasher = NGramHasher(NGramHashConfig(
        vocab_size=args.vocab, ngram_size=3, heads_per_ngram=1,
        ngram_vocab_size_base=args.rows // 2, seed=1234, eos_token_id=eos))
    addressed = StructuralSidecar(rows=hasher.padded_vocab_size, code_dim=args.code_dim,
                                  structural=int(structural.numel()), mode="fixed",
                                  heads=args.heads, seed=args.seed)
    sidecar = FactorizedSidecar(addressed, structural.cpu(), whitespace.cpu(),
                                gated=True).to(device="cuda", dtype=torch.float32)
    augmented = AugmentedHead(sidecar, structural, args.vocab, args.hidden).to("cuda")
    print("sidecar: %s | latent width %d"
          % (json.dumps(sidecar.parameter_report()), augmented.width), flush=True)

    split = {}
    for name, count, seed in (("train", args.fit_windows, 11),
                              ("calibration", args.calibration_windows, 22),
                              ("heldout", args.heldout_windows, 33)):
        source = "train" if name == "train" else name
        stream, _, _ = cached_remap(args.store, source, args.vocab, tokenizer, forward,
                                    bytes_ids, counts, kept, verbose=False)
        split[name] = windows_from(np.asarray(stream), count, args.length, seed)

    @torch.no_grad()
    def cache_states(tokens):
        """Hidden states and rows for a frozen backbone, computed once."""
        states, rows = [], []
        for start in range(0, tokens.shape[0], args.batch):
            chunk = tokens[start:start + args.batch].to("cuda")
            states.append(model.model(input_ids=chunk,
                                      attention_mask=torch.ones_like(chunk),
                                      use_cache=False).last_hidden_state.clone())
            rows.append(hasher.row_indices(chunk))
        return states, rows

    cached = {name: cache_states(tokens) for name, tokens in split.items()}
    print("cached hidden states for %s"
          % ", ".join("%s %d" % (k, v.shape[0]) for k, v in split.items()), flush=True)

    head = model.lm_head.weight

    def scored(name, index, strength):
        """Per-position NLL with and without the bias, plus the target's class."""
        states, rows = cached[name]
        tokens = split[name][index * args.batch:(index + 1) * args.batch].to("cuda")
        state, row = states[index], rows[index]
        targets = tokens[:, 1:]
        plain_logits = (state[:, :-1].float() @ head.float().T)
        code = augmented.code_for(row)[:, :-1].float()
        bias = code @ augmented.s_matrix(torch.float32).T
        biased_logits = plain_logits + strength * bias
        flat_t = targets.reshape(-1)
        biased = F.cross_entropy(biased_logits.reshape(-1, args.vocab), flat_t,
                                 reduction="none")
        plain = F.cross_entropy(plain_logits.reshape(-1, args.vocab), flat_t,
                                reduction="none")
        return biased, plain, class_of[flat_t]

    optimizer = torch.optim.AdamW(
        [p for p in sidecar.parameters() if p.requires_grad], lr=args.lr)
    batches = len(cached["train"][0])
    history = []
    for epoch in range(args.epochs):
        losses = []
        for index in range(batches):
            biased, plain, klass = scored("train", index, 1.0)
            is_content = klass == 0
            if not is_content.any() or is_content.all():
                continue
            loss = biased[~is_content].mean() + args.beta * torch.relu(
                biased[is_content].mean() - plain[is_content].mean())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        history.append(sum(losses) / max(len(losses), 1))
        print("epoch %d  loss %.5f" % (epoch, history[-1]), flush=True)

    @torch.no_grad()
    def per_class(name, strength):
        totals = torch.zeros(len(CLASSES), device="cuda")
        base = torch.zeros(len(CLASSES), device="cuda")
        seen = torch.zeros(len(CLASSES), device="cuda")
        for index in range(len(cached[name][0])):
            biased, plain, klass = scored(name, index, strength)
            totals.index_add_(0, klass, biased)
            base.index_add_(0, klass, plain)
            seen.index_add_(0, klass, torch.ones_like(biased))
        seen = seen.clamp(min=1)
        return {CLASSES[i]: {"biased": float(totals[i] / seen[i]),
                             "plain": float(base[i] / seen[i]),
                             "delta": float((totals[i] - base[i]) / seen[i]),
                             "tokens": int(seen[i])} for i in range(len(CLASSES))}

    # Strength on calibration only. heldout is untouched until this is fixed.
    grid = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    calibration = {}
    for strength in grid:
        report = per_class("calibration", strength)
        structural_delta = sum(report[c]["delta"] * report[c]["tokens"]
                               for c in CLASSES if c != "content")
        structural_tokens = sum(report[c]["tokens"] for c in CLASSES if c != "content")
        calibration[strength] = {
            "structural": structural_delta / max(structural_tokens, 1),
            "content": report["content"]["delta"]}
        calibration[strength]["guarded"] = (
            calibration[strength]["structural"]
            + args.beta * max(0.0, calibration[strength]["content"]))
        print("  strength %.2f  structural %+.4f  content %+.4f  guarded %+.4f"
              % (strength, calibration[strength]["structural"],
                 calibration[strength]["content"],
                 calibration[strength]["guarded"]), flush=True)
    # Selected on the same guarded objective the fit optimizes. Picking on structural
    # alone would let a strength that violates the content constraint win, which is the
    # trade the guardrail exists to forbid.
    chosen = min(grid, key=lambda s: calibration[s]["guarded"])
    print("chosen strength %.2f (guarded %+.4f)"
          % (chosen, calibration[chosen]["guarded"]), flush=True)

    result = per_class("heldout", chosen)
    after = digest(model)
    print()
    print("%-12s %10s %10s %10s %10s" % ("class", "plain", "biased", "delta", "tokens"))
    for name in CLASSES:
        row = result[name]
        print("%-12s %10.4f %10.4f %+10.4f %10d"
              % (name, row["plain"], row["biased"], row["delta"], row["tokens"]))
    print()
    print("backbone unchanged: %s" % (before == after))

    args.output.write_text(json.dumps({
        "checkpoint": str(args.checkpoint), "strength": chosen,
        "lr": args.lr, "beta": args.beta, "epochs": args.epochs,
        "fit_windows": args.fit_windows, "length": args.length,
        "fit_tokens": args.fit_windows * args.length * args.epochs,
        "sidecar": sidecar.parameter_report(), "latent_width": augmented.width,
        "loss_history": history, "calibration": {str(k): v for k, v in
                                                 calibration.items()},
        "heldout": result, "backbone_unchanged": before == after,
        "backbone_digest": before, "seconds": time.perf_counter() - started,
    }, indent=2), encoding="utf-8")
    if before != after:
        raise SystemExit("the backbone moved; this is not a post-hoc fit")

    # The weights, not just the numbers. A fit that cannot be reloaded has to be redone to
    # ask it anything new, which is the same mistake the baseline run made by saving no
    # checkpoint. Everything needed to rebuild the module and place it is stored with it.
    weights = args.output.with_suffix(".pt")
    torch.save({"state_dict": {k: v.cpu() for k, v in sidecar.state_dict().items()},
                "strength": chosen, "rows": args.rows, "padded_rows":
                    hasher.padded_vocab_size, "code_dim": args.code_dim,
                "heads": args.heads, "seed": args.seed, "vocab": args.vocab,
                "hidden": args.hidden, "latent_width": augmented.width,
                "structural": structural.cpu(), "whitespace": whitespace.cpu(),
                "hasher": {"vocab_size": args.vocab, "ngram_size": 3,
                           "heads_per_ngram": 1,
                           "ngram_vocab_size_base": args.rows // 2,
                           "seed": 1234, "eos_token_id": eos},
                "checkpoint": str(args.checkpoint), "backbone_digest": before},
               weights)
    print("wrote %s and %s" % (args.output, weights))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

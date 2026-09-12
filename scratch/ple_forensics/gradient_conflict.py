"""Does predicting layout cost the backbone capacity that content could have used?

The offload hypothesis is that if a cheap external memory takes over repetitive
structural prediction, the backbone stops spending gradient on it and can put that
capacity into content. That is a different claim from "the sidecar predicts content
better", which everything measured so far says it does not.

It is also falsifiable before any training. Split the loss by token class and ask, per
layer, two things about the parameter gradients:

    R_l = E||grad_l L_layout||^2 / E||grad_l L_content||^2
    A_l = E cos(grad_l L_layout, grad_l L_content)

`R_l` is what fraction of the layer's gradient energy layout is consuming. `A_l` is
whether the two objectives want to move the same weights the same way. The hypothesis
needs layout to be consuming a non-negligible share *and* to be pulling somewhere
content is not: if `R_l` is tiny there is nothing to free, and if `A_l` is near 1 the
layout objective is already doing content's work and removing it takes that away too.

Parameter gradients, not activation gradients, are the right object here. Parameters are
shared across positions, so `cos(grad L_layout, grad L_content)` is exactly "do these two
objectives want to move the same weights in the same direction". Activation gradients
live at different positions for the two classes and have no natural pairing.

Gradients are taken against a *sample* of each layer's parameters -- the attention output
projection and the MLP down projection -- rather than all 4B, which would not fit three
times over. Both are the output-side matrices of their sublayer, so a conflict that shows
up in them is a conflict in the layer. This is a sample and is labelled as one.

    python scratch/ple_forensics/gradient_conflict.py --documents 48
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from torch.nn import functional as F

from scratch.ple_forensics.token_classes import CLASSES, class_of

STUDENT = "D:/DeepThought/Projects/HybridModel/student-hf"
BUNDLE = Path("scratch/independent-eval/reply-bundle-384.json")
OUT = Path("scratch/ple_forensics")

#: Which classes get their own loss. `lexical` is the content reference every ratio is
#: taken against; `punctuation` is measured but is not a candidate for offloading.
PROBED = ("whitespace", "control", "punctuation", "lexical")


def sampled_parameters(model):
    """One attention and one MLP output matrix per decoder layer."""
    selected = []
    for index, layer in enumerate(model.model.layers):
        for path in ("self_attn.o_proj.weight", "mlp.down_proj.weight"):
            module = layer
            for part in path.split(".")[:-1]:
                module = getattr(module, part, None)
                if module is None:
                    break
            if module is None:
                continue
            parameter = getattr(module, path.split(".")[-1], None)
            if parameter is not None:
                selected.append((index, path, parameter))
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", type=int, default=48)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=OUT / "gradient-conflict.json")
    args = parser.parse_args()

    from transformers import AutoTokenizer
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(STUDENT, local_files_only=True)
    bundle = json.loads(BUNDLE.read_text(encoding="utf-8"))
    records = bundle["splits"]["screen"]["nll"][:args.documents]

    model = Qwen3_5ForCausalLM.from_pretrained(STUDENT, dtype=torch.bfloat16,
                                               local_files_only=True)
    model.config.use_cache = False
    model.to(args.device).eval()
    # Frozen in the sense that nothing is updated; gradients still have to exist to be
    # measured, so the sampled matrices are the only things that require them.
    model.requires_grad_(False)
    sampled = sampled_parameters(model)
    for _, _, parameter in sampled:
        parameter.requires_grad_(True)
    parameters = [parameter for _, _, parameter in sampled]
    print("sampling %d matrices across %d layers"
          % (len(sampled), len({index for index, _, _ in sampled})))

    # Accumulated across documents: squared norms per class, and the dot products that
    # make the cosines. Accumulating dot products rather than per-document cosines keeps
    # the estimate weighted by gradient magnitude instead of by document count.
    energy = {name: np.zeros(len(sampled)) for name in PROBED}
    dot = {name: np.zeros(len(sampled)) for name in PROBED if name != "lexical"}
    counts = {name: 0 for name in PROBED}
    # Token shares, because a class gradient taken as a *mean* over its own tokens is not
    # that class's contribution to the update. The full-batch gradient is
    # sum_c (n_c / N) grad_c, so a rare class looks large per unit of objective purely
    # because averaging over few tokens cancels less. Both normalisations are reported.
    tokens = {name: 0 for name in PROBED}
    scored = 0

    for number, record in enumerate(records):
        ids = record["ids"]
        assistant = [i for low, high in record["roles"].get("assistant", [])
                     for i in range(max(low, 1), min(high, len(ids)))]
        if not assistant:
            continue
        targets = np.array([ids[i] for i in assistant])
        labels = class_of(targets, tokenizer)
        batch = {"input_ids": torch.tensor([ids], device=args.device),
                 "attention_mask": torch.ones(1, len(ids), dtype=torch.long,
                                              device=args.device)}
        # Only the scored positions, not the whole window: a 248,320-wide logit row is
        # 1 MB in fp32 and the graph has to stay alive across several backward passes.
        positions = torch.as_tensor(np.array(assistant) - 1, device=args.device)
        logits = model(**batch, logits_to_keep=positions).logits[0].float()
        target = torch.as_tensor(targets, device=args.device)
        per_token = F.cross_entropy(logits, target, reduction="none")

        def gradient_of(name, keep_graph):
            mask = labels == name
            if not mask.any():
                return None
            # Mean within the class, so a class with more tokens does not get a larger
            # gradient purely by count -- the question is energy per unit of objective.
            tokens[name] += int(mask.sum())
            rows = torch.as_tensor(np.flatnonzero(mask), device=args.device)
            grad = torch.autograd.grad(per_token[rows].mean(), parameters,
                                       retain_graph=keep_graph, allow_unused=True)
            # Off the GPU as each one is produced. Two full gradient sets over the
            # sampled matrices are ~7.7 GB in fp32 and do not fit beside an 8 GB model
            # plus a retained graph; host RAM is not scarce here.
            pieces = []
            for slot, (piece, parameter) in enumerate(zip(grad, parameters)):
                moved = (torch.zeros(parameter.shape, dtype=torch.float32)
                         if piece is None else piece.detach().to("cpu", torch.float32))
                energy[name][slot] += float(moved.pow(2).sum())
                pieces.append(moved)
            del grad
            torch.cuda.empty_cache()
            counts[name] += 1
            return pieces

        # The content gradient is held, and each other class is computed against it one
        # at a time and released. Holding all four at once is 4 x 3.8 GB over the sampled
        # matrices and does not fit beside the model.
        content = gradient_of("lexical", keep_graph=True)
        others = [name for name in PROBED if name != "lexical"]
        for order, name in enumerate(others):
            pieces = gradient_of(name, keep_graph=order < len(others) - 1)
            if pieces is None or content is None:
                del pieces
                continue
            for slot, (left, right) in enumerate(zip(pieces, content)):
                denominator = float(left.norm()) * float(right.norm())
                if denominator > 0:
                    dot[name][slot] += float((left * right).sum()) / denominator
            del pieces
            torch.cuda.empty_cache()
        del content, logits, per_token
        torch.cuda.empty_cache()
        scored += 1
        if (number + 1) % 8 == 0:
            print("  %d/%d documents" % (number + 1, len(records)), flush=True)

    print("\n%d documents contributed; class appearances: %s"
          % (scored, {k: v for k, v in counts.items()}))

    layers = sorted({index for index, _, _ in sampled})
    total_tokens = sum(tokens.values())
    share = {name: tokens[name] / max(total_tokens, 1) for name in PROBED}
    print("token shares: %s" % {name: round(value, 4) for name, value in share.items()})
    report = {"documents": scored, "sampled_matrices": len(sampled), "layers": layers,
              "tokens": tokens, "share": share, "per_layer": {}}
    print("\nR_l = layout energy / content energy, A_l = cos(layout grad, content grad)")
    print("  %-6s %-28s %-28s %s" % ("layer", "whitespace  R / A", "control  R / A",
                                     "punctuation  R / A"))
    for position, layer in enumerate(layers):
        slots = [i for i, (index, _, _) in enumerate(sampled) if index == layer]
        content_energy = energy["lexical"][slots].sum() / max(counts["lexical"], 1)
        cells, record_row = [], {}
        for name in ("whitespace", "control", "punctuation"):
            if not counts[name]:
                cells.append("%-28s" % "n/a")
                continue
            ratio = (energy[name][slots].sum() / counts[name]) / max(content_energy, 1e-30)
            align = dot[name][slots].sum() / (len(slots) * max(counts[name], 1))
            record_row[name] = {"R": ratio, "A": align}
            cells.append("%-28s" % ("%8.4f / %+7.4f" % (ratio, align)))
        report["per_layer"][str(layer)] = record_row
        if position % 4 == 0 or layer == layers[-1]:
            print("  %-6d %s %s %s" % (layer, *cells))

    print("\nwhole-network summary over the sampled matrices")
    print("  R_obj    gradient energy per unit of that class's own mean loss")
    print("  R_share  the same, weighted by token share -- what the class actually")
    print("           contributes to the full-batch update")
    print("  A        cosine with the content gradient\n")
    summary = {}
    content_energy = energy["lexical"].sum() / max(counts["lexical"], 1)
    for name in ("whitespace", "control", "punctuation"):
        if not counts[name]:
            continue
        ratio = (energy[name].sum() / counts[name]) / max(content_energy, 1e-30)
        weighted = ratio * (share[name] / max(share["lexical"], 1e-30)) ** 2
        align = dot[name].sum() / (len(sampled) * counts[name])
        summary[name] = {"R_obj": ratio, "R_share": weighted, "A": align}
        print("  %-12s R_obj = %9.4f   R_share = %8.4f   A = %+7.4f"
              % (name, ratio, weighted, align))
    report["summary"] = summary

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nwrote %s" % args.output)

    verdict = summary.get("whitespace", {})
    floor = 1.0 / np.sqrt(sum(p.numel() for p in parameters))
    print("\ngate: the offload hypothesis needs layout to consume real capacity AND to")
    print("pull somewhere content does not.")
    print("  whitespace R_share = %.4f of content's contribution to the update"
          % verdict.get("R_share", float("nan")))
    print("  whitespace A       = %+.4f, against %.6f for random vectors in this many"
          % (verdict.get("A", float("nan")), floor))
    print("                       dimensions. Near zero is orthogonal, which is neither")
    print("                       conflict (negative) nor redundancy (near one).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

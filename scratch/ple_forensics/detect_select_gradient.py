"""Is whitespace *selection* a meaningful share of the update, or only of the loss?

The router check factored the whitespace loss exactly,

    -log P(w) = -log P(WS) + -log P(w | WS)
                 detection      selection

and found selection is 82.7% of the nats. That is an NLL share, and arm B removes a
gradient, not a loss. The two can diverge badly: a term can dominate the loss while
contributing little to the update, or the reverse. Before any training, confirm that
selection is a meaningful share of the gradient energy over the exact window arms A and B
will train -- decoder layers 20 to 28, the tuned A3 window.

All parameters in the window are used rather than the sampled matrices of
`gradient_conflict.py`, because this is the set that actually moves. Each gradient is
moved to host RAM as it is produced; three sets over 1.1B parameters is 13.6 GB in fp32,
which does not fit on the GPU beside the model but is nothing for this machine.

Losses, all means over their own positions:

``detect``   -log P(WS)                     at whitespace targets
``select``   -log P(w | WS)                 at whitespace targets
``content``  full-vocabulary CE             at non-whitespace targets

    python scratch/ple_forensics/detect_select_gradient.py --documents 48
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

from scratch.ple_forensics.router_check import whitespace_vocabulary
from scratch.ple_forensics.token_classes import class_of

STUDENT = "D:/DeepThought/Projects/HybridModel/student-hf"
BUNDLE = Path("scratch/independent-eval/reply-bundle-384.json")
WINDOW = (20, 28)
TERMS = ("detect", "select", "content")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", type=int, default=48)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--window", type=int, nargs=2, default=WINDOW)
    parser.add_argument("--output", type=Path,
                        default=Path("scratch/ple_forensics/detect-select-gradient.json"))
    args = parser.parse_args()

    from transformers import AutoTokenizer
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(STUDENT, local_files_only=True)
    model = Qwen3_5ForCausalLM.from_pretrained(STUDENT, dtype=torch.bfloat16,
                                               local_files_only=True)
    model.config.use_cache = False
    model.to(args.device).eval()
    model.requires_grad_(False)

    low, high = args.window
    parameters = []
    for index in range(low, high + 1):
        parameters += list(model.model.layers[index].parameters())
    for parameter in parameters:
        parameter.requires_grad_(True)
    total = sum(p.numel() for p in parameters)
    print("window layers %d-%d: %d tensors, %.2fB parameters"
          % (low, high, len(parameters), total / 1e9))

    whitespace = torch.as_tensor(whitespace_vocabulary(tokenizer, model.config.vocab_size),
                                 device=args.device)
    records = json.loads(BUNDLE.read_text(encoding="utf-8"))["splits"]["screen"]["nll"]
    records = records[:args.documents]

    energy = {name: 0.0 for name in TERMS}
    counts = {name: 0 for name in TERMS}
    tokens = {name: 0 for name in TERMS}
    cosine = {"select_vs_content": 0.0, "detect_vs_content": 0.0, "select_vs_detect": 0.0}
    pairs = {key: 0 for key in cosine}

    for number, record in enumerate(records):
        ids = record["ids"]
        assistant = [i for lo, hi in record["roles"].get("assistant", [])
                     for i in range(max(lo, 1), min(hi, len(ids)))]
        if not assistant:
            continue
        targets = np.array([ids[i] for i in assistant])
        is_whitespace = class_of(targets, tokenizer) == "whitespace"
        if not is_whitespace.any() or is_whitespace.all():
            continue
        positions = torch.as_tensor(np.array(assistant) - 1, device=args.device)
        logits = model(input_ids=torch.tensor([ids], device=args.device),
                       attention_mask=torch.ones(1, len(ids), dtype=torch.long,
                                                 device=args.device),
                       logits_to_keep=positions).logits[0].float()
        target = torch.as_tensor(targets, device=args.device)
        rows_ws = torch.as_tensor(np.flatnonzero(is_whitespace), device=args.device)
        rows_other = torch.as_tensor(np.flatnonzero(~is_whitespace), device=args.device)

        everything = torch.logsumexp(logits, dim=-1)
        within = torch.logsumexp(logits[:, whitespace], dim=-1)
        picked = logits.gather(1, target.unsqueeze(1)).squeeze(1)
        losses = {
            # -log P(WS): the class mass the router already gets nearly right.
            "detect": (everything - within)[rows_ws].mean(),
            # -log P(w | WS): the conditional inside the whitespace set, which is the
            # only part an expert can take over.
            "select": (within - picked)[rows_ws].mean(),
            "content": F.cross_entropy(logits[rows_other], target[rows_other]),
        }
        tokens["detect"] += len(rows_ws)
        tokens["select"] += len(rows_ws)
        tokens["content"] += len(rows_other)

        gradients = {}
        for order, name in enumerate(TERMS):
            grad = torch.autograd.grad(losses[name], parameters,
                                       retain_graph=order < len(TERMS) - 1,
                                       allow_unused=True)
            pieces = []
            for piece, parameter in zip(grad, parameters):
                moved = (torch.zeros(parameter.shape, dtype=torch.float32)
                         if piece is None else piece.detach().to("cpu", torch.float32))
                energy[name] += float(moved.pow(2).sum())
                pieces.append(moved)
            del grad
            torch.cuda.empty_cache()
            counts[name] += 1
            gradients[name] = pieces

        for key, (left, right) in (("select_vs_content", ("select", "content")),
                                   ("detect_vs_content", ("detect", "content")),
                                   ("select_vs_detect", ("select", "detect"))):
            dot = sum(float((a * b).sum())
                      for a, b in zip(gradients[left], gradients[right]))
            norms = (np.sqrt(sum(float(a.pow(2).sum()) for a in gradients[left]))
                     * np.sqrt(sum(float(b.pow(2).sum()) for b in gradients[right])))
            if norms > 0:
                cosine[key] += dot / norms
                pairs[key] += 1
        del gradients, logits
        torch.cuda.empty_cache()
        if (number + 1) % 8 == 0:
            print("  %d/%d" % (number + 1, len(records)), flush=True)

    scored = counts["content"]
    print("\n%d documents; tokens: %s" % (scored, tokens))
    grand = tokens["content"] + tokens["select"]
    share = {"detect": tokens["detect"] / grand, "select": tokens["select"] / grand,
             "content": tokens["content"] / grand}

    print("\ngradient energy over the window that A and B will train")
    print("  %-10s %14s %14s %10s" % ("term", "per objective", "share-weighted", "tokens"))
    summary = {}
    reference = energy["content"] / max(counts["content"], 1)
    for name in TERMS:
        if not counts[name]:
            continue
        per_objective = (energy[name] / counts[name]) / max(reference, 1e-30)
        weighted = per_objective * (share[name] / share["content"]) ** 2
        summary[name] = {"per_objective": per_objective, "share_weighted": weighted,
                         "tokens": tokens[name]}
        print("  %-10s %14.4f %14.4f %10d" % (name, per_objective, weighted, tokens[name]))

    whitespace_total = summary["detect"]["share_weighted"] + summary["select"]["share_weighted"]
    select_share = summary["select"]["share_weighted"] / max(whitespace_total, 1e-30)
    print("\n  of the whitespace contribution to the update, selection is %.1f%%"
          % (100 * select_share))
    print("  (the NLL split put selection at 82.7%%)")

    print("\ngradient cosines")
    for key in cosine:
        print("  %-20s %+.4f" % (key, cosine[key] / max(pairs[key], 1)))

    report = {"window": [low, high], "parameters": int(total), "documents": scored,
              "tokens": tokens, "share": share, "summary": summary,
              "select_share_of_whitespace_update": float(select_share),
              "cosine": {k: cosine[k] / max(pairs[k], 1) for k in cosine}}
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nwrote %s" % args.output)

    print("\ngate: arm B removes the selection gradient. That is worth doing only if")
    print("selection is a real share of what the window is spending.")
    print("  selection share-weighted energy, against content: %.4f"
          % summary["select"]["share_weighted"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Can a context-only router send whitespace to an expert, and is there work to send?

Arm D needs `P = pi(context) P_expert + (1 - pi(context)) P_backbone` over the full
vocabulary. Routing on the ground-truth class would leak the answer, and a
reduced-vocabulary head under oracle routing does not produce an NLL comparable to arm
A's. So `pi` has to be predictable from context, and the cheapest candidate costs nothing
to build: the frozen backbone's own whitespace mass,

    pi(context) = sum_{w in whitespace} P_backbone(w | context)

Two things get measured in one pass, and the second is the sharper gate.

**Is pi predictable?** AUC and calibration of that mass against the actual class. If the
backbone cannot tell whitespace is coming, no context-only router can be built from it.

**Is there anything to offload?** The NLL at a whitespace position factors exactly:

    -log P(w) = -log P(whitespace) + -log P(w | whitespace)
                 \_____ detection ____/   \_____ selection ____/

An expert can take over *selection* -- which whitespace token, given that one is coming.
It cannot take over detection, because the router has to do that anyway and the router is
the backbone. So if whitespace NLL is nearly all detection, arm D frees almost nothing
even if every other part works, and the 21.7% of update energy whitespace consumes is
mostly spent on a job that cannot be handed over.

    python scratch/ple_forensics/router_check.py --documents 64
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

from scratch.ple_forensics.token_classes import class_of

STUDENT = "D:/DeepThought/Projects/HybridModel/student-hf"
BUNDLE = Path("scratch/independent-eval/reply-bundle-384.json")
CACHE = Path("scratch/ple_forensics/whitespace-vocab.npy")


def whitespace_vocabulary(tokenizer, vocab_size):
    """Every id in the whole vocabulary whose text is non-empty and all whitespace.

    Over the vocabulary, not over observed targets: `pi` is a sum over everything the
    expert could emit, so a whitespace token absent from this sample still belongs in it.
    """
    if CACHE.exists():
        return np.load(CACHE)
    found = []
    step = 4096
    for start in range(0, vocab_size, step):
        ids = list(range(start, min(start + step, vocab_size)))
        for token_id, text in zip(ids, tokenizer.batch_decode([[i] for i in ids])):
            if text and not text.strip():
                found.append(token_id)
    found = np.array(sorted(found), dtype=np.int64)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.save(CACHE, found)
    return found


def auc(scores, labels):
    labels = np.asarray(labels, dtype=bool)
    positives, negatives = labels.sum(), (~labels).sum()
    if not positives or not negatives:
        return float("nan")
    order = np.argsort(np.asarray(scores, dtype=np.float64), kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    return (ranks[labels].sum() - positives * (positives + 1) / 2) / (positives * negatives)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path,
                        default=Path("scratch/ple_forensics/router-check.json"))
    args = parser.parse_args()

    from transformers import AutoTokenizer
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(STUDENT, local_files_only=True)
    model = Qwen3_5ForCausalLM.from_pretrained(STUDENT, dtype=torch.bfloat16,
                                               local_files_only=True)
    model.config.use_cache = False
    model.to(args.device).eval()

    vocab_size = model.config.vocab_size
    whitespace = whitespace_vocabulary(tokenizer, vocab_size)
    print("%d whitespace token types in the full %d-token vocabulary"
          % (len(whitespace), vocab_size))
    whitespace_index = torch.as_tensor(whitespace, device=args.device)

    records = json.loads(BUNDLE.read_text(encoding="utf-8"))["splits"]["screen"]["nll"]
    records = records[:args.documents]

    mass, is_whitespace, detection, selection, full = [], [], [], [], []
    for number, record in enumerate(records):
        ids = record["ids"]
        assistant = [i for low, high in record["roles"].get("assistant", [])
                     for i in range(max(low, 1), min(high, len(ids)))]
        if not assistant:
            continue
        positions = torch.as_tensor(np.array(assistant) - 1, device=args.device)
        logits = model(input_ids=torch.tensor([ids], device=args.device),
                       attention_mask=torch.ones(1, len(ids), dtype=torch.long,
                                                 device=args.device),
                       logits_to_keep=positions).logits[0].float()
        log_probabilities = F.log_softmax(logits, dim=-1)
        target = torch.as_tensor([ids[i] for i in assistant], device=args.device)

        # log pi: the mass the backbone already puts on the whitespace class.
        class_log = torch.logsumexp(log_probabilities[:, whitespace_index], dim=-1)
        target_log = log_probabilities.gather(1, target.unsqueeze(1)).squeeze(1)

        labels = class_of(np.array([ids[i] for i in assistant]), tokenizer) == "whitespace"
        mass.append(class_log.exp().cpu().numpy())
        is_whitespace.append(labels)
        full.append((-target_log).cpu().numpy())
        # Only defined where the target actually is whitespace.
        detection.append((-class_log).cpu().numpy())
        selection.append((-(target_log - class_log)).cpu().numpy())
        del logits, log_probabilities
        if (number + 1) % 16 == 0:
            print("  %d/%d" % (number + 1, len(records)), flush=True)

    mass = np.concatenate(mass)
    is_whitespace = np.concatenate(is_whitespace)
    detection = np.concatenate(detection)
    selection = np.concatenate(selection)
    full = np.concatenate(full)

    print("\n%d assistant positions, %d whitespace (%.1f%%)"
          % (len(mass), is_whitespace.sum(), 100 * is_whitespace.mean()))

    print("\nrouter predictability: backbone whitespace mass against the actual class")
    print("  AUC                        %.4f" % auc(mass, is_whitespace))
    print("  mean pi at whitespace      %.4f" % mass[is_whitespace].mean())
    print("  mean pi elsewhere          %.4f" % mass[~is_whitespace].mean())
    print("  median pi at whitespace    %.4f" % np.median(mass[is_whitespace]))
    print("  median pi elsewhere        %.4f" % np.median(mass[~is_whitespace]))
    for threshold in (0.5, 0.8, 0.9):
        fired = mass >= threshold
        if fired.sum():
            print("  pi >= %.1f: fires on %5.1f%% of positions, %5.1f%% precision, "
                  "%5.1f%% of whitespace recalled"
                  % (threshold, 100 * fired.mean(),
                     100 * is_whitespace[fired].mean(),
                     100 * (fired & is_whitespace).sum() / max(is_whitespace.sum(), 1)))

    whitespace_detection = detection[is_whitespace]
    whitespace_selection = selection[is_whitespace]
    whitespace_full = full[is_whitespace]
    share = whitespace_detection.sum() / max(whitespace_full.sum(), 1e-9)
    print("\nwhat the whitespace NLL is made of (only selection can be offloaded)")
    print("  total whitespace NLL       %8.2f nats over %d tokens"
          % (whitespace_full.sum(), is_whitespace.sum()))
    print("  detection  -log P(ws)      %8.2f nats  %5.1f%%"
          % (whitespace_detection.sum(), 100 * share))
    print("  selection  -log P(w|ws)    %8.2f nats  %5.1f%%"
          % (whitespace_selection.sum(), 100 * (1 - share)))
    print("  per token: %.4f detection, %.4f selection"
          % (whitespace_detection.mean(), whitespace_selection.mean()))

    report = {"positions": int(len(mass)), "whitespace": int(is_whitespace.sum()),
              "whitespace_types": int(len(whitespace)),
              "auc": float(auc(mass, is_whitespace)),
              "pi_whitespace": float(mass[is_whitespace].mean()),
              "pi_elsewhere": float(mass[~is_whitespace].mean()),
              "nll_total": float(whitespace_full.sum()),
              "nll_detection": float(whitespace_detection.sum()),
              "nll_selection": float(whitespace_selection.sum()),
              "detection_share": float(share)}
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nwrote %s" % args.output)

    print("\ngate: an expert can take selection, never detection -- the router has to do")
    print("detection anyway and the router is the backbone.")
    print("  offloadable share of whitespace NLL: %.1f%%" % (100 * (1 - share)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

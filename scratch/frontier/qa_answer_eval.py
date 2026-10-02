"""Answer-token NLL on the held-out frontier QA documents, per model, with the document
and without it.

Each held-out conversation (the long code document, then question / answer turns) is
rebuilt exactly as capture_inputs.py rendered it, and every answer's tokens are located
structurally -- the conversation rendered up to that question with a generation prompt
gives where the answer starts -- never by scanning for turn markers (two documents carry
literal `<|im_start|>assistant` in their code). One full-context forward per conversation
and model scores the answer bodies (not the closing `<|im_end|>`, reported separately as
turn completion). The control renders the same questions and reference answers with the
document removed: the gap between the two is how much the model reads the document,
rather than how well it imitates the answers' style.

Reports pooled answer NLL and, against the first arm, the paired per-document mean
difference with a document-level bootstrap interval.

    python scratch/frontier/qa_answer_eval.py --arm base=<checkpoint> --arm long2=<checkpoint> \\
        --output scratch/csa2-eval/qa-answers-long2.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scratch" / "dense_gr"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from capture_inputs import F, qa_conversation, split_of  # noqa: E402
from long_context_probe import TOKENIZER  # noqa: E402


def conversations(tokenizer, max_length, with_document=True):
    """Held-out QA conversations: (doc_id, ids, [(start, stop)] answer-body spans), rendered
    by capture_inputs.qa_conversation -- the same code that built the captured inputs."""
    docs = {d["doc_id"]: d["text"] for d in map(json.loads, open(F / "long-docs-code.jsonl", encoding="utf-8"))}
    seen, out = set(), []
    for row in map(json.loads, open(F / "frontier" / "qa-code.jsonl", encoding="utf-8")):
        if row["doc_id"] in seen or split_of(row["doc_id"]) != "eval":
            continue
        seen.add(row["doc_id"])
        ids, spans = qa_conversation(tokenizer, row["items"], docs[row["doc_id"]] if with_document else None,
                                     max_length)
        if spans:
            out.append((row["doc_id"], ids, [tuple(span) for span in spans]))
    return out


@torch.no_grad()
def answer_losses(model, ids, spans, device):
    """Summed NLL and count over answer bodies, and over the turn-closing tokens after them."""
    x = torch.tensor([ids], device=device)
    hidden = model.model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False).last_hidden_state[0]
    body = [p for a, b in spans for p in range(a, b)]
    close = [b for _, b in spans if b < len(ids)]

    def nll(positions):
        if not positions:
            return 0.0, 0
        at = torch.tensor(positions, device=device)
        logits = model.lm_head(hidden[at - 1]).float()
        return float(torch.nn.functional.cross_entropy(logits, x[0, at], reduction="sum")), len(positions)

    return nll(body), nll(close)


def bootstrap(deltas, draws=2000, seed=0):
    rng = np.random.default_rng(seed)
    d = np.asarray(deltas)
    means = d[rng.integers(0, len(d), (draws, len(d)))].mean(1)
    return [float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="append", required=True, help="name=checkpoint; the first is the reference")
    parser.add_argument("--max-length", type=int, default=32768)
    parser.add_argument("--limit", type=int, default=0, help="first N conversations only (smoke tests)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer

    from distillkit.models import Qwen35WidenedForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    sets = {"document": conversations(tokenizer, args.max_length),
            "no_document": conversations(tokenizer, args.max_length, with_document=False)}
    if args.limit:
        sets = {k: v[:args.limit] for k, v in sets.items()}
    print("held-out conversations: %d, answer tokens %d" % (
        len(sets["document"]), sum(b - a for _, _, s in sets["document"] for a, b in s)), flush=True)
    dtype = torch.float32 if args.device == "cpu" else torch.bfloat16
    report = dict(arms={}, conversations=[d for d, _, _ in sets["document"]])
    per_doc = {}
    for arm in args.arm:
        name, path = arm.split("=", 1)
        model = Qwen35WidenedForCausalLM.from_pretrained(path, dtype=dtype).to(args.device).eval()
        row = {}
        for kind, convs in sets.items():
            totals, docs = np.zeros(4), []
            for doc_id, ids, spans in convs:
                (bs, bn), (cs, cn) = answer_losses(model, ids, spans, args.device)
                totals += (bs, bn, cs, cn)
                docs.append(bs / bn)
            per_doc[name, kind] = docs
            row[kind] = dict(answer_nll=totals[0] / totals[1], answer_tokens=int(totals[1]),
                             close_nll=totals[2] / max(totals[3], 1))
        row["context_benefit"] = row["no_document"]["answer_nll"] - row["document"]["answer_nll"]
        report["arms"][name] = row
        print("%-12s answer NLL with document %.4f, without %.4f (benefit %.4f); close NLL %.4f"
              % (name, row["document"]["answer_nll"], row["no_document"]["answer_nll"],
                 row["context_benefit"], row["document"]["close_nll"]), flush=True)
        del model
        torch.cuda.empty_cache()
    reference = args.arm[0].split("=", 1)[0]
    for arm in args.arm[1:]:
        name = arm.split("=", 1)[0]
        delta = [a - b for a, b in zip(per_doc[name, "document"], per_doc[reference, "document"])]
        report["arms"][name]["paired_vs_" + reference] = bootstrap(delta)
        print("%s - %s per-document answer NLL: %+.4f [%+.4f, %+.4f]" % (name, reference, *bootstrap(delta)),
              flush=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

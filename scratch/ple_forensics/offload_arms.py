"""Arms A and B of the responsibility-transfer pilot: does removing whitespace
*selection* from the backbone's objective improve its content modelling?

Plain cross entropy, no teacher. Every existing arm in this project trained against the
distillation objective (0.7 sparse top-k KL + 0.3 hidden-state cosine); this does not, so
arm A is a fresh baseline rather than something comparable to the A3 numbers. The A3
window (decoder layers 20-28) and rate (3e-5) are borrowed as a *setup* that is known to
train stably, not as a reference point.

    A   mean CE over every assistant target.

    B   identical, except that at whitespace targets only the detection term survives:

            A at a whitespace target:  -log P(w)  =  -log P(WS) + -log P(w | WS)
            B at a whitespace target:  -log P(WS)

        Non-whitespace targets are untouched full-vocabulary CE.

The backbone already separates whitespace from everything else at AUC 0.9994, so
detection is not worth offloading and stays. Selection is 90.4% of the whitespace
gradient energy over this window and about a quarter of everything the window spends,
which is what B removes.

**Both arms divide by the same denominator** -- the total assistant target count -- so B
is exactly A with the selection terms absent, rather than A with content upweighted. That
keeps the counterfactual clean at the cost of a slightly smaller effective step in B,
which is the first thing to vary if B minus A comes out marginal.

    python scratch/ple_forensics/offload_arms.py --arm A --output ...
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from torch.nn import functional as F

from scratch.ple_forensics.router_check import whitespace_vocabulary
from scratch.ple_forensics.token_classes import class_of

ROOT = Path(__file__).resolve().parents[2]
STUDENT = str((ROOT / ".." / "student-hf").resolve())
DOCUMENTS = str((ROOT / ".." / "capture-data" / "heldout.jsonl").resolve())
MANIFESTS = [str((ROOT / ".." / name / "manifest.json").resolve())
             for name in ("teacher-cache-1m", "teacher-cache-5m")]


def assistant_targets(text, encoding, limit):
    """Target indices whose token is an assistant token, capped at `limit`."""
    from distillkit.independent_eval import role_spans

    spans = role_spans(text, encoding["offset_mapping"][:limit]).get("assistant", [])
    return [index for low, high in spans for index in range(max(low, 1), min(high, limit))]


def batches(documents, tokenizer, limit, size):
    for start in range(0, len(documents), size):
        chunk = documents[start:start + size]
        prepared = []
        for text in chunk:
            encoding = tokenizer(text, return_offsets_mapping=True)
            ids = encoding["input_ids"][:limit]
            targets = assistant_targets(text, encoding, len(ids))
            if targets:
                prepared.append((ids, np.array(targets, dtype=np.int64)))
        if prepared:
            yield prepared


def decompose(logits, target, whitespace_index):
    """Per-target full CE and its whitespace factorisation, detached, for logging.

    The arm loss is what trains; this is what is reported. Separating them means the
    trajectory shows the same quantities for A and B even though B never optimises the
    selection term, which is what distinguishes overfitting from instability.
    """
    with torch.no_grad():
        everything = torch.logsumexp(logits, dim=-1)
        within = torch.logsumexp(logits[:, whitespace_index], dim=-1)
        picked = logits.gather(1, target.unsqueeze(1)).squeeze(1)
        return ((everything - picked), (everything - within), (within - picked))


def arm_loss(logits, target, is_whitespace, whitespace_index, arm):
    """Summed loss over one document's assistant targets, under the arm's rule."""
    everything = torch.logsumexp(logits, dim=-1)
    picked = logits.gather(1, target.unsqueeze(1)).squeeze(1)
    full = everything - picked                       # -log P(w), per target
    if arm == "A":
        return full.sum()
    within = torch.logsumexp(logits[:, whitespace_index], dim=-1)
    detect = everything - within                     # -log P(WS)
    # B keeps detection at whitespace targets and full CE everywhere else. The sum is
    # over the same positions as A; only the whitespace terms are smaller.
    mask = torch.as_tensor(is_whitespace, device=logits.device)
    return torch.where(mask, detect, full).sum()


@torch.inference_mode()
def evaluate(model, documents, tokenizer, whitespace_index, limit, device):
    """Per-token NLL plus its detection/selection split, in document order."""
    out = {"nll": [], "target": [], "document": [], "detect": [], "select": []}
    for number, text in enumerate(documents):
        encoding = tokenizer(text, return_offsets_mapping=True)
        ids = encoding["input_ids"][:limit]
        targets = assistant_targets(text, encoding, len(ids))
        if not targets:
            continue
        # No `logits_to_keep`: with device_map="auto" the model spans both GPUs and that
        # index would have to live on whichever device holds the last layer. Slicing the
        # returned logits keeps the device bookkeeping in one place, at the cost of
        # materialising the full row block once.
        everything_logits = model(
            input_ids=torch.tensor([ids], device=device),
            attention_mask=torch.ones(1, len(ids), dtype=torch.long, device=device)).logits[0]
        positions = torch.as_tensor(np.array(targets) - 1, device=everything_logits.device)
        logits = everything_logits.index_select(0, positions).float()
        del everything_logits
        target = torch.as_tensor([ids[i] for i in targets], device=logits.device)
        everything = torch.logsumexp(logits, dim=-1)
        within = torch.logsumexp(logits[:, whitespace_index], dim=-1)
        picked = logits.gather(1, target.unsqueeze(1)).squeeze(1)
        out["nll"].append((everything - picked).cpu().numpy())
        out["detect"].append((everything - within).cpu().numpy())
        out["select"].append((within - picked).cpu().numpy())
        out["target"].append(target.cpu().numpy())
        out["document"].append(np.full(len(targets), number, dtype=np.int32))
        del logits
    return {key: np.concatenate(value) for key, value in out.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=["A", "B"], required=True)
    parser.add_argument("--train-docs", type=int, default=512)
    parser.add_argument("--eval-docs", type=int, default=128)
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--window", type=int, nargs=2, default=(20, 28))
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=0,
                        help="held-out evaluation every N steps, on --trajectory-docs; "
                             "0 evaluates only at the end")
    parser.add_argument("--trajectory-docs", type=int, default=48,
                        help="how many held-out documents the periodic evaluation uses")
    parser.add_argument("--trajectory", type=Path,
                        help="where to write the training/held-out trajectory as JSON")
    parser.add_argument("--baseline", action="store_true",
                        help="evaluate before training and exit; the shared reference")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from transformers import AutoTokenizer
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    from distillkit.independent_eval import unseen_records

    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(STUDENT, local_files_only=True)
    unseen = unseen_records(DOCUMENTS, MANIFESTS)
    if len(unseen) < args.train_docs + args.eval_docs:
        raise SystemExit("only %d unseen documents" % len(unseen))
    # Eval from the far end, as the capacity probe does, so changing --train-docs cannot
    # silently move which documents are scored.
    train_docs = [record["text"] for record in unseen[:args.train_docs]]
    eval_docs = [record["text"] for record in unseen[-args.eval_docs:]]
    print("%d train, %d eval, from %d unseen" % (len(train_docs), len(eval_docs), len(unseen)))

    model = Qwen3_5ForCausalLM.from_pretrained(
        STUDENT, dtype=torch.bfloat16, local_files_only=True, device_map="auto")
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.requires_grad_(False)
    low, high = args.window
    trainable = []
    for index in range(low, high + 1):
        for parameter in model.model.layers[index].parameters():
            parameter.requires_grad_(True)
            trainable.append(parameter)
    print("window %d-%d: %.2fB trainable of %.2fB"
          % (low, high, sum(p.numel() for p in trainable) / 1e9,
             sum(p.numel() for p in model.parameters()) / 1e9))
    device = next(model.parameters()).device
    embedding_device = model.get_input_embeddings().weight.device
    whitespace_index = torch.as_tensor(
        whitespace_vocabulary(tokenizer, model.config.vocab_size),
        device=model.get_output_embeddings().weight.device)

    if args.baseline:
        model.eval()
        scores = evaluate(model, eval_docs, tokenizer, whitespace_index,
                          args.tokens, embedding_device)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.output, **scores)
        print("baseline assistant NLL %.6f over %d tokens"
              % (scores["nll"].mean(), len(scores["nll"])))
        return 0

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1.0, (step + 1) / max(args.warmup, 1)))

    model.train()
    started, step, seen = time.perf_counter(), 0, 0
    trajectory = []
    running = {"content": 0.0, "content_tokens": 0, "detect": 0.0, "select": 0.0,
               "whitespace_tokens": 0}
    for prepared in batches(train_docs, tokenizer, args.tokens, args.batch):
        optimizer.zero_grad(set_to_none=True)
        total, counted = 0.0, 0
        for ids, targets in prepared:
            everything_logits = model(
                input_ids=torch.tensor([ids], device=embedding_device),
                attention_mask=torch.ones(1, len(ids), dtype=torch.long,
                                          device=embedding_device)).logits[0]
            positions = torch.as_tensor(targets - 1, device=everything_logits.device)
            logits = everything_logits.index_select(0, positions).float()
            target = torch.as_tensor([ids[i] for i in targets], device=logits.device)
            is_whitespace = class_of(np.array([ids[i] for i in targets]),
                                     tokenizer) == "whitespace"
            loss = arm_loss(logits, target, is_whitespace, whitespace_index, args.arm)
            full, detect, select = decompose(logits, target, whitespace_index)
            whitespace_rows = torch.as_tensor(is_whitespace, device=logits.device)
            running["content"] += float(full[~whitespace_rows].sum())
            running["content_tokens"] += int((~whitespace_rows).sum())
            running["detect"] += float(detect[whitespace_rows].sum())
            running["select"] += float(select[whitespace_rows].sum())
            running["whitespace_tokens"] += int(whitespace_rows.sum())
            counted += len(targets)
            total += float(loss)
            loss_for_backward = loss / max(len(targets), 1)
            loss_for_backward.backward()
            del everything_logits, logits, loss, loss_for_backward
        torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
        optimizer.step()
        scheduler.step()
        step += 1
        seen += counted
        if step % 20 == 0:
            print("  step %4d  loss/token %.5f  %d targets  %.1fs"
                  % (step, total / max(counted, 1), seen, time.perf_counter() - started),
                  flush=True)
        if args.eval_every and step % args.eval_every == 0:
            model.eval()
            sampled = evaluate(model, eval_docs[:args.trajectory_docs], tokenizer,
                               whitespace_index, args.tokens, embedding_device)
            model.train()
            held_out = class_of(sampled["target"], tokenizer) == "whitespace"
            point = {
                "step": step,
                # Training content NLL against held-out content NLL is what separates
                # overfitting (train improves, held-out worsens) from instability (both
                # worsen). Reported over the steps since the last point, not cumulative.
                "train_content": running["content"] / max(running["content_tokens"], 1),
                "train_detect": running["detect"] / max(running["whitespace_tokens"], 1),
                "train_select": running["select"] / max(running["whitespace_tokens"], 1),
                "heldout_content": float(sampled["nll"][~held_out].mean()),
                "heldout_detect": float(sampled["detect"][held_out].mean()),
                "heldout_select": float(sampled["select"][held_out].mean()),
            }
            trajectory.append(point)
            print("    [%4d] train content %.5f  held-out content %.5f  "
                  "ws detect %.5f  ws select %.5f"
                  % (step, point["train_content"], point["heldout_content"],
                     point["heldout_detect"], point["heldout_select"]), flush=True)
            running = {key: 0 if "tokens" in key else 0.0 for key in running}

    print("trained %d steps over %d targets in %.1fs"
          % (step, seen, time.perf_counter() - started))
    model.eval()
    scores = evaluate(model, eval_docs, tokenizer, whitespace_index,
                      args.tokens, embedding_device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **scores)
    summary = {"arm": args.arm, "lr": args.lr, "steps": step, "train_targets": seen,
               "eval_tokens": int(len(scores["nll"])),
               "assistant_nll": float(scores["nll"].mean())}
    if args.trajectory:
        args.trajectory.parent.mkdir(parents=True, exist_ok=True)
        args.trajectory.write_text(
            json.dumps({"summary": summary, "trajectory": trajectory}, indent=2),
            encoding="utf-8")
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

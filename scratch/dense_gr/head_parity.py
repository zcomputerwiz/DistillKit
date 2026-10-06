"""The separate head losses against shared_head.head_losses on real batches, on one GPU.

For each kind of micro-batch the trainer builds -- a 32K agent trace scored on its
assistant turns, QA scored on its answers, raw code under KL alone, the teacher's own
traces, packed looping rollouts under unlikelihood -- the student's final hidden states
are computed once, then both paths compute the step's objective from them (teacher weight
0.5, or KL plus unlikelihood for KL-only records, as backward_step does), and an fp32
reference -- the shared arithmetic on fp32 operands -- scores both. Reports each loss term,
the cosine and relative error of each path's gradients for the hidden states and the head
against the reference (and shared against old), and the head losses' time and peak memory
per path: head-phase numbers, not full-step ones. The old path's KL chunk is the rounds'
--kl-chunk (`--old-chunk`, 64), the shared path's the trainer's --head-chunk (`--head-chunk`, 512).

    python scratch/dense_gr/head_parity.py --checkpoint <ckpt> --output scratch/csa2-eval/head-parity.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

from shared_head import head_losses  # noqa: E402
from teacher_kl import CachedTeacher, grouped_tail_kl, scored_mask, unlikelihood_loss  # noqa: E402
from training_step import causal_ce  # noqa: E402

D = Path("D:/DeepThought/Projects/HybridModel")
CASES = [("agent trace, assistant turns", "teacher-cache-agent-smol-b", "assistant"),
         ("QA, answers only", "teacher-cache-frontier-qa2", "qa"),
         ("raw code, KL only", "teacher-cache-frontier-code-raw", "kl_only"),
         ("teacher math traces", "teacher-cache-teacher-math-gen", "plain"),
         ("looping rollouts, packed", "teacher-cache-onpolicy-r6-loop", "loops")]


VARIANTS = [("old", "old"), ("shared", "shared")]
CHUNKS = {"old": 64, "shared": 512}  # set from --old-chunk / --head-chunk


def batch_for(tokenizer, kind, cache, budget, context_weight=0.0):
    from smoke_train import ANSWER_MARKER

    path = D / cache
    marker = tokenizer(ANSWER_MARKER, add_special_tokens=False)["input_ids"]
    options = dict(answer_marker=marker, min_answer_tokens=2, max_length=min(32768, budget),
                   turn_close=tokenizer.convert_tokens_to_ids("<|im_end|>"),
                   think_close=tokenizer.convert_tokens_to_ids("</think>"))
    if kind in ("assistant", "qa"):
        options["assistant_only"] = [path]
    if kind == "qa":
        spans = {r["doc_id"]: r["answer_spans"] for r in map(json.loads, open(D / "capture-data" / "frontier-qa2.jsonl",
                                                                               encoding="utf-8"))}
        options.update(answer_spans=spans, answer_weight=8.0)
    if kind == "kl_only":
        options["kl_only"] = [path]
    if kind == "loops":
        options["unlikelihood"] = [path]
    teacher = CachedTeacher(path, "train", device="cuda", **options)
    teacher.pad_blocks = True
    if context_weight and kind in ("assistant", "qa"):
        teacher.context_kl = (context_weight, 8)
    groups = teacher._groups(1, 128, budget)
    # The fullest micro-batch: the most tokens, and for loops the most rollouts.
    group, width = max(groups, key=lambda g: (len(g[0]) if kind == "loops" else 0, len(g[0]) * g[1]))
    batch = teacher.read_batch(group, width)
    batch["kl_only"] = kind in ("kl_only", "loops")
    return batch


def objective(model, hidden, batch, mode, head_weight=None):
    ids, weight, negative = batch["input_ids"], batch.get("weight"), batch.get("negative")
    rows, length = ids.shape
    count = float(weight[:, :-1].sum()) if weight is not None else rows * (length - 1)
    mask = scored_mask(length, ids.device, rows)
    if weight is not None:
        mask = mask * weight
    kl_mask = mask if negative is None else mask * ~negative
    if mode != "old":
        # "reference": the same arithmetic on fp32 operands with the logits formed exactly.
        sums = head_losses(hidden, model.lm_head.weight if head_weight is None else head_weight, ids,
                           weight=weight, topk_ids=batch["topk_ids"], topk_logprobs=batch["topk_logprobs"],
                           kl_weight=torch.broadcast_to(kl_mask, ids.shape), negative=negative,
                           chunk=CHUNKS["shared"],
)
        language, carried, repelled = sums["nll"] / sums["weight"], sums["kl"] / count, sums["unlikelihood"] / count
    else:
        language = causal_ce(model, hidden, ids) if weight is None else causal_ce(model, hidden, ids, weight=weight)
        carried = grouped_tail_kl(hidden, model.lm_head, batch["topk_ids"], batch["topk_logprobs"], kl_mask,
                                  chunk_length=CHUNKS["old"]) / count
        repelled = hidden.new_zeros((), dtype=torch.float32)
        if negative is not None:
            repelled = unlikelihood_loss(hidden, model.lm_head, ids, negative, weight=weight) / count
    total = carried + repelled if batch["kl_only"] else 0.5 * language + 0.5 * carried
    return total, {"ce": float(language), "kl": float(carried), "unlikelihood": float(repelled)}


def run(model, hidden, batch, mode, repeats=3):
    times, peak = [], 0
    for _ in range(repeats if mode != "reference" else 1):
        h = (hidden.detach().float() if mode == "reference" else hidden.detach()).requires_grad_(True)
        head = model.lm_head.weight.detach().float().requires_grad_(True) if mode == "reference" else None
        model.lm_head.weight.grad = None
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        start = time.perf_counter()
        if mode == "streaming":
            from streaming_head import backward_head
            ids, weight = batch["input_ids"], batch.get("weight")
            negative = batch.get("negative")
            count = float(weight[:, :-1].sum()) if weight is not None else ids.shape[0] * (ids.shape[1] - 1)
            mask = scored_mask(ids.shape[1], ids.device, ids.shape[0])
            if weight is not None:
                mask = mask * weight
            kl_mask = mask if negative is None else mask * ~negative
            coeff = (0., 1., 1.) if batch["kl_only"] else (.5, .5, 0.)
            sums, h.grad = backward_head(h, model.lm_head.weight, ids, weight=weight,
                topk_ids=batch["topk_ids"], topk_logprobs=batch["topk_logprobs"],
                kl_weight=torch.broadcast_to(kl_mask, ids.shape), negative=negative,
                chunk=CHUNKS["shared"], coefficients=tuple(c / count for c in coeff))
            terms = {"ce": float(sums["nll"] / count), "kl": float(sums["kl"] / count),
                     "unlikelihood": float(sums["unlikelihood"] / count)}
        else:
            total, terms = objective(model, h, batch, mode, head)
            total.backward()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
        peak = max(peak, torch.cuda.max_memory_allocated() - base)
    w_grad = head.grad if mode == "reference" else model.lm_head.weight.grad
    return terms, h.grad.float(), w_grad.float(), statistics.median(times), peak / 2**30


def compare(a, b):
    cos = torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0)
    return {"cosine": float(cos), "relative_error": float((a - b).norm() / b.norm().clamp_min(1e-30))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=32768)
    parser.add_argument("--qa-budget", type=int, help="isolated head suite: explicit QA prefix cap when answers require longer context")
    parser.add_argument("--old-chunk", type=int, default=64, help="the old KL's chunk: the rounds' --kl-chunk")
    parser.add_argument("--head-chunk", type=int, default=512, help="the shared path's rows a projection")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cce-diagnostic", action="store_true", help="bounded isolated same-tile CCE diagnostic")
    parser.add_argument("--cce-bucketed", action="store_true", help="also compare experimental pre-bucketed direct-tail forward")
    parser.add_argument("--cce-unlocked", action="store_true", help="also time bucketed stable reductions without stock locked-LSE diagnostics")
    parser.add_argument("--cases", default="0,1,2", help="CCE diagnostic case indices")
    parser.add_argument("--limit-rows", type=int, default=1024, help="CCE diagnostic active-row cap")
    parser.add_argument("--timing-rows", type=int, default=4096, help="CCE diagnostic timing row cap")
    parser.add_argument("--gradient-rows", type=int, default=128, help="CCE normalizer gradient sanity rows")
    parser.add_argument("--repeats", type=int, default=3, help="CCE diagnostic timed repetitions")
    parser.add_argument("--context-weight", type=float, default=0.0, help="CCE diagnostic sampled context KL")
    parser.add_argument("--selection-replay-check", action="store_true", help="actual BF16 model gradient check of checkpoint selection caching")
    parser.add_argument("--selection-repeat-control", action="store_true", help="compare two unchanged backward passes instead of the replay cache")
    parser.add_argument("--streaming-head-check", action="store_true", help="with selection-replay-check: compare immediate head backward instead of selection caching")
    parser.add_argument("--head-update-check", action="store_true", help="compare predictions after fresh controlled optimizer updates, including ordinary repeat")
    parser.add_argument("--update-replay-cache", action="store_true", help="with head-update-check: combine streaming and checkpoint selection caching")
    parser.add_argument("--include-streaming-head", action="store_true", help="include streaming head in the existing five-case isolated head comparison")
    parser.add_argument("--selection-tp", action="store_true", help="selection check with the current two-GPU body/head placement")
    args = parser.parse_args()
    if args.selection_tp and not (args.selection_replay_check or args.head_update_check):
        parser.error("--selection-tp requires --selection-replay-check or --head-update-check")
    if args.head_update_check and (args.selection_replay_check or args.cce_diagnostic):
        parser.error("choose one diagnostic mode")
    if args.update_replay_cache and not args.head_update_check:
        parser.error("--update-replay-cache requires --head-update-check")
    if args.include_streaming_head:
        VARIANTS.append(("streaming", "streaming"))
    if args.selection_repeat_control and not args.selection_replay_check:
        parser.error("--selection-repeat-control requires --selection-replay-check")
    if args.streaming_head_check and not args.selection_replay_check:
        parser.error("--streaming-head-check requires --selection-replay-check")
    if args.selection_replay_check and args.cce_diagnostic:
        parser.error("choose one diagnostic mode")
    if args.cce_unlocked and not (args.cce_diagnostic and args.cce_bucketed):
        parser.error("--cce-unlocked requires --cce-diagnostic and --cce-bucketed")
    if args.cce_diagnostic:
        import os

        if os.environ.get("CUDA_VISIBLE_DEVICES") != "1" or torch.cuda.device_count() != 1:
            raise SystemExit("CCE diagnostic must use only physical GPU1 (CUDA_VISIBLE_DEVICES=1)")
    CHUNKS.update(old=args.old_chunk, shared=args.head_chunk)
    from transformers import AutoTokenizer

    from distillkit.models import Qwen35WidenedForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = Qwen35WidenedForCausalLM.from_pretrained(args.checkpoint, dtype=torch.bfloat16).cuda().eval()
    if args.head_update_check:
        from streaming_head import update_check
        update_check(model, args)
        return
    if args.selection_replay_check:
        from selection_replay import gradient_check
        gradient_check(model, args)
        return
    for p in model.model.parameters():
        p.requires_grad_(False)
    model.lm_head.weight.requires_grad_(True)  # tied to the embedding: the body still runs without grad
    if args.cce_diagnostic:
        import cce_diagnostics

        cce_diagnostics.run(sys.modules[__name__], model, tokenizer, args)
        return
    results = []
    for name, cache, kind in CASES:
        budget = args.qa_budget if kind == "qa" and args.qa_budget is not None else args.budget
        batch = batch_for(tokenizer, kind, cache, budget)
        with torch.no_grad():
            hidden = model.model(input_ids=batch["input_ids"], attention_mask=torch.ones_like(batch["input_ids"]),
                                 use_cache=False).last_hidden_state
        ref_terms, ref_h, ref_w, _, _ = run(model, hidden, batch, "reference")
        row = {"case": name, "shape": list(batch["input_ids"].shape),
               "scored": float((batch["weight"][:, :-1] > 0).float().mean()) if "weight" in batch else 1.0,
               "reference": ref_terms}
        for label, mode in VARIANTS:
            terms, h_grad, w_grad, seconds, peak = run(model, hidden, batch, mode)
            row[label] = {**terms, "seconds": seconds, "peak_gib": peak,
                          "hidden_grad": compare(h_grad, ref_h), "head_grad": compare(w_grad, ref_w)}
            if label == "old":
                old_h, old_w = h_grad, w_grad
            else:
                row[label]["vs old"] = {"hidden_grad": compare(h_grad, old_h), "head_grad": compare(w_grad, old_w)}
            if args.include_streaming_head and label == "shared":
                shared_h, shared_w = h_grad, w_grad
            if label == "streaming":
                row[label]["vs shared"] = {"hidden_grad": compare(h_grad, shared_h),
                                           "head_grad": compare(w_grad, shared_w)}
            del h_grad, w_grad
            torch.cuda.empty_cache()
        del ref_h, ref_w, old_h, old_w
        if args.include_streaming_head:
            del shared_h, shared_w
        results.append(row)
        print("== %s %s, %.0f%% of positions scored" % (name, row["shape"], 100 * row["scored"]))
        print("   %-18s ce %.5f kl %.5f ul %.5f" % ("fp32 reference", ref_terms["ce"], ref_terms["kl"],
                                                  ref_terms["unlikelihood"]))
        for label, *_ in VARIANTS:
            s = row[label]
            print("   %-22s ce %.5f kl %.5f ul %.5f  %.3fs  peak %.2f GiB  vs reference: grad cos hidden %.6f head %.6f"
                  "  rel err %.2e / %.2e" % (label, s["ce"], s["kl"], s["unlikelihood"], s["seconds"], s["peak_gib"],
                                             s["hidden_grad"]["cosine"], s["head_grad"]["cosine"],
                                             s["hidden_grad"]["relative_error"], s["head_grad"]["relative_error"]),
                  flush=True)
        del hidden, batch
        torch.cuda.empty_cache()
        print("   shared vs old: grad cos hidden %.7f head %.7f" % (
            row["shared"]["vs old"]["hidden_grad"]["cosine"], row["shared"]["vs old"]["head_grad"]["cosine"]), flush=True)
    args.output.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()

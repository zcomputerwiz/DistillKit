"""Objective forensics: which term makes the same sidecar help under CE and hurt under
distillation?

The two-stream pilot settled the architecture question and produced a result nobody
predicted: under plain cross entropy the single-stream retrofit ``S`` *improves* content
at every rate. The historical content harm (C1, +0.003512) was measured under the
distillation objective, so the objective is the remaining suspect, not the sidecar.

Same three architectures as ``costream_arms.py`` -- A stock, S the sidecar writing into
the ordinary residual, M the sidecar writing into a private lane -- under four objectives:

    ce         plain cross entropy over every supervised position
    kl         0.7-weight term alone: sparse top-k teacher KL, temperature 1.0,
               missing_probability_handling=zero, sparse_chunk_length=256
    cosine     0.3-weight term alone: hidden-state cosine against the cached anchors
    combined   (0.7 * kl + 0.3 * cosine) / 1.0, the historical objective exactly
    grouped    the same sparse KL with the tail grouped instead of zeroed
    grouped_ce (0.5 * ce + 0.5 * grouped) / 1.0, the proposed replacement target

**Grouped tail is ``symmetric_uniform``, exactly.** The proposed repair is to distil over
the K cached tokens plus one aggregate bucket carrying the omitted mass::

    L = -sum_i p_i log q_i  -  p_tail log q_tail

and this codebase's ``MissingProbabilityHandling.SYMMETRIC_UNIFORM`` already computes it.
Spreading the tail uniformly over the same ``V - K`` support on both sides makes the
per-token factor cancel::

    sum_{i not in K} (p_tail/(V-K)) log[(p_tail/(V-K)) / (q_tail/(V-K))]
        = p_tail log(p_tail / q_tail)

so the "uniform" assumption is vacuous and the option is the grouped-tail objective under
a misleading name. Measured against the formula above, written independently: the values
differ by exactly the coarsened teacher's entropy (-10.147993 against -10.147994 on a
random fp64 case) and the gradients agree to 2e-8. ``tests/test_grouped_tail.py`` pins it.
The repair therefore needs no new loss function, only a different flag -- and the tail is
real: mean 0.0043, above 0.01 at 5.1% of positions, up to 0.872.

Every loss is the project's own implementation called directly -- ``KLDLoss``,
``HiddenStateCosineLoss``, ``HiddenStateMapping``, ``OfflineHiddenStateSignalSource`` --
rather than a reimplementation, so "faithful to the historical objective" means the same
code path rather than the same formula written twice.

**What the anchors imply, and why M is still interesting here.** The historical mapping
is ``[[4, 0], [32, 1]]``: student layer 4 against the teacher's layer 8, and the post-norm
state against the teacher's layer 64. The sidecar writes at layer 1, *upstream of anchor
4*, so under S the cosine term sees every change the sidecar makes to layer 4 and is free
to penalise it. M's lane is read at layers 20-28 only, so anchor 4 never sees the lane at
all. If hidden-state matching is what punishes PLE, S and M should come apart here even
though they did not under CE.

## Deviations from the historical runs, all deliberate and all shared by every arm

* Trainable window is decoder layers 20-28, not the whole backbone, and the horizon is
  512 documents rather than 5M tokens. That is the ``costream`` setup, kept so the CE row
  is comparable; the cross-objective contrast is within-row.
* Training documents are the first 512 cached documents of at most 1024 tokens. The
  signal source validates tokens against the capture, so documents cannot be truncated;
  they are selected by length instead.
* Supervision covers every valid position, as the historical objective did. The earlier
  CE arms scored assistant targets only, which is why ``--objective ce`` is re-run here
  rather than reused: it is the matched control for the coverage difference.
* The projections are 2560 -> 5120 per anchor and start from ``HiddenStateMapping``'s
  xavier initialisation, where the historical stage 2 inherited trained ones from stage
  1. ``--train-projections`` reproduces that: projections only, backbone frozen, saved
  once and loaded identically by all three arms.

Evaluation is unchanged from the pilot -- held-out assistant tokens from the unseen tail,
per token class -- so every number here is comparable to the CE matrix already published.

    python scratch/ple_forensics/objective_arms.py --arm S --objective cosine --output ...
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

from scratch.ple_forensics.costream_arms import (
    DEFAULT_GGUF,
    Wiring,
    by_class,
    evaluate,
    ngram_raw_of,
)
from scratch.ple_forensics.offload_arms import DOCUMENTS, MANIFESTS, STUDENT
from scratch.ple_forensics.router_check import whitespace_vocabulary

CACHE = str((Path(__file__).resolve().parents[3] / "teacher-cache-5m").resolve())
#: The historical mapping, verbatim from `examples/qwen35_ple_stage2_5m.yml`.
LAYER_MAPPING = [(4, 0), (32, 1)]
KL_WEIGHT, COSINE_WEIGHT = 0.7, 0.3


def cached_documents(cache, limit, max_tokens):
    """The first `limit` cached training documents of at most `max_tokens` tokens."""
    chosen = []
    for doc in cache.manifest["documents"]:
        if doc["split"] != "train" or doc["length"] > max_tokens:
            continue
        chosen.append(doc["doc_id"])
        if len(chosen) >= limit:
            break
    return chosen


#: What each objective is made of. The trainer computes `sum(w_i L_i) / sum(w_i)`, so a
#: single-term objective is its term and the divisor is only ever interesting for a mix.
OBJECTIVES = {
    "ce": {"ce": 1.0},
    "kl": {"kl": 1.0},
    "cosine": {"cosine": 1.0},
    "combined": {"kl": KL_WEIGHT, "cosine": COSINE_WEIGHT},
    "grouped": {"grouped": 1.0},
    "grouped_ce": {"ce": 0.5, "grouped": 0.5},
}


def cross_entropy(outputs, ids):
    """Ground-truth CE at every position, the term that survives a truncated teacher."""
    logits = outputs.logits[0, :-1].float()
    target = torch.as_tensor(ids[1:], device=logits.device)
    return (torch.logsumexp(logits, dim=-1)
            - logits.gather(1, target.unsqueeze(1)).squeeze(1)).mean()


def objective_loss(objective, outputs, ids, signal, mask, mapping, losses):
    """One document's loss and its components, under the named objective.

    Each term is normalised the way the trainer normalises it -- the divergences by the
    supervised position count, the cosine by position count per anchor and then averaged
    over anchors -- and a mixture divides by the sum of its weights.
    """
    weights = OBJECTIVES[objective]
    parts = {}
    if "ce" in weights:
        parts["ce"] = cross_entropy(outputs, ids)
    for name in ("kl", "grouped"):
        if name in weights:
            parts[name] = losses[name](outputs, signal, mask=mask)
    if "cosine" in weights:
        parts["cosine"] = losses["cosine"](outputs, signal, mask=mask,
                                           hidden_state_mapping=mapping)
    device = parts[next(iter(weights))].device
    total = sum(weight * parts[name].to(device) for name, weight in weights.items())
    return total / sum(weights.values()), parts


@torch.inference_mode()
def tail_scores(wiring, source, doc_ids, hasher, table, device, tokenizer):
    """Ground-truth NLL split by whether the teacher's top-k contains the true token.

    `missing_probability_handling: zero` gives every token outside the cached top-64 a
    target probability of exactly zero, so the KL term actively penalises the student for
    putting mass on a true token the teacher did not rank. If that truncation is what
    turns the sidecar from a help into a harm, the harm has to live on the positions where
    the true token is outside the teacher's list -- and nowhere else.

    Scored on the cache's own eval split, which is the only held-out text that has a
    teacher distribution to be outside of.
    """
    from scratch.ple_forensics.token_classes import class_of

    collected = {key: [] for key in ("nll", "token", "in_topk", "document")}
    for number, doc_id in enumerate(doc_ids):
        cached = source.cache.read_document(doc_id, include_hidden_states=False)
        ids = cached["input_ids"].astype(np.int64)
        raw = ngram_raw_of(ids.tolist(), hasher, table)
        logits = wiring.forward(ids.tolist(), raw, device).logits[0, :-1].float()
        target = torch.as_tensor(ids[1:], device=logits.device)
        nll = (torch.logsumexp(logits, dim=-1)
               - logits.gather(1, target.unsqueeze(1)).squeeze(1))
        ranked = torch.as_tensor(cached["topk_ids"].astype(np.int64)[:-1], device=logits.device)
        collected["nll"].append(nll.cpu().numpy())
        collected["token"].append(ids[1:])
        collected["in_topk"].append(
            (ranked == target.unsqueeze(1)).any(dim=1).cpu().numpy())
        collected["document"].append(np.full(len(ids) - 1, number, dtype=np.int32))
        del logits, nll
    packed = {key: np.concatenate(value) for key, value in collected.items()}
    inside = packed["in_topk"]
    labels = class_of(packed["token"], tokenizer)
    print("\nground truth against the teacher's top-k, %d positions, %.1f%% inside"
          % (len(inside), 100 * inside.mean()))
    for name in ("lexical", "whitespace", "punctuation", "control"):
        mask = labels == name
        if not mask.any():
            continue
        print("  %-12s inside %.5f (n=%d)   outside %.5f (n=%d)"
              % (name, packed["nll"][mask & inside].mean(), (mask & inside).sum(),
                 packed["nll"][mask & ~inside].mean(), (mask & ~inside).sum()))
    return packed


@torch.inference_mode()
def write_vs_cosine(wiring, sidecar, mapping, source, doc_ids, hasher, table, device,
                    tokenizer):
    """Does the hidden-state term resist exactly the positions the sidecar writes hardest?

    For each cached held-out document, per position: the magnitude of the PLE write
    relative to the stream it lands in, and the cosine matching error at each anchor. If
    hidden-state imitation is what punishes PLE, the two should be positively related --
    the more the sidecar changes the representation, the worse it matches the teacher.

    Run on the cache's own eval split, because a position's teacher anchor has to exist;
    the unseen documents the arms are scored on have no teacher signal at all.
    """
    from scratch.ple_forensics.token_classes import class_of

    collected = {"write": [], "token": [], "cos_4": [], "cos_32": []}
    for doc_id in doc_ids:
        cached = source.cache.read_document(doc_id)
        ids = cached["input_ids"].astype(np.int64).tolist()
        raw = ngram_raw_of(ids, hasher, table)
        outputs = wiring.forward(ids, raw, device, output_hidden_states=True)
        batch = {"input_ids": torch.tensor([ids], device=device),
                 "attention_mask": torch.ones(1, len(ids), dtype=torch.long, device=device),
                 "doc_id": [doc_id]}
        signal = source.get_signal(batch, return_hidden_states=True)

        # The stream the sidecar consumed: hidden_states[1] is layer 0's output, which is
        # layer 1's input, and the sidecar runs before the rest of layer 1.
        entry = outputs.hidden_states[1]
        write = (wiring.box["memory"] if wiring.arm == "M"
                 else sidecar.write(entry, raw))
        collected["write"].append(
            (write.float().norm(dim=-1)[0] / entry.float().norm(dim=-1)[0].clamp_min(1e-9))
            .cpu().numpy())
        collected["token"].append(np.asarray(ids))
        for student_layer, anchor in mapping.layer_mapping:
            student_h = outputs.hidden_states[student_layer]
            projection = mapping.projections[anchor]
            projected = projection(student_h.to(projection.weight.device,
                                                projection.weight.dtype))
            teacher = signal.hidden_states[anchor].to(projected.device, projected.dtype)
            error = 1 - torch.nn.functional.cosine_similarity(projected, teacher, dim=-1)
            collected["cos_%d" % student_layer].append(error.float()[0].cpu().numpy())
    packed = {key: np.concatenate(value) for key, value in collected.items()}

    labels = class_of(packed["token"], tokenizer)
    print("\nwrite magnitude against hidden-state matching error, %d positions"
          % len(packed["write"]))
    for anchor in ("cos_4", "cos_32"):
        error, write = packed[anchor], packed["write"]
        pearson = float(np.corrcoef(write, error)[0, 1])
        order = np.argsort(write)
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(len(order))
        error_order = np.argsort(error)
        error_ranks = np.empty_like(error_order, dtype=np.float64)
        error_ranks[error_order] = np.arange(len(error_order))
        spearman = float(np.corrcoef(ranks, error_ranks)[0, 1])
        deciles = np.quantile(write, np.linspace(0, 1, 11))
        bottom = error[write <= deciles[1]].mean()
        top = error[write >= deciles[-2]].mean()
        print("  %-6s pearson %+.4f  spearman %+.4f   bottom decile %.5f  top decile %.5f"
              % (anchor, pearson, spearman, bottom, top))
        for name in ("whitespace", "control", "punctuation", "lexical"):
            mask = labels == name
            if mask.any():
                print("      %-12s error %.5f  write %.6f  n=%d"
                      % (name, error[mask].mean(), write[mask].mean(), mask.sum()))
    return packed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=["A", "S", "M"], required=True)
    parser.add_argument("--objective", choices=list(OBJECTIVES), required=True)
    parser.add_argument("--cache", default=CACHE)
    parser.add_argument("--gguf", default=DEFAULT_GGUF)
    parser.add_argument("--train-docs", type=int, default=512)
    parser.add_argument("--eval-docs", type=int, default=128)
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--window", type=int, nargs=2, default=(20, 28))
    parser.add_argument("--read-layers", type=int, nargs="+")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--trajectory-docs", type=int, default=48)
    parser.add_argument("--trajectory", type=Path)
    parser.add_argument("--projections", type=Path,
                        help="load anchor projections from here instead of xavier init")
    parser.add_argument("--train-projections", action="store_true",
                        help="train only the projections and save them to --output; the "
                             "shared stand-in for the historical stage-1 inheritance")
    parser.add_argument("--write-diagnostic", type=Path,
                        help="after training, relate per-position PLE write magnitude to "
                             "per-position hidden-state matching error, on the cache's "
                             "own eval split; needs a cosine-bearing objective")
    parser.add_argument("--tail-diagnostic", type=Path,
                        help="after training, score the cache's eval split and record "
                             "whether each true token was inside the teacher's top-k")
    parser.add_argument("--diagnostic-docs", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    weights = OBJECTIVES[args.objective]
    needs_teacher = bool({"kl", "grouped", "cosine"} & set(weights))
    needs_hidden = "cosine" in weights

    from transformers import AutoConfig, AutoTokenizer

    from distillkit.hsd_mapping import HiddenStateMapping
    from distillkit.independent_eval import unseen_records
    from distillkit.lossfuncs.hidden_state import HiddenStateCosineLoss, last_anchor_report
    from distillkit.lossfuncs.kl import KLDLoss
    from distillkit.missing_probability import MissingProbabilityHandling
    from distillkit.models.qwen35_sidecar import Qwen35SidecarForCausalLM
    from distillkit.ngram_hash import NGramHasher
    from distillkit.ngram_table import GGUFNGramTable
    from distillkit.signals import OfflineHiddenStateSignalSource

    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(STUDENT, local_files_only=True)
    source = OfflineHiddenStateSignalSource(args.cache)
    train_ids = cached_documents(source.cache, args.train_docs, args.tokens)
    eval_docs = [record["text"] for record in
                 unseen_records(DOCUMENTS, MANIFESTS)[-args.eval_docs:]]
    print("%d cached train documents (<= %d tokens), %d held-out eval documents"
          % (len(train_ids), args.tokens, len(eval_docs)))

    table = GGUFNGramTable(args.gguf)
    hasher = NGramHasher()
    config = AutoConfig.from_pretrained(STUDENT, local_files_only=True)
    config = getattr(config, "text_config", config)
    config.sidecar_variant = "ple"
    model = Qwen35SidecarForCausalLM.from_pretrained(
        STUDENT, config=config, dtype=torch.bfloat16, local_files_only=True,
        device_map="auto")
    model.config.use_cache = False
    model.requires_grad_(False)

    low, high = args.window
    read_layers = args.read_layers if args.read_layers else list(range(low, high + 1))
    wiring = Wiring(model, args.arm, read_layers, project=False)
    sidecar = model.model.layers[model.config.sidecar_layer_index].sidecar

    mapping = None
    if needs_hidden:
        mapping = HiddenStateMapping(student=model, teacher_hidden_size=source.hidden_size,
                                     layer_mapping=LAYER_MAPPING)
        if args.projections:
            state = torch.load(args.projections, map_location="cpu")
            mapping.projections.load_state_dict(state)
            print("loaded projections from %s" % args.projections)

    trainable = []
    if args.train_projections:
        for parameter in mapping.projections.parameters():
            parameter.requires_grad_(True)
            trainable.append(parameter)
    else:
        for index in range(low, high + 1):
            for parameter in model.model.layers[index].parameters():
                parameter.requires_grad_(True)
                trainable.append(parameter)
        if args.arm in ("S", "M"):
            for parameter in sidecar.parameters():
                parameter.requires_grad_(True)
                trainable.append(parameter)
        trainable.extend(wiring.parameters())
        if needs_hidden:
            # The projections belong to the objective, not the arm: all three arms get
            # the same ones, trained the same way, so S - A stays a clean contrast.
            for parameter in mapping.projections.parameters():
                parameter.requires_grad_(True)
                trainable.append(parameter)
    print("arm %s under %s: %.3fB trainable parameters"
          % (args.arm, args.objective, sum(p.numel() for p in trainable) / 1e9))

    losses = {"kl": KLDLoss(temperature=1.0,
                            missing_probability_handling=MissingProbabilityHandling.ZERO,
                            sparse_chunk_length=256),
              # The grouped tail, under its existing name. Same temperature, same
              # chunking; the only difference is that the omitted mass is carried rather
              # than asserted to be zero.
              "grouped": KLDLoss(
                  temperature=1.0,
                  missing_probability_handling=MissingProbabilityHandling.SYMMETRIC_UNIFORM,
                  sparse_chunk_length=256),
              "cosine": HiddenStateCosineLoss()}
    device = model.get_input_embeddings().weight.device
    whitespace_index = torch.as_tensor(
        whitespace_vocabulary(tokenizer, model.config.vocab_size),
        device=model.get_output_embeddings().weight.device)

    def score(documents, collect_reads=False):
        return evaluate(wiring, documents, tokenizer, hasher, table, whitespace_index,
                        args.tokens, device, collect_reads)

    def document_loss(doc_id):
        """The objective's value on one cached document, plus its components."""
        cached = source.cache.read_document(doc_id, tokens_only=not needs_teacher)
        ids = cached["input_ids"].astype(np.int64).tolist()
        raw = ngram_raw_of(ids, hasher, table)
        outputs = wiring.forward(ids, raw, device, output_hidden_states=needs_hidden)
        signal, mask = None, None
        if needs_teacher:
            batch = {"input_ids": torch.tensor([ids], device=device),
                     "attention_mask": torch.ones(1, len(ids), dtype=torch.long,
                                                  device=device),
                     "doc_id": [doc_id]}
            signal = source.get_signal(batch, return_hidden_states=needs_hidden)
            mask = torch.ones(1, len(ids), 1, dtype=torch.bool,
                              device=outputs.logits.device)
        return objective_loss(args.objective, outputs, ids, signal, mask, mapping, losses)

    initial = [parameter.detach().to("cpu", torch.float32).clone() for parameter in trainable]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1.0, (step + 1) / max(args.warmup, 1)))

    model.train()
    started, step = time.perf_counter(), 0
    trajectory, running = [], {}
    for start in range(0, len(train_ids), args.batch):
        chunk = train_ids[start:start + args.batch]
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for doc_id in chunk:
            value, parts = document_loss(doc_id)
            total += float(value.detach())
            for name, part in parts.items():
                running[name] = running.get(name, 0.0) + float(part.detach())
            running["documents"] = running.get("documents", 0.0) + 1
            (value / len(chunk)).backward()
            del value, parts
        torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
        optimizer.step()
        scheduler.step()
        step += 1
        if step % 20 == 0:
            seen = max(running.get("documents", 1), 1)
            terms = "  ".join("%s %.5f" % (name, running[name] / seen)
                              for name in sorted(running) if name != "documents")
            print("  step %4d  loss %.5f  [%s]  %.1fs"
                  % (step, total / len(chunk), terms, time.perf_counter() - started),
                  flush=True)
        if args.eval_every and step % args.eval_every == 0 and not args.train_projections:
            model.eval()
            sampled = score(eval_docs[:args.trajectory_docs])
            model.train()
            point = {"step": step, **by_class(sampled["nll"], sampled["target"], tokenizer)}
            seen = max(running.get("documents", 1), 1)
            point.update({name: running[name] / seen for name in running if name != "documents"})
            trajectory.append(point)
            print("    [%4d] lexical %.5f  whitespace %.5f  control %.5f"
                  % (step, point["lexical"], point["whitespace"], point["control"]),
                  flush=True)
            running = {}

    print("trained %d steps in %.1fs" % (step, time.perf_counter() - started))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.train_projections:
        torch.save(mapping.projections.state_dict(), args.output)
        print(json.dumps({"projections": str(args.output), "steps": step,
                          "anchor_report": last_anchor_report()}))
        return 0

    model.eval()
    scores = score(eval_docs, collect_reads=args.arm == "M")
    np.savez(args.output, **scores)
    summary = {"arm": args.arm, "objective": args.objective, "lr": args.lr, "steps": step,
               "eval_tokens": int(len(scores["nll"])),
               "assistant_nll": float(scores["nll"].mean()),
               "by_class": by_class(scores["nll"], scores["target"], tokenizer)}
    if needs_hidden:
        summary["anchor_report"] = last_anchor_report()
    if args.arm in ("S", "M"):
        summary["sidecar"] = {name: float(parameter.detach().float().norm())
                              for name, parameter in sidecar.named_parameters()}
    if args.arm == "M":
        wiring.box["enabled"] = False
        off = score(eval_docs)
        np.savez(args.output.with_name(args.output.stem + "-off.npz"), **off)
        wiring.box["enabled"] = True
        summary["memory_off"] = by_class(off["nll"], off["target"], tokenizer)
        summary["read_scale"] = {index: float(read.scale.item())
                                 for index, read in wiring.reads.items()}
    moved = float(np.sqrt(sum(
        float((parameter.detach().to("cpu", torch.float32) - begin).pow(2).sum())
        for parameter, begin in zip(trainable, initial))))
    summary["displacement"] = moved

    if args.tail_diagnostic:
        held_out = [doc["doc_id"] for doc in source.cache.manifest["documents"]
                    if doc["split"] == "eval" and doc["length"] <= args.tokens
                    ][:args.diagnostic_docs]
        np.savez(args.tail_diagnostic,
                 **tail_scores(wiring, source, held_out, hasher, table, device, tokenizer))

    if args.write_diagnostic and needs_hidden and args.arm in ("S", "M"):
        held_out = [doc["doc_id"] for doc in source.cache.manifest["documents"]
                    if doc["split"] == "eval" and doc["length"] <= args.tokens
                    ][:args.diagnostic_docs]
        packed = write_vs_cosine(wiring, sidecar, mapping, source, held_out, hasher,
                                 table, device, tokenizer)
        np.savez(args.write_diagnostic, **packed)
    if args.trajectory:
        args.trajectory.parent.mkdir(parents=True, exist_ok=True)
        args.trajectory.write_text(
            json.dumps({"summary": summary, "trajectory": trajectory}, indent=2),
            encoding="utf-8")
    wiring.close()
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""How many nats can a linear read of the n-gram table add to this student's responses?

Every arm so far has measured the table through an objective whose optimum is a silent
sidecar: the teacher has no n-gram table, so both the KL and the hidden-state terms are
minimised by a student whose sidecar changes nothing. That says nothing about whether the
table *contains* anything useful, only that this loss cannot ask for it.

This asks directly, with ground-truth cross entropy and no teacher at all. The backbone
is frozen and the only trainable thing is one matrix::

    logits = lm_head(final_hidden + W(features))       # W: 2560 -> 2560, zero-init

At `W = 0` this is exactly the student's own next-token distribution, so the baseline is
free and exact. Whatever `W` then buys is what a *linear* read of the table can add on
top of everything the backbone already knows -- an upper bound on the simplest possible
sidecar, and a lower bound on what a gated non-linear one could reach.

The control is the same training against features rolled across documents. A linear map
onto 248,320 logits has enough freedom to fit something from any correlated input; the
control measures how much, so the real arm can be read against it rather than against
zero.

Scored on assistant tokens only, for the reason the response-only bundle exists: prompt
prediction dominates whole-window numbers and is not what the model is for.

`heldout.jsonl` is a *pool*, not a split -- all 5,598 training documents are inside its
7,200 -- so eligibility comes from the teacher-cache manifests exactly as
`independent_eval prepare` establishes it. Both slices here are drawn from the unseen
remainder and are disjoint from each other, so the eval documents are new to the
backbone and to the readout alike. The absolute NLL is still not comparable to the
reply-bundle arms: different documents, different lengths.

`--inject-layer` moves the injection from the head to a decoder layer, which is the
comparison that matters for a real sidecar. At the head the readout writes straight into
what `lm_head` consumes and its gate reads the final hidden state, so it answers "are
there useful features here" while bypassing the entire transport problem: at layer 1 the
same features have to survive 31 more layers of a frozen backbone, and the gate has to
decide from a layer-1 representation. Expect the layer-1 number to be smaller; the
question is how much.

`--arms` also runs the gate ablation. Upstream computes `hc_count` gates, one per
hyper-connection stream, over a shared value. With a single residual stream those gates
are all functions of the *same* vector, so adding `gate_s * value` to one stream for
every s is exactly `(sum_s gate_s) * value` -- four admission decisions collapse to one
scalar, at four dot products a token and no widening at all. What that leaves open is
whether a gate earns anything here, which the `linear`, `gate1` and `gate4` arms answer
directly.

    python scratch/table_capacity_probe.py --output scratch/gpu-checks/table-capacity.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import threading
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GGUF = os.path.join(
    os.path.expanduser("~"), ".cache", "huggingface", "hub",
    "models--unsloth--Qwen3.8-Flash-Next-GGUF", "snapshots",
    "38bb39ee97821de2c9009abb7e93950eec396e66", "UD-IQ4_XS",
    "Qwen3.8-Flash-Next-UD-IQ4_XS-00002-of-00003.gguf",
)
DEFAULT_STUDENT = str((ROOT / ".." / "student-hf").resolve())
DEFAULT_DOCUMENTS = str((ROOT / ".." / "capture-data" / "heldout.jsonl").resolve())
DEFAULT_MANIFESTS = [str((ROOT / ".." / name / "manifest.json").resolve())
                     for name in ("teacher-cache-1m", "teacher-cache-5m")]


class _PadCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        longest = max(len(f["input_ids"]) for f in features)
        ids, mask = [], []
        for feature in features:
            count = len(feature["input_ids"])
            ids.append(list(feature["input_ids"]) + [self.pad_token_id] * (longest - count))
            mask.append([1] * count + [0] * (longest - count))
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(mask)}


def assistant_mask(text, encoding, length):
    """True at positions whose *target* is an assistant token."""
    from distillkit.independent_eval import role_spans

    spans = role_spans(text, encoding["offset_mapping"]).get("assistant", [])
    mask = torch.zeros(length, dtype=torch.bool)
    for start, stop in spans:
        mask[max(start - 1, 0):min(stop - 1, length)] = True
    return mask


def chunked_ce(hidden, head, targets, mask, chunk=256, backward=False, scale=1.0):
    """Cross entropy in token chunks: the full 248,320-wide logits never exist at once."""
    total = torch.zeros((), device=hidden.device, dtype=torch.float32)
    flat_hidden = hidden.reshape(-1, hidden.shape[-1])
    flat_targets = targets.reshape(-1)
    flat_mask = mask.reshape(-1)
    counted = int(flat_mask.sum())
    for start in range(0, flat_hidden.shape[0], chunk):
        stop = min(start + chunk, flat_hidden.shape[0])
        piece = flat_mask[start:stop]
        if not piece.any():
            continue
        logits = head(flat_hidden[start:stop][piece]).float()
        loss = F.cross_entropy(logits, flat_targets[start:stop][piece], reduction="sum")
        if backward:
            (loss * scale).backward(retain_graph=True)
            del logits
        total = total + loss.detach()
    return total, counted


def per_token_ce(hidden, head, targets, mask, chunk=256):
    """`chunked_ce` without the reduction, plus the target each loss belongs to.

    The aggregate path above is left exactly as it was: it produces the numbers every
    published capacity figure was computed from, and this runs beside it rather than
    replacing it. What it adds is the thing the original never persisted -- which token
    each nat was spent on -- without which the -0.0539 cannot be split into layout and
    content at all.

    Returns losses and targets in mask order, which is row-major over the batch, so a
    caller that also tracks row-to-document mapping can attribute every value.
    """
    losses, kept = [], []
    flat_hidden = hidden.reshape(-1, hidden.shape[-1])
    flat_targets = targets.reshape(-1)
    flat_mask = mask.reshape(-1)
    for start in range(0, flat_hidden.shape[0], chunk):
        stop = min(start + chunk, flat_hidden.shape[0])
        piece = flat_mask[start:stop]
        if not piece.any():
            continue
        picked = flat_targets[start:stop][piece]
        logits = head(flat_hidden[start:stop][piece]).float()
        losses.append(F.cross_entropy(logits, picked, reduction="none").detach().cpu())
        kept.append(picked.detach().cpu())
        del logits
    if not losses:
        return torch.zeros(0), torch.zeros(0, dtype=torch.long)
    return torch.cat(losses), torch.cat(kept)


class Readout(nn.Module):
    """`W(features)`, optionally admitted by a gate read off the residual stream.

    The gate is upstream's, with the query taken where this probe injects rather than at
    the sidecar's layer: RMS-normalise the stream, project onto a learned direction,
    compress with the signed square root, squash.

    `combiner="mean2"` is `2 * mean_i gate_i`, and it is the one to compare with. Summing
    was the first thing tried and it confounds the comparison: with a free value matrix
    `(sum_i g_i) W f = (mean_i g_i)(k W) f`, so k directions summed are not more
    expressive than their mean, they merely start at an effective multiplier of k/2
    instead of 1. Under `mean2` every arm starts at 1.0 with the same reachable range, so
    a difference is shape rather than scale.

    `trainable=False` freezes the directions at their random initialisation, which
    separates "a learned admission criterion helps" from "any non-linear function of the
    stream helps".
    """

    def __init__(self, hidden_size, directions=0, combiner="mean2", trainable=True):
        super().__init__()
        self.value = nn.Linear(hidden_size, hidden_size, bias=False)
        nn.init.zeros_(self.value.weight)
        self.directions = directions
        self.combiner = combiner
        if directions:
            # NOT zero, which is the instinct and is a trap: signed_sqrt has sign(0) = 0
            # and its clamp_min flattens abs() near the origin, so the gradient with
            # respect to a zero direction is exactly 0.0 and the gate is frozen at 0.5
            # for the whole run. Measured: |grad| 0.0 at g = 0, 4.85 at g ~ N(0, 0.02).
            # The first run of this ablation had zero-init gates and therefore compared
            # three constant rescalings of the value path rather than three gates.
            gate = torch.randn(directions, hidden_size) * 0.02
            self.gate = nn.Parameter(gate, requires_grad=trainable)

    def forward(self, features, stream):
        value = self.value(features)
        if not self.directions:
            return value
        normed = stream.float()
        normed = normed * torch.rsqrt(normed.pow(2).mean(-1, keepdim=True) + 1e-6)
        raw = (normed @ self.gate.T) / math.sqrt(stream.shape[-1])
        gate = torch.sigmoid(raw.abs().clamp_min(1e-6).sqrt() * raw.sign())
        combined = (gate.sum(-1, keepdim=True) if self.combiner == "sum"
                    else 2.0 * gate.mean(-1, keepdim=True))
        return value * combined


def batches(documents, tokenizer, collator, table_batch, tokens):
    for start in range(0, len(documents), table_batch):
        chunk = documents[start:start + table_batch]
        encodings = [tokenizer(text, return_offsets_mapping=True) for text in chunk]
        batch = collator([{"input_ids": e["input_ids"][:tokens]} for e in encodings])
        length = batch["input_ids"].shape[1]
        masks = torch.stack([assistant_mask(text, encoding, length)
                             for text, encoding in zip(chunk, encodings)])
        yield batch, masks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf", default=DEFAULT_GGUF)
    parser.add_argument("--student", default=DEFAULT_STUDENT)
    parser.add_argument("--documents", default=DEFAULT_DOCUMENTS)
    parser.add_argument("--manifests", nargs="+", default=DEFAULT_MANIFESTS)
    parser.add_argument("--train-docs", type=int, default=512)
    parser.add_argument("--eval-docs", type=int, default=128)
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--arms", nargs="+", default=["table", "shuffled_control"],
                        choices=["table", "shuffled_control", "gate1", "gate4",
                                 "gate4_sum", "gate4_frozen"])
    parser.add_argument("--combiner", default="mean2", choices=["mean2", "sum"],
                        help="how the per-direction gates become one scalar; mean2 is "
                             "2*mean, which starts every arm at 1.0")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--inject-layer", type=int, default=-1,
                        help="decoder layer to inject before; -1 injects at the head")
    parser.add_argument("--seed", type=int, default=7,
                        help="gate direction init and nothing else; the data order is fixed")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-token", type=Path,
                        help="directory to write per-token NLL, target id and document "
                             "index for the baseline and every arm; this is what lets "
                             "the result be split into layout and content afterwards")
    args = parser.parse_args()
    timer = threading.Timer(5400, lambda: os._exit(124))
    timer.daemon = True
    timer.start()

    from transformers import AutoTokenizer

    from distillkit.models.qwen35_sidecar import Qwen35SidecarForCausalLM
    from distillkit.ngram_hash import NGramHasher
    from distillkit.ngram_table import GGUFNGramTable, IQ4NLDequant
    from distillkit.sidecar_collator import SidecarDataCollator

    tokenizer = AutoTokenizer.from_pretrained(args.student)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    from distillkit.independent_eval import unseen_records

    unseen = unseen_records(args.documents, args.manifests)
    if len(unseen) < args.train_docs + args.eval_docs:
        raise SystemExit("only %d unseen documents; asked for %d"
                         % (len(unseen), args.train_docs + args.eval_docs))
    # Eval taken from the far end so a smaller --train-docs does not silently move which
    # documents are scored, which would make two runs of this probe incomparable.
    train_docs = [record["text"] for record in unseen[:args.train_docs]]
    eval_docs = [record["text"] for record in unseen[-args.eval_docs:]]
    print("documents: %d train, %d eval, from %d unseen"
          % (len(train_docs), len(eval_docs), len(unseen)))

    table = GGUFNGramTable(args.gguf)
    collator = SidecarDataCollator(_PadCollator(pad_id), table, NGramHasher())
    dequant = IQ4NLDequant(out_dtype=torch.float32)

    device = torch.device(args.device)
    model = Qwen35SidecarForCausalLM.from_pretrained(args.student, dtype=torch.bfloat16)
    model.config.use_cache = False
    model.to(device).eval()
    model.requires_grad_(False)
    head = model.get_output_embeddings()
    hidden_size = model.config.hidden_size
    print("student on %s, hidden %d, vocab %d" % (device, hidden_size, model.config.vocab_size))

    def hidden_of(batch):
        with torch.no_grad():
            out = model.model(input_ids=batch["input_ids"].to(device),
                              attention_mask=batch["attention_mask"].to(device),
                              sidecar_enabled=False)
        return out.last_hidden_state

    def hidden_with_injection(batch, readout, features):
        """Run the backbone with the readout writing into `--inject-layer`'s input.

        The backbone's parameters are frozen but the graph still has to be built from the
        injection point onward, which is the cost of asking the question at the place the
        sidecar actually sits.
        """
        injected = {}

        def hook(module, args, kwargs):
            stream = args[0] if args else kwargs["hidden_states"]
            addition = readout(features, stream).to(stream.dtype)
            injected["rms"] = addition.detach().float().pow(2).mean().sqrt().item()
            stream = stream + addition
            if args:
                return (stream,) + tuple(args[1:]), kwargs
            kwargs["hidden_states"] = stream
            return args, kwargs

        handle = model.model.layers[args.inject_layer].register_forward_pre_hook(
            hook, with_kwargs=True)
        try:
            out = model.model(input_ids=batch["input_ids"].to(device),
                              attention_mask=batch["attention_mask"].to(device),
                              sidecar_enabled=False)
        finally:
            handle.remove()
        return out.last_hidden_state, injected.get("rms")

    def features_of(batch, roll):
        features = dequant(batch["ngram_raw"]).flatten(-2).to(device)
        return features.roll(1, dims=0) if roll else features

    def evaluate(readout, roll, collect=None):
        total, counted, per_document = 0.0, 0, []
        document = 0
        for batch, masks in batches(eval_docs, tokenizer, collator, args.batch, args.tokens):
            targets = batch["input_ids"][:, 1:].to(device)
            keep = masks[:, :-1].to(device) & batch["attention_mask"][:, 1:].bool().to(device)
            if readout is not None and args.inject_layer >= 0:
                with torch.no_grad():
                    hidden, _ = hidden_with_injection(batch, readout, features_of(batch, roll))
                state = hidden[:, :-1]
            else:
                state = hidden_of(batch)[:, :-1]
                if readout is not None:
                    state = state + readout(features_of(batch, roll)[:, :-1],
                                            state).to(state.dtype)
            with torch.no_grad():
                loss, count = chunked_ce(state, head, targets, keep)
                for row in range(state.shape[0]):
                    row_loss, row_count = chunked_ce(state[row:row + 1], head,
                                                     targets[row:row + 1], keep[row:row + 1])
                    per_document.append((float(row_loss), row_count))
                    if collect is not None:
                        # Row by row, so the document index is exact rather than inferred
                        # from a flattened batch.
                        row_losses, row_targets = per_token_ce(
                            state[row:row + 1], head, targets[row:row + 1],
                            keep[row:row + 1])
                        collect["nll"].append(row_losses.numpy())
                        collect["target"].append(row_targets.numpy())
                        collect["document"].append(
                            np.full(len(row_losses), document + row, dtype=np.int32))
            document += state.shape[0]
            total += float(loss)
            counted += count
        return total / max(counted, 1), counted, per_document

    results = {"train_docs": len(train_docs), "eval_docs": len(eval_docs),
               "tokens": args.tokens, "lr": args.lr, "epochs": args.epochs, "seed": args.seed,
               "inject_layer": args.inject_layer, "combiner": args.combiner}
    def collector():
        return None if args.per_token is None else {
            "nll": [], "target": [], "document": []}

    def persist(name, collected):
        if collected is None:
            return
        args.per_token.mkdir(parents=True, exist_ok=True)
        packed = {key: np.concatenate(value) if value else np.zeros(0)
                  for key, value in collected.items()}
        np.savez(args.per_token / ("%s.npz" % name), **packed)
        print("  wrote %d per-token values for %s" % (len(packed["nll"]), name))

    baseline_collected = collector()
    baseline, eval_tokens, baseline_documents = evaluate(None, False, baseline_collected)
    persist("baseline", baseline_collected)
    results["baseline_nll"] = baseline
    results["eval_assistant_tokens"] = eval_tokens
    print("baseline assistant NLL %.4f over %d tokens" % (baseline, eval_tokens))

    plan = {
        "table": (False, 0, "mean2", True),
        "shuffled_control": (True, 0, "mean2", True),
        "gate1": (False, 1, args.combiner, True),
        "gate4": (False, 4, args.combiner, True),
        "gate4_sum": (False, 4, "sum", True),
        "gate4_frozen": (False, 4, args.combiner, False),
    }
    for name in args.arms:
        roll, directions, combiner, trainable = plan[name]
        torch.manual_seed(args.seed)
        readout = Readout(hidden_size, directions, combiner, trainable)
        readout = readout.to(device).to(torch.float32)
        optimizer = torch.optim.AdamW(
            [p for p in readout.parameters() if p.requires_grad], lr=args.lr)
        started = time.perf_counter()
        seen = 0
        for epoch in range(args.epochs):
            for index, (batch, masks) in enumerate(
                    batches(train_docs, tokenizer, collator, args.batch, args.tokens)):
                targets = batch["input_ids"][:, 1:].to(device)
                keep = masks[:, :-1].to(device) & batch["attention_mask"][:, 1:].bool().to(device)
                count = int(keep.sum())
                if not count:
                    continue
                if args.inject_layer >= 0:
                    hidden, injection_rms = hidden_with_injection(
                        batch, readout, features_of(batch, roll))
                    state = hidden[:, :-1]
                else:
                    hidden = hidden_of(batch)
                    stream = hidden[:, :-1]
                    state = stream + readout(features_of(batch, roll)[:, :-1],
                                             stream).to(hidden.dtype)
                    injection_rms = None
                optimizer.zero_grad(set_to_none=True)
                loss, _ = chunked_ce(state, head, targets, keep, backward=True, scale=1.0 / count)
                optimizer.step()
                seen += count
                if index % 32 == 0:
                    print("  %s epoch %d batch %d  train NLL %.4f  %d tokens%s"
                          % (name, epoch, index, float(loss) / count, seen,
                             "" if injection_rms is None
                             else "  injection rms %.4f" % injection_rms), flush=True)
        arm_collected = collector()
        trained, _, documents = evaluate(readout, roll, arm_collected)
        persist(name, arm_collected)
        results[name] = {
            "eval_nll": trained,
            "delta_vs_baseline": trained - baseline,
            "readout_norm": readout.value.weight.float().norm().item(),
            "gate_norm": (readout.gate.float().norm().item() if directions else None),
            "combiner": combiner if directions else None,
            "gate_trainable": trainable if directions else None,
            "train_tokens": seen,
            "seconds": time.perf_counter() - started,
            "per_document": documents,
        }
        print("%s: eval NLL %.4f  delta %+.4f  |W| %.3f"
              % (name, trained, trained - baseline, results[name]["readout_norm"]), flush=True)

    def paired(left, right, draws=10000, seed=0):
        """Bootstrap over documents, the unit the arms actually share."""
        import numpy as np
        delta = np.array([a[0] - b[0] for a, b in zip(left, right)])
        tokens = np.array([a[1] for a in left])
        rng = np.random.default_rng(seed)
        index = rng.integers(0, len(delta), (draws, len(delta)))
        samples = delta[index].sum(1) / tokens[index].sum(1)
        return {"estimate": float(delta.sum() / tokens.sum()),
                "ci95": np.quantile(samples, [0.025, 0.975]).tolist(),
                "documents": len(delta)}

    if "table" in args.arms:
        results["paired_against_linear"] = {}
        for name in args.arms:
            if name == "table":
                continue
            stats = paired(results[name]["per_document"], results["table"]["per_document"])
            results["paired_against_linear"][name] = stats
            print("%-14s minus linear  %+.6f [%+.6f, %+.6f] over %d documents"
                  % (name, stats["estimate"], *stats["ci95"], stats["documents"]))

    if "table" in args.arms and "shuffled_control" in args.arms:
        results["table_gain_over_control"] = (results["shuffled_control"]["delta_vs_baseline"]
                                              - results["table"]["delta_vs_baseline"])
    results["quality_evaluation"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2), flush=True)
    timer.cancel()


if __name__ == "__main__":
    main()

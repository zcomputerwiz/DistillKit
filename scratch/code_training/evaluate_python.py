"""Score a backbone and the two frozen post-hoc modules on held-out Python.

Four arms over one backbone -- stock, plus G', plus S, plus both -- on the same held-out
documents in the same order with the same targets. Plain causal CE over 100% of targets:
no assistant masking, no role weighting, no token dropped. That is a deliberate break
from every earlier evaluation in this programme, which scored assistant spans of chat
transcripts; ordinary Python has no roles, and masking anything here would be inventing
structure the data does not have.

The stock arm's per-token NLL is kept in memory and every other arm is differenced
against it position by position, so a delta is a paired comparison over identical targets
rather than two aggregates subtracted. That is what makes the per-document t-statistics
mean anything at 10^-3 nats, which is the resolution the earlier architecture-scale
results live at.

Two mechanical points that are easy to get wrong and invisible afterwards:

*Logits are never materialized for a whole batch.* The vocabulary is 248,320 wide, so a
single 32k-token document's logits would be 32 GB in float32. Hidden states are computed
once and the head is applied in position chunks, which also happens to be where the
sidecar's bias has to be added.

*Padding is on the right and masked out of the loss.* With causal attention a real token
never attends to a later pad, and neither module mixes across positions -- the gate scales
an FFN residual per position, the sidecar biases logits per position -- so padded columns
cannot influence a scored one. Right padding also keeps document-relative positions intact
for the gate's n-gram addressing.

    CUDA_VISIBLE_DEVICES=0 python scratch/code_training/evaluate_python.py \
        --backbone D:/DeepThought/Projects/HybridModel/student-2b-hf \
        --output scratch/code_training/baseline_eval/b0.json
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ffn_memo"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ffn_attenuation"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "residual_gate"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "structural_sidecar"))

import numpy as np
import torch
import torch.nn.functional as F

from distillkit.code_classes import CODE_CLASSES, HISTORICAL_CLASSES
from distillkit.experimental.residual_gate import (
    TrigramFamiliarity, install_residual_gates, remove_residual_gates)
from distillkit.experimental.structural_sidecar import (
    apply_structural_bias, structural_token_ids)
from corpus import (SPLITS, TOKENS, TokenStore, class_tables, corpus_identity,
                    evaluation_subset, load_config, load_tokenizer)

GATE = Path("scratch/gate_regime/plain-harness/gate.pt")
FAMILIARITY_CACHE = Path("scratch/ffn_memo/cache/layer-12.npz")
ARMS = {"stock": (False, False), "gate": (True, False),
        "sidecar": (False, True), "both": (True, True)}
#: Familiarity buckets, in cross-document trigram occurrences. Edges straddle 400 because
#: the original intervention study found the benefit turned positive around there; they
#: are not evenly spaced in a quantity nothing depends on.
COUNT_EDGES = (0, 1, 4, 100, 400, 800, 3000)


def paired(per_document):
    """Mean, standard error and t over per-document deltas -- the established arithmetic."""
    out = {}
    for label, values in per_document.items():
        if not values:
            continue
        mean = statistics.fmean(values)
        error = statistics.stdev(values) / len(values) ** 0.5 if len(values) > 1 else 0.0
        out[label] = {"mean": mean, "stderr": error, "t": mean / error if error else 0.0,
                      "ci95": [mean - 1.96 * error, mean + 1.96 * error],
                      "better": sum(1 for value in values if value < 0), "n": len(values)}
    return out


def batches(store, indices, max_tokens, max_length):
    """Group documents into padded batches, longest-first so padding stays cheap."""
    lengths = [(int(store.offsets[i + 1] - store.offsets[i]), i) for i in indices]
    lengths.sort(reverse=True)
    batch, widest = [], 0
    for length, index in lengths:
        width = max(widest, min(length, max_length))
        if batch and width * (len(batch) + 1) > max_tokens:
            yield batch
            batch, widest = [index], min(length, max_length)
        else:
            batch.append(index)
            widest = width
    if batch:
        yield batch


class Sidecar:
    """The frozen structural sidecar, as a per-chunk logit bias."""

    def __init__(self, module, hasher, structural, strength):
        self.module = module
        self.hasher = hasher
        self.structural = structural
        self.strength = strength

    def rows(self, ids):
        return self.hasher.row_indices(ids)

    def bias(self, rows, start, stop):
        return self.module(rows[:, start:stop], self.strength)

    def apply(self, logits, rows, start, stop):
        bias = self.bias(rows, start, stop)
        return apply_structural_bias(logits, bias, self.structural, 1.0)


@torch.inference_mode()
def run_arm(model, store, indices, code, historical, device, args, handle=None,
            sidecar=None, baseline=None, diagnostics=None):
    """One pass over the evaluation subset; returns per-token NLL plus class totals."""
    per_token = {}
    code_totals = np.zeros((len(CODE_CLASSES), 2), dtype=np.float64)   # nll, count
    per_document = collections.defaultdict(list)
    scored = 0

    for group in batches(store, indices, args.max_batch_tokens, args.max_length):
        rows = [np.asarray(store.document(i)[:args.max_length], dtype=np.int64)
                for i in group]
        width = max(len(r) for r in rows)
        ids = torch.full((len(rows), width), model.config.eos_token_id,
                         dtype=torch.long, device=device)
        mask = torch.zeros((len(rows), width), dtype=torch.long, device=device)
        for position, row in enumerate(rows):
            ids[position, :len(row)] = torch.from_numpy(row).to(device)
            mask[position, :len(row)] = 1

        if handle is not None:
            handle.set_context(ids)
            if diagnostics is not None:
                handle.reset_stats()
        hidden = model.model(input_ids=ids, attention_mask=mask).last_hidden_state
        structural_rows = sidecar.rows(ids) if sidecar is not None else None

        nll = torch.zeros((len(rows), width - 1), dtype=torch.float32, device=device)
        # The head chunk is budgeted in *total* positions, not positions per sequence:
        # the logits it materializes are [batch, chunk, 248320], so a per-sequence chunk
        # silently scales with batch size and a wide batch asks for tens of gigabytes.
        chunk = max(1, args.head_positions // len(rows))
        for start in range(0, width - 1, chunk):
            stop = min(start + chunk, width - 1)
            logits = model.lm_head(hidden[:, start:stop]).float()
            if sidecar is not None:
                logits = sidecar.apply(logits, structural_rows, start, stop)
            nll[:, start:stop] = F.cross_entropy(
                logits.transpose(1, 2), ids[:, start + 1:stop + 1], reduction="none")

        if diagnostics is not None and handle is not None:
            diagnostics.observe(handle, ids, mask, code)

        host = nll.to("cpu").numpy()
        for position, index in enumerate(group):
            length = len(rows[position]) - 1
            values = host[position, :length].astype(np.float64)
            targets = rows[position][1:length + 1]
            per_token[index] = values
            labels = code[targets]
            np.add.at(code_totals[:, 0], labels, values)
            np.add.at(code_totals[:, 1], labels, 1.0)
            scored += length
            if baseline is not None:
                delta = values - baseline[index]
                per_document["all"].append(float(delta.mean()))
                for label in np.unique(labels):
                    selected = delta[labels == label]
                    per_document[CODE_CLASSES[label]].append(float(selected.mean()))
                # Namespaced: "content" names a code class and a historical class, and
                # they are different sets of tokens -- historical content is code content
                # plus keywords. Pooling them under one key would average two different
                # quantities into a number that is neither.
                hist = historical[targets]
                for label in np.unique(hist):
                    selected = delta[hist == label]
                    per_document["hist/" + HISTORICAL_CLASSES[label]].append(
                        float(selected.mean()))

    result = {
        "tokens": scored,
        "total_nll": float(code_totals[:, 0].sum()),
        "mean_nll": float(code_totals[:, 0].sum() / max(scored, 1)),
        "code_classes": {
            name: {"nll": float(code_totals[i, 0] / code_totals[i, 1]) if code_totals[i, 1]
                   else None, "tokens": int(code_totals[i, 1]),
                   "total_nll": float(code_totals[i, 0])}
            for i, name in enumerate(CODE_CLASSES)},
    }
    # The historical five-way view is a sum over code classes, never an independent
    # classification, so the two can never disagree.
    from distillkit.code_classes import HISTORICAL

    rolled = collections.defaultdict(lambda: [0.0, 0])
    for i, name in enumerate(CODE_CLASSES):
        bucket = rolled[HISTORICAL[name]]
        bucket[0] += code_totals[i, 0]
        bucket[1] += int(code_totals[i, 1])
    result["historical_classes"] = {
        name: {"nll": (total / count) if count else None, "tokens": count,
               "total_nll": total}
        for name, (total, count) in sorted(rolled.items())}
    if baseline is not None:
        result["delta"] = paired(per_document)
    return result, per_token


class GateDiagnostics:
    """Gate behaviour on Python: reach, per-layer values, familiarity and class response.

    The question this answers is narrow and important: the gate learned a monotone policy
    in trigram familiarity on general text, and its familiarity cache was built from that
    text. If ordinary Python falls almost entirely into the unfamiliar bucket, the gate is
    effectively a constant here and any improvement it shows is not the learned policy.
    """

    def __init__(self, familiarity, vocab_size):
        self.familiarity = familiarity
        self.vocab_size = vocab_size
        self.layer_sum = collections.defaultdict(float)
        self.layer_count = collections.defaultdict(int)
        self.bucket = collections.defaultdict(lambda: [0.0, 0])
        self.by_class = collections.defaultdict(lambda: [0.0, 0])

    def observe(self, handle, ids, mask, code):
        # ``kept[layer]`` holds one [batch, sequence] tensor per forward call, and there
        # is exactly one call per batch here.
        keep = mask.bool().cpu()
        counts = torch.expm1(self.familiarity.features(ids)[..., 0]).cpu()[keep]
        labels = torch.from_numpy(code[ids.cpu().numpy()].astype(np.int64))[keep]
        buckets = [(counts >= low) & (counts < high)
                   for low, high in zip(COUNT_EDGES, COUNT_EDGES[1:] + (float("inf"),))]
        for layer, parts in handle.kept.items():
            selected = torch.cat([p.float() for p in parts], dim=0)[keep]
            self.layer_sum[layer] += float(selected.sum())
            self.layer_count[layer] += int(selected.numel())
            for (low, high), chosen in zip(
                    zip(COUNT_EDGES, COUNT_EDGES[1:] + (float("inf"),)), buckets):
                values = selected[chosen]
                if values.numel():
                    entry = self.bucket["[%g, %g)" % (low, high)]
                    entry[0] += float(values.sum())
                    entry[1] += int(values.numel())
            for index, name in enumerate(CODE_CLASSES):
                values = selected[labels == index]
                if values.numel():
                    entry = self.by_class[name]
                    entry[0] += float(values.sum())
                    entry[1] += int(values.numel())

    def report(self, handle):
        mean = lambda pair: pair[0] / pair[1] if pair[1] else None
        total = max(sum(v[1] for v in self.bucket.values()), 1)
        return {
            # ``reach`` is the largest deviation from unit admission this gate could
            # produce for any input -- zero at initialization by construction, so it is
            # the one-number answer to whether the gate learned anything at all.
            "reach": {str(layer): handle.gate(layer).gate_report("g")["g/reach"]
                      for layer in handle.layer_indices},
            "mean_by_layer": {str(layer): self.layer_sum[layer] / self.layer_count[layer]
                              for layer in sorted(self.layer_sum)},
            "mean_by_familiarity": {key: mean(value)
                                    for key, value in sorted(self.bucket.items())},
            "familiarity_token_share": {key: value[1] / total
                                        for key, value in sorted(self.bucket.items())},
            "mean_by_target_class": {key: mean(value)
                                     for key, value in sorted(self.by_class.items())},
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--label", default=None, help="name for this backbone in the report")
    parser.add_argument("--split", default="heldout", choices=SPLITS)
    parser.add_argument("--subset-tokens", type=int, default=0,
                        help="0 evaluates the whole split")
    parser.add_argument("--arms", default="stock,gate,sidecar,both")
    parser.add_argument("--gate", type=Path, default=GATE)
    parser.add_argument("--cache", type=Path, default=FAMILIARITY_CACHE)
    parser.add_argument("--max-batch-tokens", type=int, default=16384)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--head-positions", type=int, default=4096,
                        help="total positions per head chunk; logits are batch*chunk*vocab")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--memory-fraction", type=float, default=0.92,
                        help="Windows WDDM pages instead of raising OOM; this restores it")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if torch.cuda.device_count() != 1:
        raise SystemExit("mask to one card with CUDA_VISIBLE_DEVICES")
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction, 0)

    from transformers import AutoConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    from distillkit.independent_eval import build_token_classes

    started = time.monotonic()
    config = AutoConfig.from_pretrained(args.backbone, local_files_only=True)
    config = getattr(config, "text_config", config)
    config.use_cache = False
    model = Qwen3_5ForCausalLM.from_pretrained(
        args.backbone, config=config, dtype=torch.bfloat16,
        local_files_only=True).to(args.device).eval()
    model.requires_grad_(False)

    tokenizer = load_tokenizer()
    code, historical = class_tables(tokenizer, config.vocab_size)
    store = TokenStore(TOKENS, args.split)
    indices, subset = evaluation_subset(store, args.subset_tokens)
    print("%s: %d documents, %d tokens (digest %s)"
          % (args.split, subset["documents"], subset["tokens"], subset["digest"][:12]), flush=True)

    report = {"backbone": str(args.backbone), "label": args.label or Path(args.backbone).name,
              "corpus": corpus_identity(), "evaluation_subset": subset,
              "split": args.split, "arms": {}}

    baseline = None
    for arm in args.arms.split(","):
        use_gate, use_sidecar = ARMS[arm]
        handle = diagnostics = sidecar = None
        if use_gate:
            payload = torch.load(args.gate, map_location="cpu", weights_only=False)
            familiarity = TrigramFamiliarity(args.cache, config.vocab_size,
                                             device=args.device)
            handle = install_residual_gates(model, payload["layers"],
                                            family=payload["family"],
                                            familiarity=familiarity)
            model.residual_gates.load_state_dict(payload["state_dict"], strict=True)
            model.residual_gates.requires_grad_(False)
            handle.keep = True
            diagnostics = GateDiagnostics(familiarity, config.vocab_size)
            report.setdefault("gate_checkpoint", {"path": str(args.gate),
                                                  "layers": payload["layers"],
                                                  "mask": payload.get("mask"),
                                                  "regime": payload.get("regime")})
        if use_sidecar:
            from factorize import class_ids
            from gate_after_structure import build_structural
            from evaluate import split_layout

            classes = split_layout(build_token_classes(tokenizer, config.vocab_size),
                                   tokenizer, config.vocab_size)
            structural = structural_token_ids(classes, device=args.device)
            whitespace = class_ids(classes, "whitespace", device=args.device)
            module, hasher, strength = build_structural(config, structural, whitespace,
                                                        args.device)
            sidecar = Sidecar(module, hasher, structural, strength)

        began = time.monotonic()
        result, per_token = run_arm(model, store, indices, code, historical, args.device,
                                    args, handle=handle, sidecar=sidecar,
                                    baseline=baseline, diagnostics=diagnostics)
        result["seconds"] = time.monotonic() - began
        if diagnostics is not None:
            result["gate_diagnostics"] = diagnostics.report(handle)
        if arm == "stock":
            baseline = per_token
        report["arms"][arm] = result
        print("%-9s mean NLL %.6f over %d tokens  (%.0f s)"
              % (arm, result["mean_nll"], result["tokens"], result["seconds"]), flush=True)
        if handle is not None:
            remove_residual_gates(model)

    report["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(args.device))
    report["elapsed_seconds"] = time.monotonic() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("wrote %s" % args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

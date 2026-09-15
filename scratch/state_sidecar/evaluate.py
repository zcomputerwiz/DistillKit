"""Score the state-conditioned sidecar and its controls on the frozen Python heldout subset.

Arms, in the order the question needs them:

    stock          B_code alone
    blind          the matched control, g = 1, trained from identical initialization
    conditioned    the compatibility gate
    conditioned_g1 the *trained* conditioned module with g forced to 1 at evaluation --
                   the parameter-matched ablation, which separates "the gate did the work"
                   from "the extra capacity did"
    conditioned_wrong  real hash rows read from the wrong position

The stock arm is scored first and every other arm is differenced against it token by token,
so a delta is a paired comparison over identical targets rather than two aggregates
subtracted. The evaluation subset, its digest and the head-chunk size all match the
previous tasks exactly, so these numbers sit on the same scale as the G_code and S_code
tables.

    CUDA_VISIBLE_DEVICES=0 python scratch/state_sidecar/evaluate.py
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code_training"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code_gate"))

import numpy as np
import torch
import torch.nn.functional as F

from distillkit.code_classes import CODE_CLASSES, HISTORICAL, HISTORICAL_CLASSES
from distillkit.experimental.ngram_hash import NGramHashConfig, NGramHasher
from distillkit.experimental.state_sidecar import (
    StateConditionedSidecar, install_state_sidecar, remove_state_sidecar)
from corpus import TOKENS, TokenStore, class_tables, corpus_identity, load_tokenizer
from evaluate_python import batches, paired

BCODE = Path("scratch/code_training/checkpoints/v1/final")
ROOT = Path("scratch/state_sidecar")
ARMS = ("stock", "blind", "conditioned", "conditioned_g1", "conditioned_wrong")


@torch.inference_mode()
def run(model, handle, store, indices, code, historical, args, baseline=None,
        collect=False):
    per_token, per_document = {}, collections.defaultdict(list)
    totals = np.zeros((len(CODE_CLASSES), 2), dtype=np.float64)
    gates, compatibilities, ratios = [], [], []
    scored = 0

    for group in batches(store, indices, args.max_batch_tokens, args.max_length):
        rows = [np.asarray(store.document(i)[:args.max_length], dtype=np.int64)
                for i in group]
        width = max(len(r) for r in rows)
        ids = torch.full((len(rows), width), model.config.eos_token_id,
                         dtype=torch.long, device=args.device)
        mask = torch.zeros((len(rows), width), dtype=torch.long, device=args.device)
        for slot, row in enumerate(rows):
            ids[slot, :len(row)] = torch.from_numpy(row).to(args.device)
            mask[slot, :len(row)] = 1
        if handle is not None:
            handle.set_context(ids)
            handle.collect = collect
        hidden = model.model(input_ids=ids, attention_mask=mask).last_hidden_state
        if collect and handle is not None and handle.sidecar.last:
            keep = mask.bool().cpu()
            last = handle.sidecar.last
            gates.append(last["gate"].float().cpu()[keep].numpy())
            if last["compatibility"] is not None:
                compatibilities.append(
                    last["compatibility"].float().cpu()[keep].numpy())
            ratios.append((last["correction_norm"] / last["residual_norm"]
                           ).float().cpu()[keep].numpy())
        nll = torch.zeros((len(rows), width - 1), dtype=torch.float32,
                          device=args.device)
        chunk = max(1, args.head_positions // len(rows))
        for start in range(0, width - 1, chunk):
            stop = min(start + chunk, width - 1)
            logits = model.lm_head(hidden[:, start:stop]).float()
            nll[:, start:stop] = F.cross_entropy(
                logits.transpose(1, 2), ids[:, start + 1:stop + 1], reduction="none")
        host = nll.to("cpu").numpy()
        for slot, index in enumerate(group):
            span = len(rows[slot]) - 1
            values = host[slot, :span].astype(np.float64)
            targets = rows[slot][1:span + 1]
            per_token[index] = values
            labels = code[targets]
            np.add.at(totals[:, 0], labels, values)
            np.add.at(totals[:, 1], labels, 1.0)
            scored += span
            if baseline is not None:
                delta = values - baseline[index]
                per_document["all"].append(float(delta.mean()))
                for label in np.unique(labels):
                    per_document[CODE_CLASSES[label]].append(
                        float(delta[labels == label].mean()))
                hist = historical[targets]
                for label in np.unique(hist):
                    per_document["hist/" + HISTORICAL_CLASSES[label]].append(
                        float(delta[hist == label].mean()))

    rolled = collections.defaultdict(lambda: [0.0, 0])
    for index, name in enumerate(CODE_CLASSES):
        bucket = rolled[HISTORICAL[name]]
        bucket[0] += totals[index, 0]
        bucket[1] += int(totals[index, 1])
    result = {
        "tokens": scored,
        "mean_nll": float(totals[:, 0].sum() / max(scored, 1)),
        "code_classes": {
            name: {"nll": float(totals[i, 0] / totals[i, 1]) if totals[i, 1] else None,
                   "tokens": int(totals[i, 1])}
            for i, name in enumerate(CODE_CLASSES)},
        "historical_classes": {
            name: {"nll": total / count if count else None, "tokens": count}
            for name, (total, count) in sorted(rolled.items())},
    }
    if baseline is not None:
        result["delta"] = paired(per_document)
    if gates:
        flat = np.concatenate(gates)
        result["gate"] = {
            "mean": float(flat.mean()), "p10": float(np.percentile(flat, 10)),
            "p50": float(np.percentile(flat, 50)), "p90": float(np.percentile(flat, 90)),
            "min": float(flat.min()), "max": float(flat.max()),
            "std": float(flat.std()), "tokens": int(flat.size)}
        ratio = np.concatenate(ratios)
        result["correction_over_residual"] = {
            "mean": float(ratio.mean()), "p50": float(np.percentile(ratio, 50)),
            "p90": float(np.percentile(ratio, 90)), "max": float(ratio.max())}
        if compatibilities:
            score = np.concatenate(compatibilities)
            result["compatibility"] = {
                "mean": float(score.mean()), "p10": float(np.percentile(score, 10)),
                "p50": float(np.percentile(score, 50)),
                "p90": float(np.percentile(score, 90)),
                "min": float(score.min()), "max": float(score.max())}
    return result, per_token


def load_sidecar(path: Path, config, device):
    payload = torch.load(path / "sidecar.pt", map_location="cpu", weights_only=False)
    sidecar = StateConditionedSidecar(
        hidden_size=config.hidden_size, code_dim=payload["code_dim"],
        memory_dim=payload["memory_dim"], seed=payload["seed"]).to(device).to(torch.float32)
    sidecar.load_state_dict(payload["state_dict"])
    sidecar.requires_grad_(False)
    return sidecar, payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", type=Path, default=BCODE)
    parser.add_argument("--blind", type=Path, default=ROOT / "blind")
    parser.add_argument("--conditioned", type=Path, default=ROOT / "conditioned")
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--subset-tokens", type=int, default=1_250_000)
    parser.add_argument("--max-batch-tokens", type=int, default=16384)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--head-positions", type=int, default=4096)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--memory-fraction", type=float, default=0.92)
    parser.add_argument("--output", type=Path, default=ROOT / "heldout.json")
    args = parser.parse_args()

    if torch.cuda.device_count() != 1:
        raise SystemExit("mask to one card with CUDA_VISIBLE_DEVICES")
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction, 0)

    from transformers import AutoConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    from corpus import evaluation_subset
    from train_gate import digest_of

    config = AutoConfig.from_pretrained(args.backbone, local_files_only=True)
    config = getattr(config, "text_config", config)
    config.use_cache = False
    model = Qwen3_5ForCausalLM.from_pretrained(
        args.backbone, config=config, dtype=torch.bfloat16,
        local_files_only=True).to(args.device).eval()
    model.requires_grad_(False)
    before = digest_of(model, skip="state_sidecar")

    tokenizer = load_tokenizer()
    code, historical = class_tables(tokenizer, config.vocab_size)
    store = TokenStore(TOKENS, "heldout")
    indices, subset = evaluation_subset(store, args.subset_tokens)
    hasher = NGramHasher(NGramHashConfig(
        vocab_size=config.vocab_size, ngram_size=3, heads_per_ngram=1,
        ngram_vocab_size_base=(1 << 17) // 2, seed=1234,
        eos_token_id=tokenizer.eos_token_id or config.vocab_size - 1))
    print("heldout: %d documents, %d tokens (digest %s)"
          % (subset["documents"], subset["tokens"], subset["digest"][:12]), flush=True)

    report = {"backbone": str(args.backbone), "layer": None,
              "corpus": corpus_identity(), "evaluation_subset": subset, "arms": {}}
    baseline = None
    started = time.monotonic()

    for arm in args.arms.split(","):
        handle = None
        if arm != "stock":
            source = args.blind if arm == "blind" else args.conditioned
            sidecar, payload = load_sidecar(source, config, args.device)
            report["layer"] = payload["layer"]
            report.setdefault("checkpoints", {})[arm] = {
                "path": str(source), "arm": payload["arm"], "step": payload["step"],
                "parameters": sidecar.parameter_report()}
            if arm == "conditioned_g1":
                sidecar.state_conditioned = False
            if arm == "conditioned_wrong":
                sidecar.wrong_context = True
            handle = install_state_sidecar(model, sidecar, hasher, payload["layer"])
        began = time.monotonic()
        result, per_token = run(model, handle, store, indices, code, historical, args,
                                baseline=baseline,
                                collect=arm in ("conditioned", "conditioned_wrong"))
        result["seconds"] = time.monotonic() - began
        if arm == "stock":
            baseline = per_token
        report["arms"][arm] = result
        print("%-18s mean NLL %.6f over %d tokens  (%.0f s)"
              % (arm, result["mean_nll"], result["tokens"], result["seconds"]), flush=True)
        if handle is not None:
            remove_state_sidecar(model)

    after = digest_of(model, skip="state_sidecar")
    report["backbone_sha256_before"] = before
    report["backbone_sha256_after"] = after
    report["backbone_frozen"] = before == after
    report["elapsed_seconds"] = time.monotonic() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("backbone frozen: %s\nwrote %s" % (report["backbone_frozen"], args.output))
    return 0 if report["backbone_frozen"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

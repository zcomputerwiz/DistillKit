"""Independent evaluation; never imports the trainer or a distillation objective.

See docs/independent_eval.md for the protocol and the deliberately different GR
table ablation and complete-adapter bypass. Run with ``python -m
distillkit.independent_eval --help`` from the repository root.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import threading
import time

import numpy as np
import torch
import torch.nn.functional as F

from distillkit.sidecar_collator import SidecarDataCollator


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def partition(key):
    # Membership cannot change when the requested sample size or source order changes.
    return "screen" if int(digest(["independent-eval-v1", key])[:8], 16) % 2 == 0 else "confirmation"


def select_split(records, count, split):
    eligible = [r for r in records if partition(r["id"]) == split]
    eligible.sort(key=lambda r: digest(["order-v1", r["id"]]))
    if len(eligible) < count:
        raise ValueError(f"need {count} {split} records, found {len(eligible)}")
    return eligible[:count]


def unseen_records(path, manifests):
    if not manifests:
        raise ValueError("cache manifests are required to establish unseen document IDs")
    used = set()
    for manifest in manifests:
        used.update(str(d["doc_id"]) for d in json.loads(Path(manifest).read_text())["documents"])
    records, seen_ids, seen_text = [], set(), set()
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            key, text_hash = str(row["doc_id"]), digest(row["text"])
            if key in used or key in seen_ids or text_hash in seen_text:
                continue
            seen_ids.add(key)
            seen_text.add(text_hash)
            records.append({"id": key, "text": row["text"]})
    return records


def continuation_tokens(tokenizer, prompt, continuation):
    # Joint tokenization avoids inventing a different conditional distribution at a
    # BPE boundary. Fail closed if the prompt's final token is merged into the answer.
    prompt = prompt.rstrip()
    prefix = tokenizer.encode(prompt, add_special_tokens=False)
    joined = tokenizer.encode(prompt + " " + continuation, add_special_tokens=False)
    if not prefix or joined[:len(prefix)] != prefix or len(joined) <= len(prefix):
        raise ValueError("continuation boundary is not a token boundary")
    return {"ids": joined, "start": len(prefix), "chars": len(continuation)}


def benchmark_record(task, row, tokenizer, max_length):
    if task == "mmlu":
        choices = list("ABCD")
        prompt = row["question"].strip() + "\n" + "\n".join(
            f"{letter}. {choice}" for letter, choice in zip(choices, row["choices"])
        ) + "\nAnswer:"
        answer = int(row["answer"])
    else:
        prompt = f"Question: {row['question']}\nAnswer:"
        choices = row["choices"]["text"]
        answer = row["choices"]["label"].index(str(row["answerKey"]))
    encoded = [continuation_tokens(tokenizer, prompt, choice) for choice in choices]
    # Dropping only after deterministic selection would make a hidden difficulty filter.
    # Refuse long questions instead of truncating away choices or the answer.
    if max(len(c["ids"]) for c in encoded) > max_length:
        raise ValueError(f"{task} question exceeds max length {max_length}; increase --max-question-tokens")
    return {"id": task + ":" + digest([prompt, choices]), "task": task,
            "choices": encoded, "answer": answer, "subject": row.get("subject"),
            "source_id": row.get("id"), "prompt": prompt, "choice_text": choices}


def read_benchmark(path, dataset, config):
    if path:
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(obj, dict) and "rows" in obj:
            return [r.get("row", r) for r in obj["rows"]]
        return obj
    from datasets import load_dataset
    return list(load_dataset(dataset, config, split="test"))


def prepare(args):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    manifests = [Path(p) for p in args.manifests]
    unseen = unseen_records(args.documents, manifests)
    docs = []
    for row in unseen:
        ids = tokenizer.encode(row["text"], add_special_tokens=True)
        if len(ids) >= args.min_document_tokens:
            docs.append({"id": row["id"], "task": "nll", "ids": ids[:args.document_tokens],
                         "text_sha256": digest(row["text"])})
    banks = {"nll": docs}
    if not args.text_only:
        for task, source, dataset, config in [
            ("mmlu", args.mmlu_json, "cais/mmlu", "all"),
            ("arc", args.arc_json, "allenai/ai2_arc", "ARC-Challenge"),
        ]:
            rows = read_benchmark(source, dataset, config)
            bank = [benchmark_record(task, r, tokenizer, args.max_question_tokens) for r in rows]
            banks[task] = list({r["id"]: r for r in bank}.values())
    bundle = {"protocol": "independent-eval-v1", "tokenizer": str(Path(args.tokenizer).resolve()),
              "tokenizer_sha256": digest(tokenizer.backend_tokenizer.to_str()),
              "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
              "unseen_document_count": len(unseen), "eligible_document_count": len(docs),
              "manifests": {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in manifests},
              "source": {"mmlu": "cais/mmlu all test", "arc": "allenai/ai2_arc ARC-Challenge test"},
              "protocol_notes": "0-shot, no chat template; MMLU labels, ARC answer text; no question truncation",
              "splits": {}}
    for split in ("screen", "confirmation"):
        bundle["splits"][split] = {task: select_split(bank, args.docs if task == "nll" else args.questions, split)
                                    for task, bank in banks.items()}
    write_json(args.output, bundle)
    print(json.dumps({"output": args.output, "unseen": len(unseen),
                      "counts_per_split": {k: len(v) for k, v in bundle["splits"]["screen"].items()}}))


class TextCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        length = max(len(f["ids"]) for f in features)
        ids = torch.full((len(features), length), self.pad_token_id, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i, feature in enumerate(features):
            ids[i, :len(feature["ids"])] = torch.tensor(feature["ids"])
            mask[i, :len(feature["ids"])] = 1
        return {"input_ids": ids, "attention_mask": mask}


def make_collator(pad_token_id, table=None, hasher=None):
    base = TextCollator(pad_token_id)
    return base if table is None else SidecarDataCollator(base, table, hasher)


def validate_loading(info, has_sidecar):
    # The only intentionally unused weights exist exclusively for teacher loss.
    problems = {k: v for k, v in info.items() if k in ("missing_keys", "mismatched_keys", "error_msgs") and v}
    unexpected = [k for k in info.get("unexpected_keys", []) if not k.startswith("distillation_projections.")]
    if problems or unexpected:
        raise ValueError(f"checkpoint did not load exactly (sidecar={has_sidecar}): {problems}, unexpected={unexpected}")


def load_checkpoint(path, device="cpu", dtype=torch.float32):
    from safetensors import safe_open
    from transformers import AutoConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
    from distillkit.models.qwen35_sidecar import Qwen35SidecarForCausalLM

    path = Path(path)
    config = AutoConfig.from_pretrained(path, local_files_only=True)
    config = getattr(config, "text_config", config)
    stored = {}
    for file in sorted(path.glob("model*.safetensors")):
        with safe_open(file, framework="pt") as handle:
            for key in handle.keys():
                if ".sidecar." in key:
                    stored[key] = handle.get_tensor(key)
    declared = "Qwen35SidecarForCausalLM" in (config.architectures or [])
    has_sidecar = declared or bool(stored)
    if has_sidecar and (not stored or not hasattr(config, "sidecar_variant")):
        raise ValueError("sidecar checkpoint is missing its adapter weights or saved variant")
    cls = Qwen35SidecarForCausalLM if has_sidecar else Qwen3_5ForCausalLM
    model, info = cls.from_pretrained(path, config=config, local_files_only=True,
                                      dtype=dtype, output_loading_info=True, attn_implementation="sdpa")
    validate_loading(info, has_sidecar)
    actual = {k: v for k, v in model.state_dict().items() if ".sidecar." in k}
    if set(actual) != set(stored):
        raise ValueError("saved adapter keys do not match instantiated architecture")
    for key in actual:
        if not torch.equal(actual[key].cpu(), stored[key].to(dtype=actual[key].dtype)):
            raise ValueError(f"adapter tensor failed exact checkpoint verification: {key}")
    audit = {"variant": getattr(config, "sidecar_variant", None) if has_sidecar else None,
             "adapter_tensor_count": len(stored), "adapter_exact_match": True,
             "adapter_norms": {k: v.float().norm().item() for k, v in stored.items()},
             "ignored_loss_only_keys": sorted(info.get("unexpected_keys", [])),
             "config_sha256": hashlib.sha256((path / "config.json").read_bytes()).hexdigest()}
    model.requires_grad_(False)
    return model.to(device).eval(), audit


@contextmanager
def complete_bypass(model, active):
    layer = model.model.layers[model.config.sidecar_layer_index] if active else None
    saved = layer.sidecar if layer is not None else None
    try:
        if layer is not None:
            layer.sidecar = None
        yield
    finally:
        if layer is not None:
            layer.sidecar = saved


def forward_logits(model, batch, mode, positions):
    has_sidecar = hasattr(model.config, "sidecar_variant")
    kwargs = dict(batch, use_cache=False, logits_to_keep=positions)
    if has_sidecar:
        kwargs["sidecar_enabled"] = mode == "enabled"
    else:
        kwargs.pop("ngram_raw", None)
    with complete_bypass(model, has_sidecar and mode == "full_bypass"):
        return model(**kwargs).logits


@torch.inference_mode()
def score_sequences(model, features, collator, mode, device):
    batch = {k: v.to(device) for k, v in collator(features).items()}
    positions = sorted({p for f in features for p in range(f.get("start", 1) - 1, len(f["ids"]) - 1)})
    if not positions:
        raise ValueError("no causal targets to score")
    positions_tensor = torch.tensor(positions, device=device)
    logits = forward_logits(model, batch, mode, positions_tensor)
    lookup = {p: i for i, p in enumerate(positions)}
    results = []
    for i, feature in enumerate(features):
        start = feature.get("start", 1)
        indexes = [lookup[p] for p in range(start - 1, len(feature["ids"]) - 1)]
        target = torch.tensor(feature["ids"][start:], device=device)
        selected = logits[i, indexes].float()
        loss = F.cross_entropy(selected, target, reduction="sum").item()
        if not np.isfinite(loss):
            raise ValueError("non-finite next-token NLL")
        results.append({"sum_nll": loss, "tokens": len(target)})
    return results


def choice_result(scores, choices, answer):
    sums = np.array([s["sum_nll"] for s in scores])
    lengths = np.array([s["tokens"] for s in scores])
    predictions = {"acc": int(np.argmin(sums)), "acc_token_norm": int(np.argmin(sums / lengths)),
                   "acc_char_norm": int(np.argmin(sums / np.array([c["chars"] for c in choices])))}
    return {"scores": scores, "predictions": predictions,
            **{name: int(prediction == answer) for name, prediction in predictions.items()},
            "normalization_disagrees": len(set(predictions.values())) > 1}


@torch.inference_mode()
def plumbing_probe(model, collator, feature, device):
    if not hasattr(model.config, "sidecar_variant"):
        return {"stock_reference": True}
    sidecar = model.model.layers[model.config.sidecar_layer_index].sidecar
    observed = []

    def hook(module, args, output):
        observed.append({"enabled": bool(args[2]), "raw_present": args[1] is not None,
                         "residual_max": (output - args[0]).float().abs().max().item()})

    handle = sidecar.register_forward_hook(hook)
    batch = {k: v.to(device) for k, v in collator([feature]).items()}
    position = torch.tensor([len(feature["ids"]) - 2], device=device)
    try:
        enabled = forward_logits(model, batch, "enabled", position).float()
        disabled = forward_logits(model, batch, "bypassed", position).float()
    finally:
        handle.remove()
    if len(observed) != 2 or not observed[0]["enabled"] or observed[1]["enabled"] or not observed[0]["raw_present"]:
        raise ValueError("evaluation silently bypassed the sidecar or failed to forward its data/flag")
    return {"sidecar_calls": observed, "enabled_minus_bypassed_logits_max": (enabled - disabled).abs().max().item()}


def evaluate(args):
    started = time.monotonic()
    # A stalled kernel must not turn this bounded evaluation into an overnight job.
    timer = threading.Timer(args.max_seconds, lambda: os._exit(124))
    timer.daemon = True
    timer.start()
    try:
        bundle = json.loads(Path(args.bundle).read_text(encoding="utf-8"))
        tasks = bundle["splits"][args.split]
        tasks = {k: v[:args.limit] if args.limit else v for k, v in tasks.items() if k in args.tasks}
        if set(tasks) != set(args.tasks) or any(not v for v in tasks.values()):
            raise ValueError("requested evaluation tasks are missing or empty")
        model, audit = load_checkpoint(args.checkpoint, args.device, torch.bfloat16)
        table = None
        if audit["variant"]:
            from distillkit.ngram_table import GGUFNGramTable
            if not args.table:
                raise ValueError("--table is required for a sidecar checkpoint")
            table = GGUFNGramTable(args.table)
        collator = make_collator(bundle["pad_token_id"], table)
        first = next(iter(tasks.values()))[0]
        feature = first if "ids" in first else first["choices"][0]
        probe = plumbing_probe(model, collator, feature, args.device)
        modes = ["enabled", "bypassed"] + (["full_bypass"] if audit["variant"] == "gated_residual" else [])
        result = {"checkpoint": str(Path(args.checkpoint).resolve()), "split": args.split,
                  "bundle_sha256": digest(bundle), "tokenizer_sha256": bundle["tokenizer_sha256"],
                  "task_sha256": {k: digest(v) for k, v in tasks.items()}, "audit": audit, "probe": probe,
                  "dtype": "bfloat16", "device": args.device, "complete": False, "records": {}}
        for task, records in tasks.items():
            result["records"][task] = []
            for index, record in enumerate(records):
                outputs = {}
                for mode in modes:
                    if not audit["variant"] and mode == "bypassed":
                        outputs[mode] = outputs["enabled"]
                        continue
                    features = [record] if task == "nll" else record["choices"]
                    scored = score_sequences(model, features, collator, mode, args.device)
                    outputs[mode] = scored[0] if task == "nll" else choice_result(scored, features, record["answer"])
                result["records"][task].append({"id": record["id"], "modes": outputs})
                if (index + 1) % 8 == 0:
                    print(f"{Path(args.checkpoint).name} {task} {index + 1}/{len(records)} {time.monotonic()-started:.1f}s", flush=True)
                    write_json(str(args.output) + ".partial", result)
        result["elapsed_seconds"] = time.monotonic() - started
        result["complete"] = True
        write_json(args.output, result)
        print(json.dumps({"output": args.output, "elapsed_seconds": result["elapsed_seconds"], "probe": probe}), flush=True)
    finally:
        timer.cancel()


def paired_interval(a, b=None, denominators=None, draws=10000, seed=20260909):
    a = np.asarray(a, dtype=np.float64)
    b = np.zeros_like(a) if b is None else np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1 or not len(a):
        raise ValueError("paired observations must have matching nonempty shapes")
    delta = a - b
    denominator = np.ones_like(a) if denominators is None else np.asarray(denominators, dtype=np.float64)
    if denominator.shape != a.shape or np.any(denominator <= 0):
        raise ValueError("invalid paired denominators")
    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, len(a), (draws, len(a)))
    samples = delta[indexes].sum(1) / denominator[indexes].sum(1)
    return {"estimate": float(delta.sum() / denominator.sum()),
            "ci95": np.quantile(samples, [0.025, 0.975]).tolist(), "units": len(a)}


def compare_results(result, reference, draws=10000):
    if not result["complete"] or not reference["complete"]:
        raise ValueError("refusing to report an incomplete evaluation")
    if result["split"] != reference["split"] or result["tokenizer_sha256"] != reference["tokenizer_sha256"]:
        raise ValueError("reference split/tokenizer mismatch")
    rows = []
    for task, records in result["records"].items():
        refs = reference["records"][task]
        if result["task_sha256"][task] != reference["task_sha256"][task] or [r["id"] for r in records] != [r["id"] for r in refs]:
            raise ValueError("paired evaluation records differ")
        metrics = ["nll"] if task == "nll" else ["acc", "acc_token_norm", "acc_char_norm"]
        modes = list(records[0]["modes"])
        values = {mode: [r["modes"][mode] for r in records] for mode in modes}
        values["pre_retrofit"] = [r["modes"]["enabled"] for r in refs]
        comparisons = [(mode, None) for mode in values]
        comparisons += [("enabled", "bypassed"), ("enabled", "pre_retrofit"), ("bypassed", "pre_retrofit")]
        if "full_bypass" in values:
            comparisons += [("enabled", "full_bypass"), ("full_bypass", "pre_retrofit")]
        for metric in metrics:
            denominators = [v["tokens"] for v in values["enabled"]] if task == "nll" else None
            key = "sum_nll" if task == "nll" else metric
            for mode, other in comparisons:
                if task == "nll" and any([v["tokens"] for v in x] != denominators for x in values.values()):
                    raise ValueError("causal target counts differ between arms")
                stats = paired_interval([v[key] for v in values[mode]],
                                        [v[key] for v in values[other]] if other else None,
                                        denominators, draws=draws)
                rows.append({"checkpoint": Path(result["checkpoint"]).name, "task": task, "metric": metric,
                             "comparison": mode + (" - " + other if other else ""), **stats})
    return rows


def report(args):
    reference = json.loads(Path(args.reference).read_text())
    if reference["audit"]["variant"] is not None:
        raise ValueError("pre-retrofit reference must be a stock checkpoint")
    rows = []
    for path in args.results:
        rows.extend(compare_results(json.loads(Path(path).read_text()), reference, args.bootstrap))
    write_json(args.output, {"bootstrap": "paired percentile; documents for token-weighted NLL, questions for accuracy",
                             "draws": args.bootstrap, "rows": rows})
    print("checkpoint | task/metric | comparison | estimate [95% CI]")
    for row in rows:
        print(f"{row['checkpoint']} | {row['task']}/{row['metric']} | {row['comparison']} | "
              f"{row['estimate']:+.6f} [{row['ci95'][0]:+.6f}, {row['ci95'][1]:+.6f}]")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--tokenizer", required=True)
    prep.add_argument("--documents", required=True)
    prep.add_argument("--manifests", nargs="+", required=True)
    prep.add_argument("--output", required=True)
    prep.add_argument("--docs", type=int, default=32)
    prep.add_argument("--questions", type=int, default=32)
    prep.add_argument("--document-tokens", type=int, default=512)
    prep.add_argument("--min-document-tokens", type=int, default=128)
    prep.add_argument("--max-question-tokens", type=int, default=4096)
    prep.add_argument("--text-only", action="store_true")
    prep.add_argument("--mmlu-json")
    prep.add_argument("--arc-json")
    prep.set_defaults(func=prepare)
    run = sub.add_parser("evaluate")
    run.add_argument("--bundle", required=True)
    run.add_argument("--checkpoint", required=True)
    run.add_argument("--table")
    run.add_argument("--device", default="cuda:0")
    run.add_argument("--split", choices=["screen", "confirmation"], default="screen")
    run.add_argument("--tasks", nargs="+", choices=["nll", "mmlu", "arc"], default=["nll", "mmlu", "arc"])
    run.add_argument("--limit", type=int, default=0)
    run.add_argument("--max-seconds", type=int, choices=range(1, 571), default=540, metavar="1..570")
    run.add_argument("--output", required=True)
    run.set_defaults(func=evaluate)
    rep = sub.add_parser("report")
    rep.add_argument("--reference", required=True)
    rep.add_argument("--results", nargs="+", required=True)
    rep.add_argument("--output", required=True)
    rep.add_argument("--bootstrap", type=int, default=10000)
    rep.set_defaults(func=report)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

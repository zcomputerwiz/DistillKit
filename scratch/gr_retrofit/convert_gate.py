"""The real-recipient conversion gate: does the converted model still predict?

The algebra is exact and the module reproduces it bitwise in float32 and bfloat16. None
of that is evidence about a 2B recipient with 64 sublayers, a KV cache, padding and a
checkpoint round trip, which is what this script measures.

The predeclared gate, fixed before any of it was run: absolute token-weighted aggregate
and content NLL changes each at most 1e-4 nats on at least 64 representative documents.
Baseline nondeterminism is measured first, because "bitwise identical" is only a claim
worth making once the original has been shown to agree with itself.

    CUDA_VISIBLE_DEVICES=0 python scratch/gr_retrofit/convert_gate.py --mode symmetric

Writes ``scratch/gr_retrofit/conversion-<mode>.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn.functional as F

# Imported here, before anything is loaded or measured, and not inside the function that
# needs it. `distillkit.models` installs the expanded-GQA attention dispatch at import
# time, which changes how *every* Qwen3.5 attention call is computed in this process. An
# earlier draft imported it between the baseline and the comparison and read back a
# 1.3e-3 nat aggregate change against bitwise-identical logits: the original model's own
# arithmetic had moved underneath the baseline. Both models are measured under the
# deployed dispatch, which is also the one the pilot would train under.
from distillkit.models import Qwen35WidenedForCausalLM  # noqa: E402
from distillkit.models.qwen35.gqa_dispatch import fused_kernel_supports_gqa  # noqa: E402

BASE = Path("D:/DeepThought/Projects/HybridModel/student-2b-hf")
BUNDLE = Path("scratch/independent-eval/full-bundle-384.json")
#: Predeclared, and not to be loosened after seeing results.
THRESHOLD = 1e-4


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with io.open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            sha.update(block)
    return sha.hexdigest()


def parameter_digest(model, skip=("attn_residual", "mlp_residual")):
    """Backbone tensors only, so conversion can be shown not to have touched them."""
    sha = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        if any(part in name.split(".") for part in skip):
            continue
        sha.update(name.split("model.", 1)[-1].encode("utf-8"))
        # The stored bytes, not an fp32 copy of them: hashing 2B parameters through
        # float32 numpy costs several gigabytes of churn to answer the same question.
        raw = parameter.detach().cpu().contiguous().flatten()
        sha.update(raw.view(torch.int16 if raw.dtype == torch.bfloat16
                            else raw.dtype).numpy().tobytes())
    return sha.hexdigest()


def documents(count):
    bundle = json.loads(BUNDLE.read_text(encoding="utf-8"))
    # The screen split. `confirmation` is kept untouched for the retention pilot, which
    # may not be scored on anything the conversion was checked against.
    records = bundle["splits"]["screen"]["nll"][:count]
    if len(records) < count:
        raise SystemExit("bundle has %d screen documents, asked for %d"
                         % (len(records), count))
    return records


def token_classes(tokenizer, vocab_size):
    sys.path.insert(0, str(Path("scratch/residual_gate").resolve()))
    from evaluate import split_layout

    from distillkit.independent_eval import build_token_classes
    return split_layout(build_token_classes(tokenizer, vocab_size), tokenizer, vocab_size)


def load_original(device, dtype):
    from transformers import AutoConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    config = AutoConfig.from_pretrained(BASE, local_files_only=True)
    config = getattr(config, "text_config", config)
    config.use_cache = True
    model = Qwen3_5ForCausalLM.from_pretrained(BASE, config=config, dtype=dtype,
                                               local_files_only=True).to(device).eval()
    return model.requires_grad_(False), config


def load_converted(config, device, dtype, mode, seed, branches, lowrank):
    import copy

    widened = copy.deepcopy(config)
    widened.residual_stream_enabled = True
    widened.residual_stream_routing = "flash_next"
    widened.residual_stream_num_branches = branches
    widened.residual_stream_lowrank = lowrank
    widened.residual_stream_sidecar = False
    widened.residual_stream_blend = 0.0
    model = Qwen35WidenedForCausalLM.from_pretrained(
        BASE, config=widened, dtype=dtype, local_files_only=True).to(device).eval()
    records = model.recipient_initialize(asymmetric=(mode == "asymmetric"), seed=seed)
    return model.requires_grad_(False), records


@torch.inference_mode()
def logits_for(model, ids, device, hidden=False):
    tokens = torch.tensor([ids], device=device)
    output = model(input_ids=tokens, attention_mask=torch.ones_like(tokens),
                   output_hidden_states=hidden, use_cache=False)
    return output.logits[0].float(), output.hidden_states


def scored(logits, ids, classes, device):
    """Per-document summed NLL and counts, aggregate and content."""
    targets = torch.tensor(ids[1:], device=device)
    nll = F.cross_entropy(logits[:-1], targets, reduction="none")
    content = torch.tensor([classes[int(t)] == "content" for t in ids[1:]],
                           device=device)
    return {"nll": float(nll.sum()), "count": int(nll.numel()),
            "content_nll": float(nll[content].sum()), "content_count": int(content.sum()),
            "argmax": logits[:-1].argmax(-1)}


def weighted(rows, key, count_key):
    total = sum(row[count_key] for row in rows)
    return (sum(row[key] for row in rows) / total if total else float("nan")), total


def difference(left, right):
    gap = (left - right).abs()
    return {"max": float(gap.max()), "rms": float(gap.pow(2).mean().sqrt())}


@torch.inference_mode()
def decode(model, prefix, device, steps):
    """Greedy cached decoding, which is where a stale position or cache would show."""
    tokens = torch.tensor([prefix], device=device)
    produced, past = [], None
    ids = tokens
    for _ in range(steps):
        output = model(input_ids=ids, past_key_values=past, use_cache=True)
        past = output.past_key_values
        nxt = int(output.logits[0, -1].argmax())
        produced.append(nxt)
        ids = torch.tensor([[nxt]], device=device)
    return produced


@torch.inference_mode()
def padded_batch(model, first, second, device):
    """Right-padded batch against the same rows run alone."""
    width = max(len(first), len(second))
    tokens = torch.zeros(2, width, dtype=torch.long, device=device)
    mask = torch.zeros(2, width, dtype=torch.long, device=device)
    for row, ids in enumerate((first, second)):
        tokens[row, :len(ids)] = torch.tensor(ids, device=device)
        mask[row, :len(ids)] = 1
    batched = model(input_ids=tokens, attention_mask=mask, use_cache=False).logits.float()
    return [batched[row, :len(ids)] for row, ids in enumerate((first, second))]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("symmetric", "asymmetric"), required=True)
    parser.add_argument("--documents", type=int, default=64)
    parser.add_argument("--branches", type=int, default=4)
    parser.add_argument("--lowrank", type=int, default=320)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--decode-steps", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.device == "cuda" and torch.cuda.device_count() != 1:
        raise SystemExit("mask to one card with CUDA_VISIBLE_DEVICES")
    started = time.monotonic()
    device, dtype = args.device, torch.bfloat16
    from transformers import AutoTokenizer

    records = documents(args.documents)
    tokenizer = AutoTokenizer.from_pretrained(BASE, local_files_only=True)

    original, config = load_original(device, dtype)
    classes = token_classes(tokenizer, config.vocab_size)
    report = {
        "mode": args.mode, "threshold_nats": THRESHOLD,
        "recipient": str(BASE),
        "recipient_weights_sha256": digest(BASE / "model.safetensors"),
        "tokenizer_sha256": digest(BASE / "tokenizer.json"),
        "execution_dtype": str(dtype), "device": torch.cuda.get_device_name(0)
        if args.device == "cuda" else "cpu",
        "fused_kernel_supports_gqa": bool(fused_kernel_supports_gqa()),
        "branches": args.branches, "lowrank": args.lowrank, "seed": args.seed,
        "documents": len(records), "bundle": str(BUNDLE), "split": "screen",
        "document_ids": [record["id"] for record in records],
    }

    # 1. Baseline nondeterminism. Everything below is read against this floor. One
    # document's logits are 438 x 248320 floats, so nothing here holds more than a
    # document at a time -- the sums and the argmax are all that survive the loop.
    repeats, base_rows = [], []
    for record in records:
        first, _ = logits_for(original, record["ids"], device)
        second, _ = logits_for(original, record["ids"], device)
        repeats.append(difference(first, second))
        base_rows.append(scored(first, record["ids"], classes, device))
        del first, second
    report["baseline_nondeterminism"] = {
        "max": max(row["max"] for row in repeats),
        "rms": max(row["rms"] for row in repeats),
        "bitwise": all(row["max"] == 0.0 for row in repeats)}
    backbone_before = parameter_digest(original)
    if args.device == "cuda":
        torch.cuda.empty_cache()

    converted, conversion = load_converted(config, device, dtype, args.mode, args.seed,
                                           args.branches, args.lowrank)
    report["conversion"] = {"sublayers": len(conversion), "mode": conversion[0]["mode"],
                            "epsilon": conversion[0]["epsilon"],
                            "seeds": [row["seed"] for row in conversion],
                            "blend": float(next(
                                module.blend for module in converted.modules()
                                if hasattr(module, "branch_gain_delta")
                                and hasattr(module, "blend"))),
                            "gain_dtype": str(next(
                                module.branch_gain_delta.dtype
                                for module in converted.modules()
                                if hasattr(module, "branch_gain_delta")))}
    report["backbone_unchanged"] = parameter_digest(converted) == backbone_before

    # 2. Full forwards, per-document NLL, per-layer states.
    gaps, layer_gaps, rows, agreement = [], [], [], 0
    for index, (record, base_row) in enumerate(zip(records, base_rows)):
        want_states = index == 0
        logits, hidden = logits_for(converted, record["ids"], device, hidden=want_states)
        base_logits, base_hidden = logits_for(original, record["ids"], device,
                                              hidden=want_states)
        gaps.append(difference(logits, base_logits))
        row = scored(logits, record["ids"], classes, device)
        agreement += int((row["argmax"] == base_row["argmax"]).sum())
        rows.append(row)
        if want_states:
            layer_gaps = [difference(a[0].float(), b[0].float())
                          for a, b in zip(hidden, base_hidden)]
        del logits, base_logits, hidden, base_hidden
    report["logits"] = {"max": max(row["max"] for row in gaps),
                        "rms": max(row["rms"] for row in gaps),
                        "bitwise": all(row["max"] == 0.0 for row in gaps)}
    report["per_layer_states"] = {
        "layers": len(layer_gaps), "max": max(row["max"] for row in layer_gaps),
        "bitwise": all(row["max"] == 0.0 for row in layer_gaps)}
    report["argmax_agreement"] = agreement / sum(row["count"] for row in base_rows)

    base_aggregate, tokens = weighted(base_rows, "nll", "count")
    base_content, content_tokens = weighted(base_rows, "content_nll", "content_count")
    aggregate, _ = weighted(rows, "nll", "count")
    content, _ = weighted(rows, "content_nll", "content_count")
    per_document = [(row["nll"] - base["nll"]) / max(row["count"], 1)
                    for row, base in zip(rows, base_rows)]
    report["nll"] = {
        "tokens": tokens, "content_tokens": content_tokens,
        "original_aggregate": base_aggregate, "converted_aggregate": aggregate,
        "aggregate_delta": aggregate - base_aggregate,
        "original_content": base_content, "converted_content": content,
        "content_delta": content - base_content,
        "worst_document_delta": max(per_document, key=abs) if per_document else 0.0,
        "documents_changed": sum(1 for value in per_document if value != 0.0)}

    # 3. Cached decoding, padding.
    prefix = records[0]["ids"][:64]
    produced = decode(converted, prefix, device, args.decode_steps)
    expected = decode(original, prefix, device, args.decode_steps)
    report["cached_decoding"] = {"steps": args.decode_steps, "identical": produced == expected,
                                 "original": expected, "converted": produced}
    # Padding is reported against the original's own padded-versus-alone difference.
    # Batching changes the reduction order inside the matmuls, so a nonzero gap here is
    # the recipient's, not the conversion's, unless the converted model's is larger.
    first, second = records[0]["ids"][:96], records[1]["ids"][:64]
    padded = padded_batch(converted, first, second, device)
    alone = [logits_for(converted, ids, device)[0] for ids in (first, second)]
    base_padded = padded_batch(original, first, second, device)
    base_alone = [logits_for(original, ids, device)[0] for ids in (first, second)]
    report["padding"] = {
        "converted": [difference(a, b) for a, b in zip(padded, alone)],
        "original": [difference(a, b) for a, b in zip(base_padded, base_alone)],
        "padded_rows_match": [difference(a, b) for a, b in zip(padded, base_padded)]}
    del padded, alone, base_padded, base_alone

    # 4. Save, reload, and a trained-state round trip. The original is released first:
    # four 2B models on one card is 18 GB before a single activation.
    del original
    if args.device == "cuda":
        torch.cuda.empty_cache()
    output = args.output or Path("scratch/gr_retrofit/conversion-%s.json" % args.mode)
    saved = output.parent / ("checkpoint-%s" % args.mode)
    converted.save_pretrained(saved)
    restored = Qwen35WidenedForCausalLM.from_pretrained(
        saved, dtype=dtype, local_files_only=True).to(device).eval()
    reloaded = logits_for(restored, records[0]["ids"], device)[0]
    report["save_reload"] = {
        "mode_restored": restored.config.residual_stream_recipient_mode,
        "gains_bitwise": all(
            torch.equal(a.branch_gain_delta, b.branch_gain_delta)
            for a, b in zip((m for m in converted.modules()
                             if hasattr(m, "branch_gain_delta")),
                            (m for m in restored.modules()
                             if hasattr(m, "branch_gain_delta")))),
        "logits": difference(reloaded, logits_for(converted, records[0]["ids"], device)[0])}
    with torch.no_grad():
        for module in restored.modules():
            if hasattr(module, "branch_gain_delta") and hasattr(module, "W_up"):
                module.branch_gain_delta.add_(0.01)
                module.W_up.weight.normal_(0, 0.02)
    trained = logits_for(restored, records[0]["ids"], device)[0]
    initialized = next(m.branch_gain_delta.detach().clone() for m in converted.modules()
                       if hasattr(m, "branch_gain_delta"))
    restored.save_pretrained(saved.with_name(saved.name + "-trained"))
    del converted, restored
    if args.device == "cuda":
        torch.cuda.empty_cache()
    again = Qwen35WidenedForCausalLM.from_pretrained(
        saved.with_name(saved.name + "-trained"), dtype=dtype,
        local_files_only=True).to(device).eval()
    report["trained_round_trip"] = {
        "logits": difference(logits_for(again, records[0]["ids"], device)[0], trained),
        "initialization_not_reapplied": not torch.equal(
            next(m.branch_gain_delta for m in again.modules()
                 if hasattr(m, "branch_gain_delta")), initialized)}

    passes = (abs(report["nll"]["aggregate_delta"]) <= THRESHOLD
              and abs(report["nll"]["content_delta"]) <= THRESHOLD
              and report["cached_decoding"]["identical"]
              and report["backbone_unchanged"]
              and report["save_reload"]["gains_bitwise"]
              and report["save_reload"]["logits"]["max"] == 0.0
              and all(row["max"] <= base["max"] for row, base in
                      zip(report["padding"]["converted"], report["padding"]["original"]))
              and report["trained_round_trip"]["initialization_not_reapplied"])
    report["verdict"] = "PASS" if passes else "FAIL"
    report["elapsed_seconds"] = time.monotonic() - started
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items()
                      if key not in ("document_ids", "conversion")}, indent=2))
    print("wrote %s" % output)
    return 0 if passes else 1


if __name__ == "__main__":
    raise SystemExit(main())

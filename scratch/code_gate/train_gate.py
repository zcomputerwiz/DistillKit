"""Fit a fresh residual-admission gate to Python on the frozen B_code backbone.

Same 260 parameters, same four gated layers, same two familiarity features, same hidden
width, same admission range, same optimizer and the same learning rate as the corrected
plain-CE gate. The only things that change are the backbone it attaches to, the corpus it
is fitted on, and the familiarity cache it reads. That is the whole design: if a gain
appears, it is attributable to the mechanism being re-fitted rather than to any change in
what the mechanism is.

The canonical regime is reproduced by scored tokens rather than by nominal steps -- 1,536
optimizer steps over 512-token sequences with a constant rate, which is 785,920 supervised
targets. Matching step counts across different data loaders would silently deliver a
different amount of data.

Three invariants are checked rather than assumed:

    identity        at initialization the output layer is zero, so g = 1 exactly and the
                    model's logits must equal the bare backbone's bit for bit. A gate that
                    is not an identity at step zero is not measuring what it claims to.
    frozen backbone the backbone digest is taken before and after and must match. No
                    backbone parameter enters the optimizer.
    heldout unseen  training reads the train split, selection reads calibration. Heldout
                    is not opened by this script at all.

Checkpoint selection is on **calibration content NLL** with aggregate as a guardrail,
never on heldout and never on a downstream benchmark.

    CUDA_VISIBLE_DEVICES=0 python scratch/code_gate/train_gate.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code_training"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ffn_memo"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "residual_gate"))

import numpy as np
import torch
import torch.nn.functional as F

from distillkit.code_classes import CODE_CLASSES, HISTORICAL
from distillkit.experimental.residual_gate import (
    TrigramFamiliarity, calibrate_gates, install_residual_gates, remove_residual_gates)
from corpus import TOKENS, TokenStore, class_tables, corpus_identity, load_tokenizer
from train import packed_stream

BCODE = Path("scratch/code_training/checkpoints/v1/final")
PYTHON_CACHE = Path("scratch/code_gate/cache/layer-12.npz")
#: The corrected plain-CE regime, copied from scratch/gate_regime/factorial.py.
LAYERS = [10, 12, 14, 16]
FAMILY = "familiarity"
LENGTH = 512
STEPS = 1536
LR = 3e-3


def digest_of(model, skip="residual_gates.") -> str:
    """Backbone digest that ignores the gate, so 'frozen' means the backbone."""
    hasher = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        if skip and name.startswith(skip):
            continue
        hasher.update(name.encode("utf-8"))
        hasher.update(tensor.detach().to(torch.float32).cpu().numpy().tobytes())
    return hasher.hexdigest()


@torch.inference_mode()
def score(model, store, indices, code, device, length, head_positions=4096):
    """Mean NLL overall and per historical class on a fixed set of documents."""
    totals = {}
    overall = [0.0, 0]
    for start in range(0, len(indices), 8):
        group = indices[start:start + 8]
        rows = [np.asarray(store.document(i)[:length], dtype=np.int64) for i in group]
        width = max(len(r) for r in rows)
        ids = torch.full((len(rows), width), model.config.eos_token_id,
                         dtype=torch.long, device=device)
        mask = torch.zeros((len(rows), width), dtype=torch.long, device=device)
        for slot, row in enumerate(rows):
            ids[slot, :len(row)] = torch.from_numpy(row).to(device)
            mask[slot, :len(row)] = 1
        handle = getattr(model, "residual_gate_handle", None)
        if handle is not None:
            handle.set_context(ids)
        hidden = model.model(input_ids=ids, attention_mask=mask).last_hidden_state
        nll = torch.zeros((len(rows), width - 1), dtype=torch.float32, device=device)
        chunk = max(1, head_positions // len(rows))
        for begin in range(0, width - 1, chunk):
            stop = min(begin + chunk, width - 1)
            logits = model.lm_head(hidden[:, begin:stop]).float()
            nll[:, begin:stop] = F.cross_entropy(
                logits.transpose(1, 2), ids[:, begin + 1:stop + 1], reduction="none")
        host = nll.to("cpu").numpy()
        for slot, row in enumerate(rows):
            span = len(row) - 1
            values = host[slot, :span].astype(np.float64)
            labels = code[row[1:span + 1]]
            overall[0] += float(values.sum())
            overall[1] += span
            for index in np.unique(labels):
                entry = totals.setdefault(HISTORICAL[CODE_CLASSES[index]], [0.0, 0])
                chosen = values[labels == index]
                entry[0] += float(chosen.sum())
                entry[1] += int(chosen.size)
    return {"aggregate": overall[0] / max(overall[1], 1),
            "tokens": overall[1],
            **{name: value[0] / value[1] for name, value in totals.items()}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", type=Path, default=BCODE)
    parser.add_argument("--cache", type=Path, default=PYTHON_CACHE)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--length", type=int, default=LENGTH)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--calibrate-steps", type=int, default=64,
                        help="no-grad batches for the feature normalizer")
    parser.add_argument("--evaluate-every", type=int, default=256)
    parser.add_argument("--calibration-documents", type=int, default=240)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--memory-fraction", type=float, default=0.92)
    parser.add_argument("--output", type=Path, default=Path("scratch/code_gate/gcode"))
    args = parser.parse_args()

    if torch.cuda.device_count() != 1:
        raise SystemExit("mask to one card with CUDA_VISIBLE_DEVICES")
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction, 0)
    torch.manual_seed(args.seed)

    from transformers import AutoConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    config = AutoConfig.from_pretrained(args.backbone, local_files_only=True)
    config = getattr(config, "text_config", config)
    config.use_cache = False
    model = Qwen3_5ForCausalLM.from_pretrained(
        args.backbone, config=config, dtype=torch.bfloat16,
        local_files_only=True).to(args.device).eval()
    model.requires_grad_(False)
    before = digest_of(model)

    tokenizer = load_tokenizer()
    code, _ = class_tables(tokenizer, config.vocab_size)
    train = TokenStore(TOKENS, "train")
    calibration = TokenStore(TOKENS, "calibration")
    selection = list(range(min(args.calibration_documents, len(calibration))))

    familiarity = TrigramFamiliarity(args.cache, config.vocab_size, device=args.device)
    handle = install_residual_gates(model, LAYERS, family=FAMILY, familiarity=familiarity)
    model.residual_gate_handle = handle
    trainable = [p for p in handle.gates.parameters() if p.requires_grad]
    parameters = sum(p.numel() for p in handle.gates.parameters())
    print("gate: %d parameters over layers %s, family %s" % (parameters, LAYERS, FAMILY))

    # Feature normalizer from Python train, no gradient, gate inert and the model bitwise
    # stock while it runs.
    stream = packed_stream(train, args.length, args.seed)
    batches = []
    for _ in range(args.calibrate_steps):
        ids = torch.from_numpy(np.stack([next(stream)])).to(args.device)
        batches.append({"input_ids": ids, "attention_mask": torch.ones_like(ids)})
    normalizer = calibrate_gates(model, handle, batches)
    statistics_path = args.output / "normalizer.json"
    args.output.mkdir(parents=True, exist_ok=True)
    # Written where the domain-normalized transfer arm can read it: the same non-learned
    # statistics, applied to the general gate's learned weights.
    first = normalizer[str(LAYERS[0])]
    (args.output.parent / "cache" / "normalizer.json").write_text(
        json.dumps({"mean": first["mean"], "std": first["std"],
                    "source": "code_corpus/v1 train split", "layers": normalizer}),
        encoding="utf-8")
    print("normalizer over %d tokens: mean %s std %s"
          % (first["tokens"], ["%.4f" % v for v in first["mean"]],
             ["%.4f" % v for v in first["std"]]))

    # Identity check: the output layer is zero-initialized, so this must be exact.
    probe = np.asarray(train.document(0)[:args.length], dtype=np.int64)
    ids = torch.from_numpy(probe).unsqueeze(0).to(args.device)
    with torch.inference_mode():
        handle.set_context(ids)
        gated = model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits.float()
        remove_residual_gates(model)
        bare = model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits.float()
    identity = float((gated - bare).abs().max())
    handle = install_residual_gates(model, LAYERS, family=FAMILY, familiarity=familiarity)
    model.residual_gate_handle = handle
    for layer in LAYERS:
        gate = handle.gate(layer)
        gate.feature_mean.copy_(torch.tensor(normalizer[str(layer)]["mean"],
                                             dtype=gate.feature_mean.dtype))
        gate.feature_std.copy_(torch.tensor(normalizer[str(layer)]["std"],
                                            dtype=gate.feature_std.dtype))
        gate.calibrated.fill_(True)
    trainable = [p for p in handle.gates.parameters() if p.requires_grad]
    print("identity at initialization: max |logit difference| %.3e" % identity)
    if identity != 0.0:
        raise SystemExit("gate is not an exact identity at initialization")

    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    stream = packed_stream(train, args.length, args.seed)
    losses, history, scored = [], [], 0
    best = {"content": float("inf")}
    started = time.monotonic()

    for step in range(args.steps):
        row = next(stream)
        ids = torch.from_numpy(row).unsqueeze(0).to(args.device)
        handle.set_context(ids)
        logits = model(input_ids=ids,
                       attention_mask=torch.ones_like(ids)).logits[0, :-1].float()
        targets = torch.tensor(row[1:], device=args.device)
        loss = F.cross_entropy(logits, targets)          # plain CE, every target
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
        scored += len(targets)
        if (step + 1) % 200 == 0:
            print("step %d  loss %.5f  %.0f s"
                  % (step + 1, statistics.fmean(losses[-200:]),
                     time.monotonic() - started), flush=True)
        if (step + 1) % args.evaluate_every == 0 or step + 1 == args.steps:
            model.eval()
            result = score(model, calibration, selection, code, args.device, args.length)
            reach = {str(layer): handle.gate(layer).gate_report("g")["g/reach"]
                     for layer in LAYERS}
            entry = {"step": step + 1, "scored_tokens": scored,
                     "calibration": result, "reach": reach}
            history.append(entry)
            print("  step %d calibration content %.6f aggregate %.6f reach %s"
                  % (step + 1, result["content"], result["aggregate"],
                     ["%.3f" % v for v in reach.values()]), flush=True)
            # Selection: content NLL, with aggregate as a guardrail against buying
            # content at the cost of everything else.
            if (result["content"] < best["content"]
                    and result["aggregate"] <= best.get("aggregate", float("inf"))):
                best = {"content": result["content"], "aggregate": result["aggregate"],
                        "step": step + 1, "scored_tokens": scored}
                torch.save({"state_dict": {k: v.cpu() for k, v in
                                           model.residual_gates.state_dict().items()},
                            "layers": LAYERS, "family": FAMILY, "step": step + 1,
                            "mask": "plain", "regime": "harness",
                            "cache": str(args.cache), "backbone": str(args.backbone)},
                           args.output / "gate.pt")

    after = digest_of(model)
    report = {
        "backbone": str(args.backbone), "backbone_sha256_before": before,
        "backbone_sha256_after": after, "backbone_frozen": before == after,
        "cache": str(args.cache), "corpus": corpus_identity(),
        "layers": LAYERS, "family": FAMILY, "parameters": parameters,
        "trainable_parameters": sum(p.numel() for p in trainable),
        "identity_at_initialization": identity,
        "optimizer": "AdamW", "lr": args.lr, "schedule": "constant",
        "steps": args.steps, "sequence_length": args.length,
        "scored_tokens": scored, "objective": "plain CE over all causal targets",
        "normalizer": normalizer, "selection": best,
        "selection_criterion": "calibration content NLL, aggregate as guardrail",
        "history": history,
        "final_loss": statistics.fmean(losses[-100:]),
        "elapsed_seconds": time.monotonic() - started,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(args.device)),
    }
    (args.output / "training.json").write_text(json.dumps(report, indent=2),
                                               encoding="utf-8")
    print("\nbackbone frozen: %s" % report["backbone_frozen"])
    print("selected step %s, calibration content %.6f, %d scored tokens"
          % (best.get("step"), best.get("content", float("nan")), scored))
    print("wrote %s" % (args.output / "gate.pt"))
    return 0 if report["backbone_frozen"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

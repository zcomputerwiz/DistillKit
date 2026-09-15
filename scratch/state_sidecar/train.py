"""Train the state-conditioned hash sidecar and its matched state-blind control.

Two arms from identical initialization, identical data order and an identical token budget.
The only difference is whether the admission gate reads the hidden state:

    blind         g = 1, so the module is a context-addressed correction with no policy
    conditioned   g = sigmoid(cos(q, k) / temperature + b)

The primary comparison is between those two, not against stock. Beating stock only says
that adding a trained correction helps; beating the blind arm says that *state-conditioned
admission* is what helps, which is the hypothesis. Both arms are seeded identically and
draw from the same packed stream, so any difference between them is the gate.

Everything else follows the established contract: the backbone is frozen bitwise and its
digest is checked before and after, plain cross-entropy over every causal target with no
masking of any kind, training reads the train split, checkpoints are selected on
calibration, and the heldout subset is not opened by this script.

    CUDA_VISIBLE_DEVICES=0 python scratch/state_sidecar/train.py --arm conditioned
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
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code_gate"))

import numpy as np
import torch
import torch.nn.functional as F

from distillkit.experimental.ngram_hash import NGramHashConfig, NGramHasher
from distillkit.experimental.state_sidecar import (
    StateConditionedSidecar, install_state_sidecar, remove_state_sidecar)
from corpus import TOKENS, TokenStore, class_tables, corpus_identity, load_tokenizer
from train import packed_stream
from train_gate import digest_of, score

BCODE = Path("scratch/code_training/checkpoints/v1/final")
#: The canonical single-layer retrofit point. Not swept: layer 12 is where the
#: memoisation cache was built, where the intervention study's validated fixed rule sat
#: (MANUAL = {"layer": 12, ...}), and what every familiarity result in this programme is
#: expressed against.
LAYER = 12
ROWS = 1 << 17
CODE_DIM = 32
MEMORY_DIM = 64


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=("blind", "conditioned"))
    parser.add_argument("--backbone", type=Path, default=BCODE)
    parser.add_argument("--layer", type=int, default=LAYER)
    parser.add_argument("--memory-dim", type=int, default=MEMORY_DIM)
    parser.add_argument("--code-dim", type=int, default=CODE_DIM)
    parser.add_argument("--tokens", type=int, default=2_000_000)
    parser.add_argument("--length", type=int, default=512)
    parser.add_argument("--micro-batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--evaluate-every", type=int, default=200)
    parser.add_argument("--calibration-documents", type=int, default=240)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--memory-fraction", type=float, default=0.92)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or Path("scratch/state_sidecar") / args.arm

    if torch.cuda.device_count() != 1:
        raise SystemExit("mask to one card with CUDA_VISIBLE_DEVICES")
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction, 0)

    from transformers import AutoConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    config = AutoConfig.from_pretrained(args.backbone, local_files_only=True)
    config = getattr(config, "text_config", config)
    config.use_cache = False
    model = Qwen3_5ForCausalLM.from_pretrained(
        args.backbone, config=config, dtype=torch.bfloat16,
        local_files_only=True).to(args.device).eval()
    model.requires_grad_(False)
    before = digest_of(model, skip="state_sidecar")

    tokenizer = load_tokenizer()
    code, _ = class_tables(tokenizer, config.vocab_size)
    train = TokenStore(TOKENS, "train")
    calibration = TokenStore(TOKENS, "calibration")
    selection = list(range(min(args.calibration_documents, len(calibration))))

    # Seeded identically for both arms, so the two modules start from the same weights.
    torch.manual_seed(args.seed)
    sidecar = StateConditionedSidecar(
        hidden_size=config.hidden_size, code_dim=args.code_dim,
        memory_dim=args.memory_dim, seed=args.seed).to(args.device).to(torch.float32)
    sidecar.state_conditioned = args.arm == "conditioned"
    hasher = NGramHasher(NGramHashConfig(
        vocab_size=config.vocab_size, ngram_size=3, heads_per_ngram=1,
        ngram_vocab_size_base=ROWS // 2, seed=1234,
        eos_token_id=tokenizer.eos_token_id or config.vocab_size - 1))
    handle = install_state_sidecar(model, sidecar, hasher, args.layer)
    report_parameters = sidecar.parameter_report()
    # The blind arm never evaluates the query/key path, so those parameters take no
    # gradient; reporting both counts keeps the comparison honest.
    trainable = [p for p in sidecar.parameters() if p.requires_grad]
    print(json.dumps({"arm": args.arm, "layer": args.layer,
                      "parameters": report_parameters}), flush=True)

    # Identity check: the output projection is zero, so the model must be bitwise stock.
    probe = np.asarray(train.document(0)[:args.length], dtype=np.int64)
    ids = torch.from_numpy(probe).unsqueeze(0).to(args.device)
    with torch.inference_mode():
        handle.set_context(ids)
        gated = model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits.float()
        remove_state_sidecar(model)
        bare = model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits.float()
    identity = float((gated - bare).abs().max())
    handle = install_state_sidecar(model, sidecar, hasher, args.layer)
    print("identity at initialization: max |logit difference| %.3e" % identity, flush=True)
    if identity != 0.0:
        raise SystemExit("the sidecar is not an exact identity at initialization")

    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    stream = packed_stream(train, args.length, args.seed)
    losses, history = [], []
    scored = steps = 0
    best = {"content": float("inf")}
    started = time.monotonic()

    while scored < args.tokens:
        rows = np.stack([next(stream) for _ in range(args.micro_batch)])
        ids = torch.from_numpy(rows).to(args.device)
        handle.set_context(ids)
        logits = model(input_ids=ids,
                       attention_mask=torch.ones_like(ids)).logits[:, :-1].float()
        targets = ids[:, 1:]
        loss = F.cross_entropy(logits.transpose(1, 2), targets)   # plain CE, every target
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
        scored += int(targets.numel())
        steps += 1
        if steps % 100 == 0:
            print("step %d  %d tokens  loss %.5f  %.0f s"
                  % (steps, scored, statistics.fmean(losses[-100:]),
                     time.monotonic() - started), flush=True)
        if steps % args.evaluate_every == 0 or scored >= args.tokens:
            # The shared scorer refreshes per-forward context through whatever is bound to
            # ``residual_gate_handle``. Binding this handle there is duck-typing -- the
            # two expose the same ``set_context(input_ids)`` -- and without it the sidecar
            # would score using hash rows left over from the last training batch, which
            # only surfaces as a shape error when the batch sizes happen to differ.
            model.residual_gate_handle = handle
            result = score(model, calibration, selection, code, args.device, args.length)
            entry = {"step": steps, "scored_tokens": scored, "calibration": result}
            if sidecar.state_conditioned:
                with torch.inference_mode():
                    handle.collect = True
                    # Packed fixed-length sequences rather than raw documents: calibration
                    # documents differ in length and cannot be stacked, and the gate
                    # statistics want uniform coverage rather than four short files.
                    probe_stream = packed_stream(calibration, args.length, args.seed)
                    probe_ids = torch.from_numpy(
                        np.stack([next(probe_stream) for _ in range(4)])).to(args.device)
                    handle.set_context(probe_ids)
                    model(input_ids=probe_ids,
                          attention_mask=torch.ones_like(probe_ids))
                    handle.collect = False
                    gate = sidecar.last["gate"].float()
                    entry["gate"] = {"mean": float(gate.mean()),
                                     "p10": float(gate.quantile(0.10)),
                                     "p50": float(gate.quantile(0.50)),
                                     "p90": float(gate.quantile(0.90))}
            history.append(entry)
            print("  step %d calibration content %.6f aggregate %.6f%s"
                  % (steps, result["content"], result["aggregate"],
                     "" if "gate" not in entry
                     else "  gate mean %.4f" % entry["gate"]["mean"]), flush=True)
            if result["content"] < best["content"]:
                best = {"content": result["content"], "aggregate": result["aggregate"],
                        "step": steps, "scored_tokens": scored}
                output.mkdir(parents=True, exist_ok=True)
                torch.save({"state_dict": {k: v.cpu() for k, v in
                                           sidecar.state_dict().items()},
                            "arm": args.arm, "layer": args.layer,
                            "code_dim": args.code_dim, "memory_dim": args.memory_dim,
                            "seed": args.seed, "step": steps,
                            "backbone": str(args.backbone)},
                           output / "sidecar.pt")

    after = digest_of(model, skip="state_sidecar")
    report = {
        "arm": args.arm, "backbone": str(args.backbone), "layer": args.layer,
        "backbone_sha256_before": before, "backbone_sha256_after": after,
        "backbone_frozen": before == after,
        "identity_at_initialization": identity,
        "parameters": report_parameters,
        "corpus": corpus_identity(),
        "objective": "plain CE over all causal targets; no masking",
        "optimizer": "AdamW", "lr": args.lr, "schedule": "constant",
        "sequence_length": args.length, "micro_batch": args.micro_batch,
        "steps": steps, "scored_tokens": scored, "seed": args.seed,
        "selection": best,
        "selection_criterion": "calibration content NLL",
        "history": history,
        "final_loss": statistics.fmean(losses[-100:]),
        "elapsed_seconds": time.monotonic() - started,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(args.device)),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "training.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nbackbone frozen: %s   selected step %s content %.6f   %d tokens in %.0f s"
          % (report["backbone_frozen"], best.get("step"),
             best.get("content", float("nan")), scored, report["elapsed_seconds"]))
    return 0 if report["backbone_frozen"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

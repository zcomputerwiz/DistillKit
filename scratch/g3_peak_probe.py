"""G3: where the peak actually is, under the real objective and a real accumulation window.

Two things made the earlier memory numbers unrepresentative. `memory_phases.py` steps and
zeroes the gradients on every backward, while production runs
`gradient_accumulation_steps: 8`, so the gradient buffers are live across eight
microbatches rather than one. And `g2_offload_probe.py` used a synthetic loss, which put
no logit or hidden-state pressure anywhere near where the real one does -- and reported
that parking 1.25 GiB off the home card moved the peak by 1 MiB.

This runs the objective the configs actually use -- sparse top-k KL through the chunked
head plus the chunked hidden-state cosine -- across a full accumulation window, with the
peer offload on and off, and records where in the window each card's peak falls.

Synthetic teacher signals: this is a memory measurement, and the losses it prints are
plumbing, never a measure of language-model quality.

    python scratch/g3_peak_probe.py --output scratch/gpu-checks/g3-peak.json
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from pathlib import Path

import torch
import yaml

from distillkit.anchor_tap import AnchorTap
from distillkit.chunked_head import HeadContext
from distillkit.configuration import DistillationRunConfig
from distillkit.hsd_mapping import HiddenStateMapping
from distillkit.lossfuncs.hidden_state import compute_hs_loss
from distillkit.lossfuncs.kl import KLDLoss
from distillkit.main import load_student_model
from distillkit.models import qwen35_widened
from distillkit.signals import SparseSignal
from distillkit.tp_model import shard_model, sync_replicated_gradients

TEACHER_HIDDEN = 5120
VOCAB = 248320


def synchronize():
    for index in range(2):
        torch.cuda.synchronize(index)


def peaks():
    return [torch.cuda.max_memory_allocated(index) / 1024 ** 3 for index in range(2)]


def load(branches, config_path):
    data = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if branches > 1:
        data["residual_stream"] = {"num_branches": branches, "lowrank": 64}
    cfg = DistillationRunConfig.model_validate(data)
    torch.manual_seed(42)
    model = load_student_model(cfg, 248077, VOCAB)
    model = shard_model(model, ["cuda:0", "cuda:1"])
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    return cfg, model


def signals(cfg, batch, sequence, sidecar):
    torch.manual_seed(123)
    ids = torch.randint(0, 248000, (batch, sequence), device="cuda:0")
    mask = torch.ones(batch, sequence, 1, dtype=torch.bool, device="cuda:0")
    raw = torch.zeros(batch, sequence, sidecar.num_heads, sidecar.bytes_per_head,
                      dtype=torch.uint8)
    signal = SparseSignal(
        sparse_ids=torch.randint(0, VOCAB, (batch, sequence, 64), device="cuda:0"),
        sparse_values=torch.log_softmax(torch.randn(batch, sequence, 64, device="cuda:0"), -1),
        log_values=True, generation_temperature=1.0,
        hidden_states=tuple(
            torch.randn(batch, sequence, TEACHER_HIDDEN, device="cuda:0", dtype=torch.bfloat16)
            for _ in cfg.layer_mapping),
        vocab_size=VOCAB,
    )
    return ids, mask, raw, signal


def window(model, cfg, mapping, optimizer, batch_data, accumulation, offload):
    """One full accumulation window plus the optimizer step that closes it."""
    ids, mask, raw, signal = batch_data
    anchors = [anchor for anchor, _ in cfg.layer_mapping] + [model.config.num_hidden_layers]
    original_device = qwen35_widened._WidenedTextModel._stream_offload_device
    original_context = qwen35_widened.offload_stream_boundaries
    if not offload:
        qwen35_widened._WidenedTextModel._stream_offload_device = lambda self: None
    try:
        model.zero_grad(set_to_none=True)
        synchronize()
        for index in range(2):
            torch.cuda.reset_peak_memory_stats(index)
        history = []
        started = time.perf_counter()
        for step in range(accumulation):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                with AnchorTap(model, anchors) as tap:
                    out = model(input_ids=ids, ngram_raw=raw, logits_to_keep=1)
                out.hidden_states = tap.states()
                head = HeadContext(out.hidden_states[model.config.num_hidden_layers],
                                   model.lm_head, vocab_size=VOCAB, chunk_length=256)
                kl = KLDLoss(temperature=1.0, sparse_chunk_length=256)(
                    out, signal, mask=mask, hidden_state_mapping=mapping, head_context=head)
                loss = (0.7 * kl.to("cuda:0")
                        + 0.3 * compute_hs_loss("cosine", out, signal, mask, mapping))
            if not torch.isfinite(loss):
                raise AssertionError("nonfinite loss")
            (loss / accumulation).backward()
            del out, loss, kl, tap, head
            synchronize()
            history.append({"microbatch": step + 1, "peak_gib": peaks(),
                            "live_gib": [torch.cuda.memory_allocated(i) / 1024 ** 3
                                         for i in range(2)]})
        sync_replicated_gradients(model)
        before_step = peaks()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        synchronize()
        elapsed = time.perf_counter() - started
    finally:
        qwen35_widened._WidenedTextModel._stream_offload_device = original_device
        qwen35_widened.offload_stream_boundaries = original_context
    return {
        "offload": offload,
        "accumulation": accumulation,
        "seconds": elapsed,
        "peak_gib": peaks(),
        "peak_before_optimizer_step_gib": before_step,
        "microbatches": history,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="examples/qwen35_sidecar_stage2_tp.yml")
    parser.add_argument("--branches", type=int, default=2)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--sequence", type=int, default=4096)
    parser.add_argument("--accumulation", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    timer = threading.Timer(3000, lambda: os._exit(124))
    timer.daemon = True
    timer.start()
    for index in range(2):
        torch.cuda.set_per_process_memory_fraction(0.95, index)

    import bitsandbytes as bnb

    cfg, model = load(args.branches, args.config)
    mapping = HiddenStateMapping(model, TEACHER_HIDDEN, cfg.layer_mapping)
    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=1e-5)
    sidecar = model.model.layers[cfg.sidecar.layer_index].sidecar
    batch_data = signals(cfg, args.batch, args.sequence, sidecar)

    # bitsandbytes allocates its 8-bit moments lazily on the first step, about 4 GiB a
    # card. Without a warmup the second arm would carry that residency and the first
    # would not, which is a larger difference than anything being measured.
    window(model, cfg, mapping, optimizer, batch_data, 1, True)

    arms = []
    for offload in (True, False):
        record = window(model, cfg, mapping, optimizer, batch_data, args.accumulation, offload)
        arms.append(record)
        print(json.dumps({k: v for k, v in record.items() if k != "microbatches"}), flush=True)

    on, off = arms
    result = {
        "kind": "g3_accumulation_peak",
        "branches": args.branches, "batch": args.batch, "sequence": args.sequence,
        "arms": arms,
        "home_peak_gib_saved_by_offload": off["peak_gib"][0] - on["peak_gib"][0],
        "peer_peak_gib_added_by_offload": on["peak_gib"][1] - off["peak_gib"][1],
        "seconds_cost_of_offload": on["seconds"] - off["seconds"],
        "quality_evaluation": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "arms"}, indent=2), flush=True)
    timer.cancel()


if __name__ == "__main__":
    main()

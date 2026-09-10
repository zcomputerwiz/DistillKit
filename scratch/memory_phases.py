"""Where does a stage-2 step's peak actually happen, and what strands the pool?

Three OOMs in this project were diagnosed by arithmetic on a traceback and twice the
arithmetic was wrong. Card 0 currently runs at 17.17 GiB allocated against 18.81 GiB
reserved, so about 1.6 GiB is stranded in the pool and roughly 4 GiB of the 22.8 GiB
cap is left. Before optimising anything, find out which phase owns the high-water mark
and how much of the gap is fragmentation rather than live tensors.

    python scratch/memory_phases.py --config examples/qwen35_widened_ple_stage2_5m.yml
    python scratch/memory_phases.py --allocator-sweep

`--allocator-sweep` re-executes this script under each PYTORCH_CUDA_ALLOC_CONF worth
testing. Windows has no `expandable_segments`, so the only levers left are how the
allocator rounds and splits blocks; `roundup_power2_divisions` in particular collapses
the number of distinct block sizes, which is what strands a pool when shapes vary.

NEEDS THE GPU TO ITSELF.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
import yaml

sys.path.insert(0, ".")

ALLOCATOR_SWEEP = [
    "garbage_collection_threshold:0.8",
    "garbage_collection_threshold:0.8,roundup_power2_divisions:16",
    "garbage_collection_threshold:0.8,roundup_power2_divisions:8,max_split_size_mb:512",
    "roundup_power2_divisions:16",
]


class Phase:
    """Peak allocation attributable to one phase, and what it left behind."""

    def __init__(self, name, record):
        self.name, self.record = name, record

    def __enter__(self):
        for index in range(torch.cuda.device_count()):
            torch.cuda.synchronize(index)
            torch.cuda.reset_peak_memory_stats(index)
        self.started = time.perf_counter()
        return self

    def __exit__(self, *exc):
        gib = 1024**3
        for index in range(torch.cuda.device_count()):
            torch.cuda.synchronize(index)
        self.record[self.name] = {
            "seconds": round(time.perf_counter() - self.started, 3),
            "peak_gib": [round(torch.cuda.max_memory_allocated(i) / gib, 3)
                         for i in range(torch.cuda.device_count())],
            "live_gib": [round(torch.cuda.memory_allocated(i) / gib, 3)
                         for i in range(torch.cuda.device_count())],
            "reserved_gib": [round(torch.cuda.memory_reserved(i) / gib, 3)
                             for i in range(torch.cuda.device_count())],
        }
        return False


def measure(config_path, steps):
    import bitsandbytes as bnb

    from distillkit.anchor_tap import AnchorTap
    from distillkit.chunked_head import HeadContext
    from distillkit.configuration import DistillationRunConfig
    from distillkit.hsd_mapping import HiddenStateMapping
    from distillkit.lossfuncs.hidden_state import compute_hs_loss
    from distillkit.lossfuncs.kl import KLDLoss
    from distillkit.main import load_student_model
    from distillkit.signals import SparseSignal
    from distillkit.tp_model import shard_model, sync_replicated_gradients

    cfg = DistillationRunConfig.model_validate(yaml.safe_load(Path(config_path).read_text()))
    batch = cfg.training_args["per_device_train_batch_size"]
    sequence = cfg.sequence_length
    chunk = next(getattr(f, "sparse_chunk_length", None)
                 for f in cfg.loss_functions if f.function == "kl")

    record = {}
    torch.manual_seed(42)
    with Phase("load", record):
        model = load_student_model(cfg, 248077, 248320)
        model = shard_model(model, ["cuda:0", "cuda:1"])
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.train()
        mapping = HiddenStateMapping(model, 5120, cfg.layer_mapping)
        optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=1e-5)

    anchors = [a for a, _ in cfg.layer_mapping] + [model.config.num_hidden_layers]
    sidecar = model.model.layers[cfg.sidecar.layer_index].sidecar
    torch.manual_seed(123)
    ids = torch.randint(0, 248000, (batch, sequence), device="cuda:0")
    mask = torch.ones(batch, sequence, 1, dtype=torch.bool, device="cuda:0")
    raw = torch.zeros(batch, sequence, sidecar.num_heads, sidecar.bytes_per_head, dtype=torch.uint8)
    signal = SparseSignal(
        sparse_ids=torch.randint(0, 248320, (batch, sequence, 64), device="cuda:0"),
        sparse_values=torch.log_softmax(torch.randn(batch, sequence, 64, device="cuda:0"), -1),
        log_values=True, generation_temperature=1.0,
        hidden_states=tuple(torch.randn(batch, sequence, 5120, device="cuda:0", dtype=torch.bfloat16)
                            for _ in cfg.layer_mapping),
        vocab_size=248320,
    )

    for step in range(steps):
        prefix = "step%d/" % (step + 1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            with Phase(prefix + "forward", record):
                with AnchorTap(model, anchors) as tap:
                    out = model(input_ids=ids, ngram_raw=raw, logits_to_keep=1)
                out.hidden_states = tap.states()
            with Phase(prefix + "kl", record):
                kl = KLDLoss(temperature=1.0, sparse_chunk_length=chunk)(
                    out, signal, mask=mask, hidden_state_mapping=mapping,
                    head_context=HeadContext(out.hidden_states[model.config.num_hidden_layers],
                                             model.lm_head, vocab_size=248320, chunk_length=chunk))
            with Phase(prefix + "hidden_state", record):
                hs = compute_hs_loss("cosine", out, signal, mask, mapping)
            loss = 0.7 * kl.to("cuda:0") + 0.3 * hs
        with Phase(prefix + "backward", record):
            loss.backward()
            sync_replicated_gradients(model)
        with Phase(prefix + "optimizer", record):
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        del out, loss, kl, hs, tap

    gib = 1024**3
    record["summary"] = {
        "config": config_path, "batch": batch, "sequence": sequence,
        "allocator": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "(default)"),
        "peak_gib": [round(torch.cuda.max_memory_allocated(i) / gib, 3) for i in range(2)],
        "reserved_gib": [round(torch.cuda.memory_reserved(i) / gib, 3) for i in range(2)],
        # The gap is the pool the allocator holds but cannot hand out. On Windows,
        # without expandable_segments, this is the number that turns into an OOM.
        "stranded_gib": [round((torch.cuda.memory_reserved(i)
                                - torch.cuda.memory_allocated(i)) / gib, 3) for i in range(2)],
    }
    return record


def sweep(config_path, steps, output):
    results = []
    for allocator in ALLOCATOR_SWEEP:
        print("\n=== %s ===" % allocator, flush=True)
        environment = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF=allocator)
        target = Path(output).with_suffix(".%s.json" % allocator.replace(":", "").replace(",", "_"))
        code = subprocess.call([sys.executable, __file__, "--config", config_path,
                                "--steps", str(steps), "--output", str(target)],
                               env=environment)
        if code != 0:
            results.append({"allocator": allocator, "failed": code})
            print("  exit %d (a configuration that OOMs is a result)" % code, flush=True)
            continue
        results.append(json.loads(target.read_text())["summary"])
    print("\n%-64s %14s %14s %14s" % ("allocator", "peak c0/c1", "reserved", "stranded"))
    for row in results:
        if "failed" in row:
            print("%-64s %14s" % (row["allocator"], "OOM/exit %d" % row["failed"]))
            continue
        print("%-64s %14s %14s %14s" % (
            row["allocator"],
            "%.2f/%.2f" % tuple(row["peak_gib"]),
            "%.2f/%.2f" % tuple(row["reserved_gib"]),
            "%.2f/%.2f" % tuple(row["stranded_gib"])))
    Path(output).write_text(json.dumps(results, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="examples/qwen35_widened_ple_stage2_5m.yml")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--allocator-sweep", action="store_true")
    parser.add_argument("--output", default="scratch/widened-residual/memory-phases.json")
    args = parser.parse_args()
    for index in range(2):
        torch.cuda.set_per_process_memory_fraction(0.95, index)
    if args.allocator_sweep:
        sweep(args.config, args.steps, args.output)
        return 0
    record = measure(args.config, args.steps)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(record, indent=2), encoding="utf-8")
    for name, value in record.items():
        print("%-22s %s" % (name, json.dumps(value)), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

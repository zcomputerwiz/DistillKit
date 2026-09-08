"""Short real-cache training probe; writes measurements, never model checkpoints.

Run each mode in a fresh process with idle GPUs. Uses the same eight longest cache
documents in both modes and at least two AdamW8bit updates.
"""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import yaml
from transformers import TrainerCallback
from distillkit.configuration import DistillationRunConfig
import distillkit.main as entry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["serial", "threaded"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--boundary", type=int, default=12, choices=range(1, 32))
    parser.add_argument("--steps", type=int, default=2)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a new output directory for each probe")
    for device in range(2):
        free, _ = torch.cuda.mem_get_info(device)
        if free < 21 * 1024**3:
            raise RuntimeError("Probe requires both GPUs to be idle with at least 21 GiB free")
    config = yaml.safe_load(Path("examples/qwen35_sidecar_stage2_concurrent.yml").read_text())
    config["concurrent_microbatches"] = 2 if args.mode == "threaded" else 1
    for i in range(32):
        config["model_kwargs"]["device_map"][f"model.layers.{i}"] = 0 if i < args.boundary else 1
    config["output_path"] = str(args.output)
    config["sidecar"]["prefault"] = False
    config["sidecar"]["resident"] = False
    config["training_args"].update(
        gradient_accumulation_steps=4, max_steps=args.steps, eval_strategy="no",
        save_strategy="no", report_to=[], logging_steps=1, disable_tqdm=True,
        warmup_steps=0, lr_scheduler_type="constant",
    )
    import os
    report = {"mode": args.mode, "boundary": args.boundary, "steps": [],
              "allocator": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "default")}

    class Timing(TrainerCallback):
        def on_step_begin(self, args, state, control, **kwargs):
            for d in range(2):
                torch.cuda.synchronize(d)
                # These eight documents are the longest in the cache, so every
                # microbatch is near the 4096 cap and the allocator ratchets its
                # reservation up until it cannot place the next block. Releasing
                # cached blocks between steps keeps the timing question separate
                # from the fragmentation one, and costs both modes equally.
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(d)
            self.started = time.perf_counter()

        def on_step_end(self, args, state, control, model=None, optimizer=None, **kwargs):
            for d in range(2):
                torch.cuda.synchronize(d)
            # Device residency is checked after optimizer state has materialized.
            state_tensors = 0
            for p, values in optimizer.state.items():
                for key, value in values.items():
                    if isinstance(value, torch.Tensor):
                        assert value.device == p.device, (key, value.device, p.device)
                        state_tensors += 1
            item = {"step": state.global_step, "seconds": time.perf_counter()-self.started,
                    "optimizer_state_tensors_on_owner": state_tensors,
                    "peak_GiB": [torch.cuda.max_memory_allocated(d)/1024**3 for d in range(2)],
                    "reserved_GiB": [torch.cuda.max_memory_reserved(d)/1024**3 for d in range(2)]}
            report["steps"].append(item)
            print("PROBE", json.dumps(item), flush=True)
            (Path(args.output_dir)/"probe.json").write_text(json.dumps(report, indent=2))

    class ProbeTrainer(entry.HybridDistillationTrainer):
        def train(self, *args, **kwargs):
            # Median-length documents, not the longest. The longest pin every
            # microbatch at the 4096 cap, which ratchets the allocator until it OOMs
            # at step 2 -- and step 1 is the one window the threaded path handicaps
            # itself on, running the first microbatch alone to prime lazy init. A
            # timing comparison has to reach step 2. These are also closer to the real
            # corpus, whose documents average ~870 tokens.
            order = sorted(range(len(self.train_dataset)),
                           key=lambda i: (len(self.train_dataset[i]["input_ids"]), i))
            middle = len(order) // 2
            indices = sorted(order[max(0, middle - 6):middle + 6][:12])
            self.train_dataset = self.train_dataset.select(indices)
            report["documents"] = list(self.train_dataset["doc_id"])
            report["lengths"] = [len(x) for x in self.train_dataset["input_ids"]]
            self.add_callback(Timing())
            result = super().train(*args, **kwargs)
            report["metrics"] = result.metrics
            report["history"] = self.state.log_history
            (Path(self.args.output_dir)/"probe.json").write_text(json.dumps(report, indent=2))
            return result

        def save_model(self, *args, **kwargs):
            pass  # Intentionally no 8+ GB checkpoint from a timing probe.

    entry.HybridDistillationTrainer = ProbeTrainer
    entry.do_distill(DistillationRunConfig.model_validate(config))


if __name__ == "__main__":
    main()

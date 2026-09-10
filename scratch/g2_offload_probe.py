"""G2: does parking checkpoint boundaries on the peer card change the gradients?

`offload_stream_boundaries` had only ever been exercised on CPU tensors, which take
the bypass branch in `pack` -- so nothing had validated the peer copies themselves,
nor their interaction with the tensor-parallel blocks' custom autograd Functions and
the cross-device barriers those insert.

One model, two backwards, same inputs and seed, the second with the offload disabled.
A `.to(peer).to(home)` round trip is an exact copy, so the gradients should agree bit
for bit; anything else means the hooks are interfering with the graph.

    python scratch/g2_offload_probe.py --output scratch/gpu-checks/g2-offload.json
"""

import argparse
import contextlib
import functools
import json
import os
import threading
import time
from pathlib import Path

import torch
import yaml

from distillkit.configuration import DistillationRunConfig
from distillkit.main import load_student_model
from distillkit.models import qwen35_widened
from distillkit.tp_model import shard_model, sync_replicated_gradients
from distillkit.widened_residual import offload_stream_boundaries


def synchronize():
    for index in range(2):
        torch.cuda.synchronize(index)


def load(branches=2):
    data = yaml.safe_load(Path("examples/qwen35_sidecar_stage2_tp.yml").read_text(encoding="utf-8"))
    data["residual_stream"] = {"num_branches": branches, "lowrank": 64}
    cfg = DistillationRunConfig.model_validate(data)
    torch.manual_seed(42)
    model = load_student_model(cfg, 248077, 248320)
    model = shard_model(model, ["cuda:0", "cuda:1"])
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    return cfg, model


@contextlib.contextmanager
def counting_registration(tally):
    """Count what the production hooks park, by wrapping their registration.

    Nesting `saved_tensors_hooks` does not compose -- only the innermost pair is
    consulted -- so an outer observer context would simply never run. Wrapping the
    registration keeps exactly one pair installed and puts the counter around the
    production pack/unpack themselves. Checkpointing installs its own pair inside each
    layer, which is why only the offload's is wrapped.
    """
    original = torch.autograd.graph.saved_tensors_hooks

    class Wrapped(original):
        def __init__(self, pack, unpack):
            if getattr(pack, "__qualname__", "").startswith("offload_stream_boundaries"):
                tally["installed"] += 1

                def counted_pack(tensor):
                    size = tensor.numel() * tensor.element_size()
                    tally["saved"] += 1
                    tally["saved_bytes"] += size
                    payload = pack(tensor)
                    if isinstance(payload, tuple):
                        tally["parked"] += 1
                        tally["parked_bytes"] += size
                        tally["parked_on"].add(str(payload[1].device))
                    return payload

                super().__init__(counted_pack, unpack)
            else:
                super().__init__(pack, unpack)

    torch.autograd.graph.saved_tensors_hooks = Wrapped
    try:
        yield
    finally:
        torch.autograd.graph.saved_tensors_hooks = original


def fingerprint(model):
    """One pair of scalars per parameter: cheap, and sensitive to any reordering."""
    return {name: (parameter.grad.float().sum().item(), parameter.grad.float().norm().item())
            for name, parameter in model.named_parameters() if parameter.grad is not None}


def run_arm(model, ids, raw, target, offload, every=2):
    original_device = qwen35_widened._WidenedTextModel._stream_offload_device
    original_context = qwen35_widened.offload_stream_boundaries
    tally = {"installed": 0, "saved": 0, "saved_bytes": 0,
             "parked": 0, "parked_bytes": 0, "parked_on": set()}
    if not offload:
        qwen35_widened._WidenedTextModel._stream_offload_device = lambda self: None
    qwen35_widened.offload_stream_boundaries = functools.partial(original_context, every=every)
    try:
        with counting_registration(tally):
            model.zero_grad(set_to_none=True)
            synchronize()
            for index in range(2):
                torch.cuda.reset_peak_memory_stats(index)
            start = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hidden = model.model(input_ids=ids, ngram_raw=raw).last_hidden_state
                loss = (hidden.float() * target).sum() / hidden.numel()
            # What the offload is supposed to buy: less resident on the home card while
            # the whole forward's boundaries are stored. Peak is a max over the run and
            # can be set elsewhere; this is the moment the parking is meant to help.
            synchronize()
            after_forward = [torch.cuda.memory_allocated(i) / 1024 ** 3 for i in range(2)]
            loss.backward()
            sync_replicated_gradients(model)
            synchronize()
            elapsed = time.perf_counter() - start
    finally:
        qwen35_widened._WidenedTextModel._stream_offload_device = original_device
        qwen35_widened.offload_stream_boundaries = original_context
    record = {
        "offload": offload,
        "every": every if offload else None,
        "allocated_after_forward_gib": after_forward,
        "loss": loss.item(),
        "seconds": elapsed,
        "peak_gib": [torch.cuda.max_memory_allocated(i) / 1024 ** 3 for i in range(2)],
        "reserved_gib": [torch.cuda.max_memory_reserved(i) / 1024 ** 3 for i in range(2)],
        "hook_installations": tally["installed"],
        "saved_tensors": tally["saved"],
        "saved_gib": tally["saved_bytes"] / 1024 ** 3,
        "parked_tensors": tally["parked"],
        "parked_gib": tally["parked_bytes"] / 1024 ** 3,
        "parked_on": sorted(tally["parked_on"]),
    }
    return record, fingerprint(model)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--sequence", type=int, default=1024)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    timer = threading.Timer(900, lambda: os._exit(124))
    timer.daemon = True
    timer.start()
    for index in range(2):
        torch.cuda.set_per_process_memory_fraction(0.95, index)

    cfg, model = load()
    sidecar = model.model.layers[cfg.sidecar.layer_index].sidecar
    torch.manual_seed(123)
    ids = torch.randint(0, 248000, (args.batch, args.sequence), device="cuda:0")
    raw = torch.zeros(args.batch, args.sequence, sidecar.num_heads, sidecar.bytes_per_head,
                      dtype=torch.uint8)
    target = torch.randn(args.batch, args.sequence, model.config.hidden_size, device="cuda:0")

    on, grads_on = run_arm(model, ids, raw, target, offload=True)
    off, grads_off = run_arm(model, ids, raw, target, offload=False)
    # CUDA backward is not bit-reproducible run to run -- atomics in the embedding and
    # convolution backwards, and the reduction orders inside the sharded blocks. Repeating
    # the *same* arm measures that floor, so the offload comparison has something to be
    # judged against rather than against zero.
    repeat, grads_repeat = run_arm(model, ids, raw, target, offload=True)
    # every=1 parks every boundary rather than alternating: if the home card's residency
    # still does not fall, the tensors were not uniquely owned by the saved-tensor slot
    # and the parking cannot free anything at all.
    everyone, _ = run_arm(model, ids, raw, target, offload=True, every=1)
    assert on["parked_on"] == ["cuda:1"], on["parked_on"]
    assert on["parked_tensors"] > 0, "nothing was parked on the peer"
    assert off["parked_tensors"] == 0 and off["hook_installations"] == 0, off

    def compare(left, right):
        """Magnitude-weighted, because a per-parameter relative is meaningless here.

        The identity-initialised routing scalars carry gradient norms around 1e-6, so
        their relative difference swings by an order of magnitude between repeats of the
        *same* arm and drowns out every parameter that actually has signal. The global
        ratio below weights each parameter by its own gradient, and is compared against
        the same statistic measured between two runs of one arm.
        """
        shared = sorted(set(left) & set(right))
        delta = torch.tensor([left[name][1] - right[name][1] for name in shared])
        scale = torch.tensor([right[name][1] for name in shared])
        big = [name for name in shared if abs(right[name][1]) > 1e-3]
        worst = max(big, key=lambda name: abs(left[name][1] - right[name][1])
                    / abs(right[name][1])) if big else shared[0]
        return {
            "parameters_with_grad": len(shared),
            "bit_identical_parameters": sum(left[name] == right[name] for name in shared),
            "global_norm_relative": (delta.norm() / scale.norm()).item(),
            "parameters_above_1e-3": len(big),
            "worst_substantial_parameter": worst,
            "worst_substantial_relative": (abs(left[worst][1] - right[worst][1])
                                           / max(abs(right[worst][1]), 1e-12)),
        }

    offload_effect = compare(grads_on, grads_off)
    nondeterminism = compare(grads_on, grads_repeat)
    result = {
        "kind": "g2_peer_offload",
        "batch": args.batch,
        "sequence": args.sequence,
        "arms": [on, off, repeat, everyone],
        "home_gib_saved_after_forward": off["allocated_after_forward_gib"][0]
                                        - on["allocated_after_forward_gib"][0],
        "home_gib_saved_after_forward_every1": off["allocated_after_forward_gib"][0]
                                               - everyone["allocated_after_forward_gib"][0],
        "offload_vs_no_offload": offload_effect,
        "same_arm_repeated": nondeterminism,
        "offload_within_nondeterminism": (offload_effect["global_norm_relative"]
                                          <= 4 * max(nondeterminism["global_norm_relative"], 1e-12)),
        "loss_bit_identical": on["loss"] == off["loss"] == repeat["loss"],
        "quality_evaluation": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    assert result["offload_within_nondeterminism"], (
        "peer offload moved gradients by %.3g, past the %.3g run-to-run floor"
        % (offload_effect["global_norm_relative"], nondeterminism["global_norm_relative"]))
    timer.cancel()


if __name__ == "__main__":
    main()

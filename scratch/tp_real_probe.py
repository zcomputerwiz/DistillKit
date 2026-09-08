"""First real numbers for tensor parallelism on the 4B student.

Everything so far is correctness on toy models. This loads the actual student, shards
it, and reports what landed where and how long a step takes -- against the layer
split's measured 3.99 s per 4096-token microbatch at boundary 13.
"""

import time

import yaml
import torch

from distillkit.configuration import DistillationRunConfig
from distillkit.main import load_student_model
from distillkit.tp_model import shard_model, sharded_parameter_report, sync_replicated_gradients

import sys
SEQ = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
cfg = DistillationRunConfig.model_validate(
    yaml.safe_load(open("examples/qwen35_sidecar_stage2_sharded.yml"))
)
cfg.model_kwargs.pop("device_map", None)  # tensor parallel places the model itself
for index in range(torch.cuda.device_count()):
    torch.cuda.set_per_process_memory_fraction(cfg.max_vram_fraction, index)

model = load_student_model(cfg, 248077, 248320)
model = shard_model(model, ["cuda:0", "cuda:1"])
# Gradient checkpointing is OFF: the custom autograd Functions interact badly with
# non-reentrant checkpointing ("trying to save more tensors during recomputation").
# Tensor parallelism halves activations on its own, so measure whether it is needed.
model.config.use_cache = False  # training does this; without checkpointing nothing else will
model.train()

report = sharded_parameter_report(model)
print("sharded %.1f%% of %.3fB parameters" % (
    100 * report["sharded_fraction"], report["total_parameters"] / 1e9))
for device, value in report["bytes_per_device"].items():
    print("  %-8s %6.2f GiB of weights" % (device, value / 1024**3))

sidecar = model.model.layers[cfg.sidecar.layer_index].sidecar
input_ids = torch.randint(0, 248000, (1, SEQ), device="cuda:0")
ngram_raw = torch.zeros(1, SEQ, sidecar.num_heads, sidecar.bytes_per_head, dtype=torch.uint8)


def step():
    # Non-reentrant checkpointing stops recomputing once it has recovered every saved
    # tensor. A sharded module produces tensors on two devices and the recompute
    # visits them in a different order, tripping that assertion; recompute the whole
    # region instead. It is a context manager, not a setter.
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(input_ids=input_ids, ngram_raw=ngram_raw, sidecar_enabled=True,
                    return_dict=True, logits_to_keep=1)
        loss = out.logits.float().pow(2).mean()
    loss.backward()
    sync_replicated_gradients(model)
    model.zero_grad(set_to_none=True)


for index in range(2):
    torch.cuda.reset_peak_memory_stats(index)
step()
for index in range(2):
    torch.cuda.synchronize(index)

start = time.perf_counter()
for _ in range(3):
    step()
for index in range(2):
    torch.cuda.synchronize(index)
elapsed = (time.perf_counter() - start) / 3

print("")
print("forward+backward: %.3f s per %d-token microbatch" % (elapsed, SEQ))
for index in range(2):
    print("  card %d peak %6.2f GiB  reserved %6.2f GiB" % (
        index,
        torch.cuda.max_memory_allocated(index) / 1024**3,
        torch.cuda.max_memory_reserved(index) / 1024**3,
    ))
print("layer split at boundary 13: 1.049 s at seq 1024, 3.99 s at seq 4096")

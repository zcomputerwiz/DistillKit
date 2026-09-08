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
from distillkit.anchor_tap import AnchorTap
from distillkit.chunked_head import HeadContext
from distillkit.hsd_mapping import HiddenStateMapping
from distillkit.lossfuncs.hidden_state import compute_hs_loss
from distillkit.lossfuncs.kl import KLDLoss
from distillkit.signals import SparseSignal
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
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
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


hsm = HiddenStateMapping(model, 5120, cfg.layer_mapping)
anchors = [a for a, _ in cfg.layer_mapping] + [model.config.num_hidden_layers]
mask = torch.ones(1, SEQ, 1, dtype=torch.bool, device="cuda:0")
signal = SparseSignal(
    sparse_ids=torch.randint(0, 248320, (1, SEQ, 64), device="cuda:0"),
    sparse_values=torch.log_softmax(torch.randn(1, SEQ, 64, device="cuda:0"), -1),
    log_values=True, generation_temperature=1.0,
    hidden_states=(torch.randn(1, SEQ, 5120, device="cuda:0", dtype=torch.bfloat16),
                   torch.randn(1, SEQ, 5120, device="cuda:0", dtype=torch.bfloat16)),
    vocab_size=248320,
)


def step():
    """The same losses the layer-split baseline was measured with, or the
    comparison is meaningless: chunked KL over the 248,320-wide vocabulary plus
    the hidden-state cosine, with the head folded into the loss chunk loop."""
    with torch.autocast("cuda", dtype=torch.bfloat16):
        with AnchorTap(model, anchors) as tap:
            out = model(input_ids=input_ids, ngram_raw=ngram_raw, sidecar_enabled=True,
                        return_dict=True, logits_to_keep=1)
        out.hidden_states = tap.states()
        kl = KLDLoss(temperature=1.0, sparse_chunk_length=256)(
            out, signal, mask=mask, hidden_state_mapping=hsm,
            head_context=HeadContext(out.hidden_states[model.config.num_hidden_layers],
                                     model.lm_head, vocab_size=248320, chunk_length=256))
        loss = 0.7 * kl + 0.3 * compute_hs_loss("cosine", out, signal, mask, hsm)
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

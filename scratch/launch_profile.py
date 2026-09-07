"""Is the stage-1 step launch-bound, and if so where is the CPU time going?

Symptom: step time barely moved from batch 1 to batch 2, and forward+backward on a
4B model runs at ~10 TFLOP/s on a card that should do ~35. That is the signature of
many small kernels with the GPU idle between them, not of a compute bottleneck.

Measures three things:
  1. GPU busy fraction -- sum of CUDA kernel time vs wall clock. Low = launch-bound.
  2. Kernel launch count per step, and the top CPU-side operators.
  3. Where the CPU time sits: dispatch, H2D copies, or Python.
"""
import os, sys, time
import torch
from torch.profiler import profile, ProfilerActivity
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from distillkit.models.qwen35_sidecar import Qwen35SidecarForCausalLM
from distillkit.optimizers import build_mixed_optimizer

BATCH, SEQ = int(os.environ.get("B", 1)), int(os.environ.get("S", 1024))
CKPT = os.environ.get("CKPT", "1") == "1"
STUDENT = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "student-hf"))
dev = torch.device("cuda:0")

model = Qwen35SidecarForCausalLM.from_pretrained(STUDENT, dtype=torch.bfloat16).to(dev)
model.freeze_backbone()
if CKPT: model.gradient_checkpointing_enable()
model.train()
opt = build_mixed_optimizer(model, lr=1e-4)

g = torch.Generator().manual_seed(0)
ids = torch.randint(0, 248320, (BATCH, SEQ), generator=g, dtype=torch.long).to(dev)
am = torch.ones_like(ids)
# Valid IQ4_NL: random nibbles but finite fp16 scales (random scale bytes hit NaN patterns).
raw = torch.randint(0, 256, (BATCH, SEQ, 16, 5, 18), dtype=torch.uint8)
scales = (torch.randn(BATCH, SEQ, 16, 5) * 0.01).to(torch.float16)
raw[..., :2] = scales.view(torch.uint8).reshape(BATCH, SEQ, 16, 5, 2)
raw = raw.reshape(BATCH, SEQ, 16, 90).to(dev)

def one_step():
    out = model(input_ids=ids, attention_mask=am, labels=ids, ngram_raw=raw,
                sidecar_enabled=True, return_dict=True)
    out.loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
    return out.loss

for _ in range(2): one_step()
torch.cuda.synchronize()

t0 = time.perf_counter()
for _ in range(3): one_step()
torch.cuda.synchronize()
wall = (time.perf_counter() - t0) / 3
print(f"config: b{BATCH} s{SEQ} ckpt={int(CKPT)}  wall {wall*1000:.1f} ms/step "
      f"-> {BATCH*SEQ/wall:.0f} tok/s")

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
             record_shapes=False, with_stack=False) as prof:
    one_step()
    torch.cuda.synchronize()

events = prof.key_averages()
cuda_total = sum(e.self_device_time_total for e in events)
cpu_total = sum(e.self_cpu_time_total for e in events)
n_launches = sum(e.count for e in events if e.self_device_time_total > 0)
print(f"\nper step: CUDA kernel time {cuda_total/1e3:.1f} ms | CPU time {cpu_total/1e3:.1f} ms")
print(f"GPU busy fraction: {cuda_total/1e3/(wall*1000)*100:.1f}%  <- low means launch-bound")
print(f"distinct ops launching kernels: {n_launches:,} calls")

print("\ntop 12 by self CPU time (the launch/dispatch cost):")
for e in sorted(events, key=lambda e: -e.self_cpu_time_total)[:12]:
    print(f"  {e.key[:44]:44s} n={e.count:6d} cpu {e.self_cpu_time_total/1e3:8.1f} ms "
          f"cuda {e.self_device_time_total/1e3:8.1f} ms")

print("\ntop 8 by self CUDA time (the actual work):")
for e in sorted(events, key=lambda e: -e.self_device_time_total)[:8]:
    print(f"  {e.key[:44]:44s} n={e.count:6d} cuda {e.self_device_time_total/1e3:8.1f} ms")

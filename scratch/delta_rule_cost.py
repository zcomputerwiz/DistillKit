"""How much of the step is the pure-torch chunked delta rule?

24 of the 32 layers are linear_attention. Each calls torch_chunk_gated_delta_rule,
which runs `for i in range(1, chunk_size)` (63 iterations) plus one iteration per
64-token chunk. Gradient checkpointing recomputes the forward, so the loops run twice.
"""
import time, torch
from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule

dev = torch.device("cuda:0")
B, S = 1, 1024
NUM_V_HEADS, V_DIM, K_DIM = 32, 128, 128   # Qwen3.5-4B linear attention geometry
N_LINEAR_LAYERS = 24

q = torch.randn(B, S, 16, K_DIM, device=dev, dtype=torch.bfloat16)
k = torch.randn(B, S, 16, K_DIM, device=dev, dtype=torch.bfloat16)
v = torch.randn(B, S, NUM_V_HEADS, V_DIM, device=dev, dtype=torch.bfloat16)
# repeat kv heads to match value heads, as the layer does
q = q.repeat_interleave(2, dim=2); k = k.repeat_interleave(2, dim=2)
beta = torch.rand(B, S, NUM_V_HEADS, device=dev, dtype=torch.bfloat16)
g = -torch.rand(B, S, NUM_V_HEADS, device=dev, dtype=torch.float32)

def run():
    out, _ = torch_chunk_gated_delta_rule(q, k, v, g, beta, chunk_size=64,
                                          initial_state=None, output_final_state=False,
                                          use_qk_l2norm_in_kernel=True)
    return out

run(); torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(5): run()
torch.cuda.synchronize()
per_call = (time.perf_counter() - t0) / 5
print(f"one torch_chunk_gated_delta_rule call (b{B} s{S}): {per_call*1000:.1f} ms")
print(f"x {N_LINEAR_LAYERS} linear-attention layers      : {per_call*N_LINEAR_LAYERS*1000:.0f} ms per forward")
print(f"x2 for gradient-checkpoint recompute        : {per_call*N_LINEAR_LAYERS*2*1000:.0f} ms")
print(f"\nmeasured full step at b1 s1024 was ~3200 ms; forward alone was ~917 ms")

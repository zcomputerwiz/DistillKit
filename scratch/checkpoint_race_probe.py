"""Diagnostic only: trace concurrent checkpoint recomputations and controls."""
import functools
import sys
import threading

import torch
import torch.utils.checkpoint as cp

from distillkit.linear_attention_dispatch import install_device_aware_linear_attention
from distillkit.tensor_parallel import Reduce, Replicate
from distillkit.tp_blocks import TensorParallelMLP

install_device_aware_linear_attention()
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

mode = sys.argv[1] if len(sys.argv) > 1 else "baseline"
original_init = cp._CheckpointFrame.__init__
active = {}
lock = threading.Lock()

def init(self, recompute_fn, *args, **kwargs):
    def traced(*inputs):
        thread = threading.get_ident()
        with lock:
            previous = active.setdefault(id(self), set()).copy()
            active[id(self)].add(thread)
            print("RECOMPUTE", id(self), "thread", thread, "gid", torch._C._current_graph_task_id(),
                  "device", torch.cuda.current_device(), "streams",
                  [torch.cuda.current_stream(i).cuda_stream for i in range(2)],
                  "OVERLAP", previous, flush=True)
        try:
            return recompute_fn(*inputs)
        finally:
            with lock:
                active[id(self)].discard(thread)
    return original_init(self, traced, *args, **kwargs)

cp._CheckpointFrame.__init__ = init

if mode in ("sync", "device"):
    def wrap(fn):
        @functools.wraps(fn)
        def call(*args):
            if mode == "sync":
                for i in range(2):
                    torch.cuda.synchronize(i)
                result = fn(*args)
                for i in range(2):
                    torch.cuda.synchronize(i)
                return result
            with torch.cuda.device(0):
                return fn(*args)
        return call
    for cls in (Reduce, Replicate):
        cls.forward = staticmethod(wrap(cls.forward))
        cls.backward = staticmethod(wrap(cls.backward))

if mode == "sentinel":
    forward, backward = Reduce.forward, Reduce.backward
    def guarded_forward(ctx, home, *shards):
        result = forward(ctx, home, *shards)
        ctx.save_for_backward(torch.empty(0, device=home))
        return result
    def guarded_backward(ctx, grad):
        ctx.saved_tensors
        return backward(ctx, grad)
    Reduce.forward = staticmethod(guarded_forward)
    Reduce.backward = staticmethod(guarded_backward)

config = Qwen3_5TextConfig(
    vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
    num_attention_heads=4, num_key_value_heads=2, head_dim=8,
    linear_key_head_dim=8, linear_value_head_dim=8, linear_num_key_heads=2,
    linear_num_value_heads=4, linear_conv_kernel_dim=4, full_attention_interval=2,
    tie_word_embeddings=True, max_position_embeddings=64, pad_token_id=0,
    eos_token_id=3, use_cache=False,
)
torch.manual_seed(0)
model = Qwen3_5ForCausalLM(config).to("cuda:0").train()
for layer in model.model.layers:
    layer.mlp = TensorParallelMLP(layer.mlp, ["cuda:0", "cuda:1"])
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
ids = torch.randint(0, 64, (1, 8), device="cuda:0")
print("FORWARD", threading.get_ident(), "device", torch.cuda.current_device(), "streams",
      [torch.cuda.current_stream(i).cuda_stream for i in range(2)], flush=True)
model(input_ids=ids).logits.sum().backward()
print("PASS", mode, flush=True)

"""Does wrapping a decoder layer with a zero-init sidecar add preserve bit-exact logits?

Gate 3 of the spec asserts it must. Measure it on a tiny random Qwen3.5 rather than
assume it. No downloads: config is built by hand at toy size.
"""
import torch, torch.nn as nn
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

torch.manual_seed(0)

cfg = Qwen3_5TextConfig(
    vocab_size=512, hidden_size=64, intermediate_size=128,
    num_hidden_layers=8, num_attention_heads=4, num_key_value_heads=2,
    head_dim=16, linear_key_head_dim=16, linear_value_head_dim=16,
    linear_num_key_heads=2, linear_num_value_heads=4, linear_conv_kernel_dim=4,
    full_attention_interval=4, tie_word_embeddings=True, max_position_embeddings=256,
)
print("layer_types:", cfg.layer_types)

dev = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16

def build():
    torch.manual_seed(1234)
    m = Qwen3_5ForCausalLM(cfg).to(device=dev, dtype=dtype).eval()
    return m

class ZeroSidecar(nn.Module):
    """Mimics the retrofit: concat 16x160-ish features -> zero-init Linear -> add."""
    def __init__(self, layer, hidden):
        super().__init__()
        self.layer = layer
        self.proj = nn.Linear(hidden, hidden, bias=False)
        nn.init.zeros_(self.proj.weight)
    def forward(self, hidden_states, *args, **kwargs):
        side = self.proj(hidden_states)
        hidden_states = hidden_states + side
        return self.layer(hidden_states, *args, **kwargs)

ids = torch.randint(0, 512, (2, 48), device=dev)

a = build()
with torch.no_grad():
    out_a = a(input_ids=ids, output_hidden_states=True, return_dict=True)

b = build()
b.model.layers[1] = ZeroSidecar(b.model.layers[1], cfg.hidden_size).to(device=dev, dtype=dtype)
with torch.no_grad():
    out_b = b(input_ids=ids, output_hidden_states=True, return_dict=True)

la, lb = out_a.logits, out_b.logits
print("dtype", la.dtype, "device", la.device, "shape", tuple(la.shape))
print("bit-identical logits:", torch.equal(la, lb))
print("max abs diff:", (la.float() - lb.float()).abs().max().item())
print("n hidden_states:", len(out_a.hidden_states))
for i, (ha, hb) in enumerate(zip(out_a.hidden_states, out_b.hidden_states)):
    if not torch.equal(ha, hb):
        print(f"  first divergent hidden_state index: {i}, maxdiff {(ha.float()-hb.float()).abs().max().item()}")
        break
else:
    print("  all hidden_states bit-identical")

# Also: does gradient reach the zero-init projection? (zero weight, nonzero input -> grad != 0)
b.train()
for p in b.parameters(): p.requires_grad_(False)
b.model.layers[1].proj.weight.requires_grad_(True)
out = b(input_ids=ids, labels=ids, return_dict=True)
out.loss.backward()
g = b.model.layers[1].proj.weight.grad
print("zero-init proj grad is not None:", g is not None, "| grad norm:", None if g is None else g.float().norm().item())

# --- gradient checkpointing + kwarg passthrough ------------------------------
print("\n--- gradient checkpointing ---")
c = build()
c.model.layers[1] = ZeroSidecar(c.model.layers[1], cfg.hidden_size).to(device=dev, dtype=dtype)
c.gradient_checkpointing_enable()
c.train()
try:
    out = c(input_ids=ids, labels=ids, return_dict=True)
    out.loss.backward()
    print("gc forward+backward OK, loss", float(out.loss))
except Exception as e:
    print("gc FAILED:", type(e).__name__, e)

print("\n--- decoder layer forward signature ---")
import inspect
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer
print(inspect.signature(Qwen3_5DecoderLayer.forward))
print("\n--- is GradientCheckpointingLayer wrapping positional? ---")
print(type(a.model.layers[1]).__mro__)

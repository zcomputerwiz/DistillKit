"""Do the anchor hooks reproduce output.hidden_states exactly, at every index?"""
import torch
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from distillkit.sample_transformers import _AnchorTap

cfg = Qwen3_5TextConfig(vocab_size=128, hidden_size=64, intermediate_size=96,
    num_hidden_layers=8, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
    linear_key_head_dim=16, linear_value_head_dim=16, linear_num_key_heads=2,
    linear_num_value_heads=4, linear_conv_kernel_dim=4, full_attention_interval=4,
    max_position_embeddings=128)
torch.manual_seed(0)
model = Qwen3_5ForCausalLM(cfg).eval()
ids = torch.randint(0, 128, (1, 24))
with torch.no_grad():
    ref = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                output_hidden_states=True, use_cache=False, return_dict=True)
print("hidden_states tuple length:", len(ref.hidden_states), "(num_layers =", cfg.num_hidden_layers, ")")
anchors = list(range(cfg.num_hidden_layers + 1))
with torch.no_grad(), _AnchorTap(model, anchors) as tap:
    model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, return_dict=True)
    got = dict(tap.captured)
bad = []
for a in anchors:
    if a not in got:
        bad.append((a, "MISSING")); continue
    same = torch.equal(got[a], ref.hidden_states[a])
    if not same:
        # is it off by one?
        alt = [j for j in anchors if torch.equal(got[a], ref.hidden_states[j])]
        bad.append((a, f"differs; matches hidden_states{alt}"))
print("mismatches:", bad if bad else "none - all", len(anchors), "indices match exactly")

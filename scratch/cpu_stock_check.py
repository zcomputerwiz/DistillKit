"""A stock Qwen3.5 forward on CPU, WITHOUT importing the sidecar model.

This is the path a capture run or an eval would take. If fla is installed and the
dispatch patch has not been applied, this raises inside Triton.
"""
import sys, torch
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

if "--patched" in sys.argv:
    from distillkit.linear_attention_dispatch import install_device_aware_linear_attention
    print("patched:", install_device_aware_linear_attention())

cfg = Qwen3_5TextConfig(vocab_size=256, hidden_size=64, intermediate_size=96,
                        num_hidden_layers=8, num_attention_heads=4, num_key_value_heads=2,
                        head_dim=16, linear_key_head_dim=16, linear_value_head_dim=16,
                        linear_num_key_heads=2, linear_num_value_heads=4,
                        linear_conv_kernel_dim=4, full_attention_interval=4,
                        max_position_embeddings=128)
torch.manual_seed(0)
model = Qwen3_5ForCausalLM(cfg).eval()
try:
    with torch.no_grad():
        out = model(input_ids=torch.randint(0, 256, (1, 32)))
    print("CPU forward OK, logits", tuple(out.logits.shape))
except Exception as e:
    print("CPU forward FAILED:", type(e).__name__, str(e)[:90])

"""Does bitsandbytes actually work on this Windows box, and what does int8 cost?"""
import torch, bitsandbytes as bnb
print("bitsandbytes", bnb.__version__)
from transformers import BitsAndBytesConfig
lin = bnb.nn.Linear8bitLt(512, 512, has_fp16_weights=False).cuda()
x = torch.randn(4, 512, device="cuda", dtype=torch.float16)
y = lin(x)
print("Linear8bitLt forward:", tuple(y.shape), y.dtype, "finite:", torch.isfinite(y).all().item())
lin4 = bnb.nn.Linear4bit(512, 512, compute_dtype=torch.bfloat16).cuda()
y4 = lin4(x.to(torch.bfloat16))
print("Linear4bit forward:  ", tuple(y4.shape), y4.dtype, "finite:", torch.isfinite(y4).all().item())
print("\n27.3B params:")
for name, bits in (("bf16", 16), ("int8", 8), ("nf4", 4)):
    print(f"  {name:5s} {27.32e9*bits/8/1e9:6.1f} GB  fits in 48 GB VRAM: {27.32e9*bits/8/1e9 < 45}")

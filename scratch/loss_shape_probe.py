"""Which loss returns a non-scalar at batch > 1?

The trainer logs `loss.item()` per loss function, which raises once any of them
returns a per-example tensor. Batch 1 hides it: a 1-element tensor converts fine.
"""
import sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers.modeling_outputs import CausalLMOutputWithPast
from distillkit.lossfuncs import ALL_LOSS_CLASSES
from distillkit.signals import SparseSignal
from distillkit.hsd_mapping import HiddenStateMapping
import torch.nn as nn

B, T, V, K, Hs, Ht = 2, 16, 512, 8, 64, 128
torch.manual_seed(0)
logits = torch.randn(B, T, V, requires_grad=True)
hidden = [torch.randn(B, T, Hs) for _ in range(3)]
out = CausalLMOutputWithPast(loss=torch.randn(B), logits=logits, hidden_states=tuple(hidden))
sig = SparseSignal(
    sparse_ids=torch.randint(0, V, (B, T, K)),
    sparse_values=torch.log_softmax(torch.randn(B, T, K), -1),
    log_values=True, generation_temperature=1.0,
    hidden_states=(torch.randn(B, T, Ht), torch.randn(B, T, Ht)), vocab_size=V,
)
mask = torch.ones(B, T, 1, dtype=torch.bool)

class HSM:
    layer_mapping = [(1, 0), (2, 1)]
    projections = nn.ModuleList([nn.Linear(Hs, Ht, bias=False) for _ in range(2)])

for cls in ALL_LOSS_CLASSES:
    name = cls.name()
    try:
        kwargs = {}
        if name in ("kl", "jsd", "tvd"):
            kwargs = {"temperature": 1.0}
        fn = cls(**kwargs)
    except Exception as e:
        print(f"  {name:22s} construct failed: {e}"); continue
    try:
        val = fn(out, sig, mask=mask, hidden_state_mapping=HSM(), num_items_in_batch=None)
        shape = tuple(val.shape)
        flag = "SCALAR" if val.ndim == 0 else f"*** NON-SCALAR {shape} ***"
        print(f"  {name:22s} -> {flag}")
    except Exception as e:
        print(f"  {name:22s} raised {type(e).__name__}: {str(e)[:70]}")

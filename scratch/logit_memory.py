"""Is the memory wall the 248k-vocab logits rather than the backbone?

Peak was 11.80 GB at b1 x 1024 with 8.05 GB of weights -- ~3.7 GB of activations for
1024 tokens *with* gradient checkpointing, which should store only layer boundaries
(32 x 1024 x 2560 x 2 B = 168 MB). The suspect is the head: 248,320 logits per token,
plus the fp32 upcast cross-entropy does internally, plus their gradients.
"""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from distillkit.models.qwen35_sidecar import Qwen35SidecarForCausalLM

STUDENT = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "student-hf"))
dev = torch.device("cuda:0")
model = Qwen35SidecarForCausalLM.from_pretrained(STUDENT, dtype=torch.bfloat16).to(dev)
model.freeze_backbone(); model.gradient_checkpointing_enable(); model.train()
B, S, V = 1, 1024, model.config.vocab_size
g = torch.Generator().manual_seed(0)
ids = torch.randint(0, V, (B, S), generator=g, dtype=torch.long).to(dev)
am = torch.ones_like(ids)
raw = torch.zeros(B, S, 16, 90, dtype=torch.uint8, device=dev)

base = torch.cuda.memory_allocated(dev) / 1024**3
print(f"weights resident: {base:.2f} GB   vocab {V:,}")
print(f"one logits tensor b{B} s{S}: {B*S*V*2/1024**3:.2f} GB bf16 "
      f"({B*S*V*4/1024**3:.2f} GB if upcast to fp32)")

for label, kwargs in [("no loss (labels=None)", {}), ("with CE loss", {"labels": ids})]:
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(dev)
    out = model(input_ids=ids, attention_mask=am, ngram_raw=raw, sidecar_enabled=True,
                return_dict=True, **kwargs)
    tensor = out.loss if "labels" in kwargs else out.logits.float().sum()
    tensor.backward()
    peak = torch.cuda.max_memory_allocated(dev) / 1024**3
    print(f"  {label:24s} peak {peak:5.2f} GB  (activations {peak-base:5.2f} GB)")
    model.zero_grad(set_to_none=True); del out, tensor

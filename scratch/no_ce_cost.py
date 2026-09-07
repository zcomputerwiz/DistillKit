"""What does withholding labels save, for a config with no cross_entropy loss?

A KL-only or KL+hidden-state pilot never reads student_outputs.loss, so the trainer
now withholds labels and the model skips the 248k-wide cross-entropy entirely --
avoiding both its memory and the chunked version's recompute cost.
"""
import os, sys, time, json
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from distillkit.chunked_ce import chunked_causal_lm_loss
from distillkit.models.qwen35_sidecar import Qwen35SidecarForCausalLM
from distillkit.optimizers import build_mixed_optimizer

STUDENT = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "student-hf"))
dev = torch.device("cuda:0")
B, S = 3, 1024
model = Qwen35SidecarForCausalLM.from_pretrained(STUDENT, dtype=torch.bfloat16).to(dev)
model.freeze_backbone(); model.gradient_checkpointing_enable(); model.train()
opt = build_mixed_optimizer(model, lr=1e-4)
g = torch.Generator().manual_seed(0)
ids = torch.randint(0, 248320, (B, S), generator=g, dtype=torch.long).to(dev)
am = torch.ones_like(ids)
raw = torch.zeros(B, S, 16, 90, dtype=torch.uint8, device=dev)

def bench(label, use_labels, loss_fn):
    model.loss_function = loss_fn
    opt.zero_grad(set_to_none=True)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(dev)
    ts = []
    for i in range(4):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        kw = {"labels": ids} if use_labels else {}
        out = model(input_ids=ids, attention_mask=am, ngram_raw=raw,
                    sidecar_enabled=True, return_dict=True, **kw)
        # Stand-in for a logits-consuming loss. Deliberately NOT .float(): a full
        # fp32 upcast here is the very cost being measured, and adding one would make
        # the "no CE" row more expensive than the CE rows it is compared against.
        loss = out.loss if use_labels else out.logits.sum()
        loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        if i: ts.append(time.perf_counter() - t0)
    med = sorted(ts)[len(ts)//2]
    peak = torch.cuda.max_memory_allocated(dev)/1024**3
    print(f"  {label:34s} {med*1000:7.0f} ms  {B*S/med:6.0f} tok/s  peak {peak:5.2f} GB")
    return {"label": label, "step_s": med, "tok_s": B*S/med, "peak_gb": peak}

from transformers.loss.loss_utils import ForCausalLMLoss
print(f"batch {B} x seq {S}, backbone frozen:")
rows = [bench("CE, stock loss", True, ForCausalLMLoss),
        bench("CE, chunked loss", True, chunked_causal_lm_loss),
        bench("no CE (labels withheld)", False, chunked_causal_lm_loss)]
json.dump(rows, open("verification/ce-variants.json","w"), indent=2)

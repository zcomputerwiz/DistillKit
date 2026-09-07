"""Where does the stage-1 step actually go, and what batch/seq should the pilot use?

The first smoke run measured 303 tok/s at batch 1 x 1024, which is low enough to
be either a real bottleneck or just launch-latency at a tiny batch. This splits
the step into forward / backward / optimizer and sweeps shapes until OOM.
"""
import sys, os, time, json
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from distillkit.models.qwen35_sidecar import Qwen35SidecarForCausalLM
from distillkit.ngram_hash import NGramHasher
from distillkit.optimizers import build_mixed_optimizer

STUDENT = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "student-hf"))
dev = torch.device("cuda:0")
model = Qwen35SidecarForCausalLM.from_pretrained(STUDENT, dtype=torch.bfloat16).to(dev)
print("attn impl:", model.config._attn_implementation)
model.freeze_backbone(); model.gradient_checkpointing_enable(); model.train()
opt = build_mixed_optimizer(model, lr=1e-4)
hasher = NGramHasher()
g = torch.Generator().manual_seed(0)
bytes_per_head = 160 // 32 * 18

def timed(batch, seq, checkpointing=True, n=4):
    if checkpointing: model.gradient_checkpointing_enable()
    else: model.gradient_checkpointing_disable()
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(dev)
    ids = torch.randint(0, 248320, (batch, seq), generator=g, dtype=torch.long).to(dev)
    am = torch.ones_like(ids)
    raw = torch.randint(0, 256, (batch, seq, 16, bytes_per_head), dtype=torch.uint8).to(dev)
    fw = bw = op = 0.0
    for i in range(n + 1):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        out = model(input_ids=ids, attention_mask=am, labels=ids, ngram_raw=raw,
                    sidecar_enabled=True, return_dict=True)
        torch.cuda.synchronize(); t1 = time.perf_counter()
        out.loss.backward()
        torch.cuda.synchronize(); t2 = time.perf_counter()
        opt.step(); opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize(); t3 = time.perf_counter()
        if i:  # skip warmup
            fw += t1 - t0; bw += t2 - t1; op += t3 - t2
    fw, bw, op = fw/n, bw/n, op/n
    tot = fw + bw + op
    peak = torch.cuda.max_memory_allocated(dev) / 1024**3
    print(f"  b{batch} s{seq} ckpt={int(checkpointing)}: fwd {fw*1000:7.1f} bwd {bw*1000:7.1f} "
          f"opt {op*1000:6.1f} ms | total {tot*1000:7.1f} ms | {batch*seq/tot:6.0f} tok/s | peak {peak:5.2f} GB")
    return {"batch": batch, "seq": seq, "ckpt": checkpointing, "fwd_s": fw, "bwd_s": bw,
            "opt_s": op, "tok_s": batch*seq/tot, "peak_gb": peak}

rows = []
for batch, seq, ckpt in [(1,1024,True),(1,1024,False),(2,1024,True),(4,1024,True),
                         (8,1024,True),(1,4096,True),(2,4096,True),(4,4096,True),(8,4096,True)]:
    try:
        rows.append(timed(batch, seq, ckpt))
    except torch.OutOfMemoryError:
        print(f"  b{batch} s{seq} ckpt={int(ckpt)}: OOM")
        torch.cuda.empty_cache()
best = max(rows, key=lambda r: r["tok_s"])
print(f"\nbest: b{best['batch']} s{best['seq']} ckpt={int(best['ckpt'])} -> {best['tok_s']:.0f} tok/s, {best['peak_gb']:.2f} GB")
print(f"  1M tokens: {1e6/best['tok_s']/3600:.2f} h | 5M: {5e6/best['tok_s']/3600:.2f} h")
json.dump(rows, open("verification/throughput-sweep.json","w"), indent=2)

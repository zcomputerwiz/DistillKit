"""Time one batch/seq in a fresh process, so an earlier OOM cannot fragment the result."""
import sys, os, time, json
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from distillkit.models.qwen35_sidecar import Qwen35SidecarForCausalLM
from distillkit.chunked_ce import chunked_causal_lm_loss
from distillkit.optimizers import build_mixed_optimizer

batch, seq = int(sys.argv[1]), int(sys.argv[2])
STUDENT = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "student-hf"))
dev = torch.device("cuda:0")
try:
    model = Qwen35SidecarForCausalLM.from_pretrained(STUDENT, dtype=torch.bfloat16).to(dev)
    model.loss_function = chunked_causal_lm_loss  # as DistillationTrainer does
    model.freeze_backbone(); model.gradient_checkpointing_enable(); model.train()
    opt = build_mixed_optimizer(model, lr=1e-4)
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(0, 248320, (batch, seq), generator=g, dtype=torch.long).to(dev)
    am = torch.ones_like(ids)
    raw = torch.randint(0, 256, (batch, seq, 16, 90), dtype=torch.uint8).to(dev)
    ts = []
    for i in range(4):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        out = model(input_ids=ids, attention_mask=am, labels=ids, ngram_raw=raw,
                    sidecar_enabled=True, return_dict=True)
        out.loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        if i: ts.append(time.perf_counter() - t0)
    med = sorted(ts)[len(ts)//2]
    peak = torch.cuda.max_memory_allocated(dev)/1024**3
    print(json.dumps({"batch":batch,"seq":seq,"ok":True,"step_s":med,
                      "tok_s":batch*seq/med,"peak_gb":peak,"loss":out.loss.item()}))
except torch.OutOfMemoryError:
    print(json.dumps({"batch":batch,"seq":seq,"ok":False,"error":"OOM"}))

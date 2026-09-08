"""Where does teacher-capture time actually go, and what would speed it up?

The 1M capture ran at ~220 tok/s and 5M would be ~6.3 hours, so it is worth the same
treatment the student's training loop got. capture_teacher does one document per forward
(`input_ids.unsqueeze(0)`), and this corpus is median 547 tokens -- the exact starvation
that cost the student half its throughput. This times the phases separately, then times
the forward alone at several shapes, so the answer is measurement rather than inference.

    python scratch/capture_profile.py

Nothing here writes a cache; it reuses the real teacher and the real corpus.
"""

import json
import time

import numpy as np
import torch

from distillkit.sample_transformers import _AnchorTap, load_text_teacher

TEACHER = "D:/DeepThought/Projects/HybridModel/teacher-hf"
CORPUS = "D:/DeepThought/Projects/HybridModel/capture-data/run1m-full.jsonl"
ANCHORS = [8, 64]
TOP_K = 64


def timed(fn):
    for index in range(torch.cuda.device_count()):
        torch.cuda.synchronize(index)
    start = time.perf_counter()
    result = fn()
    for index in range(torch.cuda.device_count()):
        torch.cuda.synchronize(index)
    return result, time.perf_counter() - start


print("loading the 27B teacher in int8 across both cards...")
model, load_seconds = timed(lambda: load_text_teacher(TEACHER, int8=True, device_map="auto"))
model.eval()
config = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
hidden_size, vocab_size = config.hidden_size, config.vocab_size
device = next(model.parameters()).device
placement = {}
for name, parameter in model.named_parameters():
    placement[str(parameter.device)] = placement.get(str(parameter.device), 0) + parameter.numel()
print("loaded in %.0f s | %s" % (load_seconds, {k: "%.2fB" % (v / 1e9) for k, v in placement.items()}))
print("input device %s | hidden %d | vocab %d\n" % (device, hidden_size, vocab_size))

from transformers import AutoTokenizer  # noqa: E402

tokenizer = AutoTokenizer.from_pretrained(TEACHER, local_files_only=True)
docs = []
with open(CORPUS, encoding="utf-8") as handle:
    for line in handle:
        if not line.strip():
            continue
        record = json.loads(line)
        if "input_ids" in record:
            docs.append(list(record["input_ids"])[:4096])
        elif "text" in record:
            docs.append(tokenizer(record["text"], add_special_tokens=True)["input_ids"][:4096])
        if len(docs) >= 120:
            break
lengths = sorted(len(d) for d in docs)
print("corpus sample: %d docs, median %d tokens, max %d\n"
      % (len(docs), lengths[len(lengths) // 2], lengths[-1]))


def one_document(tokens, logit_chunk_tokens):
    """The real capture body, phase by phase."""
    input_ids = torch.tensor(np.asarray(tokens, dtype=np.int64), dtype=torch.long,
                             device=device).unsqueeze(0)
    phases = {}

    def forward():
        with _AnchorTap(model, ANCHORS) as tap:
            result = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                           use_cache=False, return_dict=True)
            return result, dict(tap.captured)

    (result, anchors), phases["forward"] = timed(forward)
    logits = result.logits

    def topk():
        ids = np.empty((len(tokens), TOP_K), dtype="<u4")
        values = np.empty((len(tokens), TOP_K), dtype="<f2")
        for start in range(0, len(tokens), logit_chunk_tokens):
            stop = min(start + logit_chunk_tokens, len(tokens))
            chunk = logits[0, start:stop].float()
            if not torch.isfinite(chunk).all():
                raise ValueError("nonfinite")
            best, indices = torch.topk(chunk, k=TOP_K, dim=-1)
            logprobs = best - torch.logsumexp(chunk, dim=-1, keepdim=True)
            ids[start:stop] = indices.cpu().numpy().astype("<u4")
            values[start:stop] = logprobs.to(torch.float16).cpu().numpy()
        return ids, values

    _, phases["topk"] = timed(topk)

    def fp8():
        states = np.empty((len(tokens), len(ANCHORS), hidden_size), dtype=np.uint8)
        for compact, layer in enumerate(ANCHORS):
            hidden = anchors[layer]
            if not torch.isfinite(hidden).all() or hidden.abs().max() > torch.finfo(torch.float8_e4m3fn).max:
                raise ValueError("overflow")
            states[:, compact, :] = hidden[0].to(torch.float8_e4m3fn).view(torch.uint8).cpu().numpy()
        return states

    _, phases["fp8_anchors"] = timed(fp8)
    del result, logits, anchors
    return phases


print("=== phase breakdown, real documents, logit_chunk_tokens=64 (the current default) ===")
sample = [d for d in docs if 400 <= len(d) <= 700][:6] or docs[:6]
totals = {}
tokens_done = 0
with torch.inference_mode():
    one_document(docs[0], 64)  # warm up kernels
    for tokens in sample:
        for phase, seconds in one_document(tokens, 64).items():
            totals[phase] = totals.get(phase, 0.0) + seconds
        tokens_done += len(tokens)
grand = sum(totals.values())
for phase, seconds in sorted(totals.items(), key=lambda kv: -kv[1]):
    print("  %-12s %6.3f s  %4.1f%%" % (phase, seconds, 100 * seconds / grand))
print("  %-12s %6.3f s  -> %.0f tok/s over %d tokens\n" % ("TOTAL", grand, tokens_done / grand, tokens_done))

print("=== does the top-k chunk size matter? (same documents) ===")
with torch.inference_mode():
    for chunk_size in (64, 256, 1024):
        total = 0.0
        for tokens in sample:
            total += one_document(tokens, chunk_size)["topk"]
        print("  logit_chunk_tokens %-5d topk %6.3f s" % (chunk_size, total))

print("\n=== forward cost against shape: is the teacher starved on short documents? ===")
print("  (batch>1 is padded and NOT what capture does today -- this measures the headroom)")
with torch.inference_mode():
    for batch, seq in ((1, 512), (4, 512), (8, 512), (1, 1024), (4, 1024), (1, 2048), (1, 4096), (2, 4096)):
        ids = torch.randint(0, 100000, (batch, seq), device=device)
        mask = torch.ones_like(ids)
        run = lambda: model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True)
        timed(run)  # warm
        _, seconds = timed(run)
        print("  batch %d x seq %-5d %6.3f s  %6.0f tok/s" % (batch, seq, seconds, batch * seq / seconds))

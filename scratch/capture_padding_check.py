"""Would batching the capture change what the teacher says?

Capture does one document per forward and `capture_teacher` refuses padded input. The
profile says the forward is 97.5% of the loop and that batch 1 at this corpus's median
length runs at 719 tok/s against a ~1350 tok/s plateau, so batching is worth ~1.9x -- but
only if right-padding leaves the real positions untouched.

For a causal transformer that is a theorem. This teacher is not purely one: 48 of its 64
layers are `linear_attention` (GatedDeltaNet), a recurrence whose implementation has to
mask padding correctly for the theorem to hold in practice. Cheap to check, and the cost
of being wrong is a silently corrupted 50 GB cache and a pilot built on it.

    python scratch/capture_padding_check.py
"""

import json

import torch
from transformers import AutoTokenizer

from distillkit.sample_transformers import _AnchorTap, load_text_teacher

TEACHER = "D:/DeepThought/Projects/HybridModel/teacher-hf"
CORPUS = "D:/DeepThought/Projects/HybridModel/capture-data/run1m-full.jsonl"
ANCHORS = [8, 64]

model = load_text_teacher(TEACHER, int8=True, device_map="auto")
model.eval()
device = next(model.parameters()).device
tokenizer = AutoTokenizer.from_pretrained(TEACHER, local_files_only=True)

docs = []
with open(CORPUS, encoding="utf-8") as handle:
    for line in handle:
        if not line.strip():
            continue
        record = json.loads(line)
        ids = record.get("input_ids") or tokenizer(record["text"], add_special_tokens=True)["input_ids"]
        docs.append(list(ids)[:4096])
        if len(docs) >= 40:
            break
docs.sort(key=len)
# A deliberately uneven batch: the padding is what is under test. Kept short because
# the logits are [batch, width, 248320] -- at batch 4 x 3460 that is 6.4 GiB and OOMs,
# which is itself a constraint on batched capture: it would need a chunked head.
docs = [d for d in docs if len(d) <= 700]
batch_docs = [docs[0], docs[len(docs) // 3], docs[2 * len(docs) // 3], docs[-1]]
print("document lengths in the test batch:", [len(d) for d in batch_docs])
pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
width = max(len(d) for d in batch_docs)


def run(token_lists, pad_to=None):
    """Right-pad to `pad_to` and return (logits, anchors) exactly as capture reads them."""
    pad_to = pad_to or max(len(t) for t in token_lists)
    ids = torch.full((len(token_lists), pad_to), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros((len(token_lists), pad_to), dtype=torch.long, device=device)
    for row, tokens in enumerate(token_lists):
        ids[row, : len(tokens)] = torch.tensor(tokens, dtype=torch.long, device=device)
        mask[row, : len(tokens)] = 1
    with _AnchorTap(model, ANCHORS) as tap:
        result = model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True)
        return result.logits, dict(tap.captured)


with torch.inference_mode():
    print("\nreference: each document alone, unpadded (what capture does today)")
    alone = [run([tokens]) for tokens in batch_docs]

    print("candidate: all four in one padded batch")
    batch_logits, batch_anchors = run(batch_docs, pad_to=width)

    print("\n%-6s %-8s %-14s %-14s  %s" % ("doc", "tokens", "max|dlogit|", "max|danchor|", "top-1 agree"))
    worst_logit = worst_anchor = 0.0
    worst_top1 = 1.0
    for row, tokens in enumerate(batch_docs):
        n = len(tokens)
        ref_logits, ref_anchors = alone[row]
        a = ref_logits[0, :n].float()
        b = batch_logits[row, :n].float()
        dlogit = (a - b).abs().max().item()
        top1 = (a.argmax(-1) == b.argmax(-1)).float().mean().item()
        danchor = 0.0
        for layer in ANCHORS:
            x = ref_anchors[layer][0, :n].float()
            y = batch_anchors[layer][row, :n].float()
            danchor = max(danchor, (x - y).abs().max().item())
        worst_logit = max(worst_logit, dlogit)
        worst_anchor = max(worst_anchor, danchor)
        worst_top1 = min(worst_top1, top1)
        print("%-6d %-8d %-14.6f %-14.6f  %.4f" % (row, n, dlogit, danchor, top1))

    # Scale reference: how big is a logit, and how far apart are the top-2?
    ref_logits, _ = alone[0]
    span = ref_logits[0, : len(batch_docs[0])].float()
    gap = (span.topk(2, dim=-1).values[:, 0] - span.topk(2, dim=-1).values[:, 1]).median().item()
    print("\nscale: median top1-top2 logit gap %.4f, logit magnitude ~%.1f"
          % (gap, span.abs().max().item()))
    print("worst |dlogit| %.6f, worst |danchor| %.6f, min top-1 agreement %.4f"
          % (worst_logit, worst_anchor, worst_top1))
    verdict = "SAFE" if worst_top1 == 1.0 and worst_logit < gap / 10 else "NOT SAFE"
    print("\nverdict: batching is %s for this teacher" % verdict)

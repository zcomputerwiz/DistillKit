"""Which int8 layers make batched capture disagree, and can they be fixed cheaply?

Batching the teacher changes its logits enough to flip 7-12% of top-1 predictions, while
the same test on the bf16 student moves 1-2%. The suspected mechanism is LLM.int8()'s
outlier decomposition: columns of the activation whose absmax exceeds `threshold` are
computed in fp16 and the rest in int8, and that column set is chosen per *call*, so a
batch's other rows change which columns a given row takes.

Three things, in one model load:

1. Report the threshold actually in force.
2. Localize. Divergence compounds through 64 layers, so a late layer looks guilty merely
   for inheriting error. Per module this records the relative change of its input and of
   its output, and ranks by *amplification* -- output divergence over input divergence --
   which is where new error is injected rather than passed along.
3. Test the mechanism directly by setting every threshold to 0 (no fp16 outlier path, so
   nothing is batch-dependent) and re-running the comparison. If the disagreement
   collapses, the mechanism is confirmed and the knob is real -- though disabling outlier
   handling is itself a fidelity loss, which is measured here too.

    python scratch/int8_batch_sensitivity.py
"""

import json

import torch
from transformers import AutoTokenizer

from distillkit.sample_transformers import load_text_teacher

TEACHER = "D:/DeepThought/Projects/HybridModel/teacher-hf"
CORPUS = "D:/DeepThought/Projects/HybridModel/capture-data/run1m-full.jsonl"
LENGTH, BATCH = 256, 4

model = load_text_teacher(TEACHER, int8=True, device_map="auto")
model.eval()
device = next(model.parameters()).device
tokenizer = AutoTokenizer.from_pretrained(TEACHER, local_files_only=True)

import bitsandbytes as bnb  # noqa: E402

int8_layers = [(n, m) for n, m in model.named_modules() if isinstance(m, bnb.nn.Linear8bitLt)]
thresholds = {getattr(m.state, "threshold", None) for _, m in int8_layers}
print("Linear8bitLt modules: %d | thresholds in force: %s" % (len(int8_layers), thresholds))
print("has_fp16_weights: %s\n" % {getattr(m.state, "has_fp16_weights", None) for _, m in int8_layers})

docs = []
with open(CORPUS, encoding="utf-8") as handle:
    for line in handle:
        if not line.strip():
            continue
        record = json.loads(line)
        ids = record.get("input_ids") or tokenizer(record["text"], add_special_tokens=True)["input_ids"]
        if len(ids) >= LENGTH:
            docs.append(list(ids)[:LENGTH])
        if len(docs) >= BATCH:
            break
ids = torch.tensor(docs, dtype=torch.long, device=device)


def relative(a, b):
    scale = a.abs().mean().clamp_min(1e-6)
    return ((a - b).abs().mean() / scale).item()


captured = {}


def install(tag):
    handles = []
    for name, module in int8_layers:
        def hook(_module, args, output, _name=name):
            entry = captured.setdefault((_name, tag), {})
            # To CPU immediately: keeping ~400 modules' activations on the card OOMs
            # a model that already occupies 27 GB.
            entry["in"] = args[0].detach()[0].to("cpu", torch.float32)
            entry["out"] = (output[0] if isinstance(output, tuple) else output).detach()[0].to("cpu", torch.float32)
        handles.append(module.register_forward_hook(hook, with_kwargs=False))
    return handles


def compare(label):
    captured.clear()
    with torch.inference_mode():
        handles = install("alone")
        alone_logits = model(input_ids=ids[:1], attention_mask=torch.ones_like(ids[:1]),
                             use_cache=False, return_dict=True).logits[0].float()
        for h in handles:
            h.remove()
        handles = install("batched")
        batch_logits = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                             use_cache=False, return_dict=True).logits[0].float()
        for h in handles:
            h.remove()
    top1 = (alone_logits.argmax(-1) == batch_logits.argmax(-1)).float().mean().item()
    print("%s: max|dlogit| %.4f | top-1 agreement %.4f"
          % (label, (alone_logits - batch_logits).abs().max().item(), top1))
    return top1


print("=== baseline: row 0 alone vs row 0 inside a batch of %d ===" % BATCH)
compare("  as loaded")

rows = []
for name, _ in int8_layers:
    a, b = captured.get((name, "alone")), captured.get((name, "batched"))
    if not a or not b or a["in"].shape != b["in"].shape:
        continue
    din, dout = relative(a["in"], b["in"]), relative(a["out"], b["out"])
    rows.append((dout / max(din, 1e-9), din, dout, name))

print("\n=== where new divergence is injected (top 15 by amplification) ===")
print("%-9s %-11s %-11s  %s" % ("amplify", "rel d_in", "rel d_out", "module"))
for amplify, din, dout, name in sorted(rows, reverse=True)[:15]:
    print("%-9.2f %-11.3e %-11.3e  %s" % (amplify, din, dout, name))

first = [r for r in rows if r[1] < 1e-7 and r[2] > 1e-5]
print("\nmodules that diverge from an identical input (%d):" % len(first))
for _, din, dout, name in sorted(first, key=lambda r: -r[2])[:10]:
    print("  rel d_out %.3e  %s" % (dout, name))

print("\n=== mechanism test: disable the fp16 outlier path (threshold 0) ===")
for _, module in int8_layers:
    module.state.threshold = 0.0
compare("  threshold=0")
print("\n(threshold=0 removes the batch dependence at the cost of the outlier handling")
print(" LLM.int8() exists for -- fidelity against the fp16 teacher is a separate matter.)")


# ---------------------------------------------------------------------------
# Follow-up: the divergence originates in four modules. Does neutralizing only
# those make batching safe, or does a new onset simply appear one layer deeper?
# Onset can only be detected where the input is still identical, so this has to
# be iterated rather than read off the first ranking.
# ---------------------------------------------------------------------------
print("\n=== iterating: neutralize the offenders, re-measure, repeat ===")
for _, module in int8_layers:
    module.state.threshold = 6.0  # restore

neutralized = set()
for round_index in range(6):
    agreement = compare("  round %d (%d neutralized)" % (round_index, len(neutralized)))
    if agreement == 1.0:
        print("  -> batch-independent")
        break
    onset = []
    for name, _ in int8_layers:
        a, b = captured.get((name, "alone")), captured.get((name, "batched"))
        if not a or not b or a["in"].shape != b["in"].shape or name in neutralized:
            continue
        if relative(a["in"], b["in"]) < 1e-7 and relative(a["out"], b["out"]) > 1e-6:
            onset.append(name)
    if not onset:
        print("  no further onset detectable (inputs already differ everywhere)")
        break
    print("     new onset in %d module(s): %s" % (len(onset), ", ".join(sorted(onset)[:6])))
    for name, module in int8_layers:
        if name in onset:
            module.state.threshold = 0.0
            neutralized.add(name)

print("\ntotal modules neutralized: %d of %d" % (len(neutralized), len(int8_layers)))
params = sum(p.numel() for n, m in int8_layers if n in neutralized for p in m.parameters())
print("they hold %.1fM parameters (%.0f MB extra if upcast to bf16 instead)"
      % (params / 1e6, params / 1e6))

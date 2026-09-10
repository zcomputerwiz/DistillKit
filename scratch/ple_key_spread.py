import json, sys, os, torch
sys.path.insert(0, "scratch"); sys.path.insert(0, ".")
torch.cuda.is_available = lambda: False
import ple_transfer_probe as base
from transformers import AutoTokenizer
from distillkit.ngram_hash import NGramHasher
from distillkit.ngram_table import GGUFNGramTable, IQ4NLDequant
from distillkit.sidecar_collator import SidecarDataCollator

w = torch.load(base.DEFAULT_WEIGHTS, map_location="cpu")
streams = w["key_proj.weight"].shape[0] // w["value_proj.weight"].shape[0]
tok = AutoTokenizer.from_pretrained(base.DEFAULT_STUDENT)
pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
texts = []
with open(base.DEFAULT_DOCUMENTS, encoding="utf-8") as f:
    for line in f:
        texts.append(json.loads(line)["text"])
        if len(texts) >= 32: break
coll = SidecarDataCollator(base._PadCollator(pad), GGUFNGramTable(base.DEFAULT_GGUF), NGramHasher())
b = coll([{"input_ids": tok(t)["input_ids"][:512]} for t in texts])
feat = IQ4NLDequant(out_dtype=torch.float32)(b["ngram_raw"]).flatten(-2)
key = base.grouped_rms_norm(feat @ w["key_proj.weight"].float().T, w["norm_key.weight"], streams)
keep = b["attention_mask"].bool().reshape(-1)
flat = key.reshape(-1, streams, key.shape[-1])[keep]
print("keys:", tuple(flat.shape))
for s in range(streams):
    k = flat[:, s]
    unit = k / k.norm(dim=-1, keepdim=True)
    mean_unit = unit.mean(0)
    print("stream %d  mean cosine to the mean key %.4f   ||mean||/mean|| || %.4f   mean pairwise cos %.4f"
          % (s, (unit @ (mean_unit / mean_unit.norm())).mean(),
             k.mean(0).norm() / k.norm(dim=-1).mean(),
             (mean_unit.norm() ** 2 * len(unit) - 1) / (len(unit) - 1)))

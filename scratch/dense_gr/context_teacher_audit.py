# Assisted-by: Codex
"""CPU-only cached teacher predictions on the frozen evaluation role sets."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

torch.cuda.is_available = lambda: False
torch.cuda.device_count = lambda: 0
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from atlas import ROLES, roles_of
from distillkit.offline_cache import OfflineTeacherCache

root = Path(__file__).resolve().parents[3]
metadata = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
result = {}
for domain in ("agent-claude-code", "agent-codex", "qa"):
    totals = {}
    caches = {}
    tool_top1 = {}
    tool_stop_mass = {"im_end": 0., "endoftext": 0.}
    tool_examples = []
    for doc in metadata["captures"][domain]["documents"]:
        source = doc["capture"]
        if source not in caches:
            caches[source] = OfflineTeacherCache(root / source)
        record = caches[source].read_document(doc["document_id"], include_hidden_states=False)
        width = doc["retained_tokens"]
        ids = record["input_ids"][:width]
        frozen_hash = hashlib.sha256(ids.astype(np.int32).tobytes()).hexdigest()
        if frozen_hash != doc["token_sha256"]:
            raise ValueError("cached tokens differ from the frozen evaluation prefix")
        labels = roles_of(ids)[1:]
        picks = record["topk_ids"][:width - 1]
        values = record["topk_logprobs"][:width - 1].astype(np.float64)
        probs = np.exp(values)
        matches = picks == ids[1:, None]
        true_mass = np.where(matches, probs, 0).sum(1)
        best = picks[np.arange(len(picks)), values.argmax(1)] == ids[1:]
        predicted = picks[np.arange(len(picks)), values.argmax(1)]
        mass = probs.sum(1)
        tool = labels == ROLES.index("tool-result")
        unique, counts = np.unique(predicted[tool], return_counts=True)
        for token, count in zip(unique, counts):
            tool_top1[int(token)] = tool_top1.get(int(token), 0) + int(count)
        for name, token in (("im_end", 248046), ("endoftext", 248044)):
            tool_stop_mass[name] += float(np.where(picks[tool] == token, probs[tool], 0.).sum())
        if len(tool_examples) < 12:
            for position in np.flatnonzero(tool & (predicted != ids[1:]))[:3]:
                tool_examples.append({"doc_id": doc["document_id"], "position": int(position),
                                      "prefix_ids": ids[max(0, position - 24):position + 1].tolist(),
                                      "actual": int(ids[position + 1]),
                                      "predicted": int(predicted[position]),
                                      "predicted_probability": float(probs[position].max())})
        for index, role in enumerate(ROLES):
            keep = labels == index
            n = int(keep.sum())
            if not n:
                continue
            row = totals.setdefault(role, dict(tokens=0, ranked=0, top1=0, true_mass=0., topk_mass=0.))
            row["tokens"] += n
            row["ranked"] += int((true_mass[keep] > 0).sum())
            row["top1"] += int(best[keep].sum())
            row["true_mass"] += float(true_mass[keep].sum())
            row["topk_mass"] += float(mass[keep].sum())
    for cache in caches.values():
        cache.close()
    result[domain] = {role: {"tokens": r["tokens"], **{k: v / r["tokens"] for k, v in r.items() if k != "tokens"}} for role, r in totals.items()}
    if tool_top1:
        total = sum(tool_top1.values())
        result[domain]["tool_prediction_details"] = {
            "tokens": total,
            "im_end_top1_fraction": tool_top1.get(248046, 0) / total,
            "endoftext_top1_fraction": tool_top1.get(248044, 0) / total,
            "mean_cached_stop_probability": {k: v / total for k, v in tool_stop_mass.items()},
            "most_frequent_top1_ids": sorted(tool_top1.items(), key=lambda pair: -pair[1])[:20],
            "examples": tool_examples}
assert not torch.cuda.is_initialized()
Path(sys.argv[2]).write_text(json.dumps({"source": sys.argv[1], "cuda_initialized": False, "raw_unsuppressed_teacher": True, "results": result}, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))

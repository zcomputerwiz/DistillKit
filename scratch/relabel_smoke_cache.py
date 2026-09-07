"""Make the 40-doc smoke cache usable as a training dry-run.

Every smoke document landed in the eval split (shard filenames sort test-* before
train-*), so `do_distill` would bail with "no training documents". This copies the
cache and relabels most documents as train so the *training* wiring can be exercised
end to end against real teacher signals, without re-running the teacher.
"""
import json, shutil, sys
from pathlib import Path

src = Path(r"D:\DeepThought\Projects\HybridModel\capture-smoke")
dst = Path(r"D:\DeepThought\Projects\HybridModel\capture-smoke-mixed")
if dst.exists():
    shutil.rmtree(dst)
shutil.copytree(src, dst)

manifest = json.loads((dst / "manifest.json").read_text(encoding="utf-8"))
docs = manifest["documents"]
for i, doc in enumerate(docs):
    doc["split"] = "eval" if i % 5 == 0 else "train"
(dst / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
counts = {}
for d in docs:
    counts[d["split"]] = counts.get(d["split"], 0) + 1
print(f"{dst}: {len(docs)} documents -> {counts}")

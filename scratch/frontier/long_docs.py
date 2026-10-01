"""Repository-level long documents, 8K-32K tokens, for long-context question synthesis.

Consecutive source files of one package or repository, in directory order, each under a
`# ===== File: <path> =====` header, cut to a target length drawn from Qwen3's long-context
mix (75% 16K-32K, 25% 8K-16K). Sources: the Python packages installed in the venv and the
local C/C++/CUDA repositories. llama.cpp is left out: the long-context probe scores on it.
A package contributes at most --per-source documents, so transformers and torch do not
crowd out the rest.

    python scratch/frontier/long_docs.py --output ../capture-data/long-docs-code.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = Path("D:/DeepThought/Projects/HybridModel")
SITE = ROOT / "DistillKit/.venv/Lib/site-packages"
REPOS = {"nccl": ROOT / "nccl", "flash-attention": ROOT / "flash-attention"}
C_SUFFIXES = {".c", ".cc", ".cpp", ".h", ".hpp", ".cu", ".cuh"}
TOKENIZER = "D:/DeepThought/Projects/HybridModel/DistillKit/scratch/dense_gr/merges-r6r8b/u50"


def target_length(rng):
    return rng.randint(16384, 32000) if rng.random() < 0.75 else rng.randint(8192, 16383)


def files_of(root, suffixes):
    skip = {".git", "__pycache__", "third_party", "cutlass", "build"}
    return [p for p in sorted(root.rglob("*")) if p.suffix in suffixes and p.is_file()
            and not skip & set(p.parts) and p.stat().st_size < 400_000]


def documents(name, root, files, tokenizer, rng, per_source):
    """Walk the files in order from a random start, cutting a document at each target."""
    out, start = [], rng.randrange(len(files))
    order = files[start:] + files[:start]
    parts, tokens, goal = [], 0, target_length(rng)
    for path in order:
        text = path.read_text(encoding="utf-8", errors="replace")
        block = "# ===== File: %s/%s =====\n%s\n" % (name, path.relative_to(root).as_posix(), text)
        size = len(tokenizer(block, add_special_tokens=False)["input_ids"])
        if size > goal:
            continue  # one file larger than the whole target: no room for its neighbours
        if tokens + size > goal:
            if tokens >= 8192:
                out.append("".join(parts))
                if len(out) == per_source:
                    break
            parts, tokens, goal = [], 0, target_length(rng)
        parts.append(block)
        tokens += size
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-source", type=int, default=40)
    parser.add_argument("--min-python-mb", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    rng = random.Random(args.seed)
    sources = {}
    for package in sorted(SITE.iterdir()):
        if package.is_dir() and not package.name.startswith("_") and "-info" not in package.name:
            files = files_of(package, {".py"})
            if sum(p.stat().st_size for p in files) / 2**20 >= args.min_python_mb:
                sources[package.name] = (package, files)
    for name, repo in REPOS.items():
        sources[name] = (repo, files_of(repo, C_SUFFIXES))
    total = 0
    with open(args.output, "w", encoding="utf-8") as out:
        for name, (root, files) in sorted(sources.items()):
            # Large sources get their cap; small ones as many as their code makes.
            docs = documents(name, root, files, tokenizer, rng, args.per_source)
            for index, text in enumerate(docs):
                length = len(tokenizer(text, add_special_tokens=False)["input_ids"])
                out.write(json.dumps({"doc_id": "code:%s:%d" % (name, index), "source": name,
                                      "tokens": length, "text": text}) + "\n")
                total += length
            print("%-24s %3d documents" % (name, len(docs)), flush=True)
    print("total %d tokens -> %s" % (total, args.output))


if __name__ == "__main__":
    main()

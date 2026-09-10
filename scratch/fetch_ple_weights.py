"""Fetch Flash-Next's *trained* PLE projections, and nothing else.

The n-gram table this fork reads is Flash-Next's, but the layer that reads it is not:
`ple_sidecar.py` is a faithful port of the architecture with every weight initialised
to zero. So a module that upstream trained over its whole pretraining run has been
asked to rediscover, from a million tokens, how to decode a table whose encoding it
never saw. These are the weights that were thrown away.

They are small -- about 70 MB of the repository's 131 shards -- but they sit inside two
multi-gigabyte files. safetensors puts a JSON header at the front with a byte range per
tensor, so this reads the header and then range-requests exactly the tensors it wants.
Nothing else is downloaded; in particular not `ngram_embedding.shard_*`, which is the
28.8 GB table already present locally as GGUF.

    python scratch/fetch_ple_weights.py --output ../flash-next-ple
"""

import argparse
import json
import struct
import urllib.request
from pathlib import Path

import numpy as np
import torch

REPO = "Qwen/Qwen3.8-Flash-Next"
REVISION = "de4b8e4d43b917e7706784d8bb445c9af86a3540"
PREFIX = "model.language_model.layers.1.ple."
# The table itself is excluded: it is the 28.8 GB already held as GGUF.
SKIP = "ngram_embedding.shard_"
DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32,
          "I32": torch.int32, "I64": torch.int64}


def url_for(shard):
    return f"https://huggingface.co/{REPO}/resolve/{REVISION}/{shard}"


def fetch(url, start=None, stop=None):
    request = urllib.request.Request(url)
    if start is not None:
        request.add_header("Range", f"bytes={start}-{stop}")
    with urllib.request.urlopen(request, timeout=300) as response:
        return response.read()


def header_of(url):
    """safetensors: 8-byte little-endian header length, then that many bytes of JSON."""
    length = struct.unpack("<Q", fetch(url, 0, 7))[0]
    return json.loads(fetch(url, 8, 8 + length - 1)), 8 + length


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="../flash-next-ple")
    args = parser.parse_args()

    index = json.loads(fetch(
        f"https://huggingface.co/{REPO}/resolve/{REVISION}/model.safetensors.index.json"))
    wanted = {name: shard for name, shard in index["weight_map"].items()
              if name.startswith(PREFIX) and SKIP not in name}
    if not wanted:
        raise SystemExit("no PLE tensors in the index; has the revision changed?")

    tensors, total = {}, 0
    for shard in sorted(set(wanted.values())):
        url = url_for(shard)
        header, data_start = header_of(url)
        for name in sorted(n for n, s in wanted.items() if s == shard):
            entry = header[name]
            begin, end = entry["data_offsets"]
            raw = fetch(url, data_start + begin, data_start + end - 1)
            total += len(raw)
            dtype = DTYPES[entry["dtype"]]
            value = torch.frombuffer(bytearray(raw), dtype=dtype).reshape(entry["shape"])
            short = name[len(PREFIX):]
            tensors[short] = value.clone()
            print("  %-44s %-8s %-18s %8.2f MB" % (
                short, entry["dtype"], tuple(entry["shape"]), len(raw) / 2**20))

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(tensors, out / "ple_layer.pt")
    (out / "source.json").write_text(json.dumps(
        {"repo": REPO, "revision": REVISION, "prefix": PREFIX,
         "excluded": SKIP + "* (the 28.8 GB table, held locally as GGUF)",
         "tensors": {k: list(v.shape) for k, v in tensors.items()}}, indent=2), encoding="utf-8")
    print("\n%d tensors, %.1f MB -> %s" % (len(tensors), total / 2**20, out / "ple_layer.pt"))


if __name__ == "__main__":
    main()

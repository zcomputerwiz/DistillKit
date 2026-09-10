"""Does Flash-Next's trained PLE gate mean anything on *this* student's residual stream?

The downloaded weights are the module upstream trained to read the n-gram table, and the
port has been asking a zero-initialised copy to rediscover that from a million tokens.
Before adopting them there is a prior question, because of how the gate is computed::

    key   = norm_key(key_proj(features))       # the table's side, transferable
    query = norm_query(hidden_states)          # *our* backbone's residual stream
    gate  = sigmoid(signed_sqrt(key . query / sqrt(hidden_size)))

`key_proj` and `norm_query` were trained against Flash-Next's residual basis. This
student is a different model that merely shares the hidden size, so the dot product may
be measuring agreement in a basis the two do not share -- in which case the weights are
not an initialisation, they are noise with a good pedigree.

The test is a shuffled control. Real features are scored against the stream position
they belong to; the control scores the same features against a rotated set of positions.
If the gate carries token-specific information, real pairings agree more than rotated
ones. If the two distributions coincide, the basis does not transfer.

Upstream's `hc_count` is 4 -- `key_proj` is (4*2560, 2560) -- so there are four gates,
one per hyper-connection stream, over a shared value. At identity initialisation every
branch of a widened stream carries the same activation, so scoring one hidden state
against all four keys is exactly what an n_r=4 retrofit would compute on its first step.

CPU-only: the GPUs are training.

    python scratch/ple_transfer_probe.py --report scratch/gpu-checks/ple-transfer.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_GGUF = os.path.join(
    os.path.expanduser("~"), ".cache", "huggingface", "hub",
    "models--unsloth--Qwen3.8-Flash-Next-GGUF", "snapshots",
    "38bb39ee97821de2c9009abb7e93950eec396e66", "UD-IQ4_XS",
    "Qwen3.8-Flash-Next-UD-IQ4_XS-00002-of-00003.gguf",
)
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_STUDENT = str((ROOT / ".." / "student-hf").resolve())
DEFAULT_WEIGHTS = str((ROOT / ".." / "flash-next-ple" / "ple_layer.pt").resolve())
DEFAULT_DOCUMENTS = str((ROOT / ".." / "capture-data" / "heldout.jsonl").resolve())


class _PadCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        longest = max(len(f["input_ids"]) for f in features)
        ids, mask = [], []
        for feature in features:
            count = len(feature["input_ids"])
            ids.append(list(feature["input_ids"]) + [self.pad_token_id] * (longest - count))
            mask.append([1] * count + [0] * (longest - count))
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(mask)}


class _Stop(Exception):
    """Abort the forward once the sidecar layer's input has been captured."""


def stream_at_sidecar_layer(model, ids, mask, layer_index):
    """The residual stream as the PLE layer would see it, without the other 30 layers."""
    captured = {}

    def hook(module, args, kwargs):
        captured["hidden"] = (args[0] if args else kwargs["hidden_states"]).detach()
        raise _Stop

    handle = model.model.layers[layer_index].register_forward_pre_hook(hook, with_kwargs=True)
    try:
        with torch.no_grad():
            model(input_ids=ids, attention_mask=mask, sidecar_enabled=False)
    except _Stop:
        pass
    finally:
        handle.remove()
    return captured["hidden"]


def grouped_rms_norm(x, weight, streams, eps=1e-6):
    """Upstream's `Qwen4ExpTextRMSNorm(group_size=hidden_size)`: normalise per stream."""
    grouped = x.unflatten(-1, (streams, x.shape[-1] // streams)).float()
    grouped = grouped * torch.rsqrt(grouped.pow(2).mean(-1, keepdim=True) + eps)
    scaled = grouped.flatten(-2) * (1.0 + weight.float())
    return scaled.unflatten(-1, (streams, x.shape[-1] // streams))


def gates(hidden, features, weights, streams):
    """The four per-stream gates, exactly as `Qwen4ExpTextPLELayer.forward` computes them."""
    hidden_size = hidden.shape[-1]
    key = grouped_rms_norm(features @ weights["key_proj.weight"].float().T,
                           weights["norm_key.weight"], streams)
    query = grouped_rms_norm(hidden.float().repeat(1, 1, streams),
                             weights["norm_query.weight"], streams)
    raw = (key * query).sum(-1) / math.sqrt(hidden_size)
    return torch.sigmoid(raw.abs().clamp_min(1e-6).sqrt() * raw.sign()), raw


def bootstrap(values, resamples=10000, seed=0):
    generator = torch.Generator().manual_seed(seed)
    draws = values[torch.randint(len(values), (resamples, len(values)), generator=generator)]
    means = draws.mean(dim=1).sort().values
    return means[int(0.025 * resamples)].item(), means[int(0.975 * resamples)].item()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf", default=DEFAULT_GGUF)
    parser.add_argument("--student", default=DEFAULT_STUDENT)
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--documents", default=DEFAULT_DOCUMENTS)
    parser.add_argument("--docs", type=int, default=16)
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--report")
    args = parser.parse_args()

    torch.cuda.is_available = lambda: False

    from transformers import AutoTokenizer

    from distillkit.models.qwen35_sidecar import Qwen35SidecarForCausalLM
    from distillkit.ngram_hash import NGramHasher
    from distillkit.ngram_table import GGUFNGramTable, IQ4NLDequant
    from distillkit.sidecar_collator import SidecarDataCollator

    weights = torch.load(args.weights, map_location="cpu")
    streams = weights["key_proj.weight"].shape[0] // weights["value_proj.weight"].shape[0]
    print("hc_count from the trained shapes:", streams)

    tokenizer = AutoTokenizer.from_pretrained(args.student)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    texts = []
    with open(args.documents, encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            messages = record.get("messages")
            text = (tokenizer.apply_chat_template(messages, tokenize=False)
                    if messages else record.get("text", ""))
            if text:
                texts.append(text)
            if len(texts) >= args.docs:
                break
    print("documents:", len(texts))

    table = GGUFNGramTable(args.gguf)
    collator = SidecarDataCollator(_PadCollator(pad_id), table, NGramHasher())
    batch = collator([{"input_ids": tokenizer(text)["input_ids"][: args.tokens]} for text in texts])
    ids, mask, raw = batch["input_ids"], batch["attention_mask"], batch["ngram_raw"]
    features = IQ4NLDequant(out_dtype=torch.float32)(raw).flatten(-2)
    print("batch", tuple(ids.shape), "features", tuple(features.shape))

    started = time.perf_counter()
    model = Qwen35SidecarForCausalLM.from_pretrained(args.student, dtype=torch.bfloat16).eval()
    model.config.use_cache = False
    layer_index = model.config.sidecar_layer_index
    hidden = stream_at_sidecar_layer(model, ids, mask, layer_index)
    print("stream at layer %d captured in %.1fs %s" % (
        layer_index, time.perf_counter() - started, tuple(hidden.shape)))
    del model

    real, real_raw = gates(hidden, features, weights, streams)
    # Two controls, because they ask different questions. Rolling within the sequence
    # keeps the document and moves only the position, so it isolates token-specific
    # agreement against a topically matched distractor -- the hard control. Rolling
    # across the batch pairs each stream with another document's n-grams entirely, which
    # is the easy one: whatever the gate knows should show up most strongly there.
    shift = features.shape[1] // 2
    controls = {
        "within_document": features.roll(shift, dims=1),
        "across_documents": features.roll(1, dims=0),
    }

    keep = mask.bool()
    report = {"hc_count": streams, "documents": len(texts), "tokens": int(keep.sum()),
              "shift": shift, "streams": []}
    for stream in range(streams):
        entry = {
            "stream": stream,
            "gate_mean": real[..., stream][keep].mean().item(),
            "gate_std": real[..., stream][keep].std().item(),
            "logit_std": real_raw[..., stream][keep].std().item(),
        }
        for name, shuffled in controls.items():
            control, _ = gates(hidden, shuffled, weights, streams)
            paired = (real[..., stream] - control[..., stream])[keep]
            low, high = bootstrap(paired)
            entry[name] = {
                "shuffled_gate_mean": control[..., stream][keep].mean().item(),
                "paired_delta": paired.mean().item(),
                "paired_ci": [low, high],
                "separates": low > 0 or high < 0,
                "share_of_gate_std": abs(paired.mean().item()) / entry["gate_std"],
            }
        report["streams"].append(entry)
        print("stream %d  gate %.4f+-%.4f" % (stream, entry["gate_mean"], entry["gate_std"]))
        for name in controls:
            side = entry[name]
            print("    vs %-17s paired %+.5f [%+.5f, %+.5f]  %.1f%% of gate std  %s" % (
                name, side["paired_delta"], *side["paired_ci"],
                100 * side["share_of_gate_std"],
                "separates" if side["separates"] else "spans zero"))

    report["any_stream_separates"] = any(
        entry[name]["separates"] for entry in report["streams"] for name in controls)
    report["quality_evaluation"] = False
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

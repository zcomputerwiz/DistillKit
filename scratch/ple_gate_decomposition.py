"""What is Flash-Next's trained PLE gate actually responding to?

`ple_transfer_probe.py` asked whether the gate scores a real (n-gram, position) pairing
differently from a shuffled one, and found a small separation with two oddities: stream 0
with the sign reversed, and the supposedly *harder* within-document control separating
more than the across-document one. The second of those is an artefact of the corpus --
17.2% of positions carry a token-identical n-gram across every document, 51.3% in the
first 128 tokens, because every document opens with the same system prompt -- so rolling
features across documents leaves a large share of positions untouched. It is a weaker
shuffle, not an easier question.

That probe also asked the wrong question. It held the query fixed and permuted the key,
so it could only see the gate's dependence on the *table*. The gate has two inputs::

    gate = sigmoid(signed_sqrt((norm_key(key_proj(f)) . norm_query(h)) / sqrt(hidden)))

and the interesting possibility is that upstream's gate is mostly a function of `h`: a
"how much lexical help do I want at this position" signal read off the backbone's own
state, largely indifferent to which row it is offered. That would be a useful thing to
inherit even though the key side does not transfer, and it is invisible to a key-only
shuffle.

So permute each side separately, over a full random permutation of valid positions rather
than a roll, and attribute the gate's variance::

    corr(real, key_permuted)^2    share explained by the query alone
    corr(real, query_permuted)^2  share explained by the key alone

Then regress the gate on what the query could plausibly be encoding: the residual
stream's norm, the position index, the role of the token, and -- the one that would make
the gate worth inheriting -- the backbone's own uncertainty about the next token. A gate
that opens where the model is unsure is asking for lexical help; a gate that ignores
uncertainty is doing something else.

Runs on the GPU when one is free (`--device cuda:0`), because the uncertainty terms need
the full forward and a 248,320-wide softmax per position.

    python scratch/ple_gate_decomposition.py --report scratch/gpu-checks/ple-gate-decomposition.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ple_transfer_probe as base


def gate_from(key, query, hidden_size):
    raw = (key * query).sum(-1) / math.sqrt(hidden_size)
    return torch.sigmoid(raw.abs().clamp_min(1e-6).sqrt() * raw.sign())


def correlation(left, right):
    left = left - left.mean()
    right = right - right.mean()
    return (left @ right / (left.norm() * right.norm() + 1e-12)).item()


def explained(target, *predictors):
    """R^2 of an ordinary least-squares fit with an intercept.

    Standardised first: the position index runs to 512 while the role indicators are 0/1,
    and `lstsq` on that design is ill-conditioned enough to return a negative R^2 -- which
    is a solver failure, not a fit worse than the mean. `driver="gelsd"` is the SVD-based
    path, which handles the rank deficiency a full set of role indicators would otherwise
    introduce alongside the intercept.
    """
    columns = []
    for predictor in predictors:
        centred = predictor - predictor.mean()
        scale = centred.std()
        columns.append(centred / scale if scale > 0 else centred)
    design = torch.stack([torch.ones_like(target)] + columns, dim=1)
    solution = torch.linalg.lstsq(design, target.unsqueeze(1), driver="gelsd").solution
    residual = target - (design @ solution).squeeze(1)
    return 1 - (residual.var() / target.var()).item()


def uncertainty(final_hidden, head, ids, chunk=128):
    """The backbone's own entropy over the next token, and its NLL of the real one.

    Chunked over positions: a 248,320-wide distribution for every position at once is
    several gigabytes, and only two scalars per position survive.
    """
    flat = final_hidden.reshape(-1, final_hidden.shape[-1])
    targets = torch.full((flat.shape[0],), -1, dtype=torch.long, device=flat.device)
    shifted = ids[:, 1:].reshape(-1)
    stride = ids.shape[1]
    for row in range(ids.shape[0]):
        start = row * stride
        targets[start:start + stride - 1] = shifted[row * (stride - 1):(row + 1) * (stride - 1)]
    entropies, losses = [], []
    with torch.no_grad():
        for start in range(0, flat.shape[0], chunk):
            piece = flat[start:start + chunk]
            logits = head(piece).float()
            logprobs = torch.log_softmax(logits, dim=-1)
            entropies.append(-(logprobs.exp() * logprobs).sum(-1).cpu())
            target = targets[start:start + chunk]
            safe = target.clamp_min(0)
            taken = -logprobs.gather(1, safe.unsqueeze(1)).squeeze(1)
            losses.append(torch.where(target >= 0, taken, torch.zeros_like(taken)).cpu())
    return torch.cat(entropies), torch.cat(losses)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf", default=base.DEFAULT_GGUF)
    parser.add_argument("--student", default=base.DEFAULT_STUDENT)
    parser.add_argument("--weights", default=base.DEFAULT_WEIGHTS)
    parser.add_argument("--documents", default=base.DEFAULT_DOCUMENTS)
    parser.add_argument("--docs", type=int, default=32)
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--report")
    args = parser.parse_args()

    if args.device == "cpu":
        torch.cuda.is_available = lambda: False

    from transformers import AutoTokenizer

    from distillkit.independent_eval import role_spans
    from distillkit.models.qwen35_sidecar import Qwen35SidecarForCausalLM
    from distillkit.ngram_hash import NGramHasher
    from distillkit.ngram_table import GGUFNGramTable, IQ4NLDequant
    from distillkit.sidecar_collator import SidecarDataCollator

    weights = torch.load(args.weights, map_location="cpu")
    streams = weights["key_proj.weight"].shape[0] // weights["value_proj.weight"].shape[0]

    tokenizer = AutoTokenizer.from_pretrained(args.student)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    texts, encodings = [], []
    with open(args.documents, encoding="utf-8") as handle:
        for line in handle:
            text = json.loads(line).get("text", "")
            if not text:
                continue
            texts.append(text)
            encodings.append(tokenizer(text, return_offsets_mapping=True))
            if len(texts) >= args.docs:
                break

    table = GGUFNGramTable(args.gguf)
    collator = SidecarDataCollator(base._PadCollator(pad_id), table, NGramHasher())
    batch = collator([{"input_ids": e["input_ids"][: args.tokens]} for e in encodings])
    ids, mask = batch["input_ids"], batch["attention_mask"]
    features = IQ4NLDequant(out_dtype=torch.float32)(batch["ngram_raw"]).flatten(-2)

    device = torch.device(args.device)
    model = Qwen35SidecarForCausalLM.from_pretrained(args.student, dtype=torch.bfloat16).eval()
    model.config.use_cache = False
    model.to(device)
    layer_index = model.config.sidecar_layer_index
    # One forward for both: hidden_states[layer_index] is what that layer receives, and
    # the last hidden state is what the head reads. Capturing them separately would mean
    # running the backbone twice.
    with torch.no_grad():
        out = model(input_ids=ids.to(device), attention_mask=mask.to(device),
                    sidecar_enabled=False, output_hidden_states=True, logits_to_keep=1)
    hidden = out.hidden_states[layer_index].float().cpu()
    hidden_size = hidden.shape[-1]
    entropy, actual_nll = uncertainty(out.hidden_states[-1], model.get_output_embeddings(),
                                      ids.to(device))
    del model, out
    if device.type == "cuda":
        torch.cuda.empty_cache()

    key = base.grouped_rms_norm(features @ weights["key_proj.weight"].float().T,
                                weights["norm_key.weight"], streams)
    query = base.grouped_rms_norm(hidden.float().repeat(1, 1, streams),
                                  weights["norm_query.weight"], streams)

    keep = mask.bool().reshape(-1)
    flat_key = key.reshape(-1, streams, hidden_size)[keep]
    flat_query = query.reshape(-1, streams, hidden_size)[keep]
    count = flat_key.shape[0]
    generator = torch.Generator().manual_seed(11)
    key_shuffle = torch.randperm(count, generator=generator)
    query_shuffle = torch.randperm(count, generator=generator)

    # The mean key is the limit of "the gate is not looking at the row": every position
    # is offered the same average n-gram. If the real gate tracks that, the table's
    # contribution to the gate is a constant plus noise.
    mean_key = flat_key.mean(dim=0, keepdim=True).expand_as(flat_key)
    arms = {
        "real": (flat_key, flat_query),
        "key_permuted": (flat_key[key_shuffle], flat_query),
        "query_permuted": (flat_key, flat_query[query_shuffle]),
        "both_permuted": (flat_key[key_shuffle], flat_query[query_shuffle]),
        "mean_key": (mean_key, flat_query),
    }
    gates = {name: gate_from(k, q, hidden_size) for name, (k, q) in arms.items()}

    # What the query could be encoding, at the layer the sidecar occupies.
    stream_norm = hidden.float().norm(dim=-1).reshape(-1)[keep]
    position = torch.arange(ids.shape[1]).expand(ids.shape[0], -1).reshape(-1)[keep].float()
    role_index = torch.zeros(ids.shape[0], ids.shape[1], dtype=torch.long)
    role_names = ["template", "system", "user", "assistant"]
    for row, (text, encoding) in enumerate(zip(texts, encodings)):
        spans = role_spans(text, encoding["offset_mapping"])
        for name, ranges in spans.items():
            for start, stop in ranges:
                if start < ids.shape[1]:
                    role_index[row, start:min(stop, ids.shape[1])] = role_names.index(name)
    flat_role = role_index.reshape(-1)[keep]
    flat_entropy = entropy[keep].float()
    flat_nll = actual_nll[keep].float()
    role_onehot = [(flat_role == index).float() for index in range(1, len(role_names))]

    report = {"hc_count": streams, "documents": len(texts), "tokens": int(count),
              "streams": []}
    for stream in range(streams):
        real = gates["real"][:, stream]
        entry = {
            "stream": stream,
            "gate_mean": real.mean().item(),
            "gate_std": real.std().item(),
            "query_only_r2": correlation(real, gates["key_permuted"][:, stream]) ** 2,
            "key_only_r2": correlation(real, gates["query_permuted"][:, stream]) ** 2,
            "both_permuted_r2": correlation(real, gates["both_permuted"][:, stream]) ** 2,
            "mean_key_r2": correlation(real, gates["mean_key"][:, stream]) ** 2,
            "mean_key_gate_mean": gates["mean_key"][:, stream].mean().item(),
            "mean_key_gate_std": gates["mean_key"][:, stream].std().item(),
            "r2_from_stream_norm": explained(real, stream_norm),
            "r2_from_position": explained(real, position),
            "r2_from_role": explained(real, *role_onehot),
            "r2_from_entropy": explained(real, flat_entropy),
            "r2_from_token_nll": explained(real, flat_nll),
            "r2_from_norm_position_role": explained(real, stream_norm, position, *role_onehot),
            "r2_from_everything": explained(real, stream_norm, position, flat_entropy,
                                            flat_nll, *role_onehot),
            "correlation_with_entropy": correlation(real, flat_entropy),
            "gate_by_role": {},
        }
        for index, name in enumerate(role_names):
            picked = real[flat_role == index]
            if len(picked):
                entry["gate_by_role"][name] = {"mean": picked.mean().item(),
                                               "tokens": int(len(picked))}
        report["streams"].append(entry)
        print("stream %d  gate %.4f+-%.4f" % (stream, entry["gate_mean"], entry["gate_std"]))
        print("    variance explained by  query alone %.3f   key alone %.3f   neither %.3f"
              % (entry["query_only_r2"], entry["key_only_r2"], entry["both_permuted_r2"]))
        print("    average row instead of the real one: r2 %.3f  gate %.4f+-%.4f"
              % (entry["mean_key_r2"], entry["mean_key_gate_mean"], entry["mean_key_gate_std"]))
        print("    query-side regressors:  |stream| %.3f   position %.3f   role %.3f"
              % (entry["r2_from_stream_norm"], entry["r2_from_position"], entry["r2_from_role"]))
        print("    backbone uncertainty:  entropy %.3f (r %+.3f)   token NLL %.3f   everything %.3f"
              % (entry["r2_from_entropy"], entry["correlation_with_entropy"],
                 entry["r2_from_token_nll"], entry["r2_from_everything"]))
        print("    by role: " + "  ".join(
            "%s %.4f" % (name, value["mean"]) for name, value in entry["gate_by_role"].items()))

    report["quality_evaluation"] = False
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

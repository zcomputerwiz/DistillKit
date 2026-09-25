"""Which of a trained CSA2 model's Full layers could borrow instead: Reuse or Reindex.

A borrowing layer drops its own latent -- that is the cache saving, one layer's
`latent + rope` per token -- and reads the nearest earlier Full layer's through
`kv_adapt`, with the donor's rotary key taken as it is. Reuse also takes the donor's
selection; Reindex scores the donor's index keys with its own index queries.

Measured per ordered pair (donor i, borrower j), on calibration text:

* **latent R^2** -- how much of j's latent a linear map of i's explains, which is the best
  `kv_adapt` can start from; also scored in j's key/value space, through j's `kv_b_proj`,
  which is what attention reads.
* **rotary cosine** -- the borrower reads the donor's rotary key with no adapter.
* **reuse recall** -- of the positions j reads, the share i reads too.
* **reindex recall** -- of the positions j reads, the share j's own index queries still
  pick when they score i's index keys.

Then the test that decides it: rebuild the model with a mode pattern, `kv_adapt` fitted by
least squares on the calibration latents, and score held-out NLL against the unchanged
model, paired per document. No training, so it bounds how good a start each pattern gets.

    python scratch/dense_gr/borrow_profile.py --checkpoint <ckpt> --output borrow.json \\
        --patterns F,U,F,F,F,F F,I,F,F,F,F
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import smoke_train  # noqa: E402,F401  (triton metadata shim)
import torch  # noqa: E402
from cut_cross_entropy import linear_cross_entropy  # noqa: E402

from distillkit.models import Qwen35WidenedForCausalLM  # noqa: E402
from distillkit.models.qwen35.csa2 import (Qwen35SparseLatentAttention,  # noqa: E402
                                           SparseIndexBus)
from teacher_kl import CachedTeacher  # noqa: E402

LETTER = {"F": "full", "I": "reindex", "U": "reuse"}
RIDGE = 1e-6


def documents(tokenizer, chat, count, length, device):
    """`count` WikiText validation windows and `count` held-out chat documents, twice.

    WikiText *validation*, not test: the test split is the reported yardstick, and the
    patterns chosen here should not be chosen on it.
    """
    from datasets import load_dataset

    text = "\n".join(load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1",
                                  split="validation")["text"])
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    stride = (len(ids) - length) // (2 * count)
    wiki = [torch.tensor(ids[i * stride:i * stride + length], device=device).view(1, -1)
            for i in range(2 * count)]
    teacher = CachedTeacher(chat, "eval", device=device, max_length=length)
    talk = [teacher.read(doc)["input_ids"].to(device)
            for _, doc in teacher.stratified(2 * count)]
    calibration = wiki[:count] + talk[:count]
    probe = wiki[count:] + talk[count:2 * count]
    return calibration, probe


def routing(model):
    return [(i, layer.self_attn) for i, layer in enumerate(model.model.layers)
            if isinstance(getattr(layer, "self_attn", None), Qwen35SparseLatentAttention)]


@torch.no_grad()
def capture(model, docs):
    """Per routing layer and document: latent, rotary, index keys, index queries, selection."""
    book = {i: [] for i, _ in routing(model)}
    original_publish = SparseIndexBus.publish
    original_select = Qwen35SparseLatentAttention._select_positions
    current = {}

    def publish(bus, layer_idx, index_keys, latent, rotary):
        current[layer_idx] = dict(latent=latent[0].float(), rotary=rotary[0].float(),
                                  keys=index_keys[0].float())
        return original_publish(bus, layer_idx, index_keys, latent, rotary)

    def select(module, queries, keys, weights):
        positions, valid = original_select(module, queries, keys, weights)
        current[module.layer_idx].update(queries=queries[0].float(), weights=weights[0].float(),
                                         positions=positions[0], valid=valid[0])
        return positions, valid

    SparseIndexBus.publish = publish
    Qwen35SparseLatentAttention._select_positions = select
    try:
        for ids in docs:
            current.clear()
            model.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
            for i in book:
                book[i].append({k: v.cpu() for k, v in current[i].items()})
    finally:
        SparseIndexBus.publish = original_publish
        Qwen35SparseLatentAttention._select_positions = original_select
    return book


def solve(source, target):
    """Least squares `target ~= source @ W.T`, ridge-stabilized, in float64."""
    source, target = source.double(), target.double()
    gram = source.T @ source
    gram += torch.eye(gram.shape[0], dtype=gram.dtype) * gram.diagonal().mean() * RIDGE
    return torch.linalg.solve(gram, source.T @ target).T


def r2(prediction, target):
    return float(1 - (prediction - target).pow(2).sum() / (target - target.mean(0)).pow(2).sum())


def dense(positions, valid, seq):
    allowed = torch.zeros(seq, seq, dtype=torch.int16)
    allowed.scatter_add_(-1, positions, valid.to(torch.int16))
    return allowed > 0


def pair_metrics(model, book, calibration_count):
    layers = dict(routing(model))
    order = sorted(layers)
    rows = []
    for a, i in enumerate(order):
        for j in order[a + 1:]:
            fit = [slice(0, calibration_count)]
            cat = lambda key, who, part: torch.cat([d[key] for d in book[who][part]])
            src, dst = cat("latent", i, fit[0]), cat("latent", j, fit[0])
            adapt = solve(src, dst).float()
            predicted = src @ adapt.T
            up = layers[j].kv_b_proj.weight.float().cpu()
            kv_true, kv_pred = dst @ up.T, predicted @ up.T
            rot_i, rot_j = cat("rotary", i, fit[0]), cat("rotary", j, fit[0])
            cosine = float(torch.nn.functional.cosine_similarity(rot_i, rot_j, dim=-1).mean())
            reuse, reindex, counted = 0.0, 0.0, 0
            probe = layers[j]
            for doc_i, doc_j in zip(book[i], book[j]):
                seq = doc_j["latent"].shape[0]
                if seq <= probe.top_k + probe.local_window:
                    continue
                mine = dense(doc_j["positions"], doc_j["valid"], seq)
                theirs = dense(doc_i["positions"], doc_i["valid"], seq)
                queries = doc_j["queries"].unsqueeze(0).to(up.dtype)
                weights = doc_j["weights"].unsqueeze(0)
                keys = doc_i["keys"].unsqueeze(0)
                positions, valid = probe._select_positions(queries, keys, weights)
                rescored = dense(positions[0], valid[0], seq)
                late = slice(probe.top_k + probe.local_window, seq)
                total = mine[late].sum().clamp_min(1)
                reuse += float((mine[late] & theirs[late]).sum() / total)
                reindex += float((mine[late] & rescored[late]).sum() / total)
                counted += 1
            rows.append(dict(donor=i, borrower=j, latent_r2=r2(predicted, dst),
                             kv_r2=r2(kv_pred, kv_true), rotary_cosine=cosine,
                             reuse_recall=reuse / max(counted, 1),
                             reindex_recall=reindex / max(counted, 1)))
            print("donor %2d -> %2d  latent R2 %.3f  kv R2 %.3f  rotary cos %.3f  "
                  "reuse recall %.3f  reindex recall %.3f"
                  % (i, j, rows[-1]["latent_r2"], rows[-1]["kv_r2"], cosine,
                     rows[-1]["reuse_recall"], rows[-1]["reindex_recall"]), flush=True)
    return rows


class OwnRotary:
    """A bus view for one borrower that hands it its own rotary key instead of the donor's.

    The experiment behind `--own-rotary`: the borrower keeps the 64 rotary rows of its
    down projection and caches its own rotary key, borrowing only the latent -- 86% of a
    layer's cache saved instead of all of it. Measures how much of a swap's cost is the
    donor's rotary key, which a converted model's layers do not share.
    """

    def __init__(self, bus, projection, owner):
        self.bus, self.projection, self.owner, self.rotary = bus, projection, owner, None

    def __getattr__(self, name):
        return getattr(self.bus, name)

    def require_latent(self, donor, layer_idx):
        keys, latent, _ = self.bus.require_latent(donor, layer_idx)
        return keys, latent, self.rotary


def apply_pattern(model, pattern, book, calibration_count, own_rotary=False):
    """Swap the attention modules to `pattern`; returns an undo list."""
    order = [i for i, _ in routing(model)]
    modes = [LETTER[c] for c in pattern]
    config = copy.deepcopy(model.config)
    config.csa2_modes = modes
    undo = []
    for position, (index, mode) in enumerate(zip(order, modes)):
        old = model.model.layers[index].self_attn
        if mode == "full" and old.mode == "full":
            continue
        new = Qwen35SparseLatentAttention(config, index, mode)
        new.load_state_dict(old.state_dict(), strict=False)
        new = new.to(device=old.q_proj.weight.device, dtype=old.q_proj.weight.dtype)
        new.bus, new.query_chunk = old.bus, old.query_chunk
        donor = new.latent_donor
        src = torch.cat([d["latent"] for d in book[donor][:calibration_count]])
        dst = torch.cat([d["latent"] for d in book[index][:calibration_count]])
        with torch.no_grad():
            new.kv_adapt.weight.copy_(solve(src, dst).to(new.kv_adapt.weight))
        new.eval()
        if own_rotary:
            view = OwnRotary(old.bus, old.kv_a_proj, old)
            new.bus = view

            def keep_rotary(module, args, kwargs, view=view, old=old):
                hidden = kwargs.get("hidden_states", args[0] if args else None)
                rotary = old.kv_a_proj(hidden)[..., old.latent:]
                view.rotary = old._rope_shared(rotary, kwargs.get("position_embeddings"))

            new.register_forward_pre_hook(keep_rotary, with_kwargs=True)
        model.model.layers[index].self_attn = new
        undo.append((index, old))
    return undo


@torch.no_grad()
def score(model, docs):
    out = []
    for ids in docs:
        state = model.model(input_ids=ids, attention_mask=torch.ones_like(ids),
                            use_cache=False).last_hidden_state
        out.append(float(linear_cross_entropy(state, model.lm_head.weight, ids, shift=1,
                                              reduction="mean")))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--chat", type=Path, default=Path("../teacher-cache-expand-chat"))
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--length", type=int, default=1024)
    parser.add_argument("--patterns", nargs="*", default=None,
                        help="mode letters per Full layer, F/I/U; default: every single swap")
    parser.add_argument("--skip-pairs", action="store_true")
    parser.add_argument("--save", nargs=2, action="append", default=[],
                        metavar=("PATTERN", "DIR"),
                        help="write the model converted to PATTERN, kv_adapt fitted, to DIR")
    parser.add_argument("--own-rotary", action="store_true",
                        help="borrowers keep their own rotary key; see OwnRotary")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = Qwen35WidenedForCausalLM.from_pretrained(
        args.checkpoint, dtype=torch.bfloat16).to("cuda").eval()
    calibration, probe = documents(tokenizer, [args.chat], args.count, args.length, "cuda")
    count = len(calibration)
    book = capture(model, calibration)
    report = dict(checkpoint=str(args.checkpoint), calibration_docs=count,
                  probe_docs=len(probe))
    if not args.skip_pairs:
        report["pairs"] = pair_metrics(model, book, count)

    layers = len(routing(model))
    patterns = args.patterns or [
        "".join("F" if p != k else letter for p in range(layers))
        for k in range(1, layers) for letter in "UI"]
    baseline = score(model, probe)
    half = len(probe) // 2
    report["baseline"] = dict(wiki=sum(baseline[:half]) / half,
                              chat=sum(baseline[half:]) / (len(probe) - half))
    print("baseline  wiki %.4f  chat %.4f" % (report["baseline"]["wiki"],
                                              report["baseline"]["chat"]), flush=True)
    report["patterns"] = []
    for pattern in patterns:
        pattern = pattern.replace(",", "")
        undo = apply_pattern(model, pattern, book, count, args.own_rotary)
        try:
            scores = score(model, probe)
        finally:
            for index, old in undo:
                model.model.layers[index].self_attn = old
        delta = [a - b for a, b in zip(scores, baseline)]
        row = dict(pattern=pattern)
        for name, part in (("wiki", slice(0, half)), ("chat", slice(half, None))):
            d = delta[part]
            mean = sum(d) / len(d)
            se = (sum((x - mean) ** 2 for x in d) / (len(d) - 1) / len(d)) ** 0.5
            row[name] = mean
            row[name + "_se"] = se
        report["patterns"].append(row)
        print("%s  wiki %+.4f (se %.4f)  chat %+.4f (se %.4f)"
              % (pattern, row["wiki"], row["wiki_se"], row["chat"], row["chat_se"]), flush=True)
    for pattern, target in args.save:
        target = Path(target)
        if target.exists():
            raise SystemExit("refusing to overwrite %s" % target)
        undo = apply_pattern(model, pattern.replace(",", ""), book, count)
        modes = [LETTER[c] for c in pattern.replace(",", "")]
        original = list(model.config.csa2_modes)
        model.config.csa2_modes = modes
        try:
            model.save_pretrained(target, safe_serialization=True)
            tokenizer.save_pretrained(target)
            (target / "conversion.json").write_text(json.dumps(dict(
                source=str(args.checkpoint), pattern=pattern, modes=modes,
                kv_adapt="least squares on %d calibration documents" % count), indent=2),
                encoding="utf-8")
            print("saved %s -> %s" % (pattern, target), flush=True)
        finally:
            model.config.csa2_modes = original
            for index, old in undo:
                model.model.layers[index].self_attn = old
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

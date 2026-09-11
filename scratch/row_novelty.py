"""Stratify the sidecar's effect by how novel each position's table rows are.

The question the C arms left open: does the layer-1 sidecar generalise, or did it only
memorise the rows its training corpus happened to touch? Every scored position addresses
16 rows (8 bigram heads, 8 trigram heads). Let ``k`` be how many of those 16 the training
run also touched. ``k = 16`` is a position the sidecar has seen every row of; ``k = 0`` is
a position whose entire address is new. If the pathway generalises, its benefit survives
at ``k = 0``.

Two alignment rules, both load-bearing:

* ``k`` is always computed from the **real** row set. The roll is a *training* setting:
  both arms are evaluated on real rows, so both read the same 16 addresses per position
  and ``k`` is unambiguously the same axis for each. What differs is which rows each arm
  was trained to interpret.
* The seen set is always **C1's**, the real-context training run. The question is whether
  C1 generalises, so C1's exposure is the axis. C2's own exposure covers a different
  region of the table and is not what is being asked about.

The loss on target token ``i`` comes from the logits at position ``i - 1``, which is where
the sidecar injected, so ``k`` for target ``i`` is the row novelty of position ``i - 1``.

``k = 0`` is not perfectly clean: rows are hashed into ~20M buckets per head against
6e10 possible bigrams, so a genuinely new n-gram can still land on a bucket that training
moved. That collision makes ``k = 0`` *understate* novelty, which biases toward finding
generalisation. A null result at ``k = 0`` is therefore the stronger conclusion.

    python scratch/row_novelty.py seen    # build C1's training row set, once
    python scratch/row_novelty.py score --arm win-C1-L1-real-stage1-1m --checkpoint ...
    python scratch/row_novelty.py report
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from distillkit.ngram_hash import NGramHasher

BASE = Path("scratch/independent-eval")
BUNDLE = BASE / "reply-bundle-384.json"
OUT = Path("scratch/row-novelty")
SEEN = OUT / "seen-rows-1m.npy"
TABLE = ("C:/Users/Owner/.cache/huggingface/hub/models--unsloth--Qwen3.8-Flash-Next-GGUF"
         "/snapshots/38bb39ee97821de2c9009abb7e93950eec396e66/UD-IQ4_XS"
         "/Qwen3.8-Flash-Next-UD-IQ4_XS-00002-of-00003.gguf")
CACHE = Path("D:/DeepThought/Projects/HybridModel/teacher-cache-1m")


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# --- the training row set ----------------------------------------------------


def build_seen(cache=CACHE, split="train"):
    """Every row the training run's real-context hashing addressed.

    Documents are hashed one at a time with the reference's EOS-filled previous context,
    which is exactly what the collator produces for a right-padded row starting at
    position 0. Padding positions hash an all-EOS context and contribute at most one row
    per head, already reachable from any document containing EOS; not replayed here.
    """
    manifest = load_json(cache / "manifest.json")
    hasher = NGramHasher()
    # A shard file is trimmed to the tokens it actually holds, not to shard_tokens, so
    # it is read whole rather than mapped at the declared geometry. 256 KB each.
    maps, chunks = {}, []
    for document in manifest["documents"]:
        if document["split"] != split:
            continue
        shard = document["shard"]
        if shard not in maps:
            path = cache / ("%s-%05d.input_ids.bin" % (split, shard))
            maps[shard] = np.fromfile(path, dtype="<u4")
        start = document["offset"]
        ids = np.asarray(maps[shard][start:start + document["length"]], dtype=np.int64)
        rows = hasher.row_indices(torch.from_numpy(ids).unsqueeze(0))
        chunks.append(np.unique(rows.numpy().ravel()))
    seen = np.unique(np.concatenate(chunks))
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(SEEN, seen)
    total = sum(d["length"] for d in manifest["documents"] if d["split"] == split)
    print(json.dumps({"documents": len(chunks), "tokens": total,
                      "row_touches": total * hasher.config.ngram_heads,
                      "unique_rows": int(len(seen)),
                      "table_rows": hasher.padded_vocab_size,
                      "fraction_of_table": len(seen) / hasher.padded_vocab_size}))


# --- per-token scoring -------------------------------------------------------


def novelty_per_position(ids, seen, hasher):
    """``k`` in [0, 16] for every position, from the real rows regardless of arm."""
    rows = hasher.row_indices(torch.tensor(ids, dtype=torch.long).unsqueeze(0))[0].numpy()
    index = np.searchsorted(seen, rows)
    index[index >= len(seen)] = len(seen) - 1
    return (seen[index] == rows).sum(axis=1).astype(np.int8)


def novelty_for_targets(ids, targets, seen, hasher):
    """Target ``i`` is predicted from position ``i - 1``, which is where the sidecar
    injected -- so that is the position whose row novelty governs this token's loss.
    An off-by-one here would be invisible in any total and would scramble every
    stratum, so it is its own function with its own test."""
    return novelty_per_position(ids, seen, hasher)[np.asarray(targets) - 1]


@torch.inference_mode()
def score(arm, checkpoint, device="cuda:0"):
    import torch.nn.functional as F
    from distillkit.independent_eval import forward_logits, load_checkpoint, make_collator
    from distillkit.ngram_table import GGUFNGramTable

    bundle = load_json(BUNDLE)
    records = bundle["splits"]["screen"]["nll"]
    seen = np.load(SEEN)
    hasher = NGramHasher()

    model, audit = load_checkpoint(checkpoint, device, torch.bfloat16)
    if not audit["variant"]:
        raise ValueError("%s has no sidecar; there is nothing to stratify" % checkpoint)
    collator = make_collator(bundle["pad_token_id"], GGUFNGramTable(TABLE))

    start = 1
    docs, kept = [], {"k": [], "enabled": [], "bypassed": [], "doc": []}
    for number, record in enumerate(records):
        ids = record["ids"]
        picked = np.array([i for low, high in record["roles"].get("assistant", [])
                           for i in range(max(low, start), min(high, len(ids)))], dtype=np.int64)
        if not len(picked):
            continue
        positions = torch.arange(start - 1, len(ids) - 1, device=device)
        batch = {k: v.to(device) for k, v in collator([record]).items()}
        target = torch.tensor(ids[start:], device=device)
        losses = {}
        for mode in ("enabled", "bypassed"):
            logits = forward_logits(model, batch, mode, positions)[0].float()
            losses[mode] = F.cross_entropy(logits, target, reduction="none").cpu().numpy()
        kept["k"].append(novelty_for_targets(ids, picked, seen, hasher))
        for mode in ("enabled", "bypassed"):
            kept[mode].append(losses[mode][picked - start])
        kept["doc"].append(np.full(len(picked), len(docs), dtype=np.int32))
        docs.append(record["id"])
        if (number + 1) % 64 == 0:
            print("%s %d/%d" % (arm, number + 1, len(records)), flush=True)

    packed = {name: np.concatenate(values) for name, values in kept.items()}
    packed["doc_ids"] = np.array(docs)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / ("tokens-%s.npz" % arm), **packed)
    print(json.dumps({"arm": arm, "documents": len(docs), "tokens": int(len(packed["k"])),
                      "assistant_sum_nll_enabled": float(packed["enabled"].sum()),
                      "assistant_sum_nll_bypassed": float(packed["bypassed"].sum())}))


# --- the report --------------------------------------------------------------


def by_document(packed, mask, documents):
    """Per-document sums over a stratum, so the bootstrap resamples documents."""
    enabled = np.bincount(packed["doc"][mask], packed["enabled"][mask], documents)
    bypassed = np.bincount(packed["doc"][mask], packed["bypassed"][mask], documents)
    tokens = np.bincount(packed["doc"][mask], minlength=documents).astype(np.float64)
    return enabled, bypassed, tokens


def bootstrap(numerators, tokens, draws=10000, seed=20260911):
    """Ratio of sums with documents as the resampling unit, as elsewhere in this repo."""
    rng = np.random.default_rng(seed)
    index = rng.integers(0, len(tokens), (draws, len(tokens)))
    denominator = tokens[index].sum(1)
    samples = np.where(denominator > 0,
                       numerators[index].sum(1) / np.maximum(denominator, 1), np.nan)
    low, high = np.nanquantile(samples, [0.025, 0.975])
    return numerators.sum() / tokens.sum(), low, high


def paired_gap(left, right, tokens_left, tokens_right, draws=10000, seed=20260911):
    """One document resample drives both arms: the same documents, so the draw must match."""
    rng = np.random.default_rng(seed)
    index = rng.integers(0, len(tokens_left), (draws, len(tokens_left)))
    a, b = tokens_left[index].sum(1), tokens_right[index].sum(1)
    samples = np.where((a > 0) & (b > 0),
                       left[index].sum(1) / np.maximum(a, 1) - right[index].sum(1) / np.maximum(b, 1),
                       np.nan)
    low, high = np.nanquantile(samples, [0.025, 0.975])
    return left.sum() / tokens_left.sum() - right.sum() / tokens_right.sum(), low, high


def report(arms):
    loaded = {}
    for arm in arms:
        path = OUT / ("tokens-%s.npz" % arm)
        if not path.exists():
            print("skipping %s (not scored yet)" % arm)
            continue
        loaded[arm] = dict(np.load(path, allow_pickle=True))
    if not loaded:
        return 1
    reference = None
    for arm, packed in loaded.items():
        ids = list(packed["doc_ids"])
        if reference is None:
            reference = ids
        elif ids != reference:
            raise ValueError("%s scored different documents" % arm)
    documents = len(reference)

    for arm, packed in loaded.items():
        counts = np.bincount(packed["k"], minlength=17)
        print("\n%s  %d documents, %d assistant tokens" % (arm, documents, len(packed["k"])))
        print("   k   tokens  share    enabled   bypassed   sidecar cost  [95%]")
        for k in range(17):
            if not counts[k]:
                continue
            enabled, bypassed, tokens = by_document(packed, packed["k"] == k, documents)
            cost, low, high = bootstrap(enabled - bypassed, tokens)
            print("  %2d %8d %5.1f%%  %9.4f  %9.4f   %+.6f [%+.6f, %+.6f]"
                  % (k, counts[k], 100 * counts[k] / len(packed["k"]),
                     enabled.sum() / tokens.sum(), bypassed.sum() / tokens.sum(),
                     cost, low, high))

    names = list(loaded)
    if len(names) < 2:
        return 0
    real = next((n for n in names if "real" in n), names[0])
    shuffled = next((n for n in names if "shuffled" in n), names[1])
    print("\n%s minus %s, per novelty stratum" % (real, shuffled))
    print("  (G0 is the k=0 row: what the pathway buys where its whole address is new)")
    print("   k   real       shuffled   gap        [95%]")
    for k in range(17):
        mask_real = loaded[real]["k"] == k
        mask_shuffled = loaded[shuffled]["k"] == k
        if not mask_real.sum() or not mask_shuffled.sum():
            continue
        a_enabled, a_bypassed, a_tokens = by_document(loaded[real], mask_real, documents)
        b_enabled, b_bypassed, b_tokens = by_document(loaded[shuffled], mask_shuffled, documents)
        gap, low, high = paired_gap(a_enabled - a_bypassed, b_enabled - b_bypassed,
                                    a_tokens, b_tokens)
        print("  %2d  %+.6f  %+.6f  %+.6f [%+.6f, %+.6f]%s"
              % (k, (a_enabled - a_bypassed).sum() / a_tokens.sum(),
                 (b_enabled - b_bypassed).sum() / b_tokens.sum(), gap, low, high,
                 "  <- G0" if k == 0 else ""))
    return 0


# --- is k just a frequency proxy? --------------------------------------------


FREQUENCY_BINS = ((1, 1), (2, 2), (3, 5), (6, 20), (21, 100), (101, 10 ** 9))


def eval_bigram_counts(records):
    """How often each scored position's bigram address occurs across the held-out set.

    ``k`` is confounded: a context is in the training row set largely because it is
    common, and common contexts are also the ones the backbone already predicts well
    (the k=16 stratum's *bypassed* NLL is 0.475 against 0.539 at k=0, so those positions
    are intrinsically easier before the sidecar does anything). Counting occurrences
    within the evaluation corpus gives a frequency measure that owes nothing to the
    training corpus and nothing to any model, so exposure can be compared at matched
    frequency.

    The bigram is the right key: all eight bigram heads share the address
    ``(ids[p-1], ids[p])``, and it is that address being seen or not that separates
    k=0 from k>=8.
    """
    from collections import Counter
    counts = Counter()
    for record in records:
        ids = record["ids"]
        for position in range(1, len(ids)):
            counts[(ids[position - 1], ids[position])] += 1
    return counts


def scored_positions(records, start=1):
    """The same assistant targets ``score`` kept, in the same order."""
    for record in records:
        picked = [i for low, high in record["roles"].get("assistant", [])
                  for i in range(max(low, start), min(high, len(record["ids"])))]
        if picked:
            yield record, np.array(picked, dtype=np.int64)


def frequency(arms):
    """Does exposure still buy anything once frequency is held fixed?"""
    bundle = load_json(BUNDLE)
    records = bundle["splits"]["screen"]["nll"]
    counts = eval_bigram_counts(records)
    frequencies = []
    for record, picked in scored_positions(records):
        ids = record["ids"]
        # Target i was predicted from position i-1, whose bigram address is (i-2, i-1).
        frequencies.append(np.array([counts[(ids[i - 2], ids[i - 1])] for i in picked]))
    frequencies = np.concatenate(frequencies)

    loaded = {}
    for arm in arms:
        path = OUT / ("tokens-%s.npz" % arm)
        if path.exists():
            loaded[arm] = dict(np.load(path, allow_pickle=True))
    if not loaded:
        return 1
    for arm, packed in loaded.items():
        if len(packed["k"]) != len(frequencies):
            raise ValueError("%s scored %d tokens, the bundle replays %d"
                             % (arm, len(packed["k"]), len(frequencies)))

    for arm, packed in loaded.items():
        documents = len(packed["doc_ids"])
        print("\n%s: sidecar cost at matched bigram frequency" % arm)
        print("  bigram seen    k=0 tokens          cost   k=16 tokens         cost")
        for low, high in FREQUENCY_BINS:
            window = (frequencies >= low) & (frequencies <= high)
            cells = []
            for k in (0, 16):
                mask = window & (packed["k"] == k)
                if mask.sum() < 200:
                    cells.append("%8d       --     " % int(mask.sum()))
                    continue
                enabled, bypassed, tokens = by_document(packed, mask, documents)
                estimate, _, _ = bootstrap(enabled - bypassed, tokens)
                cells.append("%8d   %+.6f" % (int(mask.sum()), estimate))
            label = ("%d" % low if low == high
                     else "%d+" % low if high > 10 ** 8 else "%d-%d" % (low, high))
            print("  %-12s %s   %s" % (label, cells[0], cells[1]))
    print("\n  (a cell under 200 tokens is left blank rather than reported as noise)")

    # Frequency on its own, which the crossed table above shows is the variable that
    # actually moves: exposure changes almost nothing within a bin, and the whole
    # benefit sits in the most frequent one.
    for arm, packed in loaded.items():
        documents = len(packed["doc_ids"])
        print("\n%s: sidecar cost by bigram frequency alone" % arm)
        print("  bigram seen    tokens  share       cost  [95%]")
        for low, high in FREQUENCY_BINS:
            mask = (frequencies >= low) & (frequencies <= high)
            if not mask.sum():
                continue
            enabled, bypassed, tokens = by_document(packed, mask, documents)
            cost, lower, upper = bootstrap(enabled - bypassed, tokens)
            label = ("%d" % low if low == high
                     else "%d+" % low if high > 10 ** 8 else "%d-%d" % (low, high))
            print("  %-12s %8d %5.1f%%  %+.6f [%+.6f, %+.6f]"
                  % (label, int(mask.sum()), 100 * mask.sum() / len(frequencies),
                     cost, lower, upper))

    names = list(loaded)
    if len(names) < 2:
        return 0
    real = next((n for n in names if "real" in n), names[0])
    shuffled = next((n for n in names if "shuffled" in n), names[1])
    documents = len(loaded[real]["doc_ids"])
    print("\n%s minus %s, by bigram frequency" % (real, shuffled))
    print("  bigram seen    real       shuffled   gap        [95%]")
    for low, high in FREQUENCY_BINS:
        mask = (frequencies >= low) & (frequencies <= high)
        if not mask.sum():
            continue
        a_enabled, a_bypassed, a_tokens = by_document(loaded[real], mask, documents)
        b_enabled, b_bypassed, b_tokens = by_document(loaded[shuffled], mask, documents)
        gap, lower, upper = paired_gap(a_enabled - a_bypassed, b_enabled - b_bypassed,
                                       a_tokens, b_tokens)
        label = ("%d" % low if low == high
                 else "%d+" % low if high > 10 ** 8 else "%d-%d" % (low, high))
        print("  %-12s %+.6f  %+.6f  %+.6f [%+.6f, %+.6f]"
              % (label, (a_enabled - a_bypassed).sum() / a_tokens.sum(),
                 (b_enabled - b_bypassed).sum() / b_tokens.sum(), gap, lower, upper))
    return 0


# --- layout against content --------------------------------------------------


#: `\n` and the think-block tags. Ranking the 101+ frequency bin's 2,845 target token
#: types by nats contributed puts `\n` at -708.0 (89.9% of the bin) and `<think>` at
#: -201.6 (25.6%); together they are 2.45x the entire net win over 5.3% of the tokens.
#: They are not reassigned to the `template` role: `role_spans` strips only the leading
#: empty think block, and moving these would silently redefine the assistant span in
#: every stored bundle and reply file while still leaving `\n` -- which is ordinary text
#: and cannot be called template -- inside it. Reporting the split says more and
#: invalidates nothing.
LAYOUT_TOKEN_IDS = (198, 248068, 248069)      # \n, <think>, </think>


def target_token_ids(records, start=1):
    """The target token behind every scored position, in `score`'s order."""
    return np.concatenate([np.array([record["ids"][i] for i in picked])
                           for record, picked in scored_positions(records)])


def layout(arms):
    """How much of an arm's result is line breaks and think tags rather than content."""
    records = load_json(BUNDLE)["splits"]["screen"]["nll"]
    targets = target_token_ids(records)
    is_layout = np.isin(targets, LAYOUT_TOKEN_IDS)
    print("%d of %d assistant tokens are layout (%.1f%%)"
          % (is_layout.sum(), len(is_layout), 100 * is_layout.mean()))

    loaded = {}
    for arm in arms:
        path = OUT / ("tokens-%s.npz" % arm)
        if not path.exists():
            continue
        packed = dict(np.load(path, allow_pickle=True))
        if len(packed["k"]) != len(targets):
            raise ValueError("%s scored %d tokens, the bundle replays %d"
                             % (arm, len(packed["k"]), len(targets)))
        loaded[arm] = packed
    if not loaded:
        return 1

    views = (("all", np.ones(len(is_layout), bool)),
             ("layout", is_layout), ("content", ~is_layout))
    for arm, packed in loaded.items():
        documents = len(packed["doc_ids"])
        print("\n%s" % arm)
        for label, mask in views:
            enabled, bypassed, tokens = by_document(packed, mask, documents)
            cost, low, high = bootstrap(enabled - bypassed, tokens)
            print("  %-8s %8d tokens  %+.6f [%+.6f, %+.6f]"
                  % (label, int(mask.sum()), cost, low, high))

    names = list(loaded)
    if len(names) < 2:
        return 0
    real = next((n for n in names if "real" in n), names[0])
    shuffled = next((n for n in names if "shuffled" in n), names[1])
    documents = len(loaded[real]["doc_ids"])
    print("\n%s minus %s" % (real, shuffled))
    for label, mask in views:
        a = by_document(loaded[real], mask, documents)
        b = by_document(loaded[shuffled], mask, documents)
        gap, low, high = paired_gap(a[0] - a[1], b[0] - b[1], a[2], b[2])
        print("  %-8s %+.6f [%+.6f, %+.6f]" % (label, gap, low, high))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("seen")
    scorer = sub.add_parser("score")
    scorer.add_argument("--arm", required=True)
    scorer.add_argument("--checkpoint", required=True)
    scorer.add_argument("--device", default="cuda:0")
    default_arms = ["win-C1-L1-real-stage1-1m", "win-C2-L1-shuffled-stage1-1m"]
    reporter = sub.add_parser("report")
    reporter.add_argument("--arms", nargs="+", default=default_arms)
    matched = sub.add_parser("frequency")
    matched.add_argument("--arms", nargs="+", default=default_arms)
    split = sub.add_parser("layout")
    split.add_argument("--arms", nargs="+", default=default_arms)
    args = parser.parse_args()
    if args.command == "seen":
        return build_seen()
    if args.command == "score":
        return score(args.arm, args.checkpoint, args.device)
    if args.command == "frequency":
        return frequency(args.arms)
    if args.command == "layout":
        return layout(args.arms)
    return report(args.arms)


if __name__ == "__main__":
    raise SystemExit(main() or 0)

"""Where does the sidecar's information go, and which part of it is worth anything?

``scratch/rho_diagnostic.py`` answers a different question -- whether the trained
sidecar's *actual* increment helps at rho = 1 -- and its answer is that on assistant
tokens both ground-truth CE and the teacher KL want it smaller. That says the injection
as trained is harmful. It does not say whether the information was there to begin with,
or where between the table and the residual stream it stops being usable.

This grades each stage of the reader separately, at *matched decoding capacity*, on
frozen weights:

    raw_table -> ungated_value -> gated_value -> (+ convolution) -> reader_update

Every arm gets the same trainable parameter count -- a frozen random projection to a
common width, then one linear map into the residual -- so an arm cannot win by being
wider.

**Read every number below as "linearly decodable at the output", never as "how much
this representation knows".** The n-gram row at position t is built from tokens t,
t-1, t-2 (``NGramHasher.row_indices`` shifts right, so it is causal and contains no
part of t+1). It is a *per-layer embedding*: Flash-Next injects it at layer 2 to enrich
the representation of the token that is already there, and the remaining thirty-one
layers are what turn that enrichment into a better prediction. It is not a next-token
predictor and grading it as one would understate it by construction. A near-zero
``raw_table`` here means "a linear map at the output cannot turn this row into a
next-token correction", which is the expected result for an input enrichment and is
*not* evidence the row is empty.

What survives that caveat is the *relative* comparison, because every arm is a layer-1
object graded through the identical protocol. ``query`` is the yardstick: it is the
widened stream the reader reads, an input enrichment exactly like the table row and
graded exactly as unfairly. Anything the reader cannot beat ``query`` by is not
information the reader supplied.

For the absolute question the protocol cannot answer, ``--identity-readout`` asks a
different one that does not treat the row as a predictor at all: with the baseline
zeroed, can a linear map from the row alone name the token *at its own position*? That
is what an embedding is for, and a row that fails it is empty in a way no framing
rescues.

Two controls decide whether a number means anything. A *shuffled* arm takes its
representation from a different document, so it keeps every scale and shape and loses
only the correspondence to this text. The reference every arm is compared against is
the bypassed model's own NLL, which is the examiner at its zero initialisation.

    python scratch/reader_visibility/run.py --docs 61 --identity-readout

Nothing here trains the model, and no arm's fit ever sees a test document.
"""

from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRITON_CACHE_DIR", os.path.abspath("scratch/reader-visibility/triton-cache"))

import argparse
import json
import sys
import time
from pathlib import Path
from types import MethodType

import numpy as np
import torch
import yaml
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transformers import AutoConfig, AutoTokenizer  # noqa: E402

from distillkit.anchor_tap import AnchorTap  # noqa: E402
from distillkit.independent_eval import make_collator, role_spans  # noqa: E402
from distillkit.models.qwen35_widened import Qwen35WidenedForCausalLM  # noqa: E402
from distillkit.ngram_table import GGUFNGramTable  # noqa: E402
from distillkit.signals import OfflineHiddenStateSignalSource  # noqa: E402
from distillkit.tp_model import shard_model  # noqa: E402

from reader_visibility.core import (  # noqa: E402
    MatchedExaminer, compare_scores, digest, donor_map, prediction_positions,
    reader_representations, split_records, token_nll, write_json,
)

CHECKPOINT = Path("../runs/widened-plegated-stage1-1m/checkpoint-72")
CONFIG_FILE = "examples/qwen35_widened_plegated_stage1_1m.yml"

#: Each arm names one key of ``reader_representations``. ``query`` is the control that
#: makes the others mean something: the reader has to beat the stream it reads.
ARMS = ("raw_table", "query", "ungated_value", "upstream_value", "gated_value",
        "convolution", "reader_update")

#: Flash-Next's own trained value projection, from the layer this sidecar transcribes.
#: Its hidden size is 2560, the same as the student's, so the map from dequantized
#: table rows into *its* residual basis applies without reshaping. Whether that basis
#: is any use to a different model is the question; this is the arm that asks it.
UPSTREAM_LAYER = Path("../flash-next-ple/ple_layer.pt")
#: Arms that also get a donor-shuffled twin. The table is the source and the update is
#: the sink; the stages between them inherit whatever those two establish.
SHUFFLED = ("raw_table", "reader_update")


def load_model(dtype):
    """The checkpoint, sharded, frozen, with its pre-fp32-pin sharpness migrated."""
    config = AutoConfig.from_pretrained(CHECKPOINT)
    config = getattr(config, "text_config", config)
    model, info = Qwen35WidenedForCausalLM.from_pretrained(
        CHECKPOINT, config=config, dtype=torch.bfloat16, attn_implementation="sdpa",
        output_loading_info=True)
    saved = {}
    with safe_open(str(CHECKPOINT / "model.safetensors"), framework="pt") as handle:
        for key in handle.keys():
            if any(part in key for part in (".sidecar.", ".attn_residual.", ".mlp_residual.")):
                saved[key] = handle.get_tensor(key)
    # This checkpoint predates the fp32 pin, so it stores `sharpness` at 1.0 where the
    # module now stores `sharpness_delta` at 0. Same function, and the run that
    # produced it could not move either (see PROGRESS.md, 2026-09-10).
    old = "model.layers.1.sidecar.ple.sharpness"
    if old in saved:
        if not torch.equal(saved[old], torch.ones_like(saved[old])):
            raise ValueError("sharpness is not 1.0; the migration to a deviation is not identity")
        model.model.layers[1].sidecar.ple.sharpness_delta.data.zero_()
        saved.pop(old)
    actual = model.state_dict()
    for key, value in saved.items():
        if not torch.equal(actual[key].cpu(), value.to(actual[key].dtype)):
            raise ValueError(f"adapter tensor did not load exactly: {key}")
    del actual, saved
    shard_model(model, ["cuda:0", "cuda:1"])
    if dtype == "fp32":
        model.float()
    model.requires_grad_(False).eval()
    return model, info


@torch.no_grad()
def depth_profile(model, batch, layer_index, injected, positions):
    """Where in *this* model's stack does the borrowed vector actually belong?

    Flash-Next injects its PLE at its own layer 2 and the rest of its stack is trained
    around that. A different model's layer 1 is not the same place: the same depth
    index need not be the same representation. So this reports, at every layer, the
    RMS of the collapsed residual stream and the mean absolute cosine between the
    stream and the vector being injected.

    Neither is proof of a correspondence -- cosine to a residual stream is a weak
    signal and a matching RMS only says the scales are compatible -- but a vector that
    is orders of magnitude off the stream's scale, or orthogonal to it everywhere, is
    not going to be absorbed wherever it is put, and that is worth knowing before
    training anything.
    """
    from distillkit.models.qwen35_widened import collapse_residual

    indices = list(range(model.config.num_hidden_layers + 1))
    with AnchorTap(model, indices) as tap:
        model(**batch, use_cache=False, logits_to_keep=1, return_dict=True)
    states = tap.states()
    reference = injected[0, positions].float()
    reference = reference / reference.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    rows = []
    for index in indices:
        state = states[index]
        if state.ndim == 4:
            state = collapse_residual(state)
        stream = state[0, positions].float().to(reference.device)
        rows.append({
            "layer": index,
            "stream_rms": float(stream.pow(2).mean().sqrt()),
            "cosine": float((stream / stream.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                             * reference).sum(-1).abs().mean()),
        })
    return rows


@torch.no_grad()
def capture(model, batch, layer_index, anchor):
    """One bypassed forward that also hands back what the reader would have seen.

    The sidecar is neutralised *inside* itself rather than by removing it, so the
    stream and the table rows it would have consumed are captured on the trajectory
    the bypassed model actually takes -- which is the trajectory the baseline hidden
    state below comes from. Removing the module would give the same hidden state but
    no query at all; leaving it enabled would give a query from a different model.
    """
    sidecar = model.model.layers[layer_index].sidecar
    reader = sidecar.ple
    held = {}
    original = reader.forward

    def intercept(self, stream, features):
        held["query"] = stream.detach()
        held["features"] = features.detach()
        return stream

    reader.forward = MethodType(intercept, reader)
    try:
        with AnchorTap(model, [anchor]) as tap:
            model(**batch, use_cache=False, logits_to_keep=1, return_dict=True)
        held["baseline_hidden"] = tap.states()[anchor].detach()
    finally:
        reader.forward = original
    if "query" not in held:
        raise ValueError("the sidecar never ran; the capture would be of nothing")
    return held


def assistant_slice(tokenizer, ids, donor_length=None):
    """Predictor positions and targets for this document's assistant spans."""
    text = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    if encoded["input_ids"] != list(ids):
        raise ValueError("role tokenisation does not reproduce the cached ids")
    return prediction_positions(ids, role_spans(text, encoded["offset_mapping"]), donor_length)


def flatten(value, positions):
    """``[1, T, ...]`` -> ``[len(positions), prod(rest)]`` on the CPU, in fp16.

    Six arms over the whole eval pool is about 1.3e9 numbers; fp32 would be 5 GiB held
    at once for no gain, since every one of these came out of a bf16 forward and fp16
    holds strictly more of a bf16 value than bf16 does.
    """
    rows = value[0, positions].float().cpu()
    return rows.reshape(rows.shape[0], -1).half()


def fit_examiner(examiner, rows, targets, head, vocab_size, steps, batch, lr, seed,
                 validation, emit, label, weight_decay=0.0):
    """Train only the correction, selecting on validation NLL, never on test.

    The budget is identical for every arm; that is what makes the comparison a
    comparison. Selection is on the validation split's mean NLL, and the returned
    weights are the best seen rather than the last, so an arm cannot be beaten by
    having overfit slightly faster than another.
    """
    device = getattr(head, "device", None) or head.weight.device
    examiner = examiner.to(device)
    project = getattr(head, "sharded_logits", None)
    head_dtype = next(head.parameters()).dtype
    optimizer = torch.optim.AdamW([examiner.correction.weight], lr=lr,
                                  weight_decay=weight_decay)
    generator = torch.Generator().manual_seed(seed)
    best = (float("inf"), examiner.correction.weight.detach().clone())

    def corrected(source, index):
        baseline, feature, target = (source[0][index], source[1][index], source[2][index])
        hidden = examiner(baseline.to(device).unsqueeze(0), feature.to(device).unsqueeze(0))
        return hidden.to(head_dtype), target.to(device).reshape(1, -1, 1)

    def nll(source, index):
        hidden, target = corrected(source, index)
        logits = project(hidden, vocab_size) if project is not None else head(hidden)
        if hasattr(logits, "sparse_logprobs"):
            return -logits.sparse_logprobs(target).squeeze(-1).squeeze(0)
        if vocab_size is not None and logits.shape[-1] > vocab_size:
            logits = logits[..., :vocab_size]
        return -logits.float().log_softmax(-1).gather(-1, target).squeeze(-1).squeeze(0)

    source = (rows[0], rows[1], targets)

    def validation_nll():
        with torch.no_grad():
            return float(torch.cat([
                nll(validation, torch.arange(start, min(start + 512, len(validation[2]))))
                for start in range(0, len(validation[2]), 512)
            ]).mean())

    # Step 0 is the zero initialisation, which returns the baseline unchanged. Without
    # it in the selection set the earliest candidate is already 25 steps at lr 3e-3, so
    # an arm whose best move is not to move has no way to say so.
    best = (validation_nll(), examiner.correction.weight.detach().clone())
    emit("fit", arm=label, step=0, train_nll=None, validation_nll=best[0])
    # Log-spaced, because these arms overfit within the first few dozen steps: a first
    # check at step 25 means the only candidates on offer are already past the point
    # where validation turned, and every arm then declines and reports exactly zero.
    schedule = {1, 2, 4, 8, 16, 32, 64, 128, 256, steps}
    schedule |= {step for step in range(0, steps + 1, max(1, steps // 12)) if step}
    for step in range(steps):
        index = torch.randint(0, len(targets), (batch,), generator=generator)
        loss = nll(source, index).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step + 1 in schedule:
            score = validation_nll()
            emit("fit", arm=label, step=step + 1, train_nll=float(loss.detach()),
                 validation_nll=score)
            if score < best[0]:
                best = (score, examiner.correction.weight.detach().clone())
    examiner.correction.weight.data.copy_(best[1])
    return examiner, best[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs", type=int, default=61,
                        help="documents to draw from the eval pool, split by a hash of the id")
    parser.add_argument("--depth-profile", action="store_true",
                        help="stream RMS and |cosine| to the injected vector, per layer")
    parser.add_argument("--identity-readout", action="store_true",
                        help="also ask each representation to name the token at its own position")
    parser.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    # 2560 * width trainable numbers against a few thousand fit positions: at 256 the
    # smoke run reached train NLL 0.064 against validation 1.374, which grades the
    # examiner's memory rather than the representation's content.
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--output", default="scratch/reader-visibility/report.json")
    arguments = parser.parse_args()

    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    log = output.with_suffix(".jsonl").open("w", encoding="utf-8")
    started = time.monotonic()

    def emit(kind, **fields):
        row = json.dumps(dict(kind=kind, elapsed=round(time.monotonic() - started, 3), **fields),
                         default=str)
        log.write(row + "\n")
        log.flush()
        print(row, flush=True)

    torch.manual_seed(arguments.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    model, info = load_model(arguments.dtype)
    layer_index = model.config.sidecar_layer_index
    anchor = model.config.num_hidden_layers
    reader = model.model.layers[layer_index].sidecar.ple
    upstream = None
    if UPSTREAM_LAYER.exists():
        upstream = torch.load(UPSTREAM_LAYER, map_location="cpu",
                              weights_only=False)["value_proj.weight"]
        if upstream.shape != (model.config.hidden_size, reader.feature_dim):
            raise ValueError(f"upstream value_proj is {tuple(upstream.shape)}, expected "
                             f"{(model.config.hidden_size, reader.feature_dim)}")
    tokenizer = AutoTokenizer.from_pretrained("../student-hf")
    source = OfflineHiddenStateSignalSource("../teacher-cache-1m")
    raw_config = yaml.safe_load(Path(CONFIG_FILE).read_text(encoding="utf-8"))
    table = GGUFNGramTable(raw_config["sidecar"]["table_path"])
    collator = make_collator(tokenizer.pad_token_id, table)
    head = model.lm_head

    records = [r for r in source.cache.manifest["documents"] if r["split"] == "eval"]
    order = np.random.default_rng(arguments.seed).permutation(len(records))
    chosen = [records[index] for index in order[: arguments.docs]]
    texts = {}
    for record in chosen:
        ids = source.cache.read_document(record["doc_id"], include_hidden_states=False)
        texts[record["doc_id"]] = ids["input_ids"].tolist()
    # Whole buckets. The 60/20/20 hash split holds only in expectation, and rebalancing
    # it to hit an exact count is the one thing a stable split must not do.
    splits = split_records(
        [{"id": r["doc_id"], "text": json.dumps(texts[r["doc_id"]])} for r in chosen],
        {"fit": None, "validation": None, "test": None}, seed=arguments.seed % 1000)
    emit("setup", checkpoint=str(CHECKPOINT.resolve()), loading_info=info, dtype=arguments.dtype,
         width=arguments.width, steps=arguments.steps, batch=arguments.batch, lr=arguments.lr,
         counts={key: len(rows) for key, rows in splits.items()},
         documents={key: [row["id"] for row in rows] for key, rows in splits.items()},
         arms=ARMS, shuffled=SHUFFLED, torch_version=torch.__version__)

    # One forward per document, then everything else is arithmetic on what it captured.
    captured, profiles = {}, []
    for split, rows in splits.items():
        for row in rows:
            ids = texts[row["id"]]
            batch = collator([{"ids": ids}])
            batch = {key: value.to("cuda:0") if torch.is_tensor(value) else value
                     for key, value in batch.items()}
            positions, targets, total = assistant_slice(tokenizer, ids)
            if len(positions) == 0:
                # The eval pool holds a few records with no assistant span at all.
                # Nothing to grade, and a zero-row arm cannot be stacked.
                emit("skipped", doc_id=row["id"], split=split, length=len(ids),
                     reason="no assistant predictor positions")
                continue
            held = capture(model, batch, layer_index, anchor)
            parts = reader_representations(reader, held["query"], held["features"],
                                           upstream_value_weight=upstream)
            captured[row["id"]] = {
                "split": split, "length": len(ids), "assistant": total,
                "positions": positions, "targets": targets,
                # The token at the position itself, for the identity readout: the
                # question an embedding can actually be asked.
                "identity": torch.tensor(ids, dtype=torch.long)[positions],
                "baseline": flatten(held["baseline_hidden"], positions),
                "alignment": digest([row["id"], targets.tolist()]),
                **{arm: flatten(parts[arm], positions) for arm in ARMS},
            }
            if arguments.depth_profile and split == "test" and len(profiles) < 8:
                for which in ("upstream_value", "ungated_value"):
                    if which not in parts:
                        continue
                    value = parts[which]
                    value = value if value.ndim == 3 else value.mean(-2)
                    profiles.append({"doc_id": row["id"], "representation": which,
                                     "layers": depth_profile(model, batch, layer_index,
                                                             value, positions)})
            emit("captured", doc_id=row["id"], split=split, length=len(ids),
                 assistant_positions=int(len(positions)), assistant_total=int(total),
                 peak_memory=[torch.cuda.max_memory_allocated(i) for i in range(2)])

    def stack(split, arm, donors=None, target_key="targets", with_baseline=True):
        """[baseline, representation], targets, and the per-document row counts.

        ``with_baseline=False`` zeroes the residual the examiner corrects, which turns
        the same machinery into a linear probe from the representation alone.
        """
        baseline, feature, target, sizes, ids = [], [], [], [], []
        for key, row in captured.items():
            if row["split"] != split:
                continue
            take = len(row["positions"])
            other = captured[donors[key]] if donors else row
            if donors:
                # Truncate both sides to the donor's natural length: no fabricated row.
                take = min(take, len(other["positions"]))
                if take == 0:
                    raise ValueError(f"{key} has no shuffled counterpart positions")
            baseline.append((row["baseline"][:take] if with_baseline
                             else torch.zeros_like(row["baseline"][:take])).float())
            feature.append(other[arm][:take].float())
            target.append(row[target_key][:take])
            sizes.append(take)
            ids.append(key)
        return ((torch.cat(baseline), torch.cat(feature)), torch.cat(target), sizes, ids)

    def split_scores(hidden_rows, targets, sizes, ids, examiner=None):
        """Per-document token NLL lists, in the shape compare_scores wants.

        The alignment fingerprint is built from the targets this call actually scored.
        A shuffled arm is truncated to its donor's length, so a fingerprint taken from
        the whole document would claim two arms lined up when they did not -- which is
        exactly the check `compare_scores` exists to make.
        """
        device = getattr(head, "device", None) or head.weight.device
        if examiner is None:
            hidden = hidden_rows[0]
        else:
            with torch.no_grad():
                hidden = torch.cat([
                    examiner(hidden_rows[0][start:start + 512].to(device).unsqueeze(0),
                             hidden_rows[1][start:start + 512].to(device).unsqueeze(0))[0].cpu()
                    for start in range(0, len(targets), 512)])
        values = token_nll(hidden, head, targets, vocab_size=None)
        rows, at = [], 0
        for size, key in zip(sizes, ids):
            rows.append({"id": key, "nll": values[at:at + size],
                         "alignment": digest([key, targets[at:at + size].tolist()])})
            at += size
        return rows

    report = {"arms": {}, "counts": {key: len(rows) for key, rows in splits.items()}}
    for arm in ARMS:
        for control in ("real",) + (("shuffled",) if arm in SHUFFLED else ()):
            label = arm if control == "real" else f"{arm}/shuffled"
            donors = None
            if control == "shuffled":
                donors = {}
                for split in splits:
                    keys = [key for key, row in captured.items() if row["split"] == split]
                    donors.update(donor_map(keys, seed=arguments.seed % 1000))
            fit_rows, fit_targets, _, _ = stack("fit", arm, donors)
            val_rows, val_targets, _, _ = stack("validation", arm, donors)
            test_rows, test_targets, test_sizes, test_ids = stack("test", arm, donors)
            reference = split_scores(test_rows, test_targets, test_sizes, test_ids)
            baseline_nll = float(sum(sum(row["nll"]) for row in reference)
                                 / sum(len(row["nll"]) for row in reference))
            if "baseline_nll" not in report:
                report["baseline_nll"] = baseline_nll
                emit("baseline", test_documents=len(reference), nll=baseline_nll)
            statistics = [(fit_rows[1].mean(0), fit_rows[1].std(0).clamp_min(1e-4))]
            examiner = MatchedExaminer([fit_rows[1].shape[-1]], model.config.hidden_size,
                                       width=arguments.width, seed=arguments.seed % 1000,
                                       statistics=statistics)
            examiner, validation_nll = fit_examiner(
                examiner, fit_rows, fit_targets, head, None, arguments.steps,
                arguments.batch, arguments.lr, arguments.seed,
                (val_rows[0], val_rows[1], val_targets), emit, label,
                arguments.weight_decay)
            candidate = split_scores(test_rows, test_targets, test_sizes, test_ids, examiner)
            result = compare_scores(reference, candidate, seed=arguments.seed % 1000)
            result["validation_nll"] = validation_nll
            result["baseline_nll"] = baseline_nll
            result["input_dim"] = int(fit_rows[1].shape[-1])
            result["trainable"] = int(examiner.correction.weight.numel())
            report["arms"][label] = result
            emit("arm", arm=label, **result)

    if arguments.identity_readout:
        # Not a next-token question. With the baseline zeroed the examiner is a linear
        # map from the representation straight into the frozen head, asked to name the
        # token at its own position -- what a per-layer embedding is for. Reported as
        # NLL and top-1 against the uniform-over-vocabulary floor, with the shuffled
        # arm as the control that says whether it is reading this document at all.
        report["identity_readout"] = {}
        for arm in ("raw_table", "ungated_value", "upstream_value", "reader_update"):
            for control in ("real", "shuffled"):
                donors = None
                if control == "shuffled":
                    donors = {}
                    for split in splits:
                        keys = [key for key, row in captured.items() if row["split"] == split]
                        donors.update(donor_map(keys, seed=arguments.seed % 1000))
                label = arm if control == "real" else f"{arm}/shuffled"
                fit_rows, fit_targets, _, _ = stack("fit", arm, donors, "identity", False)
                val_rows, val_targets, _, _ = stack("validation", arm, donors, "identity", False)
                test_rows, test_targets, sizes, ids = stack("test", arm, donors, "identity", False)
                statistics = [(fit_rows[1].mean(0), fit_rows[1].std(0).clamp_min(1e-4))]
                examiner = MatchedExaminer([fit_rows[1].shape[-1]], model.config.hidden_size,
                                           width=arguments.width, seed=arguments.seed % 1000,
                                           statistics=statistics)
                examiner, validation_nll = fit_examiner(
                    examiner, fit_rows, fit_targets, head, None, arguments.steps,
                    arguments.batch, arguments.lr, arguments.seed,
                    (val_rows[0], val_rows[1], val_targets), emit, f"identity/{label}",
                    arguments.weight_decay)
                scored = split_scores(test_rows, test_targets, sizes, ids, examiner)
                values = [value for row in scored for value in row["nll"]]
                entry = {"nll": float(np.mean(values)), "validation_nll": validation_nll,
                         "uniform_floor": float(np.log(model.config.vocab_size)),
                         "tokens": len(values), "input_dim": int(fit_rows[1].shape[-1])}
                report["identity_readout"][label] = entry
                emit("identity", arm=label, **entry)

    if profiles:
        # Averaged over documents, per representation: one row per layer.
        summary = {}
        for entry in profiles:
            for row in entry["layers"]:
                key = (entry["representation"], row["layer"])
                bucket = summary.setdefault(key, {"stream_rms": [], "cosine": []})
                bucket["stream_rms"].append(row["stream_rms"])
                bucket["cosine"].append(row["cosine"])
        report["depth_profile"] = [
            {"representation": name, "layer": layer,
             "stream_rms": float(np.mean(values["stream_rms"])),
             "cosine": float(np.mean(values["cosine"]))}
            for (name, layer), values in sorted(summary.items())]
        report["depth_profile_documents"] = len({e["doc_id"] for e in profiles})
        for row in report["depth_profile"]:
            emit("depth", **row)

    write_json(output, report)
    emit("complete", output=str(output))
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

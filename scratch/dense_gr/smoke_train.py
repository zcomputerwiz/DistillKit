"""End-to-end smoke test: real tokens, remapped vocabulary, the measured configuration.

Everything so far has been benchmarked on random token ids, which exercises the kernels
but not the pipeline. This runs the smallest configuration on the actual Python token
store with the whole stack the substrate document recommends, and checks the things that
would quietly be wrong: that the vocabulary remap round-trips, that the loss falls, that
throughput matches what the benchmark promised, and that nothing spilled.

The remap is the design from `docs/dense_gr.md`: the original tokenizer stays the only
thing that touches text, a bijection carries kept ids to a compact space, and ids below
the cut decompose into their byte tokens rather than an UNK, so the mapping is lossless.
All 256 byte tokens and every special are kept regardless of frequency -- without the byte
tokens the fallback has holes and the "lossless" claim is false.

    CUDA_VISIBLE_DEVICES=0 python scratch/dense_gr/smoke_train.py
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.metadata as metadata
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

_original = metadata.version


def _version(name):
    try:
        return _original(name)
    except metadata.PackageNotFoundError:
        if name == "triton":
            return _original("triton-windows")
        raise


metadata.version = _version

import numpy as np  # noqa: E402
import torch  # noqa: E402

from benchmark import (ATTENTION_RATIOS, SpillWatch, apply_liger, build,  # noqa: E402
                       variant_tag)
from copy_probe import copy_probe, format_probe  # noqa: E402
from cut_cross_entropy import linear_cross_entropy  # noqa: E402
from distillkit.models import Qwen35WidenedForCausalLM  # noqa: E402
from distillkit.models.qwen35.csa2 import (dense_routing,  # noqa: E402
                                           routing_report, router_parameters)
from vocab_remap import (bytes_to_unicode, build_vocabulary,  # noqa: E402,F401
                         byte_token_ids, cached_remap)
from training_step import (KahanAdamW8bit, backward_step, check_optimizer,  # noqa: E402
                           optimizer_step, synchronize)
from training_state import (PlannedBatches, WindowBatches, read_training_state,  # noqa: E402
                            restore_training_state, save_training_state, take_step)

BASE = "D:/DeepThought/Projects/HybridModel/student-2b-hf"
STORE = Path("scratch/code_training/tokens-v2")
# What the corpus was tokenized at. A model this wide reads it without a remap.
FULL_VOCABULARY = 248_320
# What opens an assistant turn in the capture's chat template. Used only to find where
# a document stops being prompt; see --min-answer-tokens.
ANSWER_MARKER = "<|im_start|>assistant"


# The fields that decide whether a checkpoint's weights mean anything in this config.
# Everything else -- learning rate, token budget, evaluation cadence -- is free to differ
# between the run that wrote a checkpoint and the run that continues it.
_ARCHITECTURE_KEYS = (
    "vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers",
    "num_attention_heads", "num_key_value_heads", "head_dim", "layer_types",
    "linear_num_key_heads", "linear_num_value_heads", "linear_key_head_dim",
    "linear_value_head_dim", "residual_stream_routing", "residual_stream_num_branches",
    "residual_stream_lowrank", "residual_stream_blend", "mla_enabled", "csa2_enabled",
    "mla_latent_dim", "csa2_modes", "csa2_top_k", "csa2_block_size",
)

# Written by `recipient_initialize` and read back by the layer constructor, which uses
# them to mark a route as already converted. They say nothing about shape, so they are
# not guarded -- they are copied, because a run that dropped them would save a checkpoint
# that no longer knows it was converted and would let itself be converted a second time.
_PROVENANCE_KEYS = ("residual_stream_recipient_mode", "residual_stream_recipient_seed",
                    "residual_stream_recipient_epsilon")

# Settings that decide which tensors a checkpoint *has*, rather than how big they are.
# They are carried from the checkpoint rather than guarded, because the flags that build
# a run do not set them and a resume would otherwise rebuild the model with this
# version's defaults: a learned shared head vector replaced by a fresh projection, the
# hierarchy switched off, or a legacy key norm dropped -- none of which the shape check
# can see, since the shapes still line up.
_INHERITED_KEYS = ("csa2_token_head_weights", "csa2_rope_index", "csa2_candidate_layer",
                   "csa2_candidate_k", "mla_content_key_norm")


def rate_scale(step, warmup, progress, decay_fraction=0.0, floor=0.1):
    """Multiplier on each group's peak learning rate.

    Linear warm-up over `warmup` steps, then constant, then -- if `decay_fraction` is
    set -- linear decay to `floor` over that final fraction of the token budget. Decay is
    keyed to tokens scored, not steps, because a run's step count is only known once its
    batching plan exists and a resumed run restores its tokens, not a schedule.
    """
    scale = (step + 1) / warmup if step < warmup else 1.0
    if decay_fraction > 0:
        start = 1.0 - decay_fraction
        if progress > start:
            through = min(1.0, (progress - start) / decay_fraction)
            scale *= 1.0 - (1.0 - floor) * through
    return scale


def excluded_documents(path):
    """Document ids to exclude, from a bare list or from contamination_audit's records."""
    if path is None:
        return set()
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    if all(isinstance(record, str) for record in records):
        return set(records)
    unknown = {record.get("verdict") for record in records} - {
        "contamination", "equivalent", "not contamination"}
    if unknown:
        raise SystemExit("%s has verdicts this does not know how to treat: %s"
                         % (path, sorted(map(str, unknown))))
    return {record["doc_id"] for record in records
            if record["verdict"] in ("contamination", "equivalent")}


def _refuse_mismatched_architecture(source: Path, config) -> None:
    """Refuse to load a checkpoint whose shape disagrees with the requested config.

    `from_pretrained` with an explicit config reports missing and unexpected keys and
    carries on, which is right for a conversion that adds parameters and wrong for a
    typo in `--hidden`. The difference is that a conversion adds *known* modules; a
    mismatch here silently trains a partly random model and reports a loss for it.
    """
    stored = json.loads((source / "config.json").read_text(encoding="utf-8"))
    for key in _PROVENANCE_KEYS:
        if key in stored:
            setattr(config, key, stored[key])
    for key in _INHERITED_KEYS:
        if key in stored and stored[key] != getattr(config, key, None):
            print("init: %s = %r, taken from the checkpoint" % (key, stored[key]),
                  flush=True)
        if key in stored:
            setattr(config, key, stored[key])
    for key in _ARCHITECTURE_KEYS:
        want = getattr(config, key, None)
        have = stored.get(key, None)
        if isinstance(want, (list, tuple)) or isinstance(have, (list, tuple)):
            want, have = list(want or []), list(have or [])
        if want != have:
            raise SystemExit(
                "%s was trained with %s = %r, but this run asks for %r. Loading it "
                "would leave part of the model randomly initialized."
                % (source, key, have, want))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocab", type=int, default=16_384)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--ratio", default="1:1", choices=sorted(ATTENTION_RATIOS),
                        help="gated-delta-net layers per full-attention layer; "
                             "1:1 is what these arms have run, 3:1 is Qwen3-Next's")
    parser.add_argument("--blend", type=float, default=0.0,
                        help="gated residual route strength; 0 leaves it inert, "
                             "which is what every arm so far has run")
    parser.add_argument("--norm-mode", default="exact",
                        choices=("exact", "fast", "fused", "compiled"),
                        help="how the branch read normalises; exact is "
                             "bit-identical to the stock norm and what a "
                             "converted model needs, fused is fastest and "
                             "loses about one bf16 ulp")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--length", type=int, default=1024)
    parser.add_argument("--accumulate", type=int, default=1,
                        help="micro-batches per optimizer step; fixed-window sampling "
                             "keeps --batch rows per step, while cached documents use "
                             "--micro-batch or --micro-tokens per forward")
    parser.add_argument("--tokens", type=int, default=30_000_000,
                        help="scored tokens; one pass over the v1 store is 30.7M")
    parser.add_argument("--passes", type=float, default=None,
                        help="passes over the corpus, which overrides --tokens. This is "
                             "the fair budget across vocabularies: equal scored tokens "
                             "would give a small vocabulary less text for the same count")
    parser.add_argument("--evaluate-every", type=int, default=0,
                        help="steps between held-out evaluations; 0 disables")
    parser.add_argument("--evaluate-windows", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1.83e-6,
                        help="peak learning rate for the pretrained body. 1.83e-6 is the "
                             "low half of the flat optimum a paired sweep found for "
                             "distilling the converted 2B under compensated updates: "
                             "0.9e-6 to 3.65e-6 within noise, 7.3e-6 worse by 0.047 nats "
                             "with an interval clear of zero (TRAINING_REPAIR.md). A model "
                             "trained from scratch wants far more; pass it explicitly.")
    parser.add_argument("--decay-fraction", type=float, default=0.0,
                        help="decay every group's learning rate linearly over this final "
                             "fraction of the token budget, down to --decay-floor of its "
                             "peak. 0 keeps it constant after warm-up, as every run before "
                             "this did.")
    parser.add_argument("--decay-floor", type=float, default=0.1,
                        help="fraction of the peak the decay ends at")
    parser.add_argument("--router-lr", type=float, default=9.37e-4,
                        help="peak learning rate for the CSA2 router (the indexer "
                             "projections), 512x the body's. Swept at 1x, 8x, 64x and 512x "
                             "with the body fixed at 1.83e-6: the router's own objective on "
                             "held-out documents fell monotonically, -0.628 nats at 512x "
                             "[-0.657, -0.601] against 1x -- 2.6x the improvement 1x makes "
                             "over the untrained start -- with held-out NLL unmoved at every "
                             "rate. It is also about the 1e-3 the router was warmed at. "
                             "Pass --router-lr equal to --lr to train it with the body.")
    parser.add_argument("--adapter-lr", type=float, default=None,
                        help="peak learning rate for the residual-route adapters "
                             "(attn_residual / mlp_residual: W_down, W_up, W_write, "
                             "branch_gain_delta), which were created at conversion rather "
                             "than pretrained. Defaults to --lr: swept at 8x, 64x and 512x "
                             "it moved held-out NLL by nothing measurable, and at 512x it "
                             "set back the router's alignment by 0.18 nats.")
    parser.add_argument("--sparse-stage", action="store_true",
                        help="the reference's second stage: every parameter trains on the "
                             "language-modeling loss, the indexer trains on its own KL "
                             "against the attention over the tokens it selected, and the "
                             "two are cut apart so neither trains the other. Needs a "
                             "model whose indexer has been warmed up by indexer_kl.py; "
                             "from a cold one the selection is random and the KL is "
                             "fitting to noise.")
    parser.add_argument("--teacher-cache", type=Path, default=None, nargs="+",
                        help="distil against an offline top-k capture instead of reading "
                             "plain tokens from the store. One document per forward at "
                             "its own length, never padded: a padded batch changes CSA2's "
                             "routing wherever the indexer's scores tie at the cutoff. "
                             "The tail is carried rather than deleted, which is only "
                             "defined at the capture temperature. Several captures are "
                             "read as one corpus, after checking that they agree about "
                             "the tokenizer, the top-k width and the temperature.")
    parser.add_argument("--tensor-parallel", action="store_true",
                        help="split the body across both cards. The embedding and head "
                             "stay whole on home, because Cut Cross-Entropy never forms "
                             "the logits and needs them that way. Measure the current "
                             "training objective with tp_bench.py before sizing batches.")
    parser.add_argument("--embedding-on", choices=("home", "away"), default="home",
                        help="which card holds the tied embedding and head under "
                             "--tensor-parallel. They stay whole either way, because Cut "
                             "Cross-Entropy needs the whole head; away moves them off the "
                             "card that also carries the residual stream. Measured at the "
                             "scale run's shape (3072 tokens, sparse stage, Kahan): home "
                             "peaks 19.4 / 8.8 GiB per card.")
    parser.add_argument("--micro-tokens", type=int, default=0, metavar="N",
                        help="tokens per forward, which is the better unit than documents: "
                             "memory follows tokens and this corpus runs 134 to 1024 of "
                             "them per document, so a fixed document count makes the "
                             "widest batch decide whether the run fits. With a budget the "
                             "wide groups get fewer rows and the narrow ones more. "
                             "Overrides --micro-batch.")
    parser.add_argument("--micro-batch", type=int, default=1,
                        help="documents per forward. They are grouped by length so a "
                             "batch is uniform without padding -- CSA2 cannot represent "
                             "padding, and a padded row routes differently from the same "
                             "row alone. Grouping costs 0.2%% of the corpus at 6.")
    parser.add_argument("--teacher-weight", type=float, default=0.5,
                        help="how much of the blend is the teacher's distribution, the "
                             "rest being ground-truth cross entropy. Repairing the tail "
                             "alone was measured and changed nothing -- two thirds of the "
                             "harm sat inside the teacher's list -- and it was keeping "
                             "cross entropy that removed it. Ground truth is also the "
                             "only term that says anything about a control token the "
                             "teacher's top-64 never ranked.")
    parser.add_argument("--teacher-max-length", type=int, default=2048,
                        help="prefix cap on a cached document. The sparse stage records "
                             "attention, which forces CSA2's gathered path, whose "
                             "selection is quadratic in the sequence: a 4096-token "
                             "document runs 24 GiB out of memory. A prefix is free of "
                             "the context mismatch a mid-document window would carry, "
                             "because causal attention means the first n positions see "
                             "exactly what the teacher saw. 2048 keeps 89.3% of this "
                             "corpus and peaks within a gigabyte of the ceiling; 1024 "
                             "keeps 74.1% and leaves about three. Measured in "
                             "teacher_kl.py.")
    parser.add_argument("--exclude-documents", type=Path, default=None,
                        help="JSON list of cache document ids to leave out of training and "
                             "held-out alike. Either bare ids, or records with `doc_id` and "
                             "`verdict` as contamination_audit writes them, of which only "
                             "`contamination` and `equivalent` are excluded -- the list "
                             "also records flags that were checked and cleared. For "
                             "teacher-cache-5m: ../capture-data/run5m-contamination.json.")
    parser.add_argument("--min-answer-tokens", type=int, default=0,
                        help="drop cached documents with fewer than this many assistant "
                             "tokens left inside the prefix cap. Both objectives score "
                             "every position but the last, so a document whose framing "
                             "outruns the cap is trained entirely on framing. Measured "
                             "on teacher-cache-5m at a 1024 cap: 115 of 5,303 train "
                             "documents and 8 of 295 held-out ones, about 3.5%% of the "
                             "scored tokens; the tail is long-context documents whose "
                             "prompts reach 7,131 tokens. Zero keeps every document, "
                             "which is what every arm measured so far did.")
    parser.add_argument("--no-kahan", action="store_true",
                        help="stock AdamW8bit on the bf16 weights, which discards every "
                             "update under half an ulp. Only for reproducing runs made "
                             "before the compensated optimizer; at lr 7.3e-6 it freezes "
                             "every weight with |w| >= 0.002.")
    parser.add_argument("--kl-chunk", type=int, default=256,
                        help="positions per head projection in the teacher KL, as a row "
                             "budget at batch 1. A full row is 248320 wide. It does not "
                             "move the peak -- the chunked head frees each chunk before "
                             "the next, so the quadratic routing sets it instead.")
    parser.add_argument("--indexer-weight", type=float, default=1.0,
                        help="how much of the indexer's KL to add. The reference gives no "
                             "figure because its two losses reach disjoint parameters, "
                             "which is also true here, so this scales a gradient rather "
                             "than trading one objective against another.")
    parser.add_argument("--checkpoint-layers", action="store_true",
                        help="recompute each layer's forward during the backward pass "
                             "instead of holding its activations. Costs roughly a third "
                             "more compute and is what buys sequence length: at 4096 the "
                             "activations are 11.5 of 15.6 GiB, and they are what makes "
                             "8192 not fit.")
    parser.add_argument("--warmup", type=int, default=100,
                        help="steps to ramp the learning rate over from zero. It exists "
                             "for the converted case: a from-scratch run has nothing to "
                             "damage on its first step, and a converted one has a model "
                             "that was fitted exactly.")
    parser.add_argument("--report-every", type=int, default=25)
    parser.add_argument("--store", type=Path, default=STORE)
    parser.add_argument("--spill-every", type=float, default=30.0,
                        help="seconds between background spill checks; the counter\n"
                             "read costs 1.8 s, so it is polled off the training thread")
    parser.add_argument("--seed", type=int, default=0,
                        help="seeds model init and the data order. The probe is seeded "
                             "separately and identically for every run, so arms and "
                             "seeds are scored on the same sequences")
    parser.add_argument("--probe-every", type=int, default=0,
                        help="steps between copy probes; 0 disables")
    parser.add_argument("--probe-half", type=int, default=256,
                        help="tokens in the block that gets repeated")
    parser.add_argument("--probe-windows", type=int, default=64)
    parser.add_argument("--mla", action="store_true",
                        help="replace the full-attention layers' key/value side with a "
                             "compressed latent")
    parser.add_argument("--csa2", action="store_true",
                        help="route the latent attention through Full/Reindex/Reuse "
                             "modes; requires --mla")
    parser.add_argument("--csa2-modes", nargs="+",
                        default=["full", "reuse", "full", "reindex", "reuse"],
                        help="one mode per full-attention layer")
    parser.add_argument("--csa2-top-k", type=int, default=256)
    parser.add_argument("--csa2-local-window", type=int, default=128)
    parser.add_argument("--csa2-block-size", type=int, default=128)
    parser.add_argument("--mla-latent-dim", type=int, default=128)
    parser.add_argument("--checkpoints", type=Path,
                        default=Path("scratch/dense_gr/checkpoints-smoke"),
                        help="root for the end-of-run checkpoint")
    parser.add_argument("--no-checkpoint", action="store_true")
    parser.add_argument("--resume", type=Path, help="resume a complete training-state directory")
    parser.add_argument("--save-every", type=int, default=0,
                        help="save resumable state every N optimizer steps (0: final only)")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="additional hard limit on total optimizer steps")
    parser.add_argument("--benchmark-warmup-steps", type=int, default=0,
                        help="exclude this many real optimizer steps from step timing")
    parser.add_argument("--init-from", type=Path, default=None,
                        help="start from this checkpoint instead of a fresh "
                             "initialization. Its architecture must match the one the "
                             "other flags describe; a mismatch is refused rather than "
                             "loaded, because a partial load trains something nobody "
                             "asked for.")
    parser.add_argument("--dense-routing", action="store_true",
                        help="run the CSA2 layers with every causal block open and the "
                             "router's bias off the logits. This is DeepSeek's dense "
                             "stage: the backbone settles on its converted key/value "
                             "path before the indexer is asked to imitate what it "
                             "attends to. The indexer takes no gradient here, because "
                             "the bias that carries it is exactly what is switched off.")
    parser.add_argument("--inherit", action="store_true",
                        help="take the architecture from --init-from instead of building "
                             "one out of --hidden and --layers. The shape flags describe "
                             "the toy arms and cannot describe a real checkpoint, and the "
                             "guard that refuses a mismatch is exactly what stops one "
                             "loading. Its vocabulary comes with it, so a model that "
                             "already covers the store's is trained on the store as it "
                             "is, with no remap.")
    parser.add_argument("--train-only", default="all",
                        choices=["all", "converted", "no-embedding"],
                        help="which parameters carry optimizer state. A converted model "
                             "differs from its source in the attention layers and the "
                             "residual route and nowhere else, which is 6%% of this 2B. "
                             "The optimizer is 8-bit, so the whole of it is affordable; "
                             "this is about what the run is allowed to move, not only "
                             "about what it costs to hold.")
    parser.add_argument("--freeze-router", action="store_true",
                        help="hold the CSA2 indexer at its initialization. Routing still "
                             "happens and still reaches the attention logits; only the "
                             "learning of it stops. The control for whether learning the "
                             "routing is worth anything.")
    parser.add_argument("--asymmetric", action="store_true",
                        help="convert with mean-zero per-branch gain offsets. The read "
                             "is unchanged in exact arithmetic, but the branches no "
                             "longer start identical -- which is what the symmetric "
                             "conversion leaves for training to undo.")
    parser.add_argument("--convert", action="store_true",
                        help="run recipient_initialize on the loaded checkpoint, which "
                             "puts it on the gated residual route at blend 1 computing "
                             "exactly what it computed before. Requires --init-from and "
                             "a recipient at blend 0.")
    parser.add_argument("--output", type=Path,
                        default=Path("scratch/dense_gr/smoke-train.json"))
    argv = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(argv)
    resume_state = read_training_state(args.resume) if args.resume else None
    if resume_state is not None:
        explicit = {arg.split("=")[0] for arg in argv if arg.startswith("--")}
        mutable = {"resume", "output", "checkpoints", "tokens", "passes", "max_steps", "save_every"}
        for action in parser._actions:
            key = action.dest
            if key not in resume_state["run_args"]:
                continue
            value = resume_state["run_args"][key]
            if action.type is Path and value is not None:
                value = [Path(v) for v in value] if isinstance(value, list) else Path(value)
            supplied = any(option in explicit for option in action.option_strings)
            if supplied and key not in mutable and getattr(args, key) != value:
                raise SystemExit("resume cannot change --" + key.replace("_", "-"))
            if not supplied:
                setattr(args, key, value)
        if "--tokens" in explicit and "--passes" not in explicit:
            args.passes = None
        if "--output" not in explicit or "--checkpoints" not in explicit:
            raise SystemExit("resume needs fresh --output and --checkpoints paths")
    if (args.tokens < 1 or args.length < 2 or args.micro_batch < 1 or args.micro_tokens < 0
            or args.save_every < 0 or args.report_every < 1 or args.evaluate_windows < 1
            or args.benchmark_warmup_steps < 0
            or (args.max_steps is not None and args.max_steps < 1)
            or (args.passes is not None and args.passes <= 0)):
        raise SystemExit("budgets, lengths and reporting counts must be positive")
    if args.no_checkpoint and args.save_every:
        raise SystemExit("--no-checkpoint conflicts with --save-every")
    inherited = None
    if args.inherit:
        if args.init_from is None:
            raise SystemExit("--inherit needs --init-from: there is nothing to inherit")
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

        inherited = Qwen3_5TextConfig.from_pretrained(args.init_from)
        # The flags stop describing a model to build and start describing the one being
        # loaded, so everything downstream that reads them -- the variant tag, the guards
        # on --dense-routing and --freeze-router, the record the run writes -- is made to
        # agree with the checkpoint rather than with this run's defaults.
        args.vocab = inherited.vocab_size
        args.hidden = inherited.hidden_size
        args.layers = inherited.num_hidden_layers
        args.mla = bool(getattr(inherited, "mla_enabled", False))
        args.csa2 = bool(getattr(inherited, "csa2_enabled", False))
        args.mla_latent_dim = getattr(inherited, "mla_latent_dim", args.mla_latent_dim)
        args.csa2_modes = list(getattr(inherited, "csa2_modes", None) or args.csa2_modes)
        args.csa2_top_k = getattr(inherited, "csa2_top_k", args.csa2_top_k)
        args.csa2_local_window = getattr(inherited, "csa2_local_window",
                                         args.csa2_local_window)
        args.csa2_block_size = getattr(inherited, "csa2_block_size", args.csa2_block_size)
        args.blend = float(getattr(inherited, "residual_stream_blend", args.blend))
    variant = variant_tag(args.ratio, args.blend, args.norm_mode, args.seed)
    if args.mla:
        variant += "-csa2" if args.csa2 else "-mla"
    if args.freeze_router:
        variant += "-frozen"
    if args.dense_routing:
        if not args.csa2:
            raise SystemExit("--dense-routing needs --csa2: there is no routing to open")
        variant += "-dense"
    if args.convert:
        if args.init_from is None:
            raise SystemExit("--convert needs --init-from: there is nothing to convert")
        if args.blend != 0.0:
            # recipient_initialize sets blend to 1 itself. `--blend` describes the
            # checkpoint being loaded, and the recipient is the model at blend 0.
            raise SystemExit("--convert loads a recipient, which runs at blend 0; got "
                             "--blend %r" % args.blend)
        variant += "-converted-asym" if args.asymmetric else "-converted"
    if args.output == Path("scratch/dense_gr/smoke-train.json"):
        args.output = Path("scratch/dense_gr/smoke-train-%s.json" % variant)
    if args.output.exists():
        raise SystemExit("refusing to overwrite existing report: %s" % args.output)
    target = args.checkpoints / ("smoke-%s" % variant)
    if not args.no_checkpoint and target.exists():
        raise SystemExit("refusing to overwrite existing checkpoint: %s" % target)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.accumulate < 1 or args.batch < 1:
        raise SystemExit("--batch and --accumulate must be positive")
    if not args.teacher_cache and args.batch % args.accumulate:
        # Fixed-window sampling uses an integer row count. Cached-document batches
        # have their own plan and are weighted by their actual supervised targets.
        raise SystemExit("--batch %d is not divisible by --accumulate %d for fixed windows"
                         % (args.batch, args.accumulate))
    store = args.store

    torch.cuda.set_per_process_memory_fraction(0.90, 0)
    started = time.perf_counter()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE, local_files_only=True)

    raw = np.memmap(store / "train.bin", dtype=np.uint32, mode="r")
    print("store: %d tokens" % raw.shape[0], flush=True)

    if inherited is not None:
        print("inherit: %s, hidden %d, %d layers, vocabulary %d"
              % (args.init_from, inherited.hidden_size,
                 inherited.num_hidden_layers, inherited.vocab_size), flush=True)

    # A model whose vocabulary already covers the store reads the store as it is. The
    # remap exists so a toy with 16,384 embeddings can train on a corpus tokenized at
    # 248,320, and running it at full width would build a 12 GiB identity.
    if args.vocab >= FULL_VOCABULARY:
        kept = None
        stream = raw
        print("vocabulary: %d, the store's own; no remap" % args.vocab, flush=True)
    else:
        # Counting 3B ids takes a couple of minutes and the answer never changes, so the
        # corpus ships the ranking beside the tokens.
        cache = store / "train-counts.npy"
        if cache.exists():
            counts = np.load(cache)
            print("counts: loaded %s" % cache.name, flush=True)
        else:
            counts = np.bincount(np.asarray(raw, dtype=np.int64),
                                 minlength=FULL_VOCABULARY)
        kept, forward, bytes_ids = build_vocabulary(counts, tokenizer, args.vocab)
        coverage = float(counts[np.asarray(kept)].sum() / counts.sum())
        print("vocabulary: kept %d ids, %.4f%% coverage" % (len(kept), 100 * coverage),
              flush=True)

        stream, original_tokens, compact_tokens = cached_remap(
            store, "train", args.vocab, tokenizer, forward, bytes_ids, counts, kept)
    if kept is None:
        # The store is memory mapped and stays that way: 12 GiB of uint32 is not worth
        # resident, and the windows are random over the whole of it either way.
        print("stream: %d tokens, %.1f GiB mapped"
              % (stream.shape[0], stream.nbytes / 2 ** 30), flush=True)
    else:
        # Into RAM: the training loop draws random windows, and a cold page cache over a
        # multi-gigabyte file would pay for that at every step of the first pass.
        stream = np.asarray(stream)
        print("remap: %d tokens -> %d, inflation %.4f, %.1f GiB resident"
              % (original_tokens, compact_tokens, compact_tokens / original_tokens,
                 stream.nbytes / 2 ** 30), flush=True)

        # Round trip: the compact stream must decode to what the original ids decode to.
        inverse = np.asarray(kept, dtype=np.int64)
        sample = stream[:4096].astype(np.int64)
        round_tripped = tokenizer.decode(inverse[sample].tolist())
        reference = tokenizer.decode(np.asarray(raw[:4096]).tolist())
        matches = round_tripped[:2000] == reference[:2000]
        print("round trip on the first 4096 compact tokens: %s" % matches, flush=True)

    if inherited is not None:
        # The architecture flags describe a model this run would build. This run is not
        # building one, and letting them through would quietly resize what it loads --
        # `--mla-latent-dim` alone defaults to 128 over a checkpoint's 384, which arrives
        # as a shape mismatch on six layers rather than as an argument.
        config = inherited
        config._attn_implementation = "flash_attention_2"
        print("inherit: latent %s, modes %s, top_k %s, block %s, taken from the checkpoint"
              % (getattr(config, "mla_latent_dim", None),
                 getattr(config, "csa2_modes", None),
                 getattr(config, "csa2_top_k", None),
                 getattr(config, "csa2_block_size", None)), flush=True)
    else:
        config = build(args.hidden, args.layers, args.vocab, ratio=args.ratio,
                       blend=args.blend,
                       attn_implementation="flash_attention_2")
        if args.mla:
            config.mla_enabled = True
            config.mla_latent_dim = args.mla_latent_dim
        if args.csa2:
            config.csa2_enabled = True
            config.csa2_modes = list(args.csa2_modes)
            config.csa2_top_k = args.csa2_top_k
            config.csa2_local_window = args.csa2_local_window
            config.csa2_block_size = args.csa2_block_size
    config.residual_stream_norm_mode = args.norm_mode
    torch.manual_seed(args.seed)
    if args.init_from is None:
        model = Qwen35WidenedForCausalLM(config).to(device="cuda", dtype=torch.bfloat16)
    else:
        # Inheriting *is* the agreement: the config came out of that checkpoint, so there
        # is nothing for the guard to compare and nothing it could catch.
        if inherited is None:
            _refuse_mismatched_architecture(args.init_from, config)
        model = Qwen35WidenedForCausalLM.from_pretrained(
            args.init_from, config=config, dtype=torch.bfloat16).to(device="cuda")
        print("init: loaded %s" % args.init_from, flush=True)
        if args.convert:
            records = model.recipient_initialize(asymmetric=args.asymmetric)
            print("convert: %d sublayers, mode %s, blend %.1f"
                  % (len(records), records[0]["mode"],
                     float(model.config.residual_stream_blend)), flush=True)
    if args.train_only != "all":
        # A conversion changes the attention layers and the residual route and nothing
        # else, so those are the parameters that have something to correct. The rest is
        # the source's own and training it risks moving what the conversion preserved.
        def wanted(name):
            if args.train_only == "no-embedding":
                return "embed" not in name and "lm_head" not in name
            return "self_attn" in name or "residual" in name

        frozen = 0
        for name, parameter in model.named_parameters():
            if not wanted(name):
                parameter.requires_grad_(False)
                frozen += parameter.numel()
        trains = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("train-only %s: %d train, %d frozen, %.2f GiB of 8-bit optimizer state"
              % (args.train_only, trains, frozen, trains * 2 / 2 ** 30), flush=True)

    if args.freeze_router:
        if not args.csa2:
            raise SystemExit("--freeze-router needs --csa2: there is no router otherwise")
        held = router_parameters(model)
        if not held:
            raise SystemExit("--freeze-router found no indexer parameters to freeze")
        for _, parameter in held:
            parameter.requires_grad_(False)
        print("router: froze %d indexer tensors, %d parameters"
              % (len(held), sum(p.numel() for _, p in held)), flush=True)
    model.train()
    if args.checkpoint_layers:
        model.model.gradient_checkpointing = True
        if args.sparse_stage:
            # The routing layers cannot be recomputed while their attention is being
            # recorded as the indexer's target: the recompute does not reproduce what was
            # captured, and `torch.utils.checkpoint` compares metadata and refuses. The
            # linear-attention layers are unaffected, and they are where the memory is --
            # measured at micro-batch 6 they hold 10,654 MiB of the home card's 14,567
            # against the routing layers' 4,476.
            model.model.gradient_checkpointing_types = ("linear_attention",)
            print("checkpointing: recomputing the linear-attention layers "
                  "(the routing layers are being recorded and cannot be)", flush=True)
        else:
            print("checkpointing: recomputing every layer's forward in backward",
                  flush=True)
    swapped = apply_liger(model, config)
    parameters = sum(p.numel() for p in model.parameters())
    print("model: %.1fM parameters, liger %s" % (parameters / 1e6, json.dumps(swapped)),
          flush=True)

    import bitsandbytes as bnb
    # Built after `shard_model`, further down, and not here. Sharding *replaces* the
    # modules it splits, so every parameter of a sharded module becomes a new tensor and
    # the ones an optimizer was holding are orphans: still stepped, no longer part of the
    # model. Building it here trained the residual adapters, the indexer, `kv_a_proj` and
    # the embedding -- everything sharding left alone -- and silently left the MLPs, the
    # gated delta net, `q_proj`, `kv_b_proj` and `o_proj` bitwise identical to the
    # checkpoint they started from, while the loss fell and nothing raised.
    optimizer = None
    # Polled on a daemon thread: reading it inline costs 1.81 s per report and
    # leaves the GPU with nothing queued for all of it.
    spill = SpillWatch(interval=args.spill_every).start()

    # Held-out: the calibration split shares no repository with train, so this measures
    # generalization rather than how much of the corpus has been memorized -- which is
    # what training loss becomes once a budget spans several passes.
    evaluation = None
    if args.evaluate_every and args.teacher_cache is None:
        if kept is None:
            held_stream = np.memmap(store / "calibration.bin", dtype=np.uint32, mode="r")
            held_original = held_compact = held_stream.shape[0]
        else:
            held_stream, held_original, held_compact = cached_remap(
                store, "calibration", args.vocab, tokenizer, forward, bytes_ids, counts,
                kept)
        rng = np.random.default_rng(12345)
        starts = rng.integers(0, held_stream.shape[0] - args.length - 1,
                              size=args.evaluate_windows)
        evaluation = torch.from_numpy(
            np.stack([held_stream[s:s + args.length] for s in starts]).astype(np.int64))
        print("held-out: %d tokens -> %d, %d fixed windows"
              % (held_original, held_compact, args.evaluate_windows), flush=True)

    @torch.no_grad()
    def evaluate():
        model.eval()
        if teacher is not None:
            # The capture's own eval split, one document per forward for the same reason
            # training uses one: these documents are median 537 tokens and padding them
            # into a batch would route differently than scoring them alone. Token
            # weighted, so a long document counts for what it is rather than for one row.
            total, scored = 0.0, 0
            # A fixed subset rather than all of it: the whole eval split is 258K tokens
            # and would cost a minute of every checkpoint. The same documents every
            # time, so the series is comparable -- and spread across every capture, so
            # a merged corpus is not scored entirely on whichever one was named first.
            by_source = {}
            for source, doc_id in held_sample:
                record = held_teacher.read(doc_id)
                ids = record["input_ids"]
                state = model.model(input_ids=ids,
                                    attention_mask=torch.ones_like(ids),
                                    use_cache=False).last_hidden_state
                count = ids.shape[1] - 1
                head = model.lm_head.weight
                with torch.cuda.device(head.device):
                    part = float(linear_cross_entropy(state.to(head.device), head,
                                                      ids.to(head.device), shift=1,
                                                      reduction="mean")) * count
                total += part
                scored += count
                row = by_source.setdefault(source, [0.0, 0])
                row[0] += part
                row[1] += count
            model.train()
            # Per source as well as in aggregate: a merged corpus's aggregate moves when
            # the mixture moves, so the column that says whether the model improved on
            # code is the code column and not the total.
            evaluate.by_source = {name: total_nll / max(tokens, 1)
                                  for name, (total_nll, tokens) in by_source.items()}
            return total / max(scored, 1)
        total, scored = 0.0, 0
        for start in range(0, evaluation.shape[0], args.batch):
            chunk = evaluation[start:start + args.batch].to("cuda")
            state = model.model(input_ids=chunk,
                                attention_mask=torch.ones_like(chunk),
                                use_cache=False).last_hidden_state
            count = chunk.numel() - chunk.shape[0]
            head = model.lm_head.weight
            with torch.cuda.device(head.device):
                total += float(linear_cross_entropy(state.to(head.device), head,
                                                    chunk.to(head.device), shift=1,
                                                    reduction="mean")) * count
            scored += count
        model.train()
        return total / max(scored, 1)

    # Nats per *original* token, so vocabularies are comparable: a cut that expands more
    # tokens is charged for the expansion rather than rewarded with an easier softmax. A
    # model reading the store at its own width expands nothing, so the charge is 1.
    inflation = 1.0 if kept is None else compact_tokens / original_tokens

    teacher = None
    if args.teacher_cache is not None:
        from teacher_kl import CachedTeacher

        if kept is not None:
            raise SystemExit(
                "--teacher-cache carries the capture's own token ids; a remapped "
                "vocabulary would address a different space than the targets")
        # The capture's ids are the original vocabulary, which is what `tokenizer`
        # speaks -- the remapped case is refused above, so there is no second space
        # this marker could be in.
        marker = (tokenizer(ANSWER_MARKER, add_special_tokens=False)["input_ids"]
                  if args.min_answer_tokens > 0 else None)
        excluded = excluded_documents(args.exclude_documents)
        teacher = CachedTeacher(args.teacher_cache, "train", seed=args.seed,
                                max_length=args.teacher_max_length,
                                answer_marker=marker,
                                min_answer_tokens=args.min_answer_tokens,
                                exclude=excluded)
        held_teacher = CachedTeacher(args.teacher_cache, "eval", seed=args.seed,
                                     max_length=args.teacher_max_length,
                                     answer_marker=marker,
                                     min_answer_tokens=args.min_answer_tokens,
                                     exclude=excluded)
        if excluded:
            print("excluded as benchmark contamination: %d train, %d held-out, of %d listed"
                  % (teacher.excluded, held_teacher.excluded, len(excluded)), flush=True)
        print("teacher: %d documents, %d tokens at a %d cap, top-%d, grouped tail, "
              "blend %.2f" % (len(teacher), teacher.tokens, args.teacher_max_length,
                              teacher.top_k, args.teacher_weight), flush=True)
        print("held-out: %d documents, %d tokens from the capture's own eval split"
              % (len(held_teacher), held_teacher.tokens), flush=True)
        if args.evaluate_every and not len(held_teacher):
            raise SystemExit("no held-out documents survive evaluation filtering")
        held_sample = held_teacher.stratified(args.evaluate_windows) if len(held_teacher) else []
        spread = {}
        for source, _ in held_sample:
            spread[source] = spread.get(source, 0) + 1
        print("held-out sample: %d documents across %d captures (%s)"
              % (len(held_sample), len(spread),
                 ", ".join("%s %d" % (Path(name).name, count)
                           for name, count in sorted(spread.items()))), flush=True)
        if args.min_answer_tokens > 0:
            print("dropped as all prompt at the cap: %d train, %d held-out"
                  % (teacher.dropped_all_prompt, held_teacher.dropped_all_prompt),
                  flush=True)

    if args.tensor_parallel:
        from distillkit.parallel.checkpoint import consolidated_state_dict
        from distillkit.parallel.model import shard_model

        if torch.cuda.device_count() < 2:
            raise SystemExit("--tensor-parallel needs two CUDA devices")
        shard_model(model, ["cuda:0", "cuda:1"], shard_embeddings=False,
                    embedding_device="cuda:1" if args.embedding_on == "away" else None)
        print("tensor parallel: body split across two cards, head kept whole on %s"
              % ("cuda:1" if args.embedding_on == "away" else "home"), flush=True)

    # After sharding, so that the parameters handed over are the model's current ones.
    # Only what trains: a frozen parameter takes no gradient, and handing it to the
    # optimizer still buys it two state tensors it will never read.
    trainable = [p for p in model.parameters() if p.requires_grad]
    # Weights are bf16 and bitsandbytes rounds each update into them to nearest, so
    # an update under half an ulp is lost every step: at lr 7.3e-6 every weight with
    # |w| >= 0.002 was frozen, 87% of the model. Kahan compensation keeps what the
    # rounding drops; see KahanAdamW8bit. `--no-kahan` reproduces the old runs.
    optimizer_class = bnb.optim.AdamW8bit if args.no_kahan else KahanAdamW8bit
    # Three groups, because the three kinds of parameter did not start from the same
    # place. The body is pretrained and moves least; the router and the residual
    # adapters were created at conversion -- the router warmed at 1e-3 in float32 by
    # indexer_kl.py -- and may want another rate. Empty groups are dropped, and each
    # carries its own peak so warm-up scales it rather than overwriting it.
    router_ids = {id(p) for _, p in router_parameters(model)}
    adapter_ids = {id(p) for name, p in model.named_parameters() if "_residual." in name}
    rates = {"body": args.lr,
             "router": args.lr if args.router_lr is None else args.router_lr,
             "adapter": args.lr if args.adapter_lr is None else args.adapter_lr}
    members = {"body": [], "router": [], "adapter": []}
    for parameter in trainable:
        kind = ("router" if id(parameter) in router_ids
                else "adapter" if id(parameter) in adapter_ids else "body")
        members[kind].append(parameter)
    optimizer = optimizer_class(
        [{"params": members[kind], "lr": rates[kind], "peak_lr": rates[kind], "name": kind}
         for kind in ("body", "router", "adapter") if members[kind]],
        lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    print("learning rates: " + ", ".join(
        "%s %.3g (%.1fM)" % (group["name"], group["peak_lr"],
                             sum(p.numel() for p in group["params"]) / 1e6)
        for group in optimizer.param_groups), flush=True)
    # The failure this guards against is silent in every other way: the loss falls, no
    # tensor is missing, and only a diff against the starting checkpoint shows that most
    # of the model never moved. So the invariant is asserted rather than assumed.
    held = {id(p) for group in optimizer.param_groups for p in group["params"]}
    stranded = [name for name, p in model.named_parameters()
                if p.requires_grad and id(p) not in held]
    if stranded:
        raise SystemExit(
            "%d trainable parameters are not in the optimizer, so they would never be "
            "updated: %s" % (len(stranded), ", ".join(stranded[:6])))
    print("optimizer: %d tensors, %.1fM parameters"
          % (len(trainable), sum(p.numel() for p in trainable) / 1e6), flush=True)
    check_optimizer(model, optimizer)

    sparse_stage = None
    if args.sparse_stage:
        from indexer_kl import routing_layers

        sparse_stage = routing_layers(model)
        if not sparse_stage:
            raise SystemExit("--sparse-stage needs routing layers")
        if getattr(model.config, "csa2_router_bias", True):
            raise SystemExit("joint sparse training requires selection-only routing")

    if teacher is not None:
        # One-row and multi-row runs use the same independently fixed prefixes.
        block = getattr(model.config, "csa2_block_size", None)
        groups = teacher._groups(args.micro_batch, block, args.micro_tokens or None)
        if not groups:
            raise SystemExit("no training documents survive the sample plan")
        corpus_targets = teacher.planned_tokens(args.micro_batch, block, args.micro_tokens or None)
        batches = PlannedBatches(teacher, groups, args.seed)
        shapes = sorted({(len(g), w) for g, w in groups})
        print("plan: %d documents, %d supervised targets, %d batches, %d shapes; "
              "dropped %d short and %d without enough answer targets"
              % (sum(len(g) for g, _ in groups), corpus_targets, len(groups), len(shapes),
                 teacher.dropped_short, teacher.dropped_truncated_answer), flush=True)
    else:
        corpus_targets = len(stream) - 1
        batches = WindowBatches(stream, args.batch // args.accumulate, args.length, args.seed)
        shapes = [(args.batch // args.accumulate, args.length)]
    if args.passes is not None:
        args.tokens = int(args.passes * corpus_targets)
    if args.tokens < 1:
        raise SystemExit("budget contains no supervised targets")

    step_options = dict(teacher_weight=args.teacher_weight if teacher is not None else 0.0,
                        indexer_weight=args.indexer_weight, sparse_stage=sparse_stage,
                        kl_chunk=args.kl_chunk)
    warmup_backward_targets = 0
    # Warm the actual objective and recorded-attention path, not a CE-only surrogate.
    if args.tensor_parallel:
        examples = {}
        if teacher is not None:
            for group, width in groups:
                examples.setdefault((len(group), width), (group, width))
        print("warming %d objective shapes: %s" % (len(shapes), shapes), flush=True)
        with torch.autograd.set_multithreading_enabled(False):
            for rows, width in shapes:
                if teacher is not None:
                    group, width = examples[(rows, width)]
                    record = teacher.read_batch(group, width)
                else:
                    record = {"input_ids": torch.randint(1, model.config.vocab_size,
                                                         (rows, width), device="cuda")}
                with dense_routing(model) if args.dense_routing else contextlib.nullcontext():
                    warmed = backward_step(model, [record], **step_options)
                    warmup_backward_targets += warmed["targets"]
                model.zero_grad(set_to_none=True)
                del record
                synchronize(model)
                torch.cuda.empty_cache()
        print("warmed", flush=True)

    history, scored_tokens, step, previous_seconds = [], 0, 0, 0.0
    step_timings = []
    if resume_state is not None:
        progress = restore_training_state(resume_state, model, optimizer, batches)
        scored_tokens, step = progress["targets"], progress["steps"]
        history = progress["history"]
        previous_seconds = progress["seconds"]
        print("resumed at step %d, %d supervised targets" % (step, scored_tokens), flush=True)
        del resume_state
    if scored_tokens >= args.tokens or (args.max_steps is not None and step >= args.max_steps):
        raise SystemExit("resume budget already exhausted; increase the total target/step limit")
    run_args = json.loads(json.dumps(vars(args), default=str))
    synchronize(model)
    session_started = time.perf_counter()
    train_started = session_started - previous_seconds
    stage = dense_routing(model) if args.dense_routing else contextlib.nullcontext()
    stage.__enter__()
    while scored_tokens < args.tokens and (args.max_steps is None or step < args.max_steps):
        microbatches = take_step(batches, args.accumulate, args.tokens - scored_tokens)
        if not microbatches:
            break
        scale = rate_scale(step, args.warmup, scored_tokens / max(args.tokens, 1),
                           args.decay_fraction, args.decay_floor)
        for group in optimizer.param_groups:
            group["lr"] = group.get("peak_lr", args.lr) * scale
        if step == args.benchmark_warmup_steps:
            for device in {p.device for p in model.parameters() if p.is_cuda}:
                torch.cuda.reset_peak_memory_stats(device)
        synchronize(model)
        step_started = time.perf_counter()
        metrics = optimizer_step(model, optimizer, microbatches,
                                 tensor_parallel=args.tensor_parallel, **step_options)
        synchronize(model)
        step_seconds = time.perf_counter() - step_started
        if step >= args.benchmark_warmup_steps:
            step_timings.append({"seconds": step_seconds, "targets": metrics["targets"]})
        scored_tokens += metrics["targets"]
        loss, teacher_cost, indexer_cost = metrics["loss"], metrics["teacher_kl"], metrics["indexer"]
        finished = (args.tokens - scored_tokens < batches.next_targets()
                    or (args.max_steps is not None and step + 1 >= args.max_steps))
        del microbatches
        if spill.breached():
            # Set by the watcher thread the moment a poll exceeded the tolerance, so the
            # step this stops on is the first one after the breach rather than the next
            # multiple of report_every.
            args.output.write_text(json.dumps(
                {"aborted": "spilled to system RAM",
                 "shared_delta_gib": spill.tripped_at, 
                 "step": step, "history": history}, indent=2), encoding="utf-8")
            raise SystemExit("spilled %.2f GiB into system RAM at step %d; stopping"
                             % (spill.tripped_at, step))
        if step % args.report_every == 0 or finished:
            synchronize(model)
            elapsed = time.perf_counter() - train_started
            # Report supervised targets, never input-token estimates.
            corpus = corpus_targets
            seen = scored_tokens
            row = {"step": step, "tokens": seen, "passes": seen / corpus,
                   "loss": float(loss), "loss_per_original_token": float(loss) * inflation,
                   "tokens_per_second": seen / elapsed}
            if teacher is not None:
                row["teacher_kl"] = teacher_cost
            # Read before the probe and the held-out pass. Both run their own forward at
            # their own sequence length, and the layers keep only the last routing they
            # computed -- reading after them reports the probe's 512-token routing, where
            # every block is reachable and the density is trivially 1.00, instead of the
            # training batch's.
            routing = routing_report(model)
            if routing:
                row["routing"] = routing
            # `evaluation` is the dense_gr held-out and stays None under a teacher cache,
            # which evaluates on the capture's own eval split instead. Testing it alone
            # silently skipped every evaluation of the first two distillation arms.
            if ((evaluation is not None or teacher is not None)
                    and args.evaluate_every
                    and (step % args.evaluate_every == 0 or finished)):
                row["heldout"] = evaluate()
                row["heldout_per_original_token"] = row["heldout"] * inflation
                if getattr(evaluate, "by_source", None):
                    row["heldout_by_source"] = dict(evaluate.by_source)
                    if len(evaluate.by_source) > 1:
                        print("            held-out  " + "  ".join(
                            "%s %.4f" % (Path(name).name, value)
                            for name, value in sorted(evaluate.by_source.items())),
                            flush=True)
            if args.probe_every and (step % args.probe_every == 0 or finished):
                row["copy"] = copy_probe(model, args.vocab, half=args.probe_half,
                                         windows=args.probe_windows, batch=args.batch)
            history.append(row)
            if sparse_stage is not None:
                row["indexer"] = indexer_cost
            print("step %5d  %5.2f passes  train %7.4f%s%s  held %8s  norm %7.4f  %8.0f tok/s"
                  % (step, row["passes"], row["loss"],
                     "  kd %7.4f" % teacher_cost if teacher is not None else "",
                     "  idx %6.3f" % (indexer_cost / max(len(sparse_stage), 1))
                     if sparse_stage is not None else "",
                     "%.4f" % row["heldout"] if "heldout" in row else "-",
                     row.get("heldout_per_original_token",
                             row["loss_per_original_token"]),
                     row["tokens_per_second"]), flush=True)
            if "copy" in row:
                print("            %s" % format_probe(row["copy"]), flush=True)
            # Windows pages CUDA allocations out to system RAM over PCIe instead of
            # raising OOM, so a spilled run keeps reporting 100% GPU utilization while
            # throughput collapses. Stop on it rather than discovering it in the summary
            # of a run that has already burned hours.
            row.update(spill.report())
            # Loss alone cannot tell a router that learned to route from one that
            # collapsed onto the blocks the local window opens for free.
            if routing:
                print("            routing " + "  ".join(
                    "L%d %s d%.2f s%.2f h%.2f" % (r["layer"], r["mode"][:3],
                                                  r["density"], r["selected"],
                                                  r["entropy"]) for r in routing),
                      flush=True)

        step += 1
        if args.save_every and step % args.save_every == 0:
            save_training_state(args.checkpoints / ("state-step-%08d" % step),
                                model, optimizer, batches,
                                dict(steps=step, targets=scored_tokens, history=history,
                                     seconds=time.perf_counter() - train_started), run_args)
        if finished:
            break

    stage.__exit__(None, None, None)
    synchronize(model)
    steps = step
    if not history:
        raise SystemExit("budget is too small for the next complete microbatch")
    # Documents vary in length under a teacher cache, so the budget is what the loop
    # actually scored rather than the step count times an expected window.
    total_scored = scored_tokens
    elapsed = time.perf_counter() - train_started
    spilled = spill.stop()
    report = {
        "vocab": args.vocab,
        "kept_ids": args.vocab if kept is None else len(kept),
        "coverage": 1.0 if kept is None else coverage,
        "store": str(store), "seed": args.seed,
        "architecture": {"ratio": args.ratio,
                         "blend": float(model.config.residual_stream_blend),
                         "init_from": None if args.init_from is None
                         else str(args.init_from),
                         "converted": bool(args.convert),
                         "frozen_router": bool(args.freeze_router),
                         "dense_routing": bool(args.dense_routing),
                         "asymmetric": bool(args.asymmetric),
                         "variant": variant, "hidden": args.hidden,
                         "norm_mode": args.norm_mode,
                         "layers": args.layers,
                         "full_attention_layers": [
                             index for index, kind in enumerate(config.layer_types)
                             if "linear" not in str(kind)],
                         "mla": bool(getattr(config, "mla_enabled", False)),
                         "csa2": bool(getattr(config, "csa2_enabled", False)),
                         "csa2_modes": list(args.csa2_modes) if args.csa2 else None,
                         "csa2_top_k": args.csa2_top_k if args.csa2 else None,
                         "csa2_local_window": (args.csa2_local_window if args.csa2
                                               else None),
                         "csa2_block_size": args.csa2_block_size if args.csa2 else None},
        "store_tokens": int(stream.shape[0] if kept is None else original_tokens),
        "compact_tokens": int(stream.shape[0] if kept is None else compact_tokens),
        "round_trip": True if kept is None else bool(matches),
        "parameters": int(parameters), "liger": swapped,
        "batch": args.batch, "length": args.length, "steps": steps,
        "accumulate": args.accumulate,
        "scored_tokens": total_scored,
        "requested_targets": args.tokens, "unused_target_budget": args.tokens - total_scored,
        "corpus_scored_targets": corpus_targets,
        "step_measurements": step_timings,
        "warmup_backward_targets": warmup_backward_targets,
        "measured_step_tokens_per_second": (sum(r["targets"] for r in step_timings)
                                            / sum(r["seconds"] for r in step_timings)
                                            if step_timings else None),
        "peak_allocated_gib_by_device": {
            str(d): torch.cuda.max_memory_allocated(d) / 2 ** 30
            for d in {p.device for p in model.parameters() if p.is_cuda}},
        "seconds": elapsed, "tokens_per_second": total_scored / elapsed,
        "first_loss": history[0]["loss"], "final_loss": history[-1]["loss"],
        "inflation": inflation,
        "final_loss_per_original_token": history[-1]["loss_per_original_token"],
        "final_heldout": history[-1].get("heldout"),
        "final_heldout_per_original_token": history[-1].get(
            "heldout_per_original_token"),
        "final_copy": history[-1].get("copy"),
        "final_routing": history[-1].get("routing"),
        "passes": history[-1]["passes"],
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2 ** 30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2 ** 30,
        **spill.report(),
        "setup_seconds": session_started - started,
        "history": history,
        "run_args": run_args,
        "heldout_sample": held_sample if teacher is not None else None,
    }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if not args.no_checkpoint:
        # Same convention as copy_train: weights, tokenizer and a milestone that
        # records what produced them, so the checkpoint can answer questions the
        # report did not think to ask.
        target = args.checkpoints / ("smoke-%s" % variant)
        if target.exists() and any(target.iterdir()):
            # The identity above should make this unreachable; if it is reached, two runs
            # differ in something the name does not carry, and overwriting would destroy
            # the earlier one's weights rather than its report.
            raise SystemExit(
                "%s already holds a checkpoint; move it aside or pass --checkpoints, "
                "rather than overwriting an arm that is not this one" % target)
        target.mkdir(parents=True, exist_ok=True)
        # A sharded model's own state dict is the shards: `mlp.gate_proj.shards.0`
        # rather than `mlp.gate_proj.weight`. Saving that directly writes a checkpoint
        # nothing can load -- every key of the plain model reads as missing -- so the
        # shards are merged back into stock names first. `--tensor-parallel` is meant
        # to be invisible to everything downstream, and this is the one place where it
        # was not.
        state = consolidated_state_dict(model) if args.tensor_parallel else None
        model.save_pretrained(target, safe_serialization=True, state_dict=state)
        tokenizer.save_pretrained(target)
        save_training_state(target / "training-state", model, optimizer, batches,
                            dict(steps=steps, targets=scored_tokens, history=history,
                                 seconds=elapsed), run_args)
        (target / "milestone.json").write_text(json.dumps({
            "variant": variant, "seed": args.seed,
            "architecture": report["architecture"],
            "scored_tokens": total_scored, "optimizer_steps": steps,
            "final_loss": report["final_loss"],
            "final_heldout": report["final_heldout"],
            "final_copy": report["final_copy"],
            "final_routing": report["final_routing"],
            "tokens_per_second": report["tokens_per_second"],
            "peak_reserved_gib": report["peak_reserved_gib"],
        }, indent=2), encoding="utf-8")
        print("CHECKPOINT %s: %d tokens, %d steps, loss %.4f"
              % (target.name, total_scored, steps, report["final_loss"]),
              flush=True)
    spill_text = ("%+.2f GiB" % spilled if report["spill_telemetry_valid"]
                  else "unknown (incomplete telemetry)")
    print("\nloss %.4f -> %.4f over %d tokens at %.0f tok/s, peak %.2f GiB, spill %s"
          % (report["first_loss"], report["final_loss"], report["scored_tokens"],
             report["tokens_per_second"], report["peak_reserved_gib"], spill_text),
          flush=True)
    print("wrote %s" % args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

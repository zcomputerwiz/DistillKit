"""Turn a checkpoint saved from a sharded model back into a portable one.

`smoke_train.py --tensor-parallel` called `model.save_pretrained` on the sharded model,
whose own state dict is the shards: `mlp.gate_proj.shards.0` rather than
`mlp.gate_proj.weight`, `linear_attn.in_proj_qkv.0.weight` rather than
`linear_attn.in_proj_qkv.weight`. Nothing can load that -- every key of the plain model
reads as missing -- and the weights themselves are fine, so this merges them rather than
retraining.

`distillkit.parallel.checkpoint.consolidated_state_dict` already does the merge, and it
needs a live sharded model to do it: the gated-delta channel permutation is not
recoverable from the tensor names alone, because a rank's `in_proj_qkv` shard holds that
rank's Q, K and V interleaved rather than a contiguous slice, and only the module's plan
says where each went. So this rebuilds the same shape of model, loads the shards into it
by their own names, and asks for the consolidated form.

The source checkpoint is left alone; the merged one is written beside it.

    python scratch/dense_gr/merge_tp_checkpoint.py \
        --checkpoint scratch/dense_gr/checkpoints-2b-chat/smoke-r1-1-gr-s0-csa2 \
        --like scratch/dense_gr/checkpoints-2b/warmed-chat32 \
        --output scratch/dense_gr/checkpoints-2b-chat/portable
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import triton_shim  # noqa: F401,E402
import torch  # noqa: E402

from distillkit.models import Qwen35WidenedForCausalLM  # noqa: E402
from distillkit.parallel.checkpoint import consolidated_state_dict  # noqa: E402
from distillkit.parallel.model import shard_model  # noqa: E402

CARRY = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
         "generation_config.json", "milestone.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="the checkpoint holding sharded tensors")
    parser.add_argument("--like", type=Path, required=True,
                        help="a portable checkpoint of the same architecture, used to "
                             "build the model the shards are loaded into")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--devices", nargs="+", default=["cuda:0", "cuda:1"])
    args = parser.parse_args()

    from safetensors.torch import load_file

    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit("%s is not empty; merging into it would overwrite weights"
                         % args.output)

    model = Qwen35WidenedForCausalLM.from_pretrained(args.like, dtype=torch.bfloat16)
    shard_model(model, args.devices, shard_embeddings=False)
    native = set(model.state_dict())

    state = load_file(str(args.checkpoint / "model.safetensors"), device="cpu")
    missing, unexpected = sorted(native - set(state)), sorted(set(state) - native)
    if model.config.tie_word_embeddings and missing == ["lm_head.weight"]:
        # The tie means the head is not written; the plain loader rebuilds it the same
        # way, so this is the checkpoint being correct rather than short.
        missing = []
    if missing or unexpected:
        raise SystemExit("the checkpoint does not match this architecture: "
                         "missing=%s unexpected=%s" % (missing[:8], unexpected[:8]))
    model.load_state_dict(state, strict=False)
    print("loaded %d sharded tensors" % len(state), flush=True)

    merged = consolidated_state_dict(model)
    print("merged into %d portable tensors" % len(merged), flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output, safe_serialization=True, state_dict=merged)
    for name in CARRY:
        source = args.checkpoint / name
        if source.is_file():
            shutil.copy2(source, args.output / name)

    # Loading it back is the only check that means anything here: the failure this
    # fixes was a checkpoint that saved without complaint and would not load.
    reloaded = Qwen35WidenedForCausalLM.from_pretrained(args.output, dtype=torch.bfloat16)
    print("reloaded %s with %d parameters"
          % (args.output, sum(p.numel() for p in reloaded.parameters())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

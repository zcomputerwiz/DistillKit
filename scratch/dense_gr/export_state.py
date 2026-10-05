"""A portable checkpoint from a training state (`state-step-*`), without training further.

`--resume` refuses a budget it has already spent, and a state's model is the sharded one:
this rebuilds the run's layout (tensor parallel over two cards, the embedding where the
run kept it), loads the state's model tensors by their own names, and saves the
consolidated form, as merge_tp_checkpoint.py does for a sharded save. The tokenizer and
template come from `--like`, the run's own final checkpoint.

    python scratch/dense_gr/export_state.py --state scratch/dense_gr/checkpoints-2b-long-r5/state-step-00000100 \\
        --like scratch/dense_gr/checkpoints-2b-long-r5/smoke-r1-1-gr-s25-csa2 --output scratch/dense_gr/long5-step100
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

CARRY = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--like", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit("%s is not empty" % args.output)
    state = torch.load(args.state / "state.pt", map_location="cpu", mmap=True, weights_only=False)
    away = state["run_args"].get("embedding_on") == "away"
    if not state["run_args"].get("tensor_parallel"):
        raise SystemExit("not a tensor-parallel state; load its model dict directly")
    model = Qwen35WidenedForCausalLM.from_pretrained(args.like, dtype=torch.bfloat16)
    shard_model(model, ["cuda:0", "cuda:1"], shard_embeddings=False,
                embedding_device="cuda:1" if away else None)
    native, saved = set(model.state_dict()), set(state["model"])
    missing = sorted(native - saved - ({"lm_head.weight"} if model.config.tie_word_embeddings else set()))
    unexpected = sorted(saved - native)
    if missing or unexpected:
        raise SystemExit("the state does not match this layout: missing=%s unexpected=%s"
                         % (missing[:6], unexpected[:6]))
    model.load_state_dict(state["model"], strict=False)
    merged = consolidated_state_dict(model)
    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output, safe_serialization=True, state_dict=merged)
    for name in CARRY:
        if (args.like / name).is_file():
            shutil.copy2(args.like / name, args.output / name)
    (args.output / "milestone.json").write_text(
        '{"exported_from": "%s", "steps": %d, "targets": %d}\n'
        % (args.state.as_posix(), state["progress"]["steps"], state["progress"]["targets"]), encoding="utf-8")
    reloaded = Qwen35WidenedForCausalLM.from_pretrained(args.output, dtype=torch.bfloat16)
    print("exported step %d (%d targets) -> %s, %d parameters"
          % (state["progress"]["steps"], state["progress"]["targets"], args.output,
             sum(p.numel() for p in reloaded.parameters())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

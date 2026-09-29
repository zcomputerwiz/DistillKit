"""Tokenize preference pairs and attach the reference model's response log-probabilities.

DPO measures how far the policy has moved from a frozen reference on each side of a pair;
the reference never changes, so its log-probabilities are computed once here rather than
by holding a second 2B model in memory through training. Prompt and response are
tokenized separately and concatenated, as the rollouts were generated. Pairs longer than
--max-length are dropped.

    python scratch/dense_gr/ref_logprobs.py --reference <ckpt> --pairs ../capture-data/pairs-r7.jsonl \\
        --output ../capture-data/pairs-r7-ref.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import smoke_train  # noqa: E402,F401  (its triton-windows version shim, which Cut Cross-Entropy needs)
import torch  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite %s" % args.output)
    from cut_cross_entropy import linear_cross_entropy
    from transformers import AutoTokenizer

    from distillkit.models import Qwen35WidenedForCausalLM

    tok = AutoTokenizer.from_pretrained(args.reference)
    model = Qwen35WidenedForCausalLM.from_pretrained(args.reference, dtype=torch.bfloat16).cuda().eval()
    encode = lambda text: tok(text, add_special_tokens=False)["input_ids"]
    kept = dropped = 0
    with open(args.output, "w", encoding="utf-8") as out:
        for line in open(args.pairs, encoding="utf-8"):
            pair = json.loads(line)
            prompt = encode(pair["prompt"])
            row = {"pair_id": pair["pair_id"], "rejected_kind": pair["rejected_kind"],
                   "source": pair.get("source")}
            for side in ("chosen", "rejected"):
                ids = prompt + encode(pair[side])
                if len(ids) > args.max_length:
                    break
                tensor = torch.tensor([ids], device="cuda")
                with torch.no_grad():
                    hidden = model.model(input_ids=tensor, attention_mask=torch.ones_like(tensor),
                                         use_cache=False).last_hidden_state
                    logp = -float(linear_cross_entropy(hidden[:, len(prompt) - 1:-1], model.lm_head.weight,
                                                       tensor[:, len(prompt):], reduction="sum"))
                row.update({side + "_ids": ids, side + "_start": len(prompt), "ref_" + side: logp})
            else:
                out.write(json.dumps(row) + "\n")
                kept += 1
                continue
            dropped += 1
    print("%d pairs with reference log-probs, %d dropped as longer than %d -> %s"
          % (kept, dropped, args.max_length, args.output))


if __name__ == "__main__":
    main()

"""Student rollouts for on-policy distillation: prompts from the corpus, answers from the model.

Teacher forcing only ever shows the student clean text, so once its own generation slips it
is somewhere it never trained, and it loops. On-policy distillation trains on the student's
own trajectories instead -- slips and loops included -- scored by the teacher, whose
distribution at a looping position says to stop. This writes those trajectories; capture
them as usual and train with `--kl-only-caches`, never cross entropy, which would
reinforce the loops.

Prompts are training documents cut at their last assistant turn, so system prompts and
earlier turns stay; the model answers in thinking mode with Qwen's recommended sampling on
`CompiledGreedy`. Prompt and answer both fit `--width` and `--new`, so a rollout fits the
training cap whole.

    python scratch/dense_gr/onpolicy_rollouts.py --checkpoint <ckpt> \\
        --inputs ../capture-data/thinking-code-math.jsonl --count 3000 \\
        --output ../capture-data/onpolicy-r1.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "downstream" / "code_bench"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

from generate import CompiledGreedy  # noqa: E402

TURN = "<|im_start|>assistant\n"


def prompt_of(text):
    """Everything before the last assistant turn, plus a thinking-mode opening."""
    cut = text.rfind(TURN)
    return None if cut < 0 else text[:cut] + TURN + "<think>\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--exclude", type=Path, default=None, help="JSON list of doc ids to skip")
    parser.add_argument("--count", type=int, default=3000)
    parser.add_argument("--width", type=int, default=512, help="prompt tokens, left padded")
    parser.add_argument("--new", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shard", default="0/1", help="i/n: this process takes every n-th prompt")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite %s" % args.output)

    from transformers import AutoTokenizer

    from distillkit.models import Qwen35WidenedForCausalLM

    tok = AutoTokenizer.from_pretrained(args.checkpoint)
    tok.padding_side = "left"
    skip = set(json.load(open(args.exclude))) if args.exclude else set()
    pool = []
    for path in args.inputs:
        for line in open(path, encoding="utf-8"):
            row = json.loads(line)
            if row.get("split", "train") != "train" or row["doc_id"] in skip:
                continue
            # A prepared prompt (`rollout_prompts.py`) is used as it stands, with any
            # reference answer carried through for the classifier.
            prompt = row["prompt"] if "prompt" in row else prompt_of(row["text"])
            if prompt and len(tok(prompt, add_special_tokens=False)["input_ids"]) <= args.width:
                pool.append((row["doc_id"], prompt, row.get("source"), row.get("domain"),
                             row.get("reference")))
    random.Random(0).shuffle(pool)  # one order for every shard, so shards are disjoint
    index, shards = (int(x) for x in args.shard.split("/"))
    pool = pool[index::shards][:args.count]
    print("%d prompts (of those fitting %d tokens)" % (len(pool), args.width), flush=True)

    model = Qwen35WidenedForCausalLM.from_pretrained(args.checkpoint, dtype=torch.bfloat16).cuda().eval()
    model.config.use_cache = True
    eos = tok.eos_token_id
    runner = CompiledGreedy(model, args.batch_size, args.width, args.new, eos,
                            sampling=(0.6, 0.95, 20), seed=args.seed)
    started, written, tokens, finished_rows = time.monotonic(), 0, 0, 0
    with open(args.output, "w", encoding="utf-8") as out:
        for start in range(0, len(pool), args.batch_size):
            chunk = pool[start:start + args.batch_size]
            texts = [p for _, p, _, _, _ in chunk]
            filled = texts + [texts[0]] * (args.batch_size - len(texts))
            batch = tok(filled, return_tensors="pt", padding="max_length", max_length=args.width,
                        add_special_tokens=False).to("cuda")
            output = runner(batch["input_ids"], batch["attention_mask"])
            for offset, (doc_id, prompt, source, domain, reference) in enumerate(chunk):
                new = output[offset, args.width:]
                done = (new == eos).nonzero()
                end = int(done[0]) + 1 if done.numel() else int(new.numel())
                answer = tok.decode(new[:end], skip_special_tokens=False)
                finished_rows += bool(done.numel())
                tokens += end
                out.write(json.dumps({"doc_id": "onpolicy:%s:s%d" % (doc_id, args.seed),  # noqa: E501
                                      "text": prompt + answer, "split": "train", "source": source,
                                      "domain": domain, "finished": bool(done.numel()),
                                      "generated_tokens": end, "prompt_chars": len(prompt),
                                      "reference": reference},
                                     ensure_ascii=False) + "\n")
                written += 1
            print("%d/%d rollouts  %.0f s  %.0f tok/s  finished %.0f%%"
                  % (written, len(pool), time.monotonic() - started,
                     tokens / (time.monotonic() - started), 100 * finished_rows / written), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

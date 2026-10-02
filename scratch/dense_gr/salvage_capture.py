"""Recover the finished shards of an interrupted logits-only capture into a complete cache,
and write the documents it did not finish for a second capture.

The writer keeps its manifest in memory until the end, but it fills shards in input order
with a fixed rollover rule, so replaying that rule over the input names every document's
shard and offset. A shard is finished once a later one exists for its split; each salvaged
document's stored tokens are checked against the input before it is copied.

    python scratch/dense_gr/salvage_capture.py --input-jsonl ../capture-data/agent-smol.jsonl \\
        --partial ../teacher-cache-agent-smol --output ../teacher-cache-agent-smol-a \\
        --remaining ../capture-data/agent-smol-rest.jsonl --model ../teacher-hf --sequence-length 32768 \\
        --attn-implementation flash_attention_2 --prefill-chunk 8192 --weight-only-int8
"""
import argparse
import json
from pathlib import Path

import numpy as np

from distillkit.offline_cache import OfflineCacheWriter, _layouts, file_sha256, tokenizer_vocab_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--partial", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None, help="omit to only check and count")
    parser.add_argument("--remaining", type=Path, default=None)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, required=True)
    parser.add_argument("--shard-tokens", type=int, default=65536)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--prefill-chunk", type=int, default=None)
    parser.add_argument("--weight-only-int8", action="store_true")
    parser.add_argument("--logit-chunk-tokens", type=int, default=64, help="what the interrupted capture used")
    args = parser.parse_args()
    from transformers import AutoConfig, AutoTokenizer

    config = AutoConfig.from_pretrained(args.model)
    config = config.get_text_config() if hasattr(config, "get_text_config") else config
    layouts = _layouts(args.top_k, 0, config.hidden_size)

    # Replay the writer's placement: shard ids are global, in creation order.
    records, placed, shards, active = [], [], 0, {}
    for ordinal, line in enumerate(open(args.input_jsonl, encoding="utf-8")):
        record = json.loads(line)
        length = min(len(record["input_ids"]), args.sequence_length)
        split = record.get("split", "eval" if ordinal % args.eval_every == 0 else "train")
        shard = active.get(split)
        if shard is None or shard[1] + length > args.shard_tokens:
            shard = active[split] = [shards, 0]
            shards += 1
        placed.append((shard[0], shard[1], length, split))
        shard[1] += length
        records.append(record)
    on_disk = {int(p.name.split("-")[1].split(".")[0]): p.name.split("-")[0]
               for p in args.partial.glob("*.input_ids.bin")}
    last = {split: max(i for i, s in on_disk.items() if s == split) for split in set(on_disk.values())}
    finished = {i for i, s in on_disk.items() if i < last[s]}
    keep = [i for i, (shard, *_rest) in enumerate(placed) if shard in finished]
    print("%d documents; %d in %d finished shards (%d tokens); %d remaining"
          % (len(records), len(keep), len(finished), sum(placed[i][2] for i in keep), len(records) - len(keep)))

    def rows(i):
        shard, offset, length, split = placed[i]
        out = {}
        for name, (dtype, shape) in layouts.items():
            width = int(np.prod(shape, dtype=np.int64))
            data = np.fromfile(args.partial / f"{split}-{shard:05d}.{name}.bin", dtype=dtype,
                               count=length * width, offset=offset * width * dtype.itemsize)
            out[name] = data.reshape(length, *shape)
        if not np.array_equal(out["input_ids"], np.asarray(records[i]["input_ids"][:length], dtype=np.uint32)):
            raise ValueError("stored tokens differ from the input for %s" % records[i]["doc_id"])
        return out

    for i in keep:
        rows(i)
    print("all salvaged documents match their input tokens")
    if args.output is None:
        return
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    metadata = {"teacher_model_type": config.model_type, "teacher_config": config.to_dict(),
                "capture": "single_teacher_forced_forward_per_document",
                "eval_policy": {"explicit_split_takes_precedence": True, "fallback_eval_every": args.eval_every},
                "model": str(args.model), "revision": None, "int8": False,
                "attn_implementation": args.attn_implementation, "prefill_chunk": args.prefill_chunk,
                "weight_only_int8": args.weight_only_int8, "logit_chunk_tokens": args.logit_chunk_tokens,
                "input_jsonl_sha256": file_sha256(args.input_jsonl), "add_special_tokens": True,
                "salvaged_from": str(args.partial)}
    with OfflineCacheWriter(args.output, tokenizer_hash=file_sha256(args.model / "tokenizer.json"),
                            tokenizer_vocab_fingerprint=tokenizer_vocab_hash(tokenizer), anchor_layers=[],
                            hidden_size=config.hidden_size, vocab_size=config.vocab_size,
                            sequence_length=args.sequence_length, top_k=args.top_k,
                            shard_tokens=args.shard_tokens, metadata=metadata) as writer:
        for i in keep:
            data = rows(i)
            writer.append(records[i]["doc_id"], data["input_ids"], data["topk_ids"], data["topk_logprobs"],
                          split=placed[i][3], original_length=len(records[i]["input_ids"]))
    if args.remaining:
        kept = set(keep)
        with open(args.remaining, "w", encoding="utf-8") as out:
            for i, record in enumerate(records):
                if i not in kept:
                    out.write(json.dumps(record) + "\n")
    print("salvaged -> %s; remaining -> %s" % (args.output, args.remaining))


if __name__ == "__main__":
    main()

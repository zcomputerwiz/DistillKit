"""What batched decoding actually costs in memory, measured, to size batches and the cache.

For each batch size: weights resident, the prefill peak (and the logits it materializes),
what the cache holds after prefill, the decode peak over a fixed number of steps, and the
cache's growth per sequence per token. Greedy, eager, HF DynamicCache -- today's path.

    CUDA_VISIBLE_DEVICES=1 python scratch/downstream/code_bench/decode_memory.py <ckpt>
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "dense_gr"))

import smoke_train  # noqa: E402,F401
import torch  # noqa: E402

GIB = 2 ** 30


def main():
    path, prompt, steps = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 320, 64
    from distillkit.models import Qwen35WidenedForCausalLM

    torch.cuda.reset_peak_memory_stats()
    model = Qwen35WidenedForCausalLM.from_pretrained(path, dtype=torch.bfloat16).cuda().eval()
    model.config.use_cache = True
    weights = torch.cuda.memory_allocated() / GIB
    print("weights %.2f GiB (%.0fM parameters)" % (
        weights, sum(p.numel() for p in model.parameters()) / 1e6), flush=True)
    rows = []
    for batch in (16, 64, 128, 256):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        ids = torch.randint(1000, 100000, (batch, prompt), device="cuda")
        try:
            with torch.inference_mode():
                out = model(input_ids=ids, use_cache=True)
                prefill_peak = torch.cuda.max_memory_allocated() / GIB
                logits_gib = out.logits.numel() * out.logits.element_size() / GIB
                cache = out.past_key_values
                token = out.logits[:, -1].argmax(-1, keepdim=True)
                del out
                after_prefill = torch.cuda.memory_allocated() / GIB
                torch.cuda.reset_peak_memory_stats()
                for _ in range(steps):
                    out = model(input_ids=token, past_key_values=cache, use_cache=True)
                    cache = out.past_key_values
                    token = out.logits[:, -1].argmax(-1, keepdim=True)
                    del out
                decode_peak = torch.cuda.max_memory_allocated() / GIB
                after_decode = torch.cuda.memory_allocated() / GIB
            del cache, token
        except torch.cuda.OutOfMemoryError as error:
            print("batch %d: out of memory (%s)" % (batch, str(error).split(".")[0]))
            break
        per_token = (after_decode - after_prefill) * GIB / (batch * steps)
        state = (after_prefill - weights) * GIB / batch - per_token * prompt
        row = dict(batch=batch, prefill_peak=prefill_peak, logits=logits_gib,
                   after_prefill=after_prefill, decode_peak=decode_peak,
                   cache_bytes_per_token_per_seq=per_token, fixed_state_mib_per_seq=state / 2 ** 20)
        rows.append(row)
        print("batch %3d  prefill peak %5.2f GiB (logits %4.2f)  held after prefill %5.2f  "
              "decode peak %5.2f  cache %6.0f B/token/seq  fixed state %5.1f MiB/seq"
              % (batch, prefill_peak, logits_gib, after_prefill, decode_peak, per_token,
                 state / 2 ** 20), flush=True)
    Path(__file__).with_name("decode-memory.json").write_text(
        json.dumps(dict(checkpoint=path, prompt=prompt, steps=steps, weights_gib=weights,
                        rows=rows), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

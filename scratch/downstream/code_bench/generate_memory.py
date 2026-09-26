"""Peak memory and step time of HF `generate` at several batch sizes, dynamic vs static cache.

    CUDA_VISIBLE_DEVICES=1 python scratch/downstream/code_bench/generate_memory.py <ckpt> [static]
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "dense_gr"))

import smoke_train  # noqa: E402,F401
import torch  # noqa: E402

GIB = 2 ** 30


def main():
    path = sys.argv[1]
    static = len(sys.argv) > 2 and sys.argv[2] == "static"
    from distillkit.models import Qwen35WidenedForCausalLM

    model = Qwen35WidenedForCausalLM.from_pretrained(path, dtype=torch.bfloat16).cuda().eval()
    model.config.use_cache = True
    prompt, new = 320, 48
    for batch in (16, 64, 128, 256):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        ids = torch.randint(1000, 100000, (batch, prompt), device="cuda")
        options = dict(max_new_tokens=new, min_new_tokens=new, do_sample=False,
                       temperature=None, top_p=None, top_k=None, pad_token_id=0)
        if static:
            options["cache_implementation"] = "static"
        try:
            with torch.inference_mode():
                torch.cuda.synchronize()
                started = time.perf_counter()
                model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), **options)
                torch.cuda.synchronize()
                seconds = time.perf_counter() - started
        except torch.cuda.OutOfMemoryError:
            print("batch %3d: out of memory" % batch, flush=True)
            break
        print("batch %3d  peak %5.2f GiB  %.1f s  %.0f new tok/s"
              % (batch, torch.cuda.max_memory_allocated() / GIB, seconds,
                 batch * new / seconds), flush=True)


if __name__ == "__main__":
    main()

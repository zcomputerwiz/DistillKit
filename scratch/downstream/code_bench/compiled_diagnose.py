"""Where does CompiledGreedy diverge from HF generate, and what does a step cost?

Three decoders over the same 16 HumanEval+ prompts, 192 new tokens:
  hf        model.generate, dynamic cache, batch-max left padding
  loop      CompiledGreedy's loop with the step left eager (static cache, global padding)
  compiled  the same loop compiled with reduce-overhead
hf vs loop isolates the loop and the padding; loop vs compiled isolates compilation.
Reports the first differing token per row and the steady-state step time.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch  # noqa: E402

from generate import TEMPLATES, CompiledGreedy, render  # noqa: E402


def first_difference(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n if len(a) == len(b) else n


def main():
    from datasets import load_dataset
    from transformers import AutoTokenizer

    from distillkit.models import Qwen35WidenedForCausalLM

    path = sys.argv[1]
    tok = AutoTokenizer.from_pretrained(path)
    tok.padding_side = "left"
    model = Qwen35WidenedForCausalLM.from_pretrained(path, dtype=torch.bfloat16).cuda().eval()
    model.config.use_cache = True
    problems = load_dataset("evalplus/humanevalplus", split="test").select(range(16))
    prompts = [tok.apply_chat_template([{"role": "user", "content": render("humaneval", p)}],
                                       tokenize=False, add_generation_prompt=True)
               for p in problems]
    new, eos = 192, tok.eos_token_id
    out = {}

    batch = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
    with torch.inference_mode():
        g = model.generate(**batch, max_new_tokens=new, min_new_tokens=new, do_sample=False,
                           temperature=None, top_p=None, top_k=None, pad_token_id=eos)
    out["hf"] = g[:, batch["input_ids"].shape[1]:].tolist()

    width = -(-batch["input_ids"].shape[1] // 64) * 64
    fixed = tok(prompts, return_tensors="pt", padding="max_length", max_length=width,
                add_special_tokens=False).to("cuda")
    for mode in ("loop", "compiled"):
        runner = CompiledGreedy(model, len(prompts), width, new, eos, check=10 ** 9)
        if mode == "loop":
            runner.step = model
        runner(fixed["input_ids"], fixed["attention_mask"])  # warm-up / compile
        torch.cuda.synchronize()
        began = time.perf_counter()
        ids = runner(fixed["input_ids"], fixed["attention_mask"])
        torch.cuda.synchronize()
        seconds = time.perf_counter() - began
        out[mode] = ids[:, width:].tolist()
        print("%-9s %.1f ms per step (batch %d)" % (mode, 1000 * seconds / new, len(prompts)))

    for a, b in (("hf", "loop"), ("loop", "compiled"), ("hf", "compiled")):
        diffs = [first_difference(x, y) for x, y in zip(out[a], out[b])]
        same = sum(d >= new for d in diffs)
        print("%-4s vs %-8s identical %2d/%d   first difference at tokens %s"
              % (a, b, same, len(diffs), sorted(d for d in diffs if d < new)))


if __name__ == "__main__":
    main()

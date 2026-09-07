"""Greedy reference generation from the converted student-hf checkpoint (CPU only).

Mirrors scratch/llama_greedy_ref.txt: same prompt, temp 0, 64 new tokens.
Output is one token ID per line so it can be diffed against llama.cpp's output.
"""

import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

import torch

torch.cuda.is_available = lambda: False
assert not torch.cuda.is_available()

PROMPT = "The lighthouse keeper climbed the spiral stairs each morning before dawn, and"
N_NEW = 64


def main():
    from transformers import AutoTokenizer
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    path = r"D:\DeepThought\Projects\HybridModel\student-hf"
    tok = AutoTokenizer.from_pretrained(path)
    model = Qwen3_5ForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16).eval()

    ids = tok(PROMPT, add_special_tokens=False)["input_ids"]
    cur = torch.tensor([ids], dtype=torch.long)
    # model.generate threads the KV cache and position offsets correctly; a manual
    # loop that feeds only the latest token back loses the prompt each step.
    with torch.no_grad():
        out = model.generate(cur, max_new_tokens=N_NEW, do_sample=False, num_beams=1)
    generated = out[0][len(ids):].tolist()

    text = tok.decode(generated)
    with open(r"D:\DeepThought\Projects\HybridModel\DistillKit\scratch\hf_greedy_ref.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(str(i) for i in generated) + "\n\n")
        f.write(text + "\n")
    print("prompt tokens:", len(ids))
    print("generated:", text)


if __name__ == "__main__":
    main()

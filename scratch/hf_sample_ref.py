"""Sampled generation from the converted student-hf checkpoint (CPU only).

Complements hf_greedy_ref.py: pure greedy (temp 0) can lock into a repetition
loop on a small fine-tuned model, which says nothing about whether the weights are
sound. This runs the same prompt under sampling + a mild repetition penalty — the
regime the model would actually be used in — to confirm it produces varied,
grammatical continuation. No GPU: torch.cuda.is_available is pinned False.
"""

import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

import torch

torch.cuda.is_available = lambda: False
assert not torch.cuda.is_available()

PROMPT = "The lighthouse keeper climbed the spiral stairs each morning before dawn, and"
N_NEW = 128


def main():
    from transformers import AutoTokenizer
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    path = r"D:\DeepThought\Projects\HybridModel\student-hf"
    tok = AutoTokenizer.from_pretrained(path)
    model = Qwen3_5ForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16).eval()

    ids = tok(PROMPT, add_special_tokens=False)["input_ids"]
    cur = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        out = model.generate(
            cur, max_new_tokens=N_NEW, do_sample=True, temperature=0.7, top_p=0.9,
            repetition_penalty=1.15, num_beams=1,
        )
    new_ids = out[0][len(ids):].tolist()
    text = tok.decode(new_ids, skip_special_tokens=True)
    with open(r"D:\DeepThought\Projects\HybridModel\DistillKit\scratch\hf_sample_ref.txt", "w", encoding="utf-8") as f:
        f.write(f"prompt: {PROMPT}\n\n")
        f.write(text + "\n")
    print("prompt tokens:", len(ids))
    print("generated:", text)


if __name__ == "__main__":
    main()

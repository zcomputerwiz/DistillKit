"""Are the divergences near-ties or a bug? Compare next-token logits along one sequence.

Teacher-forces each row's HF-generated continuation through three paths -- eager dynamic
cache at batch-max padding (what generate did), eager static cache at the global padding,
and the compiled static step -- and at every position compares the logits: the largest
absolute difference, and at positions where the argmax disagrees, the reference's gap
between its top two logits. Near-ties flip on rounding; a bug moves logits by a lot.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch  # noqa: E402

from generate import render  # noqa: E402


def stepped(model, cache, prefix_ids, prefix_mask, continuation, width, compiled=None):
    """Logits for each continuation position via prefill then single-token steps."""
    step = compiled if compiled is not None else model
    batch = prefix_ids.shape[0]
    positions = torch.arange(width, device="cuda")
    with torch.inference_mode():
        out = model(input_ids=prefix_ids, attention_mask=prefix_mask, past_key_values=cache,
                    use_cache=True, cache_position=positions,
                    position_ids=positions.expand(batch, -1), logits_to_keep=1)
        logits = [out.logits[:, -1].float()]
        key_padding = torch.ones(batch, cache.max_cache_len if hasattr(cache, "max_cache_len")
                                 else width + continuation.shape[1], dtype=torch.bool, device="cuda")
        key_padding[:, :width] = prefix_mask.bool()
        token = torch.zeros(batch, 1, dtype=torch.long, device="cuda")
        position = torch.zeros(1, dtype=torch.long, device="cuda")
        position_ids = torch.zeros(batch, 1, dtype=torch.long, device="cuda")
        for index in range(continuation.shape[1] - 1):
            token.copy_(continuation[:, index:index + 1])
            position.fill_(width + index)
            position_ids.fill_(width + index)
            out = step(input_ids=token, attention_mask={"full_attention": None,
                                                         "linear_attention": None},
                       key_padding=key_padding, past_key_values=cache, use_cache=True,
                       cache_position=position, position_ids=position_ids, logits_to_keep=1)
            logits.append(out.logits[:, -1].float().clone())
    return torch.stack(logits, 1)


def main():
    from datasets import load_dataset
    from transformers import AutoTokenizer, StaticCache

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
    new, eos = 96, tok.eos_token_id
    batch = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
    with torch.inference_mode():
        g = model.generate(**batch, max_new_tokens=new, min_new_tokens=new, do_sample=False,
                           temperature=None, top_p=None, top_k=None, pad_token_id=eos,
                           output_scores=True, return_dict_in_generate=True)
    continuation = g.sequences[:, batch["input_ids"].shape[1]:]
    reference = torch.stack([s.float() for s in g.scores], 1)  # HF generate's own logits

    width = -(-batch["input_ids"].shape[1] // 64) * 64
    fixed = tok(prompts, return_tensors="pt", padding="max_length", max_length=width,
                add_special_tokens=False).to("cuda")
    runs = {}
    for name, compiled in (("static eager", None), ("static compiled", "yes")):
        cache = StaticCache(config=model.config, max_cache_len=width + new,
                            max_batch_size=len(prompts), device="cuda", dtype=torch.bfloat16)
        step = torch.compile(model, mode="reduce-overhead", fullgraph=False) if compiled else None
        if step is not None:  # warm the graph once on a throwaway cache
            warm = StaticCache(config=model.config, max_cache_len=width + new,
                               max_batch_size=len(prompts), device="cuda", dtype=torch.bfloat16)
            stepped(model, warm, fixed["input_ids"], fixed["attention_mask"], continuation[:, :4],
                    width, step)
        runs[name] = stepped(model, cache, fixed["input_ids"], fixed["attention_mask"],
                             continuation, width, step)

    top2 = reference.topk(2, -1).values
    gap = (top2[..., 0] - top2[..., 1])
    for name, logits in runs.items():
        diff = (logits - reference).abs().amax(-1)
        flips = logits.argmax(-1) != reference.argmax(-1)
        flip_gaps = gap[flips]
        print("%-16s max |dlogit| median %.3f  p99 %.3f  max %.3f   argmax flips %d/%d, "
              "reference top-2 gap at flips: max %.3f"
              % (name, diff.median(), diff.flatten().quantile(0.99), diff.max(),
                 int(flips.sum()), flips.numel(),
                 float(flip_gaps.max()) if flips.any() else 0.0))
    print("reference top-2 gap overall: median %.3f" % gap.median())


if __name__ == "__main__":
    main()

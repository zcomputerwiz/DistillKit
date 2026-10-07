"""Generate MBPP+ and HumanEval+ completions for any checkpoint. No code is executed here.

The same protocol `mbpp_plus/generate.py` used for its arms -- chat template, greedy,
left padding with EOS, a 768-token cap, first fenced block extracted by one rule -- made
independent of those arms so the source model and any trained checkpoint are compared
on identical prompts. `prompt_sha256` must match across checkpoints for a given task.

Execution happens elsewhere (see `run_docker.ps1`): the completions are untrusted model
output and this machine holds the checkpoints.

    python scratch/downstream/code_bench/generate.py --checkpoint ../student-2b-hf \\
        --bench mbpp --output scratch/downstream/code_bench/source-mbpp
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "dense_gr"))

import smoke_train  # noqa: E402,F401  (triton metadata shim)
import torch  # noqa: E402

FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)(?:```|\Z)", re.DOTALL)

# MBPP+ is `mbpp_plus/generate.py`'s template exactly, so its old stock arm stays comparable.
TEMPLATES = {
    "mbpp": ("Write a Python function for the following task. "
             "Respond with a single fenced Python code block and no explanation.\n\n"
             "{prompt}\n\nYour function must satisfy this test:\n{test}\n"),
    "humaneval": ("Complete the following Python function. Respond with a single fenced "
                  "Python code block containing the complete function, including its "
                  "imports and signature, and no explanation.\n\n```python\n{prompt}```\n"),
}
DATASETS = {"mbpp": "evalplus/mbppplus", "humaneval": "evalplus/humanevalplus"}


def pick(logits, sampling, generator):
    """Greedy when `sampling` is None, else Qwen's filtered sampling: temperature, then
    top-k, then top-p over what top-k kept, then one draw from `generator`."""
    if sampling is None:
        return logits.argmax(-1)
    temperature, top_p, top_k = sampling
    logits = logits.float() / temperature
    values, indices = logits.topk(top_k, dim=-1)
    probs = values.softmax(-1)
    cumulative = probs.cumsum(-1)
    probs = probs.masked_fill(cumulative - probs > top_p, 0.0)
    normalized = probs / probs.sum(-1, keepdim=True)
    choice = (torch.cat([torch.multinomial(p.unsqueeze(0), 1, generator=g)
                         for p,g in zip(normalized,generator)],0)
              if isinstance(generator,list) else torch.multinomial(normalized, 1, generator=generator))
    return indices.gather(-1, choice).squeeze(-1)


def stop_mask(tokens, eos):
    """Assisted-by: Codex. Scalar EOS keeps historical callers compatible."""
    if isinstance(eos,int):
        return tokens == eos
    ids = eos if isinstance(eos,torch.Tensor) else torch.tensor(list(eos),device=tokens.device,dtype=tokens.dtype)
    if not ids.numel():
        raise ValueError('at least one stop token is required')
    return (tokens.unsqueeze(-1) == ids).any(-1)


def extract(text: str) -> str:
    """The first fenced block, else the raw text. No repair: a completion that does not
    parse is a result."""
    match = FENCE.search(text)
    return (match.group(1) if match else text).strip()


def render(bench, problem):
    if bench == "mbpp":
        return TEMPLATES[bench].format(prompt=problem["prompt"], test=problem["test_list"][0])
    return TEMPLATES[bench].format(prompt=problem["prompt"])


class CompiledGreedy:
    """Greedy decoding behind CUDA graphs: one compile for a whole benchmark.

    The recipe `dense_gr/decode_compiled.py` measured at 107.6 -> 9.1 ms a token: a
    `StaticCache`, static input buffers written in place, an explicit `cache_position`,
    `torch.compile(mode="reduce-overhead")`. Made batched here, with every shape fixed --
    batch, left-padded prompt length and cache length are the same for every batch -- so
    there is one graph, not one per batch. Prefill runs eager. A decode step passes the
    padding as `key_padding` and no 2-D mask, so nothing inside it syncs with the host;
    finished rows are checked every `check` steps for the same reason.
    """

    def __init__(self, model, batch, prompt, new, eos, check=16, sampling=None, seed=0,
                 presence=0.0):
        from transformers import StaticCache

        self.sampling = sampling
        # vLLM-style presence penalty: a flat cut to the logit of every token the row has
        # already generated (prompt tokens are exempt). Qwen suggests 0-2 against loops.
        self.presence = presence
        self.generator = torch.Generator(device=next(model.parameters()).device).manual_seed(seed)

        self.model, self.batch, self.prompt, self.new, self.eos = model, batch, prompt, new, eos
        self.check, self.device = check, next(model.parameters()).device
        self.stop_tokens = eos if isinstance(eos,int) else torch.tensor(list(eos),device=self.device,dtype=torch.long)
        self.cache = StaticCache(config=model.config, max_cache_len=prompt + new,
                                 max_batch_size=batch, device=self.device, dtype=torch.bfloat16)
        self.step = torch.compile(model, mode="reduce-overhead", fullgraph=False)
        self.token = torch.zeros(batch, 1, dtype=torch.long, device=self.device)
        self.position = torch.zeros(1, dtype=torch.long, device=self.device)
        self.position_ids = torch.zeros(batch, 1, dtype=torch.long, device=self.device)
        self.key_padding = torch.ones(batch, prompt + new, dtype=torch.bool, device=self.device)
        self.no_mask = {"full_attention": None, "linear_attention": None}
        # The stock Qwen3.5 model has no `key_padding`: it takes the ordinary 2-D mask,
        # held at the static cache's full length so its shape never changes. Positions not
        # yet written lie in every query's future and the causal mask already hides them.
        self.widened = hasattr(getattr(model, "model", None), "csa2_bus")
        self.full_mask = torch.ones(batch, prompt + new, dtype=torch.long, device=self.device)

    @torch.inference_mode()
    def __call__(self, input_ids, attention_mask, seeds=None):
        batch, width = input_ids.shape
        assert (batch, width) == (self.batch, self.prompt), (batch, width)
        if seeds is not None and len(seeds) != batch:
            raise ValueError('one sampling seed per row is required')
        generator = (self.generator if seeds is None else
                     [torch.Generator(device=self.device).manual_seed(int(s)) for s in seeds])
        self.cache.reset()
        positions = torch.arange(width, device=self.device)
        out = self.model(input_ids=input_ids, attention_mask=attention_mask,
                         past_key_values=self.cache, use_cache=True, cache_position=positions,
                         position_ids=positions.expand(batch, -1), logits_to_keep=1)
        seen = torch.zeros(batch, out.logits.shape[-1], dtype=torch.bool, device=self.device)

        def choose(logits):
            if self.presence:
                logits = logits.float() - self.presence * seen
            chosen = pick(logits, self.sampling, generator).clone()
            seen.scatter_(1, chosen.view(-1, 1), True)
            return chosen

        following = choose(out.logits[:, -1])
        self.key_padding[:, :width].copy_(attention_mask.bool())
        self.full_mask[:, :width].copy_(attention_mask)
        masks = (dict(attention_mask=self.no_mask, key_padding=self.key_padding) if self.widened
                 else dict(attention_mask=self.full_mask))
        produced = [following]
        finished = stop_mask(following, self.stop_tokens)
        for index in range(self.new - 1):
            self.token.copy_(following.view(-1, 1))
            self.position.fill_(width + index)
            self.position_ids.fill_(width + index)
            out = self.step(input_ids=self.token, past_key_values=self.cache, use_cache=True,
                            cache_position=self.position, position_ids=self.position_ids,
                            logits_to_keep=1, **masks)
            following = choose(out.logits[:, -1])
            produced.append(following)
            finished |= stop_mask(following, self.stop_tokens)
            if (index + 1) % self.check == 0 and bool(finished.all()):
                break
        return torch.cat([input_ids, torch.stack(produced, 1)], dim=1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument('--eos-token-ids',type=int,nargs='+',
                        help='explicit serving stop policy; omitted reproduces historical tokenizer EOS')
    parser.add_argument('--case-seeds',action='store_true',
                        help='compiled sampler: independent RNG stream per task')
    parser.add_argument("--bench", choices=sorted(TEMPLATES), required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-thinking", action="store_true",
                        help="render with enable_thinking=False: the assistant turn opens on an "
                             "empty, closed think block, the format the code captures train on")
    parser.add_argument("--compiled", action="store_true",
                        help="CompiledGreedy instead of HF generate")
    parser.add_argument("--sample", action="store_true",
                        help="Qwen's recommended thinking-mode sampling instead of greedy")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--server", default=None,
                        help="a llama-server URL serving this checkpoint as GGUF; the tokenizer "
                             "still comes from --checkpoint, which renders the prompts")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.server and (args.eos_token_ids or args.case_seeds):
        parser.error('explicit stop tokens and per-case RNG apply to local generation; server policy must be configured separately')
    sampling = (args.temperature, args.top_p, args.top_k) if args.sample else None
    if (args.output / "completions.jsonl").exists():
        raise SystemExit("refusing to overwrite %s" % args.output)

    from datasets import load_dataset
    from transformers import AutoTokenizer

    config = json.loads((args.checkpoint / "config.json").read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    if config.get("csa2_enabled") or config.get("residual_stream_enabled"):
        from distillkit.models import Qwen35WidenedForCausalLM as Model
    else:
        from transformers import AutoModelForCausalLM as Model
    model = None
    if args.server is None:
        model = Model.from_pretrained(args.checkpoint, dtype=torch.bfloat16).to(args.device).eval()
        model.config.use_cache = True

    problems = load_dataset(DATASETS[args.bench], split="test")
    if args.limit:
        problems = problems.select(range(args.limit))
    tokenizer.padding_side = "left"
    end_of_text = tokenizer.eos_token_id
    stops = args.eos_token_ids or [end_of_text]
    if args.case_seeds and not args.compiled:
        raise ValueError('per-case RNG requires --compiled')
    prompts = [tokenizer.apply_chat_template(
        [{"role": "user", "content": render(args.bench, p)}], tokenize=False,
        add_generation_prompt=True, enable_thinking=not args.no_thinking) for p in problems]

    runner, width_all = None, None
    if args.compiled:
        # Every batch left-padded to one length and filled to one size: one graph.
        longest = max(len(tokenizer(p, add_special_tokens=False)["input_ids"]) for p in prompts)
        width_all = -(-longest // 64) * 64
        runner = CompiledGreedy(model, args.batch_size, width_all, args.max_new_tokens,
                                stops, sampling=sampling, seed=args.seed)
    torch.manual_seed(args.seed)

    started = time.monotonic()
    records, lengths, truncated = [], [], 0
    if args.server is not None:
        from llama_client import complete_all

        for (text, length, cut), problem, prompt in zip(
                complete_all(args.server, prompts, args.max_new_tokens, sampling, args.seed),
                problems, prompts):
            lengths.append(length)
            truncated += int(cut)
            records.append({"task_id": problem["task_id"],
                            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                            "generated_tokens": length, "truncated": bool(cut),
                            "raw": text, "code": extract(text)})
        prompts = []  # the batched loop below has nothing left to do
    for start in range(0, len(prompts), args.batch_size):
        chunk = prompts[start:start + args.batch_size]
        rows = problems.select(range(start, start + len(chunk)))
        if runner is None:
            tokens = tokenizer(chunk, return_tensors="pt", padding=True,
                               add_special_tokens=False).to(args.device)
            with torch.inference_mode():
                decoding = (dict(do_sample=True, temperature=args.temperature,
                                 top_p=args.top_p, top_k=args.top_k) if args.sample else
                            dict(do_sample=False, temperature=None, top_p=None, top_k=None))
                output = model.generate(**tokens, max_new_tokens=args.max_new_tokens,
                                        pad_token_id=end_of_text, eos_token_id=stops, **decoding)
        else:
            filled = chunk + [chunk[0]] * (args.batch_size - len(chunk))
            tokens = tokenizer(filled, return_tensors="pt", padding="max_length",
                               max_length=width_all, add_special_tokens=False).to(args.device)
            seeds=None
            if args.case_seeds:
                seeds=[int.from_bytes(hashlib.sha256(f"{r['task_id']}:{args.seed}".encode()).digest()[:8],'little')%(2**63-1) for r in rows]
                seeds += [seeds[0]]*(args.batch_size-len(seeds))
            output = runner(tokens["input_ids"], tokens["attention_mask"],seeds=seeds)
        width = tokens["input_ids"].shape[1]
        for offset, problem in enumerate(rows):
            new = output[offset, width:]
            finished = stop_mask(new,stops).nonzero()
            length = int(finished[0]) if finished.numel() else int(new.numel())
            completion = tokenizer.decode(new[:length], skip_special_tokens=True)
            lengths.append(length)
            is_truncated = not finished.numel() and length >= args.max_new_tokens
            truncated += int(is_truncated)
            records.append({
                "task_id": problem["task_id"],
                "prompt_sha256": hashlib.sha256(chunk[offset].encode("utf-8")).hexdigest(),
                "prompt_tokens": int(tokens["attention_mask"][offset].sum()),
                "generated_tokens": length, "truncated": bool(is_truncated),
                "token_ids": new[:length+int(bool(finished.numel()))].tolist(),
                "raw": completion, "code": extract(completion)})
            if start + offset < 3:
                records[-1]["rendered_prompt"] = chunk[offset]
        print("%s %d/%d  %.0f s" % (args.bench, len(records), len(prompts),
                                    time.monotonic() - started), flush=True)

    elapsed = time.monotonic() - started
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.json").write_text(json.dumps({
        "checkpoint": str(args.checkpoint), "bench": args.bench,
        "dataset": DATASETS[args.bench], "problems": len(records),
        "max_new_tokens": args.max_new_tokens, "batch_size": args.batch_size,
        "eos_token_ids": stops, "per_case_rng": args.case_seeds,
        "decoding": ({"do_sample": True, "temperature": args.temperature, "top_p": args.top_p,
                      "top_k": args.top_k, "seed": args.seed} if args.sample else
                     {"do_sample": False, "greedy": True}) | {"padding_side": "left",
                                                              "pad_token": "eos"},
        "instruction_template": TEMPLATES[args.bench], "thinking": not args.no_thinking,
        "mean_generated_tokens": sum(lengths) / max(len(lengths), 1),
        "median_generated_tokens": sorted(lengths)[len(lengths) // 2],
        "truncations": truncated, "elapsed_seconds": elapsed,
        "tokens_per_second": sum(lengths) / elapsed,
        "server": args.server}, indent=2), encoding="utf-8")
    with open(args.output / "completions.jsonl", "w", encoding="utf-8") as out:
        for record in records:
            out.write(json.dumps(record) + "\n")
    print("wrote %d completions to %s" % (len(records), args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

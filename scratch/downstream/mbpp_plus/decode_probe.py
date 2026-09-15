"""Prefill and decode timed separately, per arm, on one fixed prompt.

11 tok/s from a 2B model on a 3090 is low enough to be a harness bug, and the obvious
suspect is a leaked ``use_cache=False`` from the training configuration turning every
decode step into an O(T) recompute. This settles that by measuring it rather than
reasoning about it: prefill and decode are timed apart, the cache is inspected to confirm
it exists and grows by one per step, and all four arms are compared so a slow component
would show up as a slow arm.

    CUDA_VISIBLE_DEVICES=0 python scratch/downstream/mbpp_plus/decode_probe.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ffn_memo"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ffn_attenuation"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "residual_gate"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "structural_sidecar"))

import torch
from transformers import LogitsProcessorList

from distillkit.experimental.residual_gate import (
    TrigramFamiliarity, install_residual_gates, remove_residual_gates)
from distillkit.experimental.structural_sidecar import structural_token_ids
from evaluate import DEFAULT_CACHE, split_layout
from factorize import class_ids
from fit import BACKBONES, BASE
from gate_after_structure import build_structural
from generate import ARMS, GATE, StructuralBias

PROMPT = ("Write a Python function that returns the n largest numbers from a list, "
          "in descending order.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", default="B42", choices=sorted(BACKBONES))
    parser.add_argument("--new-tokens", type=int, default=128)
    parser.add_argument("--batches", type=int, nargs="*", default=[1, 32])
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path,
                        default=Path("scratch/downstream/mbpp_plus/decode_probe.json"))
    args = parser.parse_args()

    from transformers import AutoConfig, AutoTokenizer
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    from distillkit.independent_eval import build_token_classes

    checkpoint = BACKBONES[args.backbone]
    config = AutoConfig.from_pretrained(checkpoint, local_files_only=True)
    config = getattr(config, "text_config", config)
    config.use_cache = True
    model = Qwen3_5ForCausalLM.from_pretrained(
        checkpoint, config=config, dtype=torch.bfloat16,
        local_files_only=True).to(args.device).eval()
    model.requires_grad_(False)

    tokenizer = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
    classes = split_layout(build_token_classes(tokenizer, config.vocab_size), tokenizer,
                           config.vocab_size)
    structural = structural_token_ids(classes, device=args.device)
    whitespace = class_ids(classes, "whitespace", device=args.device)
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}], tokenize=False,
        add_generation_prompt=True)

    report = {"backbone": args.backbone, "checkpoint": str(checkpoint),
              "prompt": rendered, "new_tokens": args.new_tokens,
              "eval_mode": not model.training,
              "gradient_checkpointing": bool(getattr(model, "is_gradient_checkpointing",
                                                     False)),
              "config_use_cache": model.config.use_cache,
              "generation_config_use_cache": model.generation_config.use_cache,
              "attn_implementation": model.config._attn_implementation,
              "weight_dtype": str(next(model.parameters()).dtype),
              "weight_device": str(next(model.parameters()).device),
              "arms": {}}

    for arm, (use_gate, use_sidecar) in sorted(ARMS.items()):
        handle = None
        processors = LogitsProcessorList()
        if use_gate:
            payload = torch.load(GATE, map_location="cpu", weights_only=False)
            handle = install_residual_gates(
                model, payload["layers"], family=payload["family"],
                familiarity=TrigramFamiliarity(args.cache, config.vocab_size))
            model.residual_gates.load_state_dict(payload["state_dict"], strict=True)
            model.residual_gates.requires_grad_(False)
        if use_sidecar:
            sidecar, hasher, strength = build_structural(config, structural, whitespace,
                                                         args.device)
            processors.append(StructuralBias(sidecar, hasher, structural, strength))

        entry = {}
        for size in args.batches:
            tokens = tokenizer([rendered] * size, return_tensors="pt",
                               add_special_tokens=False).to(args.device)
            torch.cuda.reset_peak_memory_stats(args.device)
            with torch.inference_mode():
                if handle is not None:
                    handle.begin_generation()
                model.generate(**tokens, max_new_tokens=4, do_sample=False,
                               logits_processor=processors,
                               pad_token_id=tokenizer.eos_token_id)
                if handle is not None:
                    handle.end_generation()

                if handle is not None:
                    handle.begin_generation()
                torch.cuda.synchronize(args.device)
                start = time.perf_counter()
                prefilled = model(**tokens, use_cache=True)
                torch.cuda.synchronize(args.device)
                prefill = time.perf_counter() - start
                past = prefilled.past_key_values
                before = past.get_seq_length()
                step = tokens["input_ids"][:, -1:].clone()
                model(input_ids=step, past_key_values=past, use_cache=True)
                after = past.get_seq_length()
                if handle is not None:
                    handle.end_generation()

                if handle is not None:
                    handle.begin_generation()
                torch.cuda.synchronize(args.device)
                start = time.perf_counter()
                output = model.generate(**tokens, max_new_tokens=args.new_tokens,
                                        min_new_tokens=args.new_tokens, do_sample=False,
                                        logits_processor=processors,
                                        pad_token_id=tokenizer.eos_token_id)
                torch.cuda.synchronize(args.device)
                total = time.perf_counter() - start
                if handle is not None:
                    handle.end_generation()

            decode = total - prefill
            entry[str(size)] = {
                "batch": size,
                "prompt_tokens": int(tokens["input_ids"].shape[1]),
                "generated_tokens": int(output.shape[1] - tokens["input_ids"].shape[1]),
                "prefill_seconds": prefill,
                "decode_seconds": decode,
                "decode_tokens_per_second": size * args.new_tokens / decode,
                "milliseconds_per_step": 1000 * decode / args.new_tokens,
                "cache_present": past is not None,
                "cache_length_after_prefill": int(before),
                "cache_length_after_one_step": int(after),
                "cache_grows": int(after) == int(before) + 1,
                "peak_allocated_gib": torch.cuda.max_memory_allocated(args.device) / 2 ** 30,
            }
            row = entry[str(size)]
            print("%-8s batch %-3d prefill %6.3f s  decode %6.3f s  %7.1f tok/s  "
                  "%5.1f ms/step  cache %d->%d  peak %.1f GiB"
                  % (arm, size, row["prefill_seconds"], row["decode_seconds"],
                     row["decode_tokens_per_second"], row["milliseconds_per_step"],
                     row["cache_length_after_prefill"],
                     row["cache_length_after_one_step"], row["peak_allocated_gib"]),
                  flush=True)
        report["arms"][arm] = entry
        if handle is not None:
            remove_residual_gates(model)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nwrote %s" % args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

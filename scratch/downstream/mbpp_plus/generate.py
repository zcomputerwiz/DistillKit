"""Generate MBPP+ completions for the four frozen arms. No code is executed here.

Four configurations of the same frozen backbone -- stock, plus the corrected plain-CE
residual gate, plus the structural sidecar, plus both -- see identical problems in
identical order with identical prompts, template, decoding and stopping. The only variable
is which post-hoc module is enabled.

Two things make this different from the teacher-forced scoring everything else in this
programme used, and both are correctness requirements rather than conveniences:

    the gate     runs with a KV cache, so it keeps its own running history; without that
                 a one-token forward hashes against an empty prefix and addresses the
                 wrong rows while still returning a value
    the sidecar  is applied as a logits processor, which is handed the full sequence so
                 far, so its 3-gram addressing is computed over real history

Generation is batched because decoding here is CPU-launch-bound, not GPU-bound: a single
decode step issues roughly 380 tiny CUDA dispatches across the 18 GatedDeltaNet layers and
costs about 70 ms of wall clock for 10 ms of GPU work. Per-step cost is therefore almost
independent of batch size -- 69.5 ms at batch 1 against 72.2 ms at batch 32 -- so batching
buys 31x throughput and changes nothing about what is computed.

Padding is on the left and uses the EOS token rather than the pad token, which is not
cosmetic. The n-gram hasher resets its history at EOS, so an EOS-padded prefix addresses
exactly the rows the unpadded sequence would; padding with anything else would silently
hash real tokens against padding.

Nothing generated here is run. Raw completions are saved before extraction, extraction is
one deterministic rule applied identically to every arm, and the execution and analysis
are somebody else's job on a machine that can afford it.

    CUDA_VISIBLE_DEVICES=0 python scratch/downstream/mbpp_plus/generate.py --arm stock
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
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ffn_memo"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ffn_attenuation"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "residual_gate"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "structural_sidecar"))

import torch
from transformers import LogitsProcessor, LogitsProcessorList

from distillkit.experimental.ngram_hash import NGramHashConfig, NGramHasher
from distillkit.experimental.residual_gate import (
    TrigramFamiliarity, install_residual_gates, remove_residual_gates)
from distillkit.experimental.structural_sidecar import (
    apply_structural_bias, structural_token_ids)
from evaluate import DEFAULT_CACHE, split_layout
from factorize import class_ids
from fit import BACKBONES, BASE
from gate_after_structure import build_structural, digest_of

# The corrected plain-CE gate. NOT the historical assistant-masked gate, which the
# factorial showed never learned the familiarity policy at all.
GATE = Path("scratch/gate_regime/plain-harness/gate.pt")
ARMS = {"stock": (False, False), "gate": (True, False),
        "sidecar": (False, True), "both": (True, True)}
FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)(?:```|\Z)", re.DOTALL)

INSTRUCTION = (
    "Write a Python function for the following task. "
    "Respond with a single fenced Python code block and no explanation.\n\n"
    "{prompt}\n\n"
    "Your function must satisfy this test:\n{test}\n"
)


def sha256_of(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def extract(text: str) -> str:
    """One deterministic rule for every arm: the first fenced block, else the raw text.

    No repair, no per-arm special cases, no indentation fixing. A completion that does
    not parse is a result, not something to be tidied until it does.
    """
    match = FENCE.search(text)
    return (match.group(1) if match else text).strip()


class StructuralBias(LogitsProcessor):
    """The frozen structural sidecar, applied where the full history is available.

    ``generate`` hands a logits processor the entire sequence produced so far, which is
    exactly what the 3-gram addressing needs. Computing the rows inside the model forward
    instead would see only the newest token under a cache.
    """

    def __init__(self, sidecar, hasher, structural, strength):
        self.sidecar = sidecar
        self.hasher = hasher
        self.structural = structural
        self.strength = strength

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        rows = self.hasher.row_indices(input_ids)[:, -1:]
        bias = self.sidecar(rows, self.strength)[:, 0]
        return apply_structural_bias(scores.unsqueeze(1), bias.unsqueeze(1),
                                     self.structural, 1.0)[:, 0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=sorted(ARMS))
    parser.add_argument("--backbone", default="B42", choices=sorted(BACKBONES))
    parser.add_argument("--gate", type=Path, default=GATE)
    parser.add_argument("--whitespace", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32,
                        help="decoding is CPU-launch-bound, so this is nearly free")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if torch.cuda.device_count() != 1:
        raise SystemExit("mask to one card with CUDA_VISIBLE_DEVICES")

    from datasets import load_dataset
    from transformers import AutoConfig, AutoTokenizer
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    from distillkit.independent_eval import build_token_classes

    started = time.monotonic()
    use_gate, use_sidecar = ARMS[args.arm]
    checkpoint = BACKBONES[args.backbone]
    config = AutoConfig.from_pretrained(checkpoint, local_files_only=True)
    config = getattr(config, "text_config", config)
    config.use_cache = True
    model = Qwen3_5ForCausalLM.from_pretrained(
        checkpoint, config=config, dtype=torch.bfloat16,
        local_files_only=True).to(args.device).eval()
    model.requires_grad_(False)
    backbone_digest = digest_of(model)

    tokenizer = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
    classes = split_layout(build_token_classes(tokenizer, config.vocab_size), tokenizer,
                           config.vocab_size)
    structural = structural_token_ids(classes, device=args.device)
    whitespace = class_ids(classes, "whitespace", device=args.device)

    handle = None
    processors = LogitsProcessorList()
    manifest = {"arm": args.arm, "backbone": args.backbone,
                "checkpoint": str(checkpoint), "backbone_sha256": backbone_digest,
                "gate": None, "sidecar": None}
    if use_gate:
        payload = torch.load(args.gate, map_location="cpu", weights_only=False)
        handle = install_residual_gates(
            model, payload["layers"], family=payload["family"],
            familiarity=TrigramFamiliarity(args.cache, config.vocab_size))
        model.residual_gates.load_state_dict(payload["state_dict"], strict=True)
        model.residual_gates.requires_grad_(False)
        manifest["gate"] = {"path": str(args.gate), "sha256": sha256_of(args.gate),
                            "layers": payload["layers"],
                            "mask": payload.get("mask"),
                            "regime": payload.get("regime")}
    if use_sidecar:
        import gate_after_structure

        if args.whitespace is not None:
            gate_after_structure.WHITESPACE = args.whitespace
        sidecar, hasher, strength = build_structural(config, structural, whitespace,
                                                     args.device)
        processors.append(StructuralBias(sidecar, hasher, structural, strength))
        manifest["sidecar"] = {
            "addressed": str(gate_after_structure.ADDRESSED),
            "addressed_sha256": sha256_of(gate_after_structure.ADDRESSED),
            "whitespace": str(gate_after_structure.WHITESPACE),
            "whitespace_sha256": sha256_of(gate_after_structure.WHITESPACE),
            "strength": strength}

    problems = load_dataset("evalplus/mbppplus", split="test")
    if args.limit:
        problems = problems.select(range(args.limit))

    # Left padding with EOS: the hasher resets its history there, so a padded row
    # addresses the same n-gram rows as the unpadded sequence would.
    tokenizer.padding_side = "left"
    end_of_text = tokenizer.eos_token_id

    prompts = []
    for problem in problems:
        text = INSTRUCTION.format(prompt=problem["prompt"],
                                  test=problem["test_list"][0])
        prompts.append(tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=False,
            add_generation_prompt=True))

    records = []
    lengths = []
    truncated = 0
    for start in range(0, len(prompts), args.batch_size):
        chunk = prompts[start:start + args.batch_size]
        rows = problems.select(range(start, start + len(chunk)))
        tokens = tokenizer(chunk, return_tensors="pt", padding=True,
                           add_special_tokens=False).to(args.device)
        if handle is not None:
            handle.begin_generation()
        with torch.inference_mode():
            output = model.generate(
                **tokens, max_new_tokens=args.max_new_tokens, do_sample=False,
                temperature=None, top_p=None, top_k=None,
                logits_processor=processors, pad_token_id=end_of_text)
        if handle is not None:
            handle.end_generation()
        width = tokens["input_ids"].shape[1]
        for offset, problem in enumerate(rows):
            new = output[offset, width:]
            # generate pads finished rows with the pad id; the real length is up to the
            # first one, and a row that never stopped is a truncation.
            finished = (new == end_of_text).nonzero()
            length = int(finished[0]) if finished.numel() else int(new.numel())
            completion = tokenizer.decode(new[:length], skip_special_tokens=True)
            lengths.append(length)
            is_truncated = not finished.numel() and length >= args.max_new_tokens
            truncated += int(is_truncated)
            records.append({
                "task_id": problem["task_id"],
                "prompt_sha256": hashlib.sha256(
                    chunk[offset].encode("utf-8")).hexdigest(),
                "prompt_tokens": int(tokens["attention_mask"][offset].sum()),
                "generated_tokens": length,
                "truncated": bool(is_truncated),
                "raw": completion,
                "code": extract(completion),
            })
            if start + offset < 3:
                records[-1]["rendered_prompt"] = chunk[offset]
        print("%s %d/%d  %.0f s" % (args.arm, len(records), len(prompts),
                                    time.monotonic() - started), flush=True)

    if digest_of(model) != backbone_digest:
        raise SystemExit("the frozen backbone changed during generation")
    if handle is not None:
        remove_residual_gates(model)

    elapsed = time.monotonic() - started
    manifest.update({
        "problems": len(records),
        "max_new_tokens": args.max_new_tokens,
        "batch_size": args.batch_size,
        "decoding": {"do_sample": False, "greedy": True, "padding_side": "left",
                     "pad_token": "eos"},
        "instruction_template": INSTRUCTION,
        "mean_generated_tokens": sum(lengths) / max(len(lengths), 1),
        "median_generated_tokens": sorted(lengths)[len(lengths) // 2],
        "truncations": truncated,
        "elapsed_seconds": elapsed,
        "tokens_per_second": sum(lengths) / elapsed,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(args.device)),
    })
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2),
                                               encoding="utf-8")
    with open(args.output / "completions.jsonl", "w", encoding="utf-8") as handle_out:
        for record in records:
            handle_out.write(json.dumps(record) + "\n")
    print("%s: %d problems, %.1f tok/s, %d truncated, %.0f s"
          % (args.arm, len(records), manifest["tokens_per_second"], truncated, elapsed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

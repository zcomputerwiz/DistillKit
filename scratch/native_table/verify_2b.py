"""Four gates on the real 2B student before any native-table training.

    identity    rho = 0 reproduces the converted checkpoint, on logits and hidden states
    causal      rho != 0 changes the output, and wrong-context rows change it differently
    lookup      [batch, seq, 16] -> [batch, seq, 16, row] -> [batch, seq, hidden]
    gradient    what receives gradient at rho = 0, and what receives it once rho moves

The fourth is the one with a history: the two-stream pilot deadlocked because the write
and the admission were *both* zero-initialised, so neither factor of a product could ever
leave zero. This design puts the zero only in the admission, and the check below is what
proves the internals still train once it opens.

    python scratch/native_table/verify_2b.py --model ../student-2b-hf
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

DEFAULT_MODEL = "D:/DeepThought/Projects/HybridModel/student-2b-hf"


def native_config(model_path, base):
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    config = getattr(config, "text_config", config)
    config.sidecar_variant = "ple"
    config.sidecar_table_mode = "native"
    config.sidecar_ngram_vocab_size_base = base
    config.sidecar_layer_index = 1
    config.use_cache = False
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base", type=int, default=131072)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    from transformers import AutoTokenizer
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    from distillkit.models.qwen35_sidecar import Qwen35SidecarForCausalLM
    from distillkit.native_ple import native_hash_config
    from distillkit.ngram_hash import NGramHasher

    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    config = native_config(args.model, args.base)
    report = {"model": args.model, "hidden_size": config.hidden_size}

    stock = Qwen3_5ForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, local_files_only=True).to(args.device).eval()
    stock.config.use_cache = False
    morphed = Qwen35SidecarForCausalLM.from_pretrained(
        args.model, config=config, dtype=torch.bfloat16,
        local_files_only=True).to(args.device).eval()
    morphed.config.use_cache = False
    sidecar = morphed.model.layers[config.sidecar_layer_index].sidecar
    hasher = NGramHasher(native_hash_config(config))

    geometry = sidecar.geometry()
    report["geometry"] = {key: value for key, value in geometry.items()
                          if key not in ("head_vocab_sizes", "head_offsets")}
    report["geometry"]["head_vocab_sizes"] = geometry["head_vocab_sizes"]
    report["geometry"]["head_offsets"] = geometry["head_offsets"]
    report["geometry"]["table_bytes_bf16"] = geometry["table_parameters"] * 2
    assert config.hidden_size % 16 == 0
    assert geometry["head_dim"] * 16 == config.hidden_size

    texts = {
        "ordinary": "The capital of France is Paris, and the weather there is mild.",
        "repeated": "na " * 80,
        "long": "The quick brown fox jumps over the lazy dog. " * 40,
    }
    cases = {}
    for name, text in texts.items():
        ids = tokenizer(text)["input_ids"][:args.tokens]
        cases[name] = (ids, hasher.row_indices(torch.tensor([ids], dtype=torch.long)))

    # --- identity ------------------------------------------------------------
    report["identity"] = {}
    for name, (ids, rows) in cases.items():
        tensor = torch.tensor([ids], device=args.device)
        mask = torch.ones_like(tensor)
        with torch.no_grad():
            reference = stock(input_ids=tensor, attention_mask=mask,
                              output_hidden_states=True)
            native = morphed(input_ids=tensor, attention_mask=mask,
                             ngram_ids=rows.to(args.device), output_hidden_states=True)
        logit_difference = (native.logits.float() - reference.logits.float()).abs()
        hidden_difference = max(
            float((a.float() - b.float()).abs().max())
            for a, b in zip(native.hidden_states, reference.hidden_states))
        report["identity"][name] = {
            "tokens": len(ids),
            "bitwise_equal": bool(torch.equal(native.logits, reference.logits)),
            "max_logit_difference": float(logit_difference.max()),
            "mean_logit_difference": float(logit_difference.mean()),
            "max_hidden_difference": hidden_difference,
        }

    # --- causal --------------------------------------------------------------
    ids, rows = cases["ordinary"]
    tensor = torch.tensor([ids], device=args.device)
    rolled = hasher.row_indices(torch.tensor([ids], dtype=torch.long).roll(3, dims=-1))
    with torch.no_grad():
        dormant = morphed(input_ids=tensor, ngram_ids=rows.to(args.device)).logits
        sidecar.rho.fill_(0.5)
        admitted = morphed(input_ids=tensor, ngram_ids=rows.to(args.device)).logits
        wrong = morphed(input_ids=tensor, ngram_ids=rolled.to(args.device)).logits
        sidecar.rho.zero_()
        closed = morphed(input_ids=tensor, ngram_ids=rows.to(args.device)).logits
    report["causal"] = {
        "rho_open_changes_logits": float((admitted.float() - dormant.float()).abs().max()),
        "wrong_context_differs_from_right": float((wrong.float() - admitted.float()).abs().max()),
        "wrong_context_still_changes_output": float((wrong.float() - dormant.float()).abs().max()),
        "closing_rho_restores_exactly": bool(torch.equal(closed, dormant)),
        "rows_in_range": bool(int(rolled.min()) >= 0
                              and int(rolled.max()) < geometry["padded_vocab_size"]),
    }

    # --- lookup shapes -------------------------------------------------------
    hidden = torch.zeros(1, len(ids), config.hidden_size, device=args.device,
                         dtype=torch.bfloat16)
    features = sidecar.features(rows.to(args.device), hidden)
    report["lookup"] = {
        "ngram_ids": list(rows.shape),
        "rows_gathered": [*rows.shape, geometry["head_dim"]],
        "features": list(features.shape),
        "matches_hidden_size": features.shape[-1] == config.hidden_size,
    }

    # --- gradient topology ---------------------------------------------------
    morphed.requires_grad_(False)
    sidecar.requires_grad_(True)
    morphed.train()
    tracked = {"rho": sidecar.rho, "table": sidecar.table.weight,
               "key_proj": sidecar.ple.key_proj.weight,
               "value_proj": sidecar.ple.value_proj.weight,
               "conv1d": sidecar.ple.conv1d.weight,
               "norm_query": sidecar.ple.norm_query.weight,
               "norm_key": sidecar.ple.norm_key.weight,
               "norm_conv": sidecar.ple.norm_conv.weight}

    def gradients(label):
        morphed.zero_grad(set_to_none=True)
        logits = morphed(input_ids=tensor,
                         ngram_ids=rows.to(args.device)).logits[0, :-1].float()
        loss = torch.nn.functional.cross_entropy(
            logits, torch.as_tensor(ids[1:], device=logits.device))
        loss.backward()
        return {name: (0.0 if parameter.grad is None
                       else float(parameter.grad.detach().float().norm()))
                for name, parameter in tracked.items()}

    report["gradients"] = {"at_rho_zero": gradients("zero")}
    with torch.no_grad():
        # One ordinary optimizer step's worth of movement, not a hand-set value: this is
        # what the first real step would do to rho.
        sidecar.rho.add_(1e-3)
    report["gradients"]["after_rho_moves"] = gradients("moved")
    with torch.no_grad():
        sidecar.rho.fill_(0.5)
    report["gradients"]["fully_admitted"] = gradients("open")
    # Named for what a failure would be, so the field cannot be misread: true here means
    # the block is stuck, which is the two-stream pilot's failure reproduced.
    report["deadlock_detected"] = all(
        value == 0.0 for value in report["gradients"]["after_rho_moves"].values())

    print(json.dumps(report, indent=2)[:3500])
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

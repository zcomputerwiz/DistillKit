"""Is the converted 2B checkpoint a real, loadable, numerically sane Qwen3.5 text model?

Conversion validation only -- format and arithmetic, not quality. Nothing here touches the
n-gram table; if this does not pass there is no point retargeting anything to the model's
geometry.

    python scratch/native_table/validate_2b.py --model ../student-2b-hf
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

CONFIG_KEYS = (
    "vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers",
    "num_attention_heads", "num_key_value_heads", "head_dim", "full_attention_interval",
    "linear_num_key_heads", "linear_num_value_heads", "linear_key_head_dim",
    "linear_value_head_dim", "linear_conv_kernel_dim", "tie_word_embeddings",
    "mtp_num_hidden_layers", "rms_norm_eps",
)


def sequences(tokenizer, vocab_size, eos):
    """Cases that exercise different paths, not just different text."""
    ordinary = tokenizer("The capital of France is Paris, and the weather there is mild.")["input_ids"]
    repeated = [tokenizer("hello")["input_ids"][0]] * 64
    boundary = ordinary[:12] + [eos] + ordinary[:12]
    # Long enough to pass through every decoder block and several full-attention layers.
    long = (ordinary * 64)[:1024]
    return {"ordinary": ordinary, "repeated": repeated, "eos_boundary": boundary,
            "long_1024": long}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="D:/DeepThought/Projects/HybridModel/student-2b-hf")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    from transformers import AutoConfig, AutoTokenizer
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    report = {"model": str(args.model)}
    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    config = getattr(config, "text_config", config)
    report["config"] = {key: getattr(config, key, None) for key in CONFIG_KEYS}
    report["config"]["layer_types_counts"] = {
        kind: config.layer_types.count(kind) for kind in sorted(set(config.layer_types))}
    report["config"]["layer_types"] = list(config.layer_types)

    model, info = Qwen3_5ForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, local_files_only=True,
        output_loading_info=True)
    model = model.to(args.device).eval()
    model.config.use_cache = False
    report["loading"] = {key: list(value) for key, value in info.items()
                         if isinstance(value, (list, tuple))}
    parameters = sum(p.numel() for p in model.parameters())
    report["parameters"] = {
        "total": parameters,
        "embedding": model.get_input_embeddings().weight.numel(),
        "non_embedding": parameters - model.get_input_embeddings().weight.numel(),
        "tied_head": model.lm_head.weight is model.get_input_embeddings().weight,
    }
    report["checkpoint_bytes"] = sum(
        path.stat().st_size for path in Path(args.model).glob("*.safetensors"))

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    eos = config.eos_token_id if isinstance(config.eos_token_id, int) else config.eos_token_id[0]
    report["smoke"] = {}
    for name, ids in sequences(tokenizer, config.vocab_size, eos).items():
        tensor = torch.tensor([ids], device=args.device)
        with torch.no_grad():
            out = model(input_ids=tensor,
                        attention_mask=torch.ones_like(tensor),
                        output_hidden_states=True)
        logits = out.logits
        report["smoke"][name] = {
            "tokens": len(ids),
            "logits_shape": list(logits.shape),
            "finite": bool(torch.isfinite(logits).all()),
            "logit_min": float(logits.min()), "logit_max": float(logits.max()),
            "final_hidden_rms": float(out.hidden_states[-1].float().pow(2).mean().sqrt()),
            "top_token": tokenizer.decode([int(logits[0, -1].argmax())]),
        }
        assert logits.shape == (1, len(ids), config.vocab_size)
        assert torch.isfinite(logits).all()

    # Both attention paths actually ran, rather than both being configured: hook one
    # layer of each kind and record that it produced output.
    seen = {}
    handles = []
    for index, kind in enumerate(config.layer_types):
        if kind in seen:
            continue
        seen[kind] = index
        module = model.model.layers[index]
        handles.append(module.register_forward_hook(
            lambda mod, inputs, output, kind=kind: report.setdefault("paths", {}).update(
                {kind: {"layer": seen[kind],
                        "output_finite": bool(torch.isfinite(
                            output[0] if isinstance(output, tuple) else output).all())}})))
    ids = torch.tensor([sequences(tokenizer, config.vocab_size, eos)["long_1024"]],
                       device=args.device)
    with torch.no_grad():
        model(input_ids=ids, attention_mask=torch.ones_like(ids))
    for handle in handles:
        handle.remove()

    print(json.dumps(report, indent=2)[:4000])
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

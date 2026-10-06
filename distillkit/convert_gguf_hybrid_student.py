"""Convert the MLA/CSA2 and Gated DeltaNet student with llama.cpp's GGUF tools.

Run this file with the DistillKit Python environment. --dry-run reads the config,
safetensors header, and small routing tensors without loading the model weights.
"""

# Assisted-by: Codex

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import struct
import sys
from pathlib import Path


ARCH = "qwen35hybrid"
_COMMON = {
    "input_layernorm.weight": "attn_norm.weight",
    "post_attention_layernorm.weight": "post_attention_norm.weight",
    "mlp.gate_proj.weight": "ffn_gate.weight",
    "mlp.up_proj.weight": "ffn_up.weight",
    "mlp.down_proj.weight": "ffn_down.weight",
}
_LINEAR = {
    "linear_attn.in_proj_qkv.weight": "attn_qkv.weight",
    "linear_attn.in_proj_z.weight": "attn_gate.weight",
    "linear_attn.in_proj_a.weight": "ssm_alpha.weight",
    "linear_attn.in_proj_b.weight": "ssm_beta.weight",
    "linear_attn.conv1d.weight": "ssm_conv1d.weight",
    "linear_attn.out_proj.weight": "ssm_out.weight",
    "linear_attn.norm.weight": "ssm_norm.weight",
    "linear_attn.A_log": "ssm_a",
    "linear_attn.dt_bias": "ssm_dt.bias",
}
_FULL = {
    "self_attn.q_proj.weight": "attn_q.weight",
    "self_attn.q_norm.weight": "attn_q_norm.weight",
    "self_attn.kv_a_proj.weight": "attn_kv_a_mqa.weight",
    "self_attn.kv_a_norm.weight": "attn_kv_a_norm.weight",
    "self_attn.kv_b_proj.weight": "attn_k_b.weight + attn_v_b.weight",
    "self_attn.o_proj.weight": "attn_output.weight",
    "self_attn.index_q_proj.weight": "indexer.q_proj.weight",
    "self_attn.index_k_proj.weight": "indexer.k_proj.weight",
    "self_attn.indexer_proj.weight": "indexer.proj.weight",
    "self_attn.index_gate": "indexer.gate.weight",
}
_RESIDUAL = {
    "W_down.weight": "down.weight",
    "W_up.weight": "up.weight",
    "W_write.weight": "inject.weight",
    "blend": "blend.weight",
    "branch_gain_delta": "norm.weight",
}


def validate_config(cfg: dict) -> None:
    required = {
        "mla_enabled": True,
        "csa2_enabled": True,
        "csa2_router_bias": False,
        "residual_stream_enabled": True,
        "residual_stream_routing": "flash_next",
        "residual_stream_norm_mode": "exact",
        "residual_stream_num_branches": 2,
        "attn_output_gate": True,
        "tie_word_embeddings": True,
        "hidden_act": "silu",
    }
    for key, value in required.items():
        if cfg.get(key) != value:
            raise ValueError(f"Unsupported {key}={cfg.get(key)!r}; expected {value!r}")
    for key in ("attention_bias", "mla_content_key_norm", "residual_stream_sidecar"):
        if cfg.get(key, False):
            raise ValueError(f"Unsupported option {key}=true")
    for key in ("csa2_rope_index", "csa2_token_head_weights"):
        if not cfg.get(key, True):
            raise ValueError(f"Unsupported option {key}=false")
    if cfg.get("csa2_candidate_k", 0) or cfg.get("mlp_only_layers", []):
        raise ValueError("Candidate hierarchy and MLP-only layers are unsupported")
    layers = cfg["layer_types"]
    if len(layers) != cfg["num_hidden_layers"] or not all(t in ("linear_attention", "full_attention") for t in layers):
        raise ValueError("layer_types must list every linear_attention/full_attention block")
    modes = cfg.get("csa2_modes", ["full"] * layers.count("full_attention"))
    if modes != ["full"] * layers.count("full_attention"):
        raise ValueError("Only independent CSA2 full layers are supported")
    rope = cfg["rope_parameters"]
    rope_dim = int(cfg["head_dim"] * rope.get("partial_rotary_factor", cfg.get("partial_rotary_factor", 0.25)))
    if rope.get("rope_type", "default") != "default" or rope_dim % 2 or not 0 < rope_dim < cfg["head_dim"]:
        raise ValueError("Expected default partial RoPE with an even nonzero dimension")
    if sum(rope.get("mrope_section", [11, 11, 10])) * 2 != rope_dim:
        raise ValueError("MRoPE sections must span the rotary dimensions")
    if cfg["linear_num_value_heads"] % cfg["linear_num_key_heads"]:
        raise ValueError("DeltaNet value heads must be divisible by key heads")
    if cfg["linear_key_head_dim"] != cfg["linear_value_head_dim"]:
        raise ValueError("DeltaNet key and value head dimensions must be equal")
    for key in ("hidden_size", "intermediate_size", "num_attention_heads", "head_dim", "mla_latent_dim", "residual_stream_lowrank", "linear_key_head_dim", "linear_value_head_dim", "linear_conv_kernel_dim", "linear_num_key_heads", "linear_num_value_heads", "csa2_top_k"):
        if int(cfg[key]) <= 0:
            raise ValueError(f"{key} must be positive")
    if int(cfg.get("csa2_local_window", 32)) < 0:
        raise ValueError("csa2_local_window must be nonnegative")
    eps = float(cfg["rms_norm_eps"])
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("rms_norm_eps must be finite and positive")


def tensor_specs(cfg: dict) -> dict[str, tuple[str, tuple[int, ...]]]:
    hidden, ff = cfg["hidden_size"], cfg["intermediate_size"]
    branches, rank = cfg["residual_stream_num_branches"], cfg["residual_stream_lowrank"]
    heads, head_dim, latent = cfg["num_attention_heads"], cfg["head_dim"], cfg["mla_latent_dim"]
    rope_dim = int(head_dim * cfg["rope_parameters"].get("partial_rotary_factor", cfg.get("partial_rotary_factor", 0.25)))
    key_width = cfg["linear_num_key_heads"] * cfg["linear_key_head_dim"]
    val_width = cfg["linear_num_value_heads"] * cfg["linear_value_head_dim"]
    index_heads, index_dim = cfg.get("csa2_index_heads", 4), cfg.get("csa2_index_dim", 64)
    specs = {
        "model.embed_tokens.weight": ("token_embd.weight", (cfg["vocab_size"], hidden)),
        "model.norm.weight": ("output_norm.weight", (hidden,)),
    }
    common_shapes = [(hidden,), (hidden,), (ff, hidden), (ff, hidden), (hidden, ff)]
    linear_shapes = [(2 * key_width + val_width, hidden), (val_width, hidden), (cfg["linear_num_value_heads"], hidden), (cfg["linear_num_value_heads"], hidden), (2 * key_width + val_width, 1, cfg["linear_conv_kernel_dim"]), (hidden, val_width), (cfg["linear_value_head_dim"],), (cfg["linear_num_value_heads"],), (cfg["linear_num_value_heads"],)]
    full_shapes = [(2 * heads * head_dim, hidden), (head_dim,), (latent + rope_dim, hidden), (latent,), (heads * (2 * head_dim - rope_dim), latent), (hidden, heads * head_dim), (index_heads * (index_dim + rope_dim), hidden), (index_dim, latent), (index_heads, hidden), ()]
    residual_shapes = [(rank, branches * hidden), (branches * hidden, rank), (branches, branches * hidden), (), (branches, hidden)]
    for bid, layer_type in enumerate(cfg["layer_types"]):
        table, shapes = (_LINEAR, linear_shapes) if layer_type == "linear_attention" else (_FULL, full_shapes)
        for mappings, shape_list in ((_COMMON, common_shapes), (table, shapes)):
            for (source, target), shape in zip(mappings.items(), shape_list, strict=True):
                specs[f"model.layers.{bid}.{source}"] = (f"blk.{bid}.{target}", shape)
        for source_prefix, target_prefix in (("attn_residual", "hc_attn"), ("mlp_residual", "hc_ffn")):
            for (source, target), shape in zip(_RESIDUAL.items(), residual_shapes, strict=True):
                specs[f"model.layers.{bid}.{source_prefix}.{source}"] = (f"blk.{bid}.{target_prefix}_{target}", shape)
    return specs


def inspect_checkpoint(checkpoint: Path) -> tuple[dict, dict]:
    cfg = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    validate_config(cfg)
    path = checkpoint / "model.safetensors"
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError("Truncated safetensors header")
        length = struct.unpack("<Q", prefix)[0]
        if length > 16 * 1024 * 1024:
            raise ValueError("Safetensors header exceeds 16 MiB")
        header = json.loads(stream.read(length))
        header.pop("__metadata__", None)
        specs = tensor_specs(cfg)
        if set(header) != set(specs):
            raise ValueError(f"Tensor key mismatch: missing={sorted(set(specs) - set(header))}, unexpected={sorted(set(header) - set(specs))}")
        ranges = []
        data_size = path.stat().st_size - length - 8
        for name, entry in header.items():
            shape = tuple(entry["shape"])
            if shape != specs[name][1]:
                raise ValueError(f"Shape mismatch for {name}: {shape} != {specs[name][1]}")
            dtype = entry["dtype"]
            if dtype not in ("F32", "F16", "BF16"):
                raise ValueError(f"Unsupported dtype {dtype!r} for {name}")
            start, end = entry["data_offsets"]
            expected_bytes = math.prod(shape) * (4 if dtype == "F32" else 2)
            if not 0 <= start <= end <= data_size or end - start != expected_bytes:
                raise ValueError(f"Invalid data range for {name}")
            ranges.append((start, end))
            if name.endswith((".blend", ".branch_gain_delta", ".index_gate")):
                stream.seek(8 + length + start)
                raw = stream.read(end - start)
                if dtype == "BF16":
                    values = [struct.unpack("<f", struct.pack("<I", value[0] << 16))[0] for value in struct.iter_unpack("<H", raw)]
                else:
                    values = [value[0] for value in struct.iter_unpack("<f" if dtype == "F32" else "<e", raw)]
                if not all(math.isfinite(value) for value in values):
                    raise ValueError(f"Nonfinite routing value in {name}")
                if name.endswith(".blend") and not 0 <= values[0] <= 1:
                    raise ValueError(f"Blend outside [0, 1] in {name}")
        cursor = 0
        for start, end in sorted(ranges):
            if start != cursor:
                raise ValueError("Safetensors data ranges overlap or contain a gap")
            cursor = end
        if cursor != data_size:
            raise ValueError("Safetensors data has unaccounted bytes")
    for filename in ("tokenizer.json", "tokenizer_config.json"):
        if not (checkpoint / filename).is_file():
            raise ValueError(f"Missing {filename}")
    return cfg, header


def tokenizer_eos_id(checkpoint: Path) -> int | None:
    tokenizer_cfg = json.loads((checkpoint / "tokenizer_config.json").read_text(encoding="utf-8"))
    token = tokenizer_cfg.get("eos_token")
    if isinstance(token, dict):
        token = token["content"]
    if token is None:
        return None
    tokenizer = json.loads((checkpoint / "tokenizer.json").read_text(encoding="utf-8"))
    for added in tokenizer.get("added_tokens", []):
        if added["content"] == token:
            return int(added["id"])
    vocab = tokenizer["model"]["vocab"]
    if token not in vocab:
        raise ValueError(f"Tokenizer EOS token {token!r} is absent from its vocabulary")
    return int(vocab[token])


def parse_eog_ids(value: str) -> list[int]:
    try:
        ids = [int(part.strip()) for part in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("EOG token IDs must be a nonempty comma-separated integer list") from exc
    return list(dict.fromkeys(ids))


def converter_class(llama_cpp: Path):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sys.path.insert(0, str(llama_cpp / "gguf-py"))
    sys.path.insert(0, str(llama_cpp))
    import gguf
    from conversion.qwen import Qwen3_5TextModel

    class HybridStudentModel(Qwen3_5TextModel):
        model_arch = gguf.MODEL_ARCH.QWEN35HYBRID
        no_mtp = True
        hybrid_eos_token_id: int | None = None
        hybrid_eog_token_ids: list[int] | None = None

        def set_vocab(self):
            super().set_vocab()
            if self.hybrid_eos_token_id is not None:
                self.gguf_writer.add_eos_token_id(self.hybrid_eos_token_id)
            if self.hybrid_eog_token_ids is not None:
                self.gguf_writer.add_array(gguf.Keys.Tokenizer.EOG_IDS, self.hybrid_eog_token_ids)

        def set_gguf_parameters(self):
            cfg, writer = self.hparams, self.gguf_writer
            self.hparams = {**cfg, "num_key_value_heads": 1}
            try:
                super().set_gguf_parameters()
            finally:
                self.hparams = cfg
            rope_dim = int(cfg["head_dim"] * self.rope_parameters.get("partial_rotary_factor", 0.25))
            writer.add_kv_lora_rank(cfg["mla_latent_dim"])
            writer.add_attention_scale(cfg["head_dim"] ** -0.5)
            writer.add_indexer_head_count(cfg.get("csa2_index_heads", 4))
            writer.add_indexer_key_length(cfg.get("csa2_index_dim", 64) + rope_dim)
            writer.add_indexer_top_k(cfg["csa2_top_k"])
            writer.add_indexer_block_size(cfg.get("csa2_block_size", 128))
            writer.add_uint32(gguf.Keys.Attention.Indexer.LOCAL_WINDOW.format(arch=ARCH), cfg.get("csa2_local_window", 32))
            writer.add_bool(gguf.Keys.Attention.Indexer.ROUTER_BIAS.format(arch=ARCH), cfg.get("csa2_router_bias", True))
            writer.add_bool(gguf.Keys.Attention.Indexer.ROPE_INDEX.format(arch=ARCH), True)
            writer.add_array(gguf.Keys.Attention.Indexer.MODES.format(arch=ARCH), cfg.get("csa2_modes", ["full"] * cfg["layer_types"].count("full_attention")))
            writer.add_hyper_connection_count(cfg["residual_stream_num_branches"])
            writer.add_hyper_connection_low_rank(cfg["residual_stream_lowrank"])
            writer.add_hyper_connection_epsilon(cfg["rms_norm_eps"])
            writer.add_string(gguf.Keys.HyperConnection.NORM_MODE.format(arch=ARCH), cfg["residual_stream_norm_mode"])
            writer.add_string(gguf.Keys.HyperConnection.ROUTING.format(arch=ARCH), cfg["residual_stream_routing"])

        def modify_tensors(self, data_torch, name, bid):
            if name.endswith("self_attn.index_k_proj.weight"):
                yield self.map_tensor_name(name), data_torch.transpose(0, 1).contiguous()
            elif name.endswith("self_attn.kv_b_proj.weight"):
                cfg = self.hparams
                rope_dim = int(cfg["head_dim"] * self.rope_parameters.get("partial_rotary_factor", 0.25))
                content_dim = cfg["head_dim"] - rope_dim
                kv = data_torch.reshape(cfg["num_attention_heads"], content_dim + cfg["head_dim"], cfg["mla_latent_dim"])
                key, value = kv.split((content_dim, cfg["head_dim"]), dim=1)
                yield self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_K_B, bid, ".weight"), key.transpose(1, 2).contiguous()
                yield self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_V_B, bid, ".weight"), value.contiguous()
            elif name.endswith(".branch_gain_delta"):
                yield self.map_tensor_name(name) + ".weight", data_torch.float() + 1.0
            elif name.endswith((".blend", ".index_gate")):
                yield self.map_tensor_name(name) + ".weight", data_torch.float().reshape(1)
            else:
                if name.endswith(".A_log") or name.endswith("norm.weight") and not name.endswith("linear_attn.norm.weight"):
                    data_torch = data_torch.float()
                yield from super().modify_tensors(data_torch, name, bid)

    return HybridStudentModel, gguf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Directory with config.json and model.safetensors")
    parser.add_argument("--llama-cpp", type=Path, default=Path(__file__).resolve().parents[2] / "llama.cpp")
    parser.add_argument("--outfile", type=Path)
    parser.add_argument("--outtype", choices=("f32", "f16", "bf16"), default="f32")
    parser.add_argument("--eos-token-id", type=int, help="Explicit primary EOS and default sole generation stop ID")
    parser.add_argument("--eog-token-ids", type=parse_eog_ids, help="Comma-separated generation stop IDs; defaults to the explicit primary EOS alone")
    parser.add_argument("--dry-run", action="store_true", help="Validate source shapes and small routing values without converting")
    parser.add_argument("--vocab-only", action="store_true", help="Write tokenizer metadata without weights")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    cfg, header = inspect_checkpoint(args.checkpoint)
    tokenizer_eos = tokenizer_eos_id(args.checkpoint)
    generation_path = args.checkpoint / "generation_config.json"
    generation_eos = json.loads(generation_path.read_text(encoding="utf-8")).get("eos_token_id") if generation_path.is_file() else cfg.get("eos_token_id")
    eos_mismatch = cfg.get("eos_token_id") != tokenizer_eos or generation_eos != tokenizer_eos
    effective_eos = args.eos_token_id if args.eos_token_id is not None else tokenizer_eos if not eos_mismatch else None
    eog_ids = args.eog_token_ids if args.eog_token_ids is not None else [args.eos_token_id] if args.eos_token_id is not None else None
    if args.eos_token_id is not None and not 0 <= args.eos_token_id < cfg["vocab_size"]:
        parser.error("--eos-token-id must be within the vocabulary")
    if eog_ids is not None:
        if not eog_ids or any(not 0 <= token_id < cfg["vocab_size"] for token_id in eog_ids):
            parser.error("--eog-token-ids must be nonempty and within the vocabulary")
        if effective_eos is not None and effective_eos not in eog_ids:
            parser.error("The primary EOS token ID must belong to --eog-token-ids")
    if args.dry_run:
        print(json.dumps({"architecture": ARCH, "source_tensors": len(header), "gguf_tensors": len(header) + cfg["layer_types"].count("full_attention"), "layers": cfg["num_hidden_layers"], "mla_layers": [i for i, kind in enumerate(cfg["layer_types"]) if kind == "full_attention"], "cache_key_width": cfg["mla_latent_dim"] + int(cfg["head_dim"] * cfg["rope_parameters"].get("partial_rotary_factor", 0.25)), "router_bias": cfg.get("csa2_router_bias", True), "routing": "token top-k plus local window", "block_size_is_provenance": True, "config_eos_token_id": cfg.get("eos_token_id"), "generation_eos_token_id": generation_eos, "tokenizer_eos_token_id": tokenizer_eos, "effective_primary_eos_token_id": effective_eos, "eog_override_token_ids": eog_ids}, indent=2))
        return
    if args.eos_token_id is None and eos_mismatch:
        parser.error(f"EOS metadata disagree: config={cfg.get('eos_token_id')}, generation={generation_eos}, tokenizer={tokenizer_eos}; select --eos-token-id explicitly")
    if args.outfile is None:
        parser.error("--outfile is required unless --dry-run is used")
    if args.outfile.exists():
        parser.error(f"Output already exists: {args.outfile}")
    if not (args.llama_cpp / "conversion" / "qwen.py").is_file():
        parser.error("--llama-cpp must point to the ported llama.cpp checkout")
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    model_cls, gguf = converter_class(args.llama_cpp.resolve())
    ftype = {"f32": gguf.LlamaFileType.ALL_F32, "f16": gguf.LlamaFileType.MOSTLY_F16, "bf16": gguf.LlamaFileType.MOSTLY_BF16}[args.outtype]
    model = model_cls(args.checkpoint.resolve(), ftype, args.outfile.resolve(), hparams=cfg, model_name="DistillKit hybrid student", use_temp_file=True)
    model.hybrid_eos_token_id = args.eos_token_id
    model.hybrid_eog_token_ids = eog_ids
    if args.vocab_only:
        model.write_vocab()
    else:
        model.write()


if __name__ == "__main__":
    main()

"""Convert a Qwen3.5 text-only GGUF (llama.cpp layout) to an HF safetensors checkpoint.

The source is the local fine-tune ``Qwen3.8-4B-BF16.gguf`` (empero-ai/
Qwen3.8-4B-Distill-GGUF:BF16): text-only, BF16/F32 tensors, tied embeddings
(no output head stored), and a trailing MTP block that the text class does not
model.

Layout facts this converter relies on (verified against the gguf 0.19.0 reader
source and llama.cpp's own HF->GGUF converter, ``conversion/qwen.py``):

* Header dims are llama.cpp ``[in, out]`` for linears, but the tensor data view
  (``ReaderTensor.data``) is already reversed to HF orientation: a 2-D tensor's
  byte stream is exactly its HF row-major ``[out, in]`` array. No transposes.
* F32 tensors decode directly; BF16 arrives as a uint8 byte view that this
  module decodes with the exact ``uint16 << 16 -> float32`` bit trick.
* ``ssm_conv1d`` arrives as ``[channels, kernel]`` and is reshaped to HF's
  ``[channels, 1, kernel]``.
* When ``linear_num_key_heads != linear_num_value_heads``, llama.cpp reorders
  the V heads from grouped-by-K-head (HF layout) to tiled order for ggml
  broadcast (``_LinearAttentionVReorderBase``). This converter applies the
  exact inverse permutation to in_proj_qkv (V rows), in_proj_z, in_proj_a/b,
  conv1d (V channels), out_proj (columns), A_log and dt_bias.
* Value transforms applied by llama.cpp's writer are inverted here:
  ``ssm_a`` stores ``-exp(A_log)`` (inverted with ``log(-x)``), and every
  RMSNorm weight except ``linear_attn.norm`` is stored as ``w + 1``.

What this module does:

* Rebuilds a ``Qwen3_5TextConfig`` from the ``qwen35.*`` KV metadata. Every
  tensor shape is cross-checked against the config-derived expectation before
  anything is written, so a wrong assumption fails the conversion instead of
  producing a silently swapped checkpoint.
* Keeps the two Mamba-dtype tensors (``A_log``, ``linear_attn.norm``) in
  float32 exactly like the stock checkpoint; everything else is bfloat16.
* Excludes the trailing MTP block: ``Qwen3_5ForCausalLM`` has no MTP module,
  and its load path ignores ``^mtp.*`` anyway. The excluded tensor list is
  printed so nothing drops silently.
* Writes a single ``model.safetensors`` plus ``config.json`` and copies the
  tokenizer files from a stock sibling checkpoint (identical family tokenizer).

Run: ``python -m distillkit.convert_gguf_student --help``
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import click
import numpy as np
import torch
from safetensors.torch import save_file

GGUF_ARCH = "qwen35"
# Family constant from the stock checkpoint's text_config (spec section 0). The
# GGUF's own tokenizer.ggml.eos_token_id is a llama.cpp-specific mapping and is
# deliberately not used.
EOS_TOKEN_ID = 248044
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
)


def read_kv(reader) -> dict[str, object]:
    return {key: field.contents() for key, field in reader.fields.items()}


def derive_text_config(
    kv: dict[str, object], vocab_size: int | None = None, tie_word_embeddings: bool = True
) -> dict:
    """Build the Qwen3_5TextConfig dict from qwen35.* KV values.

    ``vocab_size`` comes from the embedding tensor row count (the GGUF does not
    store it as a scalar). ``tie_word_embeddings`` likewise is not a KV field: it is
    whether the file carries its own ``output.weight``. The 4B student ties, the 27B
    teacher does not, so this cannot be assumed either way.
    """
    def g(key: str):
        full = f"{GGUF_ARCH}.{key}"
        if full not in kv:
            raise KeyError(f"GGUF is missing required metadata field {full!r}")
        return kv[full]

    block_count = int(g("block_count"))
    nextn = int(g("nextn_predict_layers"))
    num_hidden_layers = block_count - nextn
    hidden_size = int(g("embedding_length"))
    intermediate_size = int(g("feed_forward_length"))
    head_dim = int(g("attention.key_length"))
    if int(g("attention.value_length")) != head_dim:
        raise ValueError("attention.key_length != attention.value_length")

    state_size = int(g("ssm.state_size"))
    group_count = int(g("ssm.group_count"))
    inner_size = int(g("ssm.inner_size"))
    if inner_size % state_size:
        raise ValueError(f"ssm.inner_size {inner_size} is not a multiple of ssm.state_size {state_size}")
    linear_num_value_heads = inner_size // state_size

    rope_dim = int(g("rope.dimension_count"))
    sections = [int(s) for s in g("rope.dimension_sections")]
    while sections and sections[-1] == 0:
        sections.pop()
    if sum(sections) != rope_dim // 2:
        raise ValueError(f"mrope section sum {sum(sections)} != dimension_count/2 ({rope_dim // 2})")
    if head_dim % rope_dim:
        raise ValueError(f"rope dimension_count {rope_dim} does not divide head_dim {head_dim}")

    interval = int(g("full_attention_interval"))
    return {
        "architectures": ["Qwen3_5ForCausalLM"],
        "model_type": "qwen3_5_text",
        "attention_bias": False,
        "attention_dropout": 0.0,
        "attn_output_gate": True,
        "dtype": "bfloat16",
        "eos_token_id": EOS_TOKEN_ID,
        "full_attention_interval": interval,
        "head_dim": head_dim,
        "hidden_act": "silu",
        "hidden_size": hidden_size,
        "initializer_range": 0.02,
        "intermediate_size": intermediate_size,
        "layer_types": [
            "full_attention" if (i + 1) % interval == 0 else "linear_attention"
            for i in range(num_hidden_layers)
        ],
        "linear_conv_kernel_dim": int(g("ssm.conv_kernel")),
        "linear_key_head_dim": state_size,
        "linear_num_key_heads": group_count,
        "linear_num_value_heads": linear_num_value_heads,
        "linear_value_head_dim": state_size,
        "max_position_embeddings": int(g("context_length")),
        "mlp_only_layers": [],
        "mtp_num_hidden_layers": nextn,
        "mtp_use_dedicated_embeddings": False,
        "num_attention_heads": int(g("attention.head_count")),
        "num_hidden_layers": num_hidden_layers,
        "num_key_value_heads": int(g("attention.head_count_kv")),
        "rms_norm_eps": float(g("attention.layer_norm_rms_epsilon")),
        "tie_word_embeddings": tie_word_embeddings,
        "use_cache": True,
        "vocab_size": vocab_size,
        "mamba_ssm_dtype": "float32",
        "rope_parameters": {
            "mrope_interleaved": True,
            "mrope_section": sections,
            "rope_type": "default",
            "rope_theta": float(g("rope.freq_base")),
            "partial_rotary_factor": rope_dim / head_dim,
        },
    }


class _Dims:
    """Expected HF shapes per GGUF suffix, derived from the config."""

    def __init__(self, cfg: dict, time_step_rank: int):
        self.H = cfg["hidden_size"]
        self.I = cfg["intermediate_size"]
        key_dim = cfg["linear_num_key_heads"] * cfg["linear_key_head_dim"]
        val_dim = cfg["linear_num_value_heads"] * cfg["linear_value_head_dim"]
        self.qkv_dim = 2 * key_dim + val_dim
        self.val_dim = val_dim
        q_out = cfg["num_attention_heads"] * cfg["head_dim"]
        if cfg.get("attn_output_gate"):
            q_out *= 2
        self.q_out = q_out
        self.kv_out = cfg["num_key_value_heads"] * cfg["head_dim"]
        self.tsr = time_step_rank
        self.kernel = cfg["linear_conv_kernel_dim"]
        self.key_dim = key_dim
        self.head_v = cfg["linear_value_head_dim"]
        self.num_k_heads = cfg["linear_num_key_heads"]
        self.num_v_per_k = cfg["linear_num_value_heads"] // cfg["linear_num_key_heads"]

    _UNRESOLVED = ()

    def expected(self, layer_type: str | None, suffix: str) -> tuple[int, ...]:
        """Expected HF data-view shape. ``_UNRESOLVED`` marks the few entries the
        convert loop fills from config scalars (vocab, head_dim, key_head_dim)."""
        H, I = self.H, self.I
        common = {
            "attn_norm.weight": (H,),
            "post_attention_norm.weight": (H,),
            "ffn_gate.weight": (I, H),
            "ffn_up.weight": (I, H),
            "ffn_down.weight": (H, I),
        }
        if layer_type is None:
            table = {
                "token_embd.weight": self._UNRESOLVED,
                "output_norm.weight": (H,),
                # Untied output head. Same shape as the embedding, and present only
                # when the checkpoint does not tie them (the 27B teacher does not).
                "output.weight": self._UNRESOLVED,
            }
        elif layer_type == "linear_attention":
            table = {
                **common,
                "attn_qkv.weight": (self.qkv_dim, H),
                "attn_gate.weight": (self.val_dim, H),
                "ssm_alpha.weight": (self.tsr, H),
                "ssm_beta.weight": (self.tsr, H),
                "ssm_conv1d.weight": (self.qkv_dim, self.kernel),  # pre-reshape view
                "ssm_out.weight": (H, self.val_dim),
                "ssm_norm.weight": self._UNRESOLVED,
                "ssm_a": (self.tsr,),
                "ssm_dt.bias": (self.tsr,),
            }
        else:
            table = {
                **common,
                "attn_q.weight": (self.q_out, H),
                "attn_k.weight": (self.kv_out, H),
                "attn_v.weight": (self.kv_out, H),
                "attn_output.weight": (H, self.q_out // 2),
                "attn_q_norm.weight": self._UNRESOLVED,
                "attn_k_norm.weight": self._UNRESOLVED,
            }
        if suffix not in table:
            raise ValueError(f"Suffix {suffix!r} is not valid for layer type {layer_type!r}")
        return table[suffix]


# (hf_key suffix after model.layers.N, transform, torch dtype)
_COMMON_MAP = {
    "attn_norm.weight": ("input_layernorm.weight", "copy", torch.bfloat16),
    "post_attention_norm.weight": ("post_attention_layernorm.weight", "copy", torch.bfloat16),
    "ffn_gate.weight": ("mlp.gate_proj.weight", "copy", torch.bfloat16),
    "ffn_up.weight": ("mlp.up_proj.weight", "copy", torch.bfloat16),
    "ffn_down.weight": ("mlp.down_proj.weight", "copy", torch.bfloat16),
}
_LINEAR_MAP = {
    "attn_qkv.weight": ("linear_attn.in_proj_qkv.weight", "copy", torch.bfloat16),
    "attn_gate.weight": ("linear_attn.in_proj_z.weight", "copy", torch.bfloat16),
    "ssm_alpha.weight": ("linear_attn.in_proj_a.weight", "copy", torch.bfloat16),
    "ssm_beta.weight": ("linear_attn.in_proj_b.weight", "copy", torch.bfloat16),
    "ssm_conv1d.weight": ("linear_attn.conv1d.weight", "conv", torch.bfloat16),
    "ssm_out.weight": ("linear_attn.out_proj.weight", "copy", torch.bfloat16),
    # Mamba-dtype tensors stay float32, matching the stock checkpoint.
    "ssm_dt.bias": ("linear_attn.dt_bias", "copy", torch.bfloat16),
    # Mamba-dtype tensors stay float32, matching the stock checkpoint.
    "ssm_norm.weight": ("linear_attn.norm.weight", "copy", torch.float32),
    "ssm_a": ("linear_attn.A_log", "copy", torch.float32),
}
_FULL_MAP = {
    "attn_q.weight": ("self_attn.q_proj.weight", "copy", torch.bfloat16),
    "attn_k.weight": ("self_attn.k_proj.weight", "copy", torch.bfloat16),
    "attn_v.weight": ("self_attn.v_proj.weight", "copy", torch.bfloat16),
    "attn_output.weight": ("self_attn.o_proj.weight", "copy", torch.bfloat16),
    "attn_q_norm.weight": ("self_attn.q_norm.weight", "copy", torch.bfloat16),
    "attn_k_norm.weight": ("self_attn.k_norm.weight", "copy", torch.bfloat16),
}


def map_gguf_tensor(name: str, num_hidden_layers: int) -> tuple[str, str, torch.dtype] | None:
    """Map one GGUF tensor name to (hf_key, transform, dtype); None means MTP skip."""
    if name == "token_embd.weight":
        return ("model.embed_tokens.weight", "copy", torch.bfloat16)
    if name == "output_norm.weight":
        return ("model.norm.weight", "copy", torch.bfloat16)
    if name == "output.weight":
        # Present only when the head is untied. The 4B student ties its embeddings and
        # omits this; the 27B teacher ships a real one.
        return ("lm_head.weight", "copy", torch.bfloat16)
    match = re.fullmatch(r"blk\.(\d+)\.(.*)", name)
    if match is None:
        raise KeyError(f"Unrecognized GGUF tensor name {name!r}")
    index, suffix = int(match.group(1)), match.group(2)
    if index >= num_hidden_layers:
        return None  # trailing MTP block
    base = f"model.layers.{index}"
    for table in (_COMMON_MAP, _LINEAR_MAP, _FULL_MAP):
        if suffix in table:
            key, transform, dtype = table[suffix]
            return (f"{base}.{key}", transform, dtype)
    raise KeyError(f"No mapping for tensor {name!r}")


def expected_key_set(cfg: dict) -> set[str]:
    keys = {"model.embed_tokens.weight", "model.norm.weight"}
    if not cfg.get("tie_word_embeddings", True):
        # Untied head: the checkpoint carries its own lm_head (the 27B teacher).
        keys.add("lm_head.weight")
    for i, layer_type in enumerate(cfg["layer_types"]):
        base = f"model.layers.{i}"
        keys |= {
            f"{base}.input_layernorm.weight",
            f"{base}.post_attention_layernorm.weight",
            f"{base}.mlp.gate_proj.weight",
            f"{base}.mlp.up_proj.weight",
            f"{base}.mlp.down_proj.weight",
        }
        if layer_type == "linear_attention":
            keys |= {
                f"{base}.linear_attn.in_proj_qkv.weight",
                f"{base}.linear_attn.in_proj_z.weight",
                f"{base}.linear_attn.in_proj_a.weight",
                f"{base}.linear_attn.in_proj_b.weight",
                f"{base}.linear_attn.conv1d.weight",
                f"{base}.linear_attn.out_proj.weight",
                f"{base}.linear_attn.norm.weight",
                f"{base}.linear_attn.A_log",
                f"{base}.linear_attn.dt_bias",
            }
        else:
            keys |= {
                f"{base}.self_attn.q_proj.weight",
                f"{base}.self_attn.k_proj.weight",
                f"{base}.self_attn.v_proj.weight",
                f"{base}.self_attn.o_proj.weight",
                f"{base}.self_attn.q_norm.weight",
                f"{base}.self_attn.k_norm.weight",
            }
    return keys


def _unreorder_v_heads(array: np.ndarray, dim: int, num_k_heads: int, num_v_per_k: int, head_dim: int) -> np.ndarray:
    """Inverse of llama.cpp's grouped->tiled V-head reorder (GGUF->HF direction).

    Forward (HF->GGUF, ``_reorder_v_heads``): reshape the target dim to
    ``[K, V, D]`` and swap the first two axes -> ``[V, K, D]``. The inverse is
    the same swap with the parse order reversed: reshape to ``[V, K, D]``,
    swap, flatten. (Not an involution when K != V.)
    """
    shape = list(array.shape)
    if dim < 0:
        dim += len(shape)
    new_shape = shape[:dim] + [num_v_per_k, num_k_heads, head_dim] + shape[dim + 1:]
    array = array.reshape(new_shape)
    axes = list(range(array.ndim))
    axes[dim], axes[dim + 1] = axes[dim + 1], axes[dim]
    return np.ascontiguousarray(array.transpose(axes)).reshape(shape)


def _invert_ssm_a(gguf_values: np.ndarray) -> np.ndarray:
    """Recover HF ``A_log`` from the GGUF's stored ``-exp(A_log)``."""
    if not (gguf_values < 0).all():
        raise ValueError("ssm_a must be strictly negative (llama.cpp stores -exp(A_log))")
    return np.log(-gguf_values)


def _decode_tensor(reader, tensor) -> np.ndarray:
    """f32 array in HF orientation (data views are already [out, in])."""
    if reader.byte_order != "I":
        raise ValueError("Only little-endian GGUF files are supported")
    from gguf import GGMLQuantizationType

    if tensor.tensor_type == GGMLQuantizationType.F32:
        array = np.asarray(tensor.data)
        if array.dtype != np.float32:
            raise ValueError(f"Expected F32 data for {tensor.name}, got {array.dtype}")
        return np.array(array, dtype=np.float32)  # writable copy off the memmap
    if tensor.tensor_type == GGMLQuantizationType.BF16:
        raw = np.ascontiguousarray(tensor.data)
        if raw.dtype != np.uint8 or raw.size != tensor.n_bytes:
            raise ValueError(f"Unexpected BF16 byte view for {tensor.name}")
        bits = raw.view(np.uint16).reshape(-1)
        f32 = (bits.astype(np.uint32) << 16).view(np.float32)
        return f32.reshape(raw.shape[:-1] + (raw.shape[-1] // 2,))

    # Quantized tensors (the 27B teacher ships Q8_0 for most weights). ggml's own
    # dequantizer is the reference; hand-rolling per-type unpacking here would be a
    # second place for the nibble/scale conventions to drift.
    from gguf import quants

    try:
        array = quants.dequantize(np.ascontiguousarray(tensor.data), tensor.tensor_type)
    except Exception as exc:  # unknown/unsupported ggml type
        raise ValueError(
            f"Unsupported GGML type {tensor.tensor_type} for tensor {tensor.name}"
        ) from exc
    expected = int(np.prod(tensor.shape))
    if array.size != expected:
        raise ValueError(
            f"{tensor.name}: dequantized {array.size} elements, expected {expected}"
        )
    # GGUF ne is reversed relative to the HF [out, in] view the caller expects.
    return np.ascontiguousarray(array, dtype=np.float32).reshape(
        tuple(int(d) for d in reversed(tensor.shape))
    )


def convert_gguf_to_hf(
    gguf_path: str | Path,
    output_dir: str | Path,
    tokenizer_source: str | Path | None = None,
) -> dict:
    """Convert one GGUF file into an HF checkpoint directory. Returns a summary."""
    from gguf import GGUFReader

    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory {output} must be new or empty")
    output.mkdir(parents=True, exist_ok=True)

    reader = GGUFReader(str(gguf_path))
    kv = read_kv(reader)
    arch = kv.get("general.architecture")
    if arch != GGUF_ARCH:
        raise ValueError(f"Expected {GGUF_ARCH!r} architecture, found {arch!r}")

    embed = next(t for t in reader.tensors if t.name == "token_embd.weight")
    vocab_size, hidden_size = int(embed.shape[1]), int(embed.shape[0])
    # A stored output head means the embeddings are not tied. Asserting either way
    # would be wrong for one of the two models this converter handles.
    has_output_head = any(t.name == "output.weight" for t in reader.tensors)
    cfg = derive_text_config(
        kv, vocab_size=vocab_size, tie_word_embeddings=not has_output_head
    )
    if cfg["hidden_size"] != hidden_size:
        raise ValueError("embedding_length disagrees with token_embd row count")

    dims = _Dims(cfg, time_step_rank=int(kv[f"{GGUF_ARCH}.ssm.time_step_rank"]))
    state_dict: dict[str, torch.Tensor] = {}
    skipped_mtp: list[str] = []
    total_bytes = 0

    for tensor in reader.tensors:
        mapped = map_gguf_tensor(tensor.name, cfg["num_hidden_layers"])
        if mapped is None:
            skipped_mtp.append(tensor.name)
            continue
        hf_key, transform, dtype = mapped
        array = _decode_tensor(reader, tensor)

        match = re.fullmatch(r"model\.layers\.(\d+)\..*", hf_key)
        layer_type = cfg["layer_types"][int(match.group(1))] if match else None
        suffix = tensor.name.split(".", 2)[2] if tensor.name.startswith("blk.") else tensor.name
        expected = dims.expected(layer_type, suffix)
        if expected is dims._UNRESOLVED:
            if suffix in ("token_embd.weight", "output.weight"):
                expected = (vocab_size, hidden_size)
            elif suffix == "ssm_norm.weight":
                expected = (cfg["linear_key_head_dim"],)
            else:  # attn_q_norm / attn_k_norm
                expected = (cfg["head_dim"],)
        elif suffix == "ssm_norm.weight":
            expected = (cfg["linear_key_head_dim"],)
        elif suffix in ("attn_q_norm.weight", "attn_k_norm.weight"):
            expected = (cfg["head_dim"],)
        if tuple(int(s) for s in array.shape) != expected:
            raise ValueError(
                f"Shape mismatch for {tensor.name}: data view {tuple(array.shape)} != expected HF {expected}"
            )

        # Invert llama.cpp's V-head reorder (grouped->tiled) when key/value head
        # counts differ; the A_log value transform applies to every layout.
        if layer_type == "linear_attention" and dims.num_k_heads != cfg["linear_num_value_heads"]:
            K, R, Dv = dims.num_k_heads, dims.num_v_per_k, dims.head_v
            v_start = 2 * dims.key_dim
            if suffix == "attn_qkv.weight":
                qk, v = array[:v_start], array[v_start:]
                array = np.concatenate([qk, _unreorder_v_heads(v, 0, K, R, Dv)], axis=0)
            elif suffix == "attn_gate.weight":
                array = _unreorder_v_heads(array, 0, K, R, Dv)
            elif suffix in ("ssm_alpha.weight", "ssm_beta.weight"):
                array = _unreorder_v_heads(array, 0, K, R, 1)
            elif suffix == "ssm_conv1d.weight":
                qk, v = array[:v_start], array[v_start:]
                array = np.concatenate([qk, _unreorder_v_heads(v, 0, K, R, Dv)], axis=0)
            elif suffix == "ssm_out.weight":
                array = _unreorder_v_heads(array, 1, K, R, Dv)
            elif suffix in ("ssm_a", "ssm_dt.bias"):
                array = _unreorder_v_heads(array, 0, K, R, 1)
        if layer_type == "linear_attention" and suffix == "ssm_a":
            # GGUF stores -exp(A_log); recover A_log regardless of head layout.
            array = _invert_ssm_a(array)
        # llama.cpp stores every RMSNorm weight except linear_attn.norm as w + 1.
        if suffix in ("attn_norm.weight", "post_attention_norm.weight",
                      "attn_q_norm.weight", "attn_k_norm.weight") or tensor.name == "output_norm.weight":
            array = array - 1.0

        if transform == "conv":
            # [channels, kernel] -> [channels, 1, kernel]
            array = array.reshape(dims.qkv_dim, 1, dims.kernel)
        state_dict[hf_key] = torch.from_numpy(np.ascontiguousarray(array)).to(dtype)
        total_bytes += state_dict[hf_key].element_size() * state_dict[hf_key].numel()

    produced = set(state_dict)
    wanted = expected_key_set(cfg)
    missing = wanted - produced
    extra = produced - wanted
    if missing or extra:
        raise ValueError(
            f"Key set mismatch after mapping. missing={sorted(missing)[:8]} extra={sorted(extra)[:8]}"
        )

    save_file(state_dict, str(output / "model.safetensors"), metadata={"format": "pt"})
    (output / "config.json").write_text(json.dumps(cfg, indent=4) + "\n", encoding="utf-8")

    copied: list[str] = []
    if tokenizer_source is not None:
        source = Path(tokenizer_source)
        for filename in TOKENIZER_FILES:
            candidate = source / filename
            if candidate.exists():
                shutil.copy2(candidate, output / filename)
                copied.append(filename)

    return {
        "output": str(output),
        "tensors": len(state_dict),
        "bytes": total_bytes,
        "skipped_mtp": skipped_mtp,
        "tokenizer_files_copied": copied,
        "vocab_size": vocab_size,
        "num_hidden_layers": cfg["num_hidden_layers"],
        "gguf_tokenizer_eos": kv.get("tokenizer.ggml.eos_token_id"),
    }


@click.command("convert-gguf-student")
@click.option("--gguf", "gguf_path", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--output", type=click.Path(file_okay=False), required=True)
@click.option(
    "--tokenizer-source",
    type=click.Path(exists=True, file_okay=False),
    default=None,
    help="Directory with the family tokenizer files (e.g. a stock sibling checkpoint).",
)
def main(gguf_path, output, tokenizer_source):
    """Convert a Qwen3.5 text GGUF to an HF safetensors checkpoint."""
    summary = convert_gguf_to_hf(gguf_path, output, tokenizer_source)
    click.echo(f"Wrote {summary['tensors']} tensors ({summary['bytes'] / 2**30:.2f} GiB) to {summary['output']}")
    click.echo(f"Skipped MTP tensors: {len(summary['skipped_mtp'])}")
    for name in summary["skipped_mtp"]:
        click.echo(f"  - {name}")
    if summary["tokenizer_files_copied"]:
        click.echo(f"Copied tokenizer files: {', '.join(summary['tokenizer_files_copied'])}")
    click.echo(
        f"eos_token_id={EOS_TOKEN_ID} (GGUF tokenizer.ggml.eos_token_id was "
        f"{summary['gguf_tokenizer_eos']}; the HF family value is used)"
    )


if __name__ == "__main__":
    main()

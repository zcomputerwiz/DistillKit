"""Text-only Qwen3.5 with a dormant IQ4_NL sidecar at decoder layer index 1.

Load this class explicitly with a ``Qwen3_5TextConfig``; the stock backbone's
checkpoint names are unchanged. Only the small ``model.layers.1.sidecar`` modules
are added. The GGUF table belongs to the data collator, never to this model.

Raw rows travel through ordinary forward arguments. In particular, checkpoint
recomputation captures the original batch's rows, including when multiple forward
graphs are outstanding. There is no hook or mutable "current batch" attribute.
The backbone forward and output capture are inherited from Transformers.
"""

from __future__ import annotations

import torch
from torch import nn
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5DecoderLayer,
    Qwen3_5ForCausalLM,
    Qwen3_5PreTrainedModel,
    Qwen3_5RMSNorm,
    Qwen3_5TextModel,
    Qwen3_5TextRotaryEmbedding,
)

from distillkit.gated_residual import GatedResidual
from distillkit.linear_attention_dispatch import install_device_aware_linear_attention
from distillkit.ngram_table import IQ4NL_BLOCK, IQ4NL_KVALUES, IQ4NL_TYPE_SIZE, IQ4NLDequant

# flash-linear-attention, when installed, is bound by transformers at import time with
# no device check, and its Triton kernels reject CPU tensors. Restore per-call
# dispatch so CPU verification runs keep working. No-op without fla.
install_device_aware_linear_attention()


def _set_sidecar_defaults(config: Qwen3_5TextConfig) -> None:
    defaults = {
        "sidecar_layer_index": 1,
        "sidecar_num_heads": 16,
        "sidecar_head_dim": 160,
        "sidecar_num_branches": 4,
        "sidecar_gate_init_std": 0.02,
        "sidecar_per_channel_gate": True,
    }
    for name, value in defaults.items():
        if not hasattr(config, name):
            setattr(config, name, value)
    if not 0 <= config.sidecar_layer_index < config.num_hidden_layers:
        raise ValueError("sidecar_layer_index must identify an existing decoder layer")
    if config.sidecar_num_heads <= 0:
        raise ValueError("sidecar_num_heads must be positive")
    if config.sidecar_head_dim <= 0 or config.sidecar_head_dim % IQ4NL_BLOCK:
        raise ValueError("sidecar_head_dim must be a positive multiple of 32")
    if config.sidecar_num_branches < 1 or config.sidecar_gate_init_std < 0:
        raise ValueError("sidecar branches must be positive and gate init std nonnegative")


class _SidecarWeightInit:
    """Reapply special init for missing weights during HF's meta-device loading.

Constructor-only zeroing is insufficient: ``from_pretrained`` initializes missing
parameters after materializing them. Mark each linear independently so reloading
learned sidecar weights never resets a sibling that was present in the checkpoint.
"""

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        mode = getattr(module, "_sidecar_weight_init", None)
        if mode == "zero":
            nn.init.zeros_(module.weight)
        elif mode == "gate":
            nn.init.normal_(module.weight, std=module._sidecar_gate_init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, IQ4NLDequant):
            # Non-persistent buffers are also rematerialized empty by HF loading.
            module.kvalues.copy_(torch.tensor(IQ4NL_KVALUES, device=module.kvalues.device))


class _NGramSidecar(nn.Module):
    def __init__(self, config: Qwen3_5TextConfig):
        super().__init__()
        self.num_heads = config.sidecar_num_heads
        self.bytes_per_head = config.sidecar_head_dim // IQ4NL_BLOCK * IQ4NL_TYPE_SIZE
        self.dequant = IQ4NLDequant(out_dtype=torch.float32)
        self.W_side_proj = nn.Linear(
            config.sidecar_num_heads * config.sidecar_head_dim, config.hidden_size, bias=False
        )
        self.W_side_proj._sidecar_weight_init = "zero"
        self.gated_residual = GatedResidual(
            config.hidden_size,
            num_branches=config.sidecar_num_branches,
            gate_init_std=config.sidecar_gate_init_std,
            per_channel_gate=config.sidecar_per_channel_gate,
        )
        for branch in self.gated_residual.branches:
            branch._sidecar_weight_init = "zero"
        for gate in (self.gated_residual.W_x, self.gated_residual.W_h):
            if gate is not None:
                gate._sidecar_weight_init = "gate"
                gate._sidecar_gate_init_std = config.sidecar_gate_init_std

    def forward(self, hidden_states, ngram_raw, sidecar_enabled=True):
        if sidecar_enabled:
            expected = (*hidden_states.shape[:2], self.num_heads, self.bytes_per_head)
            if ngram_raw is None:
                raise ValueError("ngram_raw is required while sidecar_enabled=True")
            if ngram_raw.dtype != torch.uint8 or tuple(ngram_raw.shape) != expected:
                raise ValueError(f"ngram_raw must be uint8 with shape {expected}")
            raw = ngram_raw.to(device=hidden_states.device, non_blocking=True)
            features = self.dequant(raw).flatten(-2).to(dtype=hidden_states.dtype)
            hidden_states = hidden_states + self.W_side_proj(features)
        # The control arm retains exactly this same GR, with the unmodified stream
        # as its input. Branches get gradients immediately even though W_side_proj
        # is zero; gate gradients start once the branch weights leave zero.
        return self.gated_residual(hidden_states)


class _SidecarDecoderLayer(Qwen3_5DecoderLayer):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.sidecar = _NGramSidecar(config) if layer_idx == config.sidecar_layer_index else None

    def forward(self, hidden_states, *args, ngram_raw=None, sidecar_enabled=True, **kwargs):
        if self.sidecar is not None:
            hidden_states = self.sidecar(hidden_states, ngram_raw, sidecar_enabled)
        return super().forward(hidden_states, *args, **kwargs)


class _SidecarTextModel(_SidecarWeightInit, Qwen3_5TextModel):
    def __init__(self, config):
        # Same construction as Qwen3_5TextModel, with decoder subclasses that consume
        # the sidecar kwargs before they can reach an attention implementation.
        Qwen3_5PreTrainedModel.__init__(self, config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList(
            [_SidecarDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3_5TextRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.post_init()


class Qwen35SidecarForCausalLM(_SidecarWeightInit, Qwen3_5ForCausalLM):
    """Stock-compatible text decoder with explicit per-batch ``ngram_raw`` input.

``sidecar_enabled=False`` is the table ablation; the trainable gated residual is
still active. The config stores architecture dimensions but no machine-specific
table path. Use this class explicitly when reloading a saved sidecar checkpoint.
"""

    config_class = Qwen3_5TextConfig
    _no_split_modules = ["_SidecarDecoderLayer"]

    @classmethod
    def is_custom_code(cls) -> bool:
        # First-party variant of the stock qwen3_5_text architecture, not hub custom code.
        # The default (module name does not start with "transformers.") would make
        # transformers skip this model in the weight-conversion pipeline, dropping the
        # qwen3_5_text remap that loads VLM-prefixed checkpoints (model.language_model.*
        # -> model.*). Its state dict is stock layout plus sidecar additions, so the
        # stock conversions apply exactly.
        return False

    def __init__(self, config: Qwen3_5TextConfig):
        if not isinstance(config, Qwen3_5TextConfig):
            raise TypeError("Qwen35SidecarForCausalLM requires Qwen3_5TextConfig, not a VLM config")
        _set_sidecar_defaults(config)
        Qwen3_5PreTrainedModel.__init__(self, config)
        self.model = _SidecarTextModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        logits_to_keep=0,
        ngram_raw=None,
        sidecar_enabled=True,
        output_hidden_states=None,
        **kwargs,
    ):
        if sidecar_enabled and ngram_raw is None:
            raise ValueError("ngram_raw is required; use SidecarDataCollator or sidecar_enabled=False")
        if output_hidden_states is not None:
            kwargs["output_hidden_states"] = output_hidden_states
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            logits_to_keep=logits_to_keep,
            ngram_raw=ngram_raw,
            sidecar_enabled=sidecar_enabled,
            **kwargs,
        )

    def stage1_parameter_names(self) -> list[str]:
        prefix = f"model.layers.{self.config.sidecar_layer_index}.sidecar."
        return [
            name for name, _ in self.named_parameters()
            if name.startswith(prefix) or name.startswith("distillation_projections.")
        ]

    def freeze_backbone(self) -> None:
        names = set(self.stage1_parameter_names())
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name in names)
        # Reentrant checkpointing otherwise sees frozen embedding outputs and drops
        # the graph before reaching our trainable layer. This hook only supplies
        # differentiable embeddings, never any sidecar data or batch state.
        if not hasattr(self, "_require_grads_hook"):
            self.enable_input_require_grads()
            self._sidecar_enabled_input_grads = True

    def unfreeze_backbone(self) -> None:
        self.requires_grad_(True)
        if getattr(self, "_sidecar_enabled_input_grads", False):
            self.disable_input_require_grads()
            self._sidecar_enabled_input_grads = False

    def disable_sidecar_projection(self) -> None:
        """Freeze W_side_proj for the control arm (``sidecar_enabled=False`` runs).

        The projection is bypassed in forward when the sidecar is disabled, so a
        trainable copy would be an unused parameter that DDP's
        ``find_unused_parameters=False`` rejects on the next reduction. Freezing it
        before optimizer/distributed setup avoids that; the gated residual remains
        trainable.
        """
        self.model.layers[self.config.sidecar_layer_index].sidecar.W_side_proj.requires_grad_(False)

    @torch.no_grad()
    def gate_report(self, prefix="sidecar") -> dict[str, float]:
        sidecar = self.model.layers[self.config.sidecar_layer_index].sidecar
        report = sidecar.gated_residual.gate_report(f"{prefix}/gated_residual")
        report[f"{prefix}/W_side_proj_norm"] = sidecar.W_side_proj.weight.float().norm().item()
        return report

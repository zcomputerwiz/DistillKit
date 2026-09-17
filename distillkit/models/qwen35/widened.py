"""Qwen3.5 with identity-initialized residual branches at every subblock.

Stock parameter names, attention/MLP dimensions and norms are preserved. Routing
and branches stay on the TP home device; sharded blocks still communicate d-wide
inputs/outputs. The branch mean is exposed to the head and hidden-state losses.
"""
from __future__ import annotations

import torch
from torch import nn
from transformers import initialization as init
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_causal_mask, create_recurrent_attention_mask
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5DecoderLayer, Qwen3_5ForCausalLM, Qwen3_5PreTrainedModel,
    Qwen3_5RMSNorm, Qwen3_5TextModel, Qwen3_5TextRotaryEmbedding,
    Qwen3_5ModelOutputWithPast,
)
from distillkit.models.qwen35.sidecar import (
    _SidecarWeightInit, _set_sidecar_defaults, _build_sidecar,
)
from distillkit.experimental import (
    HyperConnection,
    WidenedResidual,
    collapse_residual,
    offload_stream_boundaries,
)


class _WidenedWeightInit(_SidecarWeightInit):
    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, HyperConnection):
            init.zeros_(module.branch_gain_delta)
            init.constant_(module.blend, module.initial_blend)
        if isinstance(module, WidenedResidual):
            # HF marks loaded tensors individually; its init helpers preserve them
            # even when a sibling tensor is absent from a partial checkpoint.
            init.zeros_(module.read_offset)
            init.zeros_(module.write_offset)
            init.zeros_(module.lambda_read)
            init.zeros_(module.lambda_write)
            init.zeros_(module.branch_gain_delta)


class WidenedDecoderLayer(Qwen3_5DecoderLayer):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        options = dict(hidden_size=config.hidden_size,
                       num_branches=config.residual_stream_num_branches,
                       lowrank=config.residual_stream_lowrank, layer_idx=layer_idx)
        routing = getattr(config, "residual_stream_routing", "widened")
        if routing not in ("widened", "flash_next"):
            raise ValueError(f"Unknown residual stream routing: {routing}")
        route = WidenedResidual
        if routing == "flash_next":
            route = HyperConnection
            options.update(blend=getattr(config, "residual_stream_blend", 0.0),
                           norm_eps=config.rms_norm_eps,
                           learnable_blend=getattr(
                               config, "residual_stream_learnable_blend", False))
        self.attn_residual = route(**options)
        self.mlp_residual = route(**options)
        # A converted checkpoint carries its conversion in the weights; the config only
        # says which mode produced them. Restoring the provenance here, rather than
        # re-running `recipient_initialize`, is what keeps a reload from overwriting
        # trained gains with the initialization they started from.
        mode = getattr(config, "residual_stream_recipient_mode", None)
        if mode is not None and routing == "flash_next":
            for module in (self.attn_residual, self.mlp_residual):
                module.recipient_initialized = True
                module.recipient_mode = mode
        self.sidecar = (_build_sidecar(config) if config.residual_stream_sidecar
                        and layer_idx == config.sidecar_layer_index else None)

    @staticmethod
    def distillation_hidden_state(output):
        return collapse_residual(output)

    def forward(self, hidden_states, position_embeddings, attention_mask=None,
                position_ids=None, past_key_values=None, ngram_raw=None,
                sidecar_enabled=True, **kwargs):
        if self.sidecar is not None:
            if getattr(self.sidecar, "reads_widened_stream", False):
                # Per-stream admission over one shared value: the gate is a readout of
                # the branch it admits into, so the branches have to arrive together.
                hidden_states = self.sidecar(hidden_states, ngram_raw, sidecar_enabled)
            else:
                # Applying the original sidecar independently preserves its trained
                # behavior and exact arithmetic while allowing divergent branches.
                hidden_states = torch.stack([
                    self.sidecar(branch.contiguous(), ngram_raw, sidecar_enabled)
                    for branch in hidden_states.unbind(-2)
                ], dim=-2)
        block_input, weights = self.attn_residual.read(hidden_states, self.input_layernorm)
        if self.block_type == "linear_attention":
            output = self.linear_attn(hidden_states=block_input,
                                      cache_params=past_key_values,
                                      attention_mask=attention_mask, **kwargs)
        else:
            output, _ = self.self_attn(
                hidden_states=block_input, attention_mask=attention_mask,
                position_ids=position_ids, past_key_values=past_key_values,
                position_embeddings=position_embeddings, **kwargs)
        hidden_states = self.attn_residual.write(hidden_states, output, weights)
        block_input, weights = self.mlp_residual.read(
            hidden_states, self.post_attention_layernorm)
        return self.mlp_residual.write(hidden_states, self.mlp(block_input), weights)


class _WidenedTextModel(_WidenedWeightInit, Qwen3_5TextModel):
    def __init__(self, config):
        Qwen3_5PreTrainedModel.__init__(self, config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList([WidenedDecoderLayer(config, i)
                                     for i in range(config.num_hidden_layers)])
        self.norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3_5TextRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.post_init()

    def _stream_offload_device(self):
        """The peer card, only while the stream is actually being stored for backward."""
        devices = getattr(self, "_distillkit_tp_devices", None)
        if not devices or not (self.gradient_checkpointing and self.training):
            return None
        return devices[1]

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, use_cache=None,
                output_hidden_states=None, output_attentions=None, **kwargs):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")
        if output_attentions:
            raise ValueError("Widened residual model does not expose attention weights")
        use_cache = self.config.use_cache if use_cache is None else use_cache
        output_hidden_states = (getattr(self.config, "output_hidden_states", False)
                                if output_hidden_states is None else output_hidden_states)
        if self.gradient_checkpointing and self.training:
            use_cache = False
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        if position_ids is None:
            seen = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + seen
            position_ids = position_ids.view(1, 1, -1).expand(4, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(4, position_ids.shape[0], -1)
        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids, position_ids = position_ids[0], position_ids[1:]
        else:
            text_position_ids = None
        if isinstance(attention_mask, dict):
            masks = attention_mask
        else:
            options = dict(config=self.config, inputs_embeds=inputs_embeds,
                           attention_mask=attention_mask, past_key_values=past_key_values,
                           position_ids=text_position_ids)
            masks = {"full_attention": create_causal_mask(**options),
                     "linear_attention": create_recurrent_attention_mask(**options)}
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)
        states = inputs_embeds.unsqueeze(-2).expand(
            *inputs_embeds.shape[:-1], self.config.residual_stream_num_branches,
            inputs_embeds.shape[-1])
        captured = (inputs_embeds,) if output_hidden_states else None
        with offload_stream_boundaries(self._stream_offload_device()):
            for index, layer in enumerate(self.layers):
                states = layer(states, position_embeddings=position_embeddings,
                               attention_mask=masks[self.config.layer_types[index]],
                               position_ids=text_position_ids, past_key_values=past_key_values,
                               use_cache=use_cache, **kwargs)
                if output_hidden_states and index < len(self.layers) - 1:
                    captured += (collapse_residual(states),)
        hidden_states = self.norm(collapse_residual(states).contiguous())
        if output_hidden_states:
            captured += (hidden_states,)
        return Qwen3_5ModelOutputWithPast(last_hidden_state=hidden_states,
                                        past_key_values=past_key_values,
                                        hidden_states=captured)


class Qwen35WidenedForCausalLM(_WidenedWeightInit, Qwen3_5ForCausalLM):
    """Load explicitly for widened checkpoints; config records all architecture data."""
    config_class = Qwen3_5TextConfig
    _no_split_modules = ["WidenedDecoderLayer"]
    # The branch gains are held at fp32 in memory, and the loader would undo that. A
    # converted route stores `1 + 2 * weight`, near 3, where bf16 spacing is 1.6e-2 --
    # `from_pretrained(dtype=bfloat16)` casts every loaded tensor on the way in, so a
    # checkpoint written correctly at fp32 came back rounded and the reloaded model
    # disagreed with the one that saved it. Strict, because bf16 is the deployed
    # precision here and the plain flag only fires for fp16.
    _keep_in_fp32_modules_strict = ["branch_gain_delta"]

    @classmethod
    def is_custom_code(cls):
        return False

    def __init__(self, config):
        if not isinstance(config, Qwen3_5TextConfig):
            raise TypeError("Qwen35WidenedForCausalLM requires Qwen3_5TextConfig")
        for key, value in {"residual_stream_enabled": True,
                           "residual_stream_num_branches": 2,
                           "residual_stream_lowrank": 64,
                           "residual_stream_sidecar": False}.items():
            if not hasattr(config, key):
                setattr(config, key, value)
        if not config.residual_stream_enabled:
            raise ValueError("Use the stock/sidecar model class when residual_stream is disabled")
        if config.residual_stream_sidecar:
            _set_sidecar_defaults(config)
        Qwen3_5PreTrainedModel.__init__(self, config)
        self.model = _WidenedTextModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, labels=None,
                use_cache=None, logits_to_keep=0, ngram_raw=None,
                sidecar_enabled=True, output_hidden_states=None, **kwargs):
        if self.config.residual_stream_sidecar and sidecar_enabled and ngram_raw is None:
            raise ValueError("ngram_raw is required while sidecar_enabled=True")
        return super().forward(
            input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
            past_key_values=past_key_values, inputs_embeds=inputs_embeds, labels=labels,
            use_cache=use_cache, logits_to_keep=logits_to_keep, ngram_raw=ngram_raw,
            sidecar_enabled=sidecar_enabled, output_hidden_states=output_hidden_states, **kwargs)

    #: Decoder layers that stage 1 trains alongside the adapter, as [start, stop).
    #: Empty by default: stage 1 exists to train the adapter against a fixed backbone.
    stage1_trainable_layers: tuple[int, int] | None = None

    def set_stage1_trainable_layers(self, window):
        """Open a contiguous decoder window for stage-1 training.

        Naming these through `stage1_parameter_names` rather than unfreezing them
        afterwards is what keeps the rest of the machinery consistent: the same list
        feeds `freeze_backbone_for_stage1` and `_auxiliary_parameter_ids`, so the window
        stays trainable *and* routes to AdamW rather than Muon -- which it has to, since
        Newton-Schulz does not commute with tensor-parallel slicing and
        `_refuse_trainable_muon_shards` rejects the combination outright.
        """
        if window is None:
            self.stage1_trainable_layers = None
            return
        start, stop = (int(value) for value in window)
        depth = self.config.num_hidden_layers
        if not 0 <= start < stop <= depth:
            raise ValueError(
                f"stage1 trainable window [{start}, {stop}) outside 0..{depth}")
        self.stage1_trainable_layers = (start, stop)

    def _in_stage1_window(self, name):
        window = self.stage1_trainable_layers
        if window is None:
            return False
        parts = name.split(".")
        if "layers" not in parts:
            return False
        index = parts.index("layers") + 1
        return index < len(parts) and parts[index].isdigit() and (
            window[0] <= int(parts[index]) < window[1])

    def stage1_parameter_names(self):
        return [name for name, _ in self.named_parameters()
                if self._in_stage1_window(name) or any(
                    part in name.split(".") for part in
                    ("attn_residual", "mlp_residual", "sidecar", "distillation_projections"))]

    def freeze_backbone(self, input_require_grads: bool = True):
        names = set(self.stage1_parameter_names())
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name in names)
        if input_require_grads and not hasattr(self, "_require_grads_hook"):
            self.enable_input_require_grads()

    def unfreeze_backbone(self):
        self.requires_grad_(True)
        if hasattr(self, "_require_grads_hook"):
            self.disable_input_require_grads()

    def disable_sidecar_projection(self):
        sidecar = self.model.layers[self.config.sidecar_layer_index].sidecar
        sidecar.W_side_proj.requires_grad_(False)

    @torch.no_grad()
    def recipient_initialize(self, asymmetric=False, seed=0, epsilon=None):
        """Convert every sublayer to the four-stream route without changing the model.

        Each route is initialized against the norm it replaces, so the converted model
        computes what the recipient computed with the route fully active at ``blend=1``.
        ``W_down`` is seeded per sublayer rather than once: a single seed would give
        every sublayer the same bottleneck, which is the one thing the read is supposed
        to be able to specialize.

        This is a post-construction step on purpose. It is never called from
        ``__init__``, so loading a converted checkpoint restores trained gains instead
        of overwriting them -- reload only restores the recorded mode. Converting a
        model that already records a conversion is refused for the same reason.
        """
        config = self.config
        if getattr(config, "residual_stream_routing", "widened") != "flash_next":
            raise ValueError("recipient initialization applies to the flash_next route; "
                             "this model uses %r"
                             % getattr(config, "residual_stream_routing", "widened"))
        recorded = getattr(config, "residual_stream_recipient_mode", None)
        if recorded is not None:
            raise ValueError(
                "this model already records a %r conversion. Re-running the "
                "initialization would overwrite whatever it has learned since."
                % recorded)
        records = []
        for index, layer in enumerate(self.model.layers):
            for offset, (kind, norm) in enumerate(
                    (("attn_residual", layer.input_layernorm),
                     ("mlp_residual", layer.post_attention_layernorm))):
                records.append(getattr(layer, kind).recipient_initialize(
                    norm, asymmetric=asymmetric, seed=int(seed) + 2 * index + offset,
                    epsilon=epsilon))
        if not records:
            raise ValueError("model has no layers to convert")
        config.residual_stream_recipient_mode = records[0]["mode"]
        config.residual_stream_recipient_seed = int(seed)
        config.residual_stream_recipient_epsilon = records[0]["epsilon"]
        config.residual_stream_blend = 1.0
        return records

    @torch.no_grad()
    def gate_report(self, prefix="residual_stream"):
        report = {}
        for index, layer in enumerate(self.model.layers):
            for kind in ("attn_residual", "mlp_residual"):
                report.update(getattr(layer, kind).gate_report(f"{prefix}/{index}/{kind}"))
        return report

# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team.
# Copyright 2025 ByteDance Seed. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
import logging
from typing import Optional

import torch
from torch import nn

from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_utils import PreTrainedModel

from .config import OuroMRIConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reused Ouro building blocks (vendored to avoid coupling to ouro package)
# ---------------------------------------------------------------------------

class OuroRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class OuroMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str = "silu"):
        super().__init__()
        from transformers.activations import ACT2FN
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


class UniversalTransformerCache(Cache):
    """2D-indexed KV cache for Ouro UT loop: index = ut_step * num_layers + layer_idx."""

    def __init__(self, max_cache_size: Optional[int] = None):
        self.key_cache: list[Optional[torch.Tensor]] = []
        self.value_cache: list[Optional[torch.Tensor]] = []
        self._seen_tokens = 0
        self.max_cache_size = max_cache_size

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        while len(self.key_cache) <= layer_idx:
            self.key_cache.append(None)
            self.value_cache.append(None)

        cached_key = self.key_cache[layer_idx]
        cached_value = self.value_cache[layer_idx]

        if cached_key is None:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = torch.cat([cached_key, key_states], dim=2)
            self.value_cache[layer_idx] = torch.cat([cached_value, value_states], dim=2)

        result_key = self.key_cache[layer_idx]
        result_value = self.value_cache[layer_idx]
        assert result_key is not None and result_value is not None
        self._seen_tokens = result_key.shape[2]
        return result_key, result_value

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        if layer_idx is None:
            layer_idx = 0
        if layer_idx < 0 or len(self.key_cache) <= layer_idx:
            return 0
        cached = self.key_cache[layer_idx]
        return 0 if cached is None else cached.shape[2]


class OuroAttention(nn.Module):
    """Multi-headed self-attention with RoPE and KV cache support."""

    def __init__(self, config: OuroMRIConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = config.attention_dropout

        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=False,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=False,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=False,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[UniversalTransformerCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        current_ut: int = 0,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            num_layers = self.config.num_hidden_layers
            cache_idx = current_ut * num_layers + self.layer_idx
            key_states, value_states = past_key_value.update(
                key_states, value_states, cache_idx
            )

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Scaled dot-product attention with optional mask
        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) * self.scaling

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask[:, :, :, : key_states.shape[-2]]

        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class OuroDecoderLayer(nn.Module):
    """Sandwich RMSNorm decoder layer matching Ouro's pattern."""

    def __init__(self, config: OuroMRIConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = OuroAttention(config=config, layer_idx=layer_idx)
        self.mlp = OuroMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = OuroRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm_2 = OuroRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = OuroRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm_2 = OuroRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        past_key_value: Optional[UniversalTransformerCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        current_ut: int = 0,
        **kwargs,
    ) -> torch.Tensor:
        # Sandwich norm: norm → attn → norm → residual → norm → mlp → norm → residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            past_key_value=past_key_value,
            cache_position=cache_position,
            current_ut=current_ut,
        )
        hidden_states = self.input_layernorm_2(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_attention_layernorm_2(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class OuroRotaryEmbedding(nn.Module):
    """1D RoPE position embeddings for the backbone."""

    def __init__(self, config: OuroMRIConfig, device=None):
        super().__init__()
        self.max_seq_len_cached = 4096  # sufficient for image patches
        self.rope_theta = config.rope_theta
        dim = config.hidden_size // config.num_attention_heads
        inv_freq = 1.0 / (
            self.rope_theta ** (torch.arange(0, dim, 2, device=device).float() / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq_expanded = (
            self.inv_freq[None, :, None]
            .float()
            .expand(position_ids.shape[0], -1, 1)
            .to(x.device)
        )
        position_ids_expanded = position_ids[:, None, :].float()

        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# ---------------------------------------------------------------------------
# OuroImageBackbone
# ---------------------------------------------------------------------------

class OuroImageBackbone(PreTrainedModel):
    """
    Ouro UT loop backbone adapted for image-to-image translation.

    Key differences from OuroModel:
    - No embed_tokens — receives continuous patch embeddings from VE.
    - Bidirectional (full) attention instead of causal masking.
    - Preserves the UT loop + early exit gate mechanism exactly.

    Reference: src/mobius/ouro/modeling_ouro.py:481-599
    """

    config_class = OuroMRIConfig
    supports_gradient_checkpointing = True
    _no_split_modules = ["OuroDecoderLayer"]

    def __init__(self, config: OuroMRIConfig):
        super().__init__(config)
        self.hidden_size = config.hidden_size
        self.num_hidden_layers = config.num_hidden_layers
        self.total_ut_steps = config.total_ut_steps

        self.layers = nn.ModuleList([
            OuroDecoderLayer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = OuroRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = OuroRotaryEmbedding(config=config)
        self.early_exit_gate = nn.Linear(config.hidden_size, 1)
        self.gradient_checkpointing = False
        self.post_init()

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[UniversalTransformerCache] = None,
        use_cache: bool = False,
    ) -> tuple[BaseModelOutputWithPast, list[torch.Tensor], list[torch.Tensor]]:
        """
        Args:
            inputs_embeds: [B, seq_len, hidden_size] — concatenated source + target embeddings
            attention_mask: optional [1, 1, seq_len, seq_len] float mask (0=attend, -inf=mask)
            past_key_values: optional UniversalTransformerCache for KV caching
            use_cache: whether to populate past_key_values

        Returns:
            (outputs, hidden_states_list, gate_list) where:
            - outputs: BaseModelOutputWithPast with last_hidden_state
            - hidden_states_list: list of hidden states per UT step
            - gate_list: list of early_exit_gate outputs per UT step
        """
        B, seq_len, C = inputs_embeds.shape
        device = inputs_embeds.device

        # Position IDs: sequential positions across the concatenated sequence
        past_seen_tokens = (
            past_key_values.get_seq_length() if past_key_values is not None else 0
        )
        cache_position = torch.arange(
            past_seen_tokens,
            past_seen_tokens + seq_len,
            device=device,
        )
        position_ids = cache_position.unsqueeze(0)

        if use_cache and past_key_values is None:
            max_cache_size = self.num_hidden_layers * self.total_ut_steps
            past_key_values = UniversalTransformerCache(max_cache_size)

        position_embeddings = self.rotary_emb(inputs_embeds.float(), position_ids)

        hidden_states = inputs_embeds
        hidden_states_list: list[torch.Tensor] = []
        gate_list: list[torch.Tensor] = []

        for current_ut in range(self.total_ut_steps):
            for decoder_layer in self.layers:
                hidden_states = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_embeddings=position_embeddings,
                    past_key_value=past_key_values,
                    cache_position=cache_position,
                    current_ut=current_ut,
                )

            hidden_states = self.norm(hidden_states)
            hidden_states_list.append(hidden_states)
            gate_list.append(self.early_exit_gate(hidden_states))

        outputs = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )
        return outputs, hidden_states_list, gate_list


__all__ = ["OuroImageBackbone"]

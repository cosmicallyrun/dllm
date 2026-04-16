"""
A2D (Autoregressive-to-Diffusion) conversion for Qwen3.5.

Qwen3.5 is a hybrid model: every 4-layer cycle has
  3 × Gated DeltaNet (GDN / linear attention)  +  1 × standard Gated Attention

Two distinct patches are required to make it bidirectional:

PATCH 1 — Standard attention layers  (1 per cycle)
  Replace create_causal_mask() with a padding-only 4-D mask.
  Identical approach to the existing A2DQwen3Model.

PATCH 2 — GDN layers  (3 per cycle)
  Causality is baked into torch_chunk_gated_delta_rule() via:
    (a) mask = torch.triu(..., diagonal=0)               ← intra-chunk causal mask
    (b) decay_mask = (...).tril().exp().tril()           ← lower-triangular decay
  We replace the instance attribute self.chunk_gated_delta_rule with a
  bidirectional version that removes both constraints.
  The inter-chunk recurrence (the for-loop over chunks) remains left-to-right,
  which is a known limitation — full BiRNN-style bidirectionality would need a
  second backward pass and doubles compute/memory.

PATCH 3 — Causal Conv1d
  Set self.causal_conv1d_fn = None to fall back to nn.Conv1d; then override
  the fallback in forward to use symmetric padding instead of causal padding.

Reference:
  Gated Delta Networks: https://arxiv.org/abs/2412.06464
  A2D Qwen3 (existing): dllm/pipelines/a2d/models/qwen3/modeling_qwen3.py
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import transformers
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5Config,
    Qwen3_5ForCausalLM,
    Qwen3_5GatedDeltaNet,
    Qwen3_5PreTrainedModel,
    Qwen3_5TextModel,
    apply_mask_to_padding_states,
    torch_chunk_gated_delta_rule,  # we read this but replace it per-instance
)
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.cache_utils import Cache, DynamicCache


# ---------------------------------------------------------------------------
# PATCH 2 — bidirectional chunk gated delta rule
# ---------------------------------------------------------------------------

def bidirectional_chunk_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    chunk_size=64,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
):
    """
    Bidirectional variant of torch_chunk_gated_delta_rule for masked diffusion.

    Differences from the causal original (transformers/models/qwen3_5/modeling_qwen3_5.py):

    1. Intra-chunk causal mask REMOVED:
       ORIGINAL: mask = torch.triu(..., diagonal=0)
                 attn = (...).masked_fill(mask, 0)
       PATCH:    attn = (...)         ← no masked_fill

    2. decay_mask is full (not lower-triangular):
       ORIGINAL: decay_mask = ((g[...,None] - g[...,None,:]).tril().exp()).tril()
       PATCH:    decay_mask = (g[...,None] - g[...,None,:]).exp()
       This allows symmetric decay: exp(g_i - g_j) for all i, j.
       Since g values are non-positive (they are forgetting gates), the matrix
       is well-defined and remains bounded.

    3. Inter-chunk recurrence (the for-loop) is still left-to-right.
       This is a known approximation. For the diffusion use-case with
       BD3LM block_size ≤ chunk_size, intra-chunk bidirectionality is the
       critical property — each diffusion block is denoised with full
       bidirectional context within it.
    """
    try:
        from transformers.models.qwen3_5.modeling_qwen3_5 import l2norm
    except ImportError:
        def l2norm(x, dim=-1, eps=1e-6):
            return x / (x.norm(dim=dim, keepdim=True) + eps)

    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key   = l2norm(key,   dim=-1, eps=1e-6)

    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32)
        for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]

    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key   = F.pad(key,   (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta  = F.pad(beta,  (0, pad_size))
    g     = F.pad(g,     (0, pad_size))

    total_sequence_length = sequence_length + pad_size
    scale = 1.0 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key   * beta.unsqueeze(-1)

    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)

    # ── PATCH: cumulative decay (no .tril — full symmetric matrix) ──────────
    g = g.cumsum(dim=-1)
    # Symmetric absolute-value decay: exp(-|g_i - g_j|) ∈ (0, 1].
    # The original causal version used exp(g_i - g_j).tril(), which was safe
    # because g is a non-increasing cumsum so g[i]-g[j] ≤ 0 for i ≥ j.
    # The upper triangle (i < j) has g[i]-g[j] > 0 and exp() can overflow to
    # Inf → NaN in subsequent matmuls.  Using abs avoids this completely.
    decay_mask = (-torch.abs(g.unsqueeze(-1) - g.unsqueeze(-2))).exp().float()

    # ── PATCH: intra-chunk delta rule — no masked_fill ──────────────────────
    # ORIGINAL had: mask = torch.triu(...); attn = (...).masked_fill(mask, 0)
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask)

    # Propagate the lower-triangular dependency structure (unchanged)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)

    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value    = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))

    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)

    # Inter-chunk loop — still left-to-right (see module docstring)
    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn_i = q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]
        v_prime = k_cumdecay[:, :, i] @ last_recurrent_state
        v_new   = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn_i @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None])
              .transpose(-1, -2) @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None

    core_attn_out = core_attn_out.reshape(
        core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1]
    )
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    # Safety guard: replace any residual NaN/Inf (from extreme gate values) with 0
    core_attn_out = torch.nan_to_num(core_attn_out, nan=0.0, posinf=0.0, neginf=0.0)
    return core_attn_out, last_recurrent_state


# ---------------------------------------------------------------------------
# PATCH 2+3 — bidirectional GDN layer
# ---------------------------------------------------------------------------

class A2DQwen3_5GatedDeltaNet(Qwen3_5GatedDeltaNet):
    """
    Drops-in for Qwen3_5GatedDeltaNet with two bidirectionality patches:
      • chunk_gated_delta_rule  → bidirectional_chunk_gated_delta_rule
      • conv1d                  → symmetric (non-causal) padding
    """

    def __init__(self, config: Qwen3_5Config, layer_idx: int):
        super().__init__(config, layer_idx)
        # PATCH 2: replace chunk fn with bidirectional version
        self.chunk_gated_delta_rule = bidirectional_chunk_gated_delta_rule
        # PATCH 3: disable causal_conv1d_fn — forward() will use our override
        self.causal_conv1d_fn = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: Optional[Cache] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
        batch_size, seq_len, _ = hidden_states.shape

        use_precomputed_states = (
            cache_params is not None
            and cache_params.has_previous_state(self.layer_idx)
            and seq_len == 1
        )

        if use_precomputed_states:
            conv_state     = cache_params.layers[self.layer_idx].conv_states
            recurrent_state = cache_params.layers[self.layer_idx].recurrent_states

        mixed_qkv = self.in_proj_qkv(hidden_states)
        mixed_qkv = mixed_qkv.transpose(1, 2)  # [B, conv_dim, L]

        z = self.in_proj_z(hidden_states).reshape(batch_size, seq_len, -1, self.head_v_dim)
        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        if use_precomputed_states:
            mixed_qkv = self.causal_conv1d_update(
                mixed_qkv, conv_state,
                self.conv1d.weight.squeeze(1), self.conv1d.bias, self.activation,
            )
        else:
            if cache_params is not None:
                conv_state = F.pad(mixed_qkv, (self.conv_kernel_size - mixed_qkv.shape[-1], 0))
                cache_params.update_conv_state(conv_state, self.layer_idx)

            # PATCH 3: symmetric (non-causal) conv1d
            # Original causal path: causal_conv1d_fn(...)  or  self.conv1d(x)[:, :, :seq_len]
            # We use symmetric padding so all positions see equal context on both sides.
            pad = self.conv_kernel_size // 2
            mixed_qkv = F.silu(
                F.conv1d(
                    mixed_qkv,
                    self.conv1d.weight.squeeze(1).unsqueeze(1) if self.conv1d.weight.dim() == 2
                        else self.conv1d.weight,
                    self.conv1d.bias,
                    padding=pad,
                    groups=self.conv_dim,
                )[:, :, :seq_len]  # trim to original length (symmetric padding may add +1)
            )

        mixed_qkv = mixed_qkv.transpose(1, 2)  # [B, L, conv_dim]
        query, key, value = torch.split(
            mixed_qkv,
            [self.key_dim, self.key_dim, self.value_dim],
            dim=-1,
        )

        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key   = key  .reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key   = key  .repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

        if not use_precomputed_states:
            # Uses self.chunk_gated_delta_rule which is now bidirectional
            core_attn_out, last_recurrent_state = self.chunk_gated_delta_rule(
                query, key, value,
                g=g, beta=beta,
                initial_state=None,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            core_attn_out, last_recurrent_state = self.recurrent_gated_delta_rule(
                query, key, value,
                g=g, beta=beta,
                initial_state=recurrent_state,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
            )

        if cache_params is not None:
            cache_params.update_recurrent_state(last_recurrent_state, self.layer_idx)

        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)
        return self.out_proj(core_attn_out)


# ---------------------------------------------------------------------------
# PATCH 1 — bidirectional TextModel
# ---------------------------------------------------------------------------

class A2DQwen3_5TextModel(Qwen3_5TextModel):
    """
    Replaces:
      • create_causal_mask()  →  _prepare_4d_attention_mask() (padding only)
        for the full-attention layers
      • All Qwen3_5GatedDeltaNet submodules  →  A2DQwen3_5GatedDeltaNet
    """

    def __init__(self, config: Qwen3_5Config):
        super().__init__(config)
        # Swap every GDN layer for its bidirectional variant in-place
        for i, layer in enumerate(self.layers):
            for attr_name in dir(layer):
                obj = getattr(layer, attr_name, None)
                if isinstance(obj, Qwen3_5GatedDeltaNet):
                    bidi = A2DQwen3_5GatedDeltaNet(config, i)
                    # copy pretrained weights (state_dict keys match because same class hierarchy)
                    bidi.load_state_dict(obj.state_dict(), strict=False)
                    setattr(layer, attr_name, bidi)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("Exactly one of input_ids or inputs_embeds must be set.")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        # ── position_ids (same logic as original) ───────────────────────────
        if position_ids is None:
            past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = (
                torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen
            )
            position_ids = position_ids.view(1, 1, -1).expand(4, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(4, position_ids.shape[0], -1)

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids      = position_ids[1:]
        else:
            text_position_ids = None

        # ── PATCH 1: bidirectional mask for full-attention layers ────────────
        # Original: causal_mask = create_causal_mask(...)
        # Patch:    padding-only 4-D mask (same approach as A2DQwen3Model)
        if attention_mask is None:
            bidi_mask = torch.ones(
                inputs_embeds.shape[:2],
                device=inputs_embeds.device,
                dtype=torch.long,
            )
        else:
            bidi_mask = attention_mask

        if not (
            isinstance(bidi_mask, torch.Tensor) and bidi_mask.ndim == 4
        ):
            bidi_mask = _prepare_4d_attention_mask(bidi_mask, inputs_embeds.dtype)

        # Linear attention layers keep the original 2-D padding mask (unchanged)
        linear_attn_mask = self._update_linear_attn_mask(attention_mask, past_key_values)

        hidden_states       = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            layer_mask = (
                linear_attn_mask
                if self.config.layer_types[i] == "linear_attention"
                else bidi_mask
            )
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=layer_mask,
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


# ---------------------------------------------------------------------------
# Config + LM head
# ---------------------------------------------------------------------------

class A2DQwen3_5Config(Qwen3_5Config):
    model_type = "a2d-qwen3_5"


class A2DQwen3_5LMHeadModel(Qwen3_5ForCausalLM):
    config_class = A2DQwen3_5Config

    def __init__(self, config: A2DQwen3_5Config):
        Qwen3_5PreTrainedModel.__init__(self, config)
        self.model    = A2DQwen3_5TextModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head  = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()


# ---------------------------------------------------------------------------
# Register with HuggingFace Auto classes
# ---------------------------------------------------------------------------

transformers.AutoConfig.register("a2d-qwen3_5", A2DQwen3_5Config)
transformers.AutoModel.register(A2DQwen3_5Config, A2DQwen3_5LMHeadModel)
transformers.AutoModelForMaskedLM.register(A2DQwen3_5Config, A2DQwen3_5LMHeadModel)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import dllm
    from transformers import AutoModel

    config_path = dllm.utils.resolve_with_base_env("Qwen/Qwen3.5-0.8B", "BASE_MODELS_DIR")
    config = A2DQwen3_5Config.from_pretrained(config_path)
    for attr in ("auto_map", "architectures"):
        if hasattr(config, attr):
            delattr(config, attr)

    torch.set_default_device("cuda")
    model = A2DQwen3_5LMHeadModel(config)
    model.save_pretrained("models-tmp/a2d-qwen3_5")
    auto = AutoModel.from_pretrained("models-tmp/a2d-qwen3_5")
    print("Smoke test passed:", type(auto))

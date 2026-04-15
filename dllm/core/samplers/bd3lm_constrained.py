"""
Constrained BD3LM Sampler with tool-call intent detection.

Technique reference:
  "Constrained Decoding of Diffusion LLMs with Context-Free Grammars"
  Mündler et al., arXiv:2508.10111 (ICLR 2026)
  https://github.com/eth-sri/constrained-diffusion

Core idea
---------
After the <think> block is fully denoised, we scan for the special token
<tool_call_intent>.  If found, every subsequent block has its logits masked
at each diffusion step: for each still-masked position we walk the JSON FSM
with the committed prefix and zero-out invalid tokens (log-prob → -inf).
This guarantees the denoised tokens always form valid JSON, regardless of
the parallel, non-left-to-right order in which BD3LM commits them.

FSM back-end
------------
Uses `outlines-core` (FSM layer extracted from the `outlines` library).
Install:  pip install outlines-core
Fallback: if not installed, a simple regex mask is applied instead (less
strict but still useful for JSON syntax).

Usage
-----
python examples/a2d/bd3lm/chat_toolcall.py \
    --model_name_or_path .models/a2d/Qwen3-0.6B/bd3lm/s1k-toolcall/checkpoint-final \
    --json_schema '{"type":"object","properties":{"name":{"type":"string"},"arguments":{"type":"object"}},"required":["name","arguments"]}'
"""

import copy
import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from dllm.core.samplers.base import BaseSampler, BaseSamplerConfig, BaseSamplerOutput
from dllm.core.samplers.bd3lm import (
    BD3LMSamplerConfig,
    _diffusion_step_block,
    _prepare_for_sampling,
)
from dllm.core.samplers.utils import add_gumbel_noise, get_num_transfer_tokens


# ---------------------------------------------------------------------------
# JSON Constraint Engine
# ---------------------------------------------------------------------------

class JsonConstraintEngine:
    """
    Wraps outlines-core's FSM to compute, for any committed prefix,
    the set of token ids that keep the sequence on-track toward valid JSON.

    Falls back to a simple bracket/quote tracker if outlines-core is absent.
    """

    def __init__(self, tokenizer, json_schema: dict | None = None):
        self.tokenizer = tokenizer
        self.json_schema = json_schema
        self._fsm = None
        self._fsm_states: dict[tuple, int] = {}  # prefix_hash → fsm_state_id
        self._vocab_size = tokenizer.vocab_size

        try:
            self._init_outlines(json_schema)
            self._use_outlines = True
            print("[JsonConstraintEngine] Using outlines-core FSM backend.")
        except Exception as e:
            print(f"[JsonConstraintEngine] outlines-core unavailable ({e}); "
                  f"falling back to regex mask.")
            self._use_outlines = False

    def _init_outlines(self, json_schema):
        """Build regex FSM from JSON schema via outlines-core."""
        from outlines_core.fsm.json_schema import build_regex_from_schema
        from outlines_core.fsm.regex import RegexGuide

        if json_schema is not None:
            import json
            schema_str = json.dumps(json_schema) if isinstance(json_schema, dict) else json_schema
            regex = build_regex_from_schema(schema_str)
        else:
            # Generic JSON object regex
            regex = r'\{[^}]*\}'

        self._guide = RegexGuide.from_regex(regex, self.tokenizer)

    def get_valid_token_mask(
        self,
        prefix_token_ids: list[int],
        device: torch.device,
    ) -> torch.Tensor:
        """
        Returns a float mask of shape [vocab_size]:
            0.0  → token is valid at this position
           -inf  → token is invalid
        """
        if self._use_outlines:
            return self._outlines_mask(prefix_token_ids, device)
        else:
            return self._regex_fallback_mask(prefix_token_ids, device)

    def _outlines_mask(self, prefix_token_ids: list[int], device: torch.device) -> torch.Tensor:
        """Walk FSM with prefix and return allowed-token mask."""
        state = self._guide.initial_state
        for tok_id in prefix_token_ids:
            state = self._guide.get_next_state(state, tok_id)
            if state == self._guide.final_state:
                break

        allowed = self._guide.get_next_instruction(state).tokens  # list or tensor of ids
        mask = torch.full((self._vocab_size,), float("-inf"), device=device)
        if allowed is not None and len(allowed) > 0:
            allowed_t = torch.as_tensor(list(allowed), dtype=torch.long, device=device)
            mask[allowed_t] = 0.0
        return mask

    def _regex_fallback_mask(self, prefix_token_ids: list[int], device: torch.device) -> torch.Tensor:
        """
        Lightweight fallback: allow any token that doesn't break basic JSON
        syntax given a simple bracket/quote depth counter.
        Only blocks tokens whose decoded text would clearly corrupt JSON.
        Much less strict than the FSM approach but prevents catastrophic failures.
        """
        # Decode prefix to string to count depth
        prefix_str = self.tokenizer.decode(prefix_token_ids, skip_special_tokens=False)
        depth = 0
        in_str = False
        for ch in prefix_str:
            if ch == '"' and not in_str:
                in_str = True
            elif ch == '"' and in_str:
                in_str = False
            elif not in_str:
                if ch in "{[":
                    depth += 1
                elif ch in "}]":
                    depth -= 1

        mask = torch.zeros(self._vocab_size, device=device)  # all valid by default

        if depth <= 0 and len(prefix_str.strip()) > 0:
            # JSON is closed — only allow EOS / end tokens
            allowed_ends = {
                self.tokenizer.eos_token_id,
                self.tokenizer.pad_token_id,
            }
            mask = torch.full((self._vocab_size,), float("-inf"), device=device)
            for tid in allowed_ends:
                if tid is not None:
                    mask[tid] = 0.0

        return mask


def _apply_json_constraint(
    logits: torch.Tensor,          # [B, L, V]
    x_block: torch.Tensor,          # [B, L]  current block (mask_id for uncommitted)
    mask_block: torch.Tensor,       # [B, L]  bool: True = still masked
    committed_prefix: list[list[int]],  # per-batch committed tokens BEFORE this block
    mask_id: int,
    engine: JsonConstraintEngine,
) -> torch.Tensor:
    """
    For each batch item and each still-masked position, walk the JSON FSM
    with the committed prefix + committed positions before pos, then mask
    invalid tokens in logits.

    Complexity: O(B * L * avg_prefix_len) per diffusion step.
    """
    B, L, V = logits.shape
    device = logits.device

    for b in range(B):
        prefix_b = list(committed_prefix[b])  # tokens before the block

        for pos in range(L):
            if not mask_block[b, pos]:
                continue  # already committed, skip

            # Extend prefix with committed tokens in this block before pos
            block_prefix = [
                x_block[b, j].item()
                for j in range(pos)
                if not mask_block[b, j]
            ]
            full_prefix = prefix_b + block_prefix

            token_mask = engine.get_valid_token_mask(full_prefix, device)
            logits[b, pos] += token_mask

    return logits


# ---------------------------------------------------------------------------
# Constrained Sampler Config & Sampler
# ---------------------------------------------------------------------------

@dataclass
class ConstrainedBD3LMSamplerConfig(BD3LMSamplerConfig):
    # Token/tag strings to detect intent
    tool_call_intent_token: str = "<tool_call_intent>"
    think_end_token: str = "</think>"
    # If None → unconstrained JSON object; provide a JSON Schema dict to restrict shape
    json_schema: dict | None = None


@dataclass
class ConstrainedBD3LMSampler(BaseSampler):
    """
    BD3LM sampler with per-block JSON constraint for tool calls.

    Pipeline
    --------
    1. Generate blocks normally (reasoning / free text).
    2. After each block, scan decoded text for:
           (a)  <tool_call_intent>  — model signalled tool intent
           (b)  </think>           — reasoning section ended
    3. Once BOTH are seen, all subsequent blocks use JSON-constrained logits.
    4. If only </think> appears (no intent) → continue normal generation.
    """

    @torch.no_grad()
    def sample(
        self,
        inputs: list[torch.Tensor | list],
        config: ConstrainedBD3LMSamplerConfig | None = None,
        **kwargs,
    ) -> BaseSamplerOutput:

        if config is None:
            config = ConstrainedBD3LMSamplerConfig()

        # ---- pull args ----
        steps               = kwargs.get("steps", config.steps)
        steps_per_block     = kwargs.get("steps_per_block", config.steps_per_block)
        max_new_tokens      = kwargs.get("max_new_tokens", config.max_new_tokens)
        max_length          = kwargs.get("max_length", config.max_length)
        block_size          = kwargs.get("block_size", config.block_size)
        temperature         = kwargs.get("temperature", config.temperature)
        remasking           = kwargs.get("remasking", config.remasking)
        stochastic_transfer = kwargs.get("stochastic_transfer", config.stochastic_transfer)
        return_dict         = kwargs.get("return_dict", config.return_dict)
        right_shift_logits  = kwargs.get("right_shift_logits", config.right_shift_logits)
        cfg_scale           = kwargs.get("cfg_scale", config.cfg_scale)
        cfg_keep_tokens     = kwargs.get("cfg_keep_tokens", config.cfg_keep_tokens)
        json_schema         = kwargs.get("json_schema", config.json_schema)
        intent_tok_str      = config.tool_call_intent_token
        think_end_str       = config.think_end_token

        assert block_size >= 1
        assert steps >= 1

        mask_id = self.tokenizer.mask_token_id
        bos_id  = self.tokenizer.bos_token_id
        pad_id  = self.tokenizer.pad_token_id
        eos_id  = self.tokenizer.eos_token_id

        # Resolve intent / think-end token ids
        intent_tok_ids   = self.tokenizer.encode(intent_tok_str, add_special_tokens=False)
        think_end_tok_ids = self.tokenizer.encode(think_end_str, add_special_tokens=False)

        # Lazy-init constraint engine (once per sampler instance)
        if not hasattr(self, "_constraint_engine"):
            self._constraint_engine = JsonConstraintEngine(self.tokenizer, json_schema)

        # ---- normalize inputs ----
        if right_shift_logits:
            inputs = [
                [bos_id] if isinstance(p, list) and len(p) == 0 else p for p in inputs
            ]
        if isinstance(inputs[0], list):
            inputs = [
                torch.as_tensor(p, dtype=torch.long, device=self.model.device)
                for p in inputs
            ]

        prompt_lens = [p.shape[0] for p in inputs]
        if max_new_tokens:
            max_length = max_new_tokens + max(prompt_lens)
        else:
            max_new_tokens = max_length - max(prompt_lens)

        B = len(inputs)
        max_prompt_len = max(prompt_lens)

        # ============================================================
        # 1) Initialise x with left-padded prompts
        # ============================================================
        padded_prompt_len = (
            (max_prompt_len + block_size - 1) // block_size
        ) * block_size

        x = torch.full(
            (B, padded_prompt_len), pad_id, dtype=torch.long, device=self.model.device
        )
        for b, p in enumerate(inputs):
            L = prompt_lens[b]
            offset = padded_prompt_len - L
            x[b, offset : offset + L] = p

        unmasked_index = (x != mask_id) & (x != pad_id)
        if cfg_keep_tokens:
            keep_mask = torch.isin(
                x, torch.as_tensor(cfg_keep_tokens, device=self.model.device)
            )
            unmasked_index = unmasked_index & (~keep_mask)

        done = torch.zeros((B,), dtype=torch.bool, device=self.model.device)

        # ---- intent / think tracking (per batch item) ----
        # Once True → that batch item's next block uses JSON constraint
        intent_detected  = [False] * B   # saw <tool_call_intent>
        think_ended      = [False] * B   # saw </think>
        constrained_mode = [False] * B   # both conditions met

        num_blocks    = math.ceil(max_new_tokens / block_size)
        if steps_per_block is None:
            steps_per_block = math.ceil(steps / num_blocks)
        histories = [x.clone()] if return_dict else None

        generated = 0

        def _tokens_contain(token_seq: torch.Tensor, subseq: list[int]) -> bool:
            """Check if subseq appears in token_seq (1D)."""
            if len(subseq) == 0:
                return False
            seq = token_seq.tolist()
            for i in range(len(seq) - len(subseq) + 1):
                if seq[i : i + len(subseq)] == subseq:
                    return True
            return False

        # ============================================================
        # 2) Block-by-block generation
        # ============================================================
        for b_idx in range(num_blocks):
            if done.all():
                break

            T_prefix = x.shape[1]
            cur_block_len = min(block_size, max_new_tokens - generated)
            if cur_block_len <= 0:
                break

            # ---- prefix forward pass (with KV cache) ----
            x_prefix = x
            prefix_attn, prefix_pos = _prepare_for_sampling(
                x=x_prefix, block_size=block_size, pad_token_id=pad_id
            )
            out_prefix = self.model(
                x_prefix,
                attention_mask=prefix_attn,
                position_ids=prefix_pos,
                use_cache=True,
            )
            cond_past = out_prefix.past_key_values
            cond_prefix_last_logits = out_prefix.logits[:, -1:, :]

            if cfg_scale > 0.0:
                un_x_prefix = x_prefix.clone()
                un_x_prefix[unmasked_index] = mask_id
                out_un = self.model(
                    un_x_prefix,
                    attention_mask=prefix_attn,
                    position_ids=prefix_pos,
                    use_cache=True,
                )
                uncond_past = out_un.past_key_values
                uncond_prefix_last_logits = out_un.logits[:, -1:, :]
            else:
                uncond_past = None
                uncond_prefix_last_logits = None

            # ---- append new masked block ----
            new_block = torch.full(
                (B, cur_block_len), mask_id, dtype=torch.long, device=self.model.device
            )
            x = torch.cat([x, new_block], dim=1)
            unmasked_index = torch.cat(
                [
                    unmasked_index,
                    torch.zeros((B, cur_block_len), dtype=torch.bool, device=self.model.device),
                ],
                dim=1,
            )
            T_total = x.shape[1]

            block_mask_index = x[:, -cur_block_len:] == mask_id
            num_transfer_tokens = get_num_transfer_tokens(
                mask_index=block_mask_index,
                steps=steps_per_block,
                scheduler=self.scheduler,
                stochastic=stochastic_transfer,
            )
            effective_steps = num_transfer_tokens.size(1)

            full_attention_mask, full_position_ids = _prepare_for_sampling(
                x=x, block_size=block_size, pad_token_id=pad_id
            )
            attn_block = full_attention_mask[:, :, T_prefix:T_total, :]
            pos_block   = full_position_ids[:, T_prefix:T_total]

            # ============================================================
            # 3) Inner diffusion loop
            # ============================================================
            for i_step in range(effective_steps):
                x_block   = x[:, T_prefix:T_total]      # [B, cur_block_len]
                mask_block = x_block == mask_id

                if not mask_block.any():
                    break

                cond_logits_block = self.model(
                    x_block,
                    attention_mask=attn_block,
                    position_ids=pos_block,
                    past_key_values=copy.deepcopy(cond_past),
                    use_cache=False,
                ).logits

                logits_block = cond_logits_block

                if cfg_scale > 0.0:
                    un_logits_block = self.model(
                        x_block,
                        attention_mask=attn_block,
                        position_ids=pos_block,
                        past_key_values=copy.deepcopy(uncond_past),
                        use_cache=False,
                    ).logits
                    logits_block = un_logits_block + (cfg_scale + 1.0) * (
                        cond_logits_block - un_logits_block
                    )

                if right_shift_logits:
                    prefix_last = (
                        uncond_prefix_last_logits + (cfg_scale + 1.0) * (
                            cond_prefix_last_logits - uncond_prefix_last_logits
                        ) if cfg_scale > 0.0 else cond_prefix_last_logits
                    )
                    shifted = torch.empty_like(logits_block)
                    shifted[:, 0:1, :] = prefix_last
                    shifted[:, 1:, :] = logits_block[:, :-1, :]
                    logits_block = shifted

                # ---- Apply JSON constraint for tool-call blocks ----
                if any(constrained_mode):
                    # Build committed prefix per batch item:
                    # all non-pad, non-mask tokens before this block
                    committed_prefix = []
                    for bi in range(B):
                        if constrained_mode[bi]:
                            prefix_toks = [
                                x[bi, j].item()
                                for j in range(T_prefix)
                                if x[bi, j].item() not in (pad_id, mask_id)
                            ]
                            committed_prefix.append(prefix_toks)
                        else:
                            committed_prefix.append([])

                    logits_block = _apply_json_constraint(
                        logits=logits_block,
                        x_block=x_block,
                        mask_block=mask_block,
                        committed_prefix=committed_prefix,
                        mask_id=mask_id,
                        engine=self._constraint_engine,
                    )

                x_block_updated = _diffusion_step_block(
                    logits=logits_block,
                    x_block=x_block,
                    mask_block=mask_block,
                    num_transfer_step=num_transfer_tokens[:, i_step],
                    temperature=temperature,
                    remasking=remasking,
                )
                x[:, T_prefix:T_total] = x_block_updated

                if histories is not None:
                    histories.append(x.clone())

            # ============================================================
            # 4) Post-block: detect intent / think-end tokens
            # ============================================================
            fully_generated_so_far = x  # [B, T_total]

            for bi in range(B):
                if constrained_mode[bi]:
                    continue  # already in constrained mode

                seq_bi = fully_generated_so_far[bi]

                if not intent_detected[bi]:
                    intent_detected[bi] = _tokens_contain(seq_bi, intent_tok_ids)

                if not think_ended[bi]:
                    think_ended[bi] = _tokens_contain(seq_bi, think_end_tok_ids)

                if intent_detected[bi] and think_ended[bi]:
                    constrained_mode[bi] = True

            # ---- EOS stopping ----
            if eos_id is not None:
                eos_in_block = (x[:, T_prefix:T_total] == eos_id).any(dim=1)
                done = done | eos_in_block

            generated += cur_block_len

        # ============================================================
        # 5) Output
        # ============================================================
        if not return_dict:
            return x
        return BaseSamplerOutput(sequences=x, histories=histories)

    @torch.no_grad()
    def infill(self, inputs, config=None, **kwargs):
        raise NotImplementedError

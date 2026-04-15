"""
SFT script for BD3LM tool-call training.

Teaches the model to:
  1. Reason inside <think>...</think>
  2. Emit <tool_call_intent> at the END of the think block when a tool is needed
  3. Produce a valid JSON tool call immediately after </think>

At inference time, ConstrainedBD3LMSampler watches for <tool_call_intent>
and automatically JSON-constrains the first response block.

Data format expected (one assistant turn):
    <think>
    I need to look up current weather...
    <tool_call_intent>
    </think>
    {"name": "get_weather", "arguments": {"location": "NYC"}}

Supported datasets  (via --dataset_args)
-----------------------------------------
  NousResearch/hermes-function-calling-v1
  glaive-function-calling-v2
  Any HF dataset whose rows have a "conversations" or "messages" field with tool calls

Usage
-----
Single GPU (Qwen3-0.6B):
    accelerate launch \
        --config_file scripts/accelerate_configs/ddp.yaml --num_processes 1 \
        examples/a2d/bd3lm/sft_toolcall.py

Qwen3-1.7B (gradient checkpointing):
    accelerate launch \
        --config_file scripts/accelerate_configs/ddp.yaml --num_processes 1 \
        examples/a2d/bd3lm/sft_toolcall.py \
        --model_name_or_path .models/a2d/Qwen3-1.7B \
        --output_dir .models/a2d/Qwen3-1.7B/bd3lm/toolcall \
        --per_device_train_batch_size 4 \
        --gradient_checkpointing true \
        --gradient_accumulation_steps 4
"""

import json
import os
from dataclasses import dataclass, field
from functools import partial
from typing import Optional

import accelerate
import transformers

import dllm

logger = dllm.utils.get_default_logger(__name__)

# ── Special tokens added to vocabulary ──────────────────────────────────────
TOOL_CALL_INTENT_TOKEN = "<tool_call_intent>"
TOOL_CALL_START_TOKEN  = "<tool_call>"
TOOL_CALL_END_TOKEN    = "</tool_call>"
SPECIAL_TOKENS = [TOOL_CALL_INTENT_TOKEN, TOOL_CALL_START_TOKEN, TOOL_CALL_END_TOKEN]


# ── Data formatting ──────────────────────────────────────────────────────────

def _extract_tool_call_json(tool_calls) -> str | None:
    """
    Normalise various tool-call schemas to a plain JSON string.
    Handles:
      - list of dicts with "name"/"arguments"/"parameters" keys
      - dicts directly
      - already-string JSON
    """
    if tool_calls is None:
        return None
    if isinstance(tool_calls, str):
        return tool_calls.strip()
    if isinstance(tool_calls, dict):
        return json.dumps(tool_calls, ensure_ascii=False)
    if isinstance(tool_calls, list) and len(tool_calls) > 0:
        tc = tool_calls[0]
        if isinstance(tc, str):
            return tc.strip()
        # normalise key names
        name = tc.get("name") or tc.get("function", {}).get("name", "unknown")
        args = tc.get("arguments") or tc.get("parameters") or tc.get("function", {}).get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                pass
        return json.dumps({"name": name, "arguments": args}, ensure_ascii=False)
    return None


def _format_toolcall_assistant(reasoning: str | None, tool_call_json: str | None, response: str | None) -> dict:
    """
    Build the assistant message dict for the chat template.

    • reasoning + tool_call_json  → think block ending with <tool_call_intent>, then JSON
    • reasoning only              → plain thinking, no intent token
    • response only               → no thinking at all
    """
    if tool_call_json:
        # Inject <tool_call_intent> at end of reasoning (or create a minimal think block)
        think_body = (reasoning.strip() if reasoning else "I should call a tool.")
        think_body = think_body + f"\n{TOOL_CALL_INTENT_TOKEN}"
        return {
            "role": "assistant",
            # reasoning_content is rendered as <think>...</think> by Qwen3 template
            "reasoning_content": think_body,
            "content": f"{TOOL_CALL_START_TOKEN}\n{tool_call_json}\n{TOOL_CALL_END_TOKEN}",
        }
    elif reasoning:
        return {
            "role": "assistant",
            "reasoning_content": reasoning.strip(),
            "content": response or "",
        }
    else:
        return {"role": "assistant", "content": response or ""}


import re as _re
_THINK_RE = _re.compile(r"<think>(.*?)</think>(.*)", _re.DOTALL)
_TOOL_CALL_RE = _re.compile(r"<tool_call>(.*?)</tool_call>", _re.DOTALL)


def _parse_assistant_value(value: str) -> tuple[str | None, str | None, str | None]:
    """
    Parse an assistant turn value into (thinking, tool_call_json, response).
    Handles both:
      • Hermes-thinking format: <think>...</think><tool_call>{...}</tool_call>
      • Plain Hermes format:    <tool_call>{...}</tool_call>  (no think block)
    Returns (thinking, tool_json, response_text) — any can be None.
    """
    thinking = None
    tool_json = None
    response = None

    # Extract <think> block if present
    think_match = _THINK_RE.match(value.strip())
    if think_match:
        thinking = think_match.group(1).strip()
        remainder = think_match.group(2).strip()
    else:
        remainder = value.strip()

    # Extract <tool_call> block if present
    tc_match = _TOOL_CALL_RE.search(remainder)
    if tc_match:
        raw = tc_match.group(1).strip()
        # Normalise: sometimes it's Python-style dict with single quotes
        try:
            tool_json = json.dumps(json.loads(raw))
        except Exception:
            try:
                import ast
                tool_json = json.dumps(ast.literal_eval(raw))
            except Exception:
                tool_json = raw  # keep as-is, FSM will handle it
        # Whatever's outside the tool_call tag is response text
        response = _TOOL_CALL_RE.sub("", remainder).strip() or None
    else:
        # No tool call — plain text response
        response = remainder or None

    return thinking, tool_json, response


def _hermes_thinking_map_fn(example: dict) -> dict | None:
    """
    Convert Jofthomas/hermes-function-calling-thinking-V1 rows.
    Assistant turns have real <think>...</think> blocks before tool calls.
    Schema is the same as Hermes: conversations with from/value.
    """
    messages = []
    convs = example.get("conversations", [])
    for turn in convs:
        role  = turn.get("from", "")
        value = turn.get("value", "")

        if role == "human":
            messages.append({"role": "user", "content": value})
        elif role == "gpt":
            thinking, tool_json, response = _parse_assistant_value(value)
            if tool_json:
                # Real reasoning from dataset — just append <tool_call_intent>
                messages.append(_format_toolcall_assistant(
                    reasoning=thinking,   # actual thinking trace, not fake
                    tool_call_json=tool_json,
                    response=None,
                ))
            else:
                # Plain response (possibly with thinking)
                if thinking:
                    messages.append({
                        "role": "assistant",
                        "reasoning_content": thinking,
                        "content": response or "",
                    })
                else:
                    messages.append({"role": "assistant", "content": response or ""})
        # skip tool-result turns

    if len(messages) < 2:
        return None
    return {"messages": messages}


def _hermes_map_fn(example: dict) -> dict | None:
    """
    Convert NousResearch/hermes-function-calling-v1 rows (no thinking traces).
    We still inject <tool_call_intent> so the model learns the intent signal,
    but note the reasoning body will be minimal since the dataset has none.
    For richer reasoning, use Jofthomas/hermes-function-calling-thinking-V1.
    """
    messages = []
    convs = example.get("conversations", [])
    for turn in convs:
        role  = turn.get("from", "")
        value = turn.get("value", "")

        if role == "human":
            messages.append({"role": "user", "content": value})
        elif role == "gpt":
            _, tool_json, response = _parse_assistant_value(value)
            if tool_json:
                messages.append(_format_toolcall_assistant(
                    reasoning=None,   # no reasoning in this dataset
                    tool_call_json=tool_json,
                    response=None,
                ))
            else:
                messages.append({"role": "assistant", "content": value})

    if len(messages) < 2:
        return None
    return {"messages": messages}


def _glaive_map_fn(example: dict) -> dict | None:
    """
    Convert glaive-function-calling-v2 rows.
    Schema: {"system": str, "chat": str}
    """
    chat_str = example.get("chat", "")
    # Glaive uses plain text format, try to parse
    messages = []
    system = example.get("system", "")
    if system:
        messages.append({"role": "system", "content": system})

    # Split on USER: / ASSISTANT: markers
    import re
    turns = re.split(r"(?:^|\n)(USER:|ASSISTANT:|FUNCTION RESPONSE:)", chat_str)
    current_role = None
    for part in turns:
        part = part.strip()
        if part == "USER:":
            current_role = "user"
        elif part == "ASSISTANT:":
            current_role = "assistant"
        elif part == "FUNCTION RESPONSE:":
            current_role = None  # skip function responses
        elif current_role == "user" and part:
            messages.append({"role": "user", "content": part})
        elif current_role == "assistant" and part:
            # Check for function call JSON
            fc_match = re.search(r"<functioncall>(.*?)</functioncall>", part, re.DOTALL)
            if fc_match:
                try:
                    tc_json = fc_match.group(1).strip()
                    json.loads(tc_json)
                    messages.append(_format_toolcall_assistant(
                        reasoning="I need to call a function.",
                        tool_call_json=tc_json,
                        response=None,
                    ))
                except Exception:
                    messages.append({"role": "assistant", "content": part})
            else:
                messages.append({"role": "assistant", "content": part})

    if len(messages) < 2:
        return None
    return {"messages": messages}


DATASET_MAP_FNS = {
    # With real thinking traces — prefer this over plain Hermes
    "Jofthomas/hermes-function-calling-thinking-V1": _hermes_thinking_map_fn,
    # Plain Hermes — no thinking, but larger (11K rows)
    "NousResearch/hermes-function-calling-v1": _hermes_map_fn,
    "glaive/glaive-function-calling-v2": _glaive_map_fn,
    "glaive-function-calling-v2": _glaive_map_fn,
}


def _toolcall_sft_map_fn(row, *, tokenizer, mask_prompt_loss: bool = True) -> dict:
    """
    Tokenise a row that already has row["messages"] in the standard format.
    Passes enable_thinking=True so <think>...</think> are included.
    """
    try:
        prompt_response_tokens = tokenizer.apply_chat_template(
            row["messages"],
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=True,
        )
    except Exception:
        # Fallback: template doesn't accept enable_thinking
        prompt_response_tokens = tokenizer.apply_chat_template(
            row["messages"],
            tokenize=True,
            add_generation_prompt=False,
        )

    labels = prompt_response_tokens.copy()

    if mask_prompt_loss:
        try:
            prompt_tokens = tokenizer.apply_chat_template(
                row["messages"][:-1],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=True,
            )
        except Exception:
            prompt_tokens = tokenizer.apply_chat_template(
                row["messages"][:-1],
                tokenize=True,
                add_generation_prompt=True,
            )
        labels[: len(prompt_tokens)] = [-100] * len(prompt_tokens)
        return {
            "input_ids": prompt_response_tokens,
            "labels": labels,
            "prompt_len": len(prompt_tokens),
        }

    return {"input_ids": prompt_response_tokens, "labels": labels}


# ── Args ─────────────────────────────────────────────────────────────────────

@dataclass
class ModelArguments(dllm.utils.ModelArguments):
    model_name_or_path: str = ".models/a2d/Qwen3-0.6B"


@dataclass
class DataArguments(dllm.utils.DataArguments):
    dataset_args: str = "NousResearch/hermes-function-calling-v1"
    max_length: int = 1024
    load_preprocessed_data: bool = False
    mask_prompt_loss: bool = field(default=True)


@dataclass
class TrainingArguments(dllm.core.trainers.BD3LMConfig):
    output_dir: str = ".models/a2d/Qwen3-0.6B/bd3lm/toolcall"
    group_by_length: bool = True
    num_train_epochs: int = 3
    learning_rate: float = 5e-5
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    block_size: int = 64
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 200


# ── Main ─────────────────────────────────────────────────────────────────────

def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    dllm.utils.print_args_main(model_args, data_args, training_args)
    dllm.utils.initial_training_setup(model_args, data_args, training_args)

    # ----- Model + tokenizer -----------------------------------------------
    model     = dllm.utils.get_model(model_args=model_args)
    tokenizer = dllm.utils.get_tokenizer(model_args=model_args)

    # Add special tool-call tokens to vocabulary
    num_added = tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    if num_added > 0:
        logger.info(f"Added {num_added} special tokens: {SPECIAL_TOKENS}")
        model.resize_token_embeddings(len(tokenizer))

    # ----- Dataset ----------------------------------------------------------
    with accelerate.PartialState().local_main_process_first():
        from datasets import load_dataset

        raw_name = data_args.dataset_args.strip()
        dataset_map_fn = DATASET_MAP_FNS.get(raw_name)

        if dataset_map_fn is None:
            # Generic fallback — assume the dataset already has "messages" column
            logger.warning(
                f"No custom map fn for '{raw_name}'. Assuming 'messages' column exists."
            )
            raw_ds = load_dataset(raw_name)
        else:
            raw_ds = load_dataset(raw_name)
            # Apply dataset-specific formatting first
            raw_ds = raw_ds.map(
                dataset_map_fn,
                remove_columns=raw_ds["train"].column_names,
                num_proc=data_args.num_proc,
                desc=f"Reformatting {raw_name} for tool-call format",
            )
            # Drop rows that returned None (too short / malformed)
            raw_ds = raw_ds.filter(lambda ex: ex["messages"] is not None)

        if "test" not in raw_ds:
            split = raw_ds["train"].train_test_split(test_size=min(200, len(raw_ds["train"]) // 20), seed=42)
            raw_ds = split

        # Tokenise
        dataset = raw_ds.map(
            partial(_toolcall_sft_map_fn, tokenizer=tokenizer, mask_prompt_loss=data_args.mask_prompt_loss),
            num_proc=data_args.num_proc,
            desc="Tokenising tool-call dataset",
        )
        dataset = dllm.utils.post_process_dataset(dataset, data_args)

    # ----- Training ---------------------------------------------------------
    accelerate.PartialState().wait_for_everyone()
    logger.info("Start tool-call BD3LM training...")

    trainer = dllm.core.trainers.BD3LMTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("test", None),
        args=training_args,
        data_collator=(
            dllm.core.trainers.bd3lm.AppendEOSBlockWrapper(
                transformers.DataCollatorForSeq2Seq(
                    tokenizer,
                    return_tensors="pt",
                    padding=True,
                ),
                block_size=training_args.block_size,
            )
        ),
    )
    trainer.train()
    ckpt = os.path.join(training_args.output_dir, "checkpoint-final")
    trainer.save_model(ckpt)
    trainer.processing_class.save_pretrained(ckpt)
    logger.info(f"Saved to {ckpt}")


if __name__ == "__main__":
    train()

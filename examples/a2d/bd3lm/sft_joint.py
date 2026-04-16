"""
Joint BD3LM SFT: reasoning + tool-call + general retention in one training run.

Why joint?
----------
Sequential training (reasoning → tool-call) causes catastrophic forgetting.
Training on all data simultaneously keeps both skills.

All three dataset types share the same token format:

  Reasoning only:
    <think>step-by-step reasoning...</think>
    final answer

  Tool-call (has <tool_call_intent> at end of think):
    <think>I need to call get_weather...\n<tool_call_intent></think>
    <tool_call>{"name": "get_weather", "arguments": {...}}</tool_call>

  General (no think block or empty think):
    response text

Because they all use enable_thinking=True and the same tokenizer,
they can be batched together without any special handling.

Data mix  (set via --dataset_args with + separator):
  reasoning  : simplescaling/s1K
               open-thoughts/OpenThoughts-114k[train:N]
  tool-call  : NousResearch/hermes-function-calling-v1
               glaive/glaive-function-calling-v2[train:N]
  retention  : tatsu-lab/alpaca[train:N]
               HuggingFaceH4/ultrachat_200k[train:N]

Usage
-----
# Qwen3-0.6B, recommended mix:
accelerate launch \\
    --config_file scripts/accelerate_configs/ddp.yaml --num_processes 1 \\
    examples/a2d/bd3lm/sft_joint.py \\
    --model_name_or_path .models/a2d/Qwen3-0.6B \\
    --output_dir .models/a2d/Qwen3-0.6B/bd3lm/joint

# Qwen3-1.7B with gradient checkpointing:
accelerate launch \\
    --config_file scripts/accelerate_configs/ddp.yaml --num_processes 1 \\
    examples/a2d/bd3lm/sft_joint.py \\
    --model_name_or_path .models/a2d/Qwen3-1.7B \\
    --output_dir .models/a2d/Qwen3-1.7B/bd3lm/joint \\
    --per_device_train_batch_size 4 \\
    --gradient_checkpointing true \\
    --gradient_accumulation_steps 4
"""

import json
import os
import re
from dataclasses import dataclass, field
from functools import partial

import accelerate
import transformers
from datasets import DatasetDict, concatenate_datasets

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import dllm
from dllm.data.s1k import load_dataset_s1k
from dllm.data.openthoughts import load_dataset_openthoughts
from examples.a2d.bd3lm.sft_toolcall import (
    SPECIAL_TOKENS,
    _extract_tool_call_json,
    _format_toolcall_assistant,
    _hermes_map_fn,
    _glaive_map_fn,
    DATASET_MAP_FNS as TOOLCALL_DATASET_MAP_FNS,
)

logger = dllm.utils.get_default_logger(__name__)

# ── Dataset routing ───────────────────────────────────────────────────────────

_REASONING_DATASETS = {
    "simplescaling/s1K":               load_dataset_s1k,
    "open-thoughts/OpenThoughts-114k": load_dataset_openthoughts,
}

_TRAIN_LIMIT_RE = re.compile(r"\[train:(\d+)\]")


def _strip_limit(name: str) -> tuple[str, int | None]:
    m = _TRAIN_LIMIT_RE.search(name)
    if m:
        return name[: m.start()].strip(), int(m.group(1))
    return name.strip(), None


def _load_one(raw_spec: str) -> DatasetDict:
    from datasets import load_dataset

    name, limit = _strip_limit(raw_spec)

    # Detect dataset type
    reasoning_loader = next(
        (fn for key, fn in _REASONING_DATASETS.items() if key in name), None
    )
    toolcall_reformat = next(
        (fn for key, fn in TOOLCALL_DATASET_MAP_FNS.items() if key in name), None
    )

    if reasoning_loader:
        ds = reasoning_loader(name)
    elif toolcall_reformat:
        raw = load_dataset(name)
        train_split = "train" if "train" in raw else next(k for k in raw if "train" in k)
        ds = raw.map(
            toolcall_reformat,
            remove_columns=raw[train_split].column_names,
            num_proc=1,
            desc=f"Reformatting {name}",
        ).filter(lambda ex: ex.get("messages") is not None)
    else:
        ds = load_dataset(name)  # assume already has "messages" column

    if not isinstance(ds, DatasetDict):
        ds = DatasetDict({"train": ds})

    # Normalise split names (e.g. ultrachat uses train_gen/test_gen)
    if "train" not in ds:
        train_key = next((k for k in ds if "train" in k), None)
        test_key  = next((k for k in ds if "test"  in k), None)
        if train_key:
            ds = DatasetDict({
                "train": ds[train_key],
                **({"test": ds[test_key]} if test_key else {}),
            })

    if "test" not in ds:
        eval_n = min(100, max(50, len(ds["train"]) // 20))
        split = ds["train"].train_test_split(test_size=eval_n, seed=42)
        ds = DatasetDict({"train": split["train"], "test": split["test"]})

    if limit is not None and limit < len(ds["train"]):
        ds = DatasetDict({"train": ds["train"].select(range(limit)), "test": ds.get("test")})

    return ds


def _normalize_messages(ex):
    """Ensure all message dicts have the same keys so datasets can concatenate."""
    msgs = ex.get("messages")
    if not msgs:
        return {"messages": msgs}
    return {"messages": [
        {"role": m.get("role", ""), "content": m.get("content", ""), "reasoning_content": m.get("reasoning_content", "")}
        for m in msgs
    ]}


def _load_datasets(dataset_args: str) -> DatasetDict:
    specs = [s.strip() for s in dataset_args.split("+") if s.strip()]
    parts = [_load_one(s) for s in specs]

    # Normalise schema so all parts have identical message field names
    # num_proc=1 avoids "unhashable type: slice" multiprocessing bug in datasets
    parts = [
        p.map(_normalize_messages, num_proc=1, desc="Normalising schema")
        for p in parts
    ]

    if len(parts) == 1:
        return parts[0]

    merged_train = concatenate_datasets([p["train"] for p in parts])
    test_parts   = [p["test"] for p in parts if p.get("test") is not None]
    merged_test  = concatenate_datasets(test_parts) if test_parts else None
    return DatasetDict({"train": merged_train, "test": merged_test})


# ── Unified map function ──────────────────────────────────────────────────────

def _joint_sft_map_fn(row, *, tokenizer, mask_prompt_loss: bool = True) -> dict | None:
    """
    Handles all three data types uniformly:
      - reasoning rows (reasoning_content field, no tool_call_intent)
      - tool-call rows (reasoning_content ending with <tool_call_intent>, content has JSON)
      - general rows  (no reasoning_content)
    All are tokenised with enable_thinking=True.
    """
    def _to_ids(result):
        """Extract a plain list of ints from apply_chat_template output.
        Transformers 5.x returns BatchEncoding; older versions return a plain list."""
        if isinstance(result, list):
            return result
        if hasattr(result, "input_ids"):
            ids = result.input_ids
            return ids.tolist() if hasattr(ids, "tolist") else list(ids)
        return list(result)

    _null = {"input_ids": [0], "labels": [-100], "prompt_len": -1}
    messages = row.get("messages")
    if not messages or len(messages) < 2:
        return _null

    try:
        full_tokens = _to_ids(tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False, enable_thinking=True,
        ))
    except Exception:
        try:
            full_tokens = _to_ids(tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=False,
            ))
        except Exception:
            return _null

    labels = full_tokens.copy()

    if mask_prompt_loss:
        try:
            prompt_tokens = _to_ids(tokenizer.apply_chat_template(
                messages[:-1], tokenize=True, add_generation_prompt=True, enable_thinking=True,
            ))
        except Exception:
            prompt_tokens = _to_ids(tokenizer.apply_chat_template(
                messages[:-1], tokenize=True, add_generation_prompt=True,
            ))
        labels[: len(prompt_tokens)] = [-100] * len(prompt_tokens)
        return {"input_ids": full_tokens, "labels": labels, "prompt_len": len(prompt_tokens)}

    return {"input_ids": full_tokens, "labels": labels, "prompt_len": 0}


# ── Args ──────────────────────────────────────────────────────────────────────

@dataclass
class ModelArguments(dllm.utils.ModelArguments):
    model_name_or_path: str = ".models/a2d/Qwen3-0.6B"
    bidi_gdn_mode: str = field(
        default="causal",
        metadata={"help": "GDN bidirectionality: 'causal' (safe) or 'symmetric_solve' (all layers bidi)"},
    )


@dataclass
class DataArguments(dllm.utils.DataArguments):
    # Reasoning + tool-call (with real thinking) + retention, all mixed
    dataset_args: str = (
        "open-thoughts/OpenThoughts-114k[train:15000]"
        " + Jofthomas/hermes-function-calling-thinking-V1"
        " + NousResearch/hermes-function-calling-v1[train:5000]"
        " + tatsu-lab/alpaca[train:3000]"
        " + HuggingFaceH4/ultrachat_200k[train:2000]"
    )
    max_length: int = 1024
    load_preprocessed_data: bool = False
    mask_prompt_loss: bool = field(default=True)


@dataclass
class TrainingArguments(dllm.core.trainers.BD3LMConfig):
    output_dir: str = ".models/a2d/Qwen3-0.6B/bd3lm/joint"
    group_by_length: bool = False
    num_train_epochs: int = 3
    learning_rate: float = 5e-5
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    block_size: int = 64
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 200


# ── Simple collator (avoids transformers 5.x DataCollatorForSeq2Seq compat issues) ──

import torch as _torch
from torch.nn.utils.rnn import pad_sequence as _pad_sequence

class _SimpleSeq2SeqCollator:
    """Pads input_ids (with pad_token_id) and labels (with -100) into batch tensors."""
    def __init__(self, pad_token_id: int, eos_token_id: int):
        self.pad_token_id = pad_token_id
        # Expose a tokenizer-like object so AppendEOSBlockWrapper can access eos_token_id
        from types import SimpleNamespace
        self.tokenizer = SimpleNamespace(eos_token_id=eos_token_id)

    def __call__(self, features, return_tensors=None):
        input_ids = _pad_sequence(
            [_torch.tensor(list(f["input_ids"]), dtype=_torch.long) for f in features],
            batch_first=True, padding_value=self.pad_token_id,
        )
        labels = _pad_sequence(
            [_torch.tensor(list(f["labels"]), dtype=_torch.long) for f in features],
            batch_first=True, padding_value=-100,
        )
        return {"input_ids": input_ids, "labels": labels}


# ── Main ──────────────────────────────────────────────────────────────────────

def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    dllm.utils.print_args_main(model_args, data_args, training_args)
    dllm.utils.initial_training_setup(model_args, data_args, training_args)

    # Inject bidi_gdn_mode into model config before construction
    if model_args.bidi_gdn_mode != "causal":
        from dllm.pipelines.a2d.models.qwen3_5.modeling_qwen3_5 import A2DQwen3_5Config
        cfg = A2DQwen3_5Config.from_pretrained(model_args.model_name_or_path)
        cfg.bidi_gdn_mode = model_args.bidi_gdn_mode
        logger.info(f"Bidirectional GDN mode: {model_args.bidi_gdn_mode}")
        model = dllm.utils.get_model(model_args=model_args, config=cfg)
    else:
        model = dllm.utils.get_model(model_args=model_args)
    tokenizer = dllm.utils.get_tokenizer(model_args=model_args)

    # Add tool-call special tokens (no-op if already present)
    num_added = tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    if num_added > 0:
        logger.info(f"Added {num_added} special tokens: {SPECIAL_TOKENS}")
        model.resize_token_embeddings(len(tokenizer))

    # ── NaN gradient guard ────────────────────────────────────────────────────
    # MUST be registered AFTER resize_token_embeddings so the new embedding/LM-head
    # parameters are also covered.  Without this, those parameters get unguarded NaN
    # gradients → clip_grad_norm_ returns NaN → weight corruption.
    import torch as _t
    def _nan_hook(grad):
        return _t.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
    for _p in model.parameters():
        if _p.requires_grad:
            _p.register_hook(_nan_hook)
    logger.info("Registered NaN-to-zero gradient hooks on all parameters")

    # Force single-process data loading to avoid multiprocessing slice/write bugs
    data_args.num_proc = 1

    with accelerate.PartialState().local_main_process_first():
        logger.info(f"Loading joint dataset: {data_args.dataset_args}")
        dataset = _load_datasets(data_args.dataset_args)

        logger.info(
            f"Raw sizes — train: {len(dataset['train'])} | "
            f"test: {len(dataset['test']) if dataset.get('test') else 0}"
        )

        dataset = dataset.map(
            partial(_joint_sft_map_fn, tokenizer=tokenizer, mask_prompt_loss=data_args.mask_prompt_loss),
            num_proc=1,
            remove_columns=dataset["train"].column_names,
            desc="Tokenising joint dataset",
        )
        dataset = dataset.filter(
            lambda ex: ex["prompt_len"] >= 0,
            num_proc=1,
        )
        dataset = dllm.utils.post_process_dataset(dataset, data_args)

    accelerate.PartialState().wait_for_everyone()
    logger.info(
        f"After filtering — train: {len(dataset['train'])} | "
        f"test: {len(dataset['test']) if dataset.get('test') else 0}"
    )

    trainer = dllm.core.trainers.BD3LMTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("test", None),
        args=training_args,
        data_collator=(
            dllm.core.trainers.bd3lm.AppendEOSBlockWrapper(
                _SimpleSeq2SeqCollator(pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id),
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

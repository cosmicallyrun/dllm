"""
SFT script for BD3LM reasoning training.

Supported datasets (set via --dataset_args):
  simplescaling/s1K                                      (1K, fast proof-of-concept)
  open-thoughts/OpenThoughts-114k                        (114K, use a subset)
  open-thoughts/OpenThoughts-114k[train:20000]           (20K subset)
  open-thoughts/OpenThoughts-114k[train:20000] + tatsu-lab/alpaca[train:3000]
  simplescaling/s1K + tatsu-lab/alpaca[train:3000]       (s1K + retention mix)

Both s1K and OpenThoughts are auto-detected and parsed correctly.
Any other dataset is assumed to already have a "messages" column.

Usage
-----
# Qwen3-0.6B, s1K + Alpaca retention mix (RTX 4090):
accelerate launch \
    --config_file scripts/accelerate_configs/ddp.yaml --num_processes 1 \
    examples/a2d/bd3lm/sft_reasoning.py \
    --dataset_args "simplescaling/s1K + tatsu-lab/alpaca[train:3000]"

# Qwen3-0.6B, OpenThoughts 20K subset:
accelerate launch \
    --config_file scripts/accelerate_configs/ddp.yaml --num_processes 1 \
    examples/a2d/bd3lm/sft_reasoning.py \
    --dataset_args "open-thoughts/OpenThoughts-114k[train:20000] + tatsu-lab/alpaca[train:3000]"

# Qwen3-1.7B with gradient checkpointing (A100 or 4090 tight):
accelerate launch \
    --config_file scripts/accelerate_configs/ddp.yaml --num_processes 1 \
    examples/a2d/bd3lm/sft_reasoning.py \
    --model_name_or_path .models/a2d/Qwen3-1.7B \
    --output_dir .models/a2d/Qwen3-1.7B/bd3lm/reasoning \
    --dataset_args "open-thoughts/OpenThoughts-114k[train:20000] + tatsu-lab/alpaca[train:3000]" \
    --per_device_train_batch_size 4 \
    --gradient_checkpointing true \
    --gradient_accumulation_steps 4
"""

import os
import re
from dataclasses import dataclass, field
from functools import partial

import accelerate
import transformers
from datasets import DatasetDict, concatenate_datasets

import dllm
from dllm.data.s1k import load_dataset_s1k
from dllm.data.openthoughts import load_dataset_openthoughts

logger = dllm.utils.get_default_logger(__name__)

# Datasets that need special loaders (keyed by substring match)
_SPECIAL_LOADERS = {
    "simplescaling/s1K":                  load_dataset_s1k,
    "open-thoughts/OpenThoughts-114k":    load_dataset_openthoughts,
}

_TRAIN_LIMIT_RE = re.compile(r"\[train:(\d+)\]")


def _strip_limit(name: str) -> tuple[str, int | None]:
    """Parse 'ds/name[train:N]' → ('ds/name', N)."""
    m = _TRAIN_LIMIT_RE.search(name)
    if m:
        return name[: m.start()].strip(), int(m.group(1))
    return name.strip(), None


def _load_one(raw_spec: str) -> DatasetDict:
    """Load a single dataset spec, applying train limit if given."""
    from datasets import load_dataset

    name, limit = _strip_limit(raw_spec)

    loader = None
    for key, fn in _SPECIAL_LOADERS.items():
        if key in name:
            loader = fn
            break

    ds = loader(name) if loader else load_dataset(name)

    # Normalise to DatasetDict
    if not isinstance(ds, DatasetDict):
        ds = DatasetDict({"train": ds})

    # Carve out eval split if missing
    if "test" not in ds:
        eval_n = min(100, max(50, len(ds["train"]) // 20))
        split = ds["train"].train_test_split(test_size=eval_n, seed=42)
        ds = DatasetDict({"train": split["train"], "test": split["test"]})

    # Apply train limit
    if limit is not None and limit < len(ds["train"]):
        ds = DatasetDict({
            "train": ds["train"].select(range(limit)),
            "test":  ds.get("test"),
        })

    return ds


def _load_datasets(dataset_args: str) -> DatasetDict:
    """
    Support '+'-separated specs:
      "simplescaling/s1K + tatsu-lab/alpaca[train:3000]"
    """
    specs = [s.strip() for s in dataset_args.split("+") if s.strip()]
    parts = [_load_one(s) for s in specs]

    if len(parts) == 1:
        return parts[0]

    # Merge: concatenate train and test splits
    merged_train = concatenate_datasets([p["train"] for p in parts])
    test_parts   = [p["test"] for p in parts if p.get("test") is not None]
    merged_test  = concatenate_datasets(test_parts) if test_parts else None

    return DatasetDict({"train": merged_train, "test": merged_test})


def _reasoning_sft_map_fn(row, *, tokenizer, mask_prompt_loss: bool = True) -> dict | None:
    """
    Tokenise a row with a 'messages' column.
    Passes enable_thinking=True so reasoning_content → <think>...</think>.
    Returns None for rows that fail tokenisation (filtered out afterwards).
    """
    messages = row.get("messages")
    if not messages or len(messages) < 2:
        return None

    try:
        prompt_response_tokens = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=True,
        )
    except Exception:
        try:
            prompt_response_tokens = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
            )
        except Exception:
            return None

    labels = prompt_response_tokens.copy()

    if mask_prompt_loss:
        try:
            prompt_tokens = tokenizer.apply_chat_template(
                messages[:-1],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=True,
            )
        except Exception:
            prompt_tokens = tokenizer.apply_chat_template(
                messages[:-1],
                tokenize=True,
                add_generation_prompt=True,
            )
        labels[: len(prompt_tokens)] = [-100] * len(prompt_tokens)
        return {
            "input_ids": prompt_response_tokens,
            "labels":    labels,
            "prompt_len": len(prompt_tokens),
        }

    return {"input_ids": prompt_response_tokens, "labels": labels}


@dataclass
class ModelArguments(dllm.utils.ModelArguments):
    model_name_or_path: str = ".models/a2d/Qwen3-0.6B"


@dataclass
class DataArguments(dllm.utils.DataArguments):
    dataset_args: str = "simplescaling/s1K + tatsu-lab/alpaca[train:3000]"
    max_length: int = 1024
    load_preprocessed_data: bool = False
    mask_prompt_loss: bool = field(default=True)


@dataclass
class TrainingArguments(dllm.core.trainers.BD3LMConfig):
    output_dir: str = ".models/a2d/Qwen3-0.6B/bd3lm/reasoning"
    group_by_length: bool = True
    num_train_epochs: int = 5
    learning_rate: float = 5e-5
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    block_size: int = 64
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 200


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    dllm.utils.print_args_main(model_args, data_args, training_args)
    dllm.utils.initial_training_setup(model_args, data_args, training_args)

    model     = dllm.utils.get_model(model_args=model_args)
    tokenizer = dllm.utils.get_tokenizer(model_args=model_args)

    with accelerate.PartialState().local_main_process_first():
        logger.info(f"Loading dataset(s): {data_args.dataset_args}")
        dataset = _load_datasets(data_args.dataset_args)

        if not data_args.load_preprocessed_data:
            map_fn = partial(
                _reasoning_sft_map_fn,
                tokenizer=tokenizer,
                mask_prompt_loss=data_args.mask_prompt_loss,
            )
            dataset = dataset.map(
                map_fn,
                num_proc=data_args.num_proc,
                desc="Tokenising reasoning dataset",
            )
            # Drop rows that failed tokenisation
            dataset = dataset.filter(
                lambda ex: ex["input_ids"] is not None,
                num_proc=data_args.num_proc,
                desc="Filtering invalid rows",
            )

        dataset = dllm.utils.post_process_dataset(dataset, data_args)

    accelerate.PartialState().wait_for_everyone()
    logger.info(f"Train: {len(dataset['train'])} rows | "
                f"Eval: {len(dataset['test']) if dataset.get('test') else 0} rows")

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

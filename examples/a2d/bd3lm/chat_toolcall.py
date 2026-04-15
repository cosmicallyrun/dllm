"""
Interactive chat using ConstrainedBD3LMSampler.

Automatically detects <tool_call_intent> in the <think> block and JSON-constrains
the tool call block using the FSM technique from arXiv:2508.10111.

Usage
-----
python examples/a2d/bd3lm/chat_toolcall.py \
    --model_name_or_path .models/a2d/Qwen3-0.6B/bd3lm/toolcall/checkpoint-final

With a strict JSON schema:
python examples/a2d/bd3lm/chat_toolcall.py \
    --model_name_or_path .models/a2d/Qwen3-0.6B/bd3lm/toolcall/checkpoint-final \
    --json_schema '{"type":"object","properties":{"name":{"type":"string"},"arguments":{"type":"object"}},"required":["name","arguments"]}'
"""

import json
import sys
from dataclasses import dataclass, field

import transformers

import dllm
from dllm.core.samplers.bd3lm_constrained import (
    ConstrainedBD3LMSampler,
    ConstrainedBD3LMSamplerConfig,
)


@dataclass
class ScriptArguments:
    model_name_or_path: str = ".models/a2d/Qwen3-0.6B/bd3lm/toolcall/checkpoint-final"
    seed: int = 42
    visualize: bool = True
    json_schema: str = ""  # JSON string of schema, or "" for unconstrained JSON

    def __post_init__(self):
        self.model_name_or_path = dllm.utils.resolve_with_base_env(
            self.model_name_or_path, "BASE_MODELS_DIR"
        )


@dataclass
class SamplerConfig(ConstrainedBD3LMSamplerConfig):
    steps: int = 128
    max_new_tokens: int = 256
    block_size: int = 64
    temperature: float = 0.0
    remasking: str = "low_confidence"
    right_shift_logits: bool = False
    tool_call_intent_token: str = "<tool_call_intent>"
    think_end_token: str = "</think>"


def main():
    parser = transformers.HfArgumentParser((ScriptArguments, SamplerConfig))
    script_args, sampler_config = parser.parse_args_into_dataclasses()
    transformers.set_seed(script_args.seed)

    model     = dllm.utils.get_model(model_args=script_args).eval()
    tokenizer = dllm.utils.get_tokenizer(model_args=script_args)

    # Parse JSON schema if provided
    json_schema = None
    if script_args.json_schema:
        try:
            json_schema = json.loads(script_args.json_schema)
        except json.JSONDecodeError as e:
            print(f"[WARNING] Invalid --json_schema: {e}. Falling back to generic JSON.")

    sampler = ConstrainedBD3LMSampler(model=model, tokenizer=tokenizer)
    sampler._constraint_engine = None  # will be lazy-init with schema on first call
    if json_schema is not None:
        sampler_config.json_schema = json_schema

    print("\n" + "=" * 70)
    print("Constrained BD3LM Tool-Call Chat")
    print("  • <tool_call_intent> in <think> → JSON-constrained next block")
    print("  • No intent → normal free-form generation")
    print("=" * 70)

    messages = []
    while True:
        try:
            print("\n[You]: ", end="", flush=True)
            user_msg = input().strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            sys.exit(0)

        messages.append({"role": "user", "content": user_msg})

        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            enable_thinking=True,
        )

        outputs = sampler.sample([inputs], sampler_config, return_dict=True)
        reply_tokens = outputs.sequences[0].tolist()
        # Trim prompt prefix
        reply_tokens = reply_tokens[len(inputs):]
        reply = tokenizer.decode(reply_tokens, skip_special_tokens=False).strip()

        print(f"\n[Assistant]:\n{reply}\n")
        messages.append({"role": "assistant", "content": reply})

        if script_args.visualize and outputs.histories:
            try:
                viz = dllm.utils.TerminalVisualizer(tokenizer=tokenizer)
                viz.visualize(outputs.histories, rich=True)
            except Exception as e:
                print(f"(Visualization skipped: {e})")


if __name__ == "__main__":
    main()

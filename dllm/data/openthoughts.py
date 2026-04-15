"""
Loader for open-thoughts/OpenThoughts-114k.

Schema:
  system        str   — system prompt
  conversations list  — [{from: "user"|"assistant", value: str}]

The assistant value has thinking embedded as:
  <think>reasoning...</think>actual response

We parse that into the standard messages format with reasoning_content
so the Qwen3 chat template can render it correctly with enable_thinking=True.
"""

import re

from datasets import DatasetDict, load_dataset


_THINK_RE = re.compile(r"<think>(.*?)</think>(.*)", re.DOTALL)


def _parse_assistant_value(value: str) -> tuple[str | None, str]:
    """
    Split '<think>thinking</think>response' into (thinking, response).
    Returns (None, value) if no think block found.
    """
    m = _THINK_RE.match(value.strip())
    if m:
        thinking = m.group(1).strip()
        response = m.group(2).strip()
        return thinking, response
    return None, value.strip()


def load_dataset_openthoughts(dataset_name_or_path: str) -> DatasetDict:
    """
    Load OpenThoughts-114k and convert to standard messages format.

    Output rows have:
        messages: [
            {"role": "system",    "content": "..."},   # optional
            {"role": "user",      "content": "..."},
            {"role": "assistant",
             "reasoning_content": "...",               # the <think> block
             "content":           "..."},              # the actual response
        ]
    """
    dataset = load_dataset(dataset_name_or_path)

    def map_fn(example):
        messages = []

        system = example.get("system", "").strip()
        if system:
            messages.append({"role": "system", "content": system})

        for turn in example.get("conversations", []):
            role  = turn.get("from", "")
            value = turn.get("value", "")

            if role == "user":
                messages.append({"role": "user", "content": value})
            elif role == "assistant":
                thinking, response = _parse_assistant_value(value)
                msg = {"role": "assistant", "content": response}
                if thinking:
                    msg["reasoning_content"] = thinking
                messages.append(msg)

        return {"messages": messages}

    dataset = dataset.map(
        map_fn,
        remove_columns=dataset["train"].column_names,
        num_proc=1,
        desc="Parsing OpenThoughts conversations",
    )
    return dataset


if __name__ == "__main__":
    ds = load_dataset_openthoughts("open-thoughts/OpenThoughts-114k")
    print(ds)
    print(ds["train"][0]["messages"])

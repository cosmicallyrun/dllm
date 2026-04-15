#!/bin/bash
# RunPod setup: AR → BD3LM Diffusion with reasoning + tool-call training
#
# Usage:
#   bash /workspace/dllm/scripts/runpod_setup.sh [MODEL_SIZE] [DATASET]
#
# MODEL_SIZE:
#   0.6B  — Qwen3-0.6B  (trains on 1x RTX 4090 24GB,  ~$0.44/hr)
#   1.7B  — Qwen3-1.7B  (trains on 1x A100 40GB,       ~$2.00/hr)
#
# DATASET (reasoning stage):
#   s1k          — simplescaling/s1K  (1K rows, ~30 min, smoke test)
#   openthoughts — OpenThoughts-114k 20K subset + Alpaca 3K (~3 hrs)
#
# Required env vars (set before running):
#   WANDB_API_KEY   — from wandb.ai/authorize
#   HF_TOKEN        — HuggingFace token (if downloading gated models)
#
# Examples:
#   WANDB_API_KEY=xxx bash runpod_setup.sh 0.6B openthoughts
#   WANDB_API_KEY=xxx bash runpod_setup.sh 0.6B s1k

set -e

MODEL_SIZE="${1:-0.6B}"
DATASET="${2:-s1k}"
WANDB_PROJECT="${WANDB_PROJECT:-dllm-diffusion}"

# ── Guard: wandb key required ────────────────────────────────────────────────
if [[ -z "$WANDB_API_KEY" ]]; then
    echo "ERROR: set WANDB_API_KEY before running."
    echo "  export WANDB_API_KEY=<your-key-from-wandb.ai/authorize>"
    exit 1
fi

# ── Model config ─────────────────────────────────────────────────────────────
if [[ "$MODEL_SIZE" == "1.7B" ]]; then
    HF_MODEL="Qwen/Qwen3-1.7B"
    A2D_DIR="/workspace/dllm/.models/a2d/Qwen3-1.7B"
    REASONING_OUT="/workspace/dllm/.models/a2d/Qwen3-1.7B/bd3lm/reasoning"
    TOOLCALL_OUT="/workspace/dllm/.models/a2d/Qwen3-1.7B/bd3lm/toolcall"
    BATCH=4
    GRAD_ACCUM=4
    EXTRA_ARGS="--gradient_checkpointing true"
elif [[ "$MODEL_SIZE" == "0.8B" ]]; then
    HF_MODEL="Qwen/Qwen3.5-0.8B"
    A2D_DIR="/workspace/dllm/.models/a2d/Qwen3.5-0.8B"
    REASONING_OUT="/workspace/dllm/.models/a2d/Qwen3.5-0.8B/bd3lm/reasoning"
    TOOLCALL_OUT="/workspace/dllm/.models/a2d/Qwen3.5-0.8B/bd3lm/toolcall"
    BATCH=8
    GRAD_ACCUM=2
    EXTRA_ARGS=""
else
    HF_MODEL="Qwen/Qwen3-0.6B"
    A2D_DIR="/workspace/dllm/.models/a2d/Qwen3-0.6B"
    REASONING_OUT="/workspace/dllm/.models/a2d/Qwen3-0.6B/bd3lm/reasoning"
    TOOLCALL_OUT="/workspace/dllm/.models/a2d/Qwen3-0.6B/bd3lm/toolcall"
    BATCH=8
    GRAD_ACCUM=2
    EXTRA_ARGS=""
fi

# ── Dataset config ────────────────────────────────────────────────────────────
if [[ "$DATASET" == "openthoughts" ]]; then
    REASONING_DATA="open-thoughts/OpenThoughts-114k[train:20000] + tatsu-lab/alpaca[train:3000]"
else
    REASONING_DATA="simplescaling/s1K + tatsu-lab/alpaca[train:3000]"
fi
# Joint dataset: reasoning + tool-call (w/ real thinking) + retention all mixed
JOINT_DATA="${REASONING_DATA} + Jofthomas/hermes-function-calling-thinking-V1 + NousResearch/hermes-function-calling-v1[train:5000] + HuggingFaceH4/ultrachat_200k[train:2000]"

RUN_ID="${MODEL_SIZE}-${DATASET}-$(date +%Y%m%d-%H%M)"

echo "============================================================"
echo "  Model:       $HF_MODEL"
echo "  Run ID:      $RUN_ID"
echo "  W&B project: $WANDB_PROJECT"
echo "  Joint data:  $JOINT_DATA"
echo "============================================================"

# ── 0. System deps ────────────────────────────────────────────────────────────
apt-get update -qq && apt-get install -y -qq git wget curl screen rsync

# ── 1. Repo check ─────────────────────────────────────────────────────────────
if [ ! -d /workspace/dllm ]; then
    echo "ERROR: /workspace/dllm not found. Upload the repo first:"
    echo "  rsync -avz --exclude='.git' ./ root@<POD_IP>:/workspace/dllm/ -e 'ssh -p <PORT>'"
    exit 1
fi
cd /workspace/dllm

# ── 2. Python deps ────────────────────────────────────────────────────────────
pip install -e ".[optional]" --quiet
pip install outlines-core wandb --quiet

# Log into wandb (non-interactive, using API key from env)
wandb login "$WANDB_API_KEY" --relogin

# ── 3. Download base AR model ─────────────────────────────────────────────────
echo "Downloading $HF_MODEL ..."
python - <<PYEOF
from huggingface_hub import snapshot_download
import os
snapshot_download(
    repo_id="$HF_MODEL",
    local_dir="/workspace/models/$HF_MODEL",
    ignore_patterns=["*.msgpack","*.h5","flax_model*"],
    token=os.environ.get("HF_TOKEN"),
)
print("Download complete.")
PYEOF

# ── 4. Convert AR → Diffusion ─────────────────────────────────────────────────
echo "Converting AR → Bidirectional Diffusion ..."
python dllm/pipelines/a2d/convert.py \
    --model_name_or_path "/workspace/models/$HF_MODEL" \
    --output_dir "$A2D_DIR"
echo "Converted → $A2D_DIR"

# ── 5. Joint SFT: reasoning + tool-call + retention in one run ───────────────
JOINT_OUT="/workspace/dllm/.models/a2d/${MODEL_SIZE}/bd3lm/joint"

echo "Joint SFT (reasoning + tool-call + retention) ..."
WANDB_PROJECT="$WANDB_PROJECT" \
WANDB_RUN_NAME="${RUN_ID}-joint" \
WANDB_NOTES="BD3LM joint SFT | model=$HF_MODEL | data=$JOINT_DATA" \
WANDB_TAGS="joint,reasoning,toolcall,$MODEL_SIZE,$DATASET" \
accelerate launch \
    --config_file scripts/accelerate_configs/ddp.yaml \
    --num_processes 1 \
    examples/a2d/bd3lm/sft_joint.py \
    --model_name_or_path "$A2D_DIR" \
    --output_dir "$JOINT_OUT" \
    --dataset_args "$JOINT_DATA" \
    --block_size 64 \
    --max_length 1024 \
    --num_train_epochs 3 \
    --learning_rate 5e-5 \
    --per_device_train_batch_size $BATCH \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --logging_steps 10 \
    --eval_steps 100 \
    --save_steps 200 \
    --report_to wandb \
    $EXTRA_ARGS

echo ""
echo "============================================================"
echo "  Done!  Run ID: $RUN_ID"
echo "  W&B: https://wandb.ai/$WANDB_PROJECT"
echo "  Checkpoint: $JOINT_OUT/checkpoint-final"
echo ""
echo "  Test reasoning:"
echo "    python examples/a2d/bd3lm/chat.py \\"
echo "      --model_name_or_path $JOINT_OUT/checkpoint-final"
echo ""
echo "  Test tool-calling:"
echo "    python examples/a2d/bd3lm/chat_toolcall.py \\"
echo "      --model_name_or_path $JOINT_OUT/checkpoint-final"
echo "============================================================"

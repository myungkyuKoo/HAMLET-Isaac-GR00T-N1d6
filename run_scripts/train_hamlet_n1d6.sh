#!/usr/bin/env bash
# GR00T N1.6 + HAMLET fine-tune on RoboMME -- turn the VLA into a history-aware policy.
# Usage: DATASET_PATH=/path/to/robomme bash run_scripts/train_hamlet_n1d6.sh
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# config (override via env)
DATASET_PATH="${DATASET_PATH:?set DATASET_PATH to your RoboMME dataset directory}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/robomme/hamlet_n1d6}"
BASE_MODEL="${BASE_MODEL:-nvidia/GR00T-N1.6-3B}"
NUM_GPUS="${NUM_GPUS:-4}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
MAX_STEPS="${MAX_STEPS:-60000}"
SAVE_STEPS="${SAVE_STEPS:-30000}"
MASTER_PORT="${MASTER_PORT:-$(( 20000 + RANDOM % 10000 ))}"

# HAMLET memory options
K="${K:-4}"                                   # memory window = history length
MEMORY_STRIDE="${MEMORY_STRIDE:-16}"          # env steps between snapshots; set equal to the eval n_action_steps
N_MOMENT_TOKENS="${N_MOMENT_TOKENS:-4}"       # moment tokens per step (n_q)
MEM_COND_TYPE="${MEM_COND_TYPE:-cross_attn}"  # cross_attn | adaln
MEMORY_TYPE="${MEMORY_TYPE:-moment_token}"    # moment_token | vision_feature
LOAD_MOMENT_TOKENS_FROM="${LOAD_MOMENT_TOKENS_FROM:-}"  # optional Stage-1 (TCL) ckpt; see README "Moment-token initialization"
FREEZE_MOMENT_TOKENS="${FREEZE_MOMENT_TOKENS:-0}"       # 1 = freeze moment tokens (paper recipe when TCL-initialized)

MOMENT_ARGS=()
if [ "$FREEZE_MOMENT_TOKENS" = "1" ]; then MOMENT_ARGS+=(--freeze-moment-tokens); else MOMENT_ARGS+=(--no-freeze-moment-tokens); fi
[ -n "$LOAD_MOMENT_TOKENS_FROM" ] && MOMENT_ARGS+=(--load-moment-tokens-from "$LOAD_MOMENT_TOKENS_FROM")

torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" \
    gr00t/experiment/launch_finetune.py \
    --base-model-path "$BASE_MODEL" \
    --dataset-path "$DATASET_PATH" \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path gr00t/configs/data/robomme_config.py \
    --num-gpus "$NUM_GPUS" \
    --output-dir "$OUTPUT_DIR" \
    --max-steps "$MAX_STEPS" \
    --global-batch-size "$GLOBAL_BATCH_SIZE" \
    --save-steps "$SAVE_STEPS" \
    --hamlet-mode finetune \
    --n-moment-tokens "$N_MOMENT_TOKENS" \
    --memory-window "$K" \
    --memory-stride "$MEMORY_STRIDE" \
    --memory-num-layers 2 \
    --mem-cond-type "$MEM_COND_TYPE" \
    --memory-type "$MEMORY_TYPE" \
    "${MOMENT_ARGS[@]}"

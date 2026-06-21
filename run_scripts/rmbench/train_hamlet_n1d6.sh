#!/usr/bin/env bash
# GR00T N1.6 + HAMLET fine-tune -- turn the VLA into a history-aware policy.
# Usage: DATASET_PATH=/path/to/rmbench bash run_scripts/rmbench/train_hamlet_n1d6.sh
#   RMBench modality (14-D abs-joint / 3-view / action-horizon 50) is preset (rmbench_config.py).
#   For the memory window K=8, also set GRAD_ACCUM=2 to offset the ~2x activation memory.
#
# Single-stage by default: moment tokens are randomly initialized and trained end-to-end (no TCL-initialization).
# To use the optional two-stage paper recipe instead, first run a Stage-1 TCL job (--hamlet-mode tcl),
# then point LOAD_MOMENT_TOKENS_FROM at its checkpoint and set FREEZE_MOMENT_TOKENS=1.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

# config (override via env)
DATASET_PATH="${DATASET_PATH:?set DATASET_PATH to your benchmark dataset directory}"
MODALITY_CONFIG="${MODALITY_CONFIG:-gr00t/configs/data/rmbench_config.py}"  # robomme_config.py | rmbench_config.py
OUTPUT_DIR="${OUTPUT_DIR:-runs/rmbench/hamlet_n1d6}"
BASE_MODEL="${BASE_MODEL:-nvidia/GR00T-N1.6-3B}"
NUM_GPUS="${NUM_GPUS:-4}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"                  # set 2 for K=8 (offsets ~2x memory; keeps effective batch)
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
    --modality-config-path "$MODALITY_CONFIG" \
    --num-gpus "$NUM_GPUS" \
    --output-dir "$OUTPUT_DIR" \
    --max-steps "$MAX_STEPS" \
    --global-batch-size "$GLOBAL_BATCH_SIZE" \
    --gradient-accumulation-steps "$GRAD_ACCUM" \
    --save-steps "$SAVE_STEPS" \
    --hamlet-mode finetune \
    --n-moment-tokens "$N_MOMENT_TOKENS" \
    --memory-window "$K" \
    --memory-stride "$MEMORY_STRIDE" \
    --memory-num-layers 2 \
    --mem-cond-type "$MEM_COND_TYPE" \
    --memory-type "$MEMORY_TYPE" \
    "${MOMENT_ARGS[@]}"

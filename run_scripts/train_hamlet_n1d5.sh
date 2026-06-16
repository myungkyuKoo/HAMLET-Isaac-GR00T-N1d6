#!/usr/bin/env bash
# GR00T N1.5 + HAMLET fine-tune -- turn the VLA into a history-aware policy.
# Usage: DATASET_PATH=/path/to/robocasa bash run_scripts/train_hamlet_n1d5.sh
#
# Single-stage by default: moment tokens are randomly initialized and trained end-to-end
# (no TCL-initialization). To use the optional two-stage paper recipe instead, first run a
# Stage-1 TCL job (--hamlet-mode tcl), then point LOAD_MOMENT_TOKENS_FROM at its checkpoint
# and set FREEZE_MOMENT_TOKENS=1. See README "Moment-token initialization (TCL, optional)".
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# config (override via env)
DATASET_PATH="${DATASET_PATH:?set DATASET_PATH to your RoboCasa-Kitchen dataset directory}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/robocasa/hamlet_n1d5}"
BASE_MODEL="${BASE_MODEL:-nvidia/GR00T-N1.5-3B}"
NUM_GPUS="${NUM_GPUS:-4}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
MAX_STEPS="${MAX_STEPS:-60000}"
SAVE_STEPS="${SAVE_STEPS:-30000}"
REPORT_TO="${REPORT_TO:-wandb}"                  # wandb | tensorboard

# HAMLET memory options
K="${K:-4}"                                      # memory window (history length); stride auto-bound to action chunk
N_MOMENT_TOKENS="${N_MOMENT_TOKENS:-4}"          # moment tokens per step (n_q)
MEM_COND_TYPE="${MEM_COND_TYPE:-cross_attn}"     # cross_attn | adaln
MEMORY_TYPE="${MEMORY_TYPE:-moment_token}"       # moment_token | vision_feature
LOAD_MOMENT_TOKENS_FROM="${LOAD_MOMENT_TOKENS_FROM:-}"  # optional Stage-1 (TCL) ckpt; see README "Moment-token initialization"
FREEZE_MOMENT_TOKENS="${FREEZE_MOMENT_TOKENS:-0}"       # 1 = freeze moment tokens (paper recipe when TCL-initialized)

MOMENT_ARGS=()
if [ "$FREEZE_MOMENT_TOKENS" = "1" ]; then MOMENT_ARGS+=(--freeze-moment-tokens); else MOMENT_ARGS+=(--no-freeze-moment-tokens); fi
[ -n "$LOAD_MOMENT_TOKENS_FROM" ] && MOMENT_ARGS+=(--load-moment-tokens-from "$LOAD_MOMENT_TOKENS_FROM")

python scripts/gr00t_finetune.py \
    --dataset-path "$DATASET_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --data-config single_panda_gripper_hamlet \
    --embodiment-tag new_embodiment \
    --base-model-path "$BASE_MODEL" \
    --num-gpus "$NUM_GPUS" \
    --batch-size "$BATCH_SIZE" \
    --gradient-accumulation-steps "$GRAD_ACCUM" \
    --max-steps "$MAX_STEPS" \
    --save-steps "$SAVE_STEPS" \
    --report-to "$REPORT_TO" \
    --hamlet-mode finetune \
    --n-moment-tokens "$N_MOMENT_TOKENS" \
    --memory-window "$K" \
    --memory-num-layers 2 \
    --mem-cond-type "$MEM_COND_TYPE" \
    --memory-type "$MEMORY_TYPE" \
    "${MOMENT_ARGS[@]}"

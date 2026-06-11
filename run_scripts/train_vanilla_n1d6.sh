#!/usr/bin/env bash
# Vanilla GR00T N1.6 fine-tune on RoboMME (no HAMLET) -- baseline for HAMLET comparisons.
# Usage: DATASET_PATH=/path/to/robomme bash run_scripts/train_vanilla_n1d6.sh
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# config (override via env)
DATASET_PATH="${DATASET_PATH:?set DATASET_PATH to your RoboMME dataset directory}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/robomme/vanilla_n1d6}"
BASE_MODEL="${BASE_MODEL:-nvidia/GR00T-N1.6-3B}"
NUM_GPUS="${NUM_GPUS:-4}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
MAX_STEPS="${MAX_STEPS:-60000}"
SAVE_STEPS="${SAVE_STEPS:-30000}"
MASTER_PORT="${MASTER_PORT:-$(( 20000 + RANDOM % 10000 ))}"

torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" \
    gr00t/experiment/launch_finetune.py \
    --base-model-path "$BASE_MODEL" \
    --dataset-path "$DATASET_PATH" \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path gr00t/configs/data/robomme_config.py \
    --hamlet-mode off \
    --num-gpus "$NUM_GPUS" \
    --output-dir "$OUTPUT_DIR" \
    --max-steps "$MAX_STEPS" \
    --global-batch-size "$GLOBAL_BATCH_SIZE" \
    --save-steps "$SAVE_STEPS"

#!/usr/bin/env bash
# Vanilla GR00T N1.5 fine-tune (no HAMLET) -- baseline for HAMLET comparisons.
# Usage: DATASET_PATH=/path/to/robocasa bash run_scripts/train_vanilla_n1d5.sh
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# config (override via env)
DATASET_PATH="${DATASET_PATH:?set DATASET_PATH to your RoboCasa-Kitchen dataset directory}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/robocasa/vanilla_n1d5}"
BASE_MODEL="${BASE_MODEL:-nvidia/GR00T-N1.5-3B}"
NUM_GPUS="${NUM_GPUS:-4}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_STEPS="${MAX_STEPS:-60000}"
SAVE_STEPS="${SAVE_STEPS:-30000}"
REPORT_TO="${REPORT_TO:-wandb}"                  # wandb | tensorboard

python scripts/gr00t_finetune.py \
    --dataset-path "$DATASET_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --data-config single_panda_gripper \
    --embodiment-tag new_embodiment \
    --base-model-path "$BASE_MODEL" \
    --hamlet-mode off \
    --num-gpus "$NUM_GPUS" \
    --batch-size "$BATCH_SIZE" \
    --max-steps "$MAX_STEPS" \
    --save-steps "$SAVE_STEPS" \
    --report-to "$REPORT_TO"

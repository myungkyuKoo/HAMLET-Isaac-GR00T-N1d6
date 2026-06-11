#!/usr/bin/env bash
# Evaluate a (vanilla or HAMLET) GR00T N1.6 checkpoint on RoboMME.
# Serves the policy (gr00t/eval/run_gr00t_server.py) and drives the RoboMME simulator
# with the rollout client over a local socket, one task at a time. The RoboMME
# benchmark is external; set ROBOMME_PYTHON to its venv python (see README "Evaluation").
#
# Usage:
#   MODEL_PATH=/path/to/checkpoint-60000 \
#   ROBOMME_PYTHON=/path/to/robomme_benchmark/venv/bin/python \
#   bash run_scripts/eval_n1d6.sh
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# config (override via env)
MODEL_PATH="${MODEL_PATH:?set MODEL_PATH to your checkpoint dir (.../checkpoint-N)}"
ROBOMME_PYTHON="${ROBOMME_PYTHON:?set ROBOMME_PYTHON to the RoboMME benchmark venv python (see README)}"
ROLLOUT="${ROLLOUT:-$REPO_ROOT/gr00t/eval/sim/robomme/run_robomme_rollout.py}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/eval/robomme}"
PORT="${PORT:-$(( 20000 + RANDOM % 40000 ))}"
DATASET_SPLIT="${DATASET_SPLIT:-test}"
N_EPISODES="${N_EPISODES:-50}"
N_ACTION_STEPS="${N_ACTION_STEPS:-16}"
MAX_EP_STEPS="${MAX_EP_STEPS:-1300}"
ONLY_TASKS="${ONLY_TASKS:-}"                                 # optional CSV subset of tasks
# Deterministic single-seed eval (flow-matching noise from a fixed generator).
export GR00T_INFERENCE_SEED="${GR00T_INFERENCE_SEED:-6}"
export MUJOCO_GL="${MUJOCO_GL:-egl}" PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

TASKS=(BinFill PickXtimes SwingXtimes StopCube
       VideoUnmask VideoUnmaskSwap ButtonUnmask ButtonUnmaskSwap
       PickHighlight VideoRepick VideoPlaceButton VideoPlaceOrder
       MoveCube InsertPeg PatternLock RouteStick)

run_task() {
    local task="$1" out="$OUTPUT_DIR/$task"
    mkdir -p "$out"
    echo "[eval] robomme / $task  (port $PORT)"
    python gr00t/eval/run_gr00t_server.py \
        --model-path "$MODEL_PATH" --embodiment-tag NEW_EMBODIMENT \
        --use-sim-policy-wrapper --host 127.0.0.1 --port "$PORT" &
    local serve_pid=$!
    sleep 60
    # The rollout client runs in the RoboMME benchmark venv (it imports the simulator),
    # with PYTHONPATH=REPO_ROOT so it can import this repo's policy client.
    PYTHONPATH="$REPO_ROOT" "$ROBOMME_PYTHON" "$ROLLOUT" --task-id "$task" \
        --policy-client-host 127.0.0.1 --policy-client-port "$PORT" \
        --dataset "$DATASET_SPLIT" --n-episodes "$N_EPISODES" \
        --max-episode-steps "$MAX_EP_STEPS" --n-action-steps "$N_ACTION_STEPS" \
        --model-config "$MODEL_PATH" \
        --output-dir "$out" || true
    kill "$serve_pid" 2>/dev/null || true
    sleep 3
}

for t in "${TASKS[@]}"; do
    if [ -n "$ONLY_TASKS" ] && [[ ",$ONLY_TASKS," != *",$t,"* ]]; then continue; fi
    run_task "$t"
done

echo "[eval] done. Aggregate with: python gr00t/eval/sim/robomme/aggregate_eval_summary.py $OUTPUT_DIR"

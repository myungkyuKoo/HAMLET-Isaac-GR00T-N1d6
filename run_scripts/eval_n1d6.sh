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
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# config (override via env)
MODEL_PATH="${MODEL_PATH:?set MODEL_PATH to your checkpoint dir (.../checkpoint-N)}"
MODEL_PATH="${MODEL_PATH%/}"
ROBOMME_PYTHON="${ROBOMME_PYTHON:?set ROBOMME_PYTHON to the RoboMME benchmark venv python (see README)}"
ROLLOUT="${ROLLOUT:-$REPO_ROOT/gr00t/eval/sim/robomme/run_robomme_rollout.py}"
MODEL_TAG="$(basename "$(dirname "$MODEL_PATH")")-$(basename "$MODEL_PATH")"
OUTPUT_DIR="${OUTPUT_DIR:-runs/eval/robomme/$MODEL_TAG}"
PORT="${PORT:-$(( 20000 + RANDOM % 40000 ))}"
DATASET_SPLIT="${DATASET_SPLIT:-test}"
N_EPISODES="${N_EPISODES:-50}"
N_ACTION_STEPS="${N_ACTION_STEPS:-16}"
MAX_EP_STEPS="${MAX_EP_STEPS:-1300}"
ONLY_TASKS="${ONLY_TASKS:-}"
SERVER_TIMEOUT="${SERVER_TIMEOUT:-300}"

# Deterministic single-seed eval (flow-matching noise from a fixed generator).
export GR00T_INFERENCE_SEED="${GR00T_INFERENCE_SEED:-6}"
export MUJOCO_GL="${MUJOCO_GL:-egl}" PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

TASKS=(BinFill PickXtimes SwingXtimes StopCube
       VideoUnmask VideoUnmaskSwap ButtonUnmask ButtonUnmaskSwap
       PickHighlight VideoRepick VideoPlaceButton VideoPlaceOrder
       MoveCube InsertPeg PatternLock RouteStick)
FAILED_TASKS=()

wait_for_server() {
    local pid="$1"
    local deadline=$(( SECONDS + SERVER_TIMEOUT ))
    local state
    while (( SECONDS < deadline )); do
        state="$(ps -o stat= -p "$pid" 2>/dev/null || true)"
        state="${state// /}"
        case "$state" in ""|Z*) return 1 ;; esac
        if (: < "/dev/tcp/127.0.0.1/$PORT") 2>/dev/null; then return 0; fi
        sleep 2
    done
    return 1
}

run_task() {
    local task="$1"
    local out="$OUTPUT_DIR/$task"
    mkdir -p "$out"
    echo "[eval] robomme / $task  (port $PORT)"
    python gr00t/eval/run_gr00t_server.py \
        --model-path "$MODEL_PATH" --embodiment-tag NEW_EMBODIMENT \
        --use-sim-policy-wrapper --host 127.0.0.1 --port "$PORT" &
    local serve_pid=$!
    if ! wait_for_server "$serve_pid"; then
        echo "[eval] ERROR: $task - policy server not ready within ${SERVER_TIMEOUT}s (or exited early)" >&2
        kill "$serve_pid" 2>/dev/null || true
        wait "$serve_pid" 2>/dev/null || true
        return 1
    fi
    # The rollout client runs in the RoboMME benchmark venv (it imports the simulator),
    # with PYTHONPATH=REPO_ROOT so it can import this repo's policy client.
    local rc=0
    PYTHONPATH="$REPO_ROOT" "$ROBOMME_PYTHON" "$ROLLOUT" --task-id "$task" \
        --policy-client-host 127.0.0.1 --policy-client-port "$PORT" \
        --dataset "$DATASET_SPLIT" --n-episodes "$N_EPISODES" \
        --max-episode-steps "$MAX_EP_STEPS" --n-action-steps "$N_ACTION_STEPS" \
        --model-config "$MODEL_PATH" \
        --output-dir "$out" || rc=$?
    kill "$serve_pid" 2>/dev/null || true
    wait "$serve_pid" 2>/dev/null || true
    sleep 3
    if (( rc != 0 )); then
        echo "[eval] ERROR: $task - rollout client exited with code $rc" >&2
        return 1
    fi
    if [ ! -s "$out/simulation_results.csv" ]; then
        echo "[eval] ERROR: $task - rollout produced no simulation_results.csv" >&2
        return 1
    fi
}

for t in "${TASKS[@]}"; do
    if [ -n "$ONLY_TASKS" ] && [[ ",$ONLY_TASKS," != *",$t,"* ]]; then continue; fi
    run_task "$t" || FAILED_TASKS+=("$t")
done

if (( ${#FAILED_TASKS[@]} > 0 )); then
    echo "[eval] FAILED tasks (${#FAILED_TASKS[@]}): ${FAILED_TASKS[*]}" >&2
    exit 1
fi
echo "[eval] done. Aggregate with: python gr00t/eval/sim/robomme/aggregate_eval_summary.py $OUTPUT_DIR"

#!/usr/bin/env bash
# Evaluate a (vanilla or HAMLET) GR00T N1.5 checkpoint on RoboCasa-Kitchen.
# Serves the policy (run_gr00t_server_n1d5.py) and drives the simulator with the
# vendored rollout client over a local socket, one task at a time. The RoboCasa
# simulator is external; set BENCH_PYTHON to its venv (see README "Evaluation").
#
# Usage:
#   MODEL_PATH=/path/to/checkpoint-60000 \
#   BENCH_PYTHON=/path/to/robocasa/venv/bin/python \
#   bash run_scripts/eval_n1d5.sh
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# config (override via env)
MODEL_PATH="${MODEL_PATH:?set MODEL_PATH to your checkpoint dir (.../checkpoint-N)}"
BENCH_PYTHON="${BENCH_PYTHON:?set BENCH_PYTHON to the RoboCasa venv python (see README)}"
DATA_CONFIG="${DATA_CONFIG_OVERRIDE:-single_panda_gripper}"   # HAMLET ckpt: single_panda_gripper_hamlet
ROLLOUT="${ROLLOUT:-$REPO_ROOT/gr00t/eval/rollout_policy.py}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/eval/robocasa}"
PORT="${PORT:-$(( 20000 + RANDOM % 40000 ))}"
N_EPISODES="${N_EPISODES:-50}"
N_ACTION_STEPS="${N_ACTION_STEPS:-16}"
MAX_EP_STEPS="${MAX_EP_STEPS:-1000}"
ONLY_TASKS="${ONLY_TASKS:-}"                                 # optional CSV subset of tasks
export MUJOCO_GL="${MUJOCO_GL:-egl}" PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

TASKS=(TurnSinkSpout TurnOnStove TurnOnSinkFaucet TurnOnMicrowave TurnOffStove
       TurnOffSinkFaucet TurnOffMicrowave PnPStoveToCounter PnPSinkToCounter
       PnPMicrowaveToCounter PnPCounterToStove PnPCounterToSink PnPCounterToMicrowave
       PnPCounterToCab PnPCabToCounter OpenSingleDoor OpenDrawer OpenDoubleDoor
       CoffeeSetupMug CoffeeServeMug CoffeePressButton CloseSingleDoor CloseDrawer
       CloseDoubleDoor)

run_task() {
    local task="$1" out="$OUTPUT_DIR/$task"
    mkdir -p "$out"
    echo "[eval] robocasa / $task  (port $PORT)"
    PYTHONPATH="$REPO_ROOT" python scripts/run_gr00t_server_n1d5.py \
        --model-path "$MODEL_PATH" --data-config "$DATA_CONFIG" \
        --embodiment-tag new_embodiment --host 127.0.0.1 --port "$PORT" &
    local serve_pid=$!
    sleep 90
    PYTHONPATH="$REPO_ROOT" "$BENCH_PYTHON" "$ROLLOUT" --n-episodes "$N_EPISODES" \
        --policy-client-host 127.0.0.1 --policy-client-port "$PORT" \
        --max-episode-steps "$MAX_EP_STEPS" \
        --env-name "robocasa_panda_omron/${task}_PandaOmron_Env" \
        --n-action-steps "$N_ACTION_STEPS" --n-envs 1 --video-dir "$out" || true
    kill "$serve_pid" 2>/dev/null || true
    sleep 3
}

for t in "${TASKS[@]}"; do
    if [ -n "$ONLY_TASKS" ] && [[ ",$ONLY_TASKS," != *",$t,"* ]]; then continue; fi
    run_task "$t"
done

echo "[eval] done. Aggregate with: python gr00t/eval/sim/robocasa/aggregate_eval_summary.py $OUTPUT_DIR"

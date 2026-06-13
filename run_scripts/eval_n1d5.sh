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

# HAMLET train/inference parity: the rolling memory stride is bound to the
# trained action chunk, so the eval replanning interval (N_ACTION_STEPS) must
# equal the checkpoint's action_horizon. The server/client split means the
# rollout client never receives MODEL_PATH, so the in-client parity guard
# (gr00t/eval/rollout_policy.py) cannot run -- enforce it here at launch.
if [ -f "$MODEL_PATH/config.json" ]; then
    python - "$MODEL_PATH/config.json" "$N_ACTION_STEPS" <<'PY' || exit 1
import json, sys
cfg_path, n_action_steps = sys.argv[1], int(sys.argv[2])
try:
    cfg = json.load(open(cfg_path))
except Exception as exc:
    print(f"[eval][warn] could not parse {cfg_path}: {exc} - parity check skipped.")
    sys.exit(0)
if cfg.get("hamlet_mode") == "finetune":
    ah = cfg.get("action_horizon")
    if ah is not None and int(ah) != n_action_steps:
        sys.exit(
            f"[eval][ERROR] N_ACTION_STEPS ({n_action_steps}) != trained "
            f"action_horizon ({ah}) from {cfg_path}.\n"
            "The HAMLET memory stride is bound to the action chunk, so train and "
            f"inference must match. Re-run with N_ACTION_STEPS={ah}."
        )
PY
fi

FAILED_TASKS=()   # tasks whose rollout client exited non-zero
RAN_TASKS=0

TASKS=(TurnSinkSpout TurnOnStove TurnOnSinkFaucet TurnOnMicrowave TurnOffStove
       TurnOffSinkFaucet TurnOffMicrowave PnPStoveToCounter PnPSinkToCounter
       PnPMicrowaveToCounter PnPCounterToStove PnPCounterToSink PnPCounterToMicrowave
       PnPCounterToCab PnPCabToCounter OpenSingleDoor OpenDrawer OpenDoubleDoor
       CoffeeSetupMug CoffeeServeMug CoffeePressButton CloseSingleDoor CloseDrawer
       CloseDoubleDoor)

run_task() {
    # NOTE: keep these on separate lines -- in `local a=$1 b=$a` bash expands both
    # RHS *before* assigning, so `$task` would be unset (fatal under `set -u`).
    local task="$1"
    local out="$OUTPUT_DIR/$task"
    mkdir -p "$out"
    echo "[eval] robocasa / $task  (port $PORT)"
    PYTHONPATH="$REPO_ROOT" python scripts/run_gr00t_server_n1d5.py \
        --model-path "$MODEL_PATH" --data-config "$DATA_CONFIG" \
        --embodiment-tag new_embodiment --host 127.0.0.1 --port "$PORT" &
    local serve_pid=$!
    sleep 90
    # Capture the rollout exit code instead of swallowing it with `|| true`, so a
    # broken server/sim/client surfaces instead of masquerading as a finished run.
    local rc=0
    PYTHONPATH="$REPO_ROOT" "$BENCH_PYTHON" "$ROLLOUT" --n-episodes "$N_EPISODES" \
        --policy-client-host 127.0.0.1 --policy-client-port "$PORT" \
        --max-episode-steps "$MAX_EP_STEPS" \
        --env-name "robocasa_panda_omron/${task}_PandaOmron_Env" \
        --n-action-steps "$N_ACTION_STEPS" --n-envs 1 --video-dir "$out" || rc=$?
    kill "$serve_pid" 2>/dev/null || true
    sleep 3
    RAN_TASKS=$((RAN_TASKS + 1))
    if [ "$rc" -ne 0 ]; then
        echo "[eval][FAIL] $task -- rollout exited $rc"
        FAILED_TASKS+=("$task")
    fi
}

for t in "${TASKS[@]}"; do
    if [ -n "$ONLY_TASKS" ] && [[ ",$ONLY_TASKS," != *",$t,"* ]]; then continue; fi
    run_task "$t"
done

AGG="python gr00t/eval/sim/robocasa/aggregate_eval_summary.py $OUTPUT_DIR"
if [ "${#FAILED_TASKS[@]}" -gt 0 ]; then
    echo "[eval][ERROR] ${#FAILED_TASKS[@]}/${RAN_TASKS} task(s) failed: ${FAILED_TASKS[*]}"
    echo "[eval] aggregate whatever completed with: $AGG"
    exit 1
fi
echo "[eval] done -- all ${RAN_TASKS} task(s) completed. Aggregate with: $AGG"

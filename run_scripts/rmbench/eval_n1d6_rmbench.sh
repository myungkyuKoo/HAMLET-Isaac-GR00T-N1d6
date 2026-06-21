#!/usr/bin/env bash
# Evaluate a (vanilla or HAMLET) GR00T N1.6 checkpoint on RMBench.
# Serves the policy (gr00t/eval/run_gr00t_server.py) and drives the RMBench simulator via RMBench's own
# harness (script/eval_policy.py) + our in-repo policy plugin (gr00t/eval/sim/rmbench/policy/), one task at
# a time. The RMBench benchmark is external; set RMBENCH_ROOT to its checkout and RMBENCH_PYTHON to its
# python (install steps + the warp/setuptools fixes are in README "RMBench › Evaluation").
#
# Usage:
#   MODEL_PATH=/path/to/checkpoint-60000 \
#   RMBENCH_ROOT=/path/to/RMBench RMBENCH_PYTHON=/path/to/RMBench/conda/bin/python \
#   bash run_scripts/rmbench/eval_n1d6_rmbench.sh
# Optional env: RM_EVAL_TEST_NUM (episodes/task, default 25), ONLY_TASKS, GR00T_INFERENCE_SEED.
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

MODEL_PATH="${MODEL_PATH:?set MODEL_PATH to your checkpoint dir (.../checkpoint-N)}"; MODEL_PATH="${MODEL_PATH%/}"
RMBENCH_ROOT="${RMBENCH_ROOT:?set RMBENCH_ROOT to your RMBench checkout (see header)}"; RMBENCH_ROOT="${RMBENCH_ROOT%/}"
RMBENCH_PYTHON="${RMBENCH_PYTHON:?set RMBENCH_PYTHON to the RMBench conda env python}"
MODEL_TAG="$(basename "$(dirname "$MODEL_PATH")")-$(basename "$MODEL_PATH")"
OUTPUT_DIR="${OUTPUT_DIR:-runs/eval/rmbench/$MODEL_TAG}"
PORT="${PORT:-$(( 20000 + RANDOM % 40000 ))}"
SEED="${SEED:-0}"
SERVER_TIMEOUT="${SERVER_TIMEOUT:-600}"
export GR00T_INFERENCE_SEED="${GR00T_INFERENCE_SEED:-6}"
export RM_EVAL_TEST_NUM="${RM_EVAL_TEST_NUM:-25}"
PNAME=gr00t_hamlet_rmbench   # policy plugin name inside RMBENCH_ROOT/policy/
mkdir -p "$OUTPUT_DIR"

# Official 9 RMBench tasks (place_block_mat is a repo extra, not in the paper -> excluded).
TASKS=(observe_and_pickup rearrange_blocks put_back_block swap_blocks swap_T \
       battery_try blocks_ranking_try cover_blocks press_button)
ONLY_TASKS="${ONLY_TASKS:-}"

# --- install our policy plugin into the RMBench checkout (so its eval_policy.py can import it) ---
ln -sfn "$REPO_ROOT/gr00t/eval/sim/rmbench/policy" "$RMBENCH_ROOT/policy/$PNAME"
# --- make eval_policy.py honor RM_EVAL_TEST_NUM (idempotent; default stays 100 if unset) ---
EP="$RMBENCH_ROOT/script/eval_policy.py"
if grep -q '^    test_num = 100$' "$EP" 2>/dev/null; then
    sed -i 's/^    test_num = 100$/    test_num = int(os.environ.get("RM_EVAL_TEST_NUM", 100))  # patched: episodes\/task/' "$EP"
    echo "[eval] patched $EP to honor RM_EVAL_TEST_NUM"
fi

wait_server() {
    local deadline=$(( SECONDS + SERVER_TIMEOUT ))
    while (( SECONDS < deadline )); do
        if ! kill -0 "$1" 2>/dev/null; then return 1; fi
        if (: < "/dev/tcp/127.0.0.1/$PORT") 2>/dev/null; then return 0; fi
        sleep 3
    done
    return 1
}

FAILED_TASKS=()
run_task() {
    local task="$1"
    echo "[eval] rmbench / $task  (port $PORT, ${RM_EVAL_TEST_NUM} ep)"
    python gr00t/eval/run_gr00t_server.py \
        --model-path "$MODEL_PATH" --embodiment-tag NEW_EMBODIMENT \
        --use-sim-policy-wrapper --host 127.0.0.1 --port "$PORT" &
    local spid=$!
    if ! wait_server "$spid"; then
        echo "[eval] ERROR: $task - policy server not ready within ${SERVER_TIMEOUT}s (or exited early)" >&2
        kill "$spid" 2>/dev/null; return 1
    fi
    ( cd "$RMBENCH_ROOT" && PYTHONWARNINGS=ignore::UserWarning RM_EVAL_TEST_NUM="$RM_EVAL_TEST_NUM" \
        "$RMBENCH_PYTHON" script/eval_policy.py --config "policy/$PNAME/deploy_policy.yml" \
        --overrides --task_name "$task" --task_config demo_clean \
        --ckpt_setting "$MODEL_TAG" --seed "$SEED" --policy_name "$PNAME" --server_port "$PORT" )
    local rc=$?
    kill "$spid" 2>/dev/null
    # collect: symlink the latest RMBench result dir (videos + _result.txt) under OUTPUT_DIR/<task>
    local res
    res=$(ls -dt "$RMBENCH_ROOT/eval_result/$task/$PNAME/demo_clean/$MODEL_TAG"/*/ 2>/dev/null | head -1)
    if [ -n "$res" ] && [ -f "${res%/}/_result.txt" ]; then
        ln -sfn "${res%/}" "$OUTPUT_DIR/$task"
    else
        echo "[eval] ERROR: $task - no _result.txt produced" >&2; return 1
    fi
    return $rc
}

for t in "${TASKS[@]}"; do
    if [ -n "$ONLY_TASKS" ] && [[ ",$ONLY_TASKS," != *",$t,"* ]]; then continue; fi
    run_task "$t" || FAILED_TASKS+=("$t")
done

echo "[eval] aggregating -> $OUTPUT_DIR"
python gr00t/eval/sim/rmbench/aggregate_eval_summary.py "$OUTPUT_DIR" || true

if (( ${#FAILED_TASKS[@]} > 0 )); then
    echo "[eval] FAILED tasks: ${FAILED_TASKS[*]}" >&2
    exit 1
fi
echo "[eval] done -> $OUTPUT_DIR"

#!/usr/bin/env bash
# Build the RoboCasa-Kitchen simulator environment used by the rollout client
# (gr00t/eval/rollout_policy.py). Produces a self-contained venv at
#   gr00t/eval/sim/robocasa/venv
# whose python is what you pass as BENCH_PYTHON to run_scripts/eval_n1d5.sh.
#
# Notes:
#   - RoboCasa must be the GR00T fork (github.com/squarefk/robocasa): it adds
#     robocasa/utils/gym_utils/gymnasium_groot.py, which registers the
#     `robocasa_panda_omron/<Task>_PandaOmron_Env` gymnasium environments the
#     rollout client uses. Stock robocasa/robocasa does NOT have these.
#   - The venv needs no GR00T model stack (no transformers/flash-attn): the
#     rollout client only talks to the policy server over ZMQ. (robosuite pulls
#     its own torch for IK; that torch is unused by the GR00T client code.)
#   - Kitchen assets (~5 GB) are downloaded into the robocasa checkout.
#
# Usage:  bash gr00t/eval/sim/robocasa/setup_robocasa.sh
#   ROBOCASA_REPO=<dir>  override the robocasa checkout location
#                        (default: <repo>/external_dependencies/robocasa)
#   VENV_DIR=<dir>       override the venv location (default: alongside this script)
set -euxo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/../../../.." && pwd )"
ROBOCASA_REPO="${ROBOCASA_REPO:-$REPO_ROOT/external_dependencies/robocasa}"
VENV_DIR="${VENV_DIR:-$SCRIPT_DIR/venv}"
ROBOCASA_GIT_URL="${ROBOCASA_GIT_URL:-https://github.com/squarefk/robocasa}"

# 1) RoboCasa fork (GR00T gym registration included)
if [ ! -d "$ROBOCASA_REPO/.git" ]; then
    git clone "$ROBOCASA_GIT_URL" "$ROBOCASA_REPO"
fi

# 2) Python 3.10 venv (uv if available, stdlib venv otherwise)
if command -v uv >/dev/null 2>&1; then
    uv venv --clear "$VENV_DIR" --python 3.10
    PIP=(uv pip install --python "$VENV_DIR/bin/python")
else
    python3.10 -m venv "$VENV_DIR"
    "$VENV_DIR/bin/pip" install --upgrade pip setuptools wheel
    PIP=("$VENV_DIR/bin/pip" install)
fi

# 3) Simulator stack
"${PIP[@]}" setuptools wheel
"${PIP[@]}" "numpy<2" "git+https://github.com/ARISE-Initiative/robosuite.git@master"
"${PIP[@]}" -e "$ROBOCASA_REPO" --config-settings editable_mode=compat

# 4) Rollout-client deps (gr00t itself is reached via PYTHONPATH; see eval_n1d5.sh)
"${PIP[@]}" "gymnasium==0.29.1" "av==15.0.0" opencv-python-headless \
    msgpack pyzmq pandas tyro tqdm pydantic

# 5) Kitchen assets (~5 GB). The fork's downloader has no -y flag (answer the
# "Proceed?" prompt on stdin; herestring, not `yes |`, which dies of SIGPIPE
# under pipefail) and re-downloads unconditionally - so skip it entirely when
# the asset folders are already in place.
ASSETS="$ROBOCASA_REPO/robocasa/models/assets"
if [ -d "$ASSETS/textures" ] && [ -d "$ASSETS/fixtures" ] && \
   [ -d "$ASSETS/objects/objaverse" ] && [ -d "$ASSETS/generative_textures" ]; then
    echo "[setup_robocasa] kitchen assets already present - skipping download."
else
    "$VENV_DIR/bin/python" "$ROBOCASA_REPO/robocasa/scripts/download_kitchen_assets.py" <<< 'y'
fi

# 6) Sanity check: imports always; env construction only where EGL is available
PYTHONPATH="$REPO_ROOT" "$VENV_DIR/bin/python" - <<'PY'
import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import gymnasium as gym
import robosuite, robocasa
import robocasa.utils.gym_utils.gymnasium_groot  # noqa: F401  (registers robocasa_panda_omron/*)
import gr00t.eval.rollout_policy  # noqa: F401  (client imports resolve without torch)
print("Imports OK: robosuite", robosuite.__version__)
try:
    env = gym.make("robocasa_panda_omron/OpenSingleDoor_PandaOmron_Env", enable_render=True)
    print("Env OK:", type(env))
    env.close()
except Exception as exc:  # login nodes often lack EGL; the eval job re-validates
    print(f"[warn] env construction failed on this node ({exc}); rerun on a GPU node.")
PY

echo "[setup_robocasa] done. BENCH_PYTHON=$VENV_DIR/bin/python"

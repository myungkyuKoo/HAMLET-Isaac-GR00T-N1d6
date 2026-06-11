<div align="center">

# HAMLET on GR00T&nbsp;N1.5

### HAMLET: Switch your Vision-Language-Action Model into a History-Aware Policy
### ICLR 2026

[![arXiv](https://img.shields.io/badge/arXiv-2510.00695-b31b1b.svg)](https://arxiv.org/abs/2510.00695v3)
[![Project Page](https://img.shields.io/badge/Project-Page-1f72b1.svg)](https://myungkyukoo.github.io/hamlet/)
[![N1.6 implementation](https://img.shields.io/badge/GR00T-N1.6%20version-success.svg)](https://github.com/myungkyuKoo/HAMLET-Isaac-GR00T-N1d6)

</div>

<p align="center">
  <img src="assets/overview.png" width="95%" alt="HAMLET overview"/>
</p>

> **This repository** is the official **HAMLET implementation on top of NVIDIA GR00T&nbsp;N1.5**.<br>
> The GR00T&nbsp;**N1.6** version lives in a separate repo: **[HAMLET-Isaac-GR00T-N1d6](https://github.com/myungkyuKoo/HAMLET-Isaac-GR00T-N1d6)**.<br>
> Paper: [arXiv:2510.00695](https://arxiv.org/abs/2510.00695v3) · Project page: [myungkyukoo.github.io/hamlet](https://myungkyukoo.github.io/hamlet/)

## 🧠 What is HAMLET?

Most Vision-Language-Action (VLA) models are **Markovian**: they act from the *current* observation only and forget the past. **HAMLET** turns a pre-trained VLA into a **history-aware policy** with a small, efficient memory mechanism, without retraining the VLM backbone:

- **Moment tokens**: a few learnable query tokens are appended to the VLM input each step; their post-LLM hidden states summarize "what happened now."
- **Memory module**: a lightweight block-causal transformer aggregates the moment tokens over a window of past steps (oldest to current).
- **Action conditioning**: the memory-augmented summary is injected into the action head, so action prediction is conditioned on history.

This repo applies HAMLET to **GR00T&nbsp;N1.5** and evaluates on **RoboCasa-Kitchen**.

## ✨ This implementation

The HAMLET layer is exposed through a few CLI flags on top of the standard GR00T finetune entrypoint (`scripts/gr00t_finetune.py`):

| Flag | Choices (default) | Meaning |
|---|---|---|
| `--hamlet-mode` | `off` \| `tcl` \| `finetune` (finetune) | enable HAMLET (Stage-2 fine-tune) or TCL pretraining; `off` is vanilla GR00T |
| `--n-moment-tokens` | int (4) | moment tokens per step (`n_q`) |
| `--memory-window` | int (4) | history length `K` (timestep blocks); raise it to condition on longer past context. The window spans ≈ `(K-1) × action_chunk` control steps (e.g., `K=4` with a 16-step chunk ≈ 48 steps ≈ 4.8 s at 10 Hz) |
| `--memory-num-layers` | int (2) | depth of the memory transformer module |
| `--mem-cond-type` | `cross_attn` \| `adaln` (cross_attn) | how memory conditions the action head: concat into the cross-attention KV, **or** mean-pool into the DiT timestep embedding (AdaLN-zero) |
| `--memory-type` | `moment_token` \| `vision_feature` (moment_token) | what flows through the memory module: learnable **moment tokens** (paper), or the **primary-view image tokens** (post-LLM, pooled to 64/step) *(optional; not in the paper, but may help with low-level spatial memory)* |
| `--load-moment-tokens-from` | path (none) | warm-start `backbone.moment_tokens` from a Stage-1 (TCL) checkpoint dir or safetensors file |
| `--freeze-moment-tokens` / `--no-freeze-moment-tokens` | (script default: no-freeze) | freeze the moment-token parameter during the HAMLET fine-tune (paper recipe when TCL-initialized) |

**Memory stride.** The gap between the `K` cached snapshots is *not* a free knob: it is bound to the action chunk (`len(action_indices)`), since at inference the rolling memory cache advances once per policy call (one executed action chunk). It is derived automatically, so train and inference stay consistent.

**Memory attention mask.** The memory transformer module is **block-causal**: tokens *within* a timestep block attend bidirectionally, and blocks attend causally across time (a step sees its own and earlier steps, never the future).

**Moment-token initialization (TCL, optional).** Following the paper, the moment tokens can be warm-started with time-contrastive learning before the HAMLET fine-tune, then loaded (and optionally frozen) in Stage 2:

```bash
# Stage 1 — TCL pretraining of the moment tokens (VLM frozen)
python scripts/gr00t_finetune.py ... --hamlet-mode tcl --n-moment-tokens 4

# Stage 2 — HAMLET fine-tune, warm-starting from the Stage-1 tokens
python scripts/gr00t_finetune.py ... --hamlet-mode finetune \
    --load-moment-tokens-from <stage1-output>/checkpoint-<N> \
    --freeze-moment-tokens
```

`--load-moment-tokens-from` accepts a checkpoint directory or a `model*.safetensors` file and copies **only** `backbone.moment_tokens` (everything else still initializes from `--base-model-path`). Without it, moment tokens are randomly initialized and trained end-to-end — the default in `run_scripts/train_hamlet_n1d5.sh` (override via the `LOAD_MOMENT_TOKENS_FROM` / `FREEZE_MOMENT_TOKENS=1` environment variables).


## ⚙️ Setup

Same conda + pip environment as NVIDIA Isaac-GR00T N1.5 (Python 3.10, CUDA 12.4 recommended):

```bash
conda create -n gr00t python=3.10
conda activate gr00t
pip install --upgrade setuptools
pip install -e .[base]
pip install --no-build-isolation flash-attn==2.7.1.post4
```

The base VLM is **`nvidia/GR00T-N1.5-3B`** (downloaded automatically by `--base-model-path` on first run).

All training / eval scripts read paths and knobs from **environment variables**, so they run unchanged on any machine (no paths are hard-coded).

## 📦 Dataset

We provide an example **RoboCasa-Kitchen** dataset, in LeRobot format on the Hugging Face Hub. It is the `single_panda_gripper` data from NVIDIA's [`PhysicalAI-Robotics-GR00T-X-Embodiment-Sim`](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim), subsampled to 300 demos/task:

| Benchmark | Hugging Face dataset |
|---|---|
| RoboCasa-Kitchen (24 tasks) | [`kimtaey/robocasa_mg_gr00t_300`](https://huggingface.co/datasets/kimtaey/robocasa_mg_gr00t_300) |

```bash
huggingface-cli download --repo-type dataset kimtaey/robocasa_mg_gr00t_300 --local-dir data/robocasa
```

## 🏋️ Training

Two scripts cover everything; all options are environment variables (see the config block at the top of each script).

```bash
# vanilla GR00T N1.5 baseline
DATASET_PATH=data/robocasa bash run_scripts/train_vanilla_n1d5.sh

# GR00T N1.5 + HAMLET
DATASET_PATH=data/robocasa bash run_scripts/train_hamlet_n1d5.sh
```

HAMLET options (env vars on `train_hamlet_n1d5.sh`): `K` (memory window), `MEM_COND_TYPE` (`cross_attn` | `adaln`), `MEMORY_TYPE` (`moment_token` | `vision_feature`). Examples:

```bash
MEM_COND_TYPE=adaln            DATASET_PATH=data/robocasa bash run_scripts/train_hamlet_n1d5.sh
K=8 MEMORY_TYPE=vision_feature DATASET_PATH=data/robocasa bash run_scripts/train_hamlet_n1d5.sh
```

**Compute.** On 4× GPUs (global batch 32, 60k steps), our RoboCasa-Kitchen runs took ≈ 5–6 h for vanilla N1.5 and ≈ 19 h for N1.5 + HAMLET (K=4).

## 🧪 Evaluation

Evaluation uses a **policy-server / rollout-client** split: this repo serves the trained GR00T policy over a local socket (`scripts/run_gr00t_server_n1d5.py`), and a rollout client drives the simulator and queries the server. The rollout client is **included in this repo** (`gr00t/eval/rollout_policy.py`); only the **simulator** is external and lives in its own environment.

**1) Install the benchmark simulator (separate environment).**

| Benchmark | Repository | Notes |
|---|---|---|
| RoboCasa-Kitchen | <https://github.com/robocasa/robocasa> | RoboCasa + robosuite; create a dedicated venv per the repo's instructions. |

Set `BENCH_PYTHON` to that environment's python. The rollout client runs in it (with this repo on `PYTHONPATH`, handled by the script). `ROLLOUT` defaults to the in-repo client and only needs setting if you relocate it.

**2) Run eval** (orchestrates server + client over all tasks):

```bash
# RoboCasa (HAMLET ckpt: also pass DATA_CONFIG_OVERRIDE=single_panda_gripper_hamlet)
MODEL_PATH=runs/robocasa/hamlet_n1d5/checkpoint-60000 \
BENCH_PYTHON=/path/to/robocasa/.venv/bin/python \
OUTPUT_DIR=runs/eval/robocasa \
bash run_scripts/eval_n1d5.sh
```

`ONLY_TASKS=OpenDrawer,CloseDrawer` restricts to a subset of tasks.

**3) Aggregate** per-task results into suite + overall success rates:

```bash
python gr00t/eval/sim/robocasa/aggregate_eval_summary.py  runs/eval/robocasa
```

## 📁 Repository layout

```
gr00t/                     core GR00T package + HAMLET additions
  model/memory.py          HAMLET memory transformer (block-causal) module
  model/gr00t_n1.py        model wiring: moment-token / vision-feature memory paths, AdaLN-zero pool
  model/tcl_head.py        Stage-1 TCL head
  model/backbone/          Eagle backbone (moment-token injection, primary-view feature)
  policy/server_client.py  ZMQ policy client used by the rollout client
  eval/rollout_policy.py   RoboCasa rollout client
  eval/sim/robocasa/       RoboCasa env wrappers + result aggregation
scripts/                   python entrypoints
  gr00t_finetune.py        finetune entrypoint
  run_gr00t_server_n1d5.py policy server for evaluation
run_scripts/               runnable bash launchers
  train_vanilla_n1d5.sh    vanilla GR00T N1.5 training
  train_hamlet_n1d5.sh     GR00T N1.5 + HAMLET training (all options via env vars)
  eval_n1d5.sh             eval orchestration (server + rollout client)
```

## 📝 Notes

- All paths/knobs are environment variables; nothing is machine-specific.
- Checkpoints, rollouts (`*.mp4`), logs, and datasets are git-ignored.
- The rollout client is included; only the RoboCasa simulator package is external. See Evaluation.
- HAMLET serving supports two cache modes: pass per-sample `session_ids` (+ `reset_memory` flags) for **session-isolated** rolling caches (safe under client interleaving), or omit them for the legacy row-keyed cache (single client, fixed vector-env row order).

## 🙏 Acknowledgements

Built on **[NVIDIA Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T)** (`n1.5-release` tag). The base model, license, and core VLA stack are NVIDIA's; HAMLET adds the memory module and its training/evaluation. See [`LICENSE`](LICENSE).

## 📚 Citation

```bibtex
@inproceedings{koo2026hamlet,
  title={{HAMLET}: Switch Your Vision-Language-Action Model into a History-Aware Policy},
  author={Myungkyu Koo and Daewon Choi and Taeyoung Kim and Kyungmin Lee and Changyeon Kim and Younggyo Seo and Jinwoo Shin},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
  url={https://openreview.net/forum?id=KcJ9U0x6kO}
}
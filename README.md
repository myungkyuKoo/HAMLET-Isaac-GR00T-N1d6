<div align="center">

# HAMLET on Isaac GR00T&nbsp;N1.6

### HAMLET: Switch your Vision-Language-Action Model into a History-Aware Policy
### ICLR 2026

[![arXiv](https://img.shields.io/badge/arXiv-2510.00695-b31b1b.svg)](https://arxiv.org/abs/2510.00695v3)
[![Project Page](https://img.shields.io/badge/Project-Page-1f72b1.svg)](https://myungkyukoo.github.io/hamlet/)
[![N1.5 implementation](https://img.shields.io/badge/GR00T-N1.5%20version-success.svg)](https://github.com/myungkyuKoo/HAMLET-Isaac-GR00T/tree/n1.5)

</div>

<p align="center">
  <img src="assets/overview.png" width="95%" alt="HAMLET overview"/>
</p>

> **This repository** is the official **HAMLET implementation on top of NVIDIA GR00T&nbsp;N1.6**, with ready-to-run scripts for **two example benchmarks: RoboMME and RMBench**.<br>
> The GR00T&nbsp;**N1.5** version, which evaluates on **RoboCasa-Kitchen** benchmark, lives on the **[`n1.5` branch](https://github.com/myungkyuKoo/HAMLET-Isaac-GR00T/tree/n1.5)** of this repo.<br>
> Paper: [arXiv:2510.00695](https://arxiv.org/abs/2510.00695v3) · Project page: [myungkyukoo.github.io/hamlet](https://myungkyukoo.github.io/hamlet/)

## 🧠 What is HAMLET?

Most Vision-Language-Action models (VLAs) are **Markovian**: they act from the *current* observation only and forget the past. **HAMLET** turns a pre-trained VLA into a **history-aware policy** with a small, efficient memory mechanism, without retraining the VLM backbone:

- **Moment tokens**: a few learnable query tokens are appended to the VLM input each step; their post-LLM hidden states summarize "what happened now."
- **Memory module**: a lightweight block-causal transformer aggregates the moment tokens over a window of past steps (oldest to current).
- **Action conditioning**: the memory-augmented summary is injected into the action head, so action prediction is conditioned on history.

This repo applies HAMLET to **GR00T&nbsp;N1.6** and provides scripts for two example memory-manipulation benchmarks: **RoboMME** (16 tasks across four suites — Counting, Permanence, Reference, Imitation) and **RMBench** (9 dual-arm tasks across two memory-complexity tiers).

## ✨ This implementation

The HAMLET layer is exposed through a few CLI flags on top of the standard GR00T finetune entrypoint (`gr00t/experiment/launch_finetune.py`):

| Flag | Choices (default) | Meaning |
|---|---|---|
| `--hamlet-mode` | `off` \| `tcl` \| `finetune` (finetune) | enable HAMLET (Stage-2 fine-tune) or TCL pretraining; `off` is vanilla GR00T |
| `--n-moment-tokens` | int (4) | moment tokens per step (`n_q`) |
| `--memory-window` | int (4) | history length `K` (timestep blocks); raise it to condition on longer past context. The window spans ≈ `(K-1) × action_chunk` control steps (*e.g.,* `K=4` with a 16-step chunk ≈ 48 steps ≈ 4.8 s at 10 Hz) |
| `--memory-stride` | int (16) | environment steps between the `K` cached snapshots |
| `--memory-num-layers` | int (2) | depth of the memory transformer module |
| `--mem-cond-type` | `cross_attn` \| `adaln` (cross_attn) | how memory conditions the action head: concat into the cross-attention KV, **or** mean-pool into the DiT timestep embedding (AdaLN-zero) |
| `--memory-type` | `moment_token` \| `vision_feature` (moment_token) | what flows through the memory module: learnable **moment tokens** (paper), or the **primary-view image tokens** (post-LLM, pooled to 64/step) *(optional; not in the paper, but may help with low-level spatial memory)* |
| `--load-moment-tokens-from` | path (None) | warm-start `backbone.moment_tokens` from a Stage-1 (TCL) checkpoint dir or safetensors file |
| `--freeze-moment-tokens` / `--no-freeze-moment-tokens` | (script default: no-freeze) | whether to freeze the moment tokens during the Stage-2 fine-tune. **Freezing saves training cost** (the tokens stay at their TCL-pretrained values and leave the optimizer); **unfreezing (default) gives the best performance**, since the tokens keep adapting during fine-tuning. |

**Memory stride.** `--memory-stride` is the gap, in environment steps, between the `K` cached snapshots. Set it equal to the evaluation action-execution interval (`--n-action-steps`, default 16): the rolling memory cache advances once per policy call, so a matching stride keeps the training-time and inference-time history aligned.

**Memory attention mask.** The memory transformer module is **block-causal**: tokens *within* a timestep block attend bidirectionally, and blocks attend causally across time (a step sees its own and earlier steps, never the future).

**Moment-token initialization (TCL, optional).** Following the paper, the moment tokens can be warm-started with time-contrastive learning before the HAMLET fine-tune, then loaded (and optionally frozen) in Stage 2:

```bash
# Stage 1: TCL pretraining of the moment tokens (VLM frozen)
torchrun ... gr00t/experiment/launch_finetune.py ... \
    --hamlet-mode tcl --n-moment-tokens 4

# Stage 2: HAMLET fine-tune, warm-starting from the Stage-1 tokens
torchrun ... gr00t/experiment/launch_finetune.py ... \
    --hamlet-mode finetune \
    --load-moment-tokens-from <stage1-output>/checkpoint-<N> \
    --freeze-moment-tokens   # cost-saving option; omit for full performance

# Single-stage Training: You may skip TCL-initialization and randomly initialize moment tokens, so the moment tokens are trained end-to-end.
torchrun ... gr00t/experiment/launch_finetune.py ... \
    --hamlet-mode finetune \
    --n-moment-tokens 4 \
    --no-freeze-moment-tokens
```

**Freeze vs. unfreeze (Stage 2).** Freezing the moment tokens (`--freeze-moment-tokens`) lowers training cost so it is a reasonable choice when compute is tight. For **best performance, keep them unfrozen** (`--no-freeze-moment-tokens`, the default): letting the moment tokens keep adapting during the fine-tune consistently yields higher success. Treat freezing as a cost-saving option, not the setting for peak quality.

`--load-moment-tokens-from` accepts a checkpoint directory or a `model*.safetensors` file and copies **only** `backbone.moment_tokens` (everything else still initializes from `--base-model-path`). Without it, moment tokens are randomly initialized and trained end-to-end — the single-stage default in `run_scripts/<benchmark>/train_hamlet_n1d6.sh` (opt into the two-stage recipe via the `LOAD_MOMENT_TOKENS_FROM` / `FREEZE_MOMENT_TOKENS=1` environment variables).


## ⚙️ Setup

```bash
# install this repo (uv; see NVIDIA Isaac-GR00T for full prerequisites)
uv sync && uv pip install -e .
source .venv/bin/activate
```

The base VLM is **`nvidia/GR00T-N1.6-3B`** (downloaded automatically by `--base-model-path` on first run).

All training / eval scripts read paths and knobs from **environment variables**, so they run unchanged on any machine (no paths are hard-coded).

## 🧭 Example benchmarks

This repo ships ready-to-run scripts for **two example benchmarks** — **RoboMME** and **RMBench** — exercising GR00T N1.6 (± HAMLET) on different memory-manipulation suites. Each benchmark's **dataset setup, training, and evaluation** are described in turn below; both follow the same pattern (download a LeRobot dataset → `run_scripts/<benchmark>/train_*.sh` → serve the policy and drive the external simulator via `run_scripts/<benchmark>/eval_n1d6_<benchmark>.sh`). Per-benchmark scripts live under `run_scripts/robomme/` and `run_scripts/rmbench/`.

| Benchmark | Tasks | Robot / action | Scripts |
|---|---|---|---|
| [RoboMME](https://robomme.github.io/) | 16 (4 suites) | single-arm Panda · 8-D abs joint · 2 views | `run_scripts/robomme/` |
| [RMBench](https://rmbench.github.io/) | 9 (M(1)/M(n) tiers) | dual-arm Aloha-AgileX · 14-D abs joint · 3 views | `run_scripts/rmbench/` |

## 🤖 RoboMME

### Dataset

We train on the official **[RoboMME](https://robomme.github.io/)** benchmark data, provided in LeRobot format on the Hugging Face Hub:

| Benchmark | Hugging Face dataset |
|---|---|
| RoboMME (16 tasks) | [`Yinpei/robomme_data_lerobot`](https://huggingface.co/datasets/Yinpei/robomme_data_lerobot) |

```bash
huggingface-cli download --repo-type dataset Yinpei/robomme_data_lerobot --local-dir data/robomme
```

**Dataset preparation (one-time, required).** The Hub release stores camera frames *inside* the parquet files (`image`-dtype features) and ships no `videos/` directory, no `meta/modality.json`, and no `meta/stats.json` — all three of which the GR00T loader requires. After downloading, run:

```bash
# 1) transcode the parquet-embedded frames into the videos/ layout the loader reads
python gr00t/data/make_videos.py --dataset-path data/robomme

# 2) install the GR00T modality mapping (state/action slices + camera-key renames)
cp gr00t/configs/data/robomme_modality.json data/robomme/meta/modality.json

# 3) generate normalization stats (demo frames are excluded from relative-action stats)
python gr00t/data/stats.py --dataset-path data/robomme --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path gr00t/configs/data/robomme_config.py
```

**Demonstration frames.** RoboMME episodes include demonstration (watch-phase) frames. They are automatically excluded from the action loss while still populating the memory window, and at evaluation the demo frames are replayed through the policy to prime the memory before the first action.

### Training

Two scripts cover everything; all options are environment variables (see the config block at the top of each script). Training uses `torchrun` over `NUM_GPUS` (default 4). The RoboMME modality config is preset in these scripts (no `MODALITY_CONFIG` needed).

```bash
# vanilla GR00T N1.6 baseline
DATASET_PATH=data/robomme bash run_scripts/robomme/train_vanilla_n1d6.sh

# GR00T N1.6 + HAMLET
DATASET_PATH=data/robomme bash run_scripts/robomme/train_hamlet_n1d6.sh
```

HAMLET options (env vars on `run_scripts/robomme/train_hamlet_n1d6.sh`): `K` (memory window), `MEMORY_STRIDE`, `MEM_COND_TYPE` (`cross_attn` | `adaln`), `MEMORY_TYPE` (`moment_token` | `vision_feature`). Examples:

```bash
MEM_COND_TYPE=adaln            DATASET_PATH=data/robomme bash run_scripts/robomme/train_hamlet_n1d6.sh
K=8 MEMORY_TYPE=vision_feature DATASET_PATH=data/robomme bash run_scripts/robomme/train_hamlet_n1d6.sh
```

**Compute.** On 4× GPUs (global batch 32, 60k steps), our RoboMME runs took ≈ 6 h for vanilla N1.6 and ≈ 11 h for N1.6 + HAMLET (K=4).

### Evaluation

Evaluation uses a **policy-server / rollout-client** split: this repo serves the trained GR00T policy over a local socket (`gr00t/eval/run_gr00t_server.py`), and the RoboMME rollout client drives the simulator and queries the server. The rollout client is **included in this repo** (`gr00t/eval/sim/robomme/run_robomme_rollout.py`); only the **simulator** is external.

**1) Install the RoboMME benchmark (separate environment).** Follow the official instructions at [robomme.github.io](https://robomme.github.io/) (clone [`RoboMME/robomme_benchmark`](https://github.com/RoboMME/robomme_benchmark), then `uv sync && uv pip install -e .` inside it); it provides the `robomme` package (env + `BenchmarkEnvBuilder`). Set `ROBOMME_PYTHON` to that venv's python. The rollout client runs in that environment (with this repo on `PYTHONPATH`, handled by the script), so it also needs this repo's client-side deps (`msgpack`/`pyzmq` for the ZMQ policy client, `pandas` for `simulation_results.csv`), which a default benchmark sync does not install:

```bash
# inside the robomme_benchmark checkout
uv pip install -r ./run_scripts/robomme/robomme_client_requirements.txt
```

`uv sync` prunes packages that are not in the benchmark's lockfile — re-run this install after any `uv sync` there.

**2) Run eval** (orchestrates server + client over all 16 tasks):

```bash
MODEL_PATH=runs/robomme/hamlet_n1d6/checkpoint-60000 \
ROBOMME_PYTHON=/path/to/robomme_benchmark/venv/bin/python \
bash run_scripts/robomme/eval_n1d6_robomme.sh
```

`OUTPUT_DIR` defaults to `runs/eval/robomme/<run>-<checkpoint>` (here `runs/eval/robomme/hamlet_n1d6-checkpoint-60000`), and each per-task output directory is bound to its checkpoint and eval settings via a `policy_manifest.json`: resuming into results from a different checkpoint fails instead of silently reusing them. `ONLY_TASKS=BinFill,PatternLock` restricts to a subset; `GR00T_INFERENCE_SEED` fixes the flow-matching noise for deterministic eval. Each task waits for its policy server to come up (`SERVER_TIMEOUT`, default 300 s); a task fails when the server never becomes ready, the rollout client exits non-zero, or no `simulation_results.csv` appears, and the script then exits non-zero listing the failed tasks.

**3) Aggregate** per-task results into suite + overall success rates:

```bash
python gr00t/eval/sim/robomme/aggregate_eval_summary.py runs/eval/robomme/hamlet_n1d6-checkpoint-60000
```

## 🦾 RMBench

**[RMBench](https://rmbench.github.io/)** is a memory-dependent **dual-arm Aloha-AgileX** benchmark on RoboTwin 2.0 (SAPIEN). It defines **9 official tasks** in two memory-complexity tiers: **M(1)** (single memory event) — `observe_and_pickup`, `rearrange_blocks`, `put_back_block`, `swap_blocks`, `swap_T`; **M(n)** (repeated/multi-step memory) — `battery_try`, `blocks_ranking_try`, `cover_blocks`, `press_button`. The action space is **14-D absolute joint** (6-DoF + 1 gripper per arm) over **3 views** (front/left-wrist/right-wrist); the action horizon is 50. Everything benchmark-specific lives in `gr00t/configs/data/rmbench_config.py` (preset in the `run_scripts/rmbench/` scripts).

### Dataset

Provided ready-to-use (LeRobot v2.1 with `videos/`, `meta/modality.json`, and `meta/stats.json` already included — no extra preparation needed):

| Benchmark | Hugging Face dataset |
|---|---|
| RMBench (9 official tasks) | [`Myungkyu/rmbench_lerobot`](https://huggingface.co/datasets/Myungkyu/rmbench_lerobot) |

```bash
huggingface-cli download --repo-type dataset Myungkyu/rmbench_lerobot --local-dir data/rmbench
```

> **Provenance.** `Myungkyu/rmbench_lerobot` is **derived** from the official RMBench release
> [`TianxingChen/RMBench`](https://huggingface.co/datasets/TianxingChen/RMBench) (`demo_clean` split, RoboTwin-2.0
> per-episode HDF5, 50 expert demos/task). We converted it to GR00T-compatible LeRobot — selecting **only the
> paper's 9 official tasks** (the `envs/` repo also ships `place_block_mat`, which is **not** one of the paper's 9,
> so it is excluded) — reordering the 14-D action to contiguous `[L_arm6, R_arm6, L_grip, R_grip]`, mapping the
> head/left/right cameras to the 3 views, and re-encoding video as h264. It is a **format/representation
> conversion only**; the underlying trajectories are unchanged from the official release.

### Training

The RMBench modality config is preset in these scripts (no `MODALITY_CONFIG` needed).

```bash
# vanilla GR00T N1.6
DATASET_PATH=data/rmbench bash run_scripts/rmbench/train_vanilla_n1d6.sh

# GR00T N1.6 + HAMLET (K=4)
DATASET_PATH=data/rmbench bash run_scripts/rmbench/train_hamlet_n1d6.sh

# K=8 needs GRAD_ACCUM=2 (offsets the ~2x activation memory; keeps the effective batch at 32)
K=8 GRAD_ACCUM=2 DATASET_PATH=data/rmbench bash run_scripts/rmbench/train_hamlet_n1d6.sh
```

> **Observe ("hold-still") frames are kept and supervised.** Some RMBench tasks (clearest: `observe_and_pickup`) require the policy to **stay still and observe** before information is hidden — at eval, moving during the observation window is an automatic failure. These static frames are the *robot's own required behavior*, so they are **kept and supervised** (the model learns to hold, and the memory captures the soon-to-be-hidden info). RMBench has **no `is_demo`/demonstration phase**, so there is **no demo-frame exclusion and no memory priming** at eval.

### Evaluation

A policy-server + RMBench-simulator split. The trained policy is served by this repo (`gr00t/eval/run_gr00t_server.py`); RMBench's own harness drives the simulator and queries the server through our in-repo policy plugin (`gr00t/eval/sim/rmbench/policy/`). Only the **RMBench simulator** is external.

- **Install the RMBench benchmark** (separate conda env): clone [`TianxingChen/RMBench`](https://github.com/TianxingChen/RMBench) and run its `script/_install.sh` (SAPIEN + CuRobo + pytorch3d). Two fixes for current dependencies: in `envs/curobo/.../geom/sdf/world_mesh.py` use `wp.device_from_torch(...)` (the `wp.torch.device_from_torch` path was removed in warp ≥ 1.x), and pin `setuptools<81` (SAPIEN imports `pkg_resources`). Then install the client deps into that env:

```bash
pip install -r run_scripts/rmbench/rmbench_client_requirements.txt   # msgpack + pyzmq for the ZMQ policy client
```

- **Run eval** (orchestrates server + the 9 tasks; `RM_EVAL_TEST_NUM` episodes/task, default 25):

```bash
MODEL_PATH=runs/rmbench/hamlet_n1d6/checkpoint-60000 \
RMBENCH_ROOT=/path/to/RMBench RMBENCH_PYTHON=/path/to/RMBench/conda/bin/python \
bash run_scripts/rmbench/eval_n1d6_rmbench.sh
```

`ONLY_TASKS=swap_T,press_button` restricts the task set; `GR00T_INFERENCE_SEED` fixes the flow-matching noise. RMBench's native default is 100 episodes/task, which is very long (failed episodes run to the full per-task step limit) — the orchestrator patches `eval_policy.py` to honor `RM_EVAL_TEST_NUM` and defaults it to 25.

- **Aggregate** per-task results into M(1)/M(n) suite + overall success:

```bash
python gr00t/eval/sim/rmbench/aggregate_eval_summary.py runs/eval/rmbench/<MODEL_TAG>
```

For the HAMLET **memory lift**, aggregate the vanilla and the HAMLET runs and compare the OVERALL rows.

## 📁 Repository layout

```
gr00t/                          core GR00T package + HAMLET additions
  model/modules/memory.py       HAMLET memory transformer (block-causal) module
  model/modules/eagle_backbone.py  moment-token injection, primary-view feature
  model/gr00t_n1d6/             model wiring: memory paths, AdaLN-zero pool
  model/gr00t_n1d6/tcl_head.py  Stage-1 TCL head
  configs/data/robomme_config.py   RoboMME modality config (8-D joint, 2 views)
  configs/data/rmbench_config.py   RMBench modality config (14-D joint, 3 views)
  configs/data/{robomme,rmbench}_modality.json  dataset modality overlays
  eval/run_gr00t_server.py      policy server for evaluation (both benchmarks)
  eval/sim/robomme/             RoboMME rollout client + result aggregation
  eval/sim/rmbench/             RMBench policy plugin + result aggregation
  policy/server_client.py       ZMQ policy client used by the rollout client
run_scripts/                    runnable bash launchers (per benchmark)
  robomme/                      RoboMME scripts
    train_{vanilla,hamlet}_n1d6.sh   training (RoboMME modality preset)
    eval_n1d6_robomme.sh             eval orchestration (server + rollout client)
    robomme_client_requirements.txt  client deps for the RoboMME benchmark env
  rmbench/                      RMBench scripts
    train_{vanilla,hamlet}_n1d6.sh   training (RMBench modality preset)
    eval_n1d6_rmbench.sh             eval orchestration (server + RMBench harness)
    rmbench_client_requirements.txt  client deps for the RMBench benchmark env
```

## 📝 Notes

- All paths/knobs are environment variables; nothing is machine-specific.
- Checkpoints, rollouts (`*.mp4`), logs, and datasets are git-ignored.
- The eval glue (RoboMME rollout client / RMBench policy plugin) is included; only the simulator package is external. See each benchmark's Evaluation.

## 🙏 Acknowledgements

Built on **[NVIDIA Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T)** (`n1.6.1-release` tag). The base model, license, and core VLA stack are NVIDIA's; HAMLET adds the memory module and its training/evaluation. Evaluation uses the **[RoboMME](https://robomme.github.io/)** and **[RMBench](https://rmbench.github.io/)** (RoboTwin 2.0) benchmarks. See [`LICENSE`](LICENSE).

## 📚 Citation

```bibtex
@inproceedings{koo2026hamlet,
  title={{HAMLET}: Switch Your Vision-Language-Action Model into a History-Aware Policy},
  author={Myungkyu Koo and Daewon Choi and Taeyoung Kim and Kyungmin Lee and Changyeon Kim and Younggyo Seo and Jinwoo Shin},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
  url={https://openreview.net/forum?id=KcJ9U0x6kO}
}
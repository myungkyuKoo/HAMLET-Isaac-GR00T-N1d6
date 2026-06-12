#!/usr/bin/env python3
"""RoboMME single-task rollout client (runs in the robomme_benchmark environment).

Connects to a GR00T policy server (ZMQ), runs `n_episodes` episodes for one RoboMME
task, and writes simulation_results.csv plus per-episode mp4 videos (filename:
`<env_name>_envXX-episode_N-{success|failure}.mp4`).

For HAMLET inference, the demo frames returned by env.reset() are fed through the
policy sequentially to populate the memory cache before the first action, giving
training-inference parity for the watch-then-replicate task structure.
"""
from __future__ import annotations

import io
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import tyro

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
# SAPIEN/Vulkan render device: force GPU (CPU rendering would require osmesa fallback per
# robomme README troubleshooting Q2; we have GPUs allocated via SLURM so cuda is preferred).
os.environ.setdefault("SAPIEN_RENDER_DEVICE", "cuda")
# Force unbuffered stdout/stderr so traceback from sapien/mujoco C extensions is flushed
# before potential process exit.
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
print(f"[boot] PYTHONPATH={os.environ.get('PYTHONPATH','')[:120]}", flush=True)
print(f"[boot] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','')} MUJOCO_EGL_DEVICE_ID={os.environ['MUJOCO_EGL_DEVICE_ID']}", flush=True)

import imageio
from robomme.env_record_wrapper import BenchmarkEnvBuilder

MEMORY_WINDOW = 4   # K — matches HAMLET memory_window training default; ignored when vanilla
MEMORY_STRIDE = 16  # S — must match training stride (= action chunk length per HAMLET paper)


def _to_np(x):
    return x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)


def _add_red_border(frame: np.ndarray, thickness: int = 8) -> np.ndarray:
    """Return a copy of `frame` (H, W, 3 uint8) with a solid red border drawn on
    all four edges. Used to visually mark demo (watch-phase) frames in rollout
    videos so they are distinguishable from the robot's own execution frames."""
    f = np.ascontiguousarray(frame).copy()
    t = max(1, min(thickness, f.shape[0] // 2, f.shape[1] // 2))
    red = np.array([255, 0, 0], dtype=f.dtype)
    f[:t, :, :] = red
    f[-t:, :, :] = red
    f[:, :t, :] = red
    f[:, -t:, :] = red
    return f


def _build_demo_frames(env_obs, n_demo: int) -> list[np.ndarray]:
    """Build red-bordered [front | wrist] frames for the demo (watch) phase so they
    can be prepended to the rollout video. Demo frames are entries 0..n_demo-1 of the
    env_obs *_rgb_list (the final entry is the current/first execution frame). Returns
    [] when the task has no demo phase (n_demo <= 0)."""
    demo_frames = []
    for idx in range(max(0, n_demo)):
        front = _to_np(env_obs["front_rgb_list"][idx]).astype(np.uint8)
        wrist = _to_np(env_obs["wrist_rgb_list"][idx]).astype(np.uint8)
        demo_frames.append(_add_red_border(np.hstack([front, wrist])))
    return demo_frames


def _build_step_obs(env_obs, idx: int, task_goal: str) -> dict:
    """Convert robomme env obs lists at index `idx` into gr00t single-frame format."""
    front = _to_np(env_obs["front_rgb_list"][idx]).astype(np.uint8)
    wrist = _to_np(env_obs["wrist_rgb_list"][idx]).astype(np.uint8)
    joint = _to_np(env_obs["joint_state_list"][idx]).astype(np.float32)
    gripper = _to_np(env_obs["gripper_state_list"][idx]).astype(np.float32)
    if joint.ndim == 0:
        joint = joint[None]
    joint = joint.reshape(-1)[:7]
    if gripper.size == 0:
        gripper_scalar = 0.0
    else:
        gripper_scalar = float(gripper.reshape(-1)[0])
    # Server expects (B, T, H, W, C) for video and (B, T, D) for state — add both B and T dims.
    return {
        "video.front_view": front[None, None, ...],   # (1, 1, H, W, 3)
        "video.wrist_view": wrist[None, None, ...],   # (1, 1, H, W, 3)
        "state.joint_position": joint[None, None, :].astype(np.float32),                # (1, 1, 7)
        "state.gripper_position": np.array([[[gripper_scalar]]], dtype=np.float32),    # (1, 1, 1)
        "annotation.human.action.task_description": [task_goal],                        # (B=1,)
    }


def _prime_hamlet_memory(policy, env_obs, session_id, task_goal, K: int, stride: int) -> None:
    """Prime the HAMLET memory cache with K-1 strided demo frames so that, after the
    first execution-phase policy call, the cache matches the training memory window
    [t-(K-1)S, t-(K-2)S, ..., t-S, t] at anchor t = first non-demo frame.

    Trace (K=4, S=16, anchor=n_demo):
      prime 1 (F1 = obs[n_demo - 3S], reset_memory=True)  -> cache = [F1]*K
      prime 2 (F2 = obs[n_demo - 2S], reset_memory=False) -> cache = [F1, F1, F1, F2]
      prime 3 (F3 = obs[n_demo -  S], reset_memory=False) -> cache = [F1, F1, F2, F3]
      exec  1 (E1 = obs[n_demo]    , reset_memory=False) -> cache = [F1, F2, F3, E1]

    Vanilla (no HAMLET) policies ignore reset_memory/session_ids -> effective no-op.

    Raises on any prime-step failure: continuing with a partially primed window would
    silently change what the evaluation measures for memory tasks.
    """
    n_total = len(env_obs["front_rgb_list"])
    if n_total <= 1 or K <= 1:
        return
    n_demo = n_total - 1  # last entry is the current frame, rest are demo
    # K-1 prime indices at stride S, clamped to valid demo range.
    prime_indices = [max(0, min(n_demo - (K - 1 - i) * stride, n_demo - 1)) for i in range(K - 1)]
    print(f"[i] HAMLET prime: n_demo={n_demo}, K={K}, S={stride}, indices={prime_indices}")
    for i, idx in enumerate(prime_indices):
        step_obs = _build_step_obs(env_obs, idx, task_goal)
        # prime_only: cache update without flow-matching denoising, so the seeded
        # action-noise RNG stays call-aligned with non-primed (vanilla) policies.
        options = {"session_ids": [session_id], "reset_memory": [i == 0], "prime_only": True}
        try:
            policy.get_action(step_obs, options=options)
        except Exception as exc:
            raise RuntimeError(
                f"HAMLET memory priming failed at demo frame {idx} "
                f"(prime step {i + 1}/{len(prime_indices)}, session_id={session_id!r})"
            ) from exc


def _save_episode_video(frames: list[np.ndarray], video_dir: Path, env_name: str,
                       env_idx: int, episode_idx: int, success: bool) -> str:
    video_dir.mkdir(parents=True, exist_ok=True)
    result_stem = "success" if success else "failure"
    safe_name = env_name.replace("/", "_")
    filename = f"{safe_name}_env{env_idx:02d}-episode_{episode_idx}-{result_stem}.mp4"
    path = video_dir / filename
    try:
        imageio.mimsave(str(path), frames, fps=10)
    except Exception as exc:
        print(f"[warn] failed to save video {filename}: {exc}")
    return filename


def _update_csv(csv_path: Path, env_idx: int, episode_idx: int, success: bool,
                steps: int, video_filename: str, task_instruction: str = "") -> None:
    cols = ["env_idx", "episode_idx", "success", "reward", "steps", "video_path",
            "task_instruction"]
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        if "task_instruction" not in df.columns:
            df["task_instruction"] = ""
    else:
        df = pd.DataFrame(columns=cols)
    if not df.empty:
        df = df[~((df["env_idx"] == env_idx) & (df["episode_idx"] == episode_idx))]
    new_row = {
        "env_idx": env_idx,
        "episode_idx": episode_idx,
        "success": int(success),
        "reward": 0.0,
        "steps": int(steps),
        "video_path": video_filename,
        "task_instruction": task_instruction,
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(csv_path, index=False)


def _check_output_dir_manifest(out_dir: Path, identity: dict) -> None:
    """Bind `out_dir` to one policy/eval identity via policy_manifest.json.

    The resume logic counts existing episode mp4s as done, so reusing an output
    directory across checkpoints (or changed eval settings) would silently adopt
    another run's results as this run's. The first run records its identity; later
    runs must match it exactly or are refused.
    """
    manifest_path = out_dir / "policy_manifest.json"
    if manifest_path.exists():
        recorded = json.loads(manifest_path.read_text())
        if recorded != identity:
            diffs = {
                k: {"recorded": recorded.get(k), "current": identity.get(k)}
                for k in sorted(set(recorded) | set(identity))
                if recorded.get(k) != identity.get(k)
            }
            raise RuntimeError(
                f"{out_dir} already holds results for a different policy/eval identity; "
                f"refusing to resume into it. Mismatched fields: {diffs}. "
                f"Use a fresh --output-dir or remove the stale results."
            )
    else:
        manifest_path.write_text(json.dumps(identity, indent=2) + "\n")


@dataclass
class Config:
    task_id: str
    """RoboMME task name (e.g., PickXtimes)."""

    policy_client_host: str = "127.0.0.1"
    policy_client_port: int = 5555

    dataset: str = "test"
    """train|val|test split."""

    n_episodes: int = 50
    max_episode_steps: int = 1300   # unified RoboMME eval (MME-VLA ref)
    n_action_steps: int = 16        # = HAMLET stride S (train/inference parity)

    output_dir: str = ""
    """If empty, defaults to BASE/output/robomme/<CKPT>/<task_id>."""

    memory_window: int = 0
    """HAMLET memory window K for demo-frame priming (vanilla ignores).
    0 = auto: read `memory_window` from --model-config (the checkpoint's
    config.json); vanilla checkpoints (hamlet_mode != finetune) resolve to 1
    (no priming). Falls back to 4 when no model config is available."""

    memory_stride: int = MEMORY_STRIDE
    """HAMLET memory stride S — must equal training stride for train/inference parity."""

    model_config: str = ""
    """Path to the policy checkpoint dir or its config.json. Used to resolve
    memory_window adaptively when --memory-window 0 (auto)."""


def _resolve_memory_params(cfg: Config) -> tuple[int, int]:
    """Resolve the priming window K and memory stride S from the checkpoint's
    config.json, hard-failing on any detectable train/inference mismatch.

    - The rolling memory cache advances once per policy call, so
      n_action_steps MUST equal the trained memory_stride.
    - An explicit --memory-window no longer bypasses verification: when a
      model config is readable, it must agree with the trained memory_window.
    """
    mc = None
    cfg_path = None
    if cfg.model_config:
        cfg_path = Path(cfg.model_config)
        if cfg_path.is_dir():
            cfg_path = cfg_path / "config.json"
        try:
            mc = json.loads(cfg_path.read_text())
        except Exception as exc:
            print(f"[warn] could not read model config at {cfg_path}: {exc}")
            mc = None

    K = cfg.memory_window
    stride = cfg.memory_stride

    if mc is not None:
        if mc.get("hamlet_mode") == "finetune":
            trained_K = int(mc.get("memory_window", MEMORY_WINDOW))
            assert K <= 0 or K == trained_K, (
                f"--memory-window ({K}) != trained memory_window ({trained_K}) "
                f"from {cfg_path} - drop the flag or pass the trained value."
            )
            K = trained_K
            trained_stride = mc.get("memory_stride")
            if trained_stride is None:
                raise RuntimeError(
                    f"{cfg_path} is a HAMLET checkpoint without memory_stride - cannot "
                    f"verify train/inference parity against n_action_steps "
                    f"({cfg.n_action_steps}). Backfill the trained stride into the "
                    f"checkpoint config.json before evaluating."
                )
            else:
                assert int(trained_stride) == int(cfg.n_action_steps), (
                    f"n_action_steps ({cfg.n_action_steps}) != trained memory_stride "
                    f"({trained_stride}) from {cfg_path} - the rolling memory cache "
                    f"advances once per policy call, so these must match for "
                    f"train/inference parity."
                )
                stride = int(trained_stride)
        else:
            assert K <= 1, (
                f"--memory-window ({K}) given but checkpoint is vanilla "
                f"(hamlet_mode != finetune) per {cfg_path}."
            )
            K = 1  # vanilla policy -> skip priming entirely
        print(f"[i] memory_window K={K}, stride S={stride} (source: {cfg_path})")
        return K, stride

    # No readable model config: fall back to CLI/defaults (cannot verify).
    if K <= 0:
        K = MEMORY_WINDOW
        print(f"[i] memory_window K={K} (fallback default; no model config)")
    else:
        print(f"[i] memory_window K={K} (explicit CLI; no model config to verify)")
    print(f"[i] memory stride S={stride} (unverified)")
    return K, stride


def main(cfg: Config) -> None:
    from gr00t.policy.server_client import PolicyClient  # type: ignore

    memory_window, memory_stride = _resolve_memory_params(cfg)

    out_dir = Path(cfg.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "simulation_results.csv"

    _check_output_dir_manifest(
        out_dir,
        {
            "model_config": str(Path(cfg.model_config).resolve()) if cfg.model_config else "",
            "task_id": cfg.task_id,
            "dataset": cfg.dataset,
            "n_action_steps": cfg.n_action_steps,
            "max_episode_steps": cfg.max_episode_steps,
            "inference_seed": os.environ.get("GR00T_INFERENCE_SEED", ""),
        },
    )

    # Resume detection: count existing finished episodes from filenames.
    done_eps: set[int] = set()
    successes: list[bool] = []
    import re
    pat = re.compile(r".*_env00-episode_(\d+)-(success|failure)\.mp4$")
    for mp4 in out_dir.glob("*.mp4"):
        m = pat.match(mp4.name)
        if m:
            done_eps.add(int(m.group(1)))
            successes.append(m.group(2) == "success")
    if done_eps:
        print(f"[i] Resuming: {len(done_eps)} episodes already complete in {out_dir}")
    if len(done_eps) >= cfg.n_episodes:
        print("[i] All requested episodes already done; nothing to run.")
        return

    print(f"[i] Connecting to GR00T policy server @ {cfg.policy_client_host}:{cfg.policy_client_port}")
    policy = PolicyClient(host=cfg.policy_client_host, port=cfg.policy_client_port)

    print(f"[i] Building RoboMME env: task={cfg.task_id} dataset={cfg.dataset}")
    env_builder = BenchmarkEnvBuilder(
        env_id=cfg.task_id,
        dataset=cfg.dataset,
        action_space="joint_angle",
        gui_render=False,
        max_steps=cfg.max_episode_steps,
    )
    n_episodes_total = min(cfg.n_episodes, env_builder.get_episode_num())

    start_t = time.time()
    for ep in range(n_episodes_total):
        if ep in done_eps:
            print(f"[i] Skipping episode {ep} (already complete)")
            continue
        session_id = f"{cfg.task_id}_ep{ep}_{uuid.uuid4().hex[:8]}"

        env = env_builder.make_env_for_episode(ep, max_steps=cfg.max_episode_steps)
        env_obs, info = env.reset()
        task_goal = info["task_goal"][0]
        print(f"[i] ep={ep} task_goal='{task_goal[:80]}...' demo_frames={len(env_obs['front_rgb_list']) - 1}")

        _prime_hamlet_memory(policy, env_obs, session_id, task_goal, memory_window, memory_stride)

        # Prepend the demo (watch) phase to the saved video so failure analysis can see
        # what the policy was asked to replicate. Demo frames get a red border to mark
        # them as observed-not-executed; they precede the robot's own rollout frames.
        n_demo = len(env_obs["front_rgb_list"]) - 1
        frames = _build_demo_frames(env_obs, n_demo)
        success = False
        steps = 0
        cur_obs = _build_step_obs(env_obs, -1, task_goal)
        for t in range(cfg.max_episode_steps):
            options = {"session_ids": [session_id], "reset_memory": [False]}
            actions, _ = policy.get_action(cur_obs, options=options)
            # Server returns flat dict {"action.joint_position": (B?,T,7), "action.gripper_close": (B?,T,1)}.
            # RoboMME env.step expects an 8-D joint_angle action: [7 joint, 1 gripper] per step.
            jp = np.asarray(actions["action.joint_position"], dtype=np.float32)
            gp = np.asarray(actions["action.gripper_close"], dtype=np.float32)
            if jp.ndim == 3:  # (B, T, 7) — drop B
                jp, gp = jp[0], gp[0]
            if gp.ndim == 1:  # (T,) -> (T, 1)
                gp = gp[:, None]
            act_chunk = np.concatenate([jp, gp], axis=-1)  # (T, 8)
            n_apply = min(cfg.n_action_steps, act_chunk.shape[0])
            for k in range(n_apply):
                action = act_chunk[k]
                env_obs, _, terminated, truncated, info = env.step(action)
                steps += 1
                front_now = _to_np(env_obs["front_rgb_list"][-1]).astype(np.uint8)
                wrist_now = _to_np(env_obs["wrist_rgb_list"][-1]).astype(np.uint8)
                frames.append(np.hstack([front_now, wrist_now]))
                if info.get("status") in ("success", "fail", "timeout", "error"):
                    success = info.get("status") == "success"
                if bool(terminated) or bool(truncated):
                    break
            else:
                cur_obs = _build_step_obs(env_obs, -1, task_goal)
                continue
            break

        video_filename = _save_episode_video(frames, out_dir, f"robomme/{cfg.task_id}", 0, ep, success)
        _update_csv(csv_path, env_idx=0, episode_idx=ep, success=success, steps=steps,
                    video_filename=video_filename, task_instruction=task_goal)
        successes.append(success)
        print(f"[i] ep={ep} -> {'SUCCESS' if success else 'FAIL'} (steps={steps})")

        try:
            env.close()
        except Exception:
            pass

    if successes:
        rate = float(np.mean(successes))
        (out_dir / "summary.txt").write_text(
            f"Task: {cfg.task_id}\n"
            f"Episodes: {len(successes)}\n"
            f"Success rate: {rate:.4f}\n"
            f"Elapsed: {time.time() - start_t:.1f}s\n"
        )
        print(f"[i] DONE. success_rate={rate:.4f} over {len(successes)} eps")
    else:
        print("[warn] No episodes completed")


if __name__ == "__main__":
    main(tyro.cli(Config))

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
from dataclasses import dataclass, field
from functools import partial
import os
import json
from pathlib import Path
import random
import re
import time
from typing import Any, Dict, List, Tuple
import uuid

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.eval.sim.env_utils import get_embodiment_tag_from_env_name
from gr00t.eval.sim.wrapper.multistep_wrapper import MultiStepWrapper
from gr00t.policy import BasePolicy
import gymnasium as gym
import numpy as np
import pandas as pd
from tqdm import tqdm
import tyro


@dataclass
class VideoConfig:
    """Configuration for video recording settings.

    Attributes:
        video_dir: Directory to save videos (if None, no videos are saved)
        steps_per_render: Number of steps between each call to env.render() while recording
            during rollout
        fps: Frames per second for the output video
        codec: Video codec to use for compression
        input_pix_fmt: Input pixel format
        crf: Constant Rate Factor for video compression (lower = better quality)
        thread_type: Threading strategy for video encoding
        thread_count: Number of threads to use for encoding
    """

    video_dir: str | None = None
    steps_per_render: int = 2
    max_episode_steps: int = 1000  # RoboCasa-Kitchen evaluation horizon
    fps: int = 20
    codec: str = "h264"
    input_pix_fmt: str = "rgb24"
    crf: int = 22
    thread_type: str = "FRAME"
    thread_count: int = 1
    overlay_text: bool = True
    n_action_steps: int = 8


@dataclass
class MultiStepConfig:
    """Configuration for multi-step environment settings.

    Attributes:
        video_delta_indices: Indices of video observations to stack
        state_delta_indices: Indices of state observations to stack
        n_action_steps: Number of action steps to execute
        max_episode_steps: Maximum number of steps per episode
    """

    video_delta_indices: np.ndarray = field(default_factory=lambda: np.array([0]))
    state_delta_indices: np.ndarray = field(default_factory=lambda: np.array([0]))
    n_action_steps: int = 16
    max_episode_steps: int = 1000  # RoboCasa-Kitchen evaluation horizon
    terminate_on_success: bool = False


@dataclass
class WrapperConfigs:
    """Container for various environment wrapper configurations.

    Attributes:
        video: Configuration for video recording
        multistep: Configuration for multi-step processing
    """

    video: VideoConfig = field(default_factory=VideoConfig)
    multistep: MultiStepConfig = field(default_factory=MultiStepConfig)


def get_robocasa_env_fn(env_name: str, env_idx: int = 0, seed: int | None = None):
    """RoboCasa-Kitchen factory (`robocasa_panda_omron/<task>_PandaOmron_Env` envs).
    """
    def env_fn():
        os.environ["MUJOCO_GL"] = "egl"
        os.environ["PYOPENGL_PLATFORM"] = "egl"
        import robocasa  # noqa: F401
        import robocasa.utils.gym_utils.gymnasium_groot  # noqa: F401
        sd = seed if seed is not None else env_idx
        return gym.make(env_name, enable_render=True, seed=sd)

    return env_fn


def get_gym_env(env_name: str, env_idx: int, total_n_envs: int):
    """Create the benchmark environment (without wrappers)."""
    if env_name.startswith("robocasa_panda_omron/"):
        env_fn = get_robocasa_env_fn(env_name, env_idx=env_idx)
    else:
        raise ValueError(f"Invalid environment name: {env_name}")

    return env_fn()


def create_eval_env(
    env_name: str,
    env_idx: int,
    total_n_envs: int,
    wrapper_configs: WrapperConfigs,
    start_episode_id: int = 0,
    seed: int = 42,
) -> gym.Env:
    """Create a single evaluation environment with wrappers."""

    env_seed = seed + int(env_idx)
    random.seed(env_seed)
    np.random.seed(env_seed)
    print(
        f"[i] Creating environment {env_name} (env index: {env_idx}) with seed={env_seed}, "
        f"start_episode_id={start_episode_id}"
    )

    env = get_gym_env(env_name, env_idx, total_n_envs)
    if wrapper_configs.video.video_dir is not None:
        from gr00t.eval.sim.wrapper.video_recording_wrapper import (
            VideoRecorder,
            VideoRecordingWrapper,
        )

        video_recorder = VideoRecorder.create_h264(
            fps=wrapper_configs.video.fps,
            codec=wrapper_configs.video.codec,
            input_pix_fmt=wrapper_configs.video.input_pix_fmt,
            crf=wrapper_configs.video.crf,
            thread_type=wrapper_configs.video.thread_type,
            thread_count=wrapper_configs.video.thread_count,
        )
        env = VideoRecordingWrapper(
            env,
            video_recorder,
            video_dir=Path(wrapper_configs.video.video_dir),
            steps_per_render=wrapper_configs.video.steps_per_render,
            max_episode_steps=wrapper_configs.video.max_episode_steps,
            overlay_text=wrapper_configs.video.overlay_text,
            name_prefix=f"{env_name.replace('/', '_')}_env{env_idx:02d}",
            base_seed=env_seed,
            seed_stride=100000,
            start_episode_id=start_episode_id - 1,
        )

    env = MultiStepWrapper(
        env,
        video_delta_indices=wrapper_configs.multistep.video_delta_indices,
        state_delta_indices=wrapper_configs.multistep.state_delta_indices,
        n_action_steps=wrapper_configs.multistep.n_action_steps,
        max_episode_steps=wrapper_configs.multistep.max_episode_steps,
        terminate_on_success=wrapper_configs.multistep.terminate_on_success,
    )
    return env


class _RobustAsyncVectorEnv(gym.vector.AsyncVectorEnv):
    """AsyncVectorEnv that tolerates variable-shaped info arrays across envs.

    Gymnasium's default _add_info pre-allocates a numpy array based on the
    first env's value shape and then assigns subsequent envs into it.  When
    envs return differently-shaped values (e.g. variable-length contact arrays)
    the assignment raises ValueError.  We catch that and fall back to a plain
    Python list for that key so the rest of the step can proceed normally.
    """

    def _add_info(self, infos, info, env_num):
        for k, v in info.items():
            if k not in infos:
                infos[k] = [None] * self.num_envs
                infos[f"_{k}"] = np.zeros(self.num_envs, dtype=bool)
            if isinstance(infos[k], np.ndarray):
                try:
                    infos[k][env_num] = v
                except (ValueError, TypeError):
                    lst = list(infos[k])
                    lst[env_num] = v
                    infos[k] = lst
            else:
                infos[k][env_num] = v
            infos[f"_{k}"][env_num] = True
        return infos


def run_rollout_gymnasium_policy(
    env_name: str,
    policy: BasePolicy,
    wrapper_configs: WrapperConfigs,
    n_episodes: int = 10,
    n_envs: int = 1,
    video_dir: str | None = None,
    seed: int = 42,
) -> Any:
    """Run policy rollouts in parallel environments."""
    start_time = time.time()
    n_episodes = max(n_episodes, n_envs)
    print(f"Running collecting {n_episodes} episodes for {env_name} with {n_envs} vec envs")

    existing_count, existing_successes, env_to_max_ep = _load_existing_episode_metadata(n_envs, video_dir)
    print(f"[i] Detected {existing_count} recorded episode(s) across envs; existing_successes={existing_successes}")

    if n_envs > 1:
        target_per_env = int(n_episodes / n_envs)
        env_episode_counts = [
            min((env_to_max_ep.get(i, -1) + 1) if env_to_max_ep.get(i, -1) >= 0 else 0, target_per_env)
            for i in range(n_envs)
        ]
        total_existing = sum(env_episode_counts)
        if total_existing >= n_episodes:
            print("[i] Requested episodes per env already recorded. Nothing to run.")
            return env_name, existing_successes[:n_episodes], {}
        env_start_episode_ids = env_episode_counts[:]
    else:
        target_per_env = None
        if existing_count >= n_episodes:
            print("[i] Requested episodes already recorded. Nothing to run.")
            return env_name, existing_successes[:n_episodes], {}
        env_start_episode_ids = [env_to_max_ep.get(i, -1) + 1 for i in range(n_envs)]

    env_fns = [
        partial(
            create_eval_env,
            env_idx=idx,
            env_name=env_name,
            total_n_envs=n_envs,
            wrapper_configs=wrapper_configs,
            start_episode_id=env_start_episode_ids[idx],
            seed=seed,
        )
        for idx in range(n_envs)
    ]

    if n_envs == 1:
        env = gym.vector.SyncVectorEnv(env_fns)
    else:
        env = _RobustAsyncVectorEnv(
            env_fns,
            shared_memory=False,
            context="spawn",
        )

    # Storage for results
    episode_lengths = []
    current_rewards = [0] * n_envs
    current_lengths = [0] * n_envs
    current_successes = [False] * n_envs
    episode_successes = list(existing_successes)
    episode_infos = defaultdict(list)
    env_episode_indices = env_start_episode_ids.copy()
    completed_episodes = existing_count

    csv_path = None
    if video_dir is not None:
        csv_path = f"{video_dir}/simulation_results.csv"

    # Initial reset
    observations, _ = env.reset()
    policy.reset()
    i = 0

    # HAMLET inference: per-env session id + first-step flag tell the server-side
    # policy how to initialize / FIFO-update the memory cache.
    session_ids = [f"{env_name}_env{idx}_{uuid.uuid4().hex[:8]}" for idx in range(n_envs)]
    is_first_step = [True] * n_envs

    pbar = tqdm(total=n_episodes, initial=completed_episodes, desc="Episodes")
    while completed_episodes < n_episodes:
        if n_envs > 1 and target_per_env is not None and all(
            env_episode_indices[i] >= target_per_env for i in range(n_envs)
        ):
            break
        options = {"session_ids": session_ids, "reset_memory": list(is_first_step)}
        actions, _ = policy.get_action(observations, options=options)
        is_first_step = [False] * n_envs
        next_obs, rewards, terminations, truncations, env_infos = env.step(actions)
        # NOTE (FY): Currently we don't properly handle policy reset. For now, our policy are stateless,
        # but in the future if we need policy to be stateful, we need to detect env reset and call policy.reset()
        i += 1
        # Update episode tracking
        for env_idx in range(n_envs):
            if "success" in env_infos:
                env_success = env_infos["success"][env_idx]
                if isinstance(env_success, list):
                    env_success = np.any(env_success)
                elif isinstance(env_success, np.ndarray):
                    env_success = np.any(env_success)
                elif isinstance(env_success, bool):
                    env_success = env_success
                elif isinstance(env_success, int):
                    env_success = bool(env_success)
                else:
                    raise ValueError(f"Unknown success dtype: {type(env_success)}")
                current_successes[env_idx] |= bool(env_success)
            else:
                current_successes[env_idx] = False

            if "final_info" in env_infos and env_infos["final_info"][env_idx] is not None:
                env_success = env_infos["final_info"][env_idx]["success"]
                if isinstance(env_success, list):
                    env_success = any(env_success)
                elif isinstance(env_success, np.ndarray):
                    env_success = np.any(env_success)
                elif isinstance(env_success, bool):
                    env_success = env_success
                elif isinstance(env_success, int):
                    env_success = bool(env_success)
                else:
                    raise ValueError(f"Unknown success dtype: {type(env_success)}")
                current_successes[env_idx] |= bool(env_success)
            current_rewards[env_idx] += rewards[env_idx]
            current_lengths[env_idx] += 1

            # If episode ended, store results
            if terminations[env_idx] or truncations[env_idx]:
                if n_envs > 1 and target_per_env is not None and env_episode_indices[env_idx] >= target_per_env:
                    continue
                # Mark this env's next observation as a new episode -> reset memory cache row.
                is_first_step[env_idx] = True
                if "final_info" in env_infos:
                    current_successes[env_idx] |= any(env_infos["final_info"][env_idx]["success"])
                if "task_progress" in env_infos:
                    episode_infos["task_progress"].append(env_infos["task_progress"][env_idx][-1])
                if "q_score" in env_infos:
                    episode_infos["q_score"].append(np.max(env_infos["q_score"][env_idx]))
                if "valid" in env_infos:
                    episode_infos["valid"].append(all(env_infos["valid"][env_idx]))
                # Accumulate results
                episode_lengths.append(current_lengths[env_idx])
                episode_successes.append(current_successes[env_idx])
                if csv_path is not None:
                    result_stem = "success" if current_successes[env_idx] else "failure"
                    _update_prediction_csv(
                        csv_path=csv_path,
                        env_idx=env_idx,
                        episode_idx=env_episode_indices[env_idx],
                        success=current_successes[env_idx],
                        reward=current_rewards[env_idx],
                        steps=current_lengths[env_idx],
                        video_path=f"{env_name.replace('/', '_')}_env{env_idx:02d}-episode_{env_episode_indices[env_idx]}-{result_stem}.mp4",
                    )
                env_episode_indices[env_idx] += 1
                # Reset trackers for this environment.
                current_successes[env_idx] = False
                # only update completed_episodes if valid
                if "valid" in episode_infos:
                    if episode_infos["valid"][-1]:
                        completed_episodes += 1
                        pbar.update(1)
                else:
                    # envs don't return valid
                    completed_episodes += 1
                    pbar.update(1)
                current_rewards[env_idx] = 0
                current_lengths[env_idx] = 0
        observations = next_obs
    pbar.close()

    env.reset()
    env.close()
    print(f"Collecting {n_episodes} episodes took {time.time() - start_time} seconds")

    if video_dir is not None:
        with open(f"{video_dir}/summary.txt", "w") as f:
            f.write(f"Collecting {n_episodes} episodes took {time.time() - start_time:.2f} seconds\n")
            f.write(f"Success rate: {np.mean(episode_successes) if episode_successes else 0.0:.4f}\n")

    assert len(episode_successes) >= n_episodes, (
        f"Expected at least {n_episodes} episodes, got {len(episode_successes)}"
    )

    episode_infos = dict(episode_infos)  # Convert defaultdict to dict
    for key, value in episode_infos.items():
        assert len(value) == len(episode_successes), (
            f"Length of {key} is not equal to the number of episodes"
        )

    # process valid results
    if "valid" in episode_infos:
        valids = episode_infos["valid"]
        valid_idxs = np.where(valids)[0]
        episode_successes = [episode_successes[i] for i in valid_idxs]
        episode_infos = {k: [v[i] for i in valid_idxs] for k, v in episode_infos.items()}

    return env_name, episode_successes, episode_infos


def _update_prediction_csv(
    csv_path: str,
    env_idx: int,
    episode_idx: int,
    success: bool,
    reward: float,
    steps: int,
    video_path: str = "",
):
    """Append/overwrite a row keyed by (env_idx, episode_idx)."""
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame(columns=["env_idx", "episode_idx", "success", "reward", "steps", "video_path"])
    if not df.empty:
        df = df[~((df["env_idx"] == env_idx) & (df["episode_idx"] == episode_idx))]
    new_row = {
        "env_idx": env_idx,
        "episode_idx": episode_idx,
        "success": int(success),
        "reward": float(reward),
        "steps": int(steps),
        "video_path": video_path,
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(csv_path, index=False)


def _load_existing_episode_metadata(
    n_envs: int,
    video_dir: str | None,
) -> Tuple[int, List[bool], Dict[int, int]]:
    """Scan video_dir for `..._env<NN>-episode_<MM>-success|failure.mp4` to enable resume.

    Returns (existing_count, existing_successes ordered by (episode_idx, env_idx), env_to_max_ep).
    """
    if video_dir is None:
        return 0, [], {i: -1 for i in range(n_envs)}
    path = Path(video_dir)
    if not path.exists():
        return 0, [], {i: -1 for i in range(n_envs)}

    pattern = re.compile(
        r".*_env(?P<env>\d+)-episode_(?P<episode>\d+)-(?P<status>success|failure)\.mp4$"
    )
    env_to_max_ep: Dict[int, int] = {i: -1 for i in range(n_envs)}
    episodes: List[Tuple[int, int, bool]] = []
    for file in path.glob("*.mp4"):
        match = pattern.match(file.name)
        if not match:
            try:
                file.unlink()
                print(f"[i] Removed orphaned video: {file.name}")
            except OSError as exc:
                print(f"[warn] Failed to remove orphaned video {file.name}: {exc}")
            continue
        env_idx = int(match.group("env"))
        episode_idx = int(match.group("episode"))
        success_flag = match.group("status") == "success"
        if env_idx < 0 or env_idx >= n_envs:
            print(f"[warn] Out-of-range env {env_idx} for {file.name}")
            continue
        env_to_max_ep[env_idx] = max(env_to_max_ep.get(env_idx, -1), episode_idx)
        episodes.append((episode_idx, env_idx, success_flag))

    if not episodes:
        return 0, [], env_to_max_ep
    episodes.sort(key=lambda x: (x[0], x[1]))
    existing_successes = [ep[2] for ep in episodes]
    print(f"[i] Found {len(episodes)} existing episode(s)")
    for env_idx in range(n_envs):
        if env_to_max_ep[env_idx] >= 0:
            print(f"[i] Env {env_idx}: max episode index = {env_to_max_ep[env_idx]} (next will be {env_to_max_ep[env_idx] + 1})")
    return len(episodes), existing_successes, env_to_max_ep


def create_gr00t_sim_policy(
    model_path: str,
    embodiment_tag: EmbodimentTag,
    policy_client_host: str = "",
    policy_client_port: int | None = None,
) -> BasePolicy:
    if policy_client_host and policy_client_port:
        from gr00t.policy.server_client import PolicyClient

        policy = PolicyClient(host=policy_client_host, port=policy_client_port)
    else:
        raise NotImplementedError(
            "Local (in-process) policy evaluation is not supported: serve the "
            "checkpoint with scripts/run_gr00t_server_n1d5.py and pass "
            "--policy-client-host/--policy-client-port (see run_scripts/eval_n1d5.sh)."
        )
    return policy


def run_gr00t_sim_policy(
    env_name: str,
    n_episodes: int,
    max_episode_steps: int,
    model_path: str = "",
    policy_client_host: str = "",
    policy_client_port: int | None = None,
    n_envs: int = 8,
    n_action_steps: int = 8,
    video_dir: str | None = None,
    seed: int = 42,
):
    embodiment_tag = get_embodiment_tag_from_env_name(env_name)

    # HAMLET parity guard: the rolling memory cache advances once per policy
    # call, and the memory stride is bound to the action chunk at training
    # time - so the eval replanning interval must equal the trained action
    # horizon for history-aware checkpoints.
    if model_path:
        _cfg_file = Path(model_path) / "config.json"
        if _cfg_file.exists():
            try:
                _mc = json.loads(_cfg_file.read_text())
            except Exception as _exc:
                print(f"[warn] could not parse {_cfg_file}: {_exc} - HAMLET parity check skipped.")
                _mc = {}
            if _mc.get("hamlet_mode") == "finetune":
                _ah = _mc.get("action_horizon")
                assert _ah is None or int(_ah) == int(n_action_steps), (
                    f"n_action_steps ({n_action_steps}) != trained action_horizon ({_ah}) "
                    f"from {_cfg_file} - the HAMLET memory stride is bound to the action "
                    f"chunk, so these must match for train/inference parity."
                )

    if video_dir is None:
        if model_path:
            parts = model_path.split("/")
            model_slug = parts[-3] if len(parts) >= 3 else parts[-1]
            video_dir = f"/tmp/sim_eval_videos_{model_slug}_ac{n_action_steps}_{uuid.uuid4()}"
        else:
            video_dir = f"/tmp/sim_eval_videos_{env_name}_ac{n_action_steps}_{uuid.uuid4()}"
    wrapper_configs = WrapperConfigs(
        video=VideoConfig(
            video_dir=video_dir,
            max_episode_steps=max_episode_steps,
        ),
        multistep=MultiStepConfig(
            n_action_steps=n_action_steps,
            max_episode_steps=max_episode_steps,
            terminate_on_success=True,
        ),
    )

    policy = create_gr00t_sim_policy(
        model_path,
        embodiment_tag,
        policy_client_host,
        policy_client_port,
    )

    results = run_rollout_gymnasium_policy(
        env_name=env_name,
        policy=policy,
        wrapper_configs=wrapper_configs,
        n_episodes=n_episodes,
        n_envs=n_envs,
        video_dir=video_dir,
        seed=seed,
    )
    print("Video saved to: ", wrapper_configs.video.video_dir)
    return results


@dataclass
class RolloutConfig:
    """Configuration for rollout policy evaluation."""

    max_episode_steps: int = 504
    """Maximum number of steps per episode."""

    n_episodes: int = 50
    """Number of episodes to run."""

    model_path: str = ""
    """Path to model checkpoint."""

    policy_client_host: str = ""
    """Host for policy client."""

    policy_client_port: int | None = None
    """Port for policy client."""

    env_name: str = "libero_sim/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it"
    """Environment name."""

    n_envs: int = 8
    """Number of parallel environments."""

    n_action_steps: int = 8
    """Number of action steps."""

    video_dir: str | None = None
    """Directory to save videos. If None, uses /tmp/sim_eval_videos_<env>_<uuid>."""

    seed: int = 42
    """Base RNG seed (per-env seed = seed + env_idx; per-episode seed = (seed+env_idx)*100000 + ep_id)."""


if __name__ == "__main__":
    args = tyro.cli(RolloutConfig)

    # validate policy configuration
    assert (args.model_path and not (args.policy_client_host or args.policy_client_port)) or (
        not args.model_path and args.policy_client_host and args.policy_client_port is not None
    ), (
        "Invalid policy configuration: You must provide EITHER model_path OR (policy_client_host & policy_client_port), not both.\n"
        "If all 3 arguments are provided, explicitly choose one:\n"
        '  - To use policy client: set --policy-client-host and --policy-client-port, and set --model-path ""\n'
        '  - To use model path: set --model-path, and set --policy-client-host "" (and leave --policy-client-port unset)'
    )

    results = run_gr00t_sim_policy(
        env_name=args.env_name,
        n_episodes=args.n_episodes,
        max_episode_steps=args.max_episode_steps,
        model_path=args.model_path,
        policy_client_host=args.policy_client_host,
        policy_client_port=args.policy_client_port,
        n_envs=args.n_envs,
        n_action_steps=args.n_action_steps,
        video_dir=args.video_dir,
        seed=args.seed,
    )
    print("results: ", results)
    print("success rate: ", np.mean(results[1]))

"""N1.5 inference server with N1.7-compatible wire protocol.

Wraps N1.5's Gr00tPolicy in N1.5's RobotInferenceServer, but adds endpoints
matching N1.7's PolicyClient expectations:
  - get_action expects {observation: dict, options: dict|None} -> returns [action, info]
  - reset(options) -> returns {} (no-op for vanilla N1.5; no HAMLET state to reset)
  - get_modality_config -> returns dict[str, ModalityConfig]
  - ping (already registered by BaseInferenceServer)

This lets the rollout client (gr00t/eval/rollout_policy.py) work unchanged
against this N1.5 server.

Usage:
  python scripts/run_gr00t_server_n1d5.py \\
      --model-path <ckpt-dir> --embodiment-tag new_embodiment \\
      --data-config single_panda_gripper \\
      --host 127.0.0.1 --port 5555
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any

import tyro

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.eval.service import BaseInferenceServer
from gr00t.experiment.data_config import load_data_config
from gr00t.model.policy import Gr00tPolicy


@dataclass
class ArgsConfig:
    model_path: str
    data_config: str
    embodiment_tag: str = "new_embodiment"
    host: str = "127.0.0.1"
    port: int = 5555
    denoising_steps: int = 4


def main(args: ArgsConfig) -> None:
    data_config = load_data_config(args.data_config)
    modality_config = data_config.modality_config()
    modality_transform = data_config.transform()

    policy = Gr00tPolicy(
        model_path=args.model_path,
        modality_config=modality_config,
        modality_transform=modality_transform,
        embodiment_tag=args.embodiment_tag,
        denoising_steps=args.denoising_steps,
    )

    # RoboCasa env emits `video.res256_image_{side_0,side_1,wrist_0}`, but the
    # single_panda_gripper data config expects `video.{left,right,wrist}_view`.
    # N1.7 does this remap inside Gr00tSimPolicyWrapper.check_observation; the
    # N1.5 server has no such wrapper, so we remap here.
    _robocasa_video_remap = {
        "video.left_view": "video.res256_image_side_0",
        "video.right_view": "video.res256_image_side_1",
        "video.wrist_view": "video.res256_image_wrist_0",
    }

    def _get_action_wrapper(payload: dict) -> tuple[dict, dict]:
        """Unwrap N1.7 client's {observation, options} envelope, call N1.5
        policy.get_action(observation, options), then wrap as (action, info)."""
        observation = payload.get("observation", payload)
        for dst, src in _robocasa_video_remap.items():
            if dst not in observation and src in observation:
                observation[dst] = observation[src]
        # HAMLET memory reset / session routing:
        # - with `session_ids`: Gr00tPolicy keeps a per-session rolling cache and
        #   swaps it into the model around the forward call; per-sample
        #   reset_memory flags are forwarded as-is.
        # - without session_ids (legacy row-stable path): scalar True / all-True
        #   -> full cache clear via `__reset_memory__`; mixed flags -> per-row reset
        #   mask. The rolling cache is then keyed by batch ROW - serve a single
        #   client per server and keep the vector-env row order fixed.
        client_opts = payload.get("options") or {}
        fwd_opts: dict[str, Any] = {}
        sids = client_opts.get("session_ids")
        rm = client_opts.get("reset_memory")
        if isinstance(sids, (list, tuple)) and len(sids) > 0:
            fwd_opts["session_ids"] = [str(s) for s in sids]
            if isinstance(rm, (list, tuple)):
                fwd_opts["reset_memory"] = [bool(x) for x in rm]
            elif rm is True:
                fwd_opts["reset_memory"] = [True] * len(sids)
        elif isinstance(rm, (list, tuple)):
            if all(rm):
                fwd_opts["__reset_memory__"] = True
            elif any(rm):
                fwd_opts["reset_memory"] = [bool(x) for x in rm]
        elif rm is True:
            fwd_opts["__reset_memory__"] = True
        action = policy.get_action(observation, options=fwd_opts)
        info: dict[str, Any] = {}
        return (action, info)

    def _reset_handler(payload: dict | None) -> dict:
        """No-op reset for vanilla N1.5 (no memory state to clear)."""
        return {}

    def _get_modality_config_handler() -> dict:
        return policy.get_modality_config()

    server = BaseInferenceServer(host="*", port=args.port)
    server.register_endpoint("get_action", _get_action_wrapper)
    server.register_endpoint("reset", _reset_handler)
    server.register_endpoint(
        "get_modality_config", _get_modality_config_handler, requires_input=False
    )
    print(f"[i] N1.5 server (N1.7 wire protocol) listening on port {args.port}")
    server.run()


if __name__ == "__main__":
    main(tyro.cli(ArgsConfig))

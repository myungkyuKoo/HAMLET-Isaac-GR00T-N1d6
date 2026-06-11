# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HFValidationError, RepositoryNotFoundError

from gr00t.data.dataset import ModalityConfig
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.schema import DatasetMetadata
from gr00t.data.transform.base import ComposedModalityTransform
from gr00t.model.gr00t_n1 import GR00T_N1_5

COMPUTE_DTYPE = torch.bfloat16


class BasePolicy(ABC):
    @abstractmethod
    def get_action(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """
        Abstract method to get the action for a given state.

        Args:
            observations: The observations from the environment.

        Returns:
            The action to take in the environment in dictionary format.
        """
        raise NotImplementedError

    @abstractmethod
    def get_modality_config(self) -> Dict[str, ModalityConfig]:
        """
        Return the modality config of the policy.
        """
        raise NotImplementedError


class Gr00tPolicy(BasePolicy):
    """
    A wrapper for Gr00t model checkpoints that handles loading the model, applying transforms,
    making predictions, and unapplying transforms. This loads some custom configs, stats
    and metadata related to the model checkpoints used
    in the Gr00t model.
    """

    def __init__(
        self,
        model_path: str,
        embodiment_tag: Union[str, EmbodimentTag],
        modality_config: Dict[str, ModalityConfig],
        modality_transform: ComposedModalityTransform,
        denoising_steps: Optional[int] = None,
        device: Union[int, str] = "cuda" if torch.cuda.is_available() else "cpu",
        experiment_cfg_dir: Optional[str] = None,
    ):
        """
        Initialize the Gr00tPolicy.

        Args:
            model_path (str): Path to the model checkpoint directory or the huggingface hub id.
            modality_config (Dict[str, ModalityConfig]): The modality config for the model.
            modality_transform (ComposedModalityTransform): The modality transform for the model.
            embodiment_tag (Union[str, EmbodimentTag]): The embodiment tag for the model.
            denoising_steps: Number of denoising steps to use for the action head.
            device (Union[int, str]): Device to run the model on.
            experiment_cfg_dir (Optional[str]): Path to the experiment_cfg directory containing
                metadata.json. If None, defaults to ``<model_path>/experiment_cfg``.
                Useful when the model is loaded from HuggingFace Hub and metadata is stored locally.
        """
        try:
            # NOTE(YL) this returns the local path to the model which is normally
            # saved in ~/.cache/huggingface/hub/
            model_path = snapshot_download(model_path, repo_type="model")
            # HFValidationError, RepositoryNotFoundError
        except (HFValidationError, RepositoryNotFoundError):
            print(
                f"Model not found or avail in the huggingface hub. Loading from local path: {model_path}"
            )

        self._modality_config = modality_config
        self._modality_transform = modality_transform
        self._modality_transform.eval()  # set this to eval mode
        self.model_path = Path(model_path)
        self.device = device

        # Convert string embodiment tag to EmbodimentTag enum if needed
        if isinstance(embodiment_tag, str):
            self.embodiment_tag = EmbodimentTag(embodiment_tag)
        else:
            self.embodiment_tag = embodiment_tag

        # Load model
        self._load_model(model_path)
        # Load transforms — resolve experiment_cfg directory.
        # Priority: explicit experiment_cfg_dir (if it exists) > <model_path>/experiment_cfg.
        exp_cfg = None
        if experiment_cfg_dir is not None:
            candidate = Path(experiment_cfg_dir)
            if (candidate / "metadata.json").exists():
                exp_cfg = candidate
            else:
                print(
                    f"Warning: metadata.json not found at {candidate}, "
                    "falling back to <model_path>/experiment_cfg"
                )
        if exp_cfg is None:
            exp_cfg = self.model_path / "experiment_cfg"
        self._load_metadata(exp_cfg)
        # Load horizons
        self._load_horizons()

        if denoising_steps is not None:
            if hasattr(self.model, "action_head") and hasattr(
                self.model.action_head, "num_inference_timesteps"
            ):
                self.model.action_head.num_inference_timesteps = denoising_steps
                print(f"Set action denoising steps to {denoising_steps}")

    def apply_transforms(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply transforms to the observation.

        Args:
            obs (Dict[str, Any]): The observation to transform.

        Returns:
            Dict[str, Any]: The transformed observation.
        """
        # Ensure correct dimensions before applying transforms
        return self._modality_transform(obs)

    def unapply_transforms(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Unapply transforms to the action.

        Args:
            action (Dict[str, Any]): The action to unapply transforms to.

        Returns:
            Dict[str, Any]: The untransformed action.
        """
        return self._modality_transform.unapply(action)

    def get_action(
        self,
        observations: Dict[str, Any],
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Make a prediction with the model.
        Args:
            obs (Dict[str, Any]): The observation to make a prediction for.

        e.g. obs = {
            "video.<>": np.ndarray,  # (T, H, W, C)
            "state.<>": np.ndarray, # (T, D)
            "annotation.<>": np.ndarray, # (T, )
        }

        or with batched input:
        e.g. obs = {
            "video.<>": np.ndarray,, # (B, T, H, W, C)
            "state.<>": np.ndarray, # (B, T, D)
            "annotation.<>": np.ndarray, # (B, T, )
        }

        Options (HAMLET):
            - "__reset_memory__" (bool): clear the rolling memory cache before
              this call. Useful at episode boundaries.

        Returns:
            Dict[str, Any]: The predicted action.
        """
        # HAMLET memory cache reset at episode boundaries.
        opts = dict(options) if options else {}
        if opts.pop("__reset_memory__", False) and hasattr(self.model, "reset_memory"):
            self.model.reset_memory()
            for _attr in ("_session_mq_cache", "_session_vis_cache", "_session_lru"):
                _store = getattr(self, _attr, None)
                if _store is not None:
                    _store.clear()
        # Per-sample reset mask (multi-env): convert to a bool tensor on the
        # model device; the model resets only the flagged rows of the rolling
        # cache (see GR00T_N1_5.get_action / _apply_memory).
        rm_mask = opts.get("reset_memory")
        if rm_mask is not None and not torch.is_tensor(rm_mask):
            try:
                _dev = next(self.model.parameters()).device
            except StopIteration:
                _dev = "cpu"
            opts["reset_memory"] = torch.tensor(rm_mask, dtype=torch.bool, device=_dev)

        # --- Session-isolated HAMLET cache (parity with the N1.6 policy) ----
        # When the client provides per-sample `session_ids`, the rolling memory
        # cache is stored per session on the policy and swapped into the model's
        # cache slots around the forward call. Without session_ids the legacy
        # row-stable global cache path is used (single client, fixed vector-env
        # row order).
        session_ids = opts.pop("session_ids", None)
        _session_active = (
            session_ids is not None
            and getattr(self.model, "memory_transformer", None) is not None
        )
        if _session_active:
            if getattr(self, "_session_mq_cache", None) is None:
                self._session_mq_cache: Dict[str, torch.Tensor] = {}
                self._session_vis_cache: Dict[str, torch.Tensor] = {}
                self._session_lru: list = []
                self._session_cache_cap: int = 64
            B_s = len(session_ids)
            raw_reset = opts.get("reset_memory")
            if torch.is_tensor(raw_reset):
                raw_reset = [bool(x) for x in raw_reset.tolist()]
            if raw_reset is None:
                raw_reset = [False] * B_s
            is_vision = getattr(self.model, "memory_type", "moment_token") == "vision_feature"
            try:
                _dev = next(self.model.parameters()).device
            except StopIteration:
                _dev = "cpu"
            cached_mq = [
                None if raw_reset[i] else self._session_mq_cache.get(session_ids[i])
                for i in range(B_s)
            ]
            cached_vis = [
                None if raw_reset[i] else self._session_vis_cache.get(session_ids[i])
                for i in range(B_s)
            ]
            ref = next((c for c in cached_mq if c is not None), None)
            if ref is not None:
                ph = torch.zeros_like(ref)
                self.model._cached_mq = torch.stack(
                    [c if c is not None else ph for c in cached_mq], dim=0
                ).to(_dev)
            else:
                self.model._cached_mq = None
            ref_v = next((c for c in cached_vis if c is not None), None)
            if ref_v is not None:
                ph_v = torch.zeros_like(ref_v)
                self.model._cached_vis = torch.stack(
                    [c if c is not None else ph_v for c in cached_vis], dim=0
                ).to(_dev)
            else:
                self.model._cached_vis = None
            relevant = cached_vis if is_vision else cached_mq
            opts["reset_memory"] = torch.tensor(
                [bool(raw_reset[i] or relevant[i] is None) for i in range(B_s)],
                dtype=torch.bool,
                device=_dev,
            )

        # Create a copy to avoid mutating input
        obs_copy = observations.copy()

        is_batch = self._check_state_is_batched(obs_copy)
        if not is_batch:
            obs_copy = unsqueeze_dict_values(obs_copy)

        # Convert to numpy arrays
        for k, v in obs_copy.items():
            if not isinstance(v, np.ndarray):
                obs_copy[k] = np.array(v)

        normalized_input = self.apply_transforms(obs_copy)
        normalized_action = self._get_action_from_normalized_input(normalized_input, opts)
        if _session_active:
            new_mq = self.model._cached_mq
            if new_mq is not None:
                for i, sid in enumerate(session_ids):
                    self._session_mq_cache[sid] = new_mq[i].detach().clone()
            new_vis = self.model._cached_vis
            if new_vis is not None:
                for i, sid in enumerate(session_ids):
                    self._session_vis_cache[sid] = new_vis[i].detach().clone()
            # Clear model slots so the next call rebuilds them from per-session
            # storage, then evict sessions beyond the LRU cap (oldest first).
            self.model._cached_mq = None
            self.model._cached_vis = None
            for sid in session_ids:
                if sid in self._session_lru:
                    self._session_lru.remove(sid)
                self._session_lru.append(sid)
            while len(self._session_lru) > self._session_cache_cap:
                stale = self._session_lru.pop(0)
                self._session_mq_cache.pop(stale, None)
                self._session_vis_cache.pop(stale, None)
        unnormalized_action = self._get_unnormalized_action(normalized_action)

        if not is_batch:
            unnormalized_action = squeeze_dict_values(unnormalized_action)
        return unnormalized_action

    def reset_memory(self):
        """Clear the HAMLET MQ rolling cache on the wrapped model."""
        if hasattr(self.model, "reset_memory"):
            self.model.reset_memory()

    def _get_action_from_normalized_input(
        self,
        normalized_input: Dict[str, Any],
        options: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        # Set up autocast context if needed
        # Decide ONCE (per policy instance) whether the wrapped model's get_action
        # accepts an `options` kwarg (HAMLET models do; plain GR00T_N1_5 does not).
        # A broad try/except TypeError here would mask genuine TypeErrors raised
        # inside the model and re-run inference after partial cache/RNG mutation.
        if getattr(self, "_model_accepts_options", None) is None:
            import inspect

            try:
                params = inspect.signature(self.model.get_action).parameters
                self._model_accepts_options = "options" in params or any(
                    p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
                )
            except (TypeError, ValueError):
                self._model_accepts_options = False

        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=COMPUTE_DTYPE):
            if self._model_accepts_options:
                model_pred = self.model.get_action(normalized_input, options=options)
            else:
                model_pred = self.model.get_action(normalized_input)

        normalized_action = model_pred["action_pred"].float()
        return normalized_action

    def _get_unnormalized_action(self, normalized_action: torch.Tensor) -> Dict[str, Any]:
        return self.unapply_transforms({"action": normalized_action.cpu()})

    def get_modality_config(self) -> Dict[str, ModalityConfig]:
        """
        Get the modality config for the model, overrides the base class method
        """
        return self._modality_config

    @property
    def modality_config(self) -> Dict[str, ModalityConfig]:
        return self._modality_config

    @property
    def modality_transform(self) -> ComposedModalityTransform:
        return self._modality_transform

    @property
    def video_delta_indices(self) -> np.ndarray:
        """Get the video delta indices."""
        return self._video_delta_indices

    @property
    def state_delta_indices(self) -> np.ndarray | None:
        """Get the state delta indices."""
        return self._state_delta_indices

    @property
    def denoising_steps(self) -> int:
        """Get the number of denoising steps."""
        return self.model.action_head.num_inference_timesteps

    @denoising_steps.setter
    def denoising_steps(self, value: int):
        """Set the number of denoising steps."""
        self.model.action_head.num_inference_timesteps = value

    def _check_state_is_batched(self, obs: Dict[str, Any]) -> bool:
        for k, v in obs.items():
            if "state" in k and len(v.shape) < 3:  # (B, Time, Dim)
                return False
        return True

    def _load_model(self, model_path):
        model = GR00T_N1_5.from_pretrained(model_path, torch_dtype=COMPUTE_DTYPE)
        model.eval()  # Set model to eval mode

        # Update action_horizon to match modality config
        # Get the expected action horizon from the modality config
        expected_action_horizon = len(self._modality_config["action"].delta_indices)

        if expected_action_horizon != model.action_head.config.action_horizon:
            print(
                f"Policy: Recreating action head with action_horizon {expected_action_horizon} (was {model.action_head.config.action_horizon})"
            )

            # Update the action head config
            new_action_head_config = model.action_head.config
            new_action_head_config.action_horizon = expected_action_horizon

            # Import the FlowmatchingActionHead class
            from gr00t.model.action_head.flow_matching_action_head import (
                FlowmatchingActionHead,
            )

            # Create new action head with updated config
            new_action_head = FlowmatchingActionHead(new_action_head_config)

            # Copy the weights from the old action head to the new one
            new_action_head.load_state_dict(model.action_head.state_dict(), strict=False)

            # Replace the action head
            model.action_head = new_action_head

            # Update model config AND the action_head_cfg dictionary that gets saved
            model.config.action_horizon = expected_action_horizon
            model.action_horizon = expected_action_horizon
            model.config.action_head_cfg["action_horizon"] = expected_action_horizon

        model.to(device=self.device)  # type: ignore

        self.model = model

    def _load_metadata(self, exp_cfg_dir: Path):
        """Load the transforms for the model."""
        # Load metadata for normalization stats
        metadata_path = exp_cfg_dir / "metadata.json"
        with open(metadata_path, "r") as f:
            metadatas = json.load(f)

        # Get metadata for the specific embodiment
        metadata_dict = metadatas.get(self.embodiment_tag.value)
        if metadata_dict is None:
            raise ValueError(
                f"No metadata found for embodiment tag: {self.embodiment_tag.value}",
                f"make sure the metadata.json file is present at {metadata_path}",
            )

        metadata = DatasetMetadata.model_validate(metadata_dict)

        self._modality_transform.set_metadata(metadata)
        self.metadata = metadata

    def _load_horizons(self):
        """Load the horizons needed for the model."""
        # Get modality configs
        # Video horizons
        self._video_delta_indices = np.array(self._modality_config["video"].delta_indices)
        self._assert_delta_indices(self._video_delta_indices)
        self._video_horizon = len(self._video_delta_indices)
        # State horizons (if used)
        if "state" in self._modality_config:
            self._state_delta_indices = np.array(self._modality_config["state"].delta_indices)
            self._assert_delta_indices(self._state_delta_indices)
            self._state_horizon = len(self._state_delta_indices)
        else:
            self._state_horizon = None
            self._state_delta_indices = None

    def _assert_delta_indices(self, delta_indices: np.ndarray):
        """Assert that the delta indices are valid."""
        # All delta indices should be non-positive because there's no way to get the future observations
        assert np.all(delta_indices <= 0), f"{delta_indices=}"
        # The last delta index should be 0 because it doesn't make sense to not use the latest observation
        assert delta_indices[-1] == 0, f"{delta_indices=}"
        if len(delta_indices) > 1:
            # The step is consistent
            assert np.all(
                np.diff(delta_indices) == delta_indices[1] - delta_indices[0]
            ), f"{delta_indices=}"
            # And the step is positive
            assert (delta_indices[1] - delta_indices[0]) > 0, f"{delta_indices=}"


#######################################################################################################


# Helper functions
def unsqueeze_dict_values(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Unsqueeze the values of a dictionary.
    This converts the data to be batched of size 1.
    """
    unsqueezed_data = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            unsqueezed_data[k] = np.expand_dims(v, axis=0)
        elif isinstance(v, list):
            unsqueezed_data[k] = np.expand_dims(np.array(v), axis=0)  # Fixed
        elif isinstance(v, torch.Tensor):
            unsqueezed_data[k] = v.unsqueeze(0)
        else:
            unsqueezed_data[k] = v
    return unsqueezed_data


def squeeze_dict_values(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Squeeze the values of a dictionary. This removes the batch dimension.
    """
    squeezed_data = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            squeezed_data[k] = np.squeeze(v, axis=0)  # Fixed: only remove batch dim
        elif isinstance(v, torch.Tensor):
            squeezed_data[k] = v.squeeze(0)  # Fixed: only remove batch dim
        else:
            squeezed_data[k] = v
    return squeezed_data

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

import logging
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

import numpy as np
import torch
from torch import nn
import tree
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HFValidationError, RepositoryNotFoundError
from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel
from transformers.feature_extraction_utils import BatchFeature

from .action_head.flow_matching_action_head import (
    FlowmatchingActionHead,
    FlowmatchingActionHeadConfig,
)
from .backbone import EagleBackbone
from .memory import MemoryTransformer
from .tcl_head import Gr00tTCLHead

logger = logging.getLogger(__name__)

BACKBONE_FEATURE_KEY = "backbone_features"
BACKBONE_MASK_KEY = "backbone_attention_mask"
N_MOMENT_TOKENS_KEY = "n_moment_tokens"
ACTION_KEY = "action_pred"
LOSS_KEY = "loss"
ERROR_MSG = "Error: unexpected input/output"
N_COLOR_CHANNELS = 3


# config
@dataclass
class GR00T_N1_5_Config(PretrainedConfig):
    model_type = "gr00t_n1_5"
    backbone_cfg: dict = field(init=False, metadata={"help": "Backbone configuration."})

    action_head_cfg: dict = field(init=False, metadata={"help": "Action head configuration."})

    action_horizon: int = field(init=False, metadata={"help": "Action horizon."})

    action_dim: int = field(init=False, metadata={"help": "Action dimension."})
    compute_dtype: str = field(default="float32", metadata={"help": "Compute dtype."})

    # --- HAMLET ---
    hamlet_mode: str = field(
        default="finetune",
        metadata={"help": "HAMLET training mode: one of {off, tcl, finetune}."},
    )
    n_moment_tokens: int = field(
        default=4, metadata={"help": "n_q: learnable moment tokens appended to VLM input."}
    )
    memory_window: int = field(
        default=4, metadata={"help": "T: history window length for the memory transformer."}
    )
    memory_num_layers: int = field(
        default=2, metadata={"help": "Depth of the HAMLET memory transformer."}
    )
    freeze_moment_tokens: bool = field(
        default=False, metadata={"help": "Freeze moment-token parameter (enable when loading TCL-initialized tokens)."}
    )
    mem_cond_type: str = field(
        default="cross_attn",
        metadata={"help": "HAMLET memory->action conditioning: {cross_attn, adaln}."},
    )
    memory_type: str = field(
        default="moment_token",
        metadata={"help": "What flows through the memory module: {moment_token, vision_feature}. "
                  "vision_feature: primary-view image tokens (post-LLM) pooled to 64/step."},
    )
    tcl_tau: float = field(
        default=0.07, metadata={"help": "InfoNCE temperature for the TCL stage."}
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


class _MemAdaLNPool(nn.Module):
    """Mean-pool HAMLET memory tokens (B, n_q, d_in) -> (B, d_out) for AdaLN-zero
    conditioning. The output projection is ZERO-INIT so the AdaLN-memory path starts as
    an exact no-op; it learns to inject memory over training."""

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.proj = nn.Linear(d_in, d_out)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, mem: torch.Tensor) -> torch.Tensor:  # (B, n_q, d_in)
        pooled = mem.mean(dim=1)
        return self.proj(pooled)  # (B, d_out); zero at init -> no-op


# real model
class GR00T_N1_5(PreTrainedModel):
    supports_gradient_checkpointing = True
    config_class = GR00T_N1_5_Config
    """
    we expect the backbone output to have a key 'backbone_features' with shape (batch_size, n, hidden_size)
    here n is variable and can be e.g. time, 1 or user specified
    we expect the action head output to have a key 'action_pred' with shape (batch_size, time, action_dim) during inference time
    we expect these to have type BatchFeature, and they can of course have many other user specified keys too

    HAMLET extensions (gated by config.hamlet_mode):
      - "off": vanilla GR00T N1.5 path.
      - "tcl": time-contrastive pretraining of moment_tokens + Gr00tTCLHead on
        (anchor, positive, hard-negative) triplets.
      - "finetune": K-step batching feeds K consecutive observations per sample
        through the backbone; the n_q moment-token tail of each step is
        aggregated by a block-causal memory module whose output replaces the
        current step's tail before the action head runs.
    """

    def __init__(
        self,
        config: GR00T_N1_5_Config,
        local_model_path: str,
    ):
        assert isinstance(config.backbone_cfg, dict)
        assert isinstance(config.action_head_cfg, dict)

        super().__init__(config)
        self.local_model_path = local_model_path

        # Inject HAMLET kwargs into a copy of backbone_cfg so the backbone gets
        # moment_tokens whenever hamlet_mode != "off".
        self.hamlet_mode = getattr(config, "hamlet_mode", "off")
        self.n_moment_tokens = int(getattr(config, "n_moment_tokens", 4))
        self.memory_window = int(getattr(config, "memory_window", 4))
        self.memory_num_layers = int(getattr(config, "memory_num_layers", 2))
        self.freeze_moment_tokens = bool(getattr(config, "freeze_moment_tokens", False))
        self.tcl_tau = float(getattr(config, "tcl_tau", 0.07))
        # memory_type: "moment_token" feeds the learnable moment tokens through the
        # memory module; "vision_feature" feeds the primary view's image tokens
        # (post-LLM, pooled to 64/step) and adds no moment tokens.
        self.memory_type = getattr(config, "memory_type", "moment_token")
        self._mem_tokens_per_step = (
            64 if self.memory_type == "vision_feature" else self.n_moment_tokens
        )

        backbone_cfg = dict(config.backbone_cfg)
        if self.hamlet_mode != "off":
            backbone_cfg["memory_type"] = self.memory_type
            if self.memory_type == "vision_feature":
                backbone_cfg["n_moment_tokens"] = 0
                backbone_cfg["freeze_moment_tokens"] = False
            else:
                backbone_cfg["n_moment_tokens"] = self.n_moment_tokens
                backbone_cfg["freeze_moment_tokens"] = self.freeze_moment_tokens
        self.backbone = EagleBackbone(**backbone_cfg)

        if self.hamlet_mode == "tcl":
            # Stage-1: freeze everything except moment_tokens and the TCL head.
            # TCL trains the moment tokens by definition — clear the backbone's
            # persisted freeze flag so set_trainable_parameters (re-invoked by
            # from_pretrained) does not re-freeze them.
            self.backbone.freeze_moment_tokens = False
            for p in self.backbone.parameters():
                p.requires_grad = False
            if self.backbone.moment_tokens is not None:
                self.backbone.moment_tokens.requires_grad_(True)
            backbone_embed_dim = (
                self.backbone.eagle_linear.out_features
                if isinstance(self.backbone.eagle_linear, torch.nn.Linear)
                else 2048
            )
            self.action_head = Gr00tTCLHead(
                backbone_embedding_dim=backbone_embed_dim,
                tcl_tau=self.tcl_tau,
            )
            self.memory_transformer = None
        else:
            action_head_cfg = FlowmatchingActionHeadConfig(**config.action_head_cfg)
            self.action_head = FlowmatchingActionHead(action_head_cfg)
            if self.hamlet_mode == "finetune" and self.memory_num_layers > 0:
                self.memory_transformer = MemoryTransformer(
                    dim=int(action_head_cfg.backbone_embedding_dim),
                    n_q=self._mem_tokens_per_step,
                    T=self.memory_window,
                    num_layers=self.memory_num_layers,
                )
            else:
                self.memory_transformer = None

        # Inference-time rolling cache, shape (B, T*n_q, d), oldest first to current
        # last. Seeded by replicating the current step T times, then updated FIFO.
        self._memory_cache: torch.Tensor | None = None
        self._vision_cache: torch.Tensor | None = None

        # memory-to-action conditioning. "cross_attn" replaces the moment-token tail
        # of the action-head KV; "adaln" mean-pools the memory output through a
        # zero-init projection added to the DiT timestep embedding.
        self.mem_cond_type = getattr(config, "mem_cond_type", "cross_attn")
        if self.memory_transformer is not None and self.mem_cond_type == "adaln":
            self.mem_adaln_pool = _MemAdaLNPool(
                d_in=int(config.action_head_cfg["backbone_embedding_dim"]),
                d_out=int(self.action_head.model.inner_dim),
            )
        else:
            self.mem_adaln_pool = None

        self.action_horizon = config.action_horizon
        self.action_dim = config.action_dim
        self.compute_dtype = config.compute_dtype

    def validate_inputs(self, inputs):
        # NOTE -- this should be handled internally by the model
        # however, doing that will likely be breaking changes -- so we'll need to do it after the deadline
        if self.hamlet_mode == "tcl":
            # TCL forward consumes triplet streams; skip the action/video shape
            # checks because the dataloader emits (anchor, aug, neg) triplets.
            return

        detected_error = False
        error_msg = ERROR_MSG
        if "action" in inputs:
            action = inputs["action"]
            type_ok = isinstance(action, torch.Tensor)
            shape_ok = (
                len(action.shape) == 3
                and action.shape[1] == self.action_horizon
                and action.shape[2] == self.action_dim
            )
            if not type_ok:
                error_msg += f"\n{action.dtype=}"
                detected_error = True
            if not shape_ok:
                error_msg += f"\n{action.shape=}"
                detected_error = True

        if "video" in inputs:
            video = inputs["video"]
            type_ok = isinstance(video, np.ndarray)
            dtype_ok = video.dtype == np.uint8
            shape_ok = len(video.shape) == 6 and video.shape[3] == N_COLOR_CHANNELS
            if not type_ok:
                error_msg += f"\n{type(video)=}"
                detected_error = True
            if not dtype_ok:
                error_msg += f"\n{video.dtype=}"
                detected_error = True
            if not shape_ok:
                error_msg += f"\n{video.shape=}"
                detected_error = True

        if detected_error:
            raise ValueError(error_msg)

    def validate_data(self, action_head_outputs, backbone_outputs, is_training):
        if self.hamlet_mode == "tcl":
            return  # TCL head outputs only a loss key.

        fail_backbone = (
            not isinstance(backbone_outputs, BatchFeature)
            or BACKBONE_FEATURE_KEY not in backbone_outputs
        )

        if fail_backbone:
            error_msg = ERROR_MSG
            error_msg += f"\n{isinstance(backbone_outputs, BatchFeature)=}"
            error_msg += f"\n{BACKBONE_FEATURE_KEY in backbone_outputs=}"
            error_msg += f"\n{backbone_outputs[BACKBONE_FEATURE_KEY].shape=}"
            raise ValueError(error_msg)

        fail_action_head = (not isinstance(action_head_outputs, BatchFeature)) or not (
            (
                LOSS_KEY in action_head_outputs and is_training
            )  # there might not be an action prediction during training
            or (
                ACTION_KEY in action_head_outputs
                and action_head_outputs[ACTION_KEY].shape[1] == self.action_horizon
                and action_head_outputs[ACTION_KEY].shape[2] == self.action_dim
            )
        )

        if fail_action_head:
            error_msg = ERROR_MSG
            error_msg += f"\n{isinstance(action_head_outputs, BatchFeature)=}"
            error_msg += f"\n{LOSS_KEY in action_head_outputs=}"
            error_msg += f"\n{action_head_outputs[ACTION_KEY].shape=}"
            error_msg += f"\n{self.action_horizon=}"
            error_msg += f"\n{self.action_dim=}"
            raise ValueError(error_msg)

    # ------------------------- HAMLET memory aggregation -------------------------

    def _apply_memory(
        self,
        backbone_output: BatchFeature,
        B_target: int,
        reset_memory: torch.Tensor | None = None,
    ) -> BatchFeature:
        """Fold the K backbone rows (training) or rolling cache (inference) through
        the memory module and rewrite the current step's moment-token tail.
        """
        if self.memory_transformer is None:
            return backbone_output

        # ---- vision_feature path: aggregate the primary view's pooled (64) tokens ----
        if self.memory_type == "vision_feature" and "primary_view_feature" in backbone_output:
            v_nq = self._mem_tokens_per_step  # 64
            primary = backbone_output["primary_view_feature"]  # (B*K, 64, d)
            feats = backbone_output[BACKBONE_FEATURE_KEY]
            mask = backbone_output[BACKBONE_MASK_KEY]
            BK, _, d = primary.shape
            B = B_target
            K = BK // B
            K_target = self.memory_transformer.T
            Tlen = feats.shape[1]
            if K not in (1, K_target):
                raise RuntimeError(
                    f"HAMLET memory: got K={K} backbone rows per action sample "
                    f"(expected 1 for rolling inference or memory_window={K_target} "
                    f"for K-step training). The video delta_indices / memory_window "
                    f"data config is inconsistent - refusing to silently skip "
                    f"memory augmentation."
                )
            if K == K_target:
                mem_seq = primary.view(B, K, v_nq, d).view(B, K * v_nq, d)
                mem_aug = self.memory_transformer(mem_seq)[:, -v_nq:, :]
                current = feats.view(B, K, Tlen, d)[:, -1, :, :]  # unchanged conditioning
                new_mask = mask.view(B, K, -1)[:, -1, :]
                if self.mem_cond_type == "adaln":
                    backbone_output["mem_temb_add"] = self.mem_adaln_pool(mem_aug)
                else:
                    current = torch.cat([current, mem_aug], dim=1)
                    new_mask = torch.cat([new_mask, new_mask.new_ones(B, v_nq)], dim=1)
                backbone_output[BACKBONE_FEATURE_KEY] = current
                backbone_output[BACKBONE_MASK_KEY] = new_mask
            elif K == 1:
                vis_current = primary  # (B, 64, d)
                if self._vision_cache is None or self._vision_cache.shape[0] != B:
                    self._vision_cache = vis_current.repeat(1, K_target, 1)
                elif reset_memory is not None and torch.is_tensor(reset_memory) and reset_memory.any():
                    defaults = vis_current.repeat(1, K_target, 1)
                    shifted = torch.cat([self._vision_cache[:, v_nq:, :], vis_current], dim=1)
                    reset_b = reset_memory.view(B, 1, 1).expand(B, K_target * v_nq, d)
                    self._vision_cache = torch.where(reset_b, defaults, shifted)
                else:
                    self._vision_cache = torch.cat([self._vision_cache[:, v_nq:, :], vis_current], dim=1)
                mem_aug = self.memory_transformer(self._vision_cache)[:, -v_nq:, :]
                if self.mem_cond_type == "adaln":
                    backbone_output["mem_temb_add"] = self.mem_adaln_pool(mem_aug)
                else:
                    backbone_output[BACKBONE_FEATURE_KEY] = torch.cat([feats, mem_aug], dim=1)
                    backbone_output[BACKBONE_MASK_KEY] = torch.cat(
                        [mask, mask.new_ones(B, v_nq)], dim=1
                    )
            return backbone_output

        if N_MOMENT_TOKENS_KEY not in backbone_output:
            return backbone_output

        feats = backbone_output[BACKBONE_FEATURE_KEY]
        mask = backbone_output[BACKBONE_MASK_KEY]
        n_q = int(backbone_output[N_MOMENT_TOKENS_KEY])
        BK, T, d = feats.shape
        K_target = self.memory_transformer.T
        K = BK // B_target
        assert BK == B_target * K, f"expected B*K rows, got {BK} for B={B_target}"

        if K not in (1, K_target):
            raise RuntimeError(
                f"HAMLET memory: got K={K} backbone rows per action sample "
                f"(expected 1 for rolling inference or memory_window={K_target} "
                f"for K-step training). The video delta_indices / memory_window "
                f"data config is inconsistent - refusing to silently skip "
                f"memory augmentation."
            )
        if K == K_target:
            # K-step training path
            moment_all = feats[:, -n_q:, :].contiguous().view(B_target, K, n_q, d)
            moment_seq = moment_all.view(B_target, K * n_q, d)  # oldest -> current
            memory_out = self.memory_transformer(moment_seq)
            memory_slice = memory_out[:, -n_q:, :]
            current = feats.view(B_target, K, T, d)[:, -1, :, :]
            new_mask = mask.view(B_target, K, -1)[:, -1, :]
            if self.mem_cond_type == "adaln":
                backbone_output["mem_temb_add"] = self.mem_adaln_pool(memory_slice)
                current = current[:, :-n_q, :]
                new_mask = new_mask[:, :-n_q]
            else:
                current = torch.cat([current[:, :-n_q, :], memory_slice], dim=1)
        elif K == 1:
            # Inference path with rolling FIFO cache.
            moment_current = feats[:, -n_q:, :]
            if self._memory_cache is None or self._memory_cache.shape[0] != B_target:
                self._memory_cache = moment_current.repeat(1, K_target, 1)
            else:
                if reset_memory is not None and torch.is_tensor(reset_memory) and reset_memory.any():
                    defaults = moment_current.repeat(1, K_target, 1)
                    shifted = torch.cat([self._memory_cache[:, n_q:, :], moment_current], dim=1)
                    reset_b = reset_memory.view(B_target, 1, 1).expand(B_target, K_target * n_q, d)
                    self._memory_cache = torch.where(reset_b, defaults, shifted)
                else:
                    self._memory_cache = torch.cat([self._memory_cache[:, n_q:, :], moment_current], dim=1)
            memory_out = self.memory_transformer(self._memory_cache)
            memory_slice = memory_out[:, -n_q:, :]
            new_mask = mask
            if self.mem_cond_type == "adaln":
                backbone_output["mem_temb_add"] = self.mem_adaln_pool(memory_slice)
                current = feats[:, :-n_q, :]
                new_mask = mask[:, :-n_q]
            else:
                current = torch.cat([feats[:, :-n_q, :], memory_slice], dim=1)
        else:
            return backbone_output

        backbone_output[BACKBONE_FEATURE_KEY] = current
        backbone_output[BACKBONE_MASK_KEY] = new_mask
        return backbone_output

    def reset_memory(self):
        """Clear the rolling memory cache. Call at episode boundaries."""
        self._memory_cache = None
        self._vision_cache = None

    # --------------------------- core training/inference ---------------------------

    def forward(
        self,
        inputs: dict,
    ) -> BatchFeature:
        if self.hamlet_mode == "tcl":
            return self._forward_tcl(inputs)

        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_outputs = self.backbone(backbone_inputs)

        if self.hamlet_mode == "finetune" and self.memory_transformer is not None:
            # K-step batching: action_inputs is at B (per-sample), backbone at B*K.
            # The dataset places the target action at the current step, so loss
            # falls only on the last slice without any mask edit here.
            B_target = action_inputs["action"].shape[0]
            backbone_outputs = self._apply_memory(backbone_outputs, B_target=B_target)

        action_head_outputs = self.action_head(backbone_outputs, action_inputs)
        self.validate_data(action_head_outputs, backbone_outputs, is_training=True)
        return action_head_outputs

    def _forward_tcl(self, inputs: dict) -> BatchFeature:
        """TCL Stage-1: run backbone three times on (anchor, aug, neg) streams
        and pass outputs to the TCL head. The dataloader/collator emits
        ``eagle_*`` + ``eagle_*_aug`` + ``eagle_*_neg`` keys.
        """
        # The TCL head doesn't need action_inputs.
        inputs = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in inputs.items()}

        def _slice(suffix: str) -> BatchFeature:
            data = {}
            for k in ("input_ids", "attention_mask", "pixel_values", "image_sizes"):
                src = f"eagle_{k}{suffix}"
                if src in inputs:
                    data[f"eagle_{k}"] = inputs[src]
            return BatchFeature(data=data)

        anc_out = self.backbone(_slice(""))
        aug_out = self.backbone(_slice("_aug"))
        neg_out = self.backbone(_slice("_neg"))
        return self.action_head(anc_out, aug_out, neg_out)

    def get_action(
        self,
        inputs: dict,
        options: Optional[dict[str, Any]] = None,
    ) -> BatchFeature:
        if self.hamlet_mode == "tcl":
            raise RuntimeError("TCL checkpoints do not support get_action; load a Stage-2 ckpt.")

        backbone_inputs, action_inputs = self.prepare_input(inputs)
        # Because the behavior of backbones remains the same for training and inference, we can use `forward` for backbones.
        backbone_outputs = self.backbone(backbone_inputs)

        if self.hamlet_mode == "finetune" and self.memory_transformer is not None:
            reset_memory = options.get("reset_memory") if options else None
            B_target = action_inputs["state"].shape[0]
            backbone_outputs = self._apply_memory(
                backbone_outputs, B_target=B_target, reset_memory=reset_memory
            )

        action_head_outputs = self.action_head.get_action(backbone_outputs, action_inputs)
        self.validate_data(action_head_outputs, backbone_outputs, is_training=False)
        return action_head_outputs

    def prepare_input(self, inputs) -> Tuple[BatchFeature, BatchFeature]:
        self.validate_inputs(inputs)
        backbone_inputs = self.backbone.prepare_input(inputs)
        action_inputs = self.action_head.prepare_input(inputs)

        def to_device_with_maybe_dtype(x):
            # Only cast to self.compute_dtype if the tensor is floating
            if torch.is_floating_point(x):
                return x.to(self.device, dtype=self.action_head.dtype)
            else:
                # Keep original dtype
                return x.to(self.device)

        backbone_inputs = tree.map_structure(to_device_with_maybe_dtype, backbone_inputs)
        action_inputs = tree.map_structure(to_device_with_maybe_dtype, action_inputs)
        return backbone_inputs, action_inputs

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        tune_visual = kwargs.pop("tune_visual", True)
        tune_llm = kwargs.pop("tune_llm", False)
        tune_projector = kwargs.pop("tune_projector", True)
        tune_diffusion_model = kwargs.pop("tune_diffusion_model", True)

        # HAMLET overrides plumbed through from gr00t_finetune.py.
        hamlet_mode = kwargs.pop("hamlet_mode", None)
        n_moment_tokens = kwargs.pop("n_moment_tokens", None)
        memory_window = kwargs.pop("memory_window", None)
        memory_num_layers = kwargs.pop("memory_num_layers", None)
        freeze_moment_tokens = kwargs.pop("freeze_moment_tokens", None)
        mem_cond_type = kwargs.pop("mem_cond_type", None)
        memory_type = kwargs.pop("memory_type", None)
        tcl_tau = kwargs.pop("tcl_tau", None)

        print(f"Loading pretrained dual brain from {pretrained_model_name_or_path}")
        print(f"Tune backbone vision tower: {tune_visual}")
        print(f"Tune backbone LLM: {tune_llm}")
        print(f"Tune action head projector: {tune_projector}")
        print(f"Tune action head DiT: {tune_diffusion_model}")

        # get the current model path being downloaded
        try:
            # NOTE(YL) This downloads the model to the local cache and returns the local path to the model
            # saved in ~/.cache/huggingface/hub/
            local_model_path = snapshot_download(pretrained_model_name_or_path, repo_type="model")
            # HFValidationError, RepositoryNotFoundError
        except (HFValidationError, RepositoryNotFoundError):
            print(
                f"Model not found or avail in the huggingface hub. Loading from local path: {pretrained_model_name_or_path}"
            )
            local_model_path = pretrained_model_name_or_path

        # Surface HAMLET overrides through the config before from_pretrained builds
        # the backbone / action head, since __init__ reads them via getattr(config).
        if any(
            x is not None
            for x in (
                hamlet_mode,
                n_moment_tokens,
                memory_window,
                memory_num_layers,
                freeze_moment_tokens,
                mem_cond_type,
                memory_type,
                tcl_tau,
            )
        ):
            cfg = AutoConfig.from_pretrained(local_model_path, trust_remote_code=True)
            if hamlet_mode is not None:
                cfg.hamlet_mode = hamlet_mode
            if n_moment_tokens is not None:
                cfg.n_moment_tokens = int(n_moment_tokens)
            if memory_window is not None:
                cfg.memory_window = int(memory_window)
            if memory_num_layers is not None:
                cfg.memory_num_layers = int(memory_num_layers)
            if freeze_moment_tokens is not None:
                cfg.freeze_moment_tokens = bool(freeze_moment_tokens)
            if mem_cond_type is not None:
                cfg.mem_cond_type = str(mem_cond_type)
            if memory_type is not None:
                cfg.memory_type = str(memory_type)
            if tcl_tau is not None:
                cfg.tcl_tau = float(tcl_tau)
            kwargs["config"] = cfg

        # Request loading info so we can re-initialize ONLY the HAMLET params
        # genuinely absent from the checkpoint. Without this gate, loading a
        # TRAINED HAMLET checkpoint (eval / warm-start) would re-randomize the
        # trained moment_tokens / memory_transformer / mem_adaln_pool weights.
        had_loading_info = bool(kwargs.get("output_loading_info", False))
        kwargs["output_loading_info"] = True
        pretrained_model, loading_info = super().from_pretrained(
            local_model_path, local_model_path=local_model_path, **kwargs
        )
        missing_keys = list(loading_info.get("missing_keys", []))

        pretrained_model.backbone.set_trainable_parameters(
            tune_visual=tune_visual, tune_llm=tune_llm
        )
        if pretrained_model.hamlet_mode != "tcl":
            pretrained_model.action_head.set_trainable_parameters(
                tune_projector=tune_projector, tune_diffusion_model=tune_diffusion_model
            )

        # HAMLET params are absent from the base GR00T-N1.5 checkpoint, so HF
        # allocates them uninitialized (and this class inherits a no-op
        # `_init_weights`). Re-run their initializers to avoid NaN at step 0 —
        # but ONLY for keys actually missing from the loaded checkpoint
        # (a trained HAMLET checkpoint provides them all → no-op).
        pretrained_model._hamlet_reinit_missing_params(missing_keys=missing_keys)
        if had_loading_info:
            return pretrained_model, loading_info
        return pretrained_model

    def _hamlet_reinit_missing_params(self, missing_keys: Optional[list] = None) -> None:
        """Initialize HAMLET params (backbone.moment_tokens, memory_transformer.*,
        mem_adaln_pool) that the base checkpoint does not provide.

        Args:
            missing_keys: state-dict keys absent from the loaded checkpoint
                (from HF ``output_loading_info``). ``None`` = unknown loader
                context → legacy behavior (re-init everything) with a loud
                warning; trained checkpoints must never take that path.
        """
        if missing_keys is None:
            print(
                "[HAMLET][WARN] _hamlet_reinit_missing_params called without "
                "missing_keys — re-initializing ALL HAMLET params. If this "
                "model was loaded from a trained HAMLET checkpoint, its "
                "memory weights are now destroyed!"
            )
        tokens_missing = missing_keys is None or any(
            k == "backbone.moment_tokens" or k.endswith(".moment_tokens") for k in missing_keys
        )
        memory_missing = missing_keys is None or any(
            "memory_transformer." in k for k in missing_keys
        )
        adaln_pool_missing = missing_keys is None or any(
            "mem_adaln_pool." in k for k in missing_keys
        )
        with torch.no_grad():
            if tokens_missing and getattr(self, "backbone", None) is not None and getattr(
                self.backbone, "moment_tokens", None
            ) is not None:
                p = self.backbone.moment_tokens
                # 0.02 * N(0, 1), keeping the materialized dtype/device.
                new_data = (0.02 * torch.randn_like(p, dtype=torch.float32)).to(
                    dtype=p.dtype, device=p.device
                )
                p.data.copy_(new_data)
                print(
                    f"[HAMLET] reinit backbone.moment_tokens shape={tuple(p.shape)} "
                    f"dtype={p.dtype} mean={p.float().mean().item():.4e} "
                    f"std={p.float().std().item():.4e}"
                )
            if memory_missing and getattr(self, "memory_transformer", None) is not None:
                mt = self.memory_transformer
                # Re-run the module init: nn.Linear -> N(0, init_range); _RMSNorm.weight -> 1.0.
                init_range = getattr(mt, "_init_range", 0.02)
                for m in mt.modules():
                    if isinstance(m, torch.nn.Linear):
                        m.weight.data.normal_(mean=0.0, std=init_range)
                        if m.bias is not None:
                            m.bias.data.zero_()
                    # _RMSNorm: has a `weight` Parameter and a scalar `eps`.
                    elif hasattr(m, "weight") and m.__class__.__name__ == "_RMSNorm":
                        m.weight.data.fill_(1.0)
                print(
                    f"[HAMLET] reinit memory_transformer "
                    f"(linears N(0,{init_range}), RMSNorm.weight=1.0)"
                )
            if adaln_pool_missing and getattr(self, "mem_adaln_pool", None) is not None:
                # AdaLN-zero pool: zero-init projection (no-op at init).
                self.mem_adaln_pool.reset_parameters()
                print("[HAMLET] reinit mem_adaln_pool (proj zero-init)")
            # TCL head projection (hamlet_mode="tcl" only): absent from the base
            # ckpt → same empty-allocation issue as above.
            tcl_head = getattr(self, "action_head", None)
            if (
                (missing_keys is None or any("moment_to_repr" in k for k in missing_keys))
                and tcl_head is not None
                and hasattr(tcl_head, "moment_to_repr")
            ):
                for m in tcl_head.moment_to_repr.modules():
                    if isinstance(m, torch.nn.Linear):
                        m.weight.data.normal_(mean=0.0, std=0.02)
                        if m.bias is not None:
                            m.bias.data.zero_()
                print("[HAMLET] reinit TCL head moment_to_repr (N(0,0.02))")


# register
AutoConfig.register("gr00t_n1_5", GR00T_N1_5_Config)
AutoModel.register(GR00T_N1_5_Config, GR00T_N1_5)

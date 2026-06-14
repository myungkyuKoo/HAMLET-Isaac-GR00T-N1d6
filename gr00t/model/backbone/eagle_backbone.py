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
import os

import torch
from torch import nn
from transformers import AutoConfig, AutoModel
from transformers.feature_extraction_utils import BatchFeature

import gr00t

DEFAULT_EAGLE_PATH = os.path.join(
    os.path.dirname(gr00t.__file__), "model", "backbone", "eagle2_hg_model"
)

# Eagle LLM hidden size; moment tokens live at this dim before eagle_linear.
EAGLE_HIDDEN_DIM = 2048


class EagleBackbone(nn.Module):

    def __init__(
        self,
        tune_llm: bool = False,
        tune_visual: bool = False,
        select_layer: int = -1,
        reproject_vision: bool = False,
        use_flash_attention: bool = False,
        load_bf16: bool = False,
        eagle_path: str | None = None,
        project_to_dim: int = 1536,
        n_moment_tokens: int = 0,
        freeze_moment_tokens: bool = False,
        memory_type: str = "moment_token",
    ):
        """
        Args:
            tune_llm: whether to tune the LLM model (default: True)
            tune_visual: whether to tune the visual model (default: False)
            n_moment_tokens: if >0, append this many learnable moment tokens at
                the tail of the LLM input. They attend to image+text via causal
                self-attention; their VLM output slice feeds the memory module
                (or the TCL head).
            freeze_moment_tokens: if True, freeze the moment-token parameter.
        """
        super().__init__()
        assert not reproject_vision, "Reproject vision is not implemented here, set to False"

        config = AutoConfig.from_pretrained(DEFAULT_EAGLE_PATH, trust_remote_code=True)
        self.eagle_model = AutoModel.from_config(config, trust_remote_code=True)

        if project_to_dim is not None:
            self.eagle_linear = torch.nn.Linear(EAGLE_HIDDEN_DIM, project_to_dim)
        else:
            self.eagle_linear = torch.nn.Identity()

        # needed since we don't use these layers. Also saves compute
        while len(self.eagle_model.language_model.model.layers) > select_layer:
            self.eagle_model.language_model.model.layers.pop(-1)

        self.select_layer = select_layer
        self.memory_type = memory_type

        # HAMLET moment tokens, initialized to 0.02 * N(0, 1).
        self.n_moment_tokens = int(n_moment_tokens)
        # Persist the freeze flag so set_trainable_parameters (which resets
        # requires_grad=True on ALL params, and is re-invoked after
        # from_pretrained) re-applies the moment-token freeze every time.
        self.freeze_moment_tokens = bool(freeze_moment_tokens)
        if self.n_moment_tokens > 0:
            self.moment_tokens = nn.Parameter(0.02 * torch.randn(self.n_moment_tokens, EAGLE_HIDDEN_DIM))
        else:
            self.moment_tokens = None

        self.set_trainable_parameters(tune_llm, tune_visual)

    def set_trainable_parameters(self, tune_llm: bool, tune_visual: bool):
        self.tune_llm = tune_llm
        self.tune_visual = tune_visual
        for p in self.parameters():
            p.requires_grad = True
        if not tune_llm:
            self.eagle_model.language_model.requires_grad_(False)
        if not tune_visual:
            self.eagle_model.vision_model.requires_grad_(False)
            self.eagle_model.mlp1.requires_grad_(False)
        if getattr(self, "moment_tokens", None) is not None and getattr(
            self, "freeze_moment_tokens", False
        ):
            self.moment_tokens.requires_grad_(False)
        print(f"Tune backbone llm: {self.tune_llm}")
        print(f"Tune backbone visual: {self.tune_visual}")
        # Check if any parameters are still trainable. If not, print a warning.
        if not tune_llm and not tune_visual:
            for name, p in self.named_parameters():
                if p.requires_grad:
                    print(f"Backbone trainable parameter: {name}")
        if not any(p.requires_grad for p in self.parameters()):
            print("Warning: No backbone trainable parameters found.")

    def set_frozen_modules_to_eval_mode(self):
        """
        Huggingface will call model.train() at each training_step. To ensure
        the expected behaviors for modules like dropout, batchnorm, etc., we
        need to call model.eval() for the frozen modules.
        """
        if self.training:
            if self.eagle_model.language_model and not self.tune_llm:
                self.eagle_model.language_model.eval()
            if self.eagle_model.vision_model and not self.tune_visual:
                self.eagle_model.vision_model.eval()

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    def forward_eagle(self, vl_input: BatchFeature) -> BatchFeature:
        eagle_prefix = "eagle_"
        eagle_input = {
            k.removeprefix(eagle_prefix): v
            for k, v in vl_input.items()
            if k.startswith(eagle_prefix)
        }
        del eagle_input["image_sizes"]

        if self.moment_tokens is not None:
            return self._forward_eagle_with_moment_tokens(eagle_input)

        eagle_output = self.eagle_model(**eagle_input, output_hidden_states=True, return_dict=True)
        eagle_features = eagle_output.hidden_states[self.select_layer]

        eagle_features = self.eagle_linear(eagle_features)
        return eagle_features, eagle_input["attention_mask"]

    def _forward_eagle_with_moment_tokens(self, eagle_input: dict):
        """HAMLET forward: append moment_tokens to language_model input embeds, run
        the LLM with the extended sequence, project, and return.

        Last ``n_moment_tokens`` rows of ``eagle_features`` are the moment-token
        outputs m'_t consumed by the memory transformer / TCL head.
        """
        bsz = eagle_input["input_ids"].size(0)
        device = eagle_input["input_ids"].device
        dtype = self.eagle_model.language_model.model.embed_tokens.weight.dtype

        # Build mixed token+image embeddings.
        token_emb = self.eagle_model.language_model.get_input_embeddings()(
            eagle_input["input_ids"]
        )  # (B, T, d_llm)
        image_emb = self.eagle_model.extract_feature(eagle_input["pixel_values"]).to(dtype=dtype)
        patch_emb = image_emb.reshape(-1, image_emb.size(-1))  # (sum_patches, d_llm)
        image_mask = eagle_input["input_ids"] == self.eagle_model.image_token_index
        assert int(image_mask.sum()) == patch_emb.size(0), (
            f"#<image_pad> tokens ({int(image_mask.sum())}) != #patch embeddings ({patch_emb.size(0)})"
        )
        token_emb = token_emb.clone()
        token_emb[image_mask] = patch_emb

        # Append moment tokens at the tail; causal self-attention lets them see
        # all preceding text+image tokens.
        moment_raw = self.moment_tokens.to(dtype).unsqueeze(0).expand(bsz, -1, -1)
        full_emb = torch.cat([token_emb, moment_raw], dim=1)
        attn_mask = eagle_input["attention_mask"]
        moment_mask = torch.ones(bsz, self.n_moment_tokens, dtype=attn_mask.dtype, device=device)
        full_attn = torch.cat([attn_mask, moment_mask], dim=1)

        eagle_output = self.eagle_model.language_model(
            inputs_embeds=full_emb,
            attention_mask=full_attn,
            output_hidden_states=True,
            return_dict=True,
        )
        eagle_features = eagle_output.hidden_states[self.select_layer]
        eagle_features = self.eagle_linear(eagle_features)
        return eagle_features, full_attn

    def forward(self, vl_input: BatchFeature) -> BatchFeature:
        self.set_frozen_modules_to_eval_mode()

        eagle_embeds, eagle_mask = self.forward_eagle(vl_input)

        # YL (TODO HACK): to resolve DDP issue when tune_visual=True
        # Ensure all trainable parameters in vision_model are used in the forward pass for DDP compatibility
        if self.training and self.tune_visual:
            dummy_term = torch.tensor(
                0.0, device=eagle_embeds.device, dtype=eagle_embeds.dtype, requires_grad=True
            )
            for param in self.eagle_model.vision_model.parameters():
                if param.requires_grad:
                    dummy_term = dummy_term + 0.0 * param.sum()
            eagle_embeds = eagle_embeds + dummy_term

        data = {"backbone_features": eagle_embeds, "backbone_attention_mask": eagle_mask}
        if self.moment_tokens is not None:
            data["n_moment_tokens"] = self.n_moment_tokens
        elif self.memory_type == "vision_feature" and "eagle_input_ids" in vl_input:
            # vision_feature: expose the primary (first) view's post-LLM image
            # tokens, avg-pooled to 64/step, for the memory module. Each view is a
            # separate Eagle image, so n_views = pixel_values rows / batch and
            # tokens/view comes from the per-row image-token count (square grid).
            from gr00t.model.memory import pool_primary_view

            input_ids = vl_input["eagle_input_ids"]
            pixel_values = vl_input.get("eagle_pixel_values")
            image_mask = input_ids == self.eagle_model.image_token_index
            B = input_ids.shape[0]
            n_views = max(1, pixel_values.shape[0] // B) if pixel_values is not None else 1
            total = int(image_mask[0].sum().item())
            tpv = total // n_views
            side = int(round(tpv**0.5))
            if tpv > 0 and side * side == tpv:
                data["primary_view_feature"] = pool_primary_view(
                    eagle_embeds, image_mask, tpv, (side, side)
                )
        return BatchFeature(data=data)  # [B, T2, hidden_size]

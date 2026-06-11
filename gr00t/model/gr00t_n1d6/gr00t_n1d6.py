from typing import Any, Tuple

from gr00t.configs.model.gr00t_n1d6 import Gr00tN1d6Config
from gr00t.model.modules.dit import AlternateVLDiT, DiT
from gr00t.model.modules.eagle_backbone import EagleBackbone
from gr00t.model.modules.embodiment_conditioned_mlp import (
    CategorySpecificMLP,
    MultiEmbodimentActionEncoder,
)
from gr00t.model.modules.memory import MemoryTransformer
import torch
from torch import nn
from torch.distributions import Beta
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, PreTrainedModel
from transformers.feature_extraction_utils import BatchFeature
import tree


class _MemAdaLNPool(nn.Module):
    """Mean-pool memory tokens (B, n_q, d_in) -> (B, d_out) for AdaLN-zero conditioning.

    The output projection is zero-initialized so the AdaLN-memory path starts as an
    exact no-op; it learns to inject memory as training proceeds.
    """

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.proj = nn.Linear(d_in, d_out)
        self.reset_parameters()

    def reset_parameters(self):
        """Zero-init the projection (no-op at init). Safe to call after
        `from_pretrained` leaves these as missing/uninitialized keys."""
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, mem: torch.Tensor) -> torch.Tensor:  # mem: (B, n_q, d_in)
        pooled = mem.mean(dim=1)  # (B, d_in)
        return self.proj(pooled)  # (B, d_out); zero at init -> no-op


class Gr00tN1d6ActionHead(nn.Module):
    """Action head component for flow matching diffusion policy."""

    supports_gradient_checkpointing = True

    def __init__(self, config: Gr00tN1d6Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.input_embedding_dim = config.input_embedding_dim

        # Initialize components directly from config
        if config.use_alternate_vl_dit:
            self.model = AlternateVLDiT(
                **config.diffusion_model_cfg,
                cross_attention_dim=config.backbone_embedding_dim,
                attend_text_every_n_blocks=config.attend_text_every_n_blocks,
            )
            print("Using AlternateVLDiT for diffusion model")
        else:
            self.model = DiT(
                **config.diffusion_model_cfg, cross_attention_dim=config.backbone_embedding_dim
            )
            print("Using DiT for diffusion model")
        self.action_dim = config.max_action_dim
        self.action_horizon = config.action_horizon
        self.num_inference_timesteps = config.num_inference_timesteps

        self.state_encoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=config.max_state_dim,
            hidden_dim=self.hidden_size,
            output_dim=self.input_embedding_dim,
        )
        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim=self.action_dim,
            hidden_size=self.input_embedding_dim,
            num_embodiments=config.max_num_embodiments,
        )
        self.action_decoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim,
        )

        self.vlln = (
            nn.LayerNorm(config.backbone_embedding_dim) if config.use_vlln else nn.Identity()
        )

        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        # State dropout parameters
        self.state_dropout_prob = config.state_dropout_prob
        self.mask_token = (
            nn.Parameter(0.02 * torch.randn(1, 1, self.input_embedding_dim))
            if self.state_dropout_prob > 0
            else None
        )

        # State noise parameters
        self.state_additive_noise_scale = config.state_additive_noise_scale

        self.beta_dist = Beta(config.noise_beta_alpha, config.noise_beta_beta)
        self.num_timestep_buckets = config.num_timestep_buckets

        # The memory transformer aggregates K timesteps of n_q moment tokens
        # (oldest first to current last) and replaces the current step's tail with the
        # memory-augmented output. memory_num_layers == 0 means identity pass-through.
        self.use_hamlet = getattr(config, "hamlet_mode", "off") == "finetune"
        # memory_type: "moment_token" (n_q tokens/step) or "vision_feature"
        # (primary-view image tokens pooled to 64/step).
        self.memory_type = getattr(config, "memory_type", "moment_token")
        self._mem_tokens_per_step = (
            64 if self.memory_type == "vision_feature" else config.n_moment_tokens
        )
        if self.use_hamlet and getattr(config, "memory_num_layers", 0) > 0:
            self.memory_transformer = MemoryTransformer(
                dim=config.backbone_embedding_dim,
                n_q=self._mem_tokens_per_step,
                T=config.memory_window,
                num_layers=config.memory_num_layers,
            )
        else:
            self.memory_transformer = None

        # Inference-time rolling cache. Shape: (B, K*n_q, d), oldest first to current last.
        self._memory_cache: torch.Tensor | None = None
        self._vision_cache: torch.Tensor | None = None

        # memory-to-action conditioning. "cross_attn" (default) replaces the moment-token
        # tail of the action-head KV; "adaln" mean-pools the memory output through a
        # zero-init projection added to the DiT timestep embedding.
        self.mem_cond_type = getattr(config, "mem_cond_type", "cross_attn")
        if (
            self.use_hamlet
            and self.memory_transformer is not None
            and self.mem_cond_type == "adaln"
        ):
            self.mem_adaln_pool = _MemAdaLNPool(
                d_in=config.backbone_embedding_dim,
                d_out=self.model.inner_dim,
            )
        else:
            self.mem_adaln_pool = None

        self.set_trainable_parameters(
            config.tune_projector, config.tune_diffusion_model, config.tune_vlln
        )

    def set_trainable_parameters(
        self, tune_projector: bool, tune_diffusion_model: bool, tune_vlln: bool
    ):
        self.tune_projector = tune_projector
        self.tune_diffusion_model = tune_diffusion_model
        self.tune_vlln = tune_vlln
        for p in self.parameters():
            p.requires_grad = True
        if not tune_projector:
            self.state_encoder.requires_grad_(False)
            self.action_encoder.requires_grad_(False)
            self.action_decoder.requires_grad_(False)
            if self.config.add_pos_embed:
                self.position_embedding.requires_grad_(False)
            if self.state_dropout_prob > 0:
                self.mask_token.requires_grad_(False)
        if not tune_diffusion_model:
            self.model.requires_grad_(False)
        if not tune_vlln:
            self.vlln.requires_grad_(False)
        print(f"Tune action head projector: {self.tune_projector}")
        print(f"Tune action head diffusion model: {self.tune_diffusion_model}")
        print(f"Tune action head vlln: {self.tune_vlln}")
        # Check if any parameters are still trainable. If not, print a warning.
        if not tune_projector and not tune_diffusion_model and not tune_vlln:
            for name, p in self.named_parameters():
                if p.requires_grad:
                    print(f"Action head trainable parameter: {name}")
        if not any(p.requires_grad for p in self.parameters()):
            print("Warning: No action head trainable parameters found.")

    def set_frozen_modules_to_eval_mode(self):
        """
        Huggingface will call model.train() at each training_step. To ensure
        the expected behaviors for modules like dropout, batchnorm, etc., we
        need to call model.eval() for the frozen modules.
        """
        if self.training:
            if not self.tune_projector:
                self.state_encoder.eval()
                self.action_encoder.eval()
                self.action_decoder.eval()
                if self.config.add_pos_embed:
                    self.position_embedding.eval()
            if not self.tune_diffusion_model:
                self.model.eval()

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        sample = (1 - sample) * self.config.noise_s
        return sample

    def process_backbone_output(
        self,
        backbone_output: BatchFeature,
        action_inputs_B: int | None = None,
        reset_memory: torch.Tensor | None = None,
    ) -> BatchFeature:
        backbone_features = backbone_output["backbone_features"]
        backbone_features = self.vlln(backbone_features)

        if (
            self.use_hamlet
            and self.memory_transformer is not None
            and self.memory_type == "vision_feature"
            and "primary_view_feature" in backbone_output
            and action_inputs_B is not None
        ):
            # vision_feature: memory aggregates the primary view's pooled (64) tokens.
            # backbone_features (action-head conditioning) is the vanilla VLM output and is
            # NOT modified except (cross_attn only) by APPENDING the current memory tokens.
            v_nq = self._mem_tokens_per_step  # 64
            primary = backbone_output["primary_view_feature"]  # (B*K, 64, d)
            BK, _, d = primary.shape
            B = action_inputs_B
            K = BK // B
            K_target = self.memory_transformer.T
            Tlen = backbone_features.shape[1]
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
                mem_out = self.memory_transformer(mem_seq)
                mem_aug = mem_out[:, -v_nq:, :]
                current = backbone_features.view(B, K, Tlen, d)[:, -1, :, :]  # unchanged
                am = (
                    backbone_output["backbone_attention_mask"].view(B, K, -1)[:, -1, :]
                    if "backbone_attention_mask" in backbone_output
                    else None
                )
                im = (
                    backbone_output["image_mask"].view(B, K, -1)[:, -1, :]
                    if "image_mask" in backbone_output
                    else None
                )
                if self.mem_cond_type == "adaln":
                    backbone_output["mem_temb_add"] = self.mem_adaln_pool(mem_aug)
                else:
                    current = torch.cat([current, mem_aug], dim=1)
                    if am is not None:
                        am = torch.cat([am, am.new_ones(B, v_nq)], dim=1)
                    if im is not None:
                        im = torch.cat([im, im.new_zeros(B, v_nq)], dim=1)
                backbone_features = current
                if am is not None:
                    backbone_output["backbone_attention_mask"] = am
                if im is not None:
                    backbone_output["image_mask"] = im
            elif K == 1:
                vis_current = primary  # (B, 64, d)
                if self._vision_cache is None or self._vision_cache.shape[0] != B:
                    self._vision_cache = vis_current.repeat(1, K_target, 1)
                elif reset_memory is not None and reset_memory.any():
                    defaults = vis_current.repeat(1, K_target, 1)
                    shifted = torch.cat([self._vision_cache[:, v_nq:, :], vis_current], dim=1)
                    reset_b = reset_memory.view(B, 1, 1).expand(B, K_target * v_nq, d)
                    self._vision_cache = torch.where(reset_b, defaults, shifted)
                else:
                    self._vision_cache = torch.cat(
                        [self._vision_cache[:, v_nq:, :], vis_current], dim=1
                    )
                mem_out = self.memory_transformer(self._vision_cache)
                mem_aug = mem_out[:, -v_nq:, :]
                if self.mem_cond_type == "adaln":
                    backbone_output["mem_temb_add"] = self.mem_adaln_pool(mem_aug)
                else:
                    backbone_features = torch.cat([backbone_features, mem_aug], dim=1)
                    if "backbone_attention_mask" in backbone_output:
                        am = backbone_output["backbone_attention_mask"]
                        backbone_output["backbone_attention_mask"] = torch.cat(
                            [am, am.new_ones(B, v_nq)], dim=1
                        )
                    if "image_mask" in backbone_output:
                        im = backbone_output["image_mask"]
                        backbone_output["image_mask"] = torch.cat(
                            [im, im.new_zeros(B, v_nq)], dim=1
                        )
            # else: unexpected K — pass through.
        elif (
            self.use_hamlet
            and self.memory_transformer is not None
            and "n_moment_tokens" in backbone_output
            and action_inputs_B is not None
        ):
            n_q = int(backbone_output["n_moment_tokens"])
            BK, T, d = backbone_features.shape
            B = action_inputs_B
            K = BK // B
            assert BK == B * K, f"expected B*K rows, got {BK} for B={B}"
            K_target = self.memory_transformer.T

            if K not in (1, K_target):
                raise RuntimeError(
                    f"HAMLET memory: got K={K} backbone rows per action sample "
                    f"(expected 1 for rolling inference or memory_window={K_target} "
                    f"for K-step training). The video delta_indices / memory_window "
                    f"data config is inconsistent - refusing to silently skip "
                    f"memory augmentation."
                )
            if K == K_target:
                # K-step training path: backbone gave K real timesteps per sample.
                moment_all = backbone_features[:, -n_q:, :].contiguous().view(B, K, n_q, d)
                mq_mem_seq = moment_all.view(B, K * n_q, d)  # oldest first -> current last
                mq_memory_out = self.memory_transformer(mq_mem_seq)
                mq_augmented = mq_memory_out[:, -n_q:, :]
                current = backbone_features.view(B, K, T, d)[:, -1, :, :]
                am = (
                    backbone_output["backbone_attention_mask"].view(B, K, -1)[:, -1, :]
                    if "backbone_attention_mask" in backbone_output
                    else None
                )
                im = (
                    backbone_output["image_mask"].view(B, K, -1)[:, -1, :]
                    if "image_mask" in backbone_output
                    else None
                )
                if self.mem_cond_type == "adaln":
                    # AdaLN-zero: pool memory -> temb add; slice moment-token tail off the KV.
                    backbone_output["mem_temb_add"] = self.mem_adaln_pool(mq_augmented)
                    current = current[:, :-n_q, :]
                    if am is not None:
                        am = am[:, :-n_q]
                    if im is not None:
                        im = im[:, :-n_q]
                else:
                    # cross_attn: memory-augmented tokens replace the moment-token tail.
                    current = torch.cat([current[:, :-n_q, :], mq_augmented], dim=1)
                backbone_features = current
                if am is not None:
                    backbone_output["backbone_attention_mask"] = am
                if im is not None:
                    backbone_output["image_mask"] = im
            elif K == 1:
                # Inference path with rolling FIFO cache.
                moment_current = backbone_features[:, -n_q:, :]  # (B, n_q, d)

                if self._memory_cache is None or self._memory_cache.shape[0] != B:
                    # First call (or batch-size mismatch): K-replicate current moment-token to fill the window.
                    self._memory_cache = moment_current.repeat(1, K_target, 1)
                else:
                    if reset_memory is not None and reset_memory.any():
                        defaults = moment_current.repeat(1, K_target, 1)
                        shifted = torch.cat([self._memory_cache[:, n_q:, :], moment_current], dim=1)
                        reset_b = reset_memory.view(B, 1, 1).expand(B, K_target * n_q, d)
                        self._memory_cache = torch.where(reset_b, defaults, shifted)
                    else:
                        self._memory_cache = torch.cat(
                            [self._memory_cache[:, n_q:, :], moment_current], dim=1
                        )

                mq_memory_out = self.memory_transformer(self._memory_cache)
                mq_augmented = mq_memory_out[:, -n_q:, :]
                if self.mem_cond_type == "adaln":
                    backbone_output["mem_temb_add"] = self.mem_adaln_pool(mq_augmented)
                    backbone_features = backbone_features[:, :-n_q, :]
                    if "backbone_attention_mask" in backbone_output:
                        backbone_output["backbone_attention_mask"] = backbone_output[
                            "backbone_attention_mask"
                        ][:, :-n_q]
                    if "image_mask" in backbone_output:
                        backbone_output["image_mask"] = backbone_output["image_mask"][:, :-n_q]
                else:
                    backbone_features = torch.cat(
                        [backbone_features[:, :-n_q, :], mq_augmented], dim=1
                    )
            else:
                # Unexpected K — pass through.
                pass

        backbone_output["backbone_features"] = backbone_features
        return backbone_output

    def reset_memory(self):
        """Clear the rolling memory cache. Call at episode boundary."""
        self._memory_cache = None
        self._vision_cache = None

    def forward(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        """
        Forward pass through the action head.

        Args:
            backbone_output: Output from the backbone model containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - backbone_attention_mask: [B, seq_len]
            action_input: Input containing:
                - state: [B, state_dim]
                - action: [B, action_horizon, action_dim] (during training)
                - embodiment_id: [B] (embodiment IDs)
                - action_mask: [B, action_horizon, action_dim]

        Returns:
            BatchFeature containing:
                - loss: action prediction loss
        """
        # Set frozen modules to eval
        self.set_frozen_modules_to_eval_mode()

        # K-step HAMLET: pass the action-input batch size so process_backbone_output can
        # collapse the B*K backbone rows back to B current-step rows after memory aggregation.
        B_target = action_input.state.shape[0]
        backbone_output = self.process_backbone_output(backbone_output, action_inputs_B=B_target)

        # Get vision and language embeddings.
        vl_embeds = backbone_output.backbone_features
        device = vl_embeds.device

        # Get embodiment ID.
        embodiment_id = action_input.embodiment_id

        # Embed state.
        state_features = self.state_encoder(action_input.state, embodiment_id)

        # Dropout state features.
        if self.state_dropout_prob > 0:
            do_dropout = (
                torch.rand(state_features.shape[0], device=state_features.device)
                < self.state_dropout_prob
            )
            do_dropout = do_dropout[:, None, None].to(dtype=state_features.dtype)
            state_features = state_features * (1 - do_dropout) + self.mask_token * do_dropout

        # Add Gaussian noise to state features.
        if self.training and self.state_additive_noise_scale > 0:
            print(
                f"Adding Gaussian noise to state features with scale {self.state_additive_noise_scale}"
            )
            noise = torch.randn_like(state_features) * self.state_additive_noise_scale
            state_features = state_features + noise

        # Embed noised action trajectory.
        actions = action_input.action
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
        t = t[:, None, None]  # shape (B,1,1) for broadcast

        noisy_trajectory = (1 - t) * noise + t * actions
        velocity = actions - noise

        # Convert (continuous) t -> discrete if needed
        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        action_features = self.action_encoder(noisy_trajectory, t_discretized, embodiment_id)

        # Maybe add position embedding.
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        # Join vision, language, state and action embedding along sequence dimension.
        sa_embs = torch.cat((state_features, action_features), dim=1)
        vl_attn_mask = backbone_output.backbone_attention_mask
        # AdaLN-zero HAMLET: pooled memory vector added to the DiT timestep embedding.
        mem_temb_add = backbone_output.get("mem_temb_add", None)

        if self.config.use_alternate_vl_dit:
            image_mask = backbone_output.image_mask
            backbone_attention_mask = backbone_output.backbone_attention_mask
            model_output, _ = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                encoder_attention_mask=vl_attn_mask,
                timestep=t_discretized,
                return_all_hidden_states=True,
                image_mask=image_mask,
                backbone_attention_mask=backbone_attention_mask,
                temb_add=mem_temb_add,
            )
        else:
            model_output, _ = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                encoder_attention_mask=vl_attn_mask,
                timestep=t_discretized,
                return_all_hidden_states=True,
                temb_add=mem_temb_add,
            )

        pred = self.action_decoder(model_output, embodiment_id)
        pred_actions = pred[:, -actions.shape[1] :]

        # Slice out only the action portion of pred and target.
        action_mask = action_input.action_mask
        action_loss = F.mse_loss(pred_actions, velocity, reduction="none") * action_mask
        loss = action_loss.sum() / (action_mask.sum() + 1e-6)

        return {
            "loss": loss,
            "action_loss": action_loss,
            "action_mask": action_mask,
            "backbone_features": vl_embeds,
            "state_features": state_features,
        }

    def _encode_features(
        self,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
        reset_memory: torch.Tensor | None = None,
    ) -> BatchFeature:
        """
        Encode features for the action head.
        """
        B_target = action_input.state.shape[0]
        backbone_output = self.process_backbone_output(
            backbone_output, action_inputs_B=B_target, reset_memory=reset_memory
        )

        # Get vision and language embeddings.
        vl_embeds = backbone_output.backbone_features
        embodiment_id = action_input.embodiment_id

        # Embed state.
        state_features = self.state_encoder(action_input.state, embodiment_id)

        return BatchFeature(data={"backbone_features": vl_embeds, "state_features": state_features})

    @torch.no_grad()
    def get_action_with_features(
        self,
        backbone_features: torch.Tensor,
        state_features: torch.Tensor,
        embodiment_id: torch.Tensor,
        backbone_output: BatchFeature,
    ) -> BatchFeature:
        """
        Generate actions using the flow matching diffusion process.

        Args:
            backbone_features: [B, seq_len, backbone_embedding_dim]
            state_features: [B, state_horizon, input_embedding_dim]
            embodiment_id: [B] (embodiment IDs)
            backbone_output: Output from the backbone model
        """
        vl_embeds = backbone_features
        # AdaLN-zero HAMLET: pooled memory vector added to the DiT timestep embedding.
        mem_temb_add = backbone_output.get("mem_temb_add", None)

        # Set initial actions as the sampled noise. When the env var
        # GR00T_INFERENCE_SEED is set, draw from a persistent per-device generator
        # seeded with that value so RoboMME eval is fully deterministic and
        # reproducible across runs (the generator advances per call -> varied but
        # reproducible noise). Default (unset) = nondeterministic global RNG (unchanged).
        batch_size = vl_embeds.shape[0]
        device = vl_embeds.device
        _noise_size = (batch_size, self.config.action_horizon, self.action_dim)
        import os as _os
        _inf_seed = _os.environ.get("GR00T_INFERENCE_SEED")
        if _inf_seed is not None:
            _gen = getattr(self, "_inference_gen", None)
            if _gen is None:
                _gen = torch.Generator(device=device)
                _gen.manual_seed(int(_inf_seed))
                self._inference_gen = _gen
            actions = torch.randn(size=_noise_size, dtype=vl_embeds.dtype, device=device, generator=_gen)
        else:
            actions = torch.randn(size=_noise_size, dtype=vl_embeds.dtype, device=device)

        dt = 1.0 / self.num_inference_timesteps

        # Run denoising steps.
        for t in range(self.num_inference_timesteps):
            t_cont = t / float(self.num_inference_timesteps)  # e.g. goes 0, 1/N, 2/N, ...
            t_discretized = int(t_cont * self.num_timestep_buckets)

            # Embed noised action trajectory.
            timesteps_tensor = torch.full(
                size=(batch_size,), fill_value=t_discretized, device=device
            )
            action_features = self.action_encoder(actions, timesteps_tensor, embodiment_id)
            # Add position embedding.
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            # Join vision, language, state and action embedding along sequence dimension.
            sa_embs = torch.cat((state_features, action_features), dim=1)

            # Run model forward.
            if self.config.use_alternate_vl_dit:
                model_output = self.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embeds,
                    timestep=timesteps_tensor,
                    image_mask=backbone_output.image_mask,
                    backbone_attention_mask=backbone_output.backbone_attention_mask,
                    temb_add=mem_temb_add,
                )
            else:
                model_output = self.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embeds,
                    timestep=timesteps_tensor,
                    temb_add=mem_temb_add,
                )
            pred = self.action_decoder(model_output, embodiment_id)

            pred_velocity = pred[:, -self.action_horizon :]

            # Update actions using euler integration.
            actions = actions + dt * pred_velocity
        return BatchFeature(
            data={
                "action_pred": actions,
                "backbone_features": vl_embeds,
                "state_features": state_features,
            }
        )

    @torch.no_grad()
    def get_action(
        self,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
        options: dict[str, Any] | None = None,
    ) -> BatchFeature:
        """
        Generate actions using the flow matching diffusion process.
        """
        reset_memory = options.get("reset_memory") if options else None
        features = self._encode_features(backbone_output, action_input, reset_memory=reset_memory)
        if options and options.get("prime_only"):
            # Memory-priming call: the cache update in _encode_features is all that is
            # needed. Skip flow-matching denoising so the (seeded) action-noise RNG is
            # NOT advanced — keeps action noise call-aligned with non-primed policies
            # for strict paired comparison (and makes priming much cheaper).
            ref = features.backbone_features
            zeros = torch.zeros(
                ref.shape[0], self.config.action_horizon, self.action_dim,
                device=ref.device, dtype=ref.dtype,
            )
            return BatchFeature(data={"action_pred": zeros})
        return self.get_action_with_features(
            backbone_features=features.backbone_features,
            state_features=features.state_features,
            embodiment_id=action_input.embodiment_id,
            backbone_output=backbone_output,
        )

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype

    def prepare_input(self, batch: dict) -> BatchFeature:
        """Prepare input batch for the action head."""
        return BatchFeature(data=batch)


def get_backbone_cls(config: Gr00tN1d6Config):
    if "NVEagle" in config.model_name or "nvidia/Eagle" in config.model_name:
        return EagleBackbone
    else:
        raise ValueError(f"Unsupported model name: {config.model_name}")


class Gr00tN1d6(PreTrainedModel):
    """Gr00tN1d6: Vision-Language-Action model with backbone."""

    config_class = Gr00tN1d6Config
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: Gr00tN1d6Config,
        transformers_loading_kwargs: dict = {"trust_remote_code": True},
    ):
        """
        Initialize Gr00tN1d6 model.

        Args:
            config: Model configuration
            transformers_loading_kwargs: Dict with transformers loading parameters:
                - transformers_trust_remote_code: Whether to trust remote code when loading from HF Hub
                - transformers_local_files_only: Whether to only use local files
                - model_revision: Specific model revision to use
                - transformers_cache_dir: Directory to cache downloaded models
                - transformers_access_token: HuggingFace access token for gated models

        Note: During training, transformers parameters are passed from training config.
              During inference (e.g., from_pretrained), defaults are used.
        """
        super().__init__(config)
        self.config = config

        backbone_cls = get_backbone_cls(config)
        backbone_kwargs = dict(
            model_name=config.model_name,
            tune_llm=config.tune_llm,
            tune_visual=config.tune_visual,
            select_layer=config.select_layer,
            reproject_vision=config.reproject_vision,
            use_flash_attention=config.use_flash_attention,
            load_bf16=config.load_bf16,
            tune_top_llm_layers=config.tune_top_llm_layers,
            trainable_params_fp32=config.backbone_trainable_params_fp32,
            transformers_loading_kwargs=transformers_loading_kwargs,
        )
        if getattr(config, "hamlet_mode", "off") != "off":
            mem_type = getattr(config, "memory_type", "moment_token")
            backbone_kwargs["memory_type"] = mem_type
            if mem_type == "vision_feature":
                backbone_kwargs["n_moment_tokens"] = 0
                backbone_kwargs["freeze_moment_tokens"] = False
            else:
                backbone_kwargs["n_moment_tokens"] = config.n_moment_tokens
                backbone_kwargs["freeze_moment_tokens"] = config.freeze_moment_tokens
        self.backbone = backbone_cls(**backbone_kwargs)

        # Initialize action head (TCL mode swaps in a contrastive head).
        if getattr(config, "hamlet_mode", "off") == "tcl":
            from .tcl_head import Gr00tN1d6TCLHead
            self.action_head = Gr00tN1d6TCLHead(
                backbone_embedding_dim=config.backbone_embedding_dim,
                tcl_tau=config.tcl_tau,
            )
            # TCL stage: freeze everything except moment_tokens and moment_to_repr.
            for p in self.backbone.parameters():
                p.requires_grad = False
            if getattr(self.backbone, "moment_tokens", None) is not None:
                self.backbone.moment_tokens.requires_grad = True
            for p in self.action_head.parameters():
                p.requires_grad = True
        else:
            self.action_head = Gr00tN1d6ActionHead(config)
        from .processing_gr00t_n1d6 import Gr00tN1d6DataCollator

        self.collator = Gr00tN1d6DataCollator(
            model_name=config.model_name,
            model_type=config.backbone_model_type,
            transformers_loading_kwargs=transformers_loading_kwargs,
        )

    def prepare_input(self, inputs: dict) -> Tuple[BatchFeature, BatchFeature]:
        """Prepare inputs for backbone and action head."""

        # NOTE -- currently the eval code doesn't use collator, so we need to add it here
        # this should ideally be fixed upstream
        if "vlm_content" in inputs:
            # Fix for n_envs > 1: Process all environments' VLM content, not just the first
            vlm_content_list = inputs["vlm_content"]
            # Ensure vlm_content_list is always a list for consistent processing
            if not isinstance(vlm_content_list, list):
                vlm_content_list = [vlm_content_list]

            # Process all VLM contents through the collator
            prep = self.collator([{"vlm_content": vlm} for vlm in vlm_content_list])["inputs"]
            inputs.pop("vlm_content")
            inputs.update(prep)

        backbone_inputs = self.backbone.prepare_input(inputs)
        action_inputs = self.action_head.prepare_input(inputs)

        # Move to device and dtype
        def to_device_with_dtype(x):
            if torch.is_floating_point(x):
                return x.to(self.device, dtype=self.dtype)
            else:
                return x.to(self.device)

        backbone_inputs = tree.map_structure(to_device_with_dtype, backbone_inputs)
        action_inputs = tree.map_structure(to_device_with_dtype, action_inputs)

        return backbone_inputs, action_inputs

    def forward(self, inputs: dict) -> BatchFeature:
        """Forward pass.
        - hamlet_mode == "tcl": run backbone 3× (anchor / aug / neg) and pass to TCL head.
        - else: standard single-pass.
        """
        if getattr(self.config, "hamlet_mode", "off") == "tcl":
            return self._forward_tcl(inputs)

        # Prepare inputs for backbone and action head
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_outputs = self.backbone(backbone_inputs)
        action_outputs = self.action_head(backbone_outputs, action_inputs)

        return action_outputs

    def _forward_tcl(self, inputs: dict) -> BatchFeature:
        """TCL Stage-1: split inputs into anchor / aug / neg streams, run backbone three
        times, pass the three backbone_features tensors to the TCL head for InfoNCE.
        """
        backbone_inputs, action_inputs = self.prepare_input(inputs)

        def _slice(prefix: str) -> BatchFeature:
            data = {
                "input_ids": backbone_inputs[f"{prefix}input_ids"],
                "attention_mask": backbone_inputs[f"{prefix}attention_mask"],
                "pixel_values": backbone_inputs[f"{prefix}pixel_values"],
            }
            # Forward optional fields when present (Eagle processor may include extras).
            for opt in ("image_grid_thw",):
                if f"{prefix}{opt}" in backbone_inputs:
                    data[opt] = backbone_inputs[f"{prefix}{opt}"]
            return BatchFeature(data=data)

        anc_out = self.backbone(_slice(""))
        aug_out = self.backbone(_slice("aug_"))
        neg_out = self.backbone(_slice("neg_"))
        return self.action_head(anc_out, aug_out, neg_out, action_inputs)

    def get_action(
        self, inputs: dict, options: dict[str, Any] | None = None
    ) -> BatchFeature:
        """
        Generate actions using the complete model.
        """
        # Prepare inputs for backbone and action head
        backbone_inputs, action_inputs = self.prepare_input(inputs)

        # Forward through backbone
        backbone_outputs = self.backbone(backbone_inputs)
        action_outputs = self.action_head.get_action(backbone_outputs, action_inputs, options)

        return action_outputs

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype


# Register the model with HuggingFace
AutoConfig.register("Gr00tN1d6", Gr00tN1d6Config)
AutoModel.register(Gr00tN1d6Config, Gr00tN1d6)

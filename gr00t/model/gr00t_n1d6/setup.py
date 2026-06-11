import json
import logging
from pathlib import Path

from gr00t.configs.base_config import Config
from gr00t.configs.model.gr00t_n1d6 import Gr00tN1d6Config
from gr00t.data.dataset.factory import DatasetFactory
from gr00t.experiment.dist_utils import get_rank
from gr00t.model.base.model_pipeline import ModelPipeline
from gr00t.model.gr00t_n1d6.gr00t_n1d6 import Gr00tN1d6
from gr00t.model.gr00t_n1d6.processing_gr00t_n1d6 import Gr00tN1d6Processor
from gr00t.model.registry import register_model
import numpy as np
from termcolor import colored
import torch
from transformers import AutoModel, AutoProcessor


# Convert tensors to lists for JSON serialization
def convert_tensors_to_lists(obj):
    """Recursively convert tensors to lists in nested dictionaries/lists."""
    if torch.is_tensor(obj) or isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_tensors_to_lists(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_tensors_to_lists(item) for item in obj]
    else:
        return obj


class Gr00tN1d6Pipeline(ModelPipeline):
    model_class = Gr00tN1d6
    processor_class = Gr00tN1d6Processor

    def __init__(self, config: Config, save_cfg_dir: Path):
        super().__init__(config)
        self.save_cfg_dir = save_cfg_dir

        # Build transformers loading kwargs from training config
        transformers_loading_kwargs = {
            "trust_remote_code": self.config.training.transformers_trust_remote_code,
            "local_files_only": self.config.training.transformers_local_files_only,
        }
        if self.model_config.model_revision is not None:
            transformers_loading_kwargs["revision"] = self.model_config.model_revision
        if self.config.training.transformers_cache_dir is not None:
            transformers_loading_kwargs["cache_dir"] = self.config.training.transformers_cache_dir
        if self.config.training.transformers_access_token is not None:
            transformers_loading_kwargs["token"] = self.config.training.transformers_access_token

        self.transformers_loading_kwargs = transformers_loading_kwargs

    @property
    def model_config(self):
        return self.config.model

    def setup(self):
        self.model = self._create_model()
        self.train_dataset, self.eval_dataset = self._create_dataset(self.save_cfg_dir)
        self.data_collator = self._create_collator()

    def _create_model(self):
        """Setup model with proper vocabulary expansion."""

        # Build transformers loading kwargs from training config
        skip_weight_loading = getattr(self.config.training, "skip_weight_loading", False)
        if self.config.training.start_from_checkpoint is not None and not skip_weight_loading:
            model, loading_info = AutoModel.from_pretrained(
                self.config.training.start_from_checkpoint,
                tune_llm=self.config.model.tune_llm,
                tune_visual=self.config.model.tune_visual,
                tune_projector=self.config.model.tune_projector,
                tune_diffusion_model=self.config.model.tune_diffusion_model,
                tune_vlln=self.config.model.tune_vlln,
                state_dropout_prob=self.config.model.state_dropout_prob,
                backbone_trainable_params_fp32=self.config.model.backbone_trainable_params_fp32,
                # HAMLET overrides (control whether moment_tokens / memory_transformer are instantiated).
                hamlet_mode=self.config.model.hamlet_mode,
                n_moment_tokens=self.config.model.n_moment_tokens,
                memory_window=self.config.model.memory_window,
                memory_num_layers=self.config.model.memory_num_layers,
                memory_stride=self.config.model.memory_stride,
                mem_cond_type=self.config.model.mem_cond_type,
                freeze_moment_tokens=self.config.model.freeze_moment_tokens,
                memory_type=self.config.model.memory_type,
                tcl_tau=self.config.model.tcl_tau,
                transformers_loading_kwargs=self.transformers_loading_kwargs,
                output_loading_info=True,
                **self.transformers_loading_kwargs,
            )

            # Initialize mask_tokens if they are not present in the base checkpoint
            missing_keys = loading_info.get("missing_keys", [])
            mask_token_missing = any("mask_token" in key for key in missing_keys)

            if mask_token_missing and getattr(model.action_head, "mask_token", None) is not None:
                # Initialize mask_token
                with torch.no_grad():
                    model.action_head.mask_token.data.copy_(
                        0.02 * torch.randn_like(model.action_head.mask_token)
                    )
                logging.info("mask_token not in checkpoint - initialized")

            # HAMLET — new modules are absent from a vanilla N1.6 base checkpoint.
            # HF from_pretrained fills missing keys with torch.empty() (uninitialized memory
            # -> potential NaN). Re-initialize them explicitly here.
            hamlet_key_substrings = (
                "backbone.moment_tokens",
                "memory_transformer",
                "moment_to_repr",
                "mem_adaln_pool",
            )

            def _is_tolerated(k: str) -> bool:
                if "mask_token" in k:
                    return True
                return any(sub in k for sub in hamlet_key_substrings)

            other_missing = [k for k in missing_keys if not _is_tolerated(k)]
            tolerated_missing = [k for k in missing_keys if _is_tolerated(k) and "mask_token" not in k]
            if tolerated_missing:
                logging.info(
                    f"HAMLET keys missing from checkpoint (re-initializing): {tolerated_missing[:8]}..."
                )
                with torch.no_grad():
                    if hasattr(model.backbone, "moment_tokens") and any(
                        "backbone.moment_tokens" in k for k in tolerated_missing
                    ):
                        model.backbone.moment_tokens.data.normal_(mean=0.0, std=0.02)
                    if (
                        getattr(model.action_head, "memory_transformer", None) is not None
                        and any("memory_transformer" in k for k in tolerated_missing)
                    ):
                        for m in model.action_head.memory_transformer.modules():
                            if isinstance(m, torch.nn.Linear):
                                m.weight.data.normal_(mean=0.0, std=0.02)
                        for n, p in model.action_head.memory_transformer.named_parameters():
                            if "norm" in n:
                                p.data.fill_(1.0)
                    if (
                        getattr(model.action_head, "moment_to_repr", None) is not None
                        and any("moment_to_repr" in k for k in tolerated_missing)
                    ):
                        for m in model.action_head.moment_to_repr.modules():
                            if isinstance(m, torch.nn.Linear):
                                m.weight.data.normal_(mean=0.0, std=0.02)
                    if (
                        getattr(model.action_head, "mem_adaln_pool", None) is not None
                        and any("mem_adaln_pool" in k for k in tolerated_missing)
                    ):
                        # AdaLN-zero pool: proj zero-init (no-op), attention query/attn proper init.
                        model.action_head.mem_adaln_pool.reset_parameters()

            unexpected_keys = loading_info.get("unexpected_keys", [])
            mismatched_keys = loading_info.get("mismatched_keys", [])
            # In TCL mode the action_head is replaced by Gr00tN1d6TCLHead, so the
            # diffusion-policy keys in the base ckpt are intentionally unexpected.
            if self.config.model.hamlet_mode == "tcl":
                unexpected_keys = [
                    k for k in unexpected_keys
                    if not (k.startswith("action_head.") and "moment_to_repr" not in k)
                ]
            errors = []
            if other_missing:
                errors.append(f"Missing keys ({len(other_missing)}): {other_missing}")
            if unexpected_keys:
                errors.append(f"Unexpected keys ({len(unexpected_keys)}): {unexpected_keys}")
            if mismatched_keys:
                errors.append(f"Mismatched keys ({len(mismatched_keys)}): {mismatched_keys}")
            if errors:
                raise RuntimeError(
                    "Checkpoint weight mismatch for "
                    f"{self.config.training.start_from_checkpoint}:\n" + "\n".join(errors)
                )

            # HAMLET Stage-2 entry: optionally overwrite freshly-initialized moment_tokens with
            # the TCL-pretrained version from a Stage-1 checkpoint.
            stage1_path = getattr(self.config.training, "load_moment_tokens_from", None)
            if stage1_path is not None and hasattr(model.backbone, "moment_tokens"):
                import os
                from safetensors import safe_open
                src_files = []
                if os.path.isdir(stage1_path):
                    for f in sorted(os.listdir(stage1_path)):
                        if f.endswith(".safetensors"):
                            src_files.append(os.path.join(stage1_path, f))
                elif stage1_path.endswith(".safetensors"):
                    src_files.append(stage1_path)
                loaded = False
                for f in src_files:
                    with safe_open(f, framework="pt") as st:
                        for k in st.keys():
                            if k.endswith("backbone.moment_tokens") or k == "backbone.moment_tokens":
                                t = st.get_tensor(k)
                                with torch.no_grad():
                                    model.backbone.moment_tokens.data.copy_(
                                        t.to(model.backbone.moment_tokens.dtype).to(
                                            model.backbone.moment_tokens.device
                                        )
                                    )
                                loaded = True
                                logging.info(
                                    f"[HAMLET] Loaded moment_tokens from {f} ({k}); shape={tuple(t.shape)}"
                                )
                                break
                    if loaded:
                        break
                if not loaded:
                    raise RuntimeError(
                        f"--load-moment-tokens-from set but backbone.moment_tokens not found under {stage1_path}"
                    )

        else:
            model = self.model_class(
                self.config.model, transformers_loading_kwargs=self.transformers_loading_kwargs
            )

        print(colored(f"Model Config: {model.config}", "yellow"))
        if get_rank() == 0:
            with open(self.save_cfg_dir / "final_model_config.json", "w") as f:
                f.write(model.config.to_filtered_json())
        # Print parameter statistics
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f"Total parameters: {total_params:,}")
        logging.info(
            f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)"
        )
        print("Model: ", model)

        return model

    def _get_statistics(self) -> dict[str, dict[str, dict[str, dict[str, list[float]]]]] | None:
        return None

    def _get_embodiment_id_mapping(self) -> dict[str, int]:
        return None

    def _create_dataset(self, save_cfg_dir: Path):
        """Create appropriate dataset based on task and mode."""

        if self.config.training.start_from_checkpoint is not None:
            processor = AutoProcessor.from_pretrained(
                self.config.training.start_from_checkpoint,
                # Overrides
                modality_configs=self.config.data.modality_configs,
                image_crop_size=self.model_config.image_crop_size,
                image_target_size=self.model_config.image_target_size,
                random_rotation_angle=self.model_config.random_rotation_angle,
                color_jitter_params=self.model_config.color_jitter_params,
                model_name=self.model_config.model_name,
                model_type=self.model_config.backbone_model_type,
                formalize_language=self.model_config.formalize_language,
                apply_sincos_state_encoding=self.model_config.apply_sincos_state_encoding,
                max_action_horizon=self.model_config.action_horizon,
                use_albumentations=self.model_config.use_albumentations_transforms,
                extra_augmentation_config=self.model_config.extra_augmentation_config,
                shortest_image_edge=self.model_config.shortest_image_edge,
                crop_fraction=self.model_config.crop_fraction,
                transformers_loading_kwargs=self.transformers_loading_kwargs,
                use_alternate_vl_dit=self.model_config.use_alternate_vl_dit,
                use_relative_action=self.model_config.use_relative_action,
                hamlet_mode=self.model_config.hamlet_mode,
                **self.transformers_loading_kwargs,
            )
        else:
            processor = self.processor_class(
                modality_configs=self.config.data.modality_configs,
                statistics=self._get_statistics(),  # By default is None, so this will be computed and set later.
                embodiment_id_mapping=self._get_embodiment_id_mapping(),  # By default is None, so this will be set later.
                image_crop_size=self.model_config.image_crop_size,
                image_target_size=self.model_config.image_target_size,
                random_rotation_angle=self.model_config.random_rotation_angle,
                color_jitter_params=self.model_config.color_jitter_params,
                model_name=self.model_config.model_name,
                model_type=self.model_config.backbone_model_type,
                formalize_language=self.model_config.formalize_language,
                max_state_dim=self.model_config.max_state_dim,
                max_action_dim=self.model_config.max_action_dim,
                apply_sincos_state_encoding=self.model_config.apply_sincos_state_encoding,
                max_action_horizon=self.model_config.action_horizon,
                use_albumentations=self.model_config.use_albumentations_transforms,
                extra_augmentation_config=self.model_config.extra_augmentation_config,
                shortest_image_edge=self.model_config.shortest_image_edge,
                crop_fraction=self.model_config.crop_fraction,
                use_relative_action=self.model_config.use_relative_action,
                transformers_loading_kwargs=self.transformers_loading_kwargs,
            )

        print(
            colored(
                f"These are all the processor configs for training: {json.dumps({k: str(v) for k, v in vars(processor).items()}, indent=2)}",
                "yellow",
            )
        )
        if get_rank() == 0:
            with open(self.save_cfg_dir / "final_processor_config.json", "w") as f:
                json.dump({k: str(v) for k, v in vars(processor).items()}, f, indent=2)

        self.processor = processor
        dataset_factory = DatasetFactory(config=self.config)
        train_dataset, eval_dataset = dataset_factory.build(processor=self.processor)

        # Save dataset statistics for inference
        stats = train_dataset.get_dataset_statistics()
        stats_dict = convert_tensors_to_lists(stats)
        # Save statistics
        with open(save_cfg_dir / "dataset_statistics.json", "w") as f:
            json.dump(stats_dict, f, indent=2)
        logging.info("Saved dataset statistics for inference")

        return train_dataset, eval_dataset

    def _create_collator(self):
        data_collator = self.processor.collator
        return data_collator


register_model(Gr00tN1d6Config, Gr00tN1d6Pipeline)

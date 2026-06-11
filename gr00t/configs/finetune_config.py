# Finetune config used for single node post-training.
from dataclasses import dataclass
from typing import Literal

from gr00t.data.embodiment_tags import EmbodimentTag


@dataclass
class FinetuneConfig:
    """
    Configuration for fine-tuning a Vision-Language-Action (VLA) model.

    This dataclass defines all parameters needed to launch a fine-tuning job
    on a pretrained base model using a custom dataset and embodiment-specific
    modality configuration. It controls model tuning options, data augmentation,
    and training hyperparameters.
    """

    # --- Data and Model Paths ---
    base_model_path: str
    """Path to the pretrained base model checkpoint (e.g., Hugging Face model hub or local directory)."""

    dataset_path: str
    """Path to the dataset root directory containing trajectory data for fine-tuning."""

    embodiment_tag: EmbodimentTag
    """Identifier specifying which embodiment (robot configuration) this fine-tuning run targets."""

    modality_config_path: str | None = None
    """
    Path to a Python file defining the modality configuration for the given embodiment. 
    If None, use the pre-registered modality config in `gr00t/configs/data/embodiment_configs.py`. 
    """

    # --- Model Tuning Flags ---
    tune_llm: bool = False
    """If True, fine-tune the language model (LLM) backbone during training."""

    tune_visual: bool = False
    """If True, fine-tune the visual encoder (e.g., ViT or CNN backbone)."""

    tune_projector: bool = True
    """If True, fine-tune the multimodal projector layers that map vision/language features to a shared space."""

    tune_diffusion_model: bool = True
    """If True, fine-tune the diffusion-based action decoder (if present in the model)."""

    state_dropout_prob: float = 0.0
    """
    Dropout probability applied to state inputs for regularization during training.
    """

    # --- Data Augmentation ---
    random_rotation_angle: int | None = None
    """Maximum rotation angle (in degrees) for random rotation augmentation of input images."""

    color_jitter_params: dict[str, float] | None = None
    """
    Parameters for color jitter augmentation on images.

    Expected keys include:
      - "brightness": float
      - "contrast": float
      - "saturation": float
      - "hue": float
    Example: {"brightness": 0.4, "contrast": 0.4, "saturation": 0.4, "hue": 0.1}

    If None, applying the default color jitter augmentation from the pretrained model.
    """
    extra_augmentation_config: str | None = None
    """
    JSON string for extra image augmentations (mask-based and others).

    Expected keys include:
      - "background_noise_transforms": list of dicts for noise on mask regions
          - "target_mask_values": list of int (e.g., [0])
          - "p": float (probability of applying)
      - "masked_region_transforms": list of dicts for color tint on mask regions
          - "target_mask_values": list of int (e.g., [4] or [5])
          - "p": float (probability of applying)
          - "alpha_range": [min, max] for random_tint intensity

    Example: {"background_noise_transforms": [{"target_mask_values": [0], "p": 0.9}],
              "masked_region_transforms": [{"target_mask_values": [4], "p": 1.0, "alpha_range": [0, 1]}]}

    If None, no extra augmentations are applied.
    """

    # --- Training Configuration ---
    global_batch_size: int = 64
    """Total effective batch size across all GPUs and accumulation steps."""

    dataloader_num_workers: int = 2
    """Number of parallel worker processes used for data loading."""

    learning_rate: float = 1e-4
    """Initial learning rate for optimizer."""

    gradient_accumulation_steps: int = 1
    """Number of forward passes to accumulate before performing a backward/update step."""

    output_dir: str = "./outputs"
    """Directory where model checkpoints, logs, and outputs are saved."""

    experiment_name: str | None = None
    """Optional experiment name used as the W&B run name. Defaults to the output directory basename."""

    wandb_project: str = "finetune-gr00t-n1d6"
    """W&B project name to log runs to."""

    save_steps: int = 1000
    """Frequency (in training steps) at which to save checkpoints."""

    save_total_limit: int = 5
    """Maximum number of checkpoints to keep before older ones are deleted."""

    num_gpus: int = 1
    """Number of GPUs available for distributed or single-node training."""

    use_wandb: bool = False
    """
    If True, log metrics and artifacts to Weights & Biases (wandb).
    The project is `finetune-gr00t-n1d6`.
    You need to login to wandb to view the logs.
    """

    max_steps: int = 10000
    """Total number of training steps to run before stopping."""

    weight_decay: float = 1e-5
    """Weight decay coefficient for optimizer (L2 regularization)."""

    warmup_ratio: float = 0.05
    """Proportion of total training steps used for learning rate warm-up."""

    shard_size: int = 2**10
    """Size of the shard to use for the dataset during preloading."""

    episode_sampling_rate: float = 0.1
    """Sampling rate for the episodes."""

    num_shards_per_epoch: int = int(1e5)
    """Number of shards to use for the dataset. reduce this number if vram is limited."""

    skip_weight_loading: bool = False
    """If True, skip loading model weights from base_model_path (architecture only).
    Useful for CI/testing to skip the slow checkpoint shard loading."""

    # --- HAMLET (History-Aware Memory with Learned Tokens) ---
    hamlet_mode: Literal["off", "tcl", "finetune"] = "finetune"
    """HAMLET training mode.
    - "off": vanilla GR00T N1.6 finetune (no HAMLET).
    - "tcl": Stage 1 — time-contrastive pretraining of moment tokens.
    - "finetune": Stage 2 — HAMLET end-to-end fine-tune (memory module + action head).
    """

    n_moment_tokens: int = 4
    """Number of learnable moment tokens (n_q) appended to the VLM input."""

    memory_window: int = 4
    """History window length T — number of past moment-token sets fed to the memory transformer."""

    memory_stride: int = 16
    """Stride (in env steps) between consecutive past snapshots in the HAMLET memory window.
    Must equal `n_action_steps` (the inference replanning interval) so the cache, which is
    updated once per policy call, naturally holds snapshots at [t-(K-1)S, ..., t-S, t]."""

    memory_num_layers: int = 2
    """Depth of the HAMLET memory transformer (paper default: 2)."""

    mem_cond_type: Literal["cross_attn", "adaln"] = "cross_attn"
    """How memory conditions the action head.
    - "cross_attn" (default): memory-aggregated moment tokens replace the backbone
      moment-token tail and enter the DiT as cross-attention KV.
    - "adaln": the pooled memory vector goes through a zero-init Linear and is added to
      the DiT timestep embedding; the moment-token tail is sliced off the KV."""

    memory_type: Literal["moment_token", "vision_feature"] = "moment_token"
    """What flows through the memory module (action-head VLM conditioning is unchanged).
    "moment_token": learnable moment tokens' post-LLM hidden states.
    "vision_feature": primary view (first modality_key) image tokens, post-LLM, avg-pooled
    to 64/step (no moment tokens added). Supports both mem_cond_type values."""

    load_moment_tokens_from: str | None = None
    """Stage-2 entry. Path to a Stage-1 (TCL) checkpoint or `model.safetensors`
    from which the moment-token parameter is loaded."""

    freeze_moment_tokens: bool = False
    """Stage 2 freezes moment tokens by default (matches GR00T frozen-VLM recipe)."""

    tcl_tau: float = 0.07
    """InfoNCE temperature for the TCL stage."""


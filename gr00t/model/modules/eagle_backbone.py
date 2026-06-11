import os

import torch
from transformers import AutoConfig, AutoModel
from transformers.feature_extraction_utils import BatchFeature


class EagleBackbone(torch.nn.Module):
    def __init__(
        self,
        model_name: str = "nvidia/Eagle-Block2A-2B-v2",
        tune_llm: bool = False,
        tune_visual: bool = False,
        select_layer: int = -1,
        reproject_vision: bool = True,
        use_flash_attention: bool = False,
        projector_dim: int = -1,
        load_bf16: bool = False,
        tune_top_llm_layers: int = 0,
        trainable_params_fp32: bool = False,
        transformers_loading_kwargs: dict = {},
        n_moment_tokens: int = 0,
        freeze_moment_tokens: bool = False,
        memory_type: str = "moment_token",
    ):
        """
        EagleBackbone is to generate n_queries to represent the future action hidden states.
        Args:
            model_name: nvidia/Eagle-Block2A-2B-v2
            tune_llm: whether to tune the LLM model (default: False)
            tune_visual: whether to tune the visual model (default: False)
            n_moment_tokens: HAMLET — if >0, append this many learnable moment tokens
                at the tail of the VLM input. They attend to image+text via causal
                self-attention and emerge in the VLM output as the m'_t slice.
        """

        super().__init__()

        # Add attention kwargs
        extra_kwargs = {}
        if use_flash_attention:
            extra_kwargs["attn_implementation"] = "flash_attention_2"
        if load_bf16:
            extra_kwargs["torch_dtype"] = torch.bfloat16

        if model_name == "nvidia/Eagle-Block2A-2B-v2":
            assert use_flash_attention, (
                "nvidia/Eagle-Block2A-2B-v2 requires flash attention by default"
            )
            assert load_bf16, "nvidia/Eagle-Block2A-2B-v2 requires bfloat16 by default"
            eagle_path = os.path.join(os.path.dirname(__file__), "nvidia", "Eagle-Block2A-2B-v2")
            config = AutoConfig.from_pretrained(eagle_path, trust_remote_code=True)
            self.model = AutoModel.from_config(config, trust_remote_code=True)
        else:
            raise ValueError(f"Model {model_name} not supported")

        # needed since we don't use these layers. Also saves compute
        while len(self.model.language_model.model.layers) > select_layer:
            self.model.language_model.model.layers.pop(-1)

        self.select_layer = select_layer
        self.memory_type = memory_type

        # HAMLET moment tokens, created here in __init__ so `from_pretrained` does not
        # leave them uninitialized. Stored on the backbone for use by the TCL (Stage-1)
        # and finetune (Stage-2) paths.
        self.n_moment_tokens = n_moment_tokens
        if n_moment_tokens > 0:
            hidden_size = self.model.config.text_config.hidden_size
            self.moment_tokens = torch.nn.Parameter(0.02 * torch.randn(n_moment_tokens, hidden_size))

        self.set_trainable_parameters(tune_llm, tune_visual, tune_top_llm_layers)
        if n_moment_tokens > 0 and freeze_moment_tokens:
            self.moment_tokens.requires_grad_(False)
        if load_bf16 and trainable_params_fp32:
            # cast trainable parameters to fp32
            for n, p in self.named_parameters():
                if p.requires_grad:
                    p.data = p.data.to(torch.float32)
                    print(f"Casting trainable parameter {n} to fp32")

    def set_trainable_parameters(self, tune_llm: bool, tune_visual: bool, tune_top_llm_layers: int):
        self.tune_llm = tune_llm
        self.tune_visual = tune_visual
        for p in self.parameters():
            p.requires_grad = True
        if not tune_llm:
            self.model.language_model.requires_grad_(False)
        if not tune_visual:
            self.model.vision_model.requires_grad_(False)
            self.model.mlp1.requires_grad_(False)

        if tune_top_llm_layers > 0:
            for layer in self.model.language_model.model.layers[-tune_top_llm_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True

        print(f"Tune backbone llm: {self.tune_llm}")
        print(f"Tune backbone visual: {self.tune_visual}")
        # Check if any parameters are still trainable. If not, print a warning.
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
            if self.model.language_model and not self.tune_llm:
                self.model.language_model.eval()
            if self.model.vision_model and not self.tune_visual:
                self.model.vision_model.eval()
                self.model.mlp1.eval()

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    def forward(self, vl_input: BatchFeature) -> BatchFeature:
        self.set_frozen_modules_to_eval_mode()
        # 0. Set frozen module to eval
        keys_to_use = ["input_ids", "attention_mask", "pixel_values"]
        vl_input = {k: vl_input[k] for k in keys_to_use if k in vl_input}

        if self.n_moment_tokens > 0:
            return self._forward_with_moment_tokens(vl_input)

        outputs = self.model(**vl_input, output_hidden_states=True)
        outputs = outputs["hidden_states"][-1]
        image_mask = vl_input["input_ids"] == self.model.config.image_token_index
        attention_mask = vl_input["attention_mask"] == 1
        data = {
            "backbone_features": outputs,
            "backbone_attention_mask": attention_mask,
            "image_mask": image_mask,
        }
        if self.memory_type == "vision_feature" and vl_input.get("pixel_values") is not None:
            # HAMLET vision_feature path: expose the PRIMARY (first) view's post-LLM image
            # tokens, avg-pooled to 64/step, for the memory module. Eagle packs each view as
            # a separate image -> n_views = pixel_values rows / batch; tokens/view inferred
            # from the (uniform) per-row image-token count (square grid).
            from gr00t.model.modules.memory import pool_primary_view

            B = vl_input["input_ids"].shape[0]
            n_views = max(1, vl_input["pixel_values"].shape[0] // B)
            total = int(image_mask[0].sum().item())
            tpv = total // n_views
            side = int(round(tpv**0.5))
            if side * side == tpv and tpv > 0:
                data["primary_view_feature"] = pool_primary_view(outputs, image_mask, tpv, (side, side))
        return BatchFeature(data=data)  # [B, T2, hidden_size]

    def _forward_with_moment_tokens(self, vl_input: dict) -> BatchFeature:
        """HAMLET forward for the Eagle backbone.

        Follows the Eagle3-VL text/image splicing, then appends `moment_tokens` at the
        tail of the spliced embeddings before calling `language_model` directly with
        `inputs_embeds`. The last `n_moment_tokens` rows of the LM hidden state are the
        moment-token outputs m'_t.
        """
        input_ids = vl_input["input_ids"]
        attention_mask = vl_input["attention_mask"]
        pixel_values = vl_input["pixel_values"]
        eagle = self.model  # Eagle3_VL

        # Embed text tokens.
        input_embeds = eagle.language_model.get_input_embeddings()(input_ids)

        # Extract visual features and splice them into the image placeholder slots,
        # matching Eagle3_VL.forward (modeling_eagle3_vl.py:231-255).
        vit_embeds = eagle.extract_feature(pixel_values)
        B, N, C = input_embeds.shape
        input_embeds_flat = input_embeds.reshape(B * N, C)
        input_ids_flat = input_ids.reshape(B * N)
        selected = input_ids_flat == eagle.image_token_index
        try:
            input_embeds_flat[selected] = input_embeds_flat[selected] * 0.0 + vit_embeds
        except Exception as e:
            print(
                f"warning: {e}, input_embeds_flat[selected].shape={input_embeds_flat[selected].shape}, "
                f"vit_embeds.shape={vit_embeds.shape}"
            )
            n_token = selected.sum()
            input_embeds_flat[selected] = input_embeds_flat[selected] * 0.0 + vit_embeds[:n_token]
        input_embeds = input_embeds_flat.reshape(B, N, C)

        # Append moment tokens at the tail.
        n_q = self.n_moment_tokens
        meta = self.moment_tokens.unsqueeze(0).expand(B, -1, -1).to(input_embeds.dtype)
        input_embeds = torch.cat([input_embeds, meta], dim=1)

        extra_mask = torch.ones(B, n_q, dtype=attention_mask.dtype, device=attention_mask.device)
        attention_mask_ext = torch.cat([attention_mask, extra_mask], dim=1)

        # Moment tokens sit after right-padding in sequence order, but their RoPE
        # positions continue from each sample's REAL length (= attention_mask.sum)
        # so train-time (padded batch) and inference-time (no padding) relative
        # distances to the content tokens match exactly. The prefix uses arange,
        # replicating HF's default when position_ids is None.
        seq_len = input_ids.shape[1]
        prefix_pos = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(B, -1)
        real_len = attention_mask.sum(dim=1, keepdim=True).to(torch.long)  # (B, 1)
        mq_pos = real_len + torch.arange(n_q, device=input_ids.device).unsqueeze(0)
        position_ids = torch.cat([prefix_pos, mq_pos], dim=1)

        outputs = eagle.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask_ext,
            position_ids=position_ids,
            output_hidden_states=True,
            use_cache=False,
        )
        last_hidden = outputs.hidden_states[-1]  # (B, N+n_q, d)

        image_mask = input_ids == eagle.image_token_index
        image_mask = torch.cat(
            [image_mask, torch.zeros(B, n_q, dtype=image_mask.dtype, device=image_mask.device)],
            dim=1,
        )
        backbone_attention_mask = attention_mask_ext == 1
        return BatchFeature(
            data={
                "backbone_features": last_hidden,
                "backbone_attention_mask": backbone_attention_mask,
                "image_mask": image_mask,
                "n_moment_tokens": n_q,
            }
        )

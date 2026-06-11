# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HAMLET time-contrastive learning (TCL) head.

Mean-pools the n_q moment tokens, projects them through a 2-layer SiLU MLP,
L2-normalizes, and trains with InfoNCE on (anchor, positive, hard-negative)
triplets. Trainable: backbone.moment_tokens + this head. Replaces
FlowmatchingActionHead when ``hamlet_mode == "tcl"``.
"""

from __future__ import annotations

import logging

import torch
from torch import nn
from torch.nn import functional as F
from transformers.feature_extraction_utils import BatchFeature


logger = logging.getLogger(__name__)


class Gr00tTCLHead(nn.Module):
    """Time-Contrastive Learning head for HAMLET Stage-1 pretraining."""

    supports_gradient_checkpointing = False

    def __init__(self, backbone_embedding_dim: int = 1536, tcl_tau: float = 0.07):
        super().__init__()
        d = backbone_embedding_dim
        # 2-layer Linear(d->d) + SiLU + Linear(d->d). Final L2-normalization is
        # applied at forward via F.normalize.
        self.moment_to_repr = nn.Sequential(
            nn.Linear(d, d),
            nn.SiLU(),
            nn.Linear(d, d),
        )
        for m in self.moment_to_repr.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.tcl_tau = tcl_tau

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    def set_trainable_parameters(self, *_args, **_kwargs):
        for p in self.parameters():
            p.requires_grad = True

    def set_frozen_modules_to_eval_mode(self):
        pass

    def _moment_repr(self, backbone_output: BatchFeature) -> torch.Tensor:
        """Mean-pool moment-token tail, project, L2-normalize -> (B, d)."""
        feats = backbone_output["backbone_features"]  # (B, T, d)
        n_q = int(backbone_output["n_moment_tokens"])
        moment = feats[:, -n_q:, :]
        pooled = moment.mean(dim=1)
        proj_dtype = next(self.moment_to_repr.parameters()).dtype
        z = self.moment_to_repr(pooled.to(proj_dtype))
        z = F.normalize(z, dim=-1, eps=1e-8)
        return z

    def forward(
        self,
        anchor_output: BatchFeature,
        aug_output: BatchFeature,
        neg_output: BatchFeature,
    ) -> dict:
        z_a = self._moment_repr(anchor_output)
        z_p = self._moment_repr(aug_output)
        z_n = self._moment_repr(neg_output)

        sim_ap = torch.sum(z_a * z_p, dim=-1, keepdim=True)  # (B, 1)
        sim_an = torch.sum(z_a * z_n, dim=-1, keepdim=True)  # (B, 1)
        logits = torch.cat([sim_ap, sim_an], dim=1) / self.tcl_tau  # (B, 2)
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        loss = F.cross_entropy(logits, labels)

        with torch.no_grad():
            tcl_pos_sim = sim_ap.mean()
            tcl_neg_sim = sim_an.mean()

        return BatchFeature(
            data={
                "loss": loss,
                "tcl_pos_sim": tcl_pos_sim.detach(),
                "tcl_neg_sim": tcl_neg_sim.detach(),
            }
        )

    @torch.no_grad()
    def get_action(self, *args, **kwargs):
        raise RuntimeError("TCL head does not support get_action; load a Stage-2 checkpoint.")

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HAMLET memory module.

Block-causal Transformer mixing T past sets of n_q moment tokens.

Input  : [bsz, T*n_q, d]  oldest-first across the T-block axis.
Output : [bsz, T*n_q, d]  (caller slices the last n_q rows as current-step memory)

Attention mask is block-bidirectional within each n_q-token block (one timestep)
and causal across blocks (later timesteps can't see future blocks).
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def pool_primary_view(
    backbone_features: torch.Tensor,
    image_mask: torch.Tensor,
    tokens_per_view: int,
    grid_hw: tuple[int, int],
    out_side: int = 8,
) -> torch.Tensor:
    """Slice the PRIMARY (first) view's image tokens out of post-LLM hidden states and
    spatially avg-pool them to a fixed `out_side x out_side` token grid (64 by default).

    Used by HAMLET `memory_type="vision_feature"`: the memory module consumes the primary
    view's vision features instead of learnable moment tokens. backbone_features is left
    UNMODIFIED (it still conditions the action head).
    """
    BK, _, d = backbone_features.shape
    img = backbone_features[image_mask].view(BK, -1, d)[:, :tokens_per_view, :]  # (BK, V0, d)
    gh, gw = grid_hw
    assert gh * gw == tokens_per_view, f"grid {gh}x{gw} != tokens_per_view {tokens_per_view}"
    grid = img.transpose(1, 2).reshape(BK, d, gh, gw)
    pooled = F.adaptive_avg_pool2d(grid.float(), (out_side, out_side)).to(img.dtype)
    return pooled.flatten(2).transpose(1, 2).contiguous()  # (BK, out_side**2, d)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    # q, k: (B, H, L, D)  ;  cos, sin: (1, 1, L, D)
    q_out = (q * cos) + (_rotate_half(q) * sin)
    k_out = (k * cos) + (_rotate_half(k) * sin)
    return q_out, k_out


class _RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_positions: int, theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_positions = max_positions

    def forward(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # positions: (L,) int
        freqs = torch.einsum("i,j->ij", positions.to(torch.float32), self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)  # (L, D)
        return emb.cos().unsqueeze(0).unsqueeze(0), emb.sin().unsqueeze(0).unsqueeze(0)


class _RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute in fp32 for stability, multiply weight, then cast back to input dtype.
        dtype = x.dtype
        x32 = x.float()
        rms = x32.pow(2).mean(-1, keepdim=True).clamp_min(self.eps).rsqrt()
        return ((x32 * rms) * self.weight).to(dtype)


class _Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        B, L, _ = x.shape
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        q, k = _apply_rope(q, k, cos, sin)
        # attn_mask is additive: 0 for allow, -inf for block. Shape (1, 1, L, L).
        out = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0)
        out = out.transpose(1, 2).contiguous().view(B, L, -1)
        return self.o_proj(out)


class _SwiGLU(nn.Module):
    def __init__(self, dim: int, intermediate: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, intermediate, bias=False)
        self.up_proj = nn.Linear(dim, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, dim, bias=False)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))


class _Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, ffn_mult: int = 4, rms_eps: float = 1e-5):
        super().__init__()
        self.attn_norm = _RMSNorm(dim, rms_eps)
        self.attn = _Attention(dim, num_heads)
        self.ffn_norm = _RMSNorm(dim, rms_eps)
        self.ffn = _SwiGLU(dim, ffn_mult * dim)

    def forward(self, x, attn_mask, cos, sin):
        x = x + self.attn(self.attn_norm(x), attn_mask, cos, sin)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class MemoryTransformer(nn.Module):
    """Block-causal Transformer for HAMLET history aggregation.

    Args:
        dim:        hidden dim (matches the action head's backbone embedding dim).
        n_q:        moment tokens per step (default 4).
        T:          history window length (default 4).
        num_layers: depth (default 2).
        num_heads:  attention heads (default 16).
        ffn_mult:   FFN expansion (default 4).
        rms_eps:    RMSNorm eps (default 1e-5).
        init_range: std for nn.Linear init (default 0.02).
    """

    def __init__(
        self,
        dim: int,
        n_q: int,
        T: int,
        num_layers: int = 2,
        num_heads: int = 16,
        ffn_mult: int = 4,
        rms_eps: float = 1e-5,
        init_range: float = 0.02,
    ):
        super().__init__()
        self.dim = dim
        self.n_q = n_q
        self.T = T
        self.num_layers = num_layers
        seq_len = T * n_q
        self.blocks = nn.ModuleList(
            [_Block(dim, num_heads, ffn_mult, rms_eps) for _ in range(num_layers)]
        )
        self.final_norm = _RMSNorm(dim, rms_eps)
        head_dim = dim // num_heads
        self.rope = _RotaryEmbedding(head_dim, max_positions=T)

        # Block-causal mask, cached for forward.
        # Position index per token = which timestep block it belongs to.
        positions = torch.arange(T, dtype=torch.long).repeat_interleave(n_q)  # (seq_len,)
        # Allow: positions[i] >= positions[j]  -> token i (later block) can attend to token j (earlier or same).
        # Block-bidirectional inside the same block (same position), causal across blocks.
        allow = positions.unsqueeze(0) <= positions.unsqueeze(1)  # (L, L), True = allow
        mask = torch.zeros(seq_len, seq_len, dtype=torch.float32)
        mask.masked_fill_(~allow, float("-inf"))
        self.register_buffer("attn_mask", mask.view(1, 1, seq_len, seq_len), persistent=False)
        self.register_buffer("positions", positions, persistent=False)

        self._init_range = init_range
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self._init_range)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: (B, T*n_q, d). Position 0 = oldest block, position -1 = current block.
        Returns:
            (B, T*n_q, d).
        """
        B, L, D = x.shape
        assert L == self.T * self.n_q, f"expected seq_len {self.T * self.n_q}, got {L}"
        assert D == self.dim, f"expected dim {self.dim}, got {D}"
        cos, sin = self.rope(self.positions)
        cos = cos.to(dtype=x.dtype)
        sin = sin.to(dtype=x.dtype)
        attn_mask = self.attn_mask.to(dtype=x.dtype)
        for block in self.blocks:
            x = block(x, attn_mask, cos, sin)
        return self.final_norm(x)

    def current_slice(self, x: torch.Tensor) -> torch.Tensor:
        """Helper: return the last n_q rows of the memory output (current step's tokens)."""
        return x[:, -self.n_q :, :]

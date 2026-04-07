"""Normalization layers for DiT point cloud model.

This module provides various normalization techniques including RMS normalization
and adaptive layer normalization with timestep conditioning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.embeddings import TimestepEmbedding, Timesteps


class MultiHeadRMSNorm(nn.Module):
    """Multi-head RMS normalization layer.

    Ref:
        https://github.com/lucidrains/mmdit/blob/main/mmdit/mmdit_pytorch.py

    Args:
        dim: Feature dimension.
        heads: Number of attention heads.
    """

    def __init__(self, dim: int, heads: int = 1):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(heads, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply multi-head RMS normalization."""
        orig = x.dtype
        x = F.normalize(x.float(), dim=-1, eps=1e-6)
        return (x * self.gamma * self.scale).to(orig)


class AdaptiveLayerNorm(nn.Module):
    """Adaptive layer normalization with timestep conditioning.

    When clamp_scale=True, uses tanh to soft-clamp the learned scale factor,
    keeping (1 + scale) in the range (0, 2). This prevents the Jacobian from
    compounding across layers during forward-mode AD (torch.func.jvp), which
    otherwise causes dudt explosion in MeanFlow distillation. The forward-pass
    output is virtually unchanged because learned scales are typically small,
    but the derivative through the time-conditioning path is bounded.
    """

    def __init__(
        self, dim: int, act_fn: nn.Module = nn.SiLU, num_channels: int = 256,
        clamp_scale: bool = False,
    ):
        """Initialize the adaptive layer normalization.

        Args:
            dim (int): Dimension of embeddings.
            act_fn (nn.Module): Activation function. Default: nn.SiLU.
            num_channels (int): Number of channels for timestep projection. Default: 256.
            clamp_scale (bool): Apply tanh to the scale factor to bound it in (-1, 1).
                Prevents Jacobian explosion in forward-mode AD. Default: False.
        """
        super().__init__()
        self.timestep_proj = Timesteps(
            num_channels=num_channels, flip_sin_to_cos=True, downscale_freq_shift=0
        )
        self.timestep_embedder = TimestepEmbedding(
            in_channels=num_channels, time_embed_dim=dim
        )

        self.activation = act_fn()
        self.linear = nn.Linear(dim, dim * 2)                       # for scale and shift
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.clamp_scale = clamp_scale

    def forward(self, x: torch.Tensor, timestep: torch.Tensor, extra_emb: torch.Tensor = None) -> torch.Tensor:
        """Apply adaptive layer normalization.

        Args:
            x (B, N, dim): Input tensor.
            timestep (B,): Timestep tensor.
            extra_emb (B, dim): Optional extra embedding (e.g., from MeanFlow timestep2).
                When provided, added to the timestep embedding before activation.

        Returns:
            (B, N, dim): Normalized tensor.
        """
        emb = self.timestep_embedder(self.timestep_proj(timestep))    # (B, dim)
        if extra_emb is not None:
            emb = emb + extra_emb                                     # (B, dim)
        emb = self.linear(self.activation(emb))                       # (B, dim * 2)
        scale, shift = emb.unsqueeze(1).chunk(2, dim=-1)              # (B, 1, dim) for both
        if self.clamp_scale:
            scale = scale.tanh()
        return self.norm(x) * (1 + scale) + shift

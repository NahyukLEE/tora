"""Diffusion Transformer layer for Rectified Point Flow."""

import math

import flash_attn
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.attention import FeedForward

from .norm import AdaptiveLayerNorm, MultiHeadRMSNorm

# To resolve a conflict with flash-attn's higher-order graph and DDP.
# DDP will be working normally, just skipped by torch._dynamo.
torch._dynamo.config.optimize_ddp=False


def scaled_dot_product_attention_vanilla(query, key, value, attn_mask=None, dropout_p=0.0, scale=None):
    """Vanilla scaled dot-product attention, compatible with torch.func.jvp.

    Uses only standard PyTorch ops (no Flash Attention CUDA kernels).
    Adapted from T2I-Distill/model/attn_processor.py.

    Args:
        query: (*, L, head_dim)
        key: (*, S, head_dim)
        value: (*, S, head_dim)
        attn_mask: Optional broadcastable mask.
        dropout_p: Dropout probability (unused during no_grad JVP).
        scale: Optional scale factor.

    Returns:
        (*, L, head_dim)
    """
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_weight = attn_weight.masked_fill(~attn_mask, float("-inf"))
        else:
            attn_weight = attn_weight + attn_mask
    attn_weight = torch.softmax(attn_weight, dim=-1)
    if dropout_p > 0.0:
        attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value


class DiTLayer(nn.Module):
    """Diffusion Transformer layer for Rectified Point Flow.

    This layer includes:
        1. Part-wise attention, independent for points in each part.
        2. Global attention, across all parts.
        3. Feed-forward network.

    Ref:
        Some codes are adapted from GARF https://github.com/ai4ce/GARF
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout: float = 0,
        softcap: float = 0,
        activation_fn: str = "geglu",
        qkv_proj_bias: bool = False,
        qk_norm: bool = True,
        attn_dtype: torch.dtype = torch.bfloat16,
        use_vanilla_attn: bool = False,
        clamp_scale: bool = False,
    ):
        """Initialize the DiT layer.

        Args:
            dim (int): Feature dimension.
            num_attention_heads (int): Number of attention heads.
            attention_head_dim (int): Dimension of each attention head.
            dropout (float): Dropout probability. Default: 0.0.
            softcap (float): Soft cap for attention scores. Default: 0.0.
            activation_fn (str): Activation function for feed-forward. Default: "geglu".
            qkv_proj_bias (bool): Whether to use bias in QKV projections. Default: False.
            qk_norm (bool): Whether to use query-key normalization. Default: True.
            attn_dtype (torch.dtype): Data type for attention. Default: torch.float16.
            use_vanilla_attn (bool): Use vanilla attention instead of Flash Attention.
                Required for torch.func.jvp compatibility in MeanFlow distillation.
                Default: False.
            clamp_scale (bool): Apply tanh to AdaLN scale factors to bound the
                Jacobian during forward-mode AD. Default: False.
        """
        super().__init__()

        assert dim == attention_head_dim * num_attention_heads, \
            "dim must be equal to attention_head_dim * num_attention_heads"

        self.dim = dim
        self.num_heads = num_attention_heads
        self.head_dim = attention_head_dim
        self.softcap = softcap
        self.qk_norm = qk_norm
        self.attn_dtype = attn_dtype
        self.use_vanilla_attn = use_vanilla_attn

        # Part-wise Attention
        self.self_prenorm = AdaptiveLayerNorm(dim, clamp_scale=clamp_scale)
        self.self_qkv_proj = nn.Linear(dim, dim * 3, bias=qkv_proj_bias)
        self.self_out_proj = nn.Linear(dim, dim)
        if qk_norm:
            self.self_q_norm = MultiHeadRMSNorm(self.head_dim, self.num_heads)
            self.self_k_norm = MultiHeadRMSNorm(self.head_dim, self.num_heads)

        # Global Attention
        self.global_prenorm = AdaptiveLayerNorm(dim, clamp_scale=clamp_scale)
        self.global_qkv_proj = nn.Linear(dim, dim * 3, bias=qkv_proj_bias)
        self.global_out_proj = nn.Linear(dim, dim)
        if qk_norm:
            self.global_q_norm = MultiHeadRMSNorm(self.head_dim, self.num_heads)
            self.global_k_norm = MultiHeadRMSNorm(self.head_dim, self.num_heads)

        # Feed-forward
        self.ff_norm = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn)

    @staticmethod
    def _qk_norm(qkv: torch.Tensor, q_norm: nn.Module, k_norm: nn.Module) -> torch.Tensor:
        """Apply query-key normalization and keep the dtype."""
        q, k, v = qkv.unbind(dim=1)
        q, k = q_norm(q).to(v.dtype), k_norm(k).to(v.dtype)
        return torch.stack([q, k, v], dim=1)

    def _part_attention(self, x: torch.Tensor, cu_seqlens: torch.Tensor, max_seqlen: int) -> torch.Tensor:
        """Apply part-wise attention using Flash Attention."""
        B, N, _ = x.shape
        qkv = self.self_qkv_proj(x)                                    # (B, N, 3 * embed_dim)
        qkv = qkv.reshape(-1, 3, self.num_heads, self.head_dim)        # (B*N, 3, num_heads, head_dim)

        if self.qk_norm:
            qkv = self._qk_norm(qkv, self.self_q_norm, self.self_k_norm)

        out = flash_attn.flash_attn_varlen_qkvpacked_func(
            qkv=qkv.to(self.attn_dtype),
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            softcap=self.softcap,
        ).to(x.dtype)
        out = out.view(B, N, self.dim)                                  # (B, N, embed_dim)
        return self.self_out_proj(out)

    def _vanilla_part_attention(self, x: torch.Tensor, cu_seqlens: torch.Tensor, max_seqlen: int) -> torch.Tensor:
        """Apply part-wise attention using vanilla ops (JVP-compatible).

        Loops over each part segment defined by cu_seqlens and applies
        standard scaled dot-product attention per part.
        """
        B, N, _ = x.shape
        qkv = self.self_qkv_proj(x)                                    # (B, N, 3 * embed_dim)
        qkv = qkv.reshape(-1, 3, self.num_heads, self.head_dim)        # (B*N, 3, num_heads, head_dim)

        if self.qk_norm:
            qkv = self._qk_norm(qkv, self.self_q_norm, self.self_k_norm)

        # Process each part segment via loop
        out = torch.zeros(B * N, self.num_heads, self.head_dim, device=x.device, dtype=x.dtype)
        num_segments = cu_seqlens.shape[0] - 1
        for i in range(num_segments):
            s = cu_seqlens[i]
            e = cu_seqlens[i + 1]
            part_qkv = qkv[s:e]                                        # (part_len, 3, heads, head_dim)
            q = part_qkv[:, 0].transpose(0, 1)                         # (heads, part_len, head_dim)
            k = part_qkv[:, 1].transpose(0, 1)
            v = part_qkv[:, 2].transpose(0, 1)
            part_out = scaled_dot_product_attention_vanilla(
                q.to(self.attn_dtype), k.to(self.attn_dtype), v.to(self.attn_dtype)
            ).to(x.dtype)                                              # (heads, part_len, head_dim)
            out[s:e] = part_out.transpose(0, 1)

        out = out.view(B, N, self.dim)                                  # (B, N, embed_dim)
        return self.self_out_proj(out)

    def _global_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Apply global attention using Flash Attention."""
        B, N, _ = x.shape
        qkv = self.global_qkv_proj(x)                                   # (B, N, 3 * embed_dim)
        qkv = qkv.reshape(-1, 3, self.num_heads, self.head_dim)         # (B*N, 3, num_heads, head_dim)

        if self.qk_norm:
            qkv = self._qk_norm(qkv, self.global_q_norm, self.global_k_norm)

        qkv = qkv.view(B, N, 3, self.num_heads, self.head_dim)
        out = flash_attn.flash_attn_qkvpacked_func(
            qkv=qkv.to(self.attn_dtype),
            softcap=self.softcap,
        ).to(x.dtype)
        out = out.view(B, N, self.dim)                                   # (B, N, embed_dim)
        return self.global_out_proj(out)

    def _vanilla_global_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Apply global attention using vanilla ops (JVP-compatible).

        Standard batched scaled dot-product attention over all points.
        """
        B, N, _ = x.shape
        qkv = self.global_qkv_proj(x)                                   # (B, N, 3 * embed_dim)
        qkv = qkv.reshape(-1, 3, self.num_heads, self.head_dim)         # (B*N, 3, num_heads, head_dim)

        if self.qk_norm:
            qkv = self._qk_norm(qkv, self.global_q_norm, self.global_k_norm)

        qkv = qkv.view(B, N, 3, self.num_heads, self.head_dim)
        q = qkv[:, :, 0].transpose(1, 2)                                # (B, heads, N, head_dim)
        k = qkv[:, :, 1].transpose(1, 2)
        v = qkv[:, :, 2].transpose(1, 2)

        out = scaled_dot_product_attention_vanilla(
            q.to(self.attn_dtype), k.to(self.attn_dtype), v.to(self.attn_dtype)
        ).to(x.dtype)                                                    # (B, heads, N, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, N, self.dim)      # (B, N, embed_dim)
        return self.global_out_proj(out)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        part_cu_seqlens: torch.Tensor,
        max_seqlen: int,
        extra_emb: torch.Tensor = None,
    ) -> torch.Tensor:
        """Forward pass through the DiT layer.

        Args:
            hidden_states (B, N, dim): Input tensor.
            timestep (B, ): Timestep values.
            part_cu_seqlens (N_valid_parts + 1, ): Cumulative lengths for each part.
            max_seqlen (int): Maximum sequence length for part-wise attention.
            extra_emb (B, dim): Optional extra embedding for MeanFlow timestep2.

        Returns:
            hidden_states (B, N, dim): Output tensor.
        """
        # 1. Part-wise Attention
        x = self.self_prenorm(hidden_states, timestep, extra_emb)
        if self.use_vanilla_attn:
            part_attn = self._vanilla_part_attention(x, part_cu_seqlens, max_seqlen)
        else:
            part_attn = self._part_attention(x, part_cu_seqlens, max_seqlen)
        hidden_states = hidden_states + part_attn

        # 2. Global Attention
        x = self.global_prenorm(hidden_states, timestep, extra_emb)
        if self.use_vanilla_attn:
            global_attn = self._vanilla_global_attention(x)
        else:
            global_attn = self._global_attention(x)
        hidden_states = hidden_states + global_attn

        # 3. Feed-forward
        x = self.ff_norm(hidden_states)
        hidden_states = hidden_states + self.ff(x)

        return hidden_states

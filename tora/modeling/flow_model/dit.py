"""Multi-part Point Cloud Diffusion Transformer (DiT) model."""

import torch
import torch.nn as nn
from diffusers.models.embeddings import TimestepEmbedding, Timesteps

from .embedding import PointCloudEncodingManager
from .layer import DiTLayer


class PointCloudDiT(nn.Module):
    """A transformer-based diffusion model for multi-part point cloud data.

    Ref:
        DiT: https://github.com/facebookresearch/DiT
        mmdit: https://github.com/lucidrains/mmdit/tree/main/mmdit
        GARF: https://github.com/ai4ce/GARF
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        dropout_rate: float = 0.0,
        softcap: float = 0.0,
        qk_norm: bool = True,
        attn_dtype: str = "float16",
        final_mlp_act: nn.Module = nn.SiLU,
        repa_layer: int = 2,
        use_vanilla_attn: bool = False,
        compile_layers: bool = True,
        use_dual_timestep: bool = False,
        clamp_scale: bool = False,
    ):
        """
        Args:
            in_dim: Input dimension of the point features (e.g., 64).
            out_dim: Output dimension (e.g., 3 for velocity field).
            embed_dim: Hidden dimension of the transformer layers (e.g., 512).
            num_layers: Number of transformer layers (e.g., 6).
            num_heads: Number of attention heads (e.g., 8).
            dropout_rate: Dropout rate, default 0.0.
            softcap: Soft cap for attention scores, default 0.0.
            qk_norm: Whether to use query-key normalization, default True.
            attn_dtype: Attention data type, default float16.
            final_mlp_act: Activation function for the final MLP, default SiLU.
            repa_layer: Layer index to extract intermediate representation.
            use_vanilla_attn: Use vanilla attention in all layers. Required for
                MeanFlow student model (torch.func.jvp compatibility). Default: False.
            compile_layers: Whether to torch.compile layer forward methods.
                Set to False for MeanFlow student (JVP compatibility). Default: True.
            use_dual_timestep: Create second timestep embedding for MeanFlow
                distillation (timestep2 = t - r). Default: False.
            clamp_scale: Apply tanh to AdaLN scale factors to bound the Jacobian
                during forward-mode AD. Only needed for the student in distillation.
                Default: False.
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.final_mlp_act = final_mlp_act
        self.repa_layer = repa_layer        # Layer index to extract intermediate representation
        # dual_timestep models are distillation students that require torch.func.jvp
        # for MeanFlow's JVP-based loss. torch.compile is incompatible with
        # torch.func's forward-mode AD tracing, so we force compilation off.


        # Parse attn_dtype
        if attn_dtype == "float16" or attn_dtype == "fp16":
            self.attn_dtype = torch.float16
        elif attn_dtype == "bfloat16" or attn_dtype == "bf16":
            self.attn_dtype = torch.bfloat16
        elif attn_dtype == "float32" or attn_dtype == "fp32":
            self.attn_dtype = torch.float32
        else:
            raise ValueError(f"Unsupported attn_dtype: {attn_dtype}")
        print(f"[PointCloudDiT] Using attn_dtype: {self.attn_dtype}")

        # Reference part embedding for distinguishing anchor vs. moving parts
        self.anchor_part_emb = nn.Embedding(2, self.embed_dim)

        # Point cloud encoding manager
        self.encoding_manager = PointCloudEncodingManager(
            in_dim=self.in_dim,
            embed_dim=self.embed_dim,
            multires=10
        )

        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            DiTLayer(
                dim=self.embed_dim,
                num_attention_heads=self.num_heads,
                attention_head_dim=self.embed_dim // self.num_heads,
                dropout=self.dropout_rate,
                softcap=softcap,
                qk_norm=qk_norm,
                attn_dtype=self.attn_dtype,
                use_vanilla_attn=use_vanilla_attn,
                clamp_scale=clamp_scale,
            )
            for _ in range(self.num_layers)
        ])

        # Optionally torch.compile layer forward methods
        if compile_layers:
            for layer in self.transformer_layers:
                layer.forward = torch.compile(layer.forward)

        # MLP for final predictions
        self.final_mlp = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            self.final_mlp_act(),
            nn.Linear(self.embed_dim, self.embed_dim // 2),
            self.final_mlp_act(),
            nn.Linear(self.embed_dim // 2, out_dim, bias=False)  # No bias for 3D coordinates
        )

        # MeanFlow: second timestep (t-r) embedding, only for distillation student
        self.use_dual_timestep = use_dual_timestep
        if use_dual_timestep:
            self.timestep_proj2 = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
            self.timestep_embedder2 = TimestepEmbedding(in_channels=256, time_embed_dim=self.embed_dim)

    def _add_anchor_embedding(
        self,
        x: torch.Tensor,
        anchor_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Add anchor part embeddings to distinguish anchor from moving parts.

        Args:
            x (B, N, dim): Input point cloud features.
            anchor_indices (B, N): bool tensor, True => anchor parts.

        Returns:
            (B, N, dim) Point cloud features with anchor part information added.
        """
        # anchor_part_emb.weight[0] for non-anchor part
        # anchor_part_emb.weight[1] for anchor part
        B, N = anchor_indices.shape
        anchor_part_emb = self.anchor_part_emb.weight[0].repeat(B, N, 1)
        anchor_part_emb[anchor_indices] = self.anchor_part_emb.weight[1]
        x = x + anchor_part_emb
        return x

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        latent: dict,
        scales: torch.Tensor,
        anchor_indices: torch.Tensor,
        timestep2: torch.Tensor = None,
    ) -> dict:
        """Forward pass through the PointCloudDiT model.

        Args:
            x (B, N, 3): Noise point coordinates at timestep t.
            timesteps (B, ): Timestep values.
            latent: PointTransformer's Point instance of conditional point cloud:
                - "coord" (n_points, 3): Point coordinates
                - "normal" (n_points, 3): Point normals
                - "feat" (n_points, in_dim): Point features
                - "batch" (n_points, ): Integer tensor of batch indices.
            scales (B, ): Scale factor for the point cloud.
            anchor_indices (B, N): bool tensor, True => anchor parts.
            timestep2 (B, ): Optional second timestep (t - r) for MeanFlow distillation.

        Returns:
            Tuple of (out, interm_repr):
                out (B, N, out_dim): Predicted velocity field.
                interm_repr (B, N, dim) or None: Intermediate representation at repa_layer.
        """

        # Encoding
        x = self.encoding_manager(x, latent, scales)                     # (B, N, dim)
        x = self._add_anchor_embedding(x, anchor_indices)                # (B, N, dim)

        # Prepare attention metadata
        part_seqlen = torch.bincount(latent["batch"])                    # (n_valid_parts, )
        max_seqlen = part_seqlen.max().item()                            # .item() is used to allow torch.compile
        part_cu_seqlens = nn.functional.pad(torch.cumsum(part_seqlen, 0), (1, 0))
        part_cu_seqlens = part_cu_seqlens.to(torch.int32)                # (n_valid_parts + 1, )

        # Compute extra embedding for (t-r) if MeanFlow dual timestep is enabled
        extra_emb = None
        if self.use_dual_timestep and timestep2 is not None:
            extra_emb = self.timestep_embedder2(self.timestep_proj2(timestep2))  # (B, embed_dim)

        # Transformer layers
        interm_repr = None
        for i, layer in enumerate(self.transformer_layers):
            x = layer(x, timesteps, part_cu_seqlens, max_seqlen, extra_emb)  # (B, N, dim)

            if (i+1) == self.repa_layer:
                interm_repr = x                                          # get intermediate representation

        # Final MLP, use float32 for better numerical stability
        with torch.amp.autocast(x.device.type, enabled=False):
            out = self.final_mlp(x.float())                              # (B, N, out_dim)

        return out, interm_repr


if __name__ == "__main__":
    model = PointCloudDiT(
        in_dim=64,
        out_dim=6,
        embed_dim=512,
        num_layers=6,
        num_heads=8,
        dropout_rate=0.0,
    )
    print(f"PointCloudDiT with {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")

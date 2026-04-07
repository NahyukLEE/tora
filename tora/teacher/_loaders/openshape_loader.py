"""
OpenShape model loader for Rectified-Point-Flow

Loads OpenShape (NeurIPS 2023) PointBERT backbone for point cloud feature
extraction as a REPA teacher.

OpenShape architecture (PointPatchTransformer):
  - PointNetSetAbstraction: FPS + ball-query grouping → patch features
  - Lift: Conv1d projection to transformer dim
  - CLS token + Transformer encoder (12 layers, scaling 4)
  - Linear projection to CLIP space (1280-dim for ViT-bigG-14)

The original model returns only a global CLS token. For REPA we extract the
512 patch tokens after the transformer and propagate them to all points via
k-NN interpolation from FPS centroids.

For REPA alignment, we support:
  - 1280-dim CLIP-projected patch features (language-aligned, default)
  - 512-dim raw transformer patch features

Dependencies: pointnet2_ops, torch
"""

import os
import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

logger = logging.getLogger(__name__)

# Feature configurations for known OpenShape PointBERT variants
OPENSHAPE_CONFIGS = {
    "openshape_pointbert_vitg14_rgb": {
        "dim": 512,
        "depth": 12,
        "heads": 8,
        "dim_head": 64,
        "mlp_dim": 1536,
        "sa_dim": 256,
        "patches": 512,
        "prad": 0.2,
        "nsamp": 64,
        "in_channel": 6,       # xyz + rgb
        "out_channel": 1280,   # ViT-bigG-14 CLIP dim
        "hf_repo": "OpenShape/openshape-pointbert-vitg14-rgb",
        "hf_file": "model.pt",
    },
}


# ============================================================================
# Architecture Components (adapted from OpenShape ppat.py + pointnet_util.py)
# ============================================================================

class PointNetSetAbstraction(nn.Module):
    """PointNet++ Set Abstraction: FPS + ball query + local MLP."""

    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz, points):
        """
        Args:
            xyz: (B, 3, N) point coordinates
            points: (B, C, N) point features
        Returns:
            new_xyz: (B, 3, S) sampled centroids
            new_points: (B, D, S) per-centroid features
        """
        from pointnet2_ops import pointnet2_utils

        xyz_t = xyz.permute(0, 2, 1).contiguous()  # (B, N, 3)
        points_t = points.permute(0, 2, 1).contiguous() if points is not None else None

        B, N, _ = xyz_t.shape

        if self.group_all:
            new_xyz = torch.zeros(B, 1, 3, device=xyz.device)
            grouped_xyz = xyz_t.view(B, 1, N, 3) - new_xyz.view(B, 1, 1, 3)
            if points_t is not None:
                new_points = torch.cat([grouped_xyz, points_t.view(B, 1, N, -1)], dim=-1)
            else:
                new_points = grouped_xyz
        else:
            # FPS
            fps_idx = pointnet2_utils.furthest_point_sample(xyz_t, self.npoint)  # (B, S)
            new_xyz = pointnet2_utils.gather_operation(
                xyz.contiguous(), fps_idx
            )  # (B, 3, S)
            new_xyz_t = new_xyz.permute(0, 2, 1).contiguous()  # (B, S, 3)

            # Ball query
            idx = pointnet2_utils.ball_query(
                self.radius, self.nsample, xyz_t, new_xyz_t
            )  # (B, S, nsample)

            # Gather grouped points
            grouped_xyz = pointnet2_utils.grouping_operation(
                xyz.contiguous(), idx
            )  # (B, 3, S, nsample)
            grouped_xyz -= new_xyz.unsqueeze(-1)  # relative position

            if points_t is not None:
                grouped_points = pointnet2_utils.grouping_operation(
                    points.contiguous(), idx
                )  # (B, C, S, nsample)
                new_points = torch.cat([grouped_xyz, grouped_points], dim=1)
            else:
                new_points = grouped_xyz

        # MLP
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, dim=-1)[0]  # (B, D, S)

        if self.group_all:
            new_xyz = new_xyz.permute(0, 2, 1)  # (B, 3, 1)

        return new_xyz, new_points


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, **kwargs):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, rel_pe=False):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )
        self.rel_pe = rel_pe
        if rel_pe:
            self.to_rel_pe = nn.Conv2d(3, heads, 1)
        else:
            self.to_rel_pe = None

    def forward(self, x, centroid_delta=None):
        B, N, _ = x.shape
        h, d = self.heads, self.dim_head

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(B, N, h, d).permute(0, 2, 1, 3) for t in qkv]

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        if centroid_delta is not None and self.to_rel_pe is not None:
            # centroid_delta: (B, 3, N, N)
            dots = dots + self.to_rel_pe(centroid_delta)

        attn = dots.softmax(dim=-1)
        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).reshape(B, N, h * d)
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0, rel_pe=False):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head,
                                       dropout=dropout, rel_pe=rel_pe)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
            ]))

    def forward(self, x, centroid_delta):
        for attn, ff in self.layers:
            x = attn(x, centroid_delta=centroid_delta) + x
            x = ff(x) + x
        return x


class PointPatchTransformer(nn.Module):
    """OpenShape PointBERT backbone.

    Original forward() returns only the CLS token (global feature).
    We add forward_patches() to extract per-patch features for REPA.
    """

    def __init__(self, dim, depth, heads, mlp_dim, sa_dim, patches, prad,
                 nsamp, in_dim=3, dim_head=64, rel_pe=False, patch_dropout=0):
        super().__init__()
        self.patches = patches
        self.patch_dropout = patch_dropout
        self.sa = PointNetSetAbstraction(
            npoint=patches, radius=prad, nsample=nsamp,
            in_channel=in_dim + 3, mlp=[64, 64, sa_dim], group_all=False,
        )
        self.lift = nn.Sequential(
            nn.Conv1d(sa_dim + 3, dim, 1),
            Permute021(),
            nn.LayerNorm(dim),
        )
        self.cls_token = nn.Parameter(torch.randn(dim))
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim,
                                        0.0, rel_pe)

    def forward(self, xyz, features):
        """Original forward: returns global CLS token only.

        Args:
            xyz: (B, 3, N) point coordinates
            features: (B, C, N) point features
        Returns:
            cls_feat: (B, dim) global feature
        """
        self.sa.npoint = self.patches
        if self.training:
            self.sa.npoint -= self.patch_dropout
        centroids, feature = self.sa(xyz, features)

        x = self.lift(torch.cat([centroids, feature], dim=1))  # (B, P, dim)

        B = x.shape[0]
        cls_tokens = self.cls_token.unsqueeze(0).unsqueeze(0).expand(B, 1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, P+1, dim)

        centroids_pad = torch.cat([
            centroids.new_zeros(B, 3, 1), centroids
        ], dim=-1)  # (B, 3, P+1)
        centroid_delta = centroids_pad.unsqueeze(-1) - centroids_pad.unsqueeze(-2)

        x = self.transformer(x, centroid_delta)
        return x[:, 0]  # CLS token

    def forward_patches(self, xyz, features):
        """Extract per-patch features (for REPA).

        Args:
            xyz: (B, 3, N) point coordinates
            features: (B, C, N) point features
        Returns:
            patch_features: (B, P, dim) per-patch transformer features
            centroids: (B, 3, P) patch center coordinates
        """
        self.sa.npoint = self.patches
        centroids, feature = self.sa(xyz, features)

        x = self.lift(torch.cat([centroids, feature], dim=1))  # (B, P, dim)

        B = x.shape[0]
        cls_tokens = self.cls_token.unsqueeze(0).unsqueeze(0).expand(B, 1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, P+1, dim)

        centroids_pad = torch.cat([
            centroids.new_zeros(B, 3, 1), centroids
        ], dim=-1)  # (B, 3, P+1)
        centroid_delta = centroids_pad.unsqueeze(-1) - centroids_pad.unsqueeze(-2)

        x = self.transformer(x, centroid_delta)

        patch_features = x[:, 1:]  # (B, P, dim) — exclude CLS token
        return patch_features, centroids


class Permute021(nn.Module):
    """Permute (B, C, N) -> (B, N, C)."""
    def forward(self, x):
        return x.permute(0, 2, 1)


class OpenShapeEncoder(nn.Module):
    """OpenShape = PointPatchTransformer + optional CLIP projection."""

    def __init__(self, ppat, proj):
        super().__init__()
        self.ppat = ppat
        self.proj = proj

    def forward(self, xyz, features, device=None, quantization_size=0.05):
        """Original interface: returns CLIP-projected global feature."""
        return self.proj(self.ppat(
            xyz.transpose(-1, -2).contiguous(),
            features.transpose(-1, -2).contiguous(),
        ))

    def forward_patches(self, xyz, features):
        """Extract per-patch features for REPA.

        Args:
            xyz: (B, N, 3) point coordinates
            features: (B, N, C) point features
        Returns:
            patch_features: (B, P, dim) raw patch features
            patch_features_proj: (B, P, out_channel) CLIP-projected (or None)
            centroids: (B, 3, P) patch center coordinates
        """
        patch_feat, centroids = self.ppat.forward_patches(
            xyz.transpose(-1, -2).contiguous(),
            features.transpose(-1, -2).contiguous(),
        )
        # patch_feat: (B, P, dim)
        # Optionally project to CLIP space
        if self.proj is not None:
            patch_feat_proj = self.proj(patch_feat)  # (B, P, out_channel)
        else:
            patch_feat_proj = None
        return patch_feat, patch_feat_proj, centroids


# ============================================================================
# Feature Propagation
# ============================================================================

def propagate_features_from_centroids(xyz_full, centroids, patch_features, k=3):
    """Propagate patch features to all points via k-NN interpolation.

    Args:
        xyz_full: (B, N, 3) full-resolution point coordinates
        centroids: (B, P, 3) patch center coordinates
        patch_features: (B, P, C) patch features
        k: int, number of nearest neighbors

    Returns:
        point_features: (B, N, C) interpolated features
    """
    dist = torch.cdist(xyz_full, centroids)  # (B, N, P)

    dist_k, idx_k = torch.topk(dist, k, dim=-1, largest=False)  # (B, N, k)

    dist_k = dist_k.clamp(min=1e-6)
    weights = 1.0 / dist_k
    weights = weights.clamp(max=1e6)
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)

    B, N, _ = xyz_full.shape
    C = patch_features.shape[-1]
    idx_k_flat = idx_k.view(B, -1)  # (B, N*k)
    gathered = torch.gather(
        patch_features, dim=1,
        index=idx_k_flat.unsqueeze(-1).expand(-1, -1, C),
    ).view(B, N, k, C)

    point_features = (gathered * weights.unsqueeze(-1)).sum(dim=2)  # (B, N, C)
    return point_features


# ============================================================================
# Model Loading
# ============================================================================

def load_openshape(model_size="openshape_pointbert_vitg14_rgb", ckpt_path=None,
                   device="cuda", auto_download=True):
    """Load OpenShape PointBERT model for feature extraction.

    Args:
        model_size: config key
        ckpt_path: optional path to a local checkpoint
        device: device to load model on
        auto_download: if True, download from HuggingFace

    Returns:
        model: OpenShapeEncoder (frozen, eval mode)
        feat_dim: int, backbone dim (768) — CLIP proj dim available via config
    """
    config = OPENSHAPE_CONFIGS.get(model_size, OPENSHAPE_CONFIGS["openshape_pointbert_vitg14_rgb"])

    ppat = PointPatchTransformer(
        dim=config["dim"],
        depth=config["depth"],
        heads=config["heads"],
        mlp_dim=config["mlp_dim"],
        sa_dim=config["sa_dim"],
        patches=config["patches"],
        prad=config["prad"],
        nsamp=config["nsamp"],
        in_dim=config["in_channel"],
        dim_head=config["dim_head"],
    )
    proj = nn.Linear(config["dim"], config["out_channel"])
    model = OpenShapeEncoder(ppat, proj)

    # Find or download checkpoint
    if ckpt_path and os.path.exists(ckpt_path):
        pass
    elif auto_download:
        ckpt_path = _download_openshape_checkpoint(config)

    if ckpt_path and os.path.exists(ckpt_path):
        logger.info(f"Loading OpenShape checkpoint from: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        # Strip 'module.' prefix (DDP) if present
        new_state_dict = {}
        for k, v in state_dict.items():
            new_state_dict[k.removeprefix("module.")] = v

        missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
        if missing:
            logger.warning(f"Missing keys ({len(missing)}): {missing[:5]}...")
        if unexpected:
            logger.warning(f"Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")
        logger.info("OpenShape checkpoint loaded successfully")
    else:
        logger.warning("No checkpoint found, using random initialization")

    model = model.to(device)
    model.eval()

    return model, config["out_channel"]


def _download_openshape_checkpoint(config, save_dir="pretrained"):
    """Download OpenShape checkpoint from HuggingFace."""
    os.makedirs(save_dir, exist_ok=True)
    local_path = os.path.join(save_dir, f"openshape_{config['hf_file']}")

    if os.path.exists(local_path):
        logger.info(f"Checkpoint already exists: {local_path}")
        return local_path

    try:
        from huggingface_hub import hf_hub_download
        import shutil

        logger.info(f"Downloading OpenShape from {config['hf_repo']}...")
        downloaded_path = hf_hub_download(
            repo_id=config["hf_repo"],
            filename=config["hf_file"],
        )

        shutil.copy2(downloaded_path, local_path)
        logger.info(f"Checkpoint saved to: {local_path}")
        return local_path

    except ImportError:
        logger.error("huggingface_hub not installed. Install with: pip install huggingface_hub")
        return None
    except Exception as e:
        logger.error(f"Failed to download checkpoint: {e}")
        return None


def get_openshape_repr_dim(model_size="openshape_pointbert_vitg14_rgb", use_proj=True):
    """Get the representation dimension for OpenShape features.

    Args:
        model_size: config key
        use_proj: if True, return CLIP-projected dim (1280),
                  else raw transformer dim (768)

    Returns:
        repr_dim: int
    """
    config = OPENSHAPE_CONFIGS.get(model_size, OPENSHAPE_CONFIGS["openshape_pointbert_vitg14_rgb"])
    if use_proj:
        return config["out_channel"]
    return config["dim"]

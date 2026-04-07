"""
PatchAlign3D model loader for Rectified-Point-Flow

Loads PatchAlign3D (arXiv 2601.02457) for point cloud feature extraction as a REPA teacher.
PatchAlign3D produces patch-level features aligned with CLIP text embeddings.

PatchAlign3D architecture:
  - FPS + KNN grouping (128 patches × 32 points)
  - Local encoder (Conv1d) → 256-dim per patch
  - Transformer encoder (12 blocks, 384-dim) → 384-dim patch features
  - Text projection head (optional) → 1280-dim CLIP-aligned features

For REPA alignment, we support:
  - 384-dim raw patch features (geometric)
  - 1280-dim CLIP-projected features (language-aligned)

Both are propagated to full point resolution via k-NN interpolation.

Dependencies: pointnet2_ops, knn_cuda, open_clip, timm
"""

import os
import sys
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from easydict import EasyDict

logger = logging.getLogger(__name__)

# HuggingFace model info
PATCHALIGN3D_HF_REPO = "patchalign3d/patchalign3d-encoder"

# Feature dimensions
PATCHALIGN3D_CONFIGS = {
    "patchalign3d_base": {
        "trans_dim": 384,           # Transformer hidden dimension
        "depth": 12,                # Number of transformer blocks
        "num_heads": 6,             # Number of attention heads
        "encoder_dims": 256,        # Local encoder output dimension
        "num_group": 128,           # Number of patches
        "group_size": 32,           # Points per patch
        "drop_path_rate": 0.1,
        "color": False,
        "clip_model": "ViT-bigG-14",
        "clip_pretrained": "laion2b_s39b_b160k",
        "clip_dim": 1280,           # CLIP text embedding dimension
    },
}


# ============================================================================
# Model Architecture (adapted from PatchAlign3D repository)
# ============================================================================

def fps(data, number):
    """Farthest Point Sampling using pointnet2_ops."""
    from pointnet2_ops import pointnet2_utils
    fps_idx = pointnet2_utils.furthest_point_sample(data, number)
    fps_data = pointnet2_utils.gather_operation(
        data.transpose(1, 2).contiguous(), fps_idx
    ).transpose(1, 2).contiguous()
    return fps_data


class PatchedGroup(nn.Module):
    """Groups points into patches using FPS and KNN."""

    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        try:
            from knn_cuda import KNN
            self.knn = KNN(k=self.group_size, transpose_mode=True)
        except ImportError:
            logger.warning("knn_cuda not found, falling back to torch implementation")
            self.knn = None

    def _knn_torch(self, xyz, center):
        """Fallback KNN using pure PyTorch."""
        # xyz: (B, N, 3), center: (B, G, 3)
        dist = torch.cdist(center, xyz)  # (B, G, N)
        _, idx = torch.topk(dist, self.group_size, dim=-1, largest=False)  # (B, G, M)
        return idx

    def forward(self, xyz):
        """
        Args:
            xyz: (B, N, C) where C>=3 (xyz | [extra])
        Returns:
            neighborhood: (B, G, M, C')
            center: (B, G, 3)
            patch_idx: (B, G, M)
        """
        batch_size, num_points, C = xyz.shape
        if C > 3:
            xyz_only = xyz[:, :, :3].contiguous()
            extra = xyz[:, :, 3:].contiguous()
        else:
            xyz_only = xyz.contiguous()
            extra = None

        center = fps(xyz_only, self.num_group)  # (B, G, 3)

        if self.knn is not None:
            _, idx = self.knn(xyz_only, center)  # (B, G, M)
        else:
            idx = self._knn_torch(xyz_only, center)

        idx_rel = idx.clone()
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx_flat = (idx + idx_base).view(-1)

        neigh_xyz = xyz_only.view(batch_size * num_points, -1)[idx_flat, :]
        neigh_xyz = neigh_xyz.view(batch_size, self.num_group, self.group_size, 3)

        if extra is not None:
            neigh_extra = extra.view(batch_size * num_points, -1)[idx_flat, :]
            neigh_extra = neigh_extra.view(batch_size, self.num_group, self.group_size, -1)
            neighborhood = torch.cat((neigh_xyz - center.unsqueeze(2), neigh_extra), dim=-1)
        else:
            neighborhood = neigh_xyz - center.unsqueeze(2)

        return neighborhood.contiguous(), center.contiguous(), idx_rel


class LocalEncoder(nn.Module):
    """Local patch encoder using Conv1d."""

    def __init__(self, encoder_channel, color=False):
        super().__init__()
        self.encoder_channel = encoder_channel
        in_channels = 6 if color else 3
        self.first_conv = nn.Sequential(
            nn.Conv1d(in_channels, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1),
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1),
        )

    def forward(self, point_groups):
        """
        Args:
            point_groups: (B, G, M, C)
        Returns:
            features: (B, G, encoder_channel)
        """
        bs, g, n, c = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, c).permute(0, 2, 1)  # (B*G, C, N)
        feature = self.first_conv(point_groups)
        feature_global = torch.max(feature, 2, keepdim=True)[0]
        feature_global = feature_global.repeat(1, 1, n)
        feature = torch.cat([feature_global, feature], 1)
        feature = self.second_conv(feature)
        feature = feature.max(dim=2)[0]  # (B*G, encoder_channel)
        return feature.reshape(bs, g, self.encoder_channel).contiguous()


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, drop=0.0, attn_drop=0.0, drop_path=0.0):
        super().__init__()
        from timm.models.layers import DropPath
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4.0, drop_path_rate=0.1):
        super().__init__()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop_path=dpr[i],
            )
            for i in range(depth)
        ])

    def forward(self, x, pos):
        for blk in self.blocks:
            x = blk(x + pos)
        return x


class PatchAlign3DEncoder(nn.Module):
    """PatchAlign3D encoder for extracting patch-level features."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.trans_dim = config.trans_dim
        self.depth = config.depth
        self.num_heads = config.num_heads
        self.color = config.color
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.encoder_dims = config.encoder_dims

        self.group_divider = PatchedGroup(num_group=self.num_group, group_size=self.group_size)
        self.encoder = LocalEncoder(encoder_channel=self.encoder_dims, color=self.color)
        self.reduce_dim = nn.Linear(self.encoder_dims, self.trans_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            num_heads=self.num_heads,
            drop_path_rate=config.drop_path_rate,
        )
        self.norm = nn.LayerNorm(self.trans_dim)

        self._init_weights()

    def _init_weights(self):
        from timm.models.layers import trunc_normal_
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward_patches(self, pts):
        """
        Args:
            pts: (B, C, N) with C>=3 (xyz | [extra])
        Returns:
            patch_emb: (B, trans_dim, G) - patch features
            patch_centers: (B, 3, G) - patch center coordinates
            patch_idx: (B, G, M) - point indices per patch
        """
        B, C, N = pts.shape
        pts_bn = pts.transpose(-1, -2).contiguous()  # (B, N, C)
        neighborhood, center, patch_idx = self.group_divider(pts_bn)
        group_tokens = self.encoder(neighborhood)
        group_tokens = self.reduce_dim(group_tokens)

        cls_tokens = self.cls_token.expand(group_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_tokens.size(0), -1, -1)
        pos = self.pos_embed(center)

        x = torch.cat((cls_tokens, group_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)
        feature = self.blocks(x, pos)
        patch_emb = self.norm(feature)[:, 1:, :].transpose(-1, -2).contiguous()  # (B, trans_dim, G)
        patch_centers = center.transpose(-1, -2).contiguous()  # (B, 3, G)
        return patch_emb, patch_centers, patch_idx

    def forward(self, pts):
        patch_emb, _, _ = self.forward_patches(pts)
        return patch_emb


class TextProjectionHead(nn.Module):
    """Projects patch features to CLIP text embedding space."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, patch_emb):
        """
        Args:
            patch_emb: (B, in_dim, G)
        Returns:
            projected: (B, G, out_dim) normalized
        """
        x = patch_emb.transpose(1, 2)  # (B, G, in_dim)
        x = self.proj(x)  # (B, G, out_dim)
        return F.normalize(x, dim=-1)


# ============================================================================
# Feature Propagation
# ============================================================================

def propagate_features_from_patches(xyz_full, patch_centers, patch_features, k=3):
    """Propagate patch features to all points via k-NN interpolation.

    Args:
        xyz_full: (B, N, 3) full-resolution point coordinates
        patch_centers: (B, G, 3) patch center coordinates
        patch_features: (B, G, C) patch features
        k: int, number of nearest neighbors

    Returns:
        point_features: (B, N, C) interpolated features at full resolution
    """
    B, N, _ = xyz_full.shape
    G, C = patch_features.shape[1], patch_features.shape[2]

    # Pairwise distances: (B, N, G)
    dist = torch.cdist(xyz_full, patch_centers)

    # k nearest neighbors
    dist_k, idx_k = torch.topk(dist, k, dim=-1, largest=False)  # (B, N, k)

    # Inverse distance weighting
    dist_k = dist_k.clamp(min=1e-6)
    weights = 1.0 / dist_k
    weights = weights.clamp(max=1e6)
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)

    # Gather and weighted sum
    idx_k_flat = idx_k.view(B, -1)  # (B, N*k)
    gathered = torch.gather(
        patch_features,
        dim=1,
        index=idx_k_flat.unsqueeze(-1).expand(-1, -1, C)
    )
    gathered = gathered.view(B, N, k, C)  # (B, N, k, C)
    point_features = (gathered * weights.unsqueeze(-1)).sum(dim=2)  # (B, N, C)

    return point_features


# ============================================================================
# Preprocessing
# ============================================================================

def preprocess_for_patchalign3d(pointclouds):
    """Preprocess point clouds for PatchAlign3D.

    PatchAlign3D expects:
    - Points in (B, 3, N) format
    - Y and Z axes swapped

    Args:
        pointclouds: (B, N, 3) tensor of point coordinates

    Returns:
        pts: (B, 3, N) preprocessed tensor
    """
    pts = pointclouds.transpose(1, 2).contiguous()  # (B, 3, N)
    # Swap Y and Z axes: [X, Y, Z] -> [X, Z, Y]
    pts = pts[:, [0, 2, 1], :]
    return pts


# ============================================================================
# Model Loading
# ============================================================================

def download_patchalign3d_checkpoint(save_dir='pretrained'):
    """Download PatchAlign3D checkpoint from HuggingFace."""
    os.makedirs(save_dir, exist_ok=True)
    local_path = os.path.join(save_dir, "patchalign3d_stage2.pt")

    if os.path.exists(local_path):
        logger.info(f"Checkpoint already exists: {local_path}")
        return local_path

    try:
        from huggingface_hub import hf_hub_download
        import shutil

        logger.info(f"Downloading PatchAlign3D from HuggingFace ({PATCHALIGN3D_HF_REPO})...")
        downloaded_path = hf_hub_download(
            repo_id=PATCHALIGN3D_HF_REPO,
            filename="patchalign3d.pt",
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


def load_patchalign3d(model_size="patchalign3d_base", ckpt_path=None, device="cuda", 
                       auto_download=True, use_clip_projection=True):
    """Load PatchAlign3D model for feature extraction.

    Args:
        model_size: config key (currently only 'patchalign3d_base')
        ckpt_path: optional path to a local checkpoint file
        device: device to load model on
        auto_download: if True, download pretrained weights from HuggingFace
        use_clip_projection: if True, also load the text projection head

    Returns:
        model: PatchAlign3DEncoder (frozen, eval mode)
        proj: TextProjectionHead or None
        feat_dim: int, output feature dimension
    """
    config_dict = PATCHALIGN3D_CONFIGS.get(model_size, PATCHALIGN3D_CONFIGS["patchalign3d_base"])
    config = EasyDict(config_dict)

    model = PatchAlign3DEncoder(config).to(device)
    proj = TextProjectionHead(config.trans_dim, config.clip_dim).to(device) if use_clip_projection else None

    # Find or download checkpoint
    if ckpt_path and os.path.exists(ckpt_path):
        pass
    elif auto_download:
        ckpt_path = download_patchalign3d_checkpoint()

    if ckpt_path and os.path.exists(ckpt_path):
        logger.info(f"Loading PatchAlign3D checkpoint from: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        # Load encoder weights
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        # Handle key prefixes
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.removeprefix("module.")
            new_state_dict[new_key] = v

        missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
        if missing:
            logger.warning(f"Missing keys: {missing[:5]}...")
        if unexpected:
            logger.warning(f"Unexpected keys: {unexpected[:5]}...")

        # Load projection head weights
        if proj is not None and "proj" in checkpoint:
            proj.load_state_dict(checkpoint["proj"], strict=False)
            logger.info("Loaded projection head weights")

        logger.info("PatchAlign3D checkpoint loaded successfully")
    else:
        logger.warning("No checkpoint provided or available, using random initialization")

    model.eval()
    if proj is not None:
        proj.eval()

    feat_dim = config.clip_dim if use_clip_projection else config.trans_dim
    return model, proj, feat_dim


def get_patchalign3d_repr_dim(model_size="patchalign3d_base", use_clip_projection=True):
    """Get the representation dimension for PatchAlign3D features.

    Args:
        model_size: config key
        use_clip_projection: if True, return CLIP dimension (1280),
                              else transformer dimension (384)

    Returns:
        repr_dim: int, feature dimension
    """
    config = PATCHALIGN3D_CONFIGS.get(model_size, PATCHALIGN3D_CONFIGS["patchalign3d_base"])
    if use_clip_projection:
        return config["clip_dim"]
    return config["trans_dim"]


# ============================================================================
# Full Pipeline for REPA
# ============================================================================

class PatchAlign3DFeatureExtractor:
    """Complete feature extraction pipeline for REPA."""

    def __init__(self, model, proj, device="cuda"):
        self.model = model
        self.proj = proj
        self.device = device

    @torch.inference_mode()
    def extract_features(self, pointclouds):
        """Extract point-wise features from point clouds.

        Args:
            pointclouds: (B, N, 3) tensor of point coordinates

        Returns:
            features: (B, N, feat_dim) point-wise features
        """
        B, N, _ = pointclouds.shape
        device = pointclouds.device

        # Preprocess
        pts = preprocess_for_patchalign3d(pointclouds)  # (B, 3, N)

        # Extract patch features
        with torch.autocast(device_type=str(device).split(":")[0], dtype=torch.float16):
            patch_emb, patch_centers, patch_idx = self.model.forward_patches(pts)
            # patch_emb: (B, trans_dim, G)
            # patch_centers: (B, 3, G)

            if self.proj is not None:
                # Project to CLIP space
                patch_feat = self.proj(patch_emb)  # (B, G, clip_dim)
            else:
                patch_feat = patch_emb.transpose(1, 2)  # (B, G, trans_dim)

        # Propagate to all points
        patch_centers_t = patch_centers.transpose(1, 2)  # (B, G, 3)
        # Undo the Y/Z swap for consistent k-NN
        patch_centers_t = patch_centers_t[:, :, [0, 2, 1]]

        features = propagate_features_from_patches(
            pointclouds.float(),
            patch_centers_t.float(),
            patch_feat.float(),
            k=3
        )

        return features


if __name__ == "__main__":
    # Test the loader
    print("Testing PatchAlign3D loader...")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model, proj, feat_dim = load_patchalign3d(
        model_size="patchalign3d_base",
        device=device,
        auto_download=True,
        use_clip_projection=True
    )
    print(f"Loaded PatchAlign3D with feat_dim={feat_dim}")

    # Create extractor
    extractor = PatchAlign3DFeatureExtractor(model, proj, device)

    # Test with dummy data
    B, N = 2, 2048
    dummy_points = torch.randn(B, N, 3).to(device)

    features = extractor.extract_features(dummy_points)
    print(f"Output features shape: {features.shape}")  # Expected: (2, 2048, 1280)

    print("PatchAlign3D loader test passed!")

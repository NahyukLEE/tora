"""
Find3D model loader for Rectified-Point-Flow

Loads Find3D (ICCV 2025 Highlight) for point cloud feature extraction as a REPA teacher.
Uses Find3D as a submodule and loads pretrained weights from HuggingFace.

Find3D architecture:
  - PointTransformerV3 backbone (encoder-decoder) -> 64-dim point features
  - Distillation head (4-layer MLP: 64->64->64->64->768) -> 768-dim features
    (aligned with SigLIP text embeddings via contrastive learning)

For REPA alignment, we support:
  - 768-dim distillation head output (semantically rich, language-aligned)
  - 64-dim backbone features (lower-level geometric features)

Dependencies: spconv, torch_scatter, flash_attn, addict, huggingface_hub
"""

import os
import sys
import logging
import platform
import torch
import torch.nn as nn
import numpy as np

logger = logging.getLogger(__name__)

# HuggingFace model ID for pretrained Find3D checkpoint
FIND3D_HF_REPO = "ziqima/find3d-checkpt0"

# Feature dimensions
FIND3D_CONFIGS = {
    "find3d_base": {
        "dim_output": 768,       # distillation head output (SigLIP-aligned)
        "backbone_dim": 64,      # PTV3 backbone feature dim
        "grid_size": 0.02,       # grid sampling resolution
    },
}


# ============================================================================
# Preprocessing Utilities
# ============================================================================

def fnv_hash_vec(arr):
    """FNV64-1A hashing for grid coordinate deduplication.

    Args:
        arr: (N, D) integer numpy array of grid coordinates
    Returns:
        hashes: (N,) uint64 array of hash values
    """
    assert arr.ndim == 2
    arr = arr.astype(np.uint64, copy=True)
    hashed_arr = np.uint64(14695981039346656037) * np.ones(arr.shape[0], dtype=np.uint64)
    for j in range(arr.shape[1]):
        hashed_arr *= np.uint64(1099511628211)
        hashed_arr = np.bitwise_xor(hashed_arr, arr[:, j])
    return hashed_arr


def grid_sample_numpy(coord, grid_size=0.02):
    """Grid-sample a point cloud, keeping one point per occupied voxel.

    Args:
        coord: (N, 3) numpy array of point coordinates
        grid_size: float, voxel size
    Returns:
        sub_idx: (M,) numpy array of indices into the original point cloud
    """
    scaled_coord = coord / np.array(grid_size)
    grid_coord = np.floor(scaled_coord).astype(int)

    min_coord = grid_coord.min(axis=0)
    grid_coord -= min_coord

    hash_vals = fnv_hash_vec(grid_coord)
    sort_idx = np.argsort(hash_vals)
    hash_sorted = hash_vals[sort_idx]

    # Keep first point in each voxel
    mask = np.ones(len(hash_sorted), dtype=bool)
    mask[1:] = hash_sorted[1:] != hash_sorted[:-1]
    sub_idx = sort_idx[mask]

    return sub_idx


def preprocess_single_pcd(xyz_np, normals_np=None, grid_size=0.02):
    """Preprocess a single point cloud following Find3D's evaluation pipeline.

    Steps:
      1. Normalize: center and scale to 0.75-size bounding box
      2. Axis swap: [x, y, z] -> [-x, z, y]
      3. Center shift
      4. Grid sample at grid_size resolution
      5. Additional centering (exclude z-axis)
      6. Build features: dummy RGB (normalized) + normals
      7. Compute grid coordinates

    Args:
        xyz_np: (N, 3) numpy array of coordinates
        normals_np: (N, 3) numpy array of normals, or None
        grid_size: float, grid sampling resolution

    Returns:
        data_dict: dict with 'coord', 'feat', 'grid_coord' tensors
        xyz_full_normalized: (N, 3) numpy array of all points in normalized space
                             (for k-NN propagation back to full resolution)
    """
    N = xyz_np.shape[0]
    xyz = xyz_np.copy()
    nrm = normals_np.copy() if normals_np is not None else np.zeros_like(xyz)

    # 1. Normalize: center and scale
    center = (xyz.max(axis=0) + xyz.min(axis=0)) / 2.0
    xyz -= center
    max_range = (xyz.max(axis=0) - xyz.min(axis=0)).max()
    if max_range > 1e-8:
        xyz = xyz / max_range * 0.75

    # 2. Axis swap: [x, y, z] -> [-x, z, y]
    xyz = np.stack([-xyz[:, 0], xyz[:, 2], xyz[:, 1]], axis=1)
    nrm = np.stack([-nrm[:, 0], nrm[:, 2], nrm[:, 1]], axis=1)

    # 3. Center shift
    center_shift = (xyz.max(axis=0) + xyz.min(axis=0)) / 2.0
    xyz -= center_shift

    # Save full-resolution normalized coords (before grid sampling)
    xyz_full_normalized = xyz.copy()

    # 4. Grid sample
    sub_idx = grid_sample_numpy(xyz, grid_size)
    xyz_sub = xyz[sub_idx]
    nrm_sub = nrm[sub_idx]

    # 5. Additional centering (exclude z)
    cs2 = (xyz_sub.max(axis=0) + xyz_sub.min(axis=0)) / 2.0
    cs2[2] = 0.0  # don't shift z
    xyz_sub -= cs2
    # Apply same shift to full coords for consistent k-NN space
    xyz_full_normalized -= cs2

    # 6. Build features: dummy RGB (normalized to [-1,1]) + normals
    # Find3D uses NormalizeColor: color / 127.5 - 1. For black (0,0,0) -> -1.0
    rgb_dummy = np.full_like(xyz_sub, -1.0)
    feat = np.concatenate([rgb_dummy, nrm_sub], axis=1)  # (M, 6)

    # 7. Grid coordinates
    grid_coord = np.floor(xyz_sub / grid_size).astype(int)
    grid_coord -= grid_coord.min(axis=0)

    data_dict = {
        "coord": torch.FloatTensor(xyz_sub),
        "feat": torch.FloatTensor(feat),
        "grid_coord": torch.IntTensor(grid_coord),
    }

    return data_dict, xyz_full_normalized


def preprocess_batch_for_find3d(pointclouds, normals, grid_size=0.02):
    """Preprocess a batch of point clouds for Find3D with offset-based batching.

    Args:
        pointclouds: (B, N, 3) tensor of point coordinates
        normals: (B, N, 3) tensor of surface normals

    Returns:
        batched_data: dict with 'coord', 'feat', 'grid_coord', 'offset' for Find3D
        xyz_full_norm_list: list of B (N, 3) numpy arrays in normalized space
        xyz_sub_list: list of B (M_i, 3) tensors of subsampled coordinates
    """
    B, N, _ = pointclouds.shape

    all_coords = []
    all_feats = []
    all_grid_coords = []
    xyz_full_norm_list = []
    xyz_sub_list = []
    cumulative = 0

    for i in range(B):
        xyz_np = pointclouds[i].detach().cpu().numpy()
        nrm_np = normals[i].detach().cpu().numpy()

        data_dict, xyz_full_norm = preprocess_single_pcd(xyz_np, nrm_np, grid_size)

        M = data_dict["coord"].shape[0]
        all_coords.append(data_dict["coord"])
        all_feats.append(data_dict["feat"])
        all_grid_coords.append(data_dict["grid_coord"])
        xyz_full_norm_list.append(xyz_full_norm)
        xyz_sub_list.append(data_dict["coord"])  # subsampled coords
        cumulative += M

    batched_data = {
        "coord": torch.cat(all_coords, dim=0),
        "feat": torch.cat(all_feats, dim=0),
        "grid_coord": torch.cat(all_grid_coords, dim=0),
        "offset": torch.IntTensor([
            sum(d.shape[0] for d in all_coords[:i+1]) for i in range(B)
        ]),
    }

    return batched_data, xyz_full_norm_list, xyz_sub_list


# ============================================================================
# Feature Propagation
# ============================================================================

def propagate_features_knn(xyz_full, xyz_sub, features_sub, k=3):
    """Propagate features from subsampled points to full resolution via k-NN
    inverse-distance-weighted interpolation.

    Args:
        xyz_full: (N, 3) tensor, full-resolution point coordinates (in normalized space)
        xyz_sub: (M, 3) tensor, subsampled point coordinates (in normalized space)
        features_sub: (M, C) tensor, features at subsampled points
        k: int, number of nearest neighbors

    Returns:
        features_full: (N, C) tensor, interpolated features at full resolution
    """
    # Pairwise distances: (N, M)
    dist = torch.cdist(xyz_full.unsqueeze(0), xyz_sub.unsqueeze(0)).squeeze(0)

    # k nearest neighbors
    dist_k, idx_k = torch.topk(dist, k, dim=-1, largest=False)  # (N, k)

    # Inverse distance weighting
    dist_k = dist_k.clamp(min=1e-6)
    weights = 1.0 / dist_k
    weights = weights.clamp(max=1e6)
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)

    # Gather and weighted sum
    N = xyz_full.shape[0]
    C = features_sub.shape[-1]
    gathered = features_sub[idx_k.view(-1)].view(N, k, C)  # (N, k, C)
    features_full = (gathered * weights.unsqueeze(-1)).sum(dim=1)  # (N, C)

    return features_full


# ============================================================================
# Model Loading
# ============================================================================

def _setup_find3d_imports():
    """Add Find3D submodule to sys.path so its internal imports resolve."""
    # Resolve from tora/extern/find3d (vendored submodule)
    _this_dir = os.path.dirname(__file__)
    find3d_root = os.path.normpath(os.path.join(_this_dir, os.pardir, os.pardir, "extern", "find3d"))
    if not os.path.isdir(find3d_root):
        raise ImportError(
            "Find3D submodule not found. Expected at:\n"
            f"  {find3d_root}\n"
            "Initialize it with:\n"
            "  git submodule update --init tora/extern/find3d"
        )
    # Require Find3D repo to be populated (not an empty submodule dir)
    model_dir = os.path.join(find3d_root, "model")
    if not os.path.isdir(model_dir):
        raise ImportError(
            "Find3D submodule is empty or incomplete. Populate it with:\n"
            "  cd <repo_root> && git submodule update --init tora/extern/find3d\n"
            f"  (expected: {model_dir})"
        )
    if find3d_root not in sys.path:
        sys.path.insert(0, find3d_root)


def _patch_spconv_for_arm(model):
    """On aarch64/ARM, force spconv to use Native conv algorithm to avoid GEMM failures.

    spconv's default ImplicitGemm algorithm may not have tuning results for
    certain GPU architectures (e.g., Grace Hopper on aarch64), causing
    'can't find suitable algorithm' errors. The Native fallback avoids GEMM.
    """
    if platform.machine() not in ('aarch64', 'arm64'):
        return False
    try:
        from spconv.pytorch.conv import SparseConvolution
        try:
            from spconv.pytorch import ConvAlgo
            native_algo = ConvAlgo.Native
        except (ImportError, AttributeError):
            native_algo = 0  # Native = 0 in spconv 2.x

        patched = 0
        for m in model.modules():
            if isinstance(m, SparseConvolution):
                m.algo = native_algo
                patched += 1
        if patched > 0:
            logger.info(
                f"Patched {patched} spconv layers to Native algorithm (aarch64 workaround)"
            )
        return patched > 0
    except ImportError:
        return False


def load_find3d(model_size="find3d_base", ckpt_path=None, device="cuda", auto_download=True):
    """Load Find3D model for feature extraction.

    Args:
        model_size: config key (currently only 'find3d_base')
        ckpt_path: optional path to a local checkpoint file
        device: device to load model on
        auto_download: if True, download pretrained weights from HuggingFace

    Returns:
        model: Find3D nn.Module (frozen, eval mode)
        feat_dim: int, output feature dimension (768 for distillation head)
    """
    config = FIND3D_CONFIGS.get(model_size, FIND3D_CONFIGS["find3d_base"])

    _setup_find3d_imports()
    from model.backbone.pt3.model import Find3D

    if ckpt_path and os.path.exists(ckpt_path):
        # Load from local checkpoint
        model = Find3D(dim_output=config["dim_output"])
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
        # Strip 'module.' prefix if present
        new_state_dict = {}
        for k, v in state_dict.items():
            new_state_dict[k.removeprefix("module.")] = v
        model.load_state_dict(new_state_dict, strict=False)
        logger.info(f"Loaded Find3D from local checkpoint: {ckpt_path}")
    elif auto_download:
        model = Find3D.from_pretrained(FIND3D_HF_REPO, dim_output=config["dim_output"])
        logger.info(f"Loaded Find3D from HuggingFace: {FIND3D_HF_REPO}")
    else:
        model = Find3D(dim_output=config["dim_output"])
        logger.warning("Initialized Find3D with random weights")

    model = model.to(device)
    model.eval()

    # On ARM/aarch64, patch spconv to use Native algorithm
    _patch_spconv_for_arm(model)

    return model, config["dim_output"]


def get_find3d_repr_dim(model_size="find3d_base", use_backbone_feat=False):
    """Get the representation dimension for Find3D features.

    Args:
        model_size: config key
        use_backbone_feat: if True, return backbone dim (64),
                           else distillation head dim (768)

    Returns:
        repr_dim: int, feature dimension
    """
    config = FIND3D_CONFIGS.get(model_size, FIND3D_CONFIGS["find3d_base"])
    if use_backbone_feat:
        return config["backbone_dim"]
    return config["dim_output"]

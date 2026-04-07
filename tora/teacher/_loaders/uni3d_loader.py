"""
Uni3D model loader for Rectified-Point-Flow
Contains model architecture code directly to avoid import issues with Uni3D's internal dependencies.
Supports both global features (CLS token) and point-wise features (with feature propagation).
"""

import os
import sys
import logging
import torch
import torch.nn as nn
import numpy as np
import timm

# HuggingFace model IDs for Uni3D checkpoints
UNI3D_HF_REPO = "BAAI/Uni3D"
UNI3D_HF_FILES = {
    'uni3d_base': 'modelzoo/uni3d-b/model.pt',
    'uni3d_large': 'modelzoo/uni3d-l/model.pt',
    'uni3d_giant': 'modelzoo/uni3d-g/model.pt',
}


# ============================================================================
# Utility Functions
# ============================================================================

def fps(data, number):
    """Farthest Point Sampling
    Args:
        data: (B, N, 3) point coordinates
        number: int, number of points to sample
    Returns:
        fps_data: (B, number, 3) sampled points
    """
    from pointnet2_ops import pointnet2_utils
    fps_idx = pointnet2_utils.furthest_point_sample(data, number)
    fps_data = pointnet2_utils.gather_operation(
        data.transpose(1, 2).contiguous(), fps_idx
    ).transpose(1, 2).contiguous()
    return fps_data


def knn_point(nsample, xyz, new_xyz):
    """K-Nearest Neighbors
    Args:
        nsample: int, number of neighbors
        xyz: (B, N, 3) all points
        new_xyz: (B, S, 3) query points
    Returns:
        group_idx: (B, S, nsample) indices of neighbors
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx


def square_distance(src, dst):
    """Calculate squared Euclidean distance between points.
    Args:
        src: (B, N, 3)
        dst: (B, M, 3)
    Returns:
        dist: (B, N, M)
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def propagate_features(xyz, center, group_features, k=3):
    """
    Propagate group-level features to all points using k-NN interpolation.
    
    Args:
        xyz: (B, N, 3) target point coordinates
        center: (B, G, 3) source (group center) coordinates
        group_features: (B, G, C) features at group centers
        k: number of nearest neighbors for interpolation
    
    Returns:
        point_features: (B, N, C) interpolated features for all points
    """
    B, N, _ = xyz.shape
    _, G, C = group_features.shape
    
    # Compute distances from all points to all centers
    dist = square_distance(xyz, center)  # (B, N, G)
    
    # Find k nearest centers for each point
    dist_k, idx_k = torch.topk(dist, k, dim=-1, largest=False)  # (B, N, k)
    
    # Compute interpolation weights (inverse distance weighting)
    # Use larger epsilon and clamping to prevent numerical issues
    dist_k = dist_k.clamp(min=1e-6)
    weights = 1.0 / dist_k  # (B, N, k)
    weights = weights.clamp(max=1e6)
    
    weight_sum = weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    weights = weights / weight_sum  # Normalize
    
    # Gather features from k nearest centers
    idx_k_flat = idx_k.view(B, -1)  # (B, N*k)
    
    gathered_features = torch.gather(
        group_features, 
        dim=1, 
        index=idx_k_flat.unsqueeze(-1).expand(-1, -1, C)
    )
    gathered_features = gathered_features.view(B, N, k, C)  # (B, N, k, C)
    
    # Weighted sum
    point_features = (gathered_features * weights.unsqueeze(-1)).sum(dim=2)  # (B, N, C)
    
    return point_features


# ============================================================================
# Uni3D Model Architecture
# ============================================================================

class PatchDropout(nn.Module):
    """https://arxiv.org/abs/2212.00794"""

    def __init__(self, prob, exclude_first_token=True):
        super().__init__()
        assert 0 <= prob < 1.
        self.prob = prob
        self.exclude_first_token = exclude_first_token

    def forward(self, x):
        if self.exclude_first_token:
            cls_tokens, x = x[:, :1], x[:, 1:]
        else:
            cls_tokens = torch.jit.annotate(torch.Tensor, x[:, :1])

        batch = x.size()[0]
        num_tokens = x.size()[1]

        batch_indices = torch.arange(batch)
        batch_indices = batch_indices[..., None]

        keep_prob = 1 - self.prob
        num_patches_keep = max(1, int(num_tokens * keep_prob))

        rand = torch.randn(batch, num_tokens)
        patch_indices_keep = rand.topk(num_patches_keep, dim=-1).indices

        x = x[batch_indices, patch_indices_keep]

        if self.exclude_first_token:
            x = torch.cat((cls_tokens, x), dim=1)

        return x


class Group(nn.Module):
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size

    def forward(self, xyz, color):
        """
        Args:
            xyz: (B, N, 3)
            color: (B, N, 3)
        Returns:
            neighborhood: (B, G, M, 3) normalized neighborhood
            center: (B, G, 3) group center coordinates
            features: (B, G, M, 6) neighborhood xyz + color
        """
        batch_size, num_points, _ = xyz.shape
        center = fps(xyz, self.num_group)  # (B, G, 3)
        idx = knn_point(self.group_size, xyz, center)  # (B, G, M)
        
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()

        neighborhood_color = color.view(batch_size * num_points, -1)[idx, :]
        neighborhood_color = neighborhood_color.view(batch_size, self.num_group, self.group_size, 3).contiguous()

        neighborhood = neighborhood - center.unsqueeze(2)
        features = torch.cat((neighborhood, neighborhood_color), dim=-1)
        return neighborhood, center, features


class Encoder(nn.Module):
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(6, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )

    def forward(self, point_groups):
        """
        Args:
            point_groups: (B, G, M, 6)
        Returns:
            feature_global: (B, G, C)
        """
        bs, g, n, _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 6)
        feature = self.first_conv(point_groups.transpose(2, 1))
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)
        feature = self.second_conv(feature)
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]
        return feature_global.reshape(bs, g, self.encoder_channel)


class PointcloudEncoder(nn.Module):
    def __init__(self, point_transformer, args):
        super().__init__()
        self.trans_dim = args.pc_feat_dim
        self.embed_dim = args.embed_dim
        self.group_size = args.group_size
        self.num_group = args.num_group
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)
        self.encoder_dim = args.pc_encoder_dim
        self.encoder = Encoder(encoder_channel=self.encoder_dim)
        
        self.encoder2trans = nn.Linear(self.encoder_dim, self.trans_dim)
        self.trans2embed = nn.Linear(self.trans_dim, self.embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )
        self.patch_dropout = PatchDropout(args.patch_dropout) if args.patch_dropout > 0. else nn.Identity()
        self.visual = point_transformer
        
        # Store number of blocks for intermediate feature extraction
        self.num_blocks = len(self.visual.blocks)

    def forward(self, pts, colors, return_point_features=False, intermediate_layers=None):
        """
        Args:
            pts: (B, N, 3) point coordinates
            colors: (B, N, 3) point colors
            return_point_features: if True, returns point-wise features
            intermediate_layers: list of layer indices to extract (e.g., [3, 7, 11] for H4, H8, H12)
                                 If None, uses [num_blocks//3-1, 2*num_blocks//3-1, num_blocks-1]
        
        Returns:
            if return_point_features=False: (B, embed_dim) global feature
            if return_point_features=True: dict with point-wise features
        """
        # Tokenization
        _, center, features = self.group_divider(pts, colors)  # center: (B, 512, 3)
        group_input_tokens = self.encoder(features)
        group_input_tokens = self.encoder2trans(group_input_tokens)
        
        # Prepare CLS token and position embeddings
        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)
        pos = self.pos_embed(center)
        
        x = torch.cat((cls_tokens, group_input_tokens), dim=1)  # (B, 513, trans_dim)
        pos = torch.cat((cls_pos, pos), dim=1)
        x = x + pos
        
        x = self.patch_dropout(x)
        x = self.visual.pos_drop(x)

        if not return_point_features:
            # Original behavior: only return global feature
            for blk in self.visual.blocks:
                x = blk(x)
            x = self.visual.norm(x[:, 0, :])
            x = self.visual.fc_norm(x)
            x = self.trans2embed(x)
            return x
        
        # Extract intermediate layer features
        if intermediate_layers is None:
            # Default: extract at 1/3, 2/3, and final layer
            n = self.num_blocks
            intermediate_layers = [n // 3 - 1, 2 * n // 3 - 1, n - 1]
        
        intermediate_features = {}
        for i, blk in enumerate(self.visual.blocks):
            x = blk(x)
            if i in intermediate_layers:
                # Extract patch features (exclude CLS token)
                patch_feat = x[:, 1:, :]  # (B, 512, trans_dim)
                intermediate_features[f'H{i+1}'] = patch_feat
        
        # Apply final normalization
        x_normed = self.visual.norm(x)
        
        # Global feature from CLS token
        cls_feat = self.visual.fc_norm(x_normed[:, 0, :])
        global_feat = self.trans2embed(cls_feat)  # (B, embed_dim)
        
        # Propagate each intermediate feature to all points
        point_features_list = []
        for layer_name, patch_feat in intermediate_features.items():
            # Apply normalization and projection to embed_dim
            patch_feat_proj = self.visual.fc_norm(self.visual.norm(patch_feat))
            patch_feat_proj = self.trans2embed(patch_feat_proj)  # (B, 512, embed_dim)
            
            # Propagate to all points
            point_feat = propagate_features(pts, center, patch_feat_proj, k=3)  # (B, N, embed_dim)
            point_features_list.append(point_feat)
        
        # Concatenate multi-scale features
        point_features_concat = torch.cat(point_features_list, dim=-1)  # (B, N, embed_dim * num_layers)
        
        return {
            'point_features': point_features_concat,  # (B, N, embed_dim * num_layers)
            'point_features_list': point_features_list,  # List of (B, N, embed_dim)
            'global_features': global_feat,  # (B, embed_dim)
            'center': center,  # (B, 512, 3)
            'intermediate_features': intermediate_features,  # Dict of layer_name -> (B, 512, trans_dim)
        }


class Uni3D(nn.Module):
    def __init__(self, point_encoder):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.point_encoder = point_encoder

    def encode_pc(self, pc, return_point_features=False, intermediate_layers=None):
        """
        Encode point cloud to features.
        
        Args:
            pc: (B, N, 6) point cloud with xyz and color
            return_point_features: if True, returns point-wise features
            intermediate_layers: list of layer indices for intermediate feature extraction
        
        Returns:
            if return_point_features=False: (B, embed_dim) global feature
            if return_point_features=True: dict with point_features, global_features, etc.
        """
        xyz = pc[:, :, :3].contiguous()
        color = pc[:, :, 3:].contiguous()
        return self.point_encoder(xyz, color, return_point_features, intermediate_layers)

    def forward(self, pc, text=None, image=None, return_point_features=False):
        pc_embed = self.encode_pc(pc, return_point_features=return_point_features)
        if return_point_features:
            return {
                'pc_embed': pc_embed['global_features'],
                'point_features': pc_embed['point_features'],
                'logit_scale': self.logit_scale.exp()
            }
        return {'pc_embed': pc_embed, 'logit_scale': self.logit_scale.exp()}


def create_uni3d(args):
    """Create Uni3D model"""
    point_transformer = timm.create_model(
        args.pc_model, 
        checkpoint_path=args.pretrained_pc if args.pretrained_pc else '',
        drop_path_rate=args.drop_path_rate
    )
    point_encoder = PointcloudEncoder(point_transformer, args)
    model = Uni3D(point_encoder=point_encoder)
    return model


# ============================================================================
# Model Configurations
# ============================================================================

UNI3D_CONFIGS = {
    'uni3d_tiny': {
        'pc_model': 'eva02_tiny_patch14_224',
        'pc_feat_dim': 192,
        'embed_dim': 1024,
        'group_size': 64,
        'num_group': 512,
        'pc_encoder_dim': 512,
        'drop_path_rate': 0.0,
        'patch_dropout': 0.0,
        'num_blocks': 12,
    },
    'uni3d_small': {
        'pc_model': 'eva02_small_patch14_224',
        'pc_feat_dim': 384,
        'embed_dim': 1024,
        'group_size': 64,
        'num_group': 512,
        'pc_encoder_dim': 512,
        'drop_path_rate': 0.0,
        'patch_dropout': 0.0,
        'num_blocks': 12,
    },
    'uni3d_base': {
        'pc_model': 'eva02_base_patch14_448',
        'pc_feat_dim': 768,
        'embed_dim': 1024,
        'group_size': 64,
        'num_group': 512,
        'pc_encoder_dim': 512,
        'drop_path_rate': 0.0,
        'patch_dropout': 0.0,
        'num_blocks': 12,
    },
    'uni3d_large': {
        'pc_model': 'eva02_large_patch14_448',
        'pc_feat_dim': 1024,
        'embed_dim': 1024,
        'group_size': 64,
        'num_group': 512,
        'pc_encoder_dim': 512,
        'drop_path_rate': 0.0,
        'patch_dropout': 0.0,
        'num_blocks': 24,
    },
    'uni3d_giant': {
        'pc_model': 'eva_giant_patch14_560.m30m_ft_in22k_in1k',
        'pc_feat_dim': 1408,
        'embed_dim': 1024,
        'group_size': 64,
        'num_group': 512,
        'pc_encoder_dim': 512,
        'drop_path_rate': 0.0,
        'patch_dropout': 0.0,
        'num_blocks': 40,
    },
}


# ============================================================================
# Loading Functions
# ============================================================================

def download_uni3d_checkpoint(model_size, save_dir='pretrained'):
    """Download Uni3D checkpoint from HuggingFace if not exists.
    
    Args:
        model_size: one of 'uni3d_base', 'uni3d_large', 'uni3d_giant'
        save_dir: directory to save the checkpoint
    
    Returns:
        local_path: path to the downloaded checkpoint
    """
    if model_size not in UNI3D_HF_FILES:
        print(f"No HuggingFace checkpoint available for {model_size}")
        return None
    
    os.makedirs(save_dir, exist_ok=True)
    local_path = os.path.join(save_dir, f"{model_size}.pt")
    
    if os.path.exists(local_path):
        print(f"Checkpoint already exists: {local_path}")
        return local_path
    
    try:
        from huggingface_hub import hf_hub_download
        import shutil

        hf_filename = UNI3D_HF_FILES[model_size]
        print(f"Downloading {model_size} from HuggingFace ({UNI3D_HF_REPO}/{hf_filename})...")

        # Download to HF cache (default), then copy to our local path
        downloaded_path = hf_hub_download(
            repo_id=UNI3D_HF_REPO,
            filename=hf_filename,
        )

        shutil.copy2(downloaded_path, local_path)
        print(f"Checkpoint saved to: {local_path}")
        return local_path

    except ImportError:
        print("huggingface_hub not installed. Install with: pip install huggingface_hub")
        return None
    except Exception as e:
        print(f"Failed to download checkpoint: {e}")
        return None


def load_uni3d(model_size='uni3d_base', ckpt_path=None, device='cuda', auto_download=True):
    """Load Uni3D model for feature extraction
    
    Args:
        model_size: one of 'uni3d_tiny', 'uni3d_small', 'uni3d_base', 'uni3d_large', 'uni3d_giant'
        ckpt_path: optional path to checkpoint file
        device: device to load model on
        auto_download: if True, automatically download checkpoint from HuggingFace
    
    Returns:
        model: Uni3D model
        feat_dim: output feature dimension (embed_dim)
    """
    if model_size not in UNI3D_CONFIGS:
        raise ValueError(f"Unknown model size: {model_size}. Available: {list(UNI3D_CONFIGS.keys())}")
    
    config = UNI3D_CONFIGS[model_size]
    
    from easydict import EasyDict
    args = EasyDict({
        'pc_model': config['pc_model'],
        'pc_feat_dim': config['pc_feat_dim'],
        'embed_dim': config['embed_dim'],
        'group_size': config['group_size'],
        'num_group': config['num_group'],
        'pc_encoder_dim': config['pc_encoder_dim'],
        'drop_path_rate': config['drop_path_rate'],
        'patch_dropout': config['patch_dropout'],
        'pretrained_pc': '',
    })
    
    model = create_uni3d(args)
    
    # Try to find or download checkpoint
    if ckpt_path and os.path.exists(ckpt_path):
        pass  # Use provided path
    elif auto_download and model_size in UNI3D_HF_FILES:
        # Try to download from HuggingFace
        ckpt_path = download_uni3d_checkpoint(model_size)
    
    if ckpt_path and os.path.exists(ckpt_path):
        print(f"Loading Uni3D checkpoint from: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        
        if 'module' in checkpoint:
            state_dict = checkpoint['module']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        
        model.load_state_dict(new_state_dict, strict=False)
        print("Checkpoint loaded successfully")
    else:
        print(f"No checkpoint provided or available, using random initialization")
    
    model = model.to(device)
    model.eval()
    
    return model, config['embed_dim']


def get_uni3d_repr_dim(model_size='uni3d_large', num_layers=3):
    """Get the representation dimension for Uni3D point-wise features.
    
    For REPA, we use multi-scale point features which concatenate features 
    from multiple transformer layers.
    
    Args:
        model_size: Uni3D model size
        num_layers: Number of layers to extract (default 3: at 1/3, 2/3, final)
    
    Returns:
        repr_dim: Total feature dimension (embed_dim * num_layers)
    """
    config = UNI3D_CONFIGS.get(model_size, UNI3D_CONFIGS['uni3d_large'])
    return config['embed_dim'] * num_layers

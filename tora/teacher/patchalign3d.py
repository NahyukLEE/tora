import torch
import torch.nn as nn

from .base import BaseTeacher


class PatchAlign3DTeacher(BaseTeacher):
    """Frozen PatchAlign3D teacher encoder.

    Args:
        repr_dim: Output representation dimension.
        teacher_size: Model identifier, e.g. ``"patchalign3d_base"``.
        use_clip_projection: If True, project patch features into CLIP space.
    """

    def __init__(
        self,
        repr_dim: int,
        teacher_size: str = "patchalign3d_base",
        use_clip_projection: bool = True,
    ):
        super().__init__(repr_dim=repr_dim)
        self.teacher_size = teacher_size
        self.use_clip_projection = use_clip_projection
        self.patchalign3d_proj = None
        self.patchalign3d_feat_dim: int | None = None
        self.patchalign3d_config: dict | None = None
        self.load_encoder()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def load_encoder(self, **kwargs) -> None:
        from ._loaders.patchalign3d_loader import load_patchalign3d, PATCHALIGN3D_CONFIGS

        model_size = (
            self.teacher_size
            if self.teacher_size.startswith("patchalign3d_")
            else f"patchalign3d_{self.teacher_size}"
        )
        self.encoder, self.patchalign3d_proj, self.patchalign3d_feat_dim = load_patchalign3d(
            model_size=model_size,
            device="cuda" if torch.cuda.is_available() else "cpu",
            auto_download=True,
            use_clip_projection=self.use_clip_projection,
        )
        self.patchalign3d_config = PATCHALIGN3D_CONFIGS[model_size]
        self._freeze_model(self.encoder)
        if self.patchalign3d_proj is not None:
            self._freeze_model(self.patchalign3d_proj)
        print(
            f"Loaded PatchAlign3D teacher: {model_size} "
            f"(feat_dim={self.patchalign3d_feat_dim}, clip_proj={self.use_clip_projection})"
        )

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------
    def extract_features(self, batch: dict) -> tuple:
        from ._loaders.patchalign3d_loader import (
            preprocess_for_patchalign3d,
            propagate_features_from_patches,
        )

        pointclouds = batch["pointclouds_gt"]          # (B, N, 3)
        points_per_part = batch["points_per_part"]     # (B, P)
        B, N, C = pointclouds.shape
        n_valid_parts = points_per_part != 0
        device = pointclouds.device

        pts = preprocess_for_patchalign3d(pointclouds.to(device))  # (B, 3, N)

        with torch.inference_mode():
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                patch_emb, patch_centers, patch_idx = self.encoder.forward_patches(pts)

                if self.patchalign3d_proj is not None:
                    patch_feat = self.patchalign3d_proj(patch_emb)
                else:
                    patch_feat = patch_emb.transpose(1, 2)

            patch_feat = patch_feat.float()

        patch_centers_t = patch_centers.transpose(1, 2)  # (B, G, 3)
        # Undo the Y/Z swap for consistent k-NN
        patch_centers_t = patch_centers_t[:, :, [0, 2, 1]]

        features = propagate_features_from_patches(
            pointclouds.to(device).float(),
            patch_centers_t.float(),
            patch_feat,
            k=3,
        )  # (B, N, feat_dim)

        return features, None, n_valid_parts

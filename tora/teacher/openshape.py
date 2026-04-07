import torch
import torch.nn as nn

from .base import BaseTeacher


class OpenShapeTeacher(BaseTeacher):
    """Frozen OpenShape PointBERT teacher encoder.

    Args:
        repr_dim: Output representation dimension.
        teacher_size: Model identifier, e.g. ``"openshape_pointbert_vitg14_rgb"``.
        use_clip_projection: If True, use CLIP-projected patch features (1280-dim).
    """

    def __init__(
        self,
        repr_dim: int,
        teacher_size: str = "openshape_pointbert_vitg14_rgb",
        use_clip_projection: bool = True,
    ):
        super().__init__(repr_dim=repr_dim)
        self.teacher_size = teacher_size
        self.use_clip_projection = use_clip_projection
        self.openshape_feat_dim: int | None = None
        self.openshape_config: dict | None = None
        self.load_encoder()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def load_encoder(self, **kwargs) -> None:
        from ._loaders.openshape_loader import load_openshape, OPENSHAPE_CONFIGS

        model_size = (
            self.teacher_size
            if self.teacher_size.startswith("openshape_")
            else f"openshape_{self.teacher_size}"
        )
        self.encoder, self.openshape_feat_dim = load_openshape(
            model_size=model_size,
            device="cuda" if torch.cuda.is_available() else "cpu",
            auto_download=True,
        )
        self.openshape_config = OPENSHAPE_CONFIGS[model_size]
        self._freeze_model(self.encoder)
        print(f"Loaded OpenShape teacher: {model_size} (feat_dim={self.openshape_feat_dim})")

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------
    def extract_features(self, batch: dict) -> tuple:
        from ._loaders.openshape_loader import propagate_features_from_centroids

        pointclouds = batch["pointclouds_gt"]          # (B, N, 3)
        points_per_part = batch["points_per_part"]     # (B, P)
        B, N, C = pointclouds.shape
        n_valid_parts = points_per_part != 0
        device = pointclouds.device

        coords = pointclouds.to(device).float()
        # OpenShape expects xyz + rgb (6-dim), use dummy color
        color = torch.ones_like(coords) * 0.4
        features_in = torch.cat([coords, color], dim=-1)  # (B, N, 6)

        with torch.inference_mode():
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                patch_feat, patch_feat_proj, centroids = self.encoder.forward_patches(
                    coords, features_in
                )

            if self.use_clip_projection and patch_feat_proj is not None:
                feat = patch_feat_proj.float()
            else:
                feat = patch_feat.float()

        centroids_t = centroids.transpose(1, 2).float()  # (B, P, 3)
        features = propagate_features_from_centroids(
            coords, centroids_t, feat, k=3
        )  # (B, N, feat_dim)

        return features, None, n_valid_parts

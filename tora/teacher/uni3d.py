import torch
import torch.nn as nn

from .base import BaseTeacher


class Uni3DTeacher(BaseTeacher):
    """Frozen Uni3D teacher encoder.

    Args:
        repr_dim: Output representation dimension.
        teacher_size: Model identifier, e.g. ``"uni3d_large"``.
            If the string does not already start with ``uni3d_``, the prefix is prepended.
    """

    def __init__(self, repr_dim: int, teacher_size: str = "uni3d_large"):
        super().__init__(repr_dim=repr_dim)
        self.teacher_size = teacher_size
        self.uni3d_embed_dim: int | None = None
        self.uni3d_config: dict | None = None
        self.load_encoder()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def load_encoder(self, **kwargs) -> None:
        from ._loaders.uni3d_loader import load_uni3d, UNI3D_CONFIGS

        model_size = (
            self.teacher_size
            if self.teacher_size.startswith("uni3d_")
            else f"uni3d_{self.teacher_size}"
        )
        self.encoder, self.uni3d_embed_dim = load_uni3d(
            model_size=model_size,
            device="cuda" if torch.cuda.is_available() else "cpu",
            auto_download=True,
        )
        self.uni3d_config = UNI3D_CONFIGS[model_size]
        self._freeze_model(self.encoder)
        print(f"Loaded Uni3D teacher: {model_size} (embed_dim={self.uni3d_embed_dim})")

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------
    def extract_features(self, batch: dict) -> tuple:
        from ._loaders.uni3d_loader import propagate_features

        pointclouds = batch["pointclouds_gt"]          # (B, N, 3)
        points_per_part = batch["points_per_part"]     # (B, P)
        B, N, C = pointclouds.shape
        n_valid_parts = points_per_part != 0

        device = pointclouds.device
        coords = pointclouds.to(device)

        # Uni3D expects (B, N, 6) with xyz + color
        # Use 0.4 for color since we don't have color information (see Uni3D Repo's issue)
        color = torch.ones_like(coords).to(device) * 0.4
        pc = torch.cat([coords, color], dim=-1).float()  # (B, N, 6)

        with torch.inference_mode():
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                result = self.encoder.encode_pc(
                    pc,
                    return_point_features=True,
                    intermediate_layers=None,
                )

            features = result["point_features"].float()

        return features, result, n_valid_parts

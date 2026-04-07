import torch
import torch.nn as nn

from .base import BaseTeacher


class Find3DTeacher(BaseTeacher):
    """Frozen Find3D teacher encoder.

    Args:
        repr_dim: Output representation dimension.
        teacher_size: Model identifier, e.g. ``"find3d_base"``.
            If the string does not already start with ``find3d_``, the prefix is prepended.
    """

    def __init__(self, repr_dim: int, teacher_size: str = "find3d_base"):
        super().__init__(repr_dim=repr_dim)
        self.teacher_size = teacher_size
        self.find3d_feat_dim: int | None = None
        self.find3d_config: dict | None = None
        self.load_encoder()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def load_encoder(self, **kwargs) -> None:
        from ._loaders.find3d_loader import load_find3d, FIND3D_CONFIGS

        model_size = (
            self.teacher_size
            if self.teacher_size.startswith("find3d_")
            else f"find3d_{self.teacher_size}"
        )
        self.encoder, self.find3d_feat_dim = load_find3d(
            model_size=model_size,
            device="cuda" if torch.cuda.is_available() else "cpu",
            auto_download=True,
        )
        self.find3d_config = FIND3D_CONFIGS[model_size]
        self._freeze_model(self.encoder)
        print(f"Loaded Find3D teacher: {model_size} (feat_dim={self.find3d_feat_dim})")

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------
    def extract_features(self, batch: dict) -> tuple:
        from ._loaders.find3d_loader import (
            preprocess_batch_for_find3d,
            propagate_features_knn,
            _patch_spconv_for_arm,
        )

        pointclouds = batch["pointclouds_gt"]          # (B, N, 3)
        normals = batch["pointclouds_normals_gt"]      # (B, N, 3)
        points_per_part = batch["points_per_part"]     # (B, P)
        B, N, C = pointclouds.shape
        n_valid_parts = points_per_part != 0
        device = pointclouds.device

        grid_size = self.find3d_config.get("grid_size", 0.02)
        batched_data, xyz_full_norm_list, xyz_sub_list = preprocess_batch_for_find3d(
            pointclouds, normals, grid_size=grid_size,
        )
        batched_data = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batched_data.items()
        }

        with torch.inference_mode():
            with torch.autocast(device_type=device.type, enabled=False):
                try:
                    feat_out = self.encoder(x=batched_data)
                except RuntimeError as e:
                    err_msg = str(e)
                    if "suitable algorithm" in err_msg or "all_profile_res" in err_msg:
                        if _patch_spconv_for_arm(self.encoder):
                            print("Retrying Find3D forward with Native spconv algorithm...")
                            feat_out = self.encoder(x=batched_data)
                        else:
                            raise RuntimeError(
                                "Find3D uses spconv, which failed to find a valid CUDA algorithm. "
                                "This often happens on aarch64/ARM (e.g. Grace). "
                                "Use another teacher: concerto or uni3d, "
                                "or build spconv from source for your CUDA arch."
                            ) from e
                    else:
                        raise

            feat_out = feat_out.float()

        # Split by sample and propagate back to full resolution
        offsets = batched_data["offset"].cpu().tolist()
        all_features = []
        prev = 0
        for i in range(B):
            cur_offset = offsets[i]
            feat_i = feat_out[prev:cur_offset]
            xyz_sub_i = xyz_sub_list[i].to(device).float()
            xyz_full_i = torch.FloatTensor(xyz_full_norm_list[i]).to(device)

            feat_full_i = propagate_features_knn(
                xyz_full_i, xyz_sub_i, feat_i, k=3,
            )
            all_features.append(feat_full_i)
            prev = cur_offset

        features = torch.stack(all_features, dim=0)  # (B, N, 768)
        return features, None, n_valid_parts

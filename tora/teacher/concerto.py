import torch
import torch.nn as nn

from .base import BaseTeacher


class ConcertoTeacher(BaseTeacher):
    """Frozen Concerto/Sonata teacher encoder.

    Args:
        repr_dim: Output representation dimension.
        teacher_size: Model identifier, e.g. ``"concerto_large"`` or ``"sonata_large"``.
            If the string does not already start with ``concerto_`` or ``sonata``,
            the prefix ``concerto_`` is prepended automatically.
    """

    def __init__(self, repr_dim: int, teacher_size: str = "concerto_large", upcast_k: int = 2):
        super().__init__(repr_dim=repr_dim)
        self.teacher_size = teacher_size
        self.upcast_k = upcast_k
        self.transform = None
        self.load_encoder()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def load_encoder(self, **kwargs) -> None:
        from ..extern.concerto import concerto
        from ._loaders.find3d_loader import _patch_spconv_for_arm

        model_name = (
            self.teacher_size
            if self.teacher_size.startswith("concerto_") or self.teacher_size.startswith("sonata")
            else f"concerto_{self.teacher_size}"
        )
        repo_id = "facebook/sonata" if model_name.startswith("sonata") else "Pointcept/Concerto"

        self.encoder = concerto.load(model_name, repo_id=repo_id)
        self.transform = concerto.transform.default()
        self._freeze_model(self.encoder)
        _patch_spconv_for_arm(self.encoder)
        print(
            f"Loaded {'Sonata' if model_name.startswith('sonata') else 'Concerto'} "
            f"teacher: {model_name}"
        )

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------
    def extract_features(self, batch: dict) -> tuple:
        pointclouds = batch["pointclouds_gt"]          # (B, N, 3)
        normals = batch["pointclouds_normals_gt"]      # (B, N, 3)
        points_per_part = batch["points_per_part"]     # (B, P)
        scales = batch["scales"]                       # (B,)
        B, N, C = pointclouds.shape
        n_valid_parts = points_per_part != 0

        device = pointclouds.device
        coords = pointclouds.to(device) * scales.to(device).view(B, 1, 1)
        coords = coords.view(-1, C)
        color = torch.zeros_like(coords).to(device)
        normals_flat = normals.view(-1, C).to(device)

        obj_offset = torch.arange(1, B + 1, device=device) * N
        part_counts = points_per_part[n_valid_parts].view(-1).to(device)
        part_segment = torch.arange(0, len(part_counts), device=device).repeat_interleave(part_counts)

        point = self.transform({
            "coord": coords.cpu().numpy(),
            "color": color.cpu().numpy(),
            "normal": normals_flat.cpu().numpy(),
            "offset": obj_offset.cpu().numpy(),
            "segment": part_segment.cpu().numpy(),
        })

        if isinstance(point, dict):
            point = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in point.items()}
        else:
            point = point.to(device)

        with torch.inference_mode():
            with torch.autocast(device_type=device.type, enabled=False):
                point = self.encoder(point)
                point = self._upcast_concerto(point, k=self.upcast_k)
                point["normal"] = normals_flat
                features = point.feat[point.inverse].reshape(B, N, -1)

        return features, point, n_valid_parts

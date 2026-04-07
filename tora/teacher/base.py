import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class BaseTeacher(nn.Module, ABC):
    """Base class for frozen teacher encoders."""

    def __init__(self, repr_dim: int):
        super().__init__()
        self.repr_dim = repr_dim
        self.encoder = None

    @abstractmethod
    def load_encoder(self, **kwargs) -> None:
        """Load the frozen encoder weights."""
        ...

    @abstractmethod
    def extract_features(self, batch: dict) -> tuple[torch.Tensor, ...]:
        """Extract point features using the encoder.

        Args:
            batch: Data dictionary with at least pointclouds_gt, pointclouds_normals_gt,
                   points_per_part, scales.

        Returns:
            Tuple of (features, extra_info, n_valid_parts) where features is (B, N, repr_dim).
        """
        ...

    @staticmethod
    def _freeze_model(model: nn.Module):
        model.eval()
        for module in model.modules():
            module.eval()
        for param in model.parameters():
            param.requires_grad = False

    @staticmethod
    def _upcast_concerto(point, k: int = 2):
        """Upcast Concerto pooling hierarchy."""
        for _ in range(k):
            assert "pooling_parent" in point.keys()
            assert "pooling_inverse" in point.keys()
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
            point = parent
        while "pooling_parent" in point.keys():
            assert "pooling_inverse" in point.keys()
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            parent.feat = point.feat[inverse]
            point = parent
        return point

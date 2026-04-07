import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class BaseProjector(nn.Module, ABC):
    """Base class for projectors that map intermediate representations to teacher feature space.

    Returns:
        Tuple of (projected_features, subsample_idx_or_None)
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

    @abstractmethod
    def forward(self, x: torch.Tensor, data_dict: dict) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Project intermediate representations.

        Args:
            x: (B, N, in_dim) intermediate features from flow model.
            data_dict: Dictionary with at least "pointclouds_gt" (B, N, 3).

        Returns:
            projected: (B, M, out_dim) where M <= N.
            subsample_idx: (B, M) indices or None.
        """
        ...

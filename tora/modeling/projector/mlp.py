import torch
import torch.nn as nn

from .base import BaseProjector


class MLPProjector(BaseProjector):
    """Simple 3-layer MLP projector.

    Ported from PointCloudTeacher's ``mlp`` head:
        Linear -> SiLU -> Linear -> SiLU -> Linear

    Args:
        in_dim: Input feature dimension.
        out_dim: Output representation dimension.
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__(in_dim, out_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.SiLU(),
            nn.Linear(in_dim, in_dim),
            nn.SiLU(),
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, x: torch.Tensor, data_dict: dict) -> tuple[torch.Tensor, None]:
        """
        Args:
            x: (B, N, in_dim) intermediate features.
            data_dict: Unused; kept for interface consistency.

        Returns:
            projected: (B, N, out_dim).
            subsample_idx: None.
        """
        return self.mlp(x), None

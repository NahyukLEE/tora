import torch
import torch.nn.functional as F

from .base import BaseAlignmentLoss


class CosineLoss(BaseAlignmentLoss):
    """Cosine similarity alignment loss.

    Computes ``1 - cosine_similarity`` averaged over all points and batch items.
    This is the default REPA projection loss adapted for 3D point features.
    """

    def __init__(self, lmbda: float = 0.5):
        super().__init__(lmbda=lmbda)

    def compute(self, repr_pred: torch.Tensor, repr_t: torch.Tensor, **kwargs) -> torch.Tensor:
        # repr_pred, repr_t: (B, M, D)
        return (1.0 - F.cosine_similarity(repr_pred, repr_t, dim=-1)).mean()

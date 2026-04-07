import torch
import torch.nn.functional as F

from .base import BaseAlignmentLoss


class NTXentLoss(BaseAlignmentLoss):
    """NT-Xent (Normalized Temperature-scaled Cross Entropy) alignment loss.

    Pools per-point features to per-sample embeddings via global average
    pooling, then treats (repr_pred[i], repr_t[i]) as a positive pair
    within the batch. The symmetric InfoNCE objective encourages each
    student embedding to be closest to its corresponding teacher embedding.

    Reference:
        Chen et al., "A Simple Framework for Contrastive Learning of Visual
        Representations", ICML 2020.

    Args:
        lmbda: Global scaling factor.
        temperature: Temperature parameter controlling the sharpness of the
            softmax distribution over similarities.
    """

    def __init__(self, lmbda: float = 0.5, temperature: float = 0.07):
        super().__init__(lmbda=lmbda)
        self.temperature = temperature

    def compute(self, repr_pred: torch.Tensor, repr_t: torch.Tensor, **kwargs) -> torch.Tensor:
        # Global average pool: (B, N, D) -> (B, D), then L2-normalise
        z_pred = F.normalize(repr_pred.mean(dim=1), dim=-1)  # (B, D)
        z_t = F.normalize(repr_t.mean(dim=1), dim=-1)        # (B, D)

        # Cosine similarity matrix scaled by temperature: (B, B)
        sim = z_pred @ z_t.T / self.temperature
        labels = torch.arange(sim.shape[0], device=sim.device)

        # Symmetric cross-entropy
        loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2
        return loss

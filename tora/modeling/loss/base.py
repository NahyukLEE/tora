import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class BaseAlignmentLoss(nn.Module, ABC):
    """Base class for alignment losses.

    All alignment losses follow the same pattern:
      1. Detach and clone the teacher target to stop gradients.
      2. Compute the raw loss via the subclass's ``compute`` method.
      3. Scale by ``lmbda``.
    """

    def __init__(self, lmbda: float = 0.5):
        super().__init__()
        self.lmbda = lmbda

    @abstractmethod
    def compute(self, repr_pred: torch.Tensor, repr_t: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compute the raw loss (before lambda scaling).

        Args:
            repr_pred: (B, M, D) projected student features.
            repr_t: (B, M, D) teacher features (already detached in ``forward``).
            **kwargs: Additional arguments needed by specific loss types.

        Returns:
            Scalar loss tensor.
        """
        ...

    def forward(self, repr_pred: torch.Tensor, repr_t: torch.Tensor, **kwargs) -> torch.Tensor:
        target = repr_t.clone().detach()
        loss = self.compute(repr_pred, target, **kwargs)
        return self.lmbda * loss

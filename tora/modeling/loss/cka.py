import torch

from .base import BaseAlignmentLoss


class CKALoss(BaseAlignmentLoss):
    """Centered Kernel Alignment (CKA) loss.

    CKA measures the similarity between two representation spaces using
    their centered kernel (Gram) matrices.  The loss is ``1 - CKA`` so that
    minimising it aligns the two spaces.

    Reference:
        Kornblith et al., "Similarity of Neural Network Representations
        Revisited", ICML 2019.

    Args:
        lmbda: Global scaling factor.
        max_points: Subsample to at most this many points to keep the kernel
            computation tractable.
    """

    def __init__(self, lmbda: float = 0.5, max_points: int = 1024):
        super().__init__(lmbda=lmbda)
        self.max_points = max_points

    @staticmethod
    def _center_gram(G: torch.Tensor) -> torch.Tensor:
        """Center a Gram matrix using the centering matrix H = I - 1/n 11^T."""
        n = G.shape[-1]
        mean_rows = G.mean(dim=-1, keepdim=True)
        mean_cols = G.mean(dim=-2, keepdim=True)
        mean_all = G.mean(dim=(-2, -1), keepdim=True)
        return G - mean_rows - mean_cols + mean_all

    @staticmethod
    def _hsic(K: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        """Empirical HSIC using centered Gram matrices."""
        # K, L: (B, M, M) already centered
        return (K * L).sum(dim=(-2, -1)) / (K.shape[-1] - 1) ** 2

    def compute(self, repr_pred: torch.Tensor, repr_t: torch.Tensor, **kwargs) -> torch.Tensor:
        B, M, D = repr_pred.shape

        # Subsample if necessary
        if M > self.max_points:
            idx = torch.randperm(M, device=repr_pred.device)[:self.max_points]
            repr_pred = repr_pred[:, idx]
            repr_t = repr_t[:, idx]

        # Linear kernels (Gram matrices)
        K = torch.bmm(repr_pred, repr_pred.transpose(1, 2))  # (B, M, M)
        L = torch.bmm(repr_t, repr_t.transpose(1, 2))        # (B, M, M)

        # Center
        K_c = self._center_gram(K)
        L_c = self._center_gram(L)

        # CKA = HSIC(K,L) / sqrt(HSIC(K,K) * HSIC(L,L))
        hsic_kl = self._hsic(K_c, L_c)
        hsic_kk = self._hsic(K_c, K_c)
        hsic_ll = self._hsic(L_c, L_c)

        cka = hsic_kl / (torch.sqrt(hsic_kk * hsic_ll) + 1e-8)
        loss = (1.0 - cka).mean()
        return loss

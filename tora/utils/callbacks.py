"""Lightning callbacks for training utilities and layer-wise analysis."""

import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

import lightning as L
import torch
from lightning.pytorch.callbacks import Callback

from .point_clouds import get_part_ids

logger = logging.getLogger(__name__)


# ===========================================================================
# Training callbacks
# ===========================================================================

class NaNTracker(Callback):
    """Detect NaN/Inf in loss and gradients during training.

    The loss check runs every step (negligible cost — single scalar).
    The gradient check runs every ``check_grad_every_n_steps`` steps to
    avoid scanning all gradient memory on every iteration.

    Args:
        check_grad_every_n_steps: How often to scan gradients. 0 = never.
            Default 50 gives early detection without measurable overhead.
    """

    def __init__(self, check_grad_every_n_steps: int = 50):
        super().__init__()
        self.check_grad_every_n_steps = check_grad_every_n_steps

    def on_after_backward(self, trainer, pl_module):
        if self.check_grad_every_n_steps <= 0:
            return
        if trainer.global_step % self.check_grad_every_n_steps != 0:
            return
        for name, param in pl_module.named_parameters():
            if param.grad is not None:
                if not torch.isfinite(param.grad).all():
                    print(f"\n[!] NaN Gradients detected in: {name}", flush=True)
                    raise ValueError(f"Gradient collapse in {name} at step {trainer.global_step}")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs
        if not torch.isfinite(loss):
            print(f"\n[!] NaN Loss detected at step {trainer.global_step}", flush=True)
            raise ValueError("Loss exploded to NaN")


# ===========================================================================
# Attention-to-parts helpers
# ===========================================================================

def _normalize_correspondence(part_corr: torch.Tensor, points_per_part: torch.Tensor) -> torch.Tensor:
    """Normalize part correspondence by part sizes.

    Args:
        part_corr: (B, P, P) aggregated attention sums.
        points_per_part: (B, P).

    Returns:
        (B, P, P) normalized correspondence.
    """
    counts = points_per_part.unsqueeze(-1).float()
    norm_factor = torch.bmm(counts, counts.transpose(1, 2))
    mask = (norm_factor > 0).float()
    safe_norm = norm_factor + (1 - mask)
    return (part_corr / safe_norm) * mask


def agg_attn_to_parts(attn_map: torch.Tensor, points_per_part: torch.Tensor) -> torch.Tensor:
    """Aggregate token-level attention to part-level correspondence.

    Args:
        attn_map: (B, H, N, N) or (B, N, N) attention weights.
        points_per_part: (B, P).

    Returns:
        (B, P, P) part correspondence matrix.
    """
    P = points_per_part.shape[1]
    device = attn_map.device
    points_per_part = points_per_part.to(device)

    if attn_map.dim() == 4:
        B, H, N, _ = attn_map.shape
        attn_avg = attn_map.mean(dim=1)
    else:
        B, N, _ = attn_map.shape
        attn_avg = attn_map

    cum_sums = torch.cumsum(points_per_part, dim=1)
    token_indices = torch.arange(N, device=device).expand(B, N)
    part_ids = torch.searchsorted(cum_sums, token_indices, right=True).clamp(0, P - 1)

    col_reduced = torch.zeros(B, N, P, device=device)
    col_idx = part_ids.unsqueeze(1).expand(B, N, N)
    col_reduced.scatter_add_(2, col_idx, attn_avg)

    part_corr = torch.zeros(B, P, P, device=device)
    row_idx = part_ids.unsqueeze(-1).expand(B, N, P)
    part_corr.scatter_add_(1, row_idx, col_reduced)

    return _normalize_correspondence(part_corr, points_per_part)


# ===========================================================================
# Analysis callbacks (test-time)
# ===========================================================================

class AttentionExtractionCallback(Callback):
    """Extract per-layer attention maps and save part correspondences.

    Hooks into the flow model's QKV projections during test, computes
    token-level attention, aggregates to part-level correspondence, and saves
    heatmap plots per batch.

    Args:
        save_dir: Output directory (default: ``trainer.log_dir / attention``).
        attn_type: Which attention to extract (``"global"`` or ``"part"``).
        save_plots: Whether to render matplotlib heatmaps.
        save_tensors: Whether to save raw correspondence tensors as ``.pt``.
        max_samples: Max samples to process (None = all).
    """

    def __init__(
        self,
        save_dir: Optional[str] = None,
        attn_type: str = "global",
        save_plots: bool = True,
        save_tensors: bool = False,
        max_samples: Optional[int] = None,
    ):
        super().__init__()
        self.save_dir = save_dir
        self.attn_type = attn_type
        self.save_plots = save_plots
        self.save_tensors = save_tensors
        self.max_samples = max_samples
        self._out_dir = None

    def setup(self, trainer: L.Trainer, pl_module: L.LightningModule, stage: str) -> None:
        if stage == "test":
            base = self.save_dir or os.path.join(trainer.log_dir, "attention")
            self._out_dir = Path(base)
            self._out_dir.mkdir(parents=True, exist_ok=True)

    def on_test_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: dict,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if self._out_dir is None:
            return

        from ..modeling.flow_model.utils import AttentionMapExtractor

        pcd = batch["pointclouds"]
        B, N, _ = pcd.shape
        points_per_part = batch["points_per_part"]

        extractor = AttentionMapExtractor(model=pl_module.flow_model)
        extractor.register_hooks()
        with torch.no_grad():
            pl_module.forward(batch)
        attn_maps = extractor.compute_attention_maps(B, N)
        extractor.remove_hooks()

        sorted_layers = sorted(k[0] for k in attn_maps if k[1] == self.attn_type)
        corrs = [agg_attn_to_parts(attn_maps[(l, self.attn_type)], points_per_part) for l in sorted_layers]
        correspondences = torch.stack(corrs, dim=1)  # (B, L, P, P)

        mean_attn = extractor.get_mean_attention(self.attn_type)
        mean_corr = agg_attn_to_parts(mean_attn, points_per_part)

        if self.save_tensors:
            torch.save({
                "correspondences": correspondences.cpu(),
                "mean_correspondence": mean_corr.cpu(),
                "points_per_part": points_per_part.cpu(),
            }, self._out_dir / f"batch_{batch_idx:04d}.pt")

        if self.save_plots:
            self._plot_correspondences(correspondences, points_per_part, batch_idx)

        n = min(B, self.max_samples) if self.max_samples else B
        logger.info(f"[batch {batch_idx}] Extracted attention for {n} samples, {len(sorted_layers)} layers")

    def _plot_correspondences(self, correspondences, points_per_part, batch_idx):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        B, L, P, _ = correspondences.shape
        points_per_part = points_per_part.to(correspondences.device)

        n = min(B, self.max_samples) if self.max_samples else B
        for b in range(n):
            cols = min(L, 4)
            rows = (L + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4), squeeze=False)
            fig.suptitle(f"Part Correspondence (batch {batch_idx}, sample {b})")

            for l in range(L):
                r, c = divmod(l, cols)
                ax = axes[r, c]
                mask = points_per_part[b] > 0
                matrix = correspondences[b, l][mask][:, mask].detach().cpu().numpy()
                im = ax.imshow(matrix, cmap="viridis")
                ax.set_title(f"Layer {l}")
                ax.set_xlabel("Target")
                ax.set_ylabel("Source")
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            for i in range(L, rows * cols):
                r, c = divmod(i, cols)
                axes[r, c].axis("off")

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(self._out_dir / f"batch{batch_idx:04d}_sample{b:02d}.png")
            plt.close(fig)


class FeatureExtractionCallback(Callback):
    """Extract intermediate representations from DiT layers and optionally teacher features.

    Saves per-layer representations and teacher features as ``.pt`` files.
    Optionally computes per-layer alignment (cosine similarity) against the teacher.

    Args:
        save_dir: Output directory (default: ``trainer.log_dir / features``).
        layers: List of layer indices to extract (None = all).
        compute_alignment: If True, compute cosine sim between each layer and teacher.
        save_tensors: Whether to save raw representation tensors.
        max_samples: Max samples to process per batch.
    """

    def __init__(
        self,
        save_dir: Optional[str] = None,
        layers: Optional[list[int]] = None,
        compute_alignment: bool = True,
        save_tensors: bool = True,
        max_samples: Optional[int] = None,
    ):
        super().__init__()
        self.save_dir = save_dir
        self.layers = layers
        self.compute_alignment = compute_alignment
        self.save_tensors = save_tensors
        self.max_samples = max_samples
        self._out_dir = None
        self._alignment_results = []

    def setup(self, trainer: L.Trainer, pl_module: L.LightningModule, stage: str) -> None:
        if stage == "test":
            base = self.save_dir or os.path.join(trainer.log_dir, "features")
            self._out_dir = Path(base)
            self._out_dir.mkdir(parents=True, exist_ok=True)

    def on_test_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: dict,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if self._out_dir is None:
            return

        from ..modeling.flow_model.utils import RepresentationExtractor

        extractor = RepresentationExtractor(
            model=pl_module.flow_model,
            layers=self.layers,
            detach=True,
            to_cpu=True,
        )
        extractor.register_hooks()
        with torch.no_grad():
            pl_module.forward(batch)
        representations = extractor.get_representations()
        extractor.remove_hooks()

        teacher_feat = None
        if pl_module.teacher is not None:
            with torch.no_grad():
                teacher_feat = pl_module.teacher.extract_features(batch)[0].cpu()

        if self.save_tensors:
            save_dict = {"representations": {k: v for k, v in representations.items()}}
            if teacher_feat is not None:
                save_dict["teacher_features"] = teacher_feat
            torch.save(save_dict, self._out_dir / f"batch_{batch_idx:04d}.pt")

        if self.compute_alignment and teacher_feat is not None:
            batch_alignment = {}
            for layer_idx in sorted(representations.keys()):
                layer_repr = representations[layer_idx]
                r = torch.nn.functional.normalize(layer_repr.float(), dim=-1)
                t = torch.nn.functional.normalize(teacher_feat.float(), dim=-1)
                cos_sim = (r * t).sum(dim=-1).mean().item()
                batch_alignment[layer_idx] = cos_sim
            self._alignment_results.append(batch_alignment)

        logger.info(f"[batch {batch_idx}] Extracted {len(representations)} layer representations")

    def on_test_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if not self._alignment_results or self._out_dir is None:
            return

        all_layers = sorted(self._alignment_results[0].keys())
        avg = {}
        for l in all_layers:
            vals = [r[l] for r in self._alignment_results if l in r]
            avg[l] = sum(vals) / len(vals)

        with open(self._out_dir / "layer_alignment.json", "w") as f:
            json.dump({str(k): v for k, v in avg.items()}, f, indent=2)

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            layers = sorted(avg.keys())
            values = [avg[l] for l in layers]
            plt.figure(figsize=(8, 5))
            plt.plot(layers, values, marker="o", linestyle="-")
            plt.xlabel("Layer Index")
            plt.ylabel("Mean Cosine Similarity with Teacher")
            plt.title("Layer-Teacher Alignment")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(self._out_dir / "layer_alignment.png")
            plt.close()
            logger.info(f"Saved layer alignment plot to {self._out_dir / 'layer_alignment.png'}")
        except ImportError:
            pass

        self._alignment_results.clear()


class SpatialMetricsCallback(Callback):
    """Compute spatial representation quality metrics for the teacher's features.

    Runs the spatial metrics suite (LDS, boundary contrast, part silhouette,
    effective rank, pose discrimination) on teacher features during test batches.

    Args:
        save_dir: Output directory (default: ``trainer.log_dir / spatial_metrics``).
        metrics: List of metric names to compute. None = all available.
        k_local: k for LDS and boundary contrast.
        max_batches: Stop after this many batches (None = all).
    """

    AVAILABLE_METRICS = [
        "lds", "boundary_contrast", "part_silhouette",
        "effective_rank", "pose_discrimination",
    ]

    def __init__(
        self,
        save_dir: Optional[str] = None,
        metrics: Optional[list[str]] = None,
        k_local: int = 6,
        max_batches: Optional[int] = None,
    ):
        super().__init__()
        self.save_dir = save_dir
        self.metric_names = metrics or self.AVAILABLE_METRICS
        self.k_local = k_local
        self.max_batches = max_batches
        self._out_dir = None
        self._results: dict[str, list[torch.Tensor]] = {}

    def setup(self, trainer: L.Trainer, pl_module: L.LightningModule, stage: str) -> None:
        if stage == "test":
            base = self.save_dir or os.path.join(trainer.log_dir, "spatial_metrics")
            self._out_dir = Path(base)
            self._out_dir.mkdir(parents=True, exist_ok=True)
            self._results = {m: [] for m in self.metric_names}

    def on_test_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: dict,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if self._out_dir is None:
            return
        if self.max_batches is not None and batch_idx >= self.max_batches:
            return
        if pl_module.teacher is None:
            return

        from ..eval.spatial import (
            metric_lds_3d,
            metric_boundary_contrast,
            metric_part_silhouette,
            metric_effective_rank,
            metric_pose_discrimination,
        )

        with torch.no_grad():
            features = pl_module.teacher.extract_features(batch)[0]

        coords = batch["pointclouds_gt"]
        points_per_part = batch["points_per_part"]
        B, N, _ = coords.shape
        part_ids = get_part_ids(points_per_part, N)

        if "lds" in self.metric_names:
            self._results["lds"].append(
                metric_lds_3d(features, coords, k_local=self.k_local).cpu()
            )
        if "boundary_contrast" in self.metric_names:
            self._results["boundary_contrast"].append(
                metric_boundary_contrast(features, coords, part_ids, k=self.k_local).cpu()
            )
        if "part_silhouette" in self.metric_names:
            self._results["part_silhouette"].append(
                metric_part_silhouette(features, part_ids).cpu()
            )
        if "effective_rank" in self.metric_names:
            self._results["effective_rank"].append(
                metric_effective_rank(features).cpu()
            )
        if "pose_discrimination" in self.metric_names:
            features_deformed = None
            if "pointclouds" in batch and "pointclouds_normals" in batch:
                batch_def = {
                    **batch,
                    "pointclouds_gt": batch["pointclouds"],
                    "pointclouds_normals_gt": batch["pointclouds_normals"],
                }
                with torch.no_grad():
                    features_deformed = pl_module.teacher.extract_features(batch_def)[0]
            if features_deformed is not None:
                self._results["pose_discrimination"].append(
                    metric_pose_discrimination(features, features_deformed).cpu()
                )

        logger.info(f"[batch {batch_idx}] Computed spatial metrics for {B} samples")

    def on_test_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if self._out_dir is None or not any(self._results.values()):
            return

        summary = {}
        for name, vals in self._results.items():
            if not vals:
                continue
            cat = torch.cat(vals)
            summary[name] = {
                "mean": cat.mean().item(),
                "std": cat.std().item(),
                "n": len(cat),
            }
            logger.info(f"  {name}: {summary[name]['mean']:.4f} +/- {summary[name]['std']:.4f}")

        with open(self._out_dir / "spatial_metrics.json", "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved spatial metrics to {self._out_dir / 'spatial_metrics.json'}")

        self._results = {m: [] for m in self.metric_names}

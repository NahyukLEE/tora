"""Visualization utilities for point cloud assembly."""

from pathlib import Path
import logging
from typing import Any, Optional

import lightning as L
import torch
from lightning.pytorch.callbacks import Callback

from .utils.render import visualize_point_clouds, img_tensor_to_pil, part_ids_to_colors, probs_to_colors, get_pca_colors, get_similarity_colors
from .utils.point_clouds import ppp_to_ids

logger = logging.getLogger("Visualizer")


class VisualizationCallback(Callback):
    """Base Lightning callback for visualizing point clouds during evaluation."""

    def __init__(
        self,
        save_dir: Optional[str] = None,
        renderer: str = "mitsuba",
        colormap: str = "default",
        scale_to_original_size: bool = False,
        center_points: bool = False,
        image_size: int = 512,
        point_radius: float = 0.015,
        camera_dist: float = 4.0,
        camera_elev: float = 20.0,
        camera_azim: float = 30.0,
        camera_fov: float = 45.0,
        max_samples_per_batch: Optional[int] = None,
    ):
        """Initialize base visualization callback.

        Args:
            save_dir (str): Directory to save images. If None, uses trainer.log_dir/visualizations.
            renderer (str): Renderer to use, can be "mitsuba" or "pytorch3d". Default: "mitsuba".
            colormap (str): Colormap to use. Default: "default".
            scale_to_original_size (bool): If True, scales the point clouds to the original size.
                Otherwise, keep the scaling, i.e. [-1, 1]. Default: False.
            center_points: If True, centers the point cloud around the origin. Default: False.
            image_size (int): Output image resolution (square). Default: 512.
            point_radius (float): Radius of each rendered point in world units. Default: 0.015.
            camera_dist (float): Distance (m) of camera from origin. Default: 4.0.
            camera_elev (float): Elevation angle (deg). Default: 20.0.
            camera_azim (float): Azimuth angle (deg). Default: 30.0.
            camera_fov (float): Field of view (deg). Default: 45.0.
            max_samples_per_batch (int): Maximum samples to visualize per batch. None means all.
        """
        super().__init__()
        self.save_dir = save_dir
        self.renderer = renderer
        self.colormap = colormap
        self.scale_to_original_size = scale_to_original_size
        self.max_samples_per_batch = max_samples_per_batch

        self.vis_dir = None
        self._vis_kwargs = {
            "renderer": self.renderer,
            "center_points": center_points,
            "image_size": image_size,
            "point_radius": point_radius,
            "camera_dist": camera_dist,
            "camera_elev": camera_elev,
            "camera_azim": camera_azim,
            "camera_fov": camera_fov,
        }

    def setup(self, trainer: L.Trainer, pl_module: L.LightningModule, stage: str) -> None:
        if stage == "test":
            if self.save_dir is None:
                self.vis_dir = Path(trainer.log_dir) / "visualizations"
            else:
                self.vis_dir = Path(self.save_dir) / "visualizations"
            self.vis_dir.mkdir(parents=True, exist_ok=True)

    def _save_sample_images(
        self,
        points: torch.Tensor,
        colors: torch.Tensor,
        sample_name: str,
    ) -> None:
        """Save visualization images for a single sample.

        Args:
            points (torch.Tensor): Point cloud of shape (N, 3).
            colors (torch.Tensor): Colors of shape (N, 3).
            sample_name (str): sample name for filename.
        """
        try:
            image = visualize_point_clouds(
                points=points,
                colors=colors,
                **self._vis_kwargs
            )
            image_pil = img_tensor_to_pil(image)
            sample_name = sample_name.replace('/', '_')
            image_pil.save(self.vis_dir / f"{sample_name}.png")
        except Exception as e:
            logger.error(f"Error saving visualization for sample {sample_name}: {e}")

    def on_test_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Override this method in subclasses for specific visualization logic."""
        raise NotImplementedError("Subclasses must implement on_test_batch_end")


class FlowVisualizationCallback(VisualizationCallback):
    """Visualization callback for rectified point flow models."""

    def __init__(
        self,
        save_trajectory: bool = True,
        trajectory_gif_fps: int = 25,
        trajectory_gif_pause_last_frame: float = 1.0,
        **kwargs
    ):
        """Initialize flow visualization callback.

        Args:
            save_trajectory (bool): Whether to save trajectory as GIF. Default: True.
            trajectory_gif_fps (int): Frames per second for the GIF.
            trajectory_gif_pause_last_frame (float): Pause time for the last frame in seconds.
            **kwargs: Additional arguments passed to base class.
        """
        super().__init__(**kwargs)
        self.save_trajectory = save_trajectory
        self.trajectory_gif_fps = trajectory_gif_fps
        self.trajectory_gif_pause_last_frame = trajectory_gif_pause_last_frame

    def _save_trajectory_gif(
        self,
        trajectory: torch.Tensor,
        colors: torch.Tensor,
        sample_name: str,
    ) -> None:
        """Save trajectory as GIF.

        Args:
            trajectory: Point clouds representing the trajectory steps of shape (num_steps, N, 3).
            colors: Colors of shape (N, 3). Same for all trajectory steps.
            sample_name (str): sample name for filename.
        """
        try:
            gif_path = self.vis_dir / f"{sample_name}.gif"

            # Render trajectory steps
            rendered_images = visualize_point_clouds(
                points=trajectory,                                          # (num_steps, N, 3)
                colors=colors,                                              # (N, 3)
                **self._vis_kwargs,
            )                                                               # (num_steps, H, W, 3)
            frames = []
            num_steps = trajectory.shape[0]
            for step in range(num_steps):
                frame_pil = img_tensor_to_pil(rendered_images[step])        # (H, W, 3)
                frames.append(frame_pil)

            # Frame duration and pause on last frame in ms
            duration = int(1000 / self.trajectory_gif_fps)
            durations = [duration] * len(frames)
            durations[-1] = int(duration + self.trajectory_gif_pause_last_frame * 1000)

            frames[0].save(
                gif_path,
                save_all=True,
                append_images=frames[1:],
                duration=durations,
                loop=0,  # Infinite loop
                optimize=True
            )
        except Exception as e:
            logger.error(f"Error saving trajectory GIF for sample {sample_name}: {e}")

    def on_test_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Save flow visualizations at the end of each test batch."""
        if self.vis_dir is None:
            return

        points_per_part = batch["points_per_part"]                            # (bs, max_parts)
        B, _ = points_per_part.shape
        part_ids = ppp_to_ids(points_per_part)                                # (bs, N)
        pts = batch["pointclouds"].view(B, -1, 3)                             # (bs, N, 3)
        pts_gt = batch["pointclouds_gt"].view(B, -1, 3)                       # (bs, N, 3)

        # K generations
        trajectories_list = outputs['trajectories']                           # (K, num_steps, num_points, 3)
        K = len(trajectories_list)
        pointclouds_pred_list = [traj[-1].view(B, -1, 3) for traj in trajectories_list]

        if self.scale_to_original_size:
            scale = batch["scale"][:, 0]                                      # (bs,)
            pts = pts * scale[:, None, None]                                  # (bs, N, 3)
            pointclouds_pred_list = [pred * scale[:, None, None] for pred in pointclouds_pred_list]

        for i in range(B):
            dataset_name = batch["dataset_name"][i]
            sample_name = f"{dataset_name}_sample{int(batch['index'][i]):05d}"

            colors = part_ids_to_colors(
                part_ids[i], colormap=self.colormap, part_order="random"
            )
            self._save_sample_images(
                points=pts[i],
                colors=colors,
                sample_name=f"{sample_name}_input",
            )
            self._save_sample_images(
                points=pts_gt[i],
                colors=colors,
                sample_name=f"{sample_name}_gt",
            )
            for n in range(K):
                pointclouds_pred = pointclouds_pred_list[n]
                self._save_sample_images(
                    points=pointclouds_pred[i],
                    colors=colors,
                    sample_name=f"{sample_name}_generation{n+1:02d}",
                )

                if self.save_trajectory:
                    trajectory = trajectories_list[n]
                    num_steps = trajectory.shape[0]
                    trajectory = trajectory.reshape(num_steps, B, -1, 3).permute(1, 0, 2, 3)  # (bs, num_steps, N, 3)
                    if self.scale_to_original_size:
                        trajectory = trajectory * scale[:, None, None, None]                  # (bs, num_steps, N, 3)
                    self._save_trajectory_gif(
                        trajectory=trajectory[i],
                        colors=colors,
                        sample_name=f"{sample_name}_generation{n+1:02d}",
                    )

            if self.max_samples_per_batch is not None and i >= self.max_samples_per_batch:
                break


class OverlapVisualizationCallback(VisualizationCallback):
    """Visualization callback for overlap prediction models."""

    def on_test_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Save overlap visualizations at the end of each test batch."""
        if self.vis_dir is None:
            return

        overlap_prob = outputs["overlap_prob"]                                # (total_points,)
        B, _ = batch["points_per_part"].shape
        pts_gt = batch["pointclouds_gt"].reshape(B, -1, 3)                    # (bs, N, 3)
        overlap_prob = overlap_prob.reshape(B, -1)                            # (bs, N)

        # Scale to original size
        if self.scale_to_original_size:
            scale = batch["scale"][:, 0]                                      # (bs,)
            pts_gt = pts_gt * scale[:, None, None]

        for i in range(B):
            dataset_name = batch["dataset_name"][i]
            sample_name = f"{dataset_name}_sample{int(batch['index'][i]):05d}"

            colors = probs_to_colors(overlap_prob[i], colormap=self.colormap)
            self._save_sample_images(
                points=pts_gt[i],
                colors=colors,
                sample_name=f"{sample_name}_overlap",
            )

            if self.max_samples_per_batch is not None and i >= self.max_samples_per_batch:
                break

class FeatureVisualizationCallback(VisualizationCallback):
    """Visualization callback for point feature visualization."""

    def on_test_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Save feature visualizations at the end of each test batch."""
        if self.vis_dir is None:
            return

        features = outputs["point"]["feat"]
        B, _ = batch["points_per_part"].shape
        pts_gt = batch["pointclouds_gt"].reshape(B, -1, 3)                    # (bs, N, 3)
        features = features.reshape(B, -1, features.shape[-1])                # (bs, N, feature_dim)

        trainer.logger.log_hyperparams({"output_feature_shape": features.shape})

        # Scale to original size
        if self.scale_to_original_size:
            scale = batch["scale"][:, 0]                                      # (bs,)
            pts_gt = pts_gt * scale[:, None, None]

        for i in range(B):
            dataset_name = batch["dataset_name"][i]
            sample_name = f"{dataset_name}_sample{int(batch['index'][i]):05d}"

            colors = get_pca_colors(features[i], brightness=1.2, center=True)
            self._save_sample_images(
                points=pts_gt[i],
                colors=colors,
                sample_name=f"{sample_name}_features",
            )

            if self.max_samples_per_batch is not None and i >= self.max_samples_per_batch:
                break


# ---------------------------------------------------------------------------
# Query point selection helpers (ported from RPF analysis/vis/similarity.py)
# ---------------------------------------------------------------------------

def _farthest_point_sample_vis(coords: torch.Tensor, num_queries: int) -> torch.Tensor:
    """Farthest point sampling for query point selection."""
    N = coords.shape[0]
    device = coords.device
    selected = [torch.randint(N, (1,), device=device).item()]
    for _ in range(num_queries - 1):
        sel_pts = coords[selected]
        dists = torch.cdist(coords.unsqueeze(0), sel_pts.unsqueeze(0)).squeeze(0)
        min_dists = dists.min(dim=1).values
        min_dists[selected] = -1.0
        selected.append(min_dists.argmax().item())
    return torch.tensor(selected, device=device, dtype=torch.long)


def _select_mating_points(
    gt_coords: torch.Tensor,
    part_ids: torch.Tensor,
    overlap_threshold: float,
    num_queries: int,
) -> torch.Tensor:
    """Select query points from the largest mating surface.

    Identifies all cross-part contact points, determines which part-pair
    interface has the most mating points, and restricts selection to that
    interface.
    """
    N = gt_coords.shape[0]
    device = gt_coords.device

    dists = torch.cdist(gt_coords.unsqueeze(0), gt_coords.unsqueeze(0)).squeeze(0)
    same_part = part_ids.unsqueeze(0) == part_ids.unsqueeze(1)
    dists[same_part] = float("inf")

    min_cross_dist, nearest_idx = dists.min(dim=1)
    mating_mask = min_cross_dist <= overlap_threshold
    mating_indices = mating_mask.nonzero(as_tuple=False).squeeze(-1)

    if mating_indices.numel() == 0:
        k = min(num_queries * 10, N)
        _, closest = min_cross_dist.topk(k, largest=False)
        mating_indices = closest

    if mating_indices.numel() <= num_queries:
        return mating_indices[:num_queries]

    # Find the interface with the most mating points
    own_parts = part_ids[mating_indices]
    neighbor_parts = part_ids[nearest_idx[mating_indices]]
    pair_lo = torch.min(own_parts, neighbor_parts)
    pair_hi = torch.max(own_parts, neighbor_parts)
    max_part = int(part_ids.max().item()) + 1
    pair_key = pair_lo * max_part + pair_hi

    best_key = pair_key.mode().values.item()
    best_indices = mating_indices[pair_key == best_key]

    if best_indices.numel() <= num_queries:
        return best_indices[:num_queries]

    # FPS among points on the largest interface
    local_selected = _farthest_point_sample_vis(gt_coords[best_indices], num_queries)
    return best_indices[local_selected]


def select_query_points(
    coords: torch.Tensor,
    num_queries: int = 1,
    strategy: str = "farthest",
    index: int | None = None,
    gt_coords: torch.Tensor | None = None,
    part_ids: torch.Tensor | None = None,
    overlap_threshold: float | None = None,
) -> torch.Tensor:
    """Select query point indices from a point cloud.

    Args:
        coords: (N, 3) point coordinates.
        num_queries: Number of query points to select.
        strategy: "farthest", "random", "index", or "mating".
        index: Fixed index when strategy="index".
        gt_coords: (N, 3) assembled GT coordinates (required for "mating").
        part_ids: (N,) integer part IDs (required for "mating").
        overlap_threshold: Distance threshold for mating surface (required for "mating").

    Returns:
        (num_queries,) int64 tensor of selected point indices.
    """
    N = coords.shape[0]
    device = coords.device

    if strategy == "index":
        return torch.tensor([index or 0], device=device, dtype=torch.long)
    if strategy == "random":
        return torch.randperm(N, device=device)[:num_queries]
    if strategy == "farthest":
        return _farthest_point_sample_vis(coords, num_queries)
    if strategy == "mating":
        if gt_coords is None or part_ids is None or overlap_threshold is None:
            raise ValueError("mating strategy requires gt_coords, part_ids, and overlap_threshold")
        return _select_mating_points(gt_coords, part_ids, overlap_threshold, num_queries)
    raise ValueError(f"Unknown strategy: {strategy}")


# ---------------------------------------------------------------------------
# PCA teacher feature visualization callback
# ---------------------------------------------------------------------------

class PCAVisualizationCallback(VisualizationCallback):
    """Render teacher features as PCA-colored point clouds during test."""

    def on_test_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if self.vis_dir is None or pl_module.teacher is None:
            return

        with torch.no_grad():
            features = pl_module.teacher.extract_features(batch)[0]  # (B, N, D)

        B = features.shape[0]
        pts_gt = batch["pointclouds_gt"].reshape(B, -1, 3)

        for i in range(B):
            dataset_name = batch["dataset_name"][i]
            sample_name = f"{dataset_name}_sample{int(batch['index'][i]):05d}"

            colors = get_pca_colors(features[i], brightness=1.2, center=True)
            self._save_sample_images(
                points=pts_gt[i],
                colors=colors,
                sample_name=f"{sample_name}_pca",
            )

            if self.max_samples_per_batch is not None and i >= self.max_samples_per_batch:
                break


# ---------------------------------------------------------------------------
# Similarity visualization callback
# ---------------------------------------------------------------------------

class SimilarityVisualizationCallback(VisualizationCallback):
    """Render per-query cosine similarity heatmaps on teacher features.

    Selects query points on the mating surface (or via other strategies),
    computes cosine similarity from the query to all other points, applies
    sigmoid contrast, and renders the result.

    Args:
        num_queries: Number of query points per sample.
        query_strategy: "mating", "farthest", "random", or "index".
        colormap: Diverging colormap for similarity.
        top_pct, highlight_pct, bottom_pct: Sigmoid contrast parameters.
        query_color: RGB for the query point marker.
        query_neighbors: Spatial neighbors to also color as query marker.
        **kwargs: Passed to VisualizationCallback.
    """

    def __init__(
        self,
        num_queries: int = 3,
        query_strategy: str = "mating",
        colormap: str = "matplotlib:Spectral_r",
        top_pct: float = 0.02,
        highlight_pct: float = 0.10,
        bottom_pct: float = 0.10,
        query_color: tuple[float, float, float] = (1.0, 0.0, 1.0),
        query_neighbors: int = 10,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_queries = num_queries
        self.query_strategy = query_strategy
        self.colormap = colormap
        self.top_pct = top_pct
        self.highlight_pct = highlight_pct
        self.bottom_pct = bottom_pct
        self.query_color = query_color
        self.query_neighbors = query_neighbors

    def on_test_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if self.vis_dir is None or pl_module.teacher is None:
            return

        with torch.no_grad():
            features = pl_module.teacher.extract_features(batch)[0]  # (B, N, D)

        B = features.shape[0]
        pts_gt = batch["pointclouds_gt"].reshape(B, -1, 3)
        points_per_part = batch["points_per_part"]

        for i in range(B):
            dataset_name = batch["dataset_name"][i]
            sample_name = f"{dataset_name}_sample{int(batch['index'][i]):05d}"

            # Build part IDs for mating strategy
            part_ids_i = None
            if self.query_strategy == "mating":
                from .utils.point_clouds import get_part_ids
                N = pts_gt.shape[1]
                part_ids_i = get_part_ids(points_per_part[i:i+1], N).squeeze(0)

            overlap_thresh = None
            if "overlap_threshold" in batch:
                overlap_thresh = batch["overlap_threshold"][i].item()

            query_indices = select_query_points(
                coords=pts_gt[i],
                num_queries=self.num_queries,
                strategy=self.query_strategy,
                gt_coords=pts_gt[i],
                part_ids=part_ids_i,
                overlap_threshold=overlap_thresh,
            )

            for q, qidx in enumerate(query_indices):
                colors = get_similarity_colors(
                    features=features[i],
                    coords=pts_gt[i],
                    query_idx=qidx.item(),
                    colormap=self.colormap,
                    top_pct=self.top_pct,
                    highlight_pct=self.highlight_pct,
                    bottom_pct=self.bottom_pct,
                    query_color=self.query_color,
                    query_neighbors=self.query_neighbors,
                )
                self._save_sample_images(
                    points=pts_gt[i],
                    colors=colors,
                    sample_name=f"{sample_name}_sim_query{q:02d}",
                )

            if self.max_samples_per_batch is not None and i >= self.max_samples_per_batch:
                break

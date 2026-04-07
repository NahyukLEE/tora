"""Disk-backed cache for precomputed teacher features.

Features are computed once on *canonical* (un-augmented) GT point clouds and
stored to disk.  During training the cached features are looked up by fragment
name, eliminating the teacher forward-pass cost entirely.

Storage format
--------------
Each sample's features are stored as ``<sanitized_name>.pt`` containing a single
tensor of shape ``(N, repr_dim)`` in the configured dtype (default float16).

DDP compatibility
-----------------
All ranks write to the same cache directory.  Writes are idempotent (skip if the
file already exists), so concurrent writes from different ranks are safe.
"""

import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset

logger = logging.getLogger(__name__)


class TeacherFeatureCache:
    """Disk-backed cache for precomputed teacher features.

    Args:
        cache_dir: Directory to store cached feature files.
        dtype: Storage dtype.  ``torch.float16`` halves disk usage with
            negligible quality loss for alignment targets.
    """

    def __init__(
        self,
        cache_dir: str,
        dtype: torch.dtype = torch.float16,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.dtype = dtype

    # ------------------------------------------------------------------
    # Name <-> path helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sanitize(name: str) -> str:
        """Convert a fragment name (may contain ``/``) to a safe filename stem."""
        return name.replace("/", "__").replace("\\", "__")

    @staticmethod
    def _unsanitize(stem: str) -> str:
        return stem.replace("__", "/")

    def _path_for(self, name: str) -> Path:
        return self.cache_dir / f"{self._sanitize(name)}.pt"

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return sum(1 for _ in self.cache_dir.glob("*.pt"))

    def has(self, name: str) -> bool:
        return self._path_for(name).exists()

    def has_batch(self, names: list[str] | tuple[str, ...]) -> bool:
        return all(self.has(n) for n in names)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def put(self, name: str, features: torch.Tensor) -> None:
        """Cache a single sample's features ``(N, D)`` to disk."""
        path = self._path_for(name)
        if not path.exists():
            torch.save(features.detach().cpu().to(self.dtype), path)

    def put_batch(self, names: list[str] | tuple[str, ...], features: torch.Tensor) -> None:
        """Cache a batch of features ``(B, N, D)``."""
        for i, name in enumerate(names):
            self.put(name, features[i])

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get(self, name: str, device: torch.device | str | None = None) -> torch.Tensor:
        """Load features for a single sample.  Returns float32."""
        t = torch.load(
            self._path_for(name), map_location="cpu", weights_only=True,
        )
        if device is not None:
            t = t.to(device)
        return t.float()

    def get_batch(
        self, names: list[str] | tuple[str, ...], device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """Load features for a batch.  Returns ``(B, N, D)`` in float32."""
        return torch.stack([self.get(n, device=device) for n in names])



# ----------------------------------------------------------------------
# Precompute utility
# ----------------------------------------------------------------------

@torch.no_grad()
def precompute_teacher_features(
    teacher: nn.Module,
    data_root: str,
    dataset_names: list[str],
    cache_dir: str,
    num_points_to_sample: int = 5000,
    max_parts: int = 64,
    min_parts: int = 2,
    up_axis: dict[str, str] | None = None,
    batch_size: int = 16,
    num_workers: int = 8,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.float16,
) -> TeacherFeatureCache:
    """Precompute teacher features on canonical (un-augmented) GT point clouds.

    Creates a :class:`TeacherFeatureCache` populated with features for every
    sample in the training split.  The dataset is loaded with augmentation
    disabled (no random rotation or scale) so that features are deterministic
    and can be reused across all training epochs.

    Args:
        teacher: Frozen teacher encoder (must implement ``extract_features``).
        data_root: Root directory containing dataset HDF5 / folders.
        dataset_names: List of dataset names to precompute.
        cache_dir: Directory to write cached ``.pt`` files.
        num_points_to_sample: Points per sample (must match training config).
        max_parts: Maximum number of parts (must match training config).
        min_parts: Minimum number of parts.
        up_axis: Per-dataset up-axis overrides, e.g. ``{"ikea": "y"}``.
        batch_size: Batch size for the precompute pass.
        num_workers: DataLoader workers.
        device: Device to run the teacher on.
        dtype: Storage dtype for cached features.

    Returns:
        The populated :class:`TeacherFeatureCache`.
    """
    import os
    from ..data.dataset import PointCloudDataset
    from ..data.datamodule import worker_init_fn

    up_axis = up_axis or {}
    cache = TeacherFeatureCache(cache_dir, dtype=dtype)

    # Build datasets with augmentation disabled
    datasets = []
    dataset_paths = {}
    for f in os.listdir(data_root):
        name = f.split(".")[0] if f.endswith(".hdf5") else f
        path = os.path.join(data_root, f)
        if name in dataset_names and (f.endswith(".hdf5") or os.path.isdir(path)):
            dataset_paths[name] = path

    for name in dataset_names:
        if name not in dataset_paths:
            logger.warning(f"Dataset '{name}' not found in {data_root}, skipping")
            continue
        ds = PointCloudDataset(
            split="train",
            data_path=dataset_paths[name],
            dataset_name=name,
            up_axis=up_axis.get(name, "y"),
            min_parts=min_parts,
            max_parts=max_parts,
            num_points_to_sample=num_points_to_sample,
            disable_augmentation=True,
        )
        datasets.append(ds)

    if not datasets:
        raise ValueError(f"No datasets found for {dataset_names} in {data_root}")

    concat = ConcatDataset(datasets)
    loader = DataLoader(
        concat,
        batch_size=batch_size,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        shuffle=False,
        pin_memory=True,
    )

    teacher = teacher.to(device).eval()
    total = len(concat)
    cached = 0
    skipped = 0

    logger.info(f"Precomputing teacher features for {total} samples -> {cache_dir}")

    for batch in loader:
        names = batch["name"]

        # Skip if entire batch is already cached
        if cache.has_batch(names):
            skipped += len(names)
            continue

        # Move tensors to device
        batch_gpu = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        features = teacher.extract_features(batch_gpu)[0]  # (B, N, repr_dim)
        cache.put_batch(names, features)
        cached += len(names)

        if (cached + skipped) % (batch_size * 10) == 0:
            logger.info(f"  [{cached + skipped}/{total}] cached={cached}, skipped={skipped}")

    logger.info(f"Done: {cached} newly cached, {skipped} already existed. Total: {len(cache)}")
    return cache

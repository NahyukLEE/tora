"""Vendored loader utilities for teacher encoders.

Each loader module provides ``load_*()`` functions that download pretrained
checkpoints (from HuggingFace) and return an ``nn.Module`` ready for inference.
"""

from .uni3d_loader import load_uni3d, UNI3D_CONFIGS, propagate_features
from .find3d_loader import (
    load_find3d,
    FIND3D_CONFIGS,
    preprocess_batch_for_find3d,
    propagate_features_knn,
    _patch_spconv_for_arm,
)
from .patchalign3d_loader import (
    load_patchalign3d,
    PATCHALIGN3D_CONFIGS,
    preprocess_for_patchalign3d,
    propagate_features_from_patches,
)
from .openshape_loader import (
    load_openshape,
    OPENSHAPE_CONFIGS,
    propagate_features_from_centroids,
)

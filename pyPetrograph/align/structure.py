"""Structure maps for lineup: edges + texture, not color or brightness."""
from __future__ import annotations

import numpy as np

from pyPetrograph.image_processing.brightness import relative_luminance
from pyPetrograph.image_processing.features import (
    compute_lbp_map,
    compute_local_std,
    compute_sobel_mag,
    downsample_to_approx_mp,
)


def _stretch01(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)
    lo, hi = np.percentile(a, (1.0, 99.0))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(a, dtype=np.float32)
    return np.clip((a - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def structure_map(
    rgb: np.ndarray,
    *,
    use_lbp: bool = True,
) -> np.ndarray:
    """
    HxW float32 map for matching: local_grad + local_std + LBP.

    Luminance is only used to compute texture; the map is not a brightness photo.
    """
    y = relative_luminance(rgb).astype(np.float32)
    grad = _stretch01(compute_sobel_mag(y))
    std = _stretch01(compute_local_std(y))
    if use_lbp:
        lbp = _stretch01(compute_lbp_map(y))
        out = 0.45 * grad + 0.30 * std + 0.25 * lbp
    else:
        out = 0.60 * grad + 0.40 * std
    return out.astype(np.float32)


def structure_map_preview(rgb: np.ndarray, target_mp: float = 2.0) -> np.ndarray:
    """Downsampled structure map for fast auto-align / UI."""
    small = downsample_to_approx_mp(rgb, target_mp=target_mp)
    return structure_map(small)

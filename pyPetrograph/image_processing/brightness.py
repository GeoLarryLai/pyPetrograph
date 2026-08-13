"""Brightness / luminance normalization."""
from __future__ import annotations

from typing import Optional

import numpy as np

from pyPetrograph.common.constants import DEFAULT_Y_TARGET

def relative_luminance(rgb: np.ndarray) -> np.ndarray:
    """
    Relative luminance Y from RGB (0–255 or float).

    Y = 0.2126 R + 0.7152 G + 0.0722 B
    """
    rgb = np.asarray(rgb, dtype=np.float32)
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def normalize_brightness(
    image: np.ndarray,
    y_target: float = DEFAULT_Y_TARGET,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Scale RGB so mean luminance matches ``y_target``, keeping color ratios.

    White-margin pixels (mean == 255) are left unchanged unless excluded via mask.
    Returns float32 RGB (not clipped to uint8) for ML features; display may clip.
    """
    rgb = np.asarray(image, dtype=np.float32)
    y = relative_luminance(rgb)

    if mask is None:
        mean_im = rgb.mean(axis=-1)
        use = mean_im < 255
    else:
        use = np.asarray(mask, dtype=bool)

    if not np.any(use):
        return rgb

    y_mean = float(y[use].mean())
    if y_mean <= 1e-6:
        return rgb

    scale = float(y_target) / y_mean
    out = rgb * scale
    # Preserve pure white margins.
    out[~use] = rgb[~use]
    return out.astype(np.float32)


def to_display_uint8(image: np.ndarray) -> np.ndarray:
    """Clip float/int image to uint8 for display or PNG preview."""
    return np.clip(np.asarray(image), 0, 255).astype(np.uint8)

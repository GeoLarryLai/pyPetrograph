"""Class colors, overlays, legends, and label PNG helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from pyPetrograph.common.constants import (
    DEFAULT_POLY_FILL_ALPHA,
    LEGEND_FONTSIZE,
    LEGEND_MAX_PER_ROW,
    PathLike,
)
from pyPetrograph.common.image_io import _to_rgb_uint8

_CLASS_COLOR_GOLDEN = 0.618033988749895

def class_color_rgba(class_id: int, alpha: float = 1.0) -> Tuple[float, float, float, float]:
    """
    Bright, well-separated color for a class id (HSV wheel + golden-angle step).

    Same id → same color in Label UI, label summary, and predict figures.
    Adjacent ids land far apart on the hue wheel (avoids near-duplicates).
    """
    import colorsys

    # 1-based ids; golden angle keeps neighbors dissimilar
    h = (((int(class_id) - 1) * _CLASS_COLOR_GOLDEN) + 0.12) % 1.0
    s, v = 0.78, 0.96
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (float(r), float(g), float(b), float(alpha))


def colorize_labels_hsv(pred: np.ndarray) -> np.ndarray:
    """Map label ids → RGB uint8 using ``class_color_rgba`` (0 → black)."""
    pred = np.asarray(pred, dtype=np.uint8)
    h, w = pred.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cid in np.unique(pred):
        cid = int(cid)
        if cid <= 0:
            continue
        r, g, b, _ = class_color_rgba(cid)
        rgb[pred == cid] = (
            int(round(r * 255)),
            int(round(g * 255)),
            int(round(b * 255)),
        )
    return rgb


def overlay_class_map(
    image_rgb: np.ndarray,
    labels: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """Blend HSV class colors onto the original RGB where labels > 0."""
    base = _to_rgb_uint8(image_rgb).astype(np.float32)
    color = colorize_labels_hsv(labels).astype(np.float32)
    out = base.copy()
    mask = np.asarray(labels) > 0
    if np.any(mask):
        a = float(np.clip(alpha, 0.0, 1.0))
        out[mask] = (1.0 - a) * base[mask] + a * color[mask]
    return np.clip(out, 0, 255).astype(np.uint8)


def add_class_legend(
    ax,
    class_ids: Sequence[int],
    class_names: Optional[Dict[int, str]] = None,
    *,
    fontsize: int = LEGEND_FONTSIZE,
    max_per_row: int = LEGEND_MAX_PER_ROW,
    fill_alpha: float = DEFAULT_POLY_FILL_ALPHA,
) -> None:
    """
    Legend below / outside the axes: up to ``max_per_row`` entries per row.

    Parameters: ax, class_ids, optional names; fontsize / row wrap / swatch alpha.
    """
    from matplotlib.patches import Patch

    ids = sorted({int(c) for c in class_ids if int(c) > 0})
    if not ids:
        return
    handles = []
    for cid in ids:
        r, g, b, _ = class_color_rgba(cid)
        name = (class_names or {}).get(cid, f"class_{cid}")
        handles.append(
            Patch(
                facecolor=(r, g, b, float(fill_alpha)),
                edgecolor=(r, g, b, 1.0),
                linewidth=1.5,
                label=f"{cid}: {name}",
            )
        )
    ncol = max(1, min(int(max_per_row), len(handles)))
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.04),
        ncol=ncol,
        fontsize=int(fontsize),
        frameon=False,
        handlelength=1.2,
        columnspacing=1.0,
        borderaxespad=0.0,
    )


def save_labels_png(path: PathLike, labels: np.ndarray) -> Path:
    """Save a label raster as uint8 PNG (compatible format)."""
    path = Path(path)
    Image.fromarray(np.asarray(labels, dtype=np.uint8)).save(path)
    return path


def load_labels_png(path: PathLike) -> np.ndarray:
    """Load a label PNG to uint8 2D array (handles float PNG read quirks)."""
    path = Path(path)
    with Image.open(path) as im:
        arr = np.array(im)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    if np.issubdtype(arr.dtype, np.floating):
        if arr.max() <= 1.0 + 1e-6:
            arr = np.round(arr * 255.0)
    return np.asarray(arr, dtype=np.uint8)


def try_load_legacy_label_ids(
    path: PathLike, shape: Tuple[int, int]
) -> Optional[np.ndarray]:
    """
    Load old uint8 class-id ``*_labels.png`` only if mode is ``L``.

    Current ``*_labels.png`` is an RGB summary figure — ignored here.
    """
    p = Path(path)
    if not p.exists():
        return None
    try:
        with Image.open(p) as im:
            if im.mode != "L":
                return None
            lab = np.asarray(im, dtype=np.uint8)
        if lab.shape == tuple(shape):
            return lab
    except Exception:
        return None
    return None


def colorize_prediction(
    pred: np.ndarray,
    color_palette: Dict[int, Sequence[int]],
) -> np.ndarray:
    """Map label ids to RGB using the training mean-color palette."""
    h, w = pred.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for label, color in color_palette.items():
        rgb[pred == int(label)] = np.asarray(color, dtype=np.uint8)
    return rgb

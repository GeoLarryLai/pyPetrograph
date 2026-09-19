"""
Legend swatches and color → class snapping for flat-color mineral maps.

Class codes: ``0`` background (white / outside scan), ``1..K`` legend order.
Colors farther than ``tol`` from every swatch are grouped into Unknown (not a
separate class).
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image

from pyPetrograph.common.constants import PathLike

RGB = Tuple[int, int, int]
Legend = Dict[str, RGB]

BACKGROUND_NAME = "background"
UNKNOWN_NAME = "Unknown"
UNLISTED_NAME = "unlisted"  # leftover alias; far colors now map to Unknown
UNLISTED_RGB: RGB = (128, 128, 128)
WHITE = np.array([255.0, 255.0, 255.0], dtype=np.float32)


def class_names(legend: Legend) -> List[str]:
    """Names indexed by class code: ``[background, *legend]``."""
    return [BACKGROUND_NAME, *legend.keys()]


def unknown_code(legend: Legend) -> int:
    """Class code of Unknown (name match, else the black swatch)."""
    for i, (n, c) in enumerate(legend.items(), start=1):
        if str(n).strip().lower() == "unknown":
            return i
    for i, c in enumerate(legend.values(), start=1):
        if tuple(int(v) for v in c) == (0, 0, 0):
            return i
    raise KeyError("LEGEND must include Unknown or a black (0, 0, 0) swatch")


def unlisted_code(legend: Legend) -> int:
    """Deprecated alias: far colors now use ``unknown_code``."""
    return unknown_code(legend)


def read_rgba(path: PathLike) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Read an image as ``(rgb uint8 HxWx3, alpha uint8 HxW or None)``."""
    with Image.open(Path(path)) as im:
        arr = np.asarray(im)
    if arr.ndim == 2:
        return np.repeat(arr[..., None], 3, axis=-1).astype(np.uint8), None
    if arr.shape[-1] >= 4:
        return np.ascontiguousarray(arr[..., :3]), np.ascontiguousarray(arr[..., 3])
    return np.ascontiguousarray(arr[..., :3]), None


def _pack(rgb: np.ndarray) -> np.ndarray:
    r = rgb[..., 0].astype(np.uint32)
    g = rgb[..., 1].astype(np.uint32)
    b = rgb[..., 2].astype(np.uint32)
    return (r << 16) | (g << 8) | b


def _unpack(packed: np.ndarray) -> np.ndarray:
    return np.stack([(packed >> 16) & 255, (packed >> 8) & 255, packed & 255], axis=1)


def read_legend_swatches(path: PathLike, swatch_px: Optional[int] = None) -> List[RGB]:
    """
    Swatch colors from a legend image, top to bottom.

    Swatches are same-size colored boxes. If ``swatch_px`` is None, use the pixel
    count shared by the most colors (excluding white and black). Black is skipped
    because it is also the text color.
    """
    rgb, _ = read_rgba(path)
    h, w = rgb.shape[:2]
    packed = _pack(rgb).ravel()
    u, inv, cnt = np.unique(packed, return_inverse=True, return_counts=True)
    cols = _unpack(u)
    is_white = np.all(cols == 255, axis=1)
    is_black = np.all(cols == 0, axis=1)
    cand = ~is_white & ~is_black & (cnt >= 16)
    if not cand.any():
        return []
    if swatch_px is None:
        swatch_px = Counter(int(c) for c in cnt[cand]).most_common(1)[0][0]
    keep = np.where(cand & (cnt == int(swatch_px)))[0]
    rows = np.repeat(np.arange(h), w)
    inv = inv.ravel()
    out: List[Tuple[float, RGB]] = []
    for i in keep:
        r_mean = float(rows[inv == i].mean())
        out.append((r_mean, tuple(int(v) for v in cols[i])))
    out.sort(key=lambda t: t[0])
    return [c for _, c in out]


def legend_from_image(
    path: PathLike,
    names: Optional[Sequence[str]] = None,
    swatch_px: Optional[int] = None,
    unknown: bool = True,
) -> Legend:
    """
    Build a legend dict from a swatch image (colors top to bottom).

    ``names`` must match the swatch count, or omit it to get ``swatch_1..N``.
    If ``unknown`` is True and no black entry exists, append
    ``"Unknown": (0, 0, 0)``.
    """
    colors = read_legend_swatches(path, swatch_px=swatch_px)
    if names is None:
        labels = [f"swatch_{i}" for i in range(1, len(colors) + 1)]
    elif len(names) != len(colors):
        raise ValueError(f"names has {len(names)} entries but image has {len(colors)} swatches")
    else:
        labels = [str(n) for n in names]
    legend: Legend = {n: c for n, c in zip(labels, colors)}
    if unknown:
        has_black = any(tuple(int(v) for v in c) == (0, 0, 0) for c in legend.values())
        if not has_black:
            legend["Unknown"] = (0, 0, 0)
    return legend


def check_legend(legend: Legend, swatch_colors: Sequence[RGB]) -> List[str]:
    """
    Compare typed ``legend`` (name → rgb) to swatches read from the legend image.

    Black entries (e.g. "Unknown") are skipped. Returns mismatch messages; empty = OK.
    """
    typed = [(n, tuple(int(v) for v in c)) for n, c in legend.items() if tuple(c) != (0, 0, 0)]
    sw = [tuple(int(v) for v in c) for c in swatch_colors]
    msgs: List[str] = []
    if len(typed) != len(sw):
        msgs.append(f"LEGEND has {len(typed)} non-black colors but the image has {len(sw)} swatches")
    for i, ((name, c), s) in enumerate(zip(typed, sw)):
        if c != s:
            msgs.append(f"row {i + 1}: LEGEND[{name!r}] = {c} but swatch = {s}")
    typed_set = {c for _, c in typed}
    for s in sw:
        if s not in typed_set:
            msgs.append(f"swatch {s} is not in LEGEND")
    return msgs


def snap_to_legend(
    rgb: np.ndarray,
    alpha: Optional[np.ndarray],
    legend: Legend,
    tol: float = 40.0,
    white_tol: float = 40.0,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Map every pixel to a legend class by nearest RGB color.

    Parameters
    ----------
    rgb : uint8 HxWx3
    alpha : uint8 HxW or None; ``alpha == 0`` = outside the scan (frame, footer)
    legend : name → (r, g, b), in legend order
    tol : max RGB distance to a legend color
    white_tol : max RGB distance to white to count as background

    Returns
    -------
    cmap : uint8 HxW class codes (0 background, 1..K legend; far colors = Unknown)
    valid : bool HxW, False outside the scan
    remapped : DataFrame ``r, g, b, n_px, nearest, dist`` for colors beyond ``tol``
        (those pixels are coded Unknown, not a separate class)
    """
    h, w = rgb.shape[:2]
    valid = np.ones((h, w), dtype=bool) if alpha is None else alpha > 0
    packed = _pack(rgb).ravel()
    u, inv = np.unique(packed, return_inverse=True)
    inv = inv.ravel()
    cols = _unpack(u).astype(np.float32)
    lut_colors = np.asarray(list(legend.values()), dtype=np.float32)
    names = list(legend.keys())
    d = np.sqrt(((cols[:, None, :] - lut_colors[None, :, :]) ** 2).sum(-1))
    nearest = d.argmin(axis=1)
    dmin = d.min(axis=1)
    white_d = np.sqrt(((cols - WHITE) ** 2).sum(-1))
    k_unk = unknown_code(legend)
    lut = np.full(len(u), k_unk, dtype=np.uint8)
    lut[dmin <= float(tol)] = (nearest[dmin <= float(tol)] + 1).astype(np.uint8)
    lut[white_d <= float(white_tol)] = 0
    cmap = lut[inv].reshape(h, w)
    cmap[~valid] = 0
    n_valid = np.bincount(inv[valid.ravel()], minlength=len(u))
    # colors that were forced to Unknown (not an exact/near swatch, not white)
    unl = np.where((dmin > float(tol)) & (white_d > float(white_tol)) & (n_valid > 0))[0]
    unlisted = pd.DataFrame(
        {
            "r": cols[unl, 0].astype(int),
            "g": cols[unl, 1].astype(int),
            "b": cols[unl, 2].astype(int),
            "n_px": n_valid[unl].astype(int),
            "nearest": [names[int(i)] for i in nearest[unl]],
            "dist": np.round(dmin[unl], 1),
        }
    ).sort_values("n_px", ascending=False, ignore_index=True)
    return cmap, valid, unlisted

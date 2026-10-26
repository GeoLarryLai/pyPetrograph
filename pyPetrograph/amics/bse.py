"""BSE image prep: footer strip, contrast stretch, click-based dark threshold, footer PixelSize OCR."""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Optional, Sequence, Tuple

import numpy as np

if TYPE_CHECKING:  # window_worker.py imports this file as a sibling and must not import pyPetrograph
    from pyPetrograph.common.constants import PathLike

Point = Tuple[float, float]

_PIXELSIZE_RE = re.compile(
    r"P[il1I]xe[l1I]\s*Size\s*[:;.]?\s*([0-9]+(?:[.,][0-9]+)?)\s*(µm|μm|um|pm|jm|nm|mm)",
    re.IGNORECASE,
)
_UNIT_FACTOR = {"µm": 1.0, "μm": 1.0, "um": 1.0, "pm": 1.0, "jm": 1.0, "nm": 0.001, "mm": 1000.0}


def load_bse(path: PathLike) -> Tuple[np.ndarray, np.ndarray]:
    """``(gray uint8 HxW = mean of RGB, rgba uint8 HxWx4)``; alpha = 255 if the file has none."""
    from PIL import Image

    with Image.open(path) as im:
        rgba = np.asarray(im.convert("RGBA"), dtype=np.uint8).copy()
    gray = np.rint(rgba[..., :3].astype(np.float64).mean(axis=2)).astype(np.uint8)
    return gray, rgba


def detect_footer(rgba: np.ndarray) -> int:
    """First row of the bottom scale-bar strip (white + red ruler); ``H`` when there is none."""
    h = rgba.shape[0]
    r_, g_, b_ = rgba[..., 0], rgba[..., 1], rgba[..., 2]
    red = (r_ > 180) & (g_ < 90) & (b_ < 90)
    frac = red.mean(axis=1)
    cand = np.nonzero((np.arange(h) > h // 2) & (frac >= 0.9))[0]
    if cand.size == 0:
        return h
    r = int(cand[0])
    if r > 0:
        prev = rgba[r - 1, :, :3]
        if (prev > 230).all(axis=1).mean() >= 0.9:
            return r - 1
    return r


def valid_mask(rgba: np.ndarray, footer_row: int) -> np.ndarray:
    """``alpha > 0`` and ``row < footer_row``."""
    valid = rgba[..., 3] > 0
    valid[max(int(footer_row), 0):, :] = False
    return valid


def stretch(gray: np.ndarray, valid: np.ndarray, lo: float = 1.0, hi: float = 99.0) -> np.ndarray:
    """Percentile contrast stretch over valid pixels to uint8; invalid pixels = 0."""
    valid = np.asarray(valid, dtype=bool)
    if not valid.any():
        return np.zeros(gray.shape, dtype=np.uint8)
    p_lo, p_hi = np.percentile(gray[valid], [lo, hi])
    if p_hi <= p_lo:
        p_hi = p_lo + 1
    out = np.clip((gray.astype(np.float64) - p_lo) / (p_hi - p_lo), 0, 1) * 255
    out = np.rint(out).astype(np.uint8)
    out[~valid] = 0
    return out


def threshold_from_points(
    gray: np.ndarray,
    points: Sequence[Point],
    valid: Optional[np.ndarray] = None,
    radius: int = 5,
    pct: float = 99.0,
) -> float:
    """Gray values within ``radius`` of each clicked ``(x, y)`` -> ``pct`` percentile + 1."""
    h, w = gray.shape[:2]
    pools = []
    for x, y in points:
        x0, x1 = max(int(np.floor(x - radius)), 0), min(int(np.ceil(x + radius)) + 1, w)
        y0, y1 = max(int(np.floor(y - radius)), 0), min(int(np.ceil(y + radius)) + 1, h)
        if x0 >= x1 or y0 >= y1:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        disc = (xx - x) ** 2 + (yy - y) ** 2 <= radius ** 2
        if valid is not None:
            disc &= np.asarray(valid, dtype=bool)[y0:y1, x0:x1]
        vals = gray[y0:y1, x0:x1][disc]
        if vals.size:
            pools.append(vals.astype(np.float64))
    if not pools:
        raise ValueError("no pixels near the clicked points")
    pool = np.concatenate(pools)
    return float(np.percentile(pool, pct) + 1.0)


def dark_mask(gray: np.ndarray, thr: float, valid: np.ndarray) -> np.ndarray:
    """``(gray < thr) | ~valid``."""
    return (gray < thr) | ~np.asarray(valid, dtype=bool)


def plot_threshold_preview(gray: np.ndarray, dark: np.ndarray, footer_row: int, points: Sequence[Point] = ()):
    """Gray BSE, dark pixels tinted blue (alpha 0.5), red dashed footer line, clicks as x. Returns the Figure."""
    import matplotlib.pyplot as plt

    h, w = gray.shape[:2]
    fig, ax = plt.subplots(figsize=(10, 10 * h / w))
    ax.imshow(gray, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    overlay = np.zeros((h, w, 4), dtype=np.float64)
    overlay[..., 0] = 0.0
    overlay[..., 1] = 0.4
    overlay[..., 2] = 1.0
    overlay[..., 3] = np.where(dark, 0.5, 0.0)
    ax.imshow(overlay, interpolation="nearest")
    if footer_row < h:
        ax.axhline(footer_row - 0.5, color="red", linestyle="--", linewidth=1)
        ax.text(2, footer_row - 0.5, "footer", color="red", va="bottom", ha="left")
    if len(points):
        pts = np.asarray(points, dtype=float).reshape(-1, 2)
        ax.plot(pts[:, 0], pts[:, 1], linestyle="none", marker="x", color="red", markersize=8)
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)
    ax.axis("off")
    ax.set_title(f"dark pixels (blue): {100 * float(np.mean(dark)):.1f}% of the image")
    fig.tight_layout()
    return fig


def read_footer_pixelsize(rgba: np.ndarray, footer_row: int) -> Optional[float]:
    """OCR ``PixelSize: 2.7µm`` (µm / um / nm / mm) from the footer; µm per pixel or None."""
    try:
        from PIL import Image

        h = rgba.shape[0]
        if h - footer_row < 8:
            return None
        from pyPetrograph.amics.legend_read import _get_engine

        engine = _get_engine()
        if engine is None:
            return None
        crop = Image.fromarray(np.ascontiguousarray(rgba[footer_row:, :, :3]), "RGB")
        crop = crop.resize((crop.width * 3, crop.height * 3), Image.LANCZOS)
        canvas = Image.new("RGB", (crop.width + 60, crop.height + 60), (255, 255, 255))
        canvas.paste(crop, (30, 30))
        result, _ = engine(np.asarray(canvas))
        if not result:
            return None

        def _key(item):
            box = np.asarray(item[0], dtype=float)
            return (float(box[:, 1].min()), float(box[:, 0].min()))

        text = " ".join(str(item[1]) for item in sorted(result, key=_key))
        m = _PIXELSIZE_RE.search(text)
        if not m:
            return None
        number = float(m.group(1).replace(",", "."))
        factor = _UNIT_FACTOR.get(m.group(2).lower(), _UNIT_FACTOR.get(m.group(2)))
        if factor is None:
            return None
        value = number * factor
        return value if value > 0 else None
    except Exception:
        return None

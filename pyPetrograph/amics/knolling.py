"""Knolling image: grains cut out of the BSE, long axis vertical, grouped by mineral, largest first."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from pyPetrograph.common.constants import PathLike

RGB = Tuple[int, int, int]

_GRAY: RGB = (128, 128, 128)

# A narrow sheet stays readable on screen. Grains keep their shape; every tile is drawn at this scale.
DEFAULT_SHEET_WIDTH_PX = 5500
DEFAULT_FONT_PX = 48
DEFAULT_SCALE = 2.0


def _empty_tile() -> np.ndarray:
    return np.zeros((1, 1, 4), np.uint8)


def _largest_part(poly):
    """Largest polygon of a (Multi)Polygon, or None when empty / missing."""
    if poly is None or getattr(poly, "is_empty", True):
        return None
    if hasattr(poly, "geoms"):
        parts = [g for g in poly.geoms if not g.is_empty and hasattr(g, "exterior")]
        if not parts:
            return None
        return max(parts, key=lambda g: g.area)
    if not hasattr(poly, "exterior"):
        return None
    return poly


def _rotate_mask(mask: np.ndarray, angle: float) -> np.ndarray:
    from skimage.transform import rotate

    if abs(angle) < 0.5:
        return mask
    return (
        rotate(mask.astype(float), angle, resize=True, order=0, mode="constant", cval=0, preserve_range=True) > 0.5
    )


def _mask_tilt(mask: np.ndarray) -> float:
    """abs(skimage orientation) of the mask in radians (0 = vertical long axis)."""
    from skimage.measure import regionprops

    if not mask.any():
        return float("inf")
    return abs(float(regionprops(mask.astype(int))[0].orientation))


def grain_tile(
    gray: np.ndarray,
    poly,
    rgb: RGB,
    orientation_deg: float,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    RGBA uint8 tile of one grain: BSE crop inside the polygon blended with ``rgb``
    at ``alpha``, transparent outside, rotated so the long axis is vertical.
    """
    from skimage.draw import polygon as draw_polygon
    from skimage.transform import rotate

    poly = _largest_part(poly)
    if poly is None:
        return _empty_tile()
    H, W = gray.shape[:2]
    minx, miny, maxx, maxy = poly.bounds
    x0 = max(int(np.floor(minx)) - 2, 0)
    y0 = max(int(np.floor(miny)) - 2, 0)
    x1 = min(int(np.ceil(maxx)) + 2, W)
    y1 = min(int(np.ceil(maxy)) + 2, H)
    if x1 <= x0 or y1 <= y0:
        return _empty_tile()
    shape = (y1 - y0, x1 - x0)
    xs, ys = poly.exterior.xy
    rr, cc = draw_polygon(np.asarray(ys) - y0, np.asarray(xs) - x0, shape=shape)
    mask = np.zeros(shape, bool)
    mask[rr, cc] = True
    if int(mask.sum()) < 4:
        return _empty_tile()

    crop = gray[y0:y1, x0:x1].astype(float)
    rgb_f = (1.0 - alpha) * crop[..., None] + alpha * np.asarray(rgb, float)
    rgb_f[~mask] = rgb_f[mask].mean(axis=0)

    theta = float(orientation_deg) if np.isfinite(orientation_deg) else 0.0
    angle = -theta
    rot_mask = _rotate_mask(mask, angle)
    if _mask_tilt(rot_mask) > 0.35:
        alt = _rotate_mask(mask, theta)
        if _mask_tilt(alt) < _mask_tilt(rot_mask):
            angle, rot_mask = theta, alt
    if abs(angle) < 0.5:
        rot_rgb = rgb_f
    else:
        rot_rgb = rotate(rgb_f, angle, resize=True, order=1, mode="constant", cval=0, preserve_range=True)

    rows = np.flatnonzero(rot_mask.any(axis=1))
    cols = np.flatnonzero(rot_mask.any(axis=0))
    if rows.size == 0 or cols.size == 0:
        return _empty_tile()
    sl = (slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1))
    m = rot_mask[sl]
    out = np.empty(m.shape + (4,), np.uint8)
    out[..., :3] = np.clip(np.rint(rot_rgb[sl]), 0, 255).astype(np.uint8)
    out[..., 3] = np.where(m, 255, 0).astype(np.uint8)
    out[~m, :3] = 255
    return out


def _band_order(minerals: Sequence[str]) -> List[str]:
    """Mineral names: count descending, ties alphabetical, ``Unknown`` last."""
    counts: Dict[str, int] = {}
    for m in minerals:
        counts[m] = counts.get(m, 0) + 1
    names = list(counts)
    return sorted(names, key=lambda n: (n.strip().lower() == "unknown", -counts[n], n))


def _layout(
    bands: Sequence[Tuple[str, Sequence[Tuple[int, int]]]],
    width: int,
    pad: int,
    font_px: int,
) -> dict:
    """
    Shelf-pack tiles. ``bands`` = [(mineral, [(h, w), ...]), ...] in drawing order.
    Returns ``{"order": [mineral, ...], "height": H, "bands": [{"mineral", "header_y",
    "tiles": [(index_in_band, x, y, w, h), ...]}, ...]}``.
    """
    y = 0
    out = []
    for name, sizes in bands:
        header_y = y
        y += font_px + 2 * pad
        x = pad
        shelf_h = 0
        placed = []
        for i, (h, w) in enumerate(sizes):
            if x + w + pad > width and x > pad:
                y += shelf_h + pad
                x = pad
                shelf_h = 0
            placed.append((i, x, y, w, h))
            x += w + pad
            shelf_h = max(shelf_h, h)
        y += shelf_h + 2 * pad
        out.append({"mineral": name, "header_y": header_y, "tiles": placed})
    return {"order": [b["mineral"] for b in out], "height": y, "bands": out}


def _font(font_px: int):
    from PIL import ImageFont

    try:
        return ImageFont.load_default(size=font_px)
    except TypeError:
        return ImageFont.load_default()


def _no_grains_image() -> np.ndarray:
    from PIL import Image, ImageDraw

    img = Image.fromarray(np.full((60, 200, 3), 255, np.uint8))
    ImageDraw.Draw(img).text((8, 18), "no grains", fill=(0, 0, 0), font=_font(20))
    return np.asarray(img).copy()


def _scale_tile(tile: np.ndarray, scale: float) -> np.ndarray:
    if scale == 1.0:
        return tile
    from PIL import Image

    h, w = tile.shape[:2]
    resized = Image.fromarray(tile).resize((max(1, int(round(w * scale))), max(1, int(round(h * scale)))), Image.NEAREST)
    return np.asarray(resized)


def make_knolling(
    gray: np.ndarray,
    polys: Sequence,
    table: pd.DataFrame,
    colors: Dict[str, RGB],
    *,
    alpha: float = 0.5,
    width_px: Optional[int] = None,
    pad: int = 12,
    font_px: int = DEFAULT_FONT_PX,
    scale: float = DEFAULT_SCALE,
) -> np.ndarray:
    """
    RGB uint8 image on white. One band per mineral (most grains first, Unknown last),
    header ``"{mineral}  n={n}"``, tiles sorted by ``ECD_px`` descending and
    shelf-packed. ``width_px`` None uses a fixed 5500 px sheet, so the sheet grows
    taller instead of wider. Every tile is drawn at ``scale`` (default 2) so grains
    stay in proportion and read larger than the raw pixels.
    ``table`` rows align with ``polys`` and need ``mineral``, ``ECD_px``, ``orientation_deg``.
    """
    from PIL import Image, ImageDraw

    if len(polys) != len(table):
        raise ValueError(f"polys ({len(polys)}) and table ({len(table)}) length mismatch")
    tab = table.reset_index(drop=True)
    minerals = tab["mineral"].tolist()
    sizes = tab["ECD_px"].to_numpy(float)
    orients = tab["orientation_deg"].to_numpy(float)

    groups: Dict[str, List[int]] = {}
    for i, poly in enumerate(polys):
        if not (np.isfinite(sizes[i]) and np.isfinite(orients[i])):
            continue
        if _largest_part(poly) is None:
            continue
        name = "Unknown" if pd.isna(minerals[i]) else str(minerals[i])
        groups.setdefault(name, []).append(i)
    if not groups:
        return _no_grains_image()

    # Tiles (rows whose tile collapses to nothing are dropped).
    tiles: Dict[str, List[np.ndarray]] = {}
    for name, idxs in groups.items():
        idxs = sorted(idxs, key=lambda k: -sizes[k])
        rgb = tuple(colors.get(name, _GRAY))
        got = []
        for k in idxs:
            t = grain_tile(gray, polys[k], rgb, orients[k], alpha=alpha)
            if t.shape[0] == 1 and t.shape[1] == 1 and t[0, 0, 3] == 0:
                continue
            got.append(_scale_tile(t, float(scale)))
        if got:
            tiles[name] = got
    if not tiles:
        return _no_grains_image()

    order = _band_order([n for n, ts in tiles.items() for _ in ts])
    width = int(width_px) if width_px else DEFAULT_SHEET_WIDTH_PX
    widest = max(t.shape[1] for ts in tiles.values() for t in ts)
    width = max(width, widest + 2 * pad)

    lay = _layout([(n, [t.shape[:2] for t in tiles[n]]) for n in order], width, pad, font_px)
    canvas = np.full((lay["height"], width, 3), 255, np.uint8)
    for band in lay["bands"]:
        ts = tiles[band["mineral"]]
        for i, x, y, w, h in band["tiles"]:
            t = ts[i]
            a = t[..., 3:4].astype(float) / 255.0
            region = canvas[y : y + h, x : x + w].astype(float)
            canvas[y : y + h, x : x + w] = np.clip(np.rint(t[..., :3] * a + region * (1 - a)), 0, 255).astype(np.uint8)

    img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img)
    font = _font(font_px)
    for band in lay["bands"]:
        n = len(tiles[band["mineral"]])
        draw.text((pad, band["header_y"] + pad), f"{band['mineral']}  n={n}", fill=(0, 0, 0), font=font)
    return np.asarray(img).copy()


def save_knolling(arr: np.ndarray, path: PathLike, dpi: int = 150) -> Path:
    """``Image.fromarray(arr).save(path, dpi=(dpi, dpi))`` at native pixels."""
    from PIL import Image

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    d = int(dpi)
    Image.fromarray(arr).save(out, dpi=(d, d))
    return out

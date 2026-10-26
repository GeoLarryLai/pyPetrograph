"""
Cut dark (epoxy / hole) pixels out of SEG polygons and drop mostly-dark grains.

Pure numpy / shapely / scikit-image. Imported by ``seg_worker.py`` as a sibling
file (no ``pyPetrograph`` import here) and usable from the kernel.
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np


def _parts(geom) -> List:
    """Non-empty Polygon parts of a Polygon / Multi* / GeometryCollection."""
    if geom is None or getattr(geom, "is_empty", True):
        return []
    if geom.geom_type == "Polygon":
        return [geom]
    return [p for g in getattr(geom, "geoms", []) for p in _parts(g)]


def clean_polygons(
    polys: Sequence,
    dark: np.ndarray,
    *,
    drop_frac: float = 0.5,
    min_area: float = 25.0,
) -> List:
    """
    Per polygon: ``dark_frac`` = share of its pixels where ``dark`` is True.
    Drop if ``dark_frac > drop_frac``; else remove dark pixels, keep the largest
    connected part, re-trace it, and drop it if smaller than ``min_area`` px.

    A polygon with no dark pixel is returned unchanged (no re-trace shrink).
    Coordinates: x = column, y = row; integer values are pixel centers.
    """
    from shapely.geometry import Polygon
    from skimage.draw import polygon as draw_polygon
    from skimage.measure import find_contours, label

    dark = np.asarray(dark, dtype=bool)
    h, w = dark.shape[:2]
    out: List = []
    for geom in polys:
        parts = _parts(geom)
        if not parts:
            continue
        minx = min(p.bounds[0] for p in parts)
        miny = min(p.bounds[1] for p in parts)
        maxx = max(p.bounds[2] for p in parts)
        maxy = max(p.bounds[3] for p in parts)
        x0 = max(int(np.floor(minx)) - 1, 0)
        y0 = max(int(np.floor(miny)) - 1, 0)
        x1 = min(int(np.ceil(maxx)) + 2, w)
        y1 = min(int(np.ceil(maxy)) + 2, h)
        if x1 <= x0 or y1 <= y0:
            continue
        mask = np.zeros((y1 - y0, x1 - x0), dtype=bool)
        for p in parts:
            xy = np.asarray(p.exterior.coords)
            rr, cc = draw_polygon(xy[:, 1] - y0, xy[:, 0] - x0, shape=mask.shape)
            mask[rr, cc] = True
        n_px = int(mask.sum())
        if n_px == 0:
            continue
        dcrop = dark[y0:y1, x0:x1]
        n_dark = int((mask & dcrop).sum())
        if n_dark / n_px > drop_frac:
            continue
        if n_dark == 0:
            if geom.area >= min_area:
                out.append(geom)
            continue
        keep = mask & ~dcrop
        lab = label(keep, connectivity=1)
        if lab.max() == 0:
            continue
        counts = np.bincount(lab.ravel())
        counts[0] = 0
        biggest = int(np.argmax(counts))
        if counts[biggest] < min_area:
            continue
        crop = np.pad(lab == biggest, 1).astype(np.float32)
        contours = find_contours(crop, 0.5)
        if not contours:
            continue
        c = max(contours, key=len)
        if len(c) < 4:
            continue
        pts = np.column_stack([c[:, 1] - 1 + x0, c[:, 0] - 1 + y0])
        poly = Polygon(pts)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.geom_type != "Polygon":
            ps = _parts(poly)
            if not ps:
                continue
            poly = max(ps, key=lambda q: q.area)
        if poly.is_empty or poly.area < min_area:
            continue
        out.append(poly)
    return out

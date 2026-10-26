"""
Grain a / b axes from polygon moments (ellipse with the same second moments), plus the
effective diameters D and ECD.

a = long axis, b = short axis, D = sqrt(a * b), ECD = sqrt(4 * A / pi) with A the projected area.

Port of SegmentEveryGrain ``measure_polygon`` + ``measure_ellipse``
(``segmenteverygrain/interactions.py``); pure numpy + shapely, no TensorFlow.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

MEASURE_COLUMNS = (
    "grain_id",
    "area_px",
    "a_px",
    "b_px",
    "D_px",
    "ECD_px",
    "orientation_deg",
    "centroid_x",
    "centroid_y",
    "area_um2",
    "a_um",
    "b_um",
    "D_um",
    "ECD_um",
)

# measure_grains fills these; add_size_metrics adds the rest
_MEASURED = tuple(c for c in MEASURE_COLUMNS if c not in ("D_px", "ECD_px", "D_um", "ECD_um"))

_MOMENT_KEYS = ("area", "cx", "cy", "Ixx", "Iyy", "Ixy")


def _nan_moments() -> Dict[str, float]:
    return {key: float("nan") for key in _MOMENT_KEYS}


def _polygon_parts(geom) -> list:
    """Polygons inside ``geom``, dropping empty parts. MultiPolygons are flattened."""
    if geom is None:
        return []
    kind = getattr(geom, "geom_type", None)
    if kind is None:
        return []
    try:
        if geom.is_empty:
            return []
    except Exception:
        return []
    if kind == "Polygon":
        return [geom]
    if kind in ("MultiPolygon", "GeometryCollection"):
        parts: list = []
        try:
            children = list(geom.geoms)
        except Exception:
            return []
        for child in children:
            parts.extend(_polygon_parts(child))
        return parts
    return []


def _largest_polygon(geom):
    parts = _polygon_parts(geom)
    if not parts:
        return None
    best = parts[0]
    best_area = float("-inf")
    for part in parts:
        try:
            area = float(part.area)
        except Exception:
            continue
        if np.isfinite(area) and area > best_area:
            best = part
            best_area = area
    return best


def polygon_moments(poly) -> Dict[str, float]:
    """
    Area, centroid and central second moments of a polygon (largest part of a MultiPolygon).

    Returns ``{"area", "cx", "cy", "Ixx", "Iyy", "Ixy"}`` in pixel units (x = column, y = row).
    """
    part = _largest_polygon(poly)
    if part is None:
        return _nan_moments()
    try:
        # Same winding fix as SEG: normalize, reverse, exterior ring only (holes ignored).
        polygon = part.normalize().reverse()
        coords = np.asarray(polygon.exterior.coords, dtype=float)
    except Exception:
        return _nan_moments()
    if coords.ndim != 2 or coords.shape[0] < 4 or coords.shape[1] < 2:
        return _nan_moments()

    x = coords[:-1, 0]
    y = coords[:-1, 1]
    x_next = coords[1:, 0]
    y_next = coords[1:, 1]
    common = x * y_next - x_next * y
    area = 0.5 * np.sum(common)
    if not np.isfinite(area) or abs(float(area)) == 0.0:
        return _nan_moments()

    cx = np.sum((x + x_next) * common) / (6.0 * area)
    cy = np.sum((y + y_next) * common) / (6.0 * area)
    ixx_origin = np.sum((y**2 + y * y_next + y_next**2) * common) / 12.0
    iyy_origin = np.sum((x**2 + x * x_next + x_next**2) * common) / 12.0
    ixy_origin = np.sum(
        (x * y_next + 2.0 * x * y + 2.0 * x_next * y_next + x_next * y) * common
    ) / 24.0
    ixx = ixx_origin - area * cy**2
    iyy = iyy_origin - area * cx**2
    ixy = ixy_origin - area * cx * cy
    return {
        "area": float(abs(area)),
        "cx": float(cx),
        "cy": float(cy),
        "Ixx": float(abs(ixx)),
        "Iyy": float(abs(iyy)),
        "Ixy": float(ixy),
    }


def _finite_lambda(value: float) -> Optional[float]:
    """Keep a principal moment; treat tiny negatives as 0, larger negatives as invalid."""
    if not np.isfinite(value):
        return None
    if value < 0.0:
        if value > -1e-6:
            return 0.0
        return None
    return float(value)


def ellipse_axes(moments: Dict[str, float]) -> Dict[str, float]:
    """
    Ellipse with the same second moments.

    Returns ``{"long", "short", "orientation_deg"}``; lengths are full axis lengths.
    ``orientation_deg`` uses the SEG / skimage ``regionprops`` convention.
    """
    missing = {"long": float("nan"), "short": float("nan"), "orientation_deg": float("nan")}
    try:
        area = float(moments["area"])
        ixx = float(moments["Ixx"])
        iyy = float(moments["Iyy"])
        ixy = float(moments["Ixy"])
    except (KeyError, TypeError, ValueError):
        return missing
    if not np.isfinite(area) or area <= 0.0:
        return missing
    if not (np.isfinite(ixx) and np.isfinite(iyy) and np.isfinite(ixy)):
        return missing

    theta = 0.5 * np.arctan2(2.0 * ixy, ixx - iyy)
    common = np.sqrt(((ixx - iyy) / 2.0) ** 2 + ixy**2)
    lambda_major = _finite_lambda((ixx + iyy) / 2.0 + common)
    lambda_minor = _finite_lambda((ixx + iyy) / 2.0 - common)
    if lambda_major is None or lambda_minor is None:
        return missing

    long = 4.0 * np.sqrt(lambda_major / area)
    short = 4.0 * np.sqrt(lambda_minor / area)
    if short > long:
        long, short = short, long
    return {
        "long": float(long),
        "short": float(short),
        "orientation_deg": float(np.degrees(theta)),
    }


def measure_grains(polys: Sequence, um_per_px: Optional[float] = None) -> pd.DataFrame:
    """
    One row per polygon with ``MEASURE_COLUMNS``; ``grain_id`` is 1..N in input order.

    ``area_um2`` / ``a_um`` / ``b_um`` are NaN when ``um_per_px`` is None. D and ECD come from
    ``add_size_metrics``.
    """
    scale = None if um_per_px is None else float(um_per_px)
    rows = []
    for grain_id, poly in enumerate(polys, start=1):
        moments = polygon_moments(poly)
        axes = ellipse_axes(moments)
        area_px = moments["area"]
        a_px = axes["long"]
        b_px = axes["short"]
        if scale is None:
            area_um2 = a_um = b_um = float("nan")
        else:
            area_um2 = area_px * scale * scale
            a_um = a_px * scale
            b_um = b_px * scale
        rows.append(
            {
                "grain_id": grain_id,
                "area_px": area_px,
                "a_px": a_px,
                "b_px": b_px,
                "orientation_deg": axes["orientation_deg"],
                "centroid_x": moments["cx"],
                "centroid_y": moments["cy"],
                "area_um2": area_um2,
                "a_um": a_um,
                "b_um": b_um,
            }
        )
    return pd.DataFrame(rows, columns=list(_MEASURED))


def add_size_metrics(table: pd.DataFrame, um_per_px: Optional[float] = None) -> pd.DataFrame:
    """
    Copy of ``table`` with ``D`` and ``ECD`` added, in pixels and (with a scale) in µm.

    ``D = sqrt(a * b)`` and ``ECD = sqrt(4 * A / pi)`` with A = ``area_px`` / ``area_um2``.
    Needs ``a_px``, ``b_px``, ``area_px``. The µm columns are recomputed from the pixel columns
    when ``um_per_px`` is given, and are NaN otherwise. Columns follow ``MEASURE_COLUMNS`` order,
    then any extra columns.
    """
    out = table.copy()
    a_px = pd.to_numeric(out["a_px"], errors="coerce").to_numpy(dtype=float)
    b_px = pd.to_numeric(out["b_px"], errors="coerce").to_numpy(dtype=float)
    area_px = pd.to_numeric(out["area_px"], errors="coerce").to_numpy(dtype=float)
    with np.errstate(invalid="ignore"):
        out["D_px"] = np.sqrt(a_px * b_px)
        out["ECD_px"] = np.sqrt(4.0 * area_px / np.pi)
    if um_per_px is None:
        for col in ("area_um2", "a_um", "b_um", "D_um", "ECD_um"):
            out[col] = float("nan")
    else:
        s = float(um_per_px)
        out["area_um2"] = area_px * s * s
        for src, dst in (("a_px", "a_um"), ("b_px", "b_um"), ("D_px", "D_um"), ("ECD_px", "ECD_um")):
            out[dst] = out[src] * s
    first = [c for c in MEASURE_COLUMNS if c in out.columns]
    rest = [c for c in out.columns if c not in first]
    return out[first + rest]

"""Polygon geometry helpers (GeoJSON, rasterize, magic wand)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import geopandas as gpd
import numpy as np
from affine import Affine
from rasterio.features import rasterize, shapes as rasterio_shapes
from rasterio.transform import from_bounds
from scipy.ndimage import label as ndimage_label
from shapely.geometry import Point, Polygon, mapping, shape

from pyPetrograph.common.constants import COORD_DECIMALS, DEFAULT_WAND_SIMPLIFY, DEFAULT_WAND_THRESHOLD, PathLike

def remove_duplicate_polys(polygons: Sequence[Polygon]) -> List[Polygon]:
    """Deduplicate shapely polygons by WKB hex."""
    unique: List[Polygon] = []
    seen = set()
    for poly in polygons:
        key = poly.wkb_hex
        if key not in seen:
            unique.append(poly)
            seen.add(key)
    return unique


def rasterize_polygons(
    polygons: Sequence[Polygon],
    class_ids: Sequence[int],
    shape_hw: Tuple[int, int],
) -> np.ndarray:
    """
    Rasterize polygons into a label map (uint8). Later polygons overwrite earlier.
    """
    h, w = shape_hw
    if len(polygons) == 0:
        return np.zeros((h, w), dtype=np.uint8)

    bounds = (0, h, w, 0)  # left, bottom, right, top
    transform = from_bounds(*bounds, w, h)
    shapes = ((mapping(poly), int(cid)) for poly, cid in zip(polygons, class_ids))
    rasterized = rasterize(
        shapes,
        out_shape=(h, w),
        transform=transform,
        fill=0,
        dtype="uint8",
    )
    return rasterized


def round_polygon_coords(poly: Polygon, ndigits: int = COORD_DECIMALS) -> Polygon:
    """Return a copy of ``poly`` with exterior coords rounded for compact GeoJSON."""
    coords = [
        (round(float(x), ndigits), round(float(y), ndigits))
        for x, y in poly.exterior.coords
    ]
    return Polygon(coords)


def save_polygons_geojson(
    path: PathLike,
    polygons: Sequence[Polygon],
    class_ids: Sequence[int],
    ndigits: int = COORD_DECIMALS,
) -> Path:
    """Save polygons as GeoJSON with a ``class`` column (compact rounded coords)."""
    path = Path(path)
    features = []
    for poly, cid in zip(polygons, class_ids):
        rounded = round_polygon_coords(poly, ndigits=ndigits)
        features.append(
            {
                "type": "Feature",
                "properties": {"class": int(cid)},
                "geometry": mapping(rounded),
            }
        )
    fc = {"type": "FeatureCollection", "features": features}
    path.write_text(json.dumps(fc, separators=(",", ":")))
    return path


def magic_wand_to_polygon(
    image: np.ndarray,
    row: int,
    col: int,
    threshold: float = DEFAULT_WAND_THRESHOLD,
    simplify_tol: float = DEFAULT_WAND_SIMPLIFY,
) -> Optional[List[Tuple[float, float]]]:
    """
    Grow a similar-color region from (row, col) and return a simplified polygon.

    Parameters
    ----------
    image : uint8 HxWx3
    row, col : seed pixel (image indices)
    threshold : max RGB Euclidean distance from seed
    simplify_tol : shapely simplify tolerance in pixels

    Returns
    -------
    list of (x, y) vertices in image pixel coords, or None if no region.
    """
    img = np.asarray(image, dtype=np.uint8)
    h, w = img.shape[:2]
    if not (0 <= row < h and 0 <= col < w):
        return None

    seed = img[row, col].astype(np.float32)
    diff = np.sqrt(((img.astype(np.float32) - seed) ** 2).sum(axis=-1))
    mask = diff <= float(threshold)

    labeled, nlab = ndimage_label(mask)
    if nlab == 0:
        return None
    cid = int(labeled[row, col])
    if cid == 0:
        return None
    region = (labeled == cid).astype(np.uint8)

    geoms = []
    for geom, val in rasterio_shapes(region, mask=region.astype(bool), transform=Affine.identity()):
        if int(val) != 1:
            continue
        g = shape(geom)
        if g.is_empty:
            continue
        geoms.append(g)

    if not geoms:
        return None

    # Prefer polygon that contains the seed (x=col, y=row).
    seed_pt = (float(col) + 0.5, float(row) + 0.5)
    containing = [g for g in geoms if g.contains(Point(seed_pt)) or g.covers(Point(seed_pt))]
    poly = max(containing or geoms, key=lambda g: g.area)
    if not isinstance(poly, Polygon):
        if hasattr(poly, "geoms"):
            parts = [g for g in poly.geoms if isinstance(g, Polygon)]
            if not parts:
                return None
            poly = max(parts, key=lambda g: g.area)
        else:
            return None

    poly = poly.simplify(float(simplify_tol), preserve_topology=True)
    if poly.is_empty or not isinstance(poly, Polygon):
        return None
    if poly.exterior is None or len(poly.exterior.coords) < 4:
        return None

    verts = [
        (round(float(x), COORD_DECIMALS), round(float(y), COORD_DECIMALS))
        for x, y in list(poly.exterior.coords)[:-1]
    ]
    if len(verts) < 3:
        return None
    return verts


def load_polygons_geojson(path: PathLike) -> Tuple[List[Polygon], List[int]]:
    """Load GeoJSON → (polygons, class_ids)."""
    gdf = gpd.read_file(path)
    polys: List[Polygon] = []
    ids: List[int] = []
    has_class = "class" in gdf.columns
    for idx, geom in enumerate(gdf.geometry):
        if geom is None or geom.is_empty:
            continue
        poly = None
        if isinstance(geom, Polygon):
            poly = geom
        else:
            mapped = shape(mapping(geom))
            if isinstance(mapped, Polygon):
                poly = mapped
            elif hasattr(mapped, "geoms"):
                parts = [g for g in mapped.geoms if isinstance(g, Polygon)]
                if parts:
                    poly = max(parts, key=lambda g: g.area)
        if poly is None:
            continue
        polys.append(poly)
        if has_class:
            ids.append(int(gdf.iloc[idx]["class"]))
        else:
            ids.append(1)
    return polys, ids

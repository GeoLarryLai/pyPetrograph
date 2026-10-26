"""GeoJSON grain files and per-mineral CSVs."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from shapely.geometry import Polygon, mapping, shape

from pyPetrograph.amics.naming import mineral_csv
from pyPetrograph.common.constants import PathLike

MINERAL_CSV_COLUMNS = (
    "grain_id",
    "mineral",
    "purity",
    "mixed",
    "a_px",
    "b_px",
    "D_px",
    "ECD_px",
    "a_um",
    "b_um",
    "D_um",
    "ECD_um",
    "area_px",
    "area_um2",
    "orientation_deg",
    "centroid_x",
    "centroid_y",
)


def _json_key(key: Any) -> str:
    if isinstance(key, str):
        return key
    converted = _jsonable(key)
    if isinstance(converted, str):
        return converted
    return str(converted)


def _jsonable(value: Any) -> Any:
    """Plain JSON value: numpy scalars, paths, NaN/inf, and nested containers."""
    if value is None or value is pd.NA:
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {_json_key(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _jsonable(value.item())
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float):
        number = float(value)
        if not math.isfinite(number):
            return None
        return number
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, str):
        return value
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, (bool, np.bool_)) and bool(missing):
        return None
    return str(value)


def _geometry_json(poly: Any) -> Any:
    if poly is None or bool(getattr(poly, "is_empty", False)):
        return None
    return _jsonable(mapping(poly))


def _properties_table(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    if not any(rows):
        return pd.DataFrame(index=range(len(rows)))
    frame = pd.DataFrame(rows)
    for column in frame.columns:
        values = frame[column].tolist()
        if any(value is None for value in values):
            frame[column] = [np.nan if value is None else value for value in values]
    return frame


def _mineral_label(value: Any) -> str:
    if value is None or value is pd.NA:
        return "Unknown"
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return "Unknown"
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, (bool, np.bool_)) and bool(missing):
        return "Unknown"
    return str(value)


def _unique_csv(path: Path, used: set[Path]) -> Path:
    if path not in used:
        return path
    number = 2
    while True:
        candidate = path.with_name(f"{path.stem}_{number}.csv")
        if candidate not in used:
            return candidate
        number += 1


def write_grains_geojson(path: PathLike, polys: Sequence, table: pd.DataFrame, meta: Dict[str, Any]) -> Path:
    """
    FeatureCollection, one Feature per polygon (properties = matching ``table`` row,
    NaN -> null). ``meta`` keys become top-level members
    (``sample``, ``um_per_px``, ``scale``, ``threshold``, ``source_image``).
    """
    n_poly = len(polys)
    n_row = len(table)
    if n_poly != n_row:
        raise ValueError(f"len(polys) ({n_poly}) != len(table) ({n_row})")

    columns = list(table.columns)
    features = []
    for i, poly in enumerate(polys):
        row = table.iloc[i]
        features.append(
            {
                "type": "Feature",
                "id": i,
                "geometry": _geometry_json(poly),
                "properties": {str(column): _jsonable(row[column]) for column in columns},
            }
        )

    body = {str(key): _jsonable(value) for key, value in meta.items() if key not in ("type", "features")}
    document = {"type": "FeatureCollection", **body, "features": features}
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(document, ensure_ascii=False, indent=1, allow_nan=False), encoding="utf-8")
    return out


def read_grains_geojson(path: PathLike) -> Tuple[List[Any], pd.DataFrame, Dict[str, Any]]:
    """``(polygons, properties table, top-level meta)``."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    raw_features = document.get("features") or []
    polys: List[Any] = []
    rows: List[Dict[str, Any]] = []
    for feature in raw_features:
        if not isinstance(feature, dict):
            polys.append(Polygon())
            rows.append({})
            continue
        geometry = feature.get("geometry")
        polys.append(Polygon() if geometry is None else shape(geometry))
        properties = feature.get("properties")
        rows.append(properties if isinstance(properties, dict) else {})
    meta = {key: value for key, value in document.items() if key not in ("type", "features")}
    return polys, _properties_table(rows), meta


def write_mineral_csvs(table: pd.DataFrame, sample: str, folder: PathLike) -> List[Path]:
    """One ``{sample}_{safe mineral}.csv`` per mineral with ``MINERAL_CSV_COLUMNS`` (blank when no scale)."""
    if "mineral" not in table.columns:
        raise KeyError("mineral")

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    frame = table.reindex(columns=list(MINERAL_CSV_COLUMNS)).copy()
    order: List[str] = []
    buckets: Dict[str, List[int]] = {}
    for position, value in enumerate(frame["mineral"].tolist()):
        label = _mineral_label(value)
        if label not in buckets:
            buckets[label] = []
            order.append(label)
        buckets[label].append(position)

    written: List[Path] = []
    used: set[Path] = set()
    for label in order:
        destination = _unique_csv(mineral_csv(folder, sample, label), used)
        used.add(destination)
        part = frame.iloc[buckets[label]].sort_values("a_px", ascending=False, na_position="last", kind="stable")
        part.to_csv(destination, index=False, encoding="utf-8", na_rep="")
        written.append(destination)
    return written

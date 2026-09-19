"""
Stage 4: one table row per grain (and leftover pore/cement/matrix blob).

Each row holds shape (area, axes, roundness) plus mean/std/p10/p90 of every
channel inside that object. Point-count statistics (stage 6) and LightGBM
naming (stage 5) both read this CSV.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from pyPetrograph.common.constants import PathLike
from pyPetrograph.common.paths import stem_paths
from pyPetrograph.fuse.stack import ChannelStack

DEFAULT_MIN_LEFTOVER_PX = 40

_SHAPE_COLS = (
    "object_id",
    "kind",
    "area_px",
    "equiv_diam",
    "major",
    "minor",
    "orientation",
    "eccentricity",
    "solidity",
    "perimeter",
    "circularity",
)


def _channel_stats(stack: np.ndarray, mask: np.ndarray) -> List[float]:
    """mean, std, p10, p90 per channel inside ``mask``; zeros if empty."""
    pix = stack[mask]
    if pix.size == 0:
        c = int(stack.shape[-1])
        return [0.0] * (4 * c)
    out: List[float] = []
    for c in range(stack.shape[-1]):
        v = pix[:, c]
        out.extend(
            [
                float(v.mean()),
                float(v.std()),
                float(np.percentile(v, 10)),
                float(np.percentile(v, 90)),
            ]
        )
    return out


def _channel_colnames(names: Sequence[str]) -> List[str]:
    cols: List[str] = []
    for n in names:
        cols.extend([f"{n}_mean", f"{n}_std", f"{n}_p10", f"{n}_p90"])
    return cols


def build_object_table(
    grain_labels: np.ndarray,
    stack: np.ndarray,
    names: Sequence[str],
    valid: np.ndarray,
    *,
    px_per_um: Optional[float] = None,
    min_leftover_px: int = DEFAULT_MIN_LEFTOVER_PX,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Instance grains + leftover blobs → table and a combined id map.

    Parameters
    ----------
    grain_labels : HxW instance ids from GrainPlot QC (0 = leftover)
    stack : HxWxC channel stack (same grid)
    names : C channel names
    valid : HxW coverage mask
    px_per_um : pixels per micrometre; if set, add ``major_um`` and ``phi``
        (Folk φ = −log2(major in mm))
    min_leftover_px : drop leftover blobs smaller than this

    Returns
    -------
    table : one row per object
    objects_mask : HxW uint16, leftover ids start after max grain id
    """
    from skimage.measure import label as cc_label, regionprops

    grains = np.asarray(grain_labels)
    stack = np.asarray(stack, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    h, w = grains.shape[:2]
    if stack.shape[:2] != (h, w):
        raise ValueError(f"stack {stack.shape[:2]} vs grain_labels {(h, w)}")
    names = [str(n) for n in names]
    if int(stack.shape[-1]) != len(names):
        raise ValueError(f"{len(names)} names for {stack.shape[-1]} channels")

    grain_bin = grains > 0
    leftover_bin = valid & ~grain_bin
    leftover = cc_label(leftover_bin).astype(np.int32)
    if int(min_leftover_px) > 1:
        from skimage.morphology import remove_small_objects

        leftover = np.asarray(
            remove_small_objects(leftover, min_size=int(min_leftover_px)),
            dtype=np.int32,
        )

    offset = int(grains.max())
    objects = grains.astype(np.int32).copy()
    leftover_ids = leftover > 0
    objects[leftover_ids] = leftover[leftover_ids] + offset

    grain_id_set = set(int(v) for v in np.unique(grains) if v > 0)
    ch_cols = _channel_colnames(names)
    rows = []
    for prop in regionprops(objects):
        oid = int(prop.label)
        kind = "grain" if oid in grain_id_set else "leftover"
        area = float(prop.area)
        peri = float(prop.perimeter) if prop.perimeter else 0.0
        circ = float(4.0 * np.pi * area / (peri * peri)) if peri > 0 else 0.0
        minr, minc, maxr, maxc = prop.bbox
        crop_mask = objects[minr:maxr, minc:maxc] == oid
        crop_stack = stack[minr:maxr, minc:maxc]
        stats = _channel_stats(crop_stack, crop_mask)
        row = {
            "object_id": oid,
            "kind": kind,
            "area_px": area,
            "equiv_diam": float(prop.equivalent_diameter),
            "major": float(prop.major_axis_length),
            "minor": float(prop.minor_axis_length),
            "orientation": float(prop.orientation),
            "eccentricity": float(prop.eccentricity),
            "solidity": float(prop.solidity),
            "perimeter": peri,
            "circularity": circ,
        }
        for col, val in zip(ch_cols, stats):
            row[col] = val
        if px_per_um is not None and float(px_per_um) > 0:
            major_um = float(prop.major_axis_length) / float(px_per_um)
            row["major_um"] = major_um
            d_mm = major_um / 1000.0
            row["phi"] = float(-np.log2(d_mm)) if d_mm > 0 else np.nan
        rows.append(row)

    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values("object_id").reset_index(drop=True)
        cols = [c for c in _SHAPE_COLS if c in df.columns]
        extra = [c for c in df.columns if c not in cols]
        df = df[cols + extra]
    out_mask = objects.astype(np.uint16) if int(objects.max()) <= 65535 else objects.astype(np.int32)
    return df, out_mask


def build_object_table_from_stack(
    grain_labels: np.ndarray,
    cs: ChannelStack,
    *,
    px_per_um: Optional[float] = None,
    min_leftover_px: int = DEFAULT_MIN_LEFTOVER_PX,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """``build_object_table`` using a ``ChannelStack``."""
    return build_object_table(
        grain_labels,
        cs.stack,
        cs.names,
        cs.valid,
        px_per_um=px_per_um,
        min_leftover_px=min_leftover_px,
    )


def save_object_table(
    table: pd.DataFrame,
    objects_mask: np.ndarray,
    image_path: PathLike,
) -> Tuple[Path, Path]:
    """Write ``objects/{stem}_objects.csv`` and ``objects/{stem}_objects_mask.png``."""
    from PIL import Image

    sp = stem_paths(image_path)
    sp.objects_dir.mkdir(parents=True, exist_ok=True)
    csv_p = sp.objects_csv
    mask_p = sp.objects_mask
    table.to_csv(csv_p, index=False)
    arr = np.asarray(objects_mask)
    if int(arr.max()) <= 255:
        Image.fromarray(arr.astype(np.uint8), mode="L").save(mask_p)
    else:
        Image.fromarray(arr.astype(np.uint16), mode="I;16").save(mask_p)
    return csv_p, mask_p


def load_object_table(image_path: PathLike) -> Optional[Tuple[pd.DataFrame, np.ndarray]]:
    """Reload CSV + mask, or ``None`` if either file is missing."""
    from PIL import Image

    sp = stem_paths(image_path)
    if not sp.objects_csv.exists() or not sp.objects_mask.exists():
        return None
    table = pd.read_csv(sp.objects_csv)
    arr = np.asarray(Image.open(sp.objects_mask))
    if arr.ndim > 2:
        arr = arr[..., 0]
    return table, arr

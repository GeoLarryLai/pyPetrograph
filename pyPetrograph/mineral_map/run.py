"""
Per-image pass and parallel folder driver for flat-color mineral maps.

Outputs beside the images in ``mineralmap/``::

    {stem}_grains.csv        one row per grain
    {stem}_grains.geojson    polygons (if write_geojson)
    {stem}_class_rgb.png     overlay preview (legend colors)
    {stem}_counts.csv        per-class counts with uncertainty
    {stem}_unlisted.csv      colors grouped into Unknown (audit table)
    all_counts.csv           every image stacked
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from PIL import Image

from pyPetrograph.common.constants import PathLike
from pyPetrograph.common.paths import get_output_dir
from pyPetrograph.mineral_map.counts import count_grains
from pyPetrograph.mineral_map.grains import (
    absorb_class_specks,
    absorb_specks,
    class_rgb,
    grain_table,
    grains_to_geojson,
    label_grains,
)
from pyPetrograph.mineral_map.legend import Legend, RGB, read_rgba, snap_to_legend, unknown_code

MINERALMAP_SUBDIR = "mineralmap"
ALL_COUNTS_NAME = "all_counts.csv"


def mineralmap_paths(image_path: PathLike, output_dir: Optional[PathLike] = None) -> Dict[str, Path]:
    """Output paths for one image (folder ``mineralmap/`` beside the image, or under ``output_dir``)."""
    p = Path(image_path)
    base = Path(output_dir) if output_dir is not None else get_output_dir()
    d = (base if base is not None else p.parent) / MINERALMAP_SUBDIR
    stem = p.stem
    return {
        "dir": d,
        "grains_csv": d / f"{stem}_grains.csv",
        "geojson": d / f"{stem}_grains.geojson",
        "class_rgb": d / f"{stem}_class_rgb.png",
        "counts_csv": d / f"{stem}_counts.csv",
        "unlisted_csv": d / f"{stem}_unlisted.csv",
        "all_counts": d / ALL_COUNTS_NAME,
    }


def process_image(
    image_path: PathLike,
    legend: Legend,
    *,
    color_tol: float = 40.0,
    white_tol: float = 40.0,
    speck_px: int = 30,
    unknown_speck_px: int = 50,
    min_grain_px: int = 100,
    split: bool = True,
    bridge_px: int = 4,
    h_min: float = 3.0,
    notch_solidity: float = 0.85,
    notch_halves: float = 0.88,
    unknown_frac: float = 0.95,
    display_rgb: Optional[Dict[str, RGB]] = None,
    write_geojson: bool = True,
    output_dir: Optional[PathLike] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Full pass for one image: snap → specks → grains → table → files.

    Returns the per-class counts table (small); everything else is written to
    ``mineralmap/`` (beside the image, or under ``output_dir``). Grains smaller than ``min_grain_px`` are dropped from the
    saved table / polygons; counts also report ``min_grain_px / 2`` and
    ``* 2`` as a method range. ``bridge_px`` / ``h_min`` / ``notch_solidity`` /
    ``notch_halves`` are the split rules of ``label_grains``. ``unknown_frac``
    is the Unknown-pixel share that marks a grain Unknown. ``display_rgb``
    overrides overlay colors only.
    """
    t0 = time.time()
    p = Path(image_path)
    paths = mineralmap_paths(p, output_dir=output_dir)
    paths["dir"].mkdir(parents=True, exist_ok=True)

    rgb, alpha = read_rgba(p)
    cmap, valid, unlisted = snap_to_legend(rgb, alpha, legend, tol=color_tol, white_tol=white_tol)
    del rgb, alpha
    unk = unknown_code(legend)
    cmap = absorb_specks(cmap, valid, speck_px=speck_px)
    cmap = absorb_class_specks(cmap, valid, unk, speck_px=unknown_speck_px)
    floor_px = max(1, int(min_grain_px) // 2)
    labels, labels_nosplit = label_grains(
        cmap,
        valid,
        unk,
        min_grain_px=min_grain_px,
        split=split,
        bridge_px=bridge_px,
        h_min=h_min,
        notch_solidity=notch_solidity,
        notch_halves=notch_halves,
        floor_px=floor_px,
    )
    table_all = grain_table(labels, cmap, legend, unknown_frac=unknown_frac)
    table_nosplit = grain_table(labels_nosplit, cmap, legend, unknown_frac=unknown_frac)
    del labels_nosplit
    counts = count_grains(table_all, table_nosplit, legend, p.stem, min_grain_px=min_grain_px)

    keep = table_all["area_px"] >= int(min_grain_px)
    table = table_all[keep].reset_index(drop=True)
    keep_ids = np.zeros(int(labels.max()) + 1, dtype=bool)
    keep_ids[table["grain_id"].to_numpy()] = True
    labels = np.where(keep_ids[labels], labels, 0).astype(np.int32)

    table.to_csv(paths["grains_csv"], index=False)
    Image.fromarray(class_rgb(labels, table, legend, display_rgb=display_rgb)).save(paths["class_rgb"])
    for stale in (paths["dir"] / f"{p.stem}_grain_labels.png", paths["dir"] / f"{p.stem}_grain_labels.npy"):
        if stale.exists():
            stale.unlink()
    if write_geojson:
        grains_to_geojson(labels, table, paths["geojson"])
    counts.to_csv(paths["counts_csv"], index=False)
    unlisted.to_csv(paths["unlisted_csv"], index=False)
    if verbose:
        n_tot = int(counts.loc[counts["class"] == "TOTAL", "n"].iloc[0])
        print(f"{p.name}: {n_tot} grains, {len(unlisted)} colors grouped into Unknown, {time.time() - t0:.1f}s")
    return counts


def process_folder(
    image_paths: Sequence[PathLike],
    legend: Legend,
    *,
    n_jobs: int = 4,
    load_saved: bool = True,
    **kw,
) -> pd.DataFrame:
    """
    Run ``process_image`` on many images in parallel (one image per worker).

    With ``load_saved=True`` an image whose ``{stem}_counts.csv`` already exists
    is reloaded, not recomputed. Writes ``mineralmap/all_counts.csv`` beside the
    first image (or under ``output_dir`` / ``get_output_dir()``) and returns
    the stacked counts. Workers get a concrete ``output_dir`` so they do not
    rely on the parent process's ``set_output_dir``.
    """
    paths = [Path(p) for p in image_paths]
    if not paths:
        return pd.DataFrame()
    out_dir = kw.pop("output_dir", None)
    if out_dir is None:
        out_dir = get_output_dir()
    done: List[pd.DataFrame] = []
    todo: List[Path] = []
    for p in paths:
        c = mineralmap_paths(p, output_dir=out_dir)["counts_csv"]
        if load_saved and c.exists():
            done.append(pd.read_csv(c))
        else:
            todo.append(p)
    if todo:
        n = max(1, min(int(n_jobs), len(todo)))
        t0 = time.time()
        results = Parallel(n_jobs=n, backend="loky")(
            delayed(process_image)(p, legend, output_dir=out_dir, **kw) for p in todo
        )
        done.extend(results)
        print(f"processed {len(todo)} image(s) with {n} worker(s) in {time.time() - t0:.1f}s")
    all_counts = pd.concat(done, ignore_index=True)
    order = {p.stem: i for i, p in enumerate(paths)}
    all_counts["_ord"] = all_counts["image"].map(order).fillna(len(order))
    all_counts = all_counts.sort_values(["_ord"], kind="stable").drop(columns="_ord").reset_index(drop=True)
    out = mineralmap_paths(paths[0], output_dir=out_dir)["all_counts"]
    out.parent.mkdir(parents=True, exist_ok=True)
    all_counts.to_csv(out, index=False)
    return all_counts

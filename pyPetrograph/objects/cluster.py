"""
Stage 5: group look-alike objects (PCA + k-means) then name them in a montage.

Clustering uses the numeric columns of the object table (no VGG16 by default).
The montage is SEG's ``ClusterMontageLabeler`` in a Qt subprocess: click a crop
to name it, Shift-click to name the whole look-alike group. Close the window to
write ``objects/{stem}_object_labels.json``.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from pyPetrograph.common.constants import PathLike
from pyPetrograph.common.paths import stem_paths

SKIP_COLS = {
    "object_id",
    "kind",
    "cluster_id",
    "class_id",
    "class_name",
    "proba",
}


def numeric_feature_matrix(
    table: pd.DataFrame,
    *,
    feature_cols: Optional[Sequence[str]] = None,
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    Numeric columns for clustering / LightGBM.

    Returns ``(X, columns, row_ok)`` where ``row_ok`` is a bool mask of finite rows.
    """
    if feature_cols is None:
        cols = [
            c
            for c in table.columns
            if c not in SKIP_COLS and pd.api.types.is_numeric_dtype(table[c])
        ]
    else:
        cols = [str(c) for c in feature_cols]
    if not cols:
        raise ValueError("no numeric columns to cluster on")
    X = table[cols].to_numpy(dtype=np.float64)
    row_ok = np.isfinite(X).all(axis=1)
    return X, cols, row_ok


def cluster_objects(
    table: pd.DataFrame,
    *,
    n_clusters: int = 8,
    n_components: Optional[int] = 12,
    random_state: int = 42,
    feature_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Standardise → optional PCA → k-means. Adds a ``cluster_id`` column (copy).

    Rows with NaNs get ``cluster_id = -1``.
    """
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    out = table.copy()
    X, cols, row_ok = numeric_feature_matrix(out, feature_cols=feature_cols)
    n = int(max(2, n_clusters))
    labels = np.full(len(out), -1, dtype=np.int32)
    if int(row_ok.sum()) < n:
        n = max(1, int(row_ok.sum()))
    if int(row_ok.sum()) >= 2:
        Xs = StandardScaler().fit_transform(X[row_ok])
        n_comp = n_components
        if n_comp is not None:
            n_comp = int(min(n_comp, Xs.shape[0], Xs.shape[1]))
            if n_comp >= 2:
                Xs = PCA(n_components=n_comp, random_state=random_state).fit_transform(Xs)
        km = MiniBatchKMeans(
            n_clusters=min(n, int(row_ok.sum())),
            random_state=int(random_state),
            n_init=3,
            batch_size=256,
        )
        labels[row_ok] = km.fit_predict(Xs).astype(np.int32)
    out["cluster_id"] = labels
    out.attrs["cluster_features"] = cols
    print(f"clustered {int((labels >= 0).sum())} objects into {int(labels.max()) + 1} groups  ({len(cols)} features)")
    return out


def load_object_labels(image_path: PathLike) -> Optional[Dict[str, object]]:
    """Reload ``objects/{stem}_object_labels.json``, or None."""
    p = stem_paths(image_path).object_labels
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def apply_object_labels(table: pd.DataFrame, blob: Dict[str, object]) -> pd.DataFrame:
    """Add ``class_id`` / ``class_name`` from a montage JSON blob."""
    out = table.copy()
    names = {int(k): str(v) for k, v in (blob.get("class_names") or {}).items()}
    labels = {int(k): int(v) for k, v in (blob.get("labels") or {}).items()}
    out["class_id"] = out["object_id"].map(lambda i: labels.get(int(i), 0)).astype(int)
    out["class_name"] = out["class_id"].map(lambda i: names.get(int(i), ""))
    return out


def launch_montage_labeler(
    image_path: PathLike,
    table: pd.DataFrame,
    objects_mask: np.ndarray,
    class_names: Dict[int, str],
    *,
    wait: bool = True,
    grain_size: int = 96,
    grid_cols: int = 16,
) -> Dict[str, object]:
    """
    Open SEG ClusterMontageLabeler (subprocess). Close the window to save.

    Needs a ``cluster_id`` column (run ``cluster_objects`` first). Crops come from
    ``objects_mask`` on the RGB photo ``image_path``.
    """
    image_path = Path(image_path)
    if "cluster_id" not in table.columns:
        raise ValueError("table has no cluster_id — run cluster_objects first")
    names = [class_names[k] for k in sorted(class_names)]
    use = table["cluster_id"].to_numpy(dtype=np.int32) >= 0
    ids = table.loc[use, "object_id"].to_numpy(dtype=np.int32)
    clusters = table.loc[use, "cluster_id"].to_numpy(dtype=np.int32)
    out_json = stem_paths(image_path).object_labels
    worker = Path(__file__).resolve().parent / "montage_worker.py"
    env = os.environ.copy()
    env["MPLBACKEND"] = "QtAgg"
    env.setdefault("PYTHONPATH", str(Path(__file__).resolve().parent.parent.parent))
    with tempfile.TemporaryDirectory(prefix="pypetrograph_montage_") as tmp:
        tmp = Path(tmp)
        ids_p = tmp / "ids.npy"
        cl_p = tmp / "clusters.npy"
        mask_p = tmp / "mask.png"
        np.save(ids_p, ids)
        np.save(cl_p, clusters)
        from PIL import Image

        arr = np.asarray(objects_mask)
        if int(arr.max()) <= 255:
            Image.fromarray(arr.astype(np.uint8), mode="L").save(mask_p)
        else:
            Image.fromarray(arr.astype(np.uint16), mode="I;16").save(mask_p)
        cmd = [
            sys.executable,
            str(worker),
            "--image",
            str(image_path),
            "--mask",
            str(mask_p),
            "--object-ids",
            str(ids_p),
            "--clusters",
            str(cl_p),
            "--names",
            json.dumps(names),
            "--class-ids",
            json.dumps([int(k) for k in sorted(class_names)]),
            "--out",
            str(out_json),
            "--grain-size",
            str(int(grain_size)),
            "--grid-cols",
            str(int(grid_cols)),
        ]
        print(f"montage labeler: {image_path.name}  n={len(ids)}")
        if not wait:
            subprocess.Popen(cmd, env=env)
            return {"ok": True, "path": out_json, "hint": "Close the montage window to save."}
        proc = subprocess.run(cmd, env=env)
    ok = proc.returncode == 0 and out_json.exists()
    n = 0
    if out_json.exists():
        blob = json.loads(out_json.read_text(encoding="utf-8"))
        n = len(blob.get("labels") or {})
    return {
        "ok": ok,
        "n": n,
        "path": out_json,
        "error": None if ok else f"montage exited {proc.returncode}",
    }

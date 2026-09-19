"""
Stage 5: name objects with LightGBM and write a class overlay.

Training reuses ``labeling_ml.train.train_classifier_from_xy``. The model is
``models/object_model_{channel_set}.joblib``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from pyPetrograph.common.constants import (
    DEFAULT_METHOD,
    DEFAULT_N_JOBS,
    PathLike,
)
from pyPetrograph.common.paths import project_models_dir, stem_paths
from pyPetrograph.image_processing.viz import class_color_rgba
from pyPetrograph.labeling_ml.model_io import save_model_bundle
from pyPetrograph.labeling_ml.train import train_classifier_from_xy
from pyPetrograph.objects.cluster import numeric_feature_matrix


def object_model_path(channel_set_id: str) -> Path:
    """Notebook-level ``models/object_model_{channel_set_id}.joblib``."""
    return project_models_dir() / f"object_model_{channel_set_id}.joblib"


def train_object_classifier(
    table: pd.DataFrame,
    class_names: Dict[int, str],
    *,
    method: str = DEFAULT_METHOD,
    n_jobs: int = DEFAULT_N_JOBS,
    feature_cols: Optional[Sequence[str]] = None,
    channel_set_id: str = "default",
) -> Dict[str, Any]:
    """
    Train on rows with ``class_id > 0`` (from ``apply_object_labels``).

    Saves ``models/object_model_{channel_set_id}.joblib``. Returns ``ok`` False
    if fewer than 2 classes or 4 labeled rows.
    """
    if "class_id" not in table.columns:
        raise ValueError("table has no class_id — run apply_object_labels first")
    X, cols, row_ok = numeric_feature_matrix(table, feature_cols=feature_cols)
    y = table["class_id"].to_numpy()
    keep = row_ok & (np.asarray(y) > 0)
    n = int(keep.sum())
    n_cls = int(len(np.unique(y[keep]))) if n else 0
    if n_cls < 2 or n < 4:
        return {
            "ok": False,
            "error": "need at least 4 labeled objects and 2 classes to train",
        }
    bundle = train_classifier_from_xy(
        X[keep],
        y[keep],
        method=method,
        n_jobs=n_jobs,
        feature_names=cols,
        class_names=class_names,
    )
    bundle["channel_set_id"] = channel_set_id
    bundle["feature_cols"] = cols
    bundle["class_names"] = dict(class_names)
    path = object_model_path(channel_set_id)
    save_model_bundle(path, bundle)
    return {
        "ok": True,
        "bundle": bundle,
        "path": path,
        "n": n,
        "cv_folds": bundle.get("cv_folds"),
        "accuracy": bundle.get("accuracy"),
        "report": bundle.get("report"),
    }


def _class_names_from_bundle_or_table(
    table: pd.DataFrame,
    bundle: Dict[str, Any],
) -> Dict[int, str]:
    raw = bundle.get("class_names")
    if raw:
        return {int(k): str(v) for k, v in raw.items()}
    names: Dict[int, str] = {}
    if "class_id" in table.columns and "class_name" in table.columns:
        for cid, cname in zip(table["class_id"], table["class_name"]):
            cid = int(cid)
            if cid > 0 and str(cname):
                names[cid] = str(cname)
    return names


def predict_object_classes(table: pd.DataFrame, bundle: Dict[str, Any]) -> pd.DataFrame:
    """Predict ``class_id`` / ``proba`` / ``class_name`` for every finite row."""
    out = table.copy()
    n = len(out)
    class_id = np.zeros(n, dtype=np.int32)
    proba = np.zeros(n, dtype=np.float64)
    feat_cols = bundle.get("feature_cols") or bundle.get("feature_names")
    X, cols, row_ok = numeric_feature_matrix(out, feature_cols=feat_cols)
    if int(row_ok.sum()) > 0:
        clf = bundle["model"]
        X_df = pd.DataFrame(X[row_ok], columns=cols)
        pred_m = np.asarray(clf.predict(X_df))
        inv_map = bundle.get("inv_map")
        if inv_map is None and bundle.get("label_map"):
            inv_map = {int(v): int(k) for k, v in bundle["label_map"].items()}
        if inv_map is not None:
            pred = np.array([int(inv_map[int(v)]) for v in pred_m], dtype=np.int32)
        else:
            pred = np.asarray(pred_m, dtype=np.int32)
        class_id[row_ok] = pred
        if hasattr(clf, "predict_proba"):
            p = np.asarray(clf.predict_proba(X_df), dtype=np.float64)
            proba[row_ok] = p.max(axis=1)
    out["class_id"] = class_id
    out["proba"] = proba
    names = _class_names_from_bundle_or_table(table, bundle)
    out["class_name"] = [
        names.get(int(cid), "") if int(cid) > 0 else "" for cid in class_id
    ]
    return out


def save_classes(
    table: pd.DataFrame,
    objects_mask: np.ndarray,
    image_path: PathLike,
    class_names: Dict[int, str],
) -> Tuple[Path, Path]:
    """Write ``objects/{stem}_classes.csv`` and ``objects/{stem}_classes_rgb.png``."""
    from PIL import Image

    sp = stem_paths(image_path)
    sp.objects_dir.mkdir(parents=True, exist_ok=True)
    names_map = {int(k): str(v) for k, v in (class_names or {}).items()}
    class_ids = (
        table["class_id"].to_numpy()
        if "class_id" in table.columns
        else np.zeros(len(table), dtype=np.int32)
    )
    name_col = []
    for i, cid in enumerate(class_ids):
        cid = int(cid)
        if cid <= 0:
            name_col.append("")
            continue
        if cid in names_map:
            name_col.append(names_map[cid])
        elif "class_name" in table.columns:
            name_col.append(str(table["class_name"].iloc[i] or ""))
        else:
            name_col.append("")
    csv_df = pd.DataFrame(
        {
            "object_id": table["object_id"],
            "class_id": class_ids,
            "name": name_col,
            "proba": table["proba"] if "proba" in table.columns else 0.0,
        }
    )
    if "kind" in table.columns:
        csv_df["kind"] = table["kind"].to_numpy()
    if "area_px" in table.columns:
        csv_df["area_px"] = table["area_px"].to_numpy()
    csv_p = sp.classes_csv
    csv_df.to_csv(csv_p, index=False)

    arr = np.asarray(objects_mask)
    if arr.ndim > 2:
        arr = arr[..., 0]
    h, w = arr.shape[:2]
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    max_id = int(arr.max()) if arr.size else 0
    if max_id > 0:
        lut = np.zeros((max_id + 1, 3), dtype=np.uint8)
        oids = table["object_id"].to_numpy()
        for oid, cid in zip(oids, class_ids):
            oid, cid = int(oid), int(cid)
            if oid <= 0 or cid <= 0 or oid > max_id:
                continue
            r, g, b, _ = class_color_rgba(cid)
            lut[oid] = (
                int(round(r * 255)),
                int(round(g * 255)),
                int(round(b * 255)),
            )
        rgb = lut[arr.astype(np.int64, copy=False)]
    png_p = sp.classes_rgb
    Image.fromarray(rgb, mode="RGB").save(png_p)
    return csv_p, png_p


def load_classes(image_path: PathLike) -> Optional[pd.DataFrame]:
    """Reload ``objects/{stem}_classes.csv``, or None."""
    p = stem_paths(image_path).classes_csv
    if not p.exists():
        return None
    return pd.read_csv(p)

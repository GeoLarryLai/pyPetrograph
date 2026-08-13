"""Classifier construction, CV scoring, and training APIs."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from pyPetrograph.common.constants import (
    DEFAULT_CV_FOLDS,
    DEFAULT_LBP_P,
    DEFAULT_LBP_R,
    DEFAULT_METHOD,
    DEFAULT_N_JOBS,
    DEFAULT_RF_TREES,
    DEFAULT_TEXTURE_WINDOW,
    DEFAULT_Y_TARGET,
    METHODS,
    RANDOM_STATE,
)
from pyPetrograph.common.image_io import _to_rgb_uint8
from pyPetrograph.image_processing.brightness import normalize_brightness
from pyPetrograph.image_processing.features import (
    build_feature_stack,
    extract_training_table,
    resolve_feature_toggles,
)
from pyPetrograph.labeling_ml.polygons import rasterize_polygons

def _make_classifier(method: str, n_jobs: int = DEFAULT_N_JOBS):
    method = method.lower().strip()
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}. Choose from {METHODS}.")

    if method == "random_forest":
        return RandomForestClassifier(
            n_estimators=DEFAULT_RF_TREES,
            random_state=RANDOM_STATE,
            n_jobs=n_jobs,
        )

    if method == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:
            raise ImportError(
                "lightgbm is not installed. `conda install lightgbm` or `pip install lightgbm`."
            ) from exc
        return LGBMClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            n_jobs=n_jobs,
            verbose=-1,
        )

    if method == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError(
                "xgboost is not installed. `conda install xgboost` or `pip install xgboost`."
            ) from exc
        return XGBClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            n_jobs=n_jobs,
            verbosity=0,
            objective="multi:softprob",
            eval_metric="mlogloss",
        )

    raise ValueError(f"Unknown method {method!r}. Choose from {METHODS}.")


def _xgboost_label_maps(y: np.ndarray) -> Tuple[Optional[Dict[int, int]], Optional[Dict[int, int]], np.ndarray]:
    """Map class ids to 0..C-1 for XGBoost when needed. Returns (label_map, inv_map, y_fit)."""
    y = np.asarray(y, dtype=np.int32)
    classes = np.unique(y)
    if np.array_equal(classes, np.arange(len(classes))):
        return None, None, y
    label_map = {int(c): i for i, c in enumerate(classes)}
    inv_map = {i: int(c) for c, i in label_map.items()}
    y_fit = np.array([label_map[int(v)] for v in y], dtype=np.int32)
    return label_map, inv_map, y_fit


def _class_target_names(
    y: np.ndarray,
    class_names: Optional[Dict[int, str]] = None,
) -> Tuple[List[int], List[str]]:
    """Sorted class ids and display names like ``1 (pores)`` for classification_report."""
    labels = [int(c) for c in np.unique(np.asarray(y, dtype=np.int32))]
    names: List[str] = []
    for cid in labels:
        label = (class_names or {}).get(cid, f"class_{cid}")
        names.append(f"{cid} ({label})")
    return labels, names


def _format_accuracy_pct(acc: Optional[float]) -> str:
    """Accuracy as percent with 3 decimals, e.g. ``99.736%``."""
    if acc is None:
        return "n/a"
    return f"{100.0 * float(acc):.3f}%"


def _format_metric_pct(x: float) -> str:
    """Single metric 0–1 → percent string with 3 decimals."""
    return f"{100.0 * float(x):.3f}%"


def format_classification_report_pct(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[Dict[int, str]] = None,
) -> str:
    """
    Classification report with precision / recall / f1 / accuracy as percent (3 dp).

    Rows are ``id (name)``. Support stays a raw count.
    """
    labels, target_names = _class_target_names(y_true, class_names)
    d = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    # Column widths
    name_w = max(len("class"), max(len(n) for n in target_names), len("weighted avg"))
    lines = [
        f"{'class':<{name_w}}  {'precision':>10}  {'recall':>10}  {'f1-score':>10}  {'support':>10}",
        "",
    ]
    for name in target_names:
        row = d[name]
        lines.append(
            f"{name:<{name_w}}  {_format_metric_pct(row['precision']):>10}  "
            f"{_format_metric_pct(row['recall']):>10}  "
            f"{_format_metric_pct(row['f1-score']):>10}  {int(row['support']):>10}"
        )
    lines.append("")
    acc = d.get("accuracy", accuracy_score(y_true, y_pred))
    n = int(np.asarray(y_true).shape[0])
    lines.append(
        f"{'accuracy':<{name_w}}  {'':>10}  {'':>10}  {_format_accuracy_pct(acc):>10}  {n:>10}"
    )
    for avg_key, avg_label in (
        ("macro avg", "macro avg"),
        ("weighted avg", "weighted avg"),
    ):
        row = d[avg_key]
        lines.append(
            f"{avg_label:<{name_w}}  {_format_metric_pct(row['precision']):>10}  "
            f"{_format_metric_pct(row['recall']):>10}  "
            f"{_format_metric_pct(row['f1-score']):>10}  {int(row['support']):>10}"
        )
    return "\n".join(lines) + "\n"


def _cv_score_then_fit_all(
    X_df,
    y: np.ndarray,
    method: str,
    n_jobs: int = DEFAULT_N_JOBS,
    cv_folds: int = DEFAULT_CV_FOLDS,
    class_names: Optional[Dict[int, str]] = None,
) -> Dict[str, Any]:
    """
    Stratified k-fold OOF score (folds parallel via n_jobs), then fit final model on all rows.

    Classifier uses n_jobs=1 during CV to avoid nested oversubscription; final fit uses n_jobs.
    """
    y = np.asarray(y, dtype=np.int32)
    method_l = method.lower().strip()
    label_map = None
    inv_map = None
    y_fit = y
    if method_l == "xgboost":
        label_map, inv_map, y_fit = _xgboost_label_maps(y)

    _, counts = np.unique(y_fit, return_counts=True)
    n_splits = int(min(max(int(cv_folds), 2), int(counts.min())))
    if n_splits < 2:
        clf = _make_classifier(method_l, n_jobs=n_jobs)
        clf.fit(X_df, y_fit)
        y_hat_m = clf.predict(X_df)
        if inv_map is not None:
            y_hat = np.array([inv_map[int(v)] for v in y_hat_m], dtype=np.int32)
        else:
            y_hat = y_hat_m
        acc = float(accuracy_score(y, y_hat))
        report = format_classification_report_pct(y, y_hat, class_names=class_names)
        return {
            "model": clf,
            "accuracy": acc,
            "report": report,
            "label_map": label_map,
            "inv_map": inv_map,
            "cv_folds": 0,
        }

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    clf_cv = _make_classifier(method_l, n_jobs=1)
    y_oof_m = cross_val_predict(clf_cv, X_df, y_fit, cv=cv, n_jobs=n_jobs)
    if inv_map is not None:
        y_oof = np.array([inv_map[int(v)] for v in y_oof_m], dtype=np.int32)
    else:
        y_oof = y_oof_m
    acc = float(accuracy_score(y, y_oof))
    report = format_classification_report_pct(y, y_oof, class_names=class_names)

    clf = _make_classifier(method_l, n_jobs=n_jobs)
    clf.fit(X_df, y_fit)
    return {
        "model": clf,
        "accuracy": acc,
        "report": report,
        "label_map": label_map,
        "inv_map": inv_map,
        "cv_folds": n_splits,
    }


def train_classifier(
    image_rgb: np.ndarray,
    labels: np.ndarray,
    method: str = DEFAULT_METHOD,
    y_target: float = DEFAULT_Y_TARGET,
    n_jobs: int = DEFAULT_N_JOBS,
    cv_folds: int = DEFAULT_CV_FOLDS,
    apply_norm: bool = True,
    toggles: Optional[Dict[str, bool]] = None,
    texture_window: int = DEFAULT_TEXTURE_WINDOW,
    lbp_p: int = DEFAULT_LBP_P,
    lbp_r: float = DEFAULT_LBP_R,
    polygons: Optional[Sequence[Polygon]] = None,
    class_names: Optional[Dict[int, str]] = None,
) -> Dict[str, Any]:
    """
    Train RF / LightGBM / XGBoost on labeled pixels with selected features.

    Scores with stratified k-fold CV (out-of-fold), then fits the saved model on all labels.
    Returns bundle including feature_names and feature_toggles.
    """
    import pandas as pd

    X, y, names = extract_training_table(
        image_rgb,
        labels,
        y_target=y_target,
        apply_norm=apply_norm,
        toggles=toggles,
        texture_window=texture_window,
        lbp_p=lbp_p,
        lbp_r=lbp_r,
        polygons=polygons,
    )
    X_df = pd.DataFrame(X, columns=names)
    fitted = _cv_score_then_fit_all(
        X_df,
        y,
        method=method,
        n_jobs=n_jobs,
        cv_folds=cv_folds,
        class_names=class_names,
    )

    img_u8 = _to_rgb_uint8(image_rgb)
    flat_u8 = img_u8.reshape(-1, 3)
    lab = np.asarray(labels).reshape(-1)
    palette: Dict[int, List[int]] = {0: [255, 255, 255]}
    for cid in np.unique(lab[lab > 0]):
        pix = flat_u8[lab == cid]
        palette[int(cid)] = [int(round(x)) for x in pix.mean(axis=0)]

    return {
        "model": fitted["model"],
        "method": method.lower(),
        "accuracy": fitted["accuracy"],
        "report": fitted["report"],
        "cv_folds": fitted["cv_folds"],
        "color_palette": palette,
        "y_target": float(y_target),
        "apply_norm": bool(apply_norm),
        "label_map": fitted["label_map"],
        "inv_map": fitted["inv_map"],
        "n_jobs": int(n_jobs),
        "feature_names": list(names),
        "feature_toggles": resolve_feature_toggles(toggles),
        "texture_window": int(texture_window),
        "lbp_p": int(lbp_p),
        "lbp_r": float(lbp_r),
    }


def train_classifier_from_xy(
    X: np.ndarray,
    y: np.ndarray,
    method: str = DEFAULT_METHOD,
    y_target: float = DEFAULT_Y_TARGET,
    n_jobs: int = DEFAULT_N_JOBS,
    cv_folds: int = DEFAULT_CV_FOLDS,
    apply_norm: bool = True,
    color_palette: Optional[Dict[int, List[int]]] = None,
    feature_names: Optional[Sequence[str]] = None,
    toggles: Optional[Dict[str, bool]] = None,
    texture_window: int = DEFAULT_TEXTURE_WINDOW,
    lbp_p: int = DEFAULT_LBP_P,
    lbp_r: float = DEFAULT_LBP_R,
    class_names: Optional[Dict[int, str]] = None,
) -> Dict[str, Any]:
    """Train from pre-built float32 X (N,C) and int y (N,). CV then fit-all (see train_classifier)."""
    import pandas as pd

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int32)
    if X.ndim != 2 or X.shape[0] == 0 or y.shape[0] != X.shape[0]:
        raise ValueError("X and y must be non-empty and have the same length.")
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(X.shape[1])]
    feature_names = list(feature_names)
    if len(feature_names) != X.shape[1]:
        raise ValueError("feature_names length must match X columns.")

    X_df = pd.DataFrame(X, columns=feature_names)
    fitted = _cv_score_then_fit_all(
        X_df,
        y,
        method=method,
        n_jobs=n_jobs,
        cv_folds=cv_folds,
        class_names=class_names,
    )
    palette = color_palette or {0: [255, 255, 255]}
    for cid in np.unique(y):
        cid = int(cid)
        if cid not in palette:
            palette[cid] = [128, 128, 128]

    return {
        "model": fitted["model"],
        "method": method.lower(),
        "accuracy": fitted["accuracy"],
        "report": fitted["report"],
        "cv_folds": fitted["cv_folds"],
        "color_palette": palette,
        "y_target": float(y_target),
        "apply_norm": bool(apply_norm),
        "label_map": fitted["label_map"],
        "inv_map": fitted["inv_map"],
        "n_jobs": int(n_jobs),
        "feature_names": feature_names,
        "feature_toggles": resolve_feature_toggles(toggles),
        "texture_window": int(texture_window),
        "lbp_p": int(lbp_p),
        "lbp_r": float(lbp_r),
    }

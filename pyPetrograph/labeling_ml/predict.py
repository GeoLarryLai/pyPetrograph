"""Image prediction, smoothing, and confidence maps."""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.ndimage import median_filter

from pyPetrograph.common.constants import (
    DEFAULT_CHUNK_PIXELS,
    DEFAULT_LBP_P,
    DEFAULT_LBP_R,
    DEFAULT_MEDIAN_SIZE,
    DEFAULT_TEXTURE_WINDOW,
    DEFAULT_Y_TARGET,
)
from pyPetrograph.image_processing.brightness import normalize_brightness
from pyPetrograph.image_processing.features import build_feature_stack, resolve_feature_toggles

def predict_image(
    image_rgb: np.ndarray,
    model_bundle: Dict[str, Any],
    chunk_pixels: int = DEFAULT_CHUNK_PIXELS,
    polygons: Optional[Sequence[Polygon]] = None,
) -> np.ndarray:
    """
    Predict labels for every pixel using the bundle's feature stack.

    Returns uint8 label map (H, W).
    """
    return predict_image_proba(
        image_rgb,
        model_bundle,
        chunk_pixels=chunk_pixels,
        polygons=polygons,
    )["labels"]


def predict_image_proba(
    image_rgb: np.ndarray,
    model_bundle: Dict[str, Any],
    chunk_pixels: int = DEFAULT_CHUNK_PIXELS,
    polygons: Optional[Sequence[Polygon]] = None,
) -> Dict[str, Any]:
    """
    Predict labels + class probabilities for every pixel.

    Returns
    -------
    dict with:
      labels : uint8 (H, W) — argmax class (0 on near-white margins)
      proba : float32 (H, W, C) — P(class) per model class column
      class_ids : list[int] length C — class id for each proba channel
      confidence : float32 (H, W) — max class probability (0 on margins)
    """
    import pandas as pd

    clf = model_bundle["model"]
    y_target = model_bundle.get("y_target", DEFAULT_Y_TARGET)
    apply_norm = model_bundle.get("apply_norm", True)
    inv_map = model_bundle.get("inv_map")
    toggles = model_bundle.get("feature_toggles")
    names = model_bundle.get("feature_names")
    texture_window = int(model_bundle.get("texture_window", DEFAULT_TEXTURE_WINDOW))
    lbp_p = int(model_bundle.get("lbp_p", DEFAULT_LBP_P))
    lbp_r = float(model_bundle.get("lbp_r", DEFAULT_LBP_R))

    stack, built_names = build_feature_stack(
        image_rgb,
        toggles=toggles,
        y_target=y_target,
        apply_norm=apply_norm,
        texture_window=texture_window,
        lbp_p=lbp_p,
        lbp_r=lbp_r,
        polygons=polygons,
    )
    if names is not None and list(names) != list(built_names):
        if set(names) != set(built_names):
            raise RuntimeError(
                "Model feature names do not match current toggles.\n"
                f"Model expects: {names}\n"
                f"Built: {built_names}\n"
                "Retrain, or load a matching model."
            )
        idx = [built_names.index(n) for n in names]
        stack = stack[..., idx]
        col_names = list(names)
    else:
        col_names = built_names

    h, w = stack.shape[:2]
    flat = stack.reshape(-1, stack.shape[-1]).astype(np.float32)
    n = flat.shape[0]

    if not hasattr(clf, "predict_proba"):
        raise RuntimeError(
            f"Model {type(clf).__name__} has no predict_proba; cannot estimate uncertainty."
        )
    probe = pd.DataFrame(flat[:1], columns=col_names)
    n_classes = int(clf.predict_proba(probe).shape[1])
    model_classes = [int(c) for c in np.asarray(clf.classes_).tolist()]
    if inv_map is not None:
        class_ids = [int(inv_map[int(c)]) for c in model_classes]
    else:
        class_ids = list(model_classes)

    proba_flat = np.empty((n, n_classes), dtype=np.float32)
    for start_i in range(0, n, chunk_pixels):
        end_i = min(start_i + chunk_pixels, n)
        chunk = pd.DataFrame(flat[start_i:end_i], columns=col_names)
        proba_flat[start_i:end_i] = np.asarray(clf.predict_proba(chunk), dtype=np.float32)

    pred_cols = np.argmax(proba_flat, axis=1)
    labels_flat = np.array([class_ids[int(i)] for i in pred_cols], dtype=np.int32)
    conf_flat = proba_flat.max(axis=1)

    labels = labels_flat.reshape(h, w).astype(np.uint8)
    confidence = conf_flat.reshape(h, w).astype(np.float32)
    proba = proba_flat.reshape(h, w, n_classes).astype(np.float32)

    mean_im = np.asarray(image_rgb, dtype=np.float32).mean(axis=-1)
    margin = mean_im >= 254.5
    labels[margin] = 0
    confidence[margin] = 0.0
    proba[margin] = 0.0

    return {
        "labels": labels,
        "proba": proba,
        "class_ids": class_ids,
        "confidence": confidence,
    }


def smooth_prediction(pred: np.ndarray, size: int = DEFAULT_MEDIAN_SIZE) -> np.ndarray:
    """Median-filter a label map."""
    if size <= 1:
        return np.asarray(pred, dtype=np.uint8)
    return median_filter(np.asarray(pred, dtype=np.uint8), size=int(size))


def confidence_for_labels(
    pred: np.ndarray,
    proba: np.ndarray,
    class_ids: Sequence[int],
) -> np.ndarray:
    """
    Per-pixel probability of the assigned label (0 where pred==0).

    Parameters: pred (H,W), proba (H,W,C), class_ids length C.
    """
    pred = np.asarray(pred, dtype=np.int32)
    proba = np.asarray(proba, dtype=np.float32)
    out = np.zeros(pred.shape, dtype=np.float32)
    id_to_col = {int(cid): i for i, cid in enumerate(class_ids)}
    for cid, col in id_to_col.items():
        m = pred == cid
        if np.any(m):
            out[m] = proba[..., col][m]
    return out

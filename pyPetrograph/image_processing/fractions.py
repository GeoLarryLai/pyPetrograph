"""Modal (area) fraction helpers."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from pyPetrograph.common.constants import DEFAULT_SURE_PROBA
from pyPetrograph.labeling_ml.predict import confidence_for_labels

def compute_fractions(
    pred: np.ndarray,
    class_names: Optional[Dict[int, str]] = None,
) -> Dict[int, Dict[str, Any]]:
    """
    Area fractions per class, excluding label 0 (background / white margins).

    Returns {class_id: {"name": str, "fraction": float, "count": int, ...bounds}}.
    """
    return compute_fractions_uncertain(
        pred, class_names=class_names, proba=None, class_ids=None
    )


def compute_fractions_uncertain(
    pred: np.ndarray,
    class_names: Optional[Dict[int, str]] = None,
    proba: Optional[np.ndarray] = None,
    class_ids: Optional[Sequence[int]] = None,
    sure_threshold: float = DEFAULT_SURE_PROBA,
    confidence: Optional[np.ndarray] = None,
) -> Dict[int, Dict[str, Any]]:
    """
    Hard area fractions plus asymmetric bounds from class probabilities.

    - area / fraction: hard labels (argmax, usually after median smooth)
    - area_pct_low: sure % — this class and P(class) >= sure_threshold
    - area_pct_high: soft expected % — mean P(class) over non-zero pixels

    Without ``proba``, low=high=hard (no uncertainty).
    """
    flat = np.asarray(pred, dtype=np.uint8).ravel()
    mask = flat != 0
    total = int(mask.sum())
    if total == 0:
        return {}

    if confidence is None and proba is not None and class_ids is not None:
        confidence = confidence_for_labels(pred, proba, class_ids)
    conf_flat = (
        np.asarray(confidence, dtype=np.float32).ravel()
        if confidence is not None
        else None
    )

    soft_by_id: Dict[int, float] = {}
    if proba is not None and class_ids is not None:
        proba_arr = np.asarray(proba, dtype=np.float32)
        flat_proba = proba_arr.reshape(-1, proba_arr.shape[-1])
        for j, cid in enumerate(class_ids):
            soft_by_id[int(cid)] = float(flat_proba[mask, j].mean()) if total else 0.0

    counts = np.bincount(flat[mask])
    out: Dict[int, Dict[str, Any]] = {}
    all_cids = set(int(c) for c in range(1, len(counts)) if counts[c] > 0)
    all_cids.update(soft_by_id.keys())

    for cid in sorted(all_cids):
        if cid <= 0:
            continue
        count = int(counts[cid]) if cid < len(counts) else 0
        hard_frac = float(count) / float(total) if total else 0.0
        name = (class_names or {}).get(cid, f"class_{cid}")

        if conf_flat is not None and count > 0:
            sure_count = int(
                np.sum((flat == cid) & (conf_flat >= float(sure_threshold)))
            )
            sure_frac = float(sure_count) / float(total)
        elif conf_flat is not None:
            sure_count = 0
            sure_frac = 0.0
        else:
            sure_count = count
            sure_frac = hard_frac

        soft_frac = soft_by_id.get(cid, hard_frac)
        area_pct = 100.0 * hard_frac
        area_pct_low = 100.0 * sure_frac
        # Upper bound from +- always ≥ main: max(hard, soft expected)
        area_pct_high = max(area_pct, 100.0 * soft_frac)
        plus = area_pct_high - area_pct
        minus = max(0.0, area_pct - area_pct_low)

        out[cid] = {
            "name": name,
            "fraction": hard_frac,
            "count": count,
            "sure_count": int(sure_count),
            "soft_fraction": soft_frac,
            "sure_fraction": sure_frac,
            "area_pct": area_pct,
            "area_pct_low": area_pct_low,
            "area_pct_high": area_pct_high,
            "area_pct_plus": plus,
            "area_pct_minus": minus,
            "sure_threshold": float(sure_threshold),
        }
    return out

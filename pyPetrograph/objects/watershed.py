"""
No-training grain outlines: edge map from the channel stack + watershed.

Edge strength at a pixel = weighted sum of Sobel magnitude on every channel
(each channel is z-scored first so a 0–255 map does not drown a −1..1 angle
map). Markers = local maxima of the inverted edge map (distance from edges),
then watershed. Leftover (pores, cement) is left as 0 — stage 4 will pick
those up as objects.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from pyPetrograph.common.constants import (
    DEFAULT_WATERSHED_MIN_AREA,
    DEFAULT_WATERSHED_MIN_DISTANCE,
)
from pyPetrograph.fuse.stack import ChannelStack


def gradient_magnitude(stack: np.ndarray, *, weights: Optional[Sequence[float]] = None) -> np.ndarray:
    """HxW edge strength: per-channel Sobel magnitude, z-scored, then weighted sum."""
    from scipy.ndimage import sobel

    arr = np.asarray(stack, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[..., None]
    h, w, c = arr.shape
    wts = np.ones(c, dtype=np.float32) if weights is None else np.asarray(weights, dtype=np.float32)
    if wts.size != c:
        raise ValueError(f"{wts.size} weights for {c} channels")
    acc = np.zeros((h, w), dtype=np.float32)
    for d in range(c):
        ch = arr[..., d]
        sd = float(ch.std())
        if sd < 1e-6:
            continue
        zn = (ch - float(ch.mean())) / sd
        mag = np.hypot(sobel(zn, axis=0), sobel(zn, axis=1))
        acc += float(wts[d]) * mag
    return acc


def watershed_labels(
    elev: np.ndarray,
    *,
    min_distance: int = DEFAULT_WATERSHED_MIN_DISTANCE,
    min_area: int = DEFAULT_WATERSHED_MIN_AREA,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Instance labels from an edge-elevation map.

    Parameters
    ----------
    elev : HxW edge strength (high = boundary)
    min_distance : minimum pixel gap between grain seeds
    min_area : drop regions smaller than this
    mask : optional HxW bool; False pixels are never labeled
    """
    from skimage.feature import peak_local_max
    from skimage.morphology import remove_small_objects
    from skimage.segmentation import watershed

    elev = np.asarray(elev, dtype=np.float32)
    inv = elev.max() - elev
    kw: Dict[str, object] = {"min_distance": int(min_distance), "exclude_border": False}
    if mask is not None and (not np.all(mask)):
        kw["labels"] = np.asarray(mask, dtype=np.int32)
    coords = peak_local_max(inv, **kw)
    markers = np.zeros(elev.shape, dtype=np.int32)
    for i, rc in enumerate(coords, start=1):
        markers[int(rc[0]), int(rc[1])] = i
    if markers.max() == 0:
        return markers
    lab = np.asarray(watershed(elev, markers, mask=mask), dtype=np.int32)
    if int(lab.max()) >= 2:
        lab = np.asarray(remove_small_objects(lab, min_size=int(min_area)), dtype=np.int32)
    elif int(lab.max()) == 1 and int((lab == 1).sum()) < int(min_area):
        lab[:] = 0
    return lab


def run_watershed(
    cs: ChannelStack,
    *,
    min_distance: int = DEFAULT_WATERSHED_MIN_DISTANCE,
    min_area: int = DEFAULT_WATERSHED_MIN_AREA,
    weights: Optional[Sequence[float]] = None,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """
    Watershed on the channel stack.

    Parameters
    ----------
    cs : stage-2 ChannelStack
    min_distance, min_area : seed gap and smallest grain
    weights : optional per-channel weights (default equal)

    Returns
    -------
    labels : HxW int32 instance ids (0 = leftover)
    info : method, n, min_distance, min_area
    """
    elev = gradient_magnitude(cs.stack, weights=weights)
    mask = cs.valid if cs.valid is not None else None
    lab = watershed_labels(elev, min_distance=min_distance, min_area=min_area, mask=mask)
    n = int(np.sum(np.unique(lab) > 0))
    return lab, {
        "method": "watershed",
        "n": n,
        "min_distance": int(min_distance),
        "min_area": int(min_area),
        "ok": n > 0,
    }


def instance_to_three_class(labels: np.ndarray) -> np.ndarray:
    """Instance mask → HxW uint8 {0 background, 1 grain, 2 outer boundary} (SEG layout)."""
    from skimage.segmentation import find_boundaries

    lab = np.asarray(labels)
    out = np.zeros(lab.shape[:2], dtype=np.uint8)
    out[lab > 0] = 1
    if int(lab.max()) > 0:
        out[find_boundaries(lab, mode="outer")] = 2
    return out


def three_class_to_probs(mask: np.ndarray) -> np.ndarray:
    """HxW {0,1,2} → HxWx3 float32 one-hot (bg, grain, boundary) for ``label_grains``."""
    y = np.asarray(mask)
    oh = np.zeros(y.shape + (3,), dtype=np.float32)
    for k in range(3):
        oh[..., k] = (y == k).astype(np.float32)
    return oh

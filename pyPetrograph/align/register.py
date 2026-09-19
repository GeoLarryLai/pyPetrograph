"""Auto lineup on structure maps (edges + texture). Always try auto first."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from skimage.feature import match_template
from skimage.registration import phase_cross_correlation
from skimage.transform import rescale

from pyPetrograph.align.scene import Affine2D, Scene
from pyPetrograph.align.structure import structure_map


def _pad_to_same(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    h = max(a.shape[0], b.shape[0])
    w = max(a.shape[1], b.shape[1])
    aa = np.zeros((h, w), dtype=np.float32)
    bb = np.zeros((h, w), dtype=np.float32)
    aa[: a.shape[0], : a.shape[1]] = a
    bb[: b.shape[0], : b.shape[1]] = b
    return aa, bb


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    mask = (a != 0) | (b != 0)
    if int(mask.sum()) < 64:
        return -1.0
    aa = a[mask].astype(np.float64)
    bb = b[mask].astype(np.float64)
    aa -= aa.mean()
    bb -= bb.mean()
    denom = float(np.sqrt(np.sum(aa * aa) * np.sum(bb * bb))) + 1e-8
    return float(np.sum(aa * bb) / denom)


def _shift_moving(mov: np.ndarray, dy: float, dx: float, shape: Tuple[int, int]) -> np.ndarray:
    from scipy.ndimage import shift

    out = shift(mov, shift=(dy, dx), order=1, mode="constant", cval=0.0)
    h, w = shape
    if out.shape[0] != h or out.shape[1] != w:
        canvas = np.zeros((h, w), dtype=np.float32)
        hh, ww = min(h, out.shape[0]), min(w, out.shape[1])
        canvas[:hh, :ww] = out[:hh, :ww]
        return canvas
    return out.astype(np.float32)


def _downsample_pair(
    base_s: np.ndarray, mov_s: np.ndarray, target_mp: float = 1.5
) -> Tuple[np.ndarray, np.ndarray, float]:
    bh, bw = base_s.shape[:2]
    cur = (bh * bw) / 1_000_000.0
    if cur <= target_mp:
        return base_s, mov_s, 1.0
    factor = (target_mp / cur) ** 0.5
    b_small = np.asarray(
        rescale(base_s, factor, order=1, anti_aliasing=True, preserve_range=True),
        dtype=np.float32,
    )
    scale = b_small.shape[0] / float(bh)
    m_small = np.asarray(
        rescale(mov_s, scale, order=1, anti_aliasing=True, preserve_range=True),
        dtype=np.float32,
    )
    if m_small.shape[0] < 8 or m_small.shape[1] < 8:
        return base_s, mov_s, 1.0
    return b_small, m_small, float(scale)


def auto_register(
    base_rgb: np.ndarray,
    moving_rgb: np.ndarray,
    *,
    scale_hint: float = 1.0,
    scale_range: Tuple[float, float] = (0.5, 2.0),
    n_scales: int = 9,
    tx_hint: float = 0.0,
    ty_hint: float = 0.0,
    neighborhood: Optional[Tuple[int, int, int, int]] = None,
) -> Affine2D:
    """
    Always-run auto lineup on structure maps.

    ``neighborhood`` is (r0, r1, c0, c1) in base pixels to search (re-auto in view).
    Hints (scale/tx/ty) set the start; a scale sweep is allowed inside ``scale_range``.
    """
    base_s = structure_map(base_rgb)
    mov_s = structure_map(moving_rgb)
    if neighborhood is not None:
        r0, r1, c0, c1 = neighborhood
        r0, r1 = max(0, int(r0)), min(base_s.shape[0], int(r1))
        c0, c1 = max(0, int(c0)), min(base_s.shape[1], int(c1))
        if r1 > r0 + 16 and c1 > c0 + 16:
            base_s = base_s[r0:r1, c0:c1]
            ty_hint = float(ty_hint) - r0
            tx_hint = float(tx_hint) - c0
        else:
            neighborhood = None

    lo, hi = float(scale_range[0]), float(scale_range[1])
    hint = float(np.clip(scale_hint, lo, hi))
    same_size = (
        abs(base_s.shape[0] - mov_s.shape[0]) <= 2
        and abs(base_s.shape[1] - mov_s.shape[1]) <= 2
    )
    if same_size and abs(hint - 1.0) < 0.05:
        scales = np.array([1.0], dtype=np.float64)
    else:
        scales = np.unique(
            np.clip(
                np.geomspace(max(lo, hint * 0.7), min(hi, hint * 1.4), num=max(3, n_scales)),
                lo,
                hi,
            )
        )
        if hint not in scales:
            scales = np.sort(np.append(scales, hint))
        if lo <= 1.0 <= hi and 1.0 not in scales:
            scales = np.sort(np.append(scales, 1.0))

    best: Optional[Tuple[float, Affine2D]] = None
    bh, bw = base_s.shape[:2]

    for sc in scales:
        mov_sc = np.asarray(
            rescale(mov_s, float(sc), order=1, anti_aliasing=True, preserve_range=True),
            dtype=np.float32,
        )
        if mov_sc.shape[0] < 8 or mov_sc.shape[1] < 8:
            continue
        nested = mov_sc.shape[0] <= 0.95 * bh and mov_sc.shape[1] <= 0.95 * bw
        b_small, m_small, ds = _downsample_pair(base_s, mov_sc, target_mp=2.0)
        if nested and m_small.shape[0] < b_small.shape[0] and m_small.shape[1] < b_small.shape[1]:
            corr = match_template(b_small, m_small, pad_input=False)
            peak = np.unravel_index(int(np.argmax(corr)), corr.shape)
            dy = peak[0] / ds
            dx = peak[1] / ds
        else:
            aa, bb = _pad_to_same(b_small, m_small)
            shift, _, _ = phase_cross_correlation(aa, bb, upsample_factor=10)
            dy, dx = float(shift[0]) / ds, float(shift[1]) / ds

        placed = _shift_moving(mov_sc, dy, dx, (bh, bw))
        ncc = _ncc(base_s, placed)
        tform = Affine2D(scale=float(sc), rotation_deg=0.0, tx=float(dx), ty=float(dy))
        if neighborhood is not None:
            r0, r1, c0, c1 = neighborhood
            tform.tx += c0
            tform.ty += r0
        if best is None or ncc > best[0]:
            best = (ncc, tform)

    if best is None:
        return Affine2D(scale=float(hint), tx=float(tx_hint), ty=float(ty_hint))
    return best[1]


def refine_register(
    scene: Scene,
    slot: str,
    *,
    neighborhood: Optional[Tuple[int, int, int, int]] = None,
) -> Affine2D:
    """Re-run auto around the current transform of ``slot``."""
    base = scene.working().ensure_rgb()
    moving = scene.layers[slot].ensure_rgb()
    cur = scene.layers[slot].transform
    tform = auto_register(
        base,
        moving,
        scale_hint=cur.scale,
        scale_range=scene.scale_range,
        tx_hint=cur.tx,
        ty_hint=cur.ty,
        neighborhood=neighborhood,
    )
    scene.layers[slot].transform = tform
    return tform


def fit_click_pairs(
    pairs: Sequence[Dict[str, List[float]]],
    *,
    tolerance_px: float = 8.0,
) -> Optional[Affine2D]:
    """
    Click pairs are loose hints: each dict has ``base`` [x, y] and ``moving`` [x, y].

    ``tolerance_px`` is stored for the UI; with 1 pair we take translation only,
    with 2+ we fit scale + translation (no fake points).
    """
    if not pairs:
        return None
    b = np.array([p["base"] for p in pairs], dtype=np.float64)
    m = np.array([p["moving"] for p in pairs], dtype=np.float64)
    if len(pairs) == 1:
        return Affine2D(tx=float(b[0, 0] - m[0, 0]), ty=float(b[0, 1] - m[0, 1]))
    # Similarity: base ≈ scale * R * moving + t. Ignore rotation if tiny sample.
    # Use translation + scale from centroid distances.
    bc, mc = b.mean(axis=0), m.mean(axis=0)
    b0, m0 = b - bc, m - mc
    denom = float(np.sum(m0 * m0))
    scale = 1.0 if denom < 1e-6 else float(np.sum(b0 * m0) / denom)
    scale = float(np.clip(scale, 0.05, 20.0))
    tx = float(bc[0] - scale * mc[0])
    ty = float(bc[1] - scale * mc[1])
    _ = tolerance_px  # UI uses this when scoring; pairs are never required to be exact
    return Affine2D(scale=scale, tx=tx, ty=ty)

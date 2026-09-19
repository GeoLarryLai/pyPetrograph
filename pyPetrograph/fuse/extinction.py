"""
Extinction fit across cross-polarized (XPL) angles.

Under crossed polars a grain's brightness repeats every 90° of stage rotation,
so per pixel we fit  I(θ) = a0 + a1·cos(4θ) + a2·sin(4θ)  by linear least squares
(three unknowns → at least three distinct angles). From the fit:

- ``max``  = a0 + amp   brightest the pixel ever gets (extinction removed; the
  classic "max image")
- ``min``  = a0 − amp   darkest (extinction)
- ``amp``  = sqrt(a1² + a2²)   how strongly the pixel blinks; ~0 for isotropic
  or opaque grains and for pores
- ``phi_cos``, ``phi_sin`` = (a1, a2) / amp   the extinction angle stored as a
  point on a circle so 0° and 90° are neighbours (no wrap seam for a CNN)
- ``phi_deg``  angle of maximum brightness in [0, 90)  (for tables / display)
- ``resid``  RMS misfit; large where the sine model is wrong (twins, zoning,
  saturated pixels)

Inside one grain these maps are flat; at a grain boundary they jump. That is why
they make better CNN input than the raw angle photos.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

MIN_ANGLES = 3
PERIOD_DEG = 90.0

FIT_NAMES = ("max", "amp", "phi_cos", "phi_sin")  # channels used in the stack
ALL_NAMES = ("mean", "amp", "max", "min", "phi_cos", "phi_sin", "phi_deg", "resid")


def design_matrix(angles_deg: Sequence[float]) -> np.ndarray:
    """Rows [1, cos4θ, sin4θ] for each angle (float64, n×3)."""
    th = np.deg2rad(np.asarray(angles_deg, dtype=np.float64))
    return np.column_stack([np.ones_like(th), np.cos(4.0 * th), np.sin(4.0 * th)])


def check_angles(angles_deg: Sequence[float]) -> None:
    """Raise a plain-language ValueError if the angles cannot support the 3-term fit."""
    ang = np.asarray(angles_deg, dtype=np.float64)
    if ang.ndim != 1 or ang.size < MIN_ANGLES:
        raise ValueError(
            f"need at least {MIN_ANGLES} XPL angles for the extinction fit, got {ang.size}"
        )
    folded = np.unique(np.round(np.mod(ang, PERIOD_DEG), 3))
    if folded.size < MIN_ANGLES:
        raise ValueError(
            "XPL angles repeat every 90°; need at least 3 distinct angles inside 0–90°, "
            f"got {folded.tolist()}"
        )
    if np.linalg.matrix_rank(design_matrix(ang)) < 3:
        raise ValueError(f"XPL angles {ang.tolist()} are degenerate for the sine fit")


def fit_extinction(
    gray_stack: np.ndarray,
    angles_deg: Sequence[float],
    *,
    valid: Optional[np.ndarray] = None,
    chunk_rows: int = 256,
    amp_eps: float = 1e-3,
) -> Dict[str, np.ndarray]:
    """
    Per-pixel sine fit over a stack of gray XPL photos.

    Parameters
    ----------
    gray_stack : (n, H, W) float or uint8 — one gray (luminance) image per angle
    angles_deg : n polarizer / stage angles in degrees
    valid : optional HxW bool; outputs are zero outside it
    chunk_rows : rows per least-squares chunk (memory control)
    amp_eps : below this amplitude the angle is undefined → phi_cos = phi_sin = 0

    Returns
    -------
    dict of float32 HxW maps with keys ``ALL_NAMES``
    """
    stack = np.asarray(gray_stack)
    if stack.ndim != 3:
        raise ValueError(f"gray_stack must be (n, H, W); got {stack.shape}")
    n, h, w = stack.shape
    ang = np.asarray(angles_deg, dtype=np.float64)
    if ang.size != n:
        raise ValueError(f"{n} images but {ang.size} angles")
    check_angles(ang)

    A = design_matrix(ang)  # n×3
    pinv = np.linalg.pinv(A)  # 3×n

    out = {k: np.zeros((h, w), dtype=np.float32) for k in ALL_NAMES}
    step = max(1, int(chunk_rows))
    for r0 in range(0, h, step):
        r1 = min(h, r0 + step)
        block = stack[:, r0:r1, :].reshape(n, -1).astype(np.float64)  # n × m
        coef = pinv @ block  # 3 × m
        a0, a1, a2 = coef[0], coef[1], coef[2]
        amp = np.hypot(a1, a2)
        resid = np.sqrt(np.mean((A @ coef - block) ** 2, axis=0))
        safe = amp > amp_eps
        pc = np.where(safe, a1 / np.where(safe, amp, 1.0), 0.0)
        ps = np.where(safe, a2 / np.where(safe, amp, 1.0), 0.0)
        # I(θ) = a0 + amp·cos(4θ − δ), δ = atan2(a2, a1) → brightest at θ = δ/4
        phi = np.mod(np.rad2deg(np.arctan2(a2, a1)) / 4.0, PERIOD_DEG)
        phi = np.where(safe, phi, 0.0)
        shape = (r1 - r0, w)
        out["mean"][r0:r1] = a0.reshape(shape)
        out["amp"][r0:r1] = amp.reshape(shape)
        out["max"][r0:r1] = (a0 + amp).reshape(shape)
        out["min"][r0:r1] = np.maximum(a0 - amp, 0.0).reshape(shape)
        out["phi_cos"][r0:r1] = pc.reshape(shape)
        out["phi_sin"][r0:r1] = ps.reshape(shape)
        out["phi_deg"][r0:r1] = phi.reshape(shape)
        out["resid"][r0:r1] = resid.reshape(shape)

    if valid is not None:
        bad = ~np.asarray(valid, dtype=bool)
        for k in out:
            out[k][bad] = 0.0
    return out


def phi_deg_from_cos_sin(phi_cos: np.ndarray, phi_sin: np.ndarray) -> np.ndarray:
    """Recover the max-brightness angle in [0, 90) from the two circle channels."""
    return np.mod(
        np.rad2deg(np.arctan2(np.asarray(phi_sin, dtype=np.float64), np.asarray(phi_cos, dtype=np.float64))) / 4.0,
        PERIOD_DEG,
    ).astype(np.float32)

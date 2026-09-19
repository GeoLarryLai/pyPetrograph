"""Feature toggles, texture maps, feature stack, and training tables."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from shapely.geometry import Polygon

from pyPetrograph.common.constants import (
    DEFAULT_FEATURE_TOGGLES,
    DEFAULT_GABOR_ANGLES_DEG,
    DEFAULT_GABOR_FREQUENCY,
    DEFAULT_HESSIAN_SIGMA,
    DEFAULT_LBP_P,
    DEFAULT_LBP_R,
    DEFAULT_PREVIEW_TARGET_MP,
    DEFAULT_SCREEN_VIEW_FILL,
    DEFAULT_TEXTURE_WINDOW,
    DEFAULT_Y_TARGET,
    FEATURE_ENTROPY_WINDOWS,
    FEATURE_LBP_RADII,
    FEATURE_STD_WINDOWS,
    GLCM_PROP_NAMES,
)
from pyPetrograph.common.image_io import _to_rgb_uint8
from pyPetrograph.image_processing.brightness import normalize_brightness, relative_luminance
from pyPetrograph.labeling_ml.polygons import rasterize_polygons

def default_feature_toggles() -> Dict[str, bool]:
    return dict(DEFAULT_FEATURE_TOGGLES)


def resolve_feature_toggles(toggles: Optional[Dict[str, bool]] = None) -> Dict[str, bool]:
    """Known features default to False unless explicitly set True."""
    out = {k: False for k in DEFAULT_FEATURE_TOGGLES}
    if toggles:
        for k, v in toggles.items():
            if k in out:
                out[k] = bool(v)
    if not any(out.values()):
        # Train/predict need at least one channel — fall back to RGB
        out["r"] = out["g"] = out["b"] = True
    return out


def downsample_to_approx_mp(image: np.ndarray, target_mp: float = DEFAULT_PREVIEW_TARGET_MP) -> np.ndarray:
    """Shrink image so H×W ≈ target_mp megapixels (approximate)."""
    img = np.asarray(image)
    h, w = img.shape[:2]
    cur_mp = (h * w) / 1_000_000.0
    if target_mp <= 0 or cur_mp <= float(target_mp):
        return img
    scale = (float(target_mp) / cur_mp) ** 0.5
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    if img.ndim == 3:
        return np.asarray(
            Image.fromarray(_to_rgb_uint8(img)).resize((nw, nh), Image.Resampling.BILINEAR)
        )
    return np.asarray(
        Image.fromarray(img.astype(np.uint8)).resize((nw, nh), Image.Resampling.BILINEAR)
    )


def _downsample_for_view(
    image: np.ndarray, target_mp: float = DEFAULT_PREVIEW_TARGET_MP
) -> np.ndarray:
    """LOD overview for the label window (alias of downsample_to_approx_mp)."""
    return downsample_to_approx_mp(image, target_mp=target_mp)


def get_screen_size_px() -> Tuple[int, int]:
    """Primary screen width × height in pixels (fallback 1920×1080).

    Prefer Qt (works on macOS / Windows / Linux when a QApplication exists,
    e.g. inside the label subprocess). Do **not** use tkinter — ``tk.Tk()``
    hard-crashes on recent macOS (``NSApplication macOSVersion`` / libtk).
    """
    try:
        from PyQt5.QtWidgets import QApplication

        app = QApplication.instance()
        if app is not None:
            screen = app.primaryScreen()
            if screen is not None:
                geo = screen.geometry()
                dpr = float(screen.devicePixelRatio())
                w = int(geo.width() * dpr)
                h = int(geo.height() * dpr)
                if w > 0 and h > 0:
                    return w, h
    except Exception:
        pass
    try:
        from AppKit import NSScreen  # type: ignore

        screen = NSScreen.mainScreen()
        if screen is not None:
            frame = screen.frame()
            scale = float(screen.backingScaleFactor())
            w = int(frame.size.width * scale)
            h = int(frame.size.height * scale)
            if w > 0 and h > 0:
                return w, h
    except Exception:
        pass
    return 1920, 1080


def screen_view_target_mp(fill: float = DEFAULT_SCREEN_VIEW_FILL) -> float:
    """
    Megapixels to roughly fill the primary screen (for label overview / feature preview).

    ``fill`` is a fraction of screen W×H (default 0.85). Zoomed UI still pulls
    high-res crops from the full working image.
    """
    w, h = get_screen_size_px()
    return max(0.5, (w * h / 1_000_000.0) * float(fill))


def resolve_view_target_mp(explicit: Optional[float] = None) -> float:
    """Use explicit MP if set; otherwise size from the screen."""
    if explicit is not None and float(explicit) > 0:
        return float(explicit)
    return screen_view_target_mp()


def max_side_for_target_mp(target_mp: float = DEFAULT_PREVIEW_TARGET_MP) -> int:
    """Square-side length whose area ≈ target_mp (for LOD overview)."""
    return max(256, int(round((max(float(target_mp), 0.1) * 1_000_000.0) ** 0.5)))


def compute_local_std(gray: np.ndarray, window: int = DEFAULT_TEXTURE_WINDOW) -> np.ndarray:
    """Local standard deviation of a 2D float image (odd window)."""
    from scipy.ndimage import uniform_filter

    g = np.asarray(gray, dtype=np.float32)
    w = int(window)
    if w < 3:
        w = 3
    if w % 2 == 0:
        w += 1
    mean = uniform_filter(g, size=w, mode="nearest")
    mean_sq = uniform_filter(g * g, size=w, mode="nearest")
    var = np.maximum(mean_sq - mean * mean, 0.0)
    return np.sqrt(var, dtype=np.float32)


def compute_sobel_mag(gray: np.ndarray) -> np.ndarray:
    """Sobel gradient magnitude of a 2D image."""
    from scipy.ndimage import sobel

    g = np.asarray(gray, dtype=np.float32)
    gx = sobel(g, axis=1, mode="nearest")
    gy = sobel(g, axis=0, mode="nearest")
    return np.sqrt(gx * gx + gy * gy).astype(np.float32)


def compute_lbp_map(
    gray: np.ndarray,
    p: int = DEFAULT_LBP_P,
    r: float = DEFAULT_LBP_R,
) -> np.ndarray:
    """Uniform LBP codes on gray (float32)."""
    from skimage.feature import local_binary_pattern

    g = np.asarray(gray, dtype=np.float32)
    g8 = np.clip(g, 0, 255).astype(np.uint8)
    codes = local_binary_pattern(g8, P=int(p), R=float(r), method="uniform")
    return codes.astype(np.float32)


def compute_gabor_maps(
    gray: np.ndarray,
    *,
    frequency: float = DEFAULT_GABOR_FREQUENCY,
    angles_deg: Sequence[float] = DEFAULT_GABOR_ANGLES_DEG,
) -> Dict[str, np.ndarray]:
    """
    Gabor filter magnitudes — stripes at given angles (cleavage / twins).

    Default angles 0°, 45°, 90°. Returns ``gabor_0``, ``gabor_45``, ``gabor_90``.
    """
    from skimage.filters import gabor

    g = np.asarray(gray, dtype=np.float32)
    g01 = g / 255.0 if float(np.nanmax(g)) > 1.5 else g
    out: Dict[str, np.ndarray] = {}
    for deg in angles_deg:
        theta = float(deg) * np.pi / 180.0
        real, imag = gabor(g01, frequency=float(frequency), theta=theta)
        mag = np.sqrt(real.astype(np.float32) ** 2 + imag.astype(np.float32) ** 2)
        out[f"gabor_{int(round(float(deg)))}"] = mag.astype(np.float32)
    return out


def compute_hessian_maps(
    gray: np.ndarray,
    *,
    sigma: float = DEFAULT_HESSIAN_SIGMA,
) -> Dict[str, np.ndarray]:
    """
    Hessian eigenvalue maps — “ridge vs blob?” (crack vs grain).

    Returns:
    - ``hessian_ridge`` — ridge strength (max |eigenvalue|)
    - ``hessian_aniso`` — how directional (|l1−l2| / (|l1|+|l2|))
    """
    from skimage.feature import hessian_matrix, hessian_matrix_eigvals

    g = np.asarray(gray, dtype=np.float32)
    g01 = g / 255.0 if float(np.nanmax(g)) > 1.5 else g
    H = hessian_matrix(g01, sigma=float(sigma), use_gaussian_derivatives=True)
    l1, l2 = hessian_matrix_eigvals(H)
    l1 = l1.astype(np.float32)
    l2 = l2.astype(np.float32)
    ridge = np.maximum(np.abs(l1), np.abs(l2))
    denom = np.abs(l1) + np.abs(l2) + 1e-6
    aniso = np.abs(l1 - l2) / denom
    return {
        "hessian_ridge": ridge.astype(np.float32),
        "hessian_aniso": aniso.astype(np.float32),
    }


def compute_local_entropy(gray: np.ndarray, window: int = 21) -> np.ndarray:
    """Local entropy of Y in an odd window (how mixed the grays are)."""
    from skimage.filters.rank import entropy
    from skimage.morphology import disk

    g8 = np.clip(np.asarray(gray), 0, 255).astype(np.uint8)
    w = int(window)
    if w < 3:
        w = 3
    radius = max(1, w // 2)
    return entropy(g8, disk(radius)).astype(np.float32)


def compute_fft_power_spectrum(gray: np.ndarray) -> np.ndarray:
    """Centered log power spectrum of a 2D image (whole-field fabric fingerprint)."""
    g = np.asarray(gray, dtype=np.float32)
    g = g - float(g.mean())
    spec = np.fft.fftshift(np.fft.fft2(g))
    power = np.log1p(np.abs(spec).astype(np.float32))
    return power


def _glcm_props_from_crop(gray_u8: np.ndarray) -> Dict[str, float]:
    from skimage.feature import graycomatrix, graycoprops

    if gray_u8.size < 16 or gray_u8.shape[0] < 2 or gray_u8.shape[1] < 2:
        return {name: 0.0 for name in GLCM_PROP_NAMES}
    # Quantize to 32 levels for speed
    q = (gray_u8.astype(np.float32) / 8.0).astype(np.uint8)
    q = np.clip(q, 0, 31)
    glcm = graycomatrix(
        q,
        distances=[1],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=32,
        symmetric=True,
        normed=True,
    )
    out: Dict[str, float] = {}
    for name in GLCM_PROP_NAMES:
        vals = graycoprops(glcm, name)
        out[name] = float(np.mean(vals))
    return out


def glcm_feature_maps(
    gray: np.ndarray,
    polygons: Sequence[Polygon],
    shape_hw: Optional[Tuple[int, int]] = None,
) -> Dict[str, np.ndarray]:
    """
    Polygon-level GLCM props broadcast onto pixels inside each polygon.
    Pixels outside polygons stay 0. If no polygons, fill image-wide GLCM constants.
    """
    g = np.asarray(gray, dtype=np.float32)
    h, w = shape_hw if shape_hw is not None else g.shape[:2]
    g8 = np.clip(g, 0, 255).astype(np.uint8)
    maps = {name: np.zeros((h, w), dtype=np.float32) for name in GLCM_PROP_NAMES}

    if not polygons:
        props = _glcm_props_from_crop(g8)
        for name, val in props.items():
            maps[name][:] = val
        return maps

    for poly in polygons:
        if poly is None or poly.is_empty:
            continue
        minx, miny, maxx, maxy = poly.bounds
        c0 = max(0, int(np.floor(minx)))
        c1 = min(w, int(np.ceil(maxx)) + 1)
        r0 = max(0, int(np.floor(miny)))
        r1 = min(h, int(np.ceil(maxy)) + 1)
        if c1 <= c0 or r1 <= r0:
            continue
        crop = g8[r0:r1, c0:c1]
        # Mask crop to polygon
        from matplotlib.path import Path as _MplPath

        yy, xx = np.mgrid[r0:r1, c0:c1]
        pts = np.column_stack([xx.ravel(), yy.ravel()])
        mask = _MplPath(list(poly.exterior.coords)).contains_points(pts).reshape(crop.shape)
        if not np.any(mask):
            continue
        props = _glcm_props_from_crop(crop)
        for name, val in props.items():
            region = maps[name][r0:r1, c0:c1]
            region[mask] = val
    return maps


def build_feature_stack(
    image_rgb: np.ndarray,
    toggles: Optional[Dict[str, bool]] = None,
    y_target: float = DEFAULT_Y_TARGET,
    apply_norm: bool = True,
    texture_window: int = DEFAULT_TEXTURE_WINDOW,
    lbp_p: int = DEFAULT_LBP_P,
    lbp_r: float = DEFAULT_LBP_R,
    polygons: Optional[Sequence[Polygon]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Build HxWxC float32 feature stack and ordered feature names.

    Training uses full working resolution. GLCM uses polygons when provided.
    """
    toggles = resolve_feature_toggles(toggles)
    rgb = (
        normalize_brightness(image_rgb, y_target=y_target)
        if apply_norm
        else np.asarray(image_rgb, dtype=np.float32)
    )
    y = relative_luminance(rgb).astype(np.float32)
    channels: List[np.ndarray] = []
    names: List[str] = []

    if toggles.get("r"):
        channels.append(rgb[..., 0].astype(np.float32))
        names.append("r")
    if toggles.get("g"):
        channels.append(rgb[..., 1].astype(np.float32))
        names.append("g")
    if toggles.get("b"):
        channels.append(rgb[..., 2].astype(np.float32))
        names.append("b")
    if toggles.get("y"):
        channels.append(y)
        names.append("y")
    if toggles.get("local_std"):
        for win in FEATURE_STD_WINDOWS:
            channels.append(compute_local_std(y, window=int(win)))
            names.append(f"local_std_{int(win)}")
    if toggles.get("local_grad"):
        channels.append(compute_sobel_mag(y))
        names.append("local_grad")
    if toggles.get("lbp"):
        for rad in FEATURE_LBP_RADII:
            tag = int(rad) if float(rad) == int(rad) else rad
            channels.append(compute_lbp_map(y, p=lbp_p, r=float(rad)))
            names.append(f"lbp_r{tag}")
    if toggles.get("gabor"):
        for name, arr in compute_gabor_maps(y).items():
            channels.append(arr)
            names.append(name)
    if toggles.get("hessian"):
        hmaps = compute_hessian_maps(y)
        channels.append(hmaps["hessian_ridge"])
        names.append("hessian_ridge")
        if toggles.get("hessian_aniso"):
            channels.append(hmaps["hessian_aniso"])
            names.append("hessian_aniso")
    elif toggles.get("hessian_aniso"):
        hmaps = compute_hessian_maps(y)
        channels.append(hmaps["hessian_aniso"])
        names.append("hessian_aniso")
    if toggles.get("entropy"):
        for win in FEATURE_ENTROPY_WINDOWS:
            channels.append(compute_local_entropy(y, window=int(win)))
            names.append(f"entropy_{int(win)}")
    if toggles.get("glcm"):
        gmaps = glcm_feature_maps(y, polygons or [], shape_hw=y.shape[:2])
        for prop in GLCM_PROP_NAMES:
            channels.append(gmaps[prop])
            names.append(f"glcm_{prop}")

    stack = np.stack(channels, axis=-1).astype(np.float32)
    return stack, names


def extract_training_table(
    image_rgb: np.ndarray,
    labels: np.ndarray,
    y_target: float = DEFAULT_Y_TARGET,
    apply_norm: bool = True,
    toggles: Optional[Dict[str, bool]] = None,
    texture_window: int = DEFAULT_TEXTURE_WINDOW,
    lbp_p: int = DEFAULT_LBP_P,
    lbp_r: float = DEFAULT_LBP_R,
    polygons: Optional[Sequence[Polygon]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Build float32 feature matrix X (N, C) and y (N,) from labeled pixels.

    Returns X, y, feature_names.
    """
    stack, names = build_feature_stack(
        image_rgb,
        toggles=toggles,
        y_target=y_target,
        apply_norm=apply_norm,
        texture_window=texture_window,
        lbp_p=lbp_p,
        lbp_r=lbp_r,
        polygons=polygons,
    )
    flat = stack.reshape(-1, stack.shape[-1])
    lab = np.asarray(labels).reshape(-1)
    valid = lab > 0
    if not np.any(valid):
        raise ValueError("No labeled pixels (labels > 0). Draw training polygons first.")
    X = flat[valid].astype(np.float32)
    y = lab[valid].astype(np.int32)
    return X, y, names

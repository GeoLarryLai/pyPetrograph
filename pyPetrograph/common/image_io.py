"""Image load/save and process-lifetime RGB RAM cache."""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image

from pyPetrograph.common.constants import CACHE_SUBDIR, PathLike, SUPPORTED_EXTENSIONS
from pyPetrograph.common.paths import stem_paths

# Process-lifetime RGB RAM cache (cleared on kernel restart)
_RGB_RAM_CACHE: Dict[str, np.ndarray] = {}
_RGB_RAM_META: Dict[str, Dict[str, Any]] = {}


def _to_rgb_uint8(arr: np.ndarray) -> np.ndarray:
    """Convert an array to HxWx3 uint8 RGB."""
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3 and arr.shape[2] >= 3:
        arr = arr[:, :, :3]
    else:
        raise ValueError(f"Unsupported image shape: {arr.shape}")

    if np.issubdtype(arr.dtype, np.floating):
        if arr.max() <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255)

    return np.asarray(arr, dtype=np.uint8)


def load_image(
    path: PathLike,
    scale: float = 1.0,
    max_side: Optional[int] = None,
    target_mp: Optional[float] = None,
) -> np.ndarray:
    """
    Load a common image format and return RGB uint8 (H, W, 3).

    Resize priority: ``max_side`` → ``target_mp`` (approx megapixels) → ``scale``.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()
    if suffix == ".geotiff":
        suffix = ".tiff"
    if suffix not in SUPPORTED_EXTENSIONS and suffix not in {".tif", ".tiff"}:
        warnings.warn(f"Unusual extension {suffix}; attempting to load anyway.")

    arr = None

    # Prefer PIL for ordinary RGB stills (quiet, broad format support).
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            arr = np.array(im)
    except Exception:
        arr = None

    # Fall back to rasterio for multi-band / GeoTIFF cases PIL cannot open.
    if arr is None and suffix in {".tif", ".tiff"}:
        try:
            import rasterio

            with rasterio.open(path) as src:
                if src.count >= 3:
                    data = src.read([1, 2, 3])
                    arr = np.transpose(data, (1, 2, 0))
                elif src.count == 1:
                    band = src.read(1)
                    arr = np.stack([band, band, band], axis=-1)
                else:
                    data = src.read()
                    arr = np.transpose(data[:3], (1, 2, 0))
        except Exception as exc:
            raise RuntimeError(f"Could not load image: {path}") from exc

    if arr is None:
        raise RuntimeError(f"Could not load image: {path}")

    arr = _to_rgb_uint8(arr)

    h, w = arr.shape[:2]
    new_w, new_h = w, h
    if max_side is not None and max(h, w) > max_side:
        if h >= w:
            new_h = max_side
            new_w = max(1, int(round(w * max_side / h)))
        else:
            new_w = max_side
            new_h = max(1, int(round(h * max_side / w)))
    elif target_mp is not None and target_mp > 0:
        cur_mp = (h * w) / 1_000_000.0
        if cur_mp > float(target_mp):
            s = (float(target_mp) / cur_mp) ** 0.5
            new_w = max(1, int(round(w * s)))
            new_h = max(1, int(round(h * s)))
    elif scale != 1.0:
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))

    if (new_w, new_h) != (w, h):
        arr = np.array(
            Image.fromarray(arr).resize((new_w, new_h), Image.Resampling.LANCZOS)
        )

    return _to_rgb_uint8(arr)


def save_rgb_png(path: PathLike, image: np.ndarray) -> Path:
    """Save an RGB or single-channel array as PNG."""
    path = Path(path)
    arr = np.asarray(image)
    if arr.ndim == 2:
        Image.fromarray(arr.astype(np.uint8)).save(path)
    else:
        Image.fromarray(_to_rgb_uint8(arr)).save(path)
    return path


def _load_settings_dict(
    scale: float,
    max_side: Optional[int],
    load_target_mp: Optional[float],
) -> Dict[str, Any]:
    return {
        "scale": float(scale),
        "max_side": None if max_side is None else int(max_side),
        "load_target_mp": None if load_target_mp is None else float(load_target_mp),
    }


def save_rgb_cache(
    image_path: PathLike,
    image: np.ndarray,
    *,
    scale: float = 1.0,
    max_side: Optional[int] = None,
    load_target_mp: Optional[float] = None,
    stem: Optional[str] = None,
) -> Path:
    """Store working RGB in RAM only (no disk). Returns a synthetic path for callers."""
    pth = Path(image_path).resolve()
    key = str(pth)
    arr = _to_rgb_uint8(image)
    _RGB_RAM_CACHE[key] = arr
    _RGB_RAM_META[key] = _load_settings_dict(scale, max_side, load_target_mp)
    return pth.parent / CACHE_SUBDIR / f"{stem or pth.stem}_rgb_cache.npy"


def load_rgb_cache(
    image_path: PathLike,
    *,
    scale: float = 1.0,
    max_side: Optional[int] = None,
    load_target_mp: Optional[float] = None,
    stem: Optional[str] = None,
) -> Optional[np.ndarray]:
    """Return RAM-cached RGB if present and load settings match."""
    key = str(Path(image_path).resolve())
    arr = _RGB_RAM_CACHE.get(key)
    meta = _RGB_RAM_META.get(key)
    if arr is None or meta is None:
        return None
    want = _load_settings_dict(scale, max_side, load_target_mp)
    for k in ("scale", "max_side", "load_target_mp"):
        if meta.get(k) != want.get(k):
            return None
    return arr


def ensure_rgb_cache(
    image_path: PathLike,
    *,
    scale: float = 1.0,
    max_side: Optional[int] = None,
    load_target_mp: Optional[float] = None,
    stem: Optional[str] = None,
) -> np.ndarray:
    """Return working RGB from RAM cache or load_image (store in RAM; no disk)."""
    path_r = Path(image_path).resolve()
    cached = load_rgb_cache(
        path_r,
        scale=scale,
        max_side=max_side,
        load_target_mp=load_target_mp,
        stem=stem,
    )
    if cached is not None:
        return cached
    arr = load_image(
        path_r, scale=scale, max_side=max_side, target_mp=load_target_mp
    )
    save_rgb_cache(
        path_r,
        arr,
        scale=scale,
        max_side=max_side,
        load_target_mp=load_target_mp,
        stem=stem,
    )
    return arr


def clear_rgb_ram_cache() -> None:
    """Drop all RAM RGB caches (normally cleared on kernel restart)."""
    _RGB_RAM_CACHE.clear()
    _RGB_RAM_META.clear()

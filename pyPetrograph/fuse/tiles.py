"""
Tiling for whole-slide stacks and Zeiss Axioscan folder helpers.

The Axioscan exports are gigapixel at full size (≈55,000 × 33,000 px). Never load
those whole. Pick a ``NN scale`` folder (``02 scale`` ≈ 0.75 MP, ``25 scale`` ≈ 117 MP)
with ``axioscan_scale_folder`` and, above a few megapixels, run per-pixel work
tile by tile with ``apply_tiled``.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Dict, Iterator, Optional, Sequence, Tuple

import numpy as np

from pyPetrograph.align.scene import Scene, xpl_slot
from pyPetrograph.common.constants import SUPPORTED_EXTENSIONS, PathLike

DEFAULT_TILE = 1024
DEFAULT_OVERLAP = 128


def iter_tiles(
    h: int, w: int, *, tile: int = DEFAULT_TILE, overlap: int = DEFAULT_OVERLAP
) -> Iterator[Tuple[int, int, int, int]]:
    """Yield ``(r0, r1, c0, c1)`` windows covering an ``h×w`` grid with overlap."""
    tile = int(tile)
    overlap = int(min(max(0, overlap), tile - 1))
    step = tile - overlap
    rows = list(range(0, max(1, h - tile + 1), step))
    cols = list(range(0, max(1, w - tile + 1), step))
    if rows[-1] + tile < h:
        rows.append(h - tile)
    if cols[-1] + tile < w:
        cols.append(w - tile)
    for r0 in rows:
        r0 = max(0, r0)
        r1 = min(h, r0 + tile)
        for c0 in cols:
            c0 = max(0, c0)
            c1 = min(w, c0 + tile)
            yield r0, r1, c0, c1


def _ramp_weight(th: int, tw: int, overlap: int) -> np.ndarray:
    """2-D weight that fades linearly to ~0 over ``overlap`` px at each tile edge."""
    if overlap <= 0:
        return np.ones((th, tw), dtype=np.float32)

    def _1d(n: int) -> np.ndarray:
        x = np.ones(n, dtype=np.float32)
        k = min(int(overlap), n // 2)
        if k > 0:
            ramp = (np.arange(1, k + 1, dtype=np.float32)) / float(k + 1)
            x[:k] = ramp
            x[-k:] = ramp[::-1]
        return x

    return np.outer(_1d(th), _1d(tw)).astype(np.float32)


def apply_tiled(
    fn: Callable[[np.ndarray], np.ndarray],
    stack: np.ndarray,
    *,
    tile: int = DEFAULT_TILE,
    overlap: int = DEFAULT_OVERLAP,
    out_channels: Optional[int] = None,
    progress: bool = False,
) -> np.ndarray:
    """
    Run ``fn(tile_HxWxC) -> HxWxK`` over overlapping tiles and blend the overlaps.

    Overlaps are averaged with edge-fading weights, so seams do not show. If the
    stack fits one tile, ``fn`` is called once. Output is float32 HxWxK.
    """
    arr = np.asarray(stack)
    if arr.ndim == 2:
        arr = arr[..., None]
    h, w = arr.shape[:2]
    if h <= tile and w <= tile:
        out = np.asarray(fn(arr), dtype=np.float32)
        return out if out.ndim == 3 else out[..., None]

    acc: Optional[np.ndarray] = None
    weight = np.zeros((h, w), dtype=np.float32)
    windows = list(iter_tiles(h, w, tile=tile, overlap=overlap))
    for i, (r0, r1, c0, c1) in enumerate(windows):
        res = np.asarray(fn(arr[r0:r1, c0:c1]), dtype=np.float32)
        if res.ndim == 2:
            res = res[..., None]
        if acc is None:
            k = int(out_channels) if out_channels is not None else int(res.shape[-1])
            acc = np.zeros((h, w, k), dtype=np.float32)
        wgt = _ramp_weight(r1 - r0, c1 - c0, overlap)
        acc[r0:r1, c0:c1] += res * wgt[..., None]
        weight[r0:r1, c0:c1] += wgt
        if progress:
            print(f"  tile {i + 1}/{len(windows)}", end="\r")
    if progress:
        print()
    assert acc is not None
    return acc / np.maximum(weight, 1e-6)[..., None]


# --- Zeiss Axioscan polarization exports ---

# ..._ScanRegion0_Bright_scale0p02.jpg / ..._pPol_0°_... / ..._cPol_... / ...__xPol_15°_...
_AXIO_RE = re.compile(r"_(Bright|pPol|cPol|xPol)(?:_(\d+)\s*[°º˚]?)?_scale", re.IGNORECASE)
_AXIO_KIND = {"bright": "bright", "ppol": "ppl", "cpol": "cpol", "xpol": "xpl"}


def axioscan_layers(folder: PathLike) -> Dict[str, Tuple[Path, Optional[float]]]:
    """
    Map Zeiss Axioscan exports in one ``NN scale`` folder to slots.

    Returns ``{slot: (path, angle_deg)}`` with slots ``bright``, ``ppl``, ``cpol``,
    ``xpl_000``, ``xpl_015``, … Files that do not match the naming are ignored.
    """
    out: Dict[str, Tuple[Path, Optional[float]]] = {}
    folder = Path(folder)
    for f in sorted(folder.iterdir()):
        if not f.is_file() or f.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        m = _AXIO_RE.search(f.name)
        if not m:
            continue
        kind = _AXIO_KIND[m.group(1).lower()]
        angle = float(m.group(2)) if m.group(2) is not None else None
        slot = xpl_slot(angle if angle is not None else 0.0) if kind == "xpl" else kind
        out[slot] = (f, angle)
    return out


def axioscan_scale_folder(sample_dir: PathLike, scale: float) -> Path:
    """
    Find the ``02 scale`` / ``25scale`` / ``50 scale`` folder for ``scale`` (0.02, 0.25, 0.5).

    Tolerates the missing space seen in some exports.
    """
    sample_dir = Path(sample_dir)
    pct = int(round(float(scale) * 100))
    want = {f"{pct:02d}scale", f"{pct}scale"}
    for d in sorted(sample_dir.iterdir()):
        if d.is_dir() and d.name.replace(" ", "").lower() in want:
            return d
    raise FileNotFoundError(f"no '{pct:02d} scale' folder under {sample_dir}")


def add_axioscan_layers(
    scene: Scene,
    folder: PathLike,
    *,
    working: str = "ppl",
    include: Sequence[str] = ("ppl", "bright", "xpl"),
) -> Scene:
    """
    Add every matching Axioscan export in ``folder`` to ``scene``.

    ``working`` is the slot used as the pixel grid (falls back to ``bright`` if there
    is no ``ppl``). These exports come from one scanner pass, so they are already on
    one grid: transforms stay identity and the auto-align cell can be skipped.
    """
    layers = axioscan_layers(folder)
    if not layers:
        raise FileNotFoundError(f"no Axioscan exports found in {folder}")
    kinds_ok = {k.lower() for k in include}
    work = working if working in layers else ("bright" if "bright" in layers else next(iter(layers)))
    for slot, (path, angle) in layers.items():
        if slot.split("_")[0] not in kinds_ok:
            continue
        scene.add_layer(slot, path, is_working=(slot == work), angle_deg=angle)
    return scene

"""Grain QC: launch GrainPlot (subprocess) and load saved grains."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from pyPetrograph.common.constants import PathLike
from pyPetrograph.common.paths import project_models_dir, stem_paths

SEG_UNET_NAME = "seg_model_smooth_labels.keras"
SEG_SAM_NAME = "sam2.1_hiera_large.pt"
SEG_PYTHON_ENV = "PYPETROGRAPH_SEG_PYTHON"
SEG_UNET_URL = (
    "https://raw.githubusercontent.com/zsylvester/segmenteverygrain/main/models/"
    "seg_model_smooth_labels.keras"
)
SEG_SAM_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"


def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"Downloading {dest.name} …")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest)
    print(f"  saved {dest}")
    return dest


def ensure_seg_weights(
    *,
    sam2: bool = False,
    models_dir: Optional[PathLike] = None,
) -> Dict[str, Path]:
    """
    Download SegmentEveryGrain U-Net (26 MB) into ``models/`` if missing.
    Leave ``sam2=False`` (default). ``sam2=True`` also fetches the SAM 2.1
    large checkpoint (~860 MB) for optional shape refine. Existing files stay.
    """
    d = Path(models_dir) if models_dir is not None else project_models_dir()
    d.mkdir(parents=True, exist_ok=True)
    unet = d / SEG_UNET_NAME
    if not unet.exists():
        _download(SEG_UNET_URL, unet)
    out: Dict[str, Path] = {"unet": unet}
    if sam2:
        sam = d / SEG_SAM_NAME
        if not sam.exists():
            _download(SEG_SAM_URL, sam)
        out["sam"] = sam
    return out


def grain_qc_paths(image_path: PathLike) -> Tuple[Path, Path]:
    """``labels/{stem}_grains.geojson`` and ``labels/{stem}_grains_mask.png``."""
    sp = stem_paths(image_path)
    return sp.grains, sp.grains_mask


def overlay_labels(
    rgb: np.ndarray,
    labels: np.ndarray,
    color: Tuple[int, int, int] = (255, 40, 40),
) -> np.ndarray:
    """Draw grain boundaries from a label map on an RGB copy."""
    from skimage.segmentation import find_boundaries

    out = np.asarray(rgb, dtype=np.uint8).copy()
    if out.ndim == 2:
        out = np.stack([out, out, out], axis=-1)
    bound = find_boundaries(np.asarray(labels), mode="outer")
    out[bound] = color
    return out


def seg_setup_hint() -> str:
    models = project_models_dir()
    return (
        "Grain U-Net / GrainPlot run as a subprocess of this Python (conda work). "
        "The notebook kernel does not import TensorFlow.\n"
        f"  {models / SEG_UNET_NAME}\n"
        f"  {models / SEG_SAM_NAME}"
    )


def _seg_python() -> Optional[Path]:
    override = os.environ.get(SEG_PYTHON_ENV, "").strip()
    if override:
        p = Path(override).expanduser()
        return p if p.exists() else None
    exe = Path(sys.executable)
    return exe if exe.exists() else None


def _tf_worker_env(*, mpl_backend: str = "Agg") -> Dict[str, str]:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env["MPLBACKEND"] = mpl_backend
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    env.setdefault("KERAS_BACKEND", "tensorflow")
    env["PYTHONFAULTHANDLER"] = "1"
    return env


def load_grain_mask(image_path: PathLike) -> Optional[np.ndarray]:
    """Load saved grain QC mask, or None."""
    _geo, mask = grain_qc_paths(image_path)
    if not mask.exists():
        return None
    from PIL import Image

    arr = np.asarray(Image.open(mask))
    if arr.ndim > 2:
        arr = arr[..., 0]
    return arr


def _n_labels(labels: np.ndarray) -> int:
    u = np.unique(labels)
    return int(np.sum(u > 0))


def load_grain_qc(image_path: PathLike) -> Dict[str, Any]:
    """
    Load grains already saved by GrainPlot (this notebook or SegmentEveryGrain_PPL).

    Does not open GrainPlot. If only the GeoJSON exists, rasterize a mask (no TF).
    """
    image_path = Path(image_path)
    geo, mask_p = grain_qc_paths(image_path)
    lab = load_grain_mask(image_path)
    if lab is None and geo.exists():
        from PIL import Image
        from shapely.geometry import MultiPolygon, shape as shapely_shape
        from skimage.draw import polygon as draw_polygon

        rgb = np.asarray(Image.open(image_path).convert("RGB"))
        h, w = rgb.shape[:2]
        lab = np.zeros((h, w), dtype=np.int32)
        blob = json.loads(geo.read_text(encoding="utf-8"))
        for i, feat in enumerate(blob.get("features") or [], start=1):
            geom = feat.get("geometry")
            if not geom:
                continue
            g = shapely_shape(geom)
            geoms = list(g.geoms) if isinstance(g, MultiPolygon) else [g]
            for poly in geoms:
                if poly is None or poly.is_empty:
                    continue
                xs, ys = poly.exterior.xy
                rr, cc = draw_polygon(
                    np.round(ys).astype(int),
                    np.round(xs).astype(int),
                    shape=(h, w),
                )
                lab[rr, cc] = i
        from PIL import Image

        if int(lab.max()) <= 255:
            Image.fromarray(lab.astype(np.uint8), mode="L").save(mask_p)
        else:
            Image.fromarray(lab.astype(np.uint16), mode="I;16").save(mask_p)
    if lab is None:
        return {
            "ok": False,
            "n": 0,
            "loaded": False,
            "geojson": geo,
            "mask": mask_p,
            "error": f"no saved grains at {geo.name} / {mask_p.name}",
            "hint": "Set LOAD_SAVED_GRAINS = False to run GrainPlot first.",
        }
    n = _n_labels(lab)
    print(f"Loaded saved grains: n={n}  {geo.name}")
    return {
        "ok": True,
        "n": n,
        "loaded": True,
        "geojson": geo,
        "mask": mask_p,
        "error": None,
        "hint": None,
    }


def launch_ppl_grain_qc(
    image_path: PathLike,
    *,
    wait: bool = True,
    reuse: bool = True,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Alias for ``launch_grain_qc`` (PPL teacher / ``SegmentEveryGrain_PPL``)."""
    return launch_grain_qc(image_path, wait=wait, reuse=reuse, **kwargs)


def load_ppl_grain_qc(image_path: PathLike) -> Dict[str, Any]:
    """Alias for ``load_grain_qc``."""
    return load_grain_qc(image_path)


def launch_grain_qc(
    image_path: PathLike,
    *,
    wait: bool = True,
    reuse: bool = True,
    pred_path: Optional[PathLike] = None,
    labels_path: Optional[PathLike] = None,
    use_sam: bool = False,
    interactive: bool = True,
) -> Dict[str, Any]:
    """
    Open GrainPlot on ``image_path`` (subprocess; kernel never imports TensorFlow).

    ``pred_path``: HxWx3 probability cube from the N-channel U-Net (skips RGB U-Net).
    ``labels_path``: HxW instance ids (e.g. watershed bootstrap).
    ``reuse``: if geojson exists and neither pred/labels is given, reopen it.
    ``use_sam``: ``False`` (default) is U-Net only — enough for counts and
    the object table. ``True`` downloads SAM 2.1 weights (~860 MB) if missing
    and refines polygon shape.
    ``interactive``: ``True`` opens the GrainPlot window (close it to save);
    ``False`` writes grains without a window.
    """
    image_path = Path(image_path).expanduser().resolve()
    py = _seg_python()
    unet_w = project_models_dir() / SEG_UNET_NAME
    sam_w = project_models_dir() / SEG_SAM_NAME
    geo, mask = grain_qc_paths(image_path)
    if py is None:
        return {"ok": False, "error": "no Python for GrainPlot worker", "hint": seg_setup_hint()}
    worker = Path(__file__).resolve().parent / "qc_worker.py"
    env = _tf_worker_env(mpl_backend="QtAgg" if interactive else "Agg")
    cmd = [
        str(py),
        str(worker),
        "--image",
        str(image_path),
        "--geojson",
        str(geo),
        "--mask",
        str(mask),
    ]
    if unet_w.exists():
        cmd.extend(["--unet", str(unet_w)])
    if use_sam and sam_w.exists():
        cmd.extend(["--sam", str(sam_w)])
    if pred_path is not None:
        cmd.extend(["--pred", str(pred_path)])
    elif labels_path is not None:
        cmd.extend(["--labels", str(labels_path)])
    elif reuse and geo.exists():
        cmd.extend(["--load-geojson", str(geo)])
    if not interactive:
        cmd.append("--no-ui")
    print(f"GrainPlot QC: {image_path.name}" + ("" if interactive else " (no window)"))
    if not wait:
        subprocess.Popen(cmd, env=env, cwd=str(image_path.parent))
        return {
            "ok": True,
            "n": 0,
            "geojson": geo,
            "mask": mask,
            "error": None,
            "hint": "GrainPlot window opened. Close it to save." if interactive else None,
        }
    proc = subprocess.run(cmd, env=env, cwd=str(image_path.parent))
    ok = proc.returncode == 0 and mask.exists()
    n = 0
    if mask.exists():
        from PIL import Image

        n = _n_labels(np.asarray(Image.open(mask)))
    return {
        "ok": ok,
        "n": n,
        "geojson": geo,
        "mask": mask,
        "error": None if ok else f"QC worker exited {proc.returncode}",
        "hint": None if ok else seg_setup_hint(),
    }

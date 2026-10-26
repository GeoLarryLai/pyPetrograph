"""Kernel-side launcher for ``seg_worker.py`` (the kernel never imports TensorFlow)."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

import numpy as np

from pyPetrograph.common.constants import SEG_INSTALL_URL, PathLike


def read_polygons_geojson(path: PathLike) -> List:
    """Shapely polygons from a SEG ``save_polygons`` GeoJSON (json + shapely only)."""
    from shapely.geometry import shape

    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    polys = []
    for feat in doc.get("features") or []:
        geom = (feat or {}).get("geometry")
        if geom:
            g = shape(geom)
            if not g.is_empty:
                polys.append(g)
    return polys


def run_seg_bse(
    stretched: np.ndarray,
    dark: np.ndarray,
    out_geojson: PathLike,
    *,
    use_sam: bool = True,
    edit: bool = True,
    min_area: float = 25.0,
    drop_frac: float = 0.5,
    load_from: Optional[PathLike] = None,
) -> List:
    """
    Run SEG on the stretched BSE in a subprocess and return shapely polygons.

    ``ensure_seg_weights(sam2=use_sam)``; temp PNG + ``.npy`` for the worker;
    ``edit=True`` opens GrainPlot (close the window to save). Raises on worker failure.
    ``load_from`` (an existing grains file) skips SEG and opens those grains in GrainPlot; with no
    editor (``edit=False`` or no screen) it just returns them.
    """
    from PIL import Image

    from pyPetrograph.amics.windows import has_screen
    from pyPetrograph.objects.qc import _seg_python, _tf_worker_env, ensure_seg_weights

    if edit and not has_screen():
        print("No screen for the GrainPlot editor: running SEG without it (edit=False).")
        edit = False
    if load_from is not None and not edit:
        return read_polygons_geojson(load_from)

    stretched = np.asarray(stretched)
    dark = np.asarray(dark, dtype=bool)
    if stretched.ndim == 2:
        rgb = np.repeat(stretched[..., None], 3, axis=-1)
    else:
        rgb = stretched[..., :3]
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    if dark.shape != rgb.shape[:2]:
        raise ValueError(f"dark {dark.shape} vs image {rgb.shape[:2]}")

    py = _seg_python()
    if py is None:
        raise RuntimeError(
            "No Python found for the SEG worker. It runs in the environment this was launched from "
            f"({sys.prefix}) unless PYPETROGRAPH_SEG_PYTHON points to another python."
        )
    if Path(py) == Path(sys.executable):
        import importlib.util

        if importlib.util.find_spec("segmenteverygrain") is None:
            raise RuntimeError(
                f"SegmentEveryGrain is not installed in this environment ({sys.prefix}). It is optional and "
                "installed with its own instructions: https://github.com/zsylvester/segmenteverygrain"
            )
    weights = ensure_seg_weights(sam2=bool(use_sam))
    worker = Path(__file__).resolve().parent / "seg_worker.py"
    out = Path(out_geojson)
    out.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="amics_seg_") as tmp:
        tmp = Path(tmp)
        png, npy, log = tmp / "bse.png", tmp / "dark.npy", tmp / "stderr.log"
        Image.fromarray(rgb).save(png)
        np.save(npy, dark)
        cmd = [
            str(py), str(worker),
            "--image", str(png), "--dark", str(npy),
            "--unet", str(weights["unet"]),
            "--min-area", str(float(min_area)),
            "--drop-frac", str(float(drop_frac)),
            "--out", str(out),
        ]
        if use_sam and "sam" in weights:
            cmd += ["--sam", str(weights["sam"])]
        if load_from is not None:
            cmd += ["--load", str(Path(load_from))]
        if not edit:
            cmd.append("--no-ui")
        env = _tf_worker_env(mpl_backend="QtAgg" if edit else "Agg")
        env["PYTHONIOENCODING"] = "utf-8"
        if load_from is not None:
            print("GrainPlot opens on the saved grains (loading SAM 2.1 first if on); close the window to save.")
        else:
            print("SEG worker started" + (" (GrainPlot opens when done; close it to save)" if edit else ""))
        if edit:
            print(
                "GrainPlot: click grain = select | D delete | M merge | Z undo | Esc clear | "
                "click empty area = add grain (SAM 2.1 on) | close window = save"
            )
        with open(log, "w", encoding="utf-8") as errf:
            proc = subprocess.run(cmd, env=env, stderr=errf)
        if proc.returncode != 0:
            tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-25:]
            print("\n".join(tail))
            raise RuntimeError(f"SEG worker exited {proc.returncode} (SegmentEveryGrain setup: {SEG_INSTALL_URL})")
    if not out.exists():
        raise RuntimeError(f"SEG worker finished but wrote no file: {out}")
    return read_polygons_geojson(out)

"""
Native-lib prep for TensorFlow / SAM2 subprocesses.

Copied from ``align/seg_worker.py`` so outline workers never import
``pyPetrograph`` (geopandas → pyarrow SIGSEGV with TF 2.21). Import this file
as a sibling of the worker scripts, never from the Jupyter kernel.
"""
from __future__ import annotations

import faulthandler
import os
import sys
from typing import Any, List


def _block_pyarrow() -> None:
    """Stop pyarrow from loading. Its C library SIGSEGVs with TensorFlow 2.21."""

    class _BlockPyarrow:
        def find_spec(self, fullname, path=None, target=None):  # noqa: ANN001
            if fullname == "pyarrow" or str(fullname).startswith("pyarrow."):
                raise ModuleNotFoundError(
                    "pyarrow blocked (native conflict with TensorFlow)"
                )
            return None

    if any(type(finder).__name__ == "_BlockPyarrow" for finder in sys.meta_path):
        return
    sys.meta_path.insert(0, _BlockPyarrow())


def _prepare_native_libs() -> None:
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("KERAS_BACKEND", "tensorflow")
    faulthandler.enable()
    _block_pyarrow()


def _device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _polygons_to_labels(grains: List[Any], hw: tuple):
    import numpy as np
    from shapely.geometry import MultiPolygon
    from skimage.draw import polygon as draw_polygon

    h, w = int(hw[0]), int(hw[1])
    lab = np.zeros((h, w), dtype=np.int32)
    for i, g in enumerate(grains, start=1):
        if g is None or getattr(g, "is_empty", False):
            continue
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
    return lab


def _save_mask_png(path, labels) -> None:
    from pathlib import Path

    from PIL import Image
    import numpy as np

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lab = np.asarray(labels)
    if int(lab.max()) <= 255:
        Image.fromarray(lab.astype(np.uint8), mode="L").save(path)
    else:
        Image.fromarray(lab.astype(np.uint16), mode="I;16").save(path)


def _load_sam_model(ckpt: str):
    """Raw SAM2 model (what ``sam_segmentation`` wants) plus a predictor for GrainPlot."""
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    device = _device()
    sam = build_sam2("configs/sam2.1/sam2.1_hiera_l.yaml", ckpt, device=device)
    return sam, SAM2ImagePredictor(sam)

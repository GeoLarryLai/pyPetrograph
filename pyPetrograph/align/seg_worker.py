#!/usr/bin/env python3
"""
SegmentEveryGrain worker. Run in a subprocess, not inside the Jupyter kernel.

Safe to use conda env ``work``'s Python (TensorFlow 2.21 + numpy 2 imports).
The worker blocks pyarrow (native crash with TensorFlow). The notebook still
must not import this module in-process.
"""
from __future__ import annotations

import argparse
import faulthandler
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional


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


def _polygons_to_labels(grains: List[Any], hw: tuple) -> Any:
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


def _load_unet(path: str):
    from keras.saving import load_model
    from segmenteverygrain import weighted_crossentropy

    return load_model(
        path, custom_objects={"weighted_crossentropy": weighted_crossentropy}
    )


def _load_sam(ckpt: str):
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    device = _device()
    sam = build_sam2("configs/sam2.1/sam2.1_hiera_l.yaml", ckpt, device=device)
    return SAM2ImagePredictor(sam)


def _run_one(
    image_path: str,
    unet,
    sam,
    *,
    use_sam: bool,
    hw: tuple,
) -> Dict[str, Any]:
    import numpy as np
    from segmenteverygrain import predict_large_image

    grains, _pred, _coords = predict_large_image(
        image_path,
        unet,
        sam=sam if use_sam else None,
        use_sam=use_sam,
        remove_edge_grains=False,
    )
    grains = list(grains or [])
    lab = _polygons_to_labels(grains, hw)
    method = "seg_unet_sam" if use_sam else "seg_unet"
    return {
        "ok": True,
        "n": int(len(grains)),
        "method": method,
        "error": None,
        "labels": lab,
        "polygons": grains,
    }


def main(argv: Optional[List[str]] = None) -> int:
    _prepare_native_libs()
    p = argparse.ArgumentParser(description="SEG worker (subprocess; not the Jupyter kernel)")
    p.add_argument("--image", required=True)
    p.add_argument("--unet", required=True)
    p.add_argument("--sam", default="")
    p.add_argument("--out-dir", required=True)
    p.add_argument(
        "--modes",
        default="unet",
        help="comma list: unet and/or sam",
    )
    args = p.parse_args(argv)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    modes = {m.strip() for m in str(args.modes).split(",") if m.strip()}
    result: Dict[str, Any] = {
        "unet": {"ok": False, "n": 0, "method": "seg_unet", "error": "not run"},
        "sam": {"ok": False, "n": 0, "method": "seg_unet_sam", "error": "not run"},
    }

    try:
        from PIL import Image

        im = Image.open(args.image)
        hw = (int(im.size[1]), int(im.size[0]))
        unet = _load_unet(args.unet)

        if "unet" in modes:
            one = _run_one(args.image, unet, None, use_sam=False, hw=hw)
            lab = one.pop("labels")
            result["unet"] = one
            import numpy as np

            np.save(out / "unet_labels.npy", lab)

        if "sam" in modes:
            if not args.sam:
                result["sam"]["error"] = "SAM2 weights path was empty"
            else:
                try:
                    sam_pred = _load_sam(args.sam)
                    one = _run_one(args.image, unet, sam_pred, use_sam=True, hw=hw)
                    lab = one.pop("labels")
                    result["sam"] = one
                    import numpy as np

                    np.save(out / "sam_labels.npy", lab)
                except Exception as sam_exc:
                    result["sam"]["error"] = f"{type(sam_exc).__name__}: {sam_exc}"
                    traceback.print_exc()
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
        for key in modes:
            if key in result and not result[key].get("ok"):
                result[key]["error"] = err
        (out / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(err, file=sys.stderr)
        return 1

    (out / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    _prepare_native_libs()
    raise SystemExit(main())

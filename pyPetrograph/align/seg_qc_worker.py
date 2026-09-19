#!/usr/bin/env python3
"""
PPL GrainPlot QC worker. Subprocess only — never import in the Jupyter kernel.

Blocks pyarrow (TensorFlow SIGSEGV). Opens SEG's GrainPlot; on close writes
``*_grains.geojson`` and ``*_grains_mask.png``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, List, Optional

# Sibling import — do not load pyPetrograph (geopandas/pyarrow SIGSEGV with TF).
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from seg_worker import (  # noqa: E402
    _load_sam,
    _load_unet,
    _polygons_to_labels,
    _prepare_native_libs,
    _run_one,
)

_prepare_native_libs()


def _save_mask_png(path: Path, labels: Any) -> None:
    from PIL import Image
    import numpy as np

    lab = np.asarray(labels)
    if lab.max() <= 255:
        Image.fromarray(lab.astype(np.uint8), mode="L").save(path)
    else:
        Image.fromarray(lab.astype(np.uint16), mode="I;16").save(path)


def main(argv: Optional[List[str]] = None) -> int:
    os.environ.setdefault("MPLBACKEND", "QtAgg")
    p = argparse.ArgumentParser(description="SEG GrainPlot QC (subprocess)")
    p.add_argument("--image", required=True)
    p.add_argument("--unet", required=True)
    p.add_argument("--sam", default="")
    p.add_argument("--geojson", required=True)
    p.add_argument("--mask", required=True)
    p.add_argument("--load-geojson", default="")
    args = p.parse_args(argv)

    geojson = Path(args.geojson)
    mask_path = Path(args.mask)
    geojson.parent.mkdir(parents=True, exist_ok=True)
    mask_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib

        matplotlib.use("QtAgg")
        import matplotlib.pyplot as plt
        import numpy as np
        from PIL import Image
        from segmenteverygrain.interactions import GrainPlot, polygons_to_grains, save_grains

        rgb = np.asarray(Image.open(args.image).convert("RGB"))
        hw = (int(rgb.shape[0]), int(rgb.shape[1]))
        sam_pred = None
        if args.sam:
            sam_pred = _load_sam(args.sam)

        load_path = Path(args.load_geojson) if args.load_geojson else None
        grains_list: List[Any] = []
        if load_path is not None and load_path.exists():
            from segmenteverygrain import read_polygons

            grains_list = polygons_to_grains(read_polygons(str(load_path)), rgb)
            print(f"Loaded {len(grains_list)} grains from {load_path.name}")
        else:
            unet = _load_unet(args.unet)
            one = _run_one(
                args.image,
                unet,
                sam_pred,
                use_sam=sam_pred is not None,
                hw=hw,
            )
            from shapely.geometry import Polygon
            from skimage.measure import find_contours, regionprops

            polys = list(one.get("polygons") or [])
            if not polys:
                lab = one["labels"]
                for prop in regionprops(lab):
                    minr, minc, maxr, maxc = prop.bbox
                    if (maxr - minr) < 2 or (maxc - minc) < 2:
                        continue
                    crop = (lab[minr:maxr, minc:maxc] == prop.label).astype(np.float32)
                    for contour in find_contours(crop, 0.5):
                        if len(contour) < 6:
                            continue
                        xy = np.column_stack([contour[:, 1] + minc, contour[:, 0] + minr])
                        poly = Polygon(xy)
                        if poly.is_valid and not poly.is_empty:
                            polys.append(poly)
                        break
            grains_list = polygons_to_grains(polys, rgb)
            print(f"SEG found {len(grains_list)} grains (U-Net+SAM)")

        gp = GrainPlot(grains=grains_list, image=rgb, predictor=sam_pred)
        print(
            "GrainPlot: click to add, D delete, M merge. Close the window to save."
        )
        plt.show()
        edited = list(gp.get_grains() or [])
        save_grains(str(geojson), edited)
        polys = [g.polygon for g in edited if getattr(g, "polygon", None) is not None]
        lab = _polygons_to_labels(polys, hw)
        _save_mask_png(mask_path, lab)
        meta = {"ok": True, "n": int(len(edited)), "geojson": str(geojson), "mask": str(mask_path)}
        (geojson.parent / f"{geojson.stem}_qc.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
        print(f"Saved {len(edited)} grains → {geojson.name} and {mask_path.name}")
        return 0
    except Exception as exc:
        traceback.print_exc()
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    _prepare_native_libs()
    raise SystemExit(main())

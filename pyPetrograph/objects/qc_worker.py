#!/usr/bin/env python3
"""
GrainPlot QC worker. Subprocess only — never import in the Jupyter kernel.

If ``--pred`` (HxWx3 bg/grain/boundary probabilities) is given, skip the RGB
U-Net and feed those maps to SEG ``label_grains`` + optional SAM2. If
``--labels`` (HxW instance ids) is given, turn them into polygons and skip both.
Otherwise run the shipped RGB U-Net (+ SAM2) on ``--image`` like the old
PPL teacher.

On close: ``*_grains.geojson`` + ``*_grains_mask.png``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, List, Optional

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from tf_env import (  # noqa: E402
    _load_sam_model,
    _polygons_to_labels,
    _prepare_native_libs,
    _save_mask_png,
)

_prepare_native_libs()


def _labels_to_polys(lab) -> list:
    import numpy as np
    from shapely.geometry import Polygon
    from skimage.measure import find_contours, regionprops

    lab = np.asarray(lab)
    polys = []
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
    return polys


def _grains_from_pred(rgb, pred, sam_model, min_area: int = 50) -> list:
    """SEG ``label_grains`` (+ optional ``sam_segmentation``) on a 3-class probability cube."""
    import numpy as np
    from segmenteverygrain import label_grains, labels_to_polygons, sam_segmentation
    from segmenteverygrain.interactions import polygons_to_grains

    pred = np.asarray(pred, dtype=np.float32)
    if pred.ndim != 3 or pred.shape[-1] != 3:
        raise ValueError(f"--pred must be HxWx3, got {pred.shape}")
    if pred.shape[:2] != rgb.shape[:2]:
        raise ValueError(f"--pred {pred.shape[:2]} vs image {rgb.shape[:2]}")
    labels_simple, coords = label_grains(rgb, pred, min_grain_area=int(min_area))
    if sam_model is not None and coords is not None and len(coords) > 0:
        packed = sam_segmentation(
            sam_model,
            rgb,
            pred,
            coords,
            labels_simple,
            min_area=int(min_area),
            plot_image=False,
            remove_edge_grains=False,
        )
        all_grains = packed[0]
        return polygons_to_grains(list(all_grains or []), rgb)
    polys = labels_to_polygons(labels_simple, min_area=float(min_area), remove_edge_grains=False)
    return polygons_to_grains(polys, rgb)


def main(argv: Optional[List[str]] = None) -> int:
    argv_list = list(sys.argv[1:] if argv is None else argv)
    if "--no-ui" in argv_list:
        os.environ["MPLBACKEND"] = "Agg"
    else:
        os.environ.setdefault("MPLBACKEND", "QtAgg")
    p = argparse.ArgumentParser(description="GrainPlot QC (subprocess)")
    p.add_argument("--image", required=True)
    p.add_argument("--unet", default="")
    p.add_argument("--sam", default="")
    p.add_argument("--geojson", required=True)
    p.add_argument("--mask", required=True)
    p.add_argument("--load-geojson", default="")
    p.add_argument("--pred", default="")  # HxWx3 probabilities
    p.add_argument("--labels", default="")  # HxW instance ids
    p.add_argument("--no-ui", action="store_true", help="save grains without GrainPlot window")
    args = p.parse_args(argv_list)

    geojson = Path(args.geojson)
    mask_path = Path(args.mask)
    geojson.parent.mkdir(parents=True, exist_ok=True)
    mask_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import numpy as np
        from PIL import Image
        from segmenteverygrain.interactions import polygons_to_grains, save_grains

        if not args.no_ui:
            import matplotlib

            matplotlib.use("QtAgg")
            import matplotlib.pyplot as plt  # noqa: F401

        rgb = np.asarray(Image.open(args.image).convert("RGB"))
        hw = (int(rgb.shape[0]), int(rgb.shape[1]))
        sam_model = None
        sam_pred = None
        if args.sam:
            sam_model, sam_pred = _load_sam_model(args.sam)

        load_path = Path(args.load_geojson) if args.load_geojson else None
        grains_list: List[Any] = []
        if args.pred:
            pred = np.load(args.pred)
            grains_list = _grains_from_pred(rgb, pred, sam_model)
            print(f"From --pred: {len(grains_list)} grains (label_grains"
                  f"{'+SAM2' if sam_model is not None else ''})")
        elif args.labels:
            lab = np.load(args.labels)
            grains_list = polygons_to_grains(_labels_to_polys(lab), rgb)
            print(f"From --labels: {len(grains_list)} grains")
        elif load_path is not None and load_path.exists():
            from segmenteverygrain import read_polygons

            grains_list = polygons_to_grains(read_polygons(str(load_path)), rgb)
            print(f"Loaded {len(grains_list)} grains from {load_path.name}")
        else:
            if not args.unet:
                raise ValueError("need --pred, --labels, an existing geojson, or --unet")
            align_dir = str(_HERE.parent / "align")
            if align_dir not in sys.path:
                sys.path.insert(0, align_dir)
            from seg_worker import _load_unet, _run_one  # type: ignore  # noqa: E402

            unet = _load_unet(args.unet)
            one = _run_one(
                args.image,
                unet,
                sam_pred,
                use_sam=sam_pred is not None,
                hw=hw,
            )
            polys = one.get("polygons") or _labels_to_polys(one["labels"])
            grains_list = polygons_to_grains(polys, rgb)
            print(f"SEG RGB U-Net found {len(grains_list)} grains")

        if args.no_ui:
            edited = list(grains_list)
        else:
            from segmenteverygrain.interactions import GrainPlot

            gp = GrainPlot(grains=grains_list, image=rgb, predictor=sam_pred)
            print("GrainPlot: click to add, D delete, M merge. Close the window to save.")
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

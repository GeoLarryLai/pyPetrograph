#!/usr/bin/env python3
"""
Cluster montage labeler. Subprocess only (Qt). Close the window to save.

Does not import TensorFlow or pyPetrograph.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import List, Optional


def _polys_from_mask(lab, object_ids):
    import numpy as np
    from shapely.geometry import Polygon
    from skimage.measure import find_contours, regionprops

    lab = np.asarray(lab)
    by_id = {}
    for prop in regionprops(lab):
        oid = int(prop.label)
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
                by_id[oid] = poly
            break
    return [by_id.get(int(i)) for i in object_ids]


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Cluster montage labeler")
    p.add_argument("--image", required=True)
    p.add_argument("--mask", required=True)
    p.add_argument("--object-ids", required=True)
    p.add_argument("--clusters", required=True)
    p.add_argument("--names", required=True)
    p.add_argument("--class-ids", default="")
    p.add_argument("--out", required=True)
    p.add_argument("--grain-size", type=int, default=96)
    p.add_argument("--grid-cols", type=int, default=16)
    args = p.parse_args(argv)
    try:
        import matplotlib

        matplotlib.use("QtAgg")
        import matplotlib.pyplot as plt
        import numpy as np
        from PIL import Image
        from segmenteverygrain.grain_utils import ClusterMontageLabeler, extract_all_grains

        rgb = np.asarray(Image.open(args.image).convert("RGB"))
        mask = np.asarray(Image.open(args.mask))
        if mask.ndim > 2:
            mask = mask[..., 0]
        object_ids = np.load(args.object_ids).astype(np.int32)
        clusters = np.load(args.clusters).astype(np.int32)
        names = json.loads(args.names)
        class_ids = json.loads(args.class_ids) if args.class_ids else list(range(1, len(names) + 1))
        name_to_id = {str(n): int(i) for n, i in zip(names, class_ids)}

        polys = _polys_from_mask(mask, object_ids)
        keep = [i for i, g in enumerate(polys) if g is not None]
        polys = [polys[i] for i in keep]
        object_ids = object_ids[keep]
        clusters = clusters[keep]
        if not polys:
            raise RuntimeError("no polygons in objects mask")
        grain_images, _masks, _preds = extract_all_grains(polys, rgb, target_size=int(args.grain_size))
        # ClusterMontageLabeler wants a numpy array of images
        labeler = ClusterMontageLabeler(
            cluster_labels=clusters,
            grain_images=grain_images,
            all_grains=polys,
            label_names=names,
            grid_cols=int(args.grid_cols),
            grain_size=int(args.grain_size),
            figsize=(16, 10),
        )
        labeler.activate()
        print("Montage: click a crop to name it; Shift-click names the whole group.")
        print("Number keys 1–9 pick the class. Close the window to save.")
        plt.show()
        labeled = dict(getattr(labeler, "grain_labels", {}) or {})
        labels_out = {}
        for idx, name in labeled.items():
            oid = int(object_ids[int(idx)])
            cid = name_to_id.get(str(name))
            if cid is not None:
                labels_out[str(oid)] = cid
        blob = {
            "class_names": {str(i): n for i, n in zip(class_ids, names)},
            "labels": labels_out,
        }
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(blob, indent=2), encoding="utf-8")
        print(f"Saved {len(labels_out)} names → {out.name}")
        return 0
    except Exception as exc:
        traceback.print_exc()
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

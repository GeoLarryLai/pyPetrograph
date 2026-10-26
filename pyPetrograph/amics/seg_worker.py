#!/usr/bin/env python3
"""
SEG on a BSE image. Subprocess only — never import in the Jupyter kernel and
never import ``pyPetrograph`` here (geopandas -> pyarrow crashes with TF 2.21).

Args: ``--image`` stretched RGB PNG (footer / invalid = 0), ``--dark`` .npy bool,
``--unet``, ``--sam`` (optional), ``--min-area``, ``--drop-frac``, ``--out`` geojson,
``--no-ui``, ``--load`` (an existing grains GeoJSON). Steps: ``predict_large_image``
(U-Net, + SAM 2.1 when ``--sam``) -> ``grain_filter.clean_polygons`` -> GrainPlot edit unless
``--no-ui`` -> ``save_grains``. With ``--load`` the first two steps are skipped and the saved
grains open in GrainPlot (``--dark`` / ``--unet`` not needed).
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path
from typing import List, Optional

_HERE = Path(__file__).resolve().parent  # .../pyPetrograph/amics
_OBJECTS = _HERE.parent / "objects"  # tf_env.py lives here (stdlib-only file)
for _d in (str(_HERE), str(_OBJECTS)):
    if _d not in sys.path:
        sys.path.insert(0, _d)

from tf_env import _apply_fast_seg, _load_sam_model, _load_unet, _prepare_native_libs, _read_polygons  # noqa: E402,F401

_prepare_native_libs()


def _help_text(can_add: bool) -> str:
    """Two short lines: mouse, then keys."""
    mouse = "Click grain: select / unselect"
    if can_add:
        mouse += " | Click empty area: add grain | Right click empty area: add as background"
    return mouse + "\nD: delete selected | M: merge selected | Z: undo last add | Esc: clear | hold Ctrl: hide outlines | hold H: show gaps | close window: save"


def _add_help_strip(gp, text: str, strip: float = 0.075) -> None:
    """Show ``text`` in a strip above the image (shrinks the plot a little so nothing is covered)."""
    gp.ax.set_position([0.0, 0.0, 1.0, 1.0 - strip])
    gp.fig.text(0.5, 1.0 - strip / 2, text, ha="center", va="center", fontsize=9, color="0.15", linespacing=1.4)


def segment_polys(image_path: str, rgb, dark, unet, sam_model, min_area: float, drop_frac: float) -> List:
    """SEG (U-Net, + SAM when ``sam_model``) -> dark-pixel cleanup. Shared with ``app/seg_server.py``."""
    from segmenteverygrain import predict_large_image

    from grain_filter import clean_polygons

    if dark.shape != rgb.shape[:2]:
        raise ValueError(f"--dark {dark.shape} vs image {rgb.shape[:2]}")
    use_sam = sam_model is not None
    print(f"SEG on {Path(image_path).name}: U-Net{' + SAM 2.1' if use_sam else ''} ...", flush=True)
    # predict_large_image wraps ``sam`` in its own SAM2ImagePredictor, so it needs the raw model.
    grains, _pred, _coords = predict_large_image(
        image_path,
        unet,
        sam=sam_model,
        min_area=int(min_area),
        remove_edge_grains=False,
        use_sam=use_sam,
        verbose=False,
    )
    polys = list(grains or [])
    print(f"SEG found {len(polys)} grains", flush=True)
    polys = clean_polygons(polys, dark, drop_frac=drop_frac, min_area=min_area)
    print(f"After dark-pixel cleanup: {len(polys)} grains", flush=True)
    return polys


def main(argv: Optional[List[str]] = None) -> int:
    argv_list = list(sys.argv[1:] if argv is None else argv)
    if "--no-ui" in argv_list:
        os.environ["MPLBACKEND"] = "Agg"
    else:
        os.environ.setdefault("MPLBACKEND", "QtAgg")
    p = argparse.ArgumentParser(description="SEG on BSE (subprocess)")
    p.add_argument("--image", required=True)
    p.add_argument("--dark", default="")
    p.add_argument("--unet", default="")
    p.add_argument("--load", default="", help="open these saved grains instead of running SEG")
    p.add_argument("--sam", default="")
    p.add_argument("--min-area", type=float, default=25.0)
    p.add_argument("--drop-frac", type=float, default=0.5)
    p.add_argument("--out", required=True)
    p.add_argument("--no-ui", action="store_true", help="save grains without the GrainPlot window")
    args = p.parse_args(argv_list)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        import numpy as np
        from PIL import Image
        from segmenteverygrain import predict_large_image

        if _apply_fast_seg():
            print("SEG fast path on (set PYPETROGRAPH_SEG_FAST=0 for the original code)", flush=True)
        from segmenteverygrain.interactions import polygons_to_grains, save_grains

        from grain_filter import clean_polygons

        if not args.no_ui:
            import matplotlib

            matplotlib.use("QtAgg")
            import matplotlib.pyplot as plt

        rgb = np.asarray(Image.open(args.image).convert("RGB"))
        sam_model = sam_pred = None
        if args.sam:
            sam_model, sam_pred = _load_sam_model(args.sam)
        use_sam = sam_model is not None

        if args.load:
            polys = _read_polygons(args.load)
            print(f"Loaded {len(polys)} saved grains from {Path(args.load).name}", flush=True)
        else:
            if not (args.dark and args.unet):
                raise ValueError("--dark and --unet are required unless --load is given")
            dark = np.load(args.dark).astype(bool)
            unet = _load_unet(args.unet)
            polys = segment_polys(args.image, rgb, dark, unet, sam_model, args.min_area, args.drop_frac)

        grain_objs = polygons_to_grains(polys, rgb)
        if args.no_ui:
            edited = list(grain_objs)
        else:
            from segmenteverygrain.interactions import GrainPlot

            if sam_pred is not None:
                # GrainPlot click-to-add calls predictor.predict, which needs the image set first.
                print("SAM 2.1: preparing the full image for click-to-add ...", flush=True)
                sam_pred.set_image(rgb)
            gp = GrainPlot(grains=grain_objs, image=rgb, predictor=sam_pred)
            gp.activate()  # connects the click / key handlers; without it the window ignores input
            _add_help_strip(gp, _help_text(sam_pred is not None))
            try:
                gp.fig.canvas.manager.set_window_title("AMICS: edit grains (close the window to save)")
            except Exception:
                pass
            print(_help_text(sam_pred is not None).replace("\n", " | "), flush=True)
            plt.show()
            edited = list(gp.get_grains() or [])
        save_grains(str(out), edited)
        print(f"Saved {len(edited)} grains -> {out.name}", flush=True)
        return 0
    except Exception as exc:
        traceback.print_exc()
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""
Persistent SEG model server for the desktop app (a subprocess, never imported by the GUI).

Keeps the U-Net and SAM 2.1 loaded between jobs. Never imports ``pyPetrograph``
(geopandas -> pyarrow crashes with TF 2.21); it uses the stdlib-safe worker
helpers in ``objects/tf_env.py`` and ``amics/seg_worker.py``.

Protocol: one JSON request per stdin line ``{"id", "op", ...}``; each reply is one
stdout line starting with ``@@PP@@ `` followed by JSON. Other stdout / stderr lines are
log text. Ops:

* ``ping``
* ``segment``  image (RGB PNG), dark (.npy), unet, sam ("" = U-Net only), min_area,
  drop_frac, out -> the grains GeoJSON, written exactly like ``amics/seg_worker.py --no-ui``.
  While it runs, progress lines ``{"id", "progress": {stage, done, total, unit, patch,
  patches, frac}}`` are sent (stages ``U-Net``, ``SAM``, ``merge``); see :class:`_Progress`.
* ``normalize`` src (GeoJSON), image, out -> GrainPlot-style save of edited grains
* ``set_image`` image, sam -> prepare SAM for click-to-add
* ``prompt``  points [[x, y]...], labels [bool...], box [x0, y0, x1, y1] -> coords
* ``point``    x, y -> one polygon (GrainPlot ``predict_from_prompts``) or null
* ``quit``

Only this file touches the model runtime: an ONNX backend can replace ``_Models``
without changing the GUI side (``app/backend.py``).
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PKG = _HERE.parent
for _d in (str(_PKG / "amics"), str(_PKG / "objects")):
    if _d not in sys.path:
        sys.path.insert(0, _d)

from tf_env import _apply_fast_seg, _load_sam_model, _load_unet, _prepare_native_libs, _read_polygons  # noqa: E402

_prepare_native_libs()

_TAG = "@@PP@@ "
_OUT = sys.stdout


def _reply(obj) -> None:
    _OUT.write(_TAG + json.dumps(obj) + "\n")
    _OUT.flush()


class _Models:
    """Model cache keyed by weight path (the swap point for another runtime)."""

    def __init__(self) -> None:
        self._unet = {}
        self._sam = {}
        self._image_key = None

    def unet(self, path: str):
        if path not in self._unet:
            print(f"loading U-Net {Path(path).name}", flush=True)
            self._unet[path] = _load_unet(path)
        return self._unet[path]

    def sam(self, path: str):
        """(raw model, predictor) or (None, None)."""
        if not path:
            return None, None
        if path not in self._sam:
            print(f"loading SAM 2.1 {Path(path).name}", flush=True)
            self._sam[path] = _load_sam_model(path)
        return self._sam[path]


def _rgb(path: str):
    import numpy as np
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"))


class _Progress:
    """Progress of one ``segment`` request, read from the segmenteverygrain loops.

    segmenteverygrain itself is not changed: while the request runs, the module-level
    ``predict_image`` / ``sam_segmentation`` / ``one_point_prompt`` that ``predict_large_image``
    looks up are wrapped (and restored afterwards). The wrappers only count calls and pass
    every argument and result through unchanged, so the grains are byte-identical.
    U-Net progress is counted per model batch (tiles of 256 px), SAM per point prompt."""

    def __init__(self, rid, shape, use_sam: bool) -> None:
        import inspect
        import time

        from segmenteverygrain import segmenteverygrain as sm

        self.sm, self.rid, self.use_sam, self.time = sm, rid, use_sam, time
        sig = inspect.signature(sm.predict_large_image).parameters
        ps = int(sig["patch_size"].default) if "patch_size" in sig else 2000
        ov = int(sig["overlap"].default) if "overlap" in sig else 300
        h, w = int(shape[0]), int(shape[1])
        step = ps - ov
        self.patch_tiles = []
        for i in range(0, max(1, h - ps + step + 1), step):
            for j in range(0, max(1, w - ps + step + 1), step):
                self.patch_tiles.append(self.tiles(min(i + ps, h) - i, min(j + ps, w) - j))
        self.total_tiles = sum(self.patch_tiles)
        self.patches = len(self.patch_tiles)
        self.patch = 0          # patches started
        self.tiles_done = 0
        self.prompts_done = self.prompts_total = 0
        self._last = 0.0
        self._saved = {}

    @staticmethod
    def tiles(h: int, w: int, I: int = 256) -> int:
        from fast_seg import tile_count

        return tile_count(h, w, I)

    def frac(self) -> float:
        P = max(1, self.patches)
        if not self.use_sam:
            return min(1.0, self.tiles_done / max(1, self.total_tiles))
        k = max(0, self.patch - 1)
        pt = self.patch_tiles[k] if k < len(self.patch_tiles) else 1
        before = sum(self.patch_tiles[:k])
        u = min(1.0, (self.tiles_done - before) / max(1, pt))
        sm_ = self.prompts_done / self.prompts_total if self.prompts_total else 0.0
        return min(1.0, (k + 0.25 * u + 0.75 * sm_) / P)

    def send(self, stage: str, force: bool = False) -> None:
        now = self.time.monotonic()
        if not force and now - self._last < 0.2:
            return
        self._last = now
        if stage == "SAM":
            done, total, unit = self.prompts_done, self.prompts_total, "prompts"
        elif stage == "U-Net":
            done, total, unit = self.tiles_done, self.total_tiles, "tiles"
        else:
            done, total, unit = 0, 0, ""
        _reply({"id": self.rid, "progress": {
            "stage": stage, "done": int(done), "total": int(total), "unit": unit,
            "patch": int(max(1, self.patch)), "patches": int(self.patches),
            "frac": 1.0 if stage == "merge" else round(self.frac(), 4)}})

    def __enter__(self):
        sm, me = self.sm, self
        self._saved = {k: getattr(sm, k) for k in ("predict_image", "sam_segmentation", "one_point_prompt")}
        pi, ss, opp = (self._saved[k] for k in ("predict_image", "sam_segmentation", "one_point_prompt"))

        class _Counted:  # model proxy: counts tiles per predict() batch
            def __init__(self, model):
                self._m = model

            def predict(self, x, *a, **k):
                out = self._m.predict(x, *a, **k)
                me.tiles_done += int(len(x))
                me.send("U-Net")
                return out

            def __getattr__(self, name):
                return getattr(self._m, name)

        def predict_image(image, model, I):
            me.patch += 1
            me.prompts_done = me.prompts_total = 0
            me.send("U-Net", force=True)
            return pi(image, _Counted(model), I)

        def sam_segmentation(sam, image, image_pred, coords, *a, **k):
            me.prompts_done, me.prompts_total = 0, int(len(coords))
            me.send("SAM", force=True)
            return ss(sam, image, image_pred, coords, *a, **k)

        def one_point_prompt(*a, **k):
            out = opp(*a, **k)
            me.prompts_done += 1
            me.send("SAM")
            return out

        sm.predict_image, sm.sam_segmentation, sm.one_point_prompt = predict_image, sam_segmentation, one_point_prompt
        return self

    def __exit__(self, *exc):
        for k, v in self._saved.items():
            setattr(self.sm, k, v)
        return False


def main() -> int:
    import numpy as np

    sys.stdout = sys.stderr  # stray prints (TF, SEG progress) go to the log stream
    fast = _apply_fast_seg()
    print(f"seg_server ready (fast SEG {'on' if fast else 'off'})", flush=True)
    models = _Models()
    sam_pred = None
    from seg_worker import segment_polys
    from segmenteverygrain.interactions import polygons_to_grains, predict_from_prompts, save_grains

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        rid, op = req.get("id"), req.get("op")
        try:
            if op == "ping":
                _reply({"id": rid, "ok": True})
            elif op == "quit":
                _reply({"id": rid, "ok": True})
                return 0
            elif op == "segment":
                rgb = _rgb(req["image"])
                dark = np.load(req["dark"]).astype(bool)
                unet = models.unet(req["unet"])
                sam_model, _ = models.sam(req.get("sam") or "")
                with _Progress(rid, rgb.shape, sam_model is not None) as prog:
                    polys = segment_polys(
                        req["image"], rgb, dark, unet, sam_model, float(req["min_area"]), float(req["drop_frac"])
                    )
                prog.send("merge", force=True)  # non-grain cleanup + saving the grains
                out = Path(req["out"])
                out.parent.mkdir(parents=True, exist_ok=True)
                save_grains(str(out), list(polygons_to_grains(polys, rgb)))
                _reply({"id": rid, "ok": True, "out": str(out)})
            elif op == "normalize":
                rgb = _rgb(req["image"])
                polys = _read_polygons(req["src"])
                save_grains(str(req["out"]), list(polygons_to_grains(polys, rgb)))
                _reply({"id": rid, "ok": True, "out": req["out"]})
            elif op == "set_image":
                _, sam_pred = models.sam(req.get("sam") or "")
                if sam_pred is None:
                    raise RuntimeError("SAM 2.1 weights not given")
                key = (req["image"], req.get("sam"))
                if models._image_key != key:
                    sam_pred.set_image(_rgb(req["image"]))
                    models._image_key = key
                _reply({"id": rid, "ok": True})
            elif op in ("point", "prompt"):
                if sam_pred is None:
                    raise RuntimeError("call set_image first")
                if op == "point":
                    pts, labels, box = [(float(req["x"]), float(req["y"]))], [True], None
                else:  # SAM's normal prompts: points (+/-) and / or one box
                    pts = [(float(a), float(b)) for a, b in req.get("points") or []]
                    labels = [bool(v) for v in req.get("labels") or []]
                    box = [float(v) for v in req["box"]] if req.get("box") else None
                    if len(pts) != len(labels):
                        raise ValueError("points and labels differ in length")
                    if not pts and box is None:
                        raise ValueError("no prompt given")
                sx, sy = predict_from_prompts(
                    sam_pred, box=box, points=pts or None, point_labels=labels or None
                )
                coords = None if sx is None else [[float(a), float(b)] for a, b in zip(sx, sy)]
                _reply({"id": rid, "ok": True, "coords": coords})
            else:
                raise ValueError(f"unknown op {op!r}")
        except Exception as exc:  # report and keep serving
            traceback.print_exc()
            _reply({"id": rid, "ok": False, "error": f"{type(exc).__name__}: {exc}"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

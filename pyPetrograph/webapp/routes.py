# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""FastAPI routes of the browser app. Thin: the step logic is in ``state.py``, the science in
``pyPetrograph.app.amics_pipeline`` / ``pyPetrograph.amics`` (the same calls as the former Qt app)."""
from __future__ import annotations

import time
import io
import threading
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from pyPetrograph.webapp.shell import JobRunner, mount_shell
from pyPetrograph.webapp.state import DOWNSTREAM_KEYS, MINERAL_FIGURES, MISSING_FIG, SETTINGS_FIELDS, ORDER, TITLES, UserError, WebState, short_count

STATIC = Path(__file__).parent / "static"
NON_GRAIN_RGBA = (26, 102, 255, 140)   # former Qt app / notebook blue tint
PREVIEW_MAX = 1600                      # longest side of the zoomed-out preview
IMG_EXT = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")


def _u8(a) -> np.ndarray:
    a = np.ascontiguousarray(a)
    if a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8)
    return a


def png_bytes(arr, max_side: Optional[int] = None, nearest: bool = False) -> bytes:
    from PIL import Image

    im = Image.fromarray(_u8(arr))
    if max_side and max(im.size) > max_side:
        k = max_side / max(im.size)
        im = im.resize((max(1, round(im.size[0] * k)), max(1, round(im.size[1] * k))),
                       Image.NEAREST if nearest else Image.BOX)
    b = io.BytesIO()
    im.save(b, format="PNG", compress_level=1)
    return b.getvalue()


def fig_array(fig) -> np.ndarray:
    import matplotlib.pyplot as plt

    fig.canvas.draw()
    a = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return a


def read_rgb(path) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"))


def create_app(state: Optional[WebState] = None, backend=None, testing: bool = False):
    from fastapi import Body, FastAPI, HTTPException, Query
    from fastapi.responses import FileResponse, JSONResponse, Response
    from fastapi.staticfiles import StaticFiles

    from pyPetrograph.app.backend import SEG_INSTALL_URL, SegBackend

    app = FastAPI(title="pyPetrograph")
    app.state.testing = testing
    st = state or WebState()
    runner = JobRunner(error_types=(UserError,))
    lock = threading.RLock()        # one state change at a time (jobs and clicks)
    app.state.st, app.state.runner = st, runner
    _backend = {"b": backend}

    def be():
        if _backend["b"] is None:
            _backend["b"] = SegBackend(log=st.log)
        return _backend["b"]

    app.state.backend = be
    mount_shell(app, runner)
    app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")
    S = st.session
    imgcache: Dict[Any, bytes] = {}

    def need(cond, msg="open the earlier steps first"):
        if not cond:
            raise HTTPException(409, msg)

    def job(title, fn):
        def body(j):
            with lock:
                return fn(j)
        jb = runner.submit(title, body)
        t0 = time.monotonic()  # short jobs (exports, small recomputes) answer in this reply: no poll round trip
        while jb.state == "running" and time.monotonic() - t0 < 0.15:
            time.sleep(0.004)
        return {"job": jb.as_dict()}

    def err(fn):
        try:
            with lock:
                return fn()
        except UserError as e:
            raise HTTPException(400, str(e))

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    # -- shared ------------------------------------------------------------------------------
    @app.get("/api/state")
    def state_():
        st.availability()
        msg, st.status = st.status, ""
        return {"steps": [{"name": n, "title": t, "done": n in st.done, "ready": st.ready(i)}
                          for i, (n, t) in enumerate(zip(ORDER, TITLES))],
                "sample": S.sample, "status": msg, "log": st.logs[-80:], "avail": st.avail,
                "seg_url": SEG_INSTALL_URL, "version": st.version,
                "folder": str(S.P["folder"]) if S.P else None}

    @app.get("/api/img/{key}")
    def img(key: str, max: Optional[int] = Query(None), v: str = ""):
        """Image layers for the viewer. ``max`` = longest side (preview); full size without it."""
        ck = (key, v, max)
        if ck in imgcache and not key.startswith(("legend_plot", "mask")):
            return Response(imgcache[ck], media_type="image/png")
        with lock:
            arr, nearest = _layer(key)
        if arr is None:
            raise HTTPException(404, f"no image {key}")
        b = png_bytes(arr, max, nearest)
        if len(imgcache) > 40:
            imgcache.clear()
        imgcache[ck] = b
        return Response(b, media_type="image/png", headers={"Cache-Control": "no-store"})

    @app.get("/api/imgsize/{key}")
    def imgsize(key: str):
        with lock:
            arr, _ = _layer(key)
        if arr is None:
            raise HTTPException(404, f"no image {key}")
        return {"w": int(arr.shape[1]), "h": int(arr.shape[0])}

    def _layer(key):
        if key == "bse_gray":
            if S.bse_gray is None:
                from pyPetrograph.amics import load_bse

                p = st.preview_bse
                return (load_bse(p)[0] if p and Path(p).is_file() else None), False
            return S.bse_gray, False
        if key == "bse_stretch":
            return S.bse_stretch, False
        if key == "legend_img":
            return (read_rgb(S.legend_file) if S.legend_file else None), False
        if key == "mineral_img":
            return (read_rgb(S.mineral_file) if S.mineral_file else None), False
        if key == "legend_plot":
            fig = st.legend_plot()
            return (fig_array(fig) if fig is not None else None), False
        if key == "mask":
            m, _ = st.ng_mask()
            rgba = np.zeros(m.shape + (4,), np.uint8)
            rgba[m] = NON_GRAIN_RGBA
            return rgba, True
        if key == "knoll":
            k = S.knoll
            return (None if k is None else (k if k.ndim == 2 else k[..., :3])), False
        if key == "fig:" + MISSING_FIG:
            raw = getattr(S, "_raw_map", None)
            if raw is None or not len(raw[2]):
                return None, False
            from pyPetrograph.amics import plot_missing_colors

            return fig_array(plot_missing_colors(raw[2])), False
        if key.startswith("fig:"):
            p = _figure(key[4:])
            return (read_rgb(p) if p else None), False
        if key.startswith("export:"):
            p = Path(S.P.get("folder", ".")) / Path(key[7:]).name
            return (read_rgb(p) if p.suffix.lower() in (".png", ".jpg") and p.exists() else None), False
        return None, False

    def _figure(name):
        if not S.C:
            return None
        key = dict(MINERAL_FIGURES).get(name)
        p = S.C.get(key) if key else S.gsd_figures().get(name)
        return p if p is not None and Path(p).exists() else None

    @app.get("/api/pixel")
    def pixel(src: str, x: float, y: float):
        """Eyedropper: the colour of one pixel of the legend image or the mineral map."""
        arr, _ = _layer(src if src in ("legend_img", "mineral_img") else "mineral_img")
        need(arr is not None, "no image")
        h, w = arr.shape[:2]
        xi, yi = int(x), int(y)
        if not (0 <= xi < w and 0 <= yi < h):
            raise HTTPException(400, "outside the image")
        rgb = [int(c) for c in np.atleast_1d(arr[yi, xi])[:3]]
        return {"rgb": rgb, "x": xi, "y": yi}

    # -- 1 files -----------------------------------------------------------------------------
    @app.get("/api/files")
    def files():
        return st.last_files()

    @app.post("/api/files/complete")
    def files_complete(body: dict = Body(...)):
        return st.complete_files(body["key"], body.get("cur") or {})

    @app.get("/api/ls")
    def ls(path: str = "", dirs_only: bool = False):
        p = Path(path).expanduser() if path else Path.home()
        if p.is_file():
            p = p.parent
        if not p.is_dir():
            p = Path.home()
        out = []
        try:
            for c in sorted(p.iterdir(), key=lambda c: (not c.is_dir(), c.name.lower())):
                if c.name.startswith("."):
                    continue
                if c.is_dir():
                    out.append({"name": c.name, "dir": True})
                elif not dirs_only and c.suffix.lower() in IMG_EXT:
                    out.append({"name": c.name, "dir": False})
        except PermissionError:
            pass
        return {"path": str(p.resolve()), "parent": str(p.resolve().parent), "items": out}

    @app.post("/api/files/preview")
    def files_preview(body: dict = Body(...)):
        st.preview_bse = body.get("bse") or ""
        imgcache.clear()
        return {"ok": Path(st.preview_bse).is_file()}

    @app.post("/api/open")
    def open_(body: dict = Body(...)):
        def f():
            imgcache.clear()
            name = st.open(body.get("legend", ""), body.get("bse", ""), body.get("mineral", ""),
                           body.get("out", ""), body.get("sample", ""))
            return {"sample": name}
        return err(f)

    # -- 2 legend ----------------------------------------------------------------------------
    def legend_view():
        msg, kind = st.legend_state()
        return {"rows": st.legend_rows(), "report": "\n".join(S.legend_report[-6:]), "state": [msg, kind],
                "can_undo": bool(st.legend_hist), "need_read": st.legend_read_for != S.legend_file,
                "saved": "legend" in S.saved()}

    @app.get("/api/legend")
    def legend_get():
        need("open" in st.done)
        with lock:
            return legend_view()

    @app.post("/api/legend/read")
    def legend_read():
        need("open" in st.done)
        return job("Reading legend (OCR)", lambda j: (j.progress(0, 1, "OCR"), st.read_legend(), None)[-1])

    @app.post("/api/legend/op")
    def legend_op(body: dict = Body(...)):
        need("open" in st.done)
        body = dict(body)
        op = body.pop("op")
        return err(lambda: (st.legend_op(op, **body), legend_view())[1])

    @app.post("/api/legend/check")
    def legend_check(body: dict = Body(...)):
        return {"problem": st.entry_problem(body["rgb"], body.get("name", ""), body.get("row"))}

    @app.post("/api/legend/apply")
    def legend_apply():
        need("open" in st.done)
        return job("Loading BSE", lambda j: (st.apply_legend(j), imgcache.clear())[0])

    # -- 3 footer ----------------------------------------------------------------------------
    @app.get("/api/footer")
    def footer_get():
        need(S.bse_gray is not None)
        h, w = S.bse_gray.shape[:2]
        return {"footer": S.footer, "detected": S.detected_footer(), "h": h, "w": w, "sample": S.sample,
                "done": "footer" in st.done}

    @app.post("/api/footer/apply")
    def footer_apply(body: dict = Body(...)):
        need(S.bse_gray is not None)
        return err(lambda: (st.apply_footer(int(body["row"])), imgcache.clear(), {"footer": S.footer})[2])

    # -- 4 non-grain -------------------------------------------------------------------------
    def ng_view():
        _, info = st.ng_mask()
        msg, kind = st.ng_state()
        h, w = S.bse_gray.shape[:2]
        return {"points": st.points, "typed": st.thr_typed, "info": info, "state": [msg, kind],
                "footer": S.footer, "h": h, "w": w, "saved": "threshold" in S.saved()}

    @app.get("/api/nongrain")
    def ng_get(activate: bool = False):
        need(S.bse_stretch is not None)
        with lock:
            if activate:
                st.ng_load_saved()
            return ng_view()

    @app.post("/api/nongrain/op")
    def ng_op(body: dict = Body(...)):
        need(S.bse_stretch is not None)
        op = body.get("op")

        def f():
            if op == "click":
                st.ng_click(float(body["x"]), float(body["y"]), int(body.get("button", 0)), float(body.get("scale", 1)))
            elif op == "undo":
                if st.points:
                    st.points.pop()
            elif op == "reset":
                st.points, st.thr_typed = [], None
            elif op == "typed":
                v = body.get("value")
                st.thr_typed = None if v in (None, "") or float(v) < 0 else float(v)
            elif op == "load_saved":
                st.ng_load_saved(force=True)
            elif op == "clear_saved":
                st.ng_clear_saved()
            elif op == "apply":
                st.ng_apply()
            else:
                raise UserError(f"unknown op {op}")
            return ng_view()
        return err(f)

    # -- 5 scale -----------------------------------------------------------------------------
    def sc_view():
        msg, kind = st.sc_state()
        h, w = S.bse_gray.shape[:2]
        from pyPetrograph.amics import UNIT_TO_UM

        return {"pts": st.pts, "length": st.length, "unit": st.unit, "um": st.um_text, "saved": st.saved_scale,
                "info": st.sc_info(), "state": [msg, kind], "footer": S.footer, "h": h, "w": w,
                "units": [u for u in ("µm", "mm", "nm") if u in UNIT_TO_UM], "done": "scale" in st.done}

    @app.get("/api/scale")
    def sc_get(activate: bool = False):
        need(S.bse_gray is not None)
        with lock:
            if activate:
                st.sc_activate()
            return sc_view()

    @app.post("/api/scale/op")
    def sc_op(body: dict = Body(...)):
        need(S.bse_gray is not None)
        op = body.get("op")

        def f():
            if op == "click":
                st.sc_click(float(body["x"]), float(body["y"]), int(body.get("button", 0)), float(body.get("scale", 1)))
            elif op == "set":
                if "length" in body:
                    st.length = float(body["length"] or 0)
                if "unit" in body:
                    st.unit = str(body["unit"])
                if "um" in body:
                    st.um_text = str(body["um"])
            elif op == "undo":
                if st.pts:
                    st.pts.pop()
            elif op == "reset":
                st.pts, st.um_text = [], ""
            elif op == "clear_saved":
                st.sc_clear_saved()
            elif op == "apply":
                st.sc_apply()
            elif op == "skip":
                st.sc_skip()
            else:
                raise UserError(f"unknown op {op}")
            return sc_view()
        return err(f)

    @app.post("/api/scale/ocr")
    def sc_ocr():
        need(S.bse_gray is not None)

        def f(j):
            v = S.footer_ocr()
            cur = st.sc_typed() or (st.sc_drawn() or {}).get("um_per_px") or (st.saved_scale or {}).get("um_per_px")
            txt = f"footer text says {v} µm/px" if v else "footer text: not read"
            if v and cur:
                txt += f" (line: {cur:.4f}, {100 * (cur / v - 1):+.1f}%)"
            return {"text": txt}
        return job("Reading footer", f)

    # -- 6 grains ----------------------------------------------------------------------------
    def seg_view():
        av = st.availability()
        msg, kind = st.seg_state()
        s = S.s
        note, st.status = st.status, ""
        return {"status": note, "n": len(st.polys), "sel": sorted(st.sel), "edits": len(st.ghist), "edited": st.edited,
                "poly_version": st.poly_version, "state": [msg, kind], "avail": av,
                "settings": {"use_sam2": bool(s.use_sam2) and bool(av.get("sam2")), "min_px": int(s.min_grain_px),
                             "drop": float(s.dark_drop_frac)},
                "has_saved": bool(S.P) and S.saved_path("grains") is not None,
                "prompts": [list(p) for p in st.prompt_pts], "box": st.prompt_box,
                "preview": None if st.preview is None else [list(c) for c in st.preview.exterior.coords],
                "sam_ready": st.sam_ready, "done": "segment" in st.done,
                "use_ok": bool(st.polys) and not (st.edited and not av.get("seg"))}

    @app.get("/api/segment")
    def seg_get(activate: bool = False):
        need("nongrain" in st.done)
        with lock:
            if activate:
                st.seg_activate()
            return seg_view()

    @app.get("/api/segment/polys")
    def seg_polys(v: str = ""):
        """Grain outlines as flat [x0, y0, x1, y1, ...] rings (1 decimal), index = grain number."""
        with lock:
            out = []
            for p in st.polys:
                g = p if p.geom_type == "Polygon" else max(getattr(p, "geoms", [p]), key=lambda q: q.area)
                a = np.round(np.asarray(g.exterior.coords, float), 1).ravel()
                out.append(a.tolist())
            return JSONResponse({"v": st.poly_version, "polys": out})

    @app.post("/api/segment/settings")
    def seg_settings(body: dict = Body(...)):
        return err(lambda: (st.sync_settings(body.get("use_sam2"), body.get("min_px"), body.get("drop")), seg_view())[1])

    @app.post("/api/segment/run")
    def seg_run(body: dict = Body(default={})):
        need("nongrain" in st.done)
        if not st.availability().get("seg"):
            raise HTTPException(400, "SegmentEveryGrain is not installed: Run segmentation is off")
        st.sync_settings(body.get("use_sam2"), body.get("min_px"), body.get("drop"))
        st.sam_ready = False

        def f(j):
            def prog(d):
                msg = d.get("stage", "")
                if int(d.get("patches") or 0) > 1:
                    msg += f" · patch {d.get('patch')}/{d.get('patches')}"
                if d.get("stage") == "merge":
                    j.progress(int(1000 * float(d.get("frac") or 0)), 1000, "merge: non-grain cleanup and saving")
                else:
                    j.progress(int(d.get("done") or 0), int(d.get("total") or 1), f"{msg} {d.get('unit', '')}".strip())
            polys = S.segment(be(), progress=prog)
            st.got_grains(polys, "segmented")
        return job("Segmenting (models stay loaded for the next run)", f)

    @app.post("/api/segment/load")
    def seg_load():
        need("nongrain" in st.done)
        if not S.P or S.saved_path("grains") is None:
            raise HTTPException(400, "no saved grains for this sample")
        return job("Loading grains", lambda j: st.got_grains(S.load_saved_grains(), "saved"))

    @app.post("/api/segment/op")
    def seg_op(body: dict = Body(...)):
        op = body.get("op")

        def f():
            if op == "click":
                st.grain_click(float(body["x"]), float(body["y"]))
            elif op == "add_point":
                st.prompt_pts.append((float(body["x"]), float(body["y"]), bool(body.get("positive", True))))
            elif op == "box":
                st.prompt_box = tuple(float(c) for c in body["box"])
            elif op == "accept":
                st.accept_prompt()
            elif op == "clear_prompts":
                st.clear_prompts()
            elif op == "clear_saved":
                st.seg_clear_saved()
            else:
                st.grain_op(op)
            return seg_view()
        return err(f)

    @app.post("/api/segment/sam_prepare")
    def sam_prepare():
        if not st.availability().get("sam2"):
            raise HTTPException(400, "SAM 2.1 (sam2 + torch) is not installed")

        def f(j):
            if not st.sam_ready:
                w = S.weights(sam2=True)
                be().request("set_image", image=str(S.seg_image_png()), sam=str(w["sam"]))
                st.sam_ready = True
            return {"ok": True}
        return job("Preparing SAM 2.1 for click-to-add", f)

    @app.post("/api/segment/prompt")
    def sam_prompt():
        if not st.sam_ready or (not st.prompt_pts and not st.prompt_box):
            return {"job": None, **seg_view()}

        def f(j):
            from shapely.geometry import Polygon

            msg = be().request("prompt", **st.prompt_request())
            coords = msg.get("coords")
            st.preview = Polygon(coords) if coords and len(coords) >= 3 else None
            if st.preview is None:
                st.status = "SAM found no grain for these prompts"
            return {"ok": True}
        return job("SAM", f)

    @app.post("/api/segment/use")
    def seg_use(body: dict = Body(default={})):
        need("nongrain" in st.done)
        st.sync_settings(body.get("use_sam2"), body.get("min_px"), body.get("drop"))
        return job("Measuring grains, voting minerals, drawing figures", lambda j: st.use_grain_map(be(), j))

    # -- Advanced settings (notebook 4 Settings cell) -----------------------------------------
    @app.get("/api/settings")
    def settings_get():
        return {"fields": [dict(f, value=_sval(f["key"])) for f in SETTINGS_FIELDS], "computed": "segment" in st.done}

    def _sval(k):
        v = getattr(S.s, k)
        return ", ".join(v) if isinstance(v, list) else v

    @app.post("/api/settings")
    def settings_set(body: dict = Body(...)):
        changed = []
        with lock:
            for f in SETTINGS_FIELDS:
                k = f["key"]
                if k not in body:
                    continue
                raw = body[k]
                try:
                    if f["type"] == "names":
                        v = [x.strip() for x in str(raw).split(",") if x.strip()]
                    elif f["type"] == "bool":
                        v = bool(raw)
                    elif f["type"] == "int?":
                        v = None if raw in (None, "") else int(float(raw))
                    elif f["type"] == "int":
                        v = int(float(raw))
                    else:
                        v = float(raw)
                except (TypeError, ValueError):
                    raise HTTPException(400, f"{f['label']}: not a number")
                if v != getattr(S.s, k):
                    setattr(S.s, k, v)
                    changed.append(k)
        redo = "segment" in st.done and any(k in DOWNSTREAM_KEYS for k in changed)

        def f(j):
            if redo:
                st.recount(dict(S.extra_legend), j)
                imgcache.clear()
            return {"changed": changed, "recomputed": redo}
        return job("Applying settings" + (": colours, votes, figures again" if redo else ""), f)

    # -- 7 minerals --------------------------------------------------------------------------
    @app.get("/api/minerals")
    def minerals_get():
        need("segment" in st.done, "Not computed yet: press 'Use this grain map' in step 6")
        with lock:
            rows = S.mineral_table()
            for r in rows:
                r["pixels_text"] = f"{short_count(r['pixels'])} ({r['pct']:.1f}%)" if r["pixels"] else "0"
            n = {k: sum(r["status"] == k for r in rows) for k in ("ok", "reassigned", "missing", "unlisted")}
            mapped = len(S.mapped) if S.mapped is not None else 0
            unm = int((S.grains["mineral"] == "Unmapped").sum()) if S.grains is not None and "mineral" in S.grains else 0
            raw = getattr(S, "_raw_map", None)
            figs = [n_ for n_, _ in MINERAL_FIGURES] + ([MISSING_FIG] if raw is not None and len(raw[2]) else [])
            matches = []
            cm = getattr(S, "color_matches", None)
            if cm is not None:
                for _, m in cm.iterrows():
                    how = {"free name": "unused legend name", "shared name": "nearest mineral (name also has its exact colour)"}.get(m.status, "left unnamed: counts as Unknown")
                    matches.append({"rgb": [int(m.map_red), int(m.map_green), int(m.map_blue)], "pixels": int(m.pixels),
                                    "assigned": str(m.assigned) if m.status in ("free name", "shared name") else "",
                                    "distance": float(m.distance) if m.distance == m.distance else None, "how": how})
            return {"rows": rows, "figures": figs, "matches": matches, "auto_match": bool(S.s.auto_match_colors), "version": st.version,
                    "summary": f"{mapped} grains with a mineral, {unm} unmapped (left out). Added colours: {len(S.extra_legend)}.",
                    "state": [f"\u2713 {n['ok']}   \u25B2 {n['reassigned']}   \u2717 {n['missing'] + n['unlisted']}  · computed, not exported yet", "saved"]}

    @app.post("/api/minerals/op")
    def minerals_op(body: dict = Body(...)):
        need("segment" in st.done)
        op = body.get("op")
        if op == "suggest":
            return {"rgb": list(st.suggest_rgb(body.get("row")))}
        if op == "add":
            from pyPetrograph.webapp.state import parse_rgb

            rgb = parse_rgb(body.get("rgb", ""))
            name = str(body.get("name", "")).strip()
            if not name or rgb is None:
                raise HTTPException(400, "need a mineral name and a colour (R, G, B 0–255 or #rrggbb)")
            extra = {**S.extra_legend, name: rgb}
        elif op == "remove":
            extra = {k: v for k, v in S.extra_legend.items() if k != body.get("mineral")}
        elif op == "clear_saved":
            S.clear_saved("extra")
            extra = {}
        else:
            raise HTTPException(400, f"unknown op {op}")
        return job("Counting minerals again", lambda j: st.recount(extra, j))

    # -- 8 grain size ------------------------------------------------------------------------
    @app.get("/api/gsd")
    def gsd_get():
        need("segment" in st.done, "Not computed yet: press 'Use this grain map' in step 6")
        g = S.grains
        figs = list(S.gsd_figures())
        big = int((g["area_px"] >= S.s.min_grain_px).sum())
        unit = "µm" if S.size_unit == "um" else "px"
        sel = g.loc[g["area_px"] >= S.s.min_grain_px]
        med = {n: float(sel[f"{n}_{S.size_unit}"].median()) for n in ("a", "b", "D", "ECD") if f"{n}_{S.size_unit}" in sel and len(sel)}
        return {"figures": figs, "version": st.version, "medians": med, "unit": unit,
                "summary": f"{len(g)} grains measured; {big} with area ≥ {S.s.min_grain_px:g} px in the curves "
                           f"(minerals with ≥ {S.s.min_grains_curve} grains get their own curve). "
                           f"Sizes in {unit}" + (f" ({S.um_per_px:.4f} µm/px)." if S.um_per_px else " (no scale)."),
                "state": [f"{len(figs)} curves ready · not exported yet", "saved"]}

    @app.post("/api/gsd/replot")
    def gsd_replot():
        need("segment" in st.done)

        def f(j):
            S.replot_gsd()
            st.version += 1
        return job("Replotting grain size curves", f)

    # -- 9 outputs ---------------------------------------------------------------------------
    @app.get("/api/outputs")
    def outputs_get():
        need("segment" in st.done, "Not computed yet: press 'Use this grain map' in step 6")
        C = S.C
        d, f = st.exported["data"], st.exported["figures"]
        return {"folder": str(S.P["folder"]),
                "n_csv": len(list(C["folder"].glob("*.csv"))), "n_png": len(list(C["folder"].glob("*.png"))),
                "files": [Path(p).name for p in d + f], "version": st.version,
                "state": ([f"Exported {len(d)} data files and {len(f)} figures to {S.P['folder'].name}/", "saved"]
                          if (d or f) else ["Nothing exported yet", "none"])}

    def _export(kind, fn):
        def f(j):
            files = fn()
            st.exported[kind] = [str(p) for p in files]
            st.done.add("outputs")
            return {"files": [Path(p).name for p in files]}
        return f

    @app.post("/api/outputs/export_data")
    def export_data():
        need("segment" in st.done)
        return job("Exporting CSV data sheets", _export("data", S.export_data))

    @app.post("/api/outputs/export_figures")
    def export_figures():
        need("segment" in st.done)
        return job("Exporting figures", _export("figures", S.export_figures))

    @app.post("/api/outputs/replot")
    def knoll_replot():
        need("segment" in st.done)

        def f(j):
            S.replot_knolling()
            st.version += 1
            imgcache.clear()
        return job("Replotting the knolling diagram", f)

    @app.post("/api/outputs/folder")
    def outputs_folder():
        """The export folder, created if needed (as the former Qt app), then opened."""
        need(bool(S.P), "open a sample first")
        d = Path(S.P["folder"])
        d.mkdir(parents=True, exist_ok=True)
        if not testing:
            from pyPetrograph.webapp.shell.server import open_folder

            open_folder(d)
        return {"opened": str(d)}

    @app.on_event("shutdown")
    def _close():
        if _backend["b"] is not None:
            _backend["b"].close()

    from .optical import mount_optical
    mount_optical(app, job, err, lock)
    st.preview_bse = ""
    return app

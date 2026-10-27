# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""optical.py - the optical (PPL / CPL photo) job of the web app: notebook 1 step by step.

Photos -> Classes -> Mark examples -> Features -> Train -> Predict -> Porosity %.
All the work is done by the package (``Session``, ``run_feature_preview``, ``run_train_cell``,
``print_train_metrics``, ``run_predict_cell``, ``porosity_summary``), so the files written are
the same as the notebook's: labels in ``<output>/labels/`` (shared with notebook 1 and the Qt
labeller), the model in ``<output>/models/universal_model.joblib``, predictions in
``<output>/predictions/``.
"""
from __future__ import annotations

import contextlib
import io
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

ORDER = ["photos", "classes", "label", "features", "train", "predict", "porosity"]
TITLES = ["1 · Photos", "2 · Classes", "3 · Mark examples", "4 · Features", "5 · Train",
          "6 · Predict", "7 · Porosity %"]
DEFAULT_CLASSES = {1: "pores", 2: "grains"}
FEATURES = [("r", "Red"), ("g", "Green"), ("b", "Blue"), ("y", "Brightness (Y)"),
            ("local_std", "Local texture (std)"), ("local_grad", "Edges (gradient)"),
            ("lbp", "LBP texture"), ("entropy", "Entropy")]
NB1_FEATURES = {"r", "g", "b", "local_std", "local_grad"}


class OpticalState:
    def __init__(self, log=None):
        self.S = None                      # pyPetrograph Session
        self.out: Optional[Path] = None
        self.done: set = set()
        self.version = 0
        self.status = ""
        self.logs: List[str] = []
        self.train_text = ""
        self.train_acc = None
        self.pred: List[Dict[str, Any]] = []
        self.features_on = {k: (k in NB1_FEATURES) for k, _ in FEATURES}
        self.wand_threshold = 70.0

    # -- helpers --------------------------------------------------------------------------
    def log(self, text: str) -> None:
        for line in str(text).splitlines():
            if line.strip():
                self.logs.append(line)
        self.logs = self.logs[-400:]

    @contextlib.contextmanager
    def ctx(self):
        """Run package code with this job's output folder, and keep what it prints for the log."""
        from pyPetrograph.common.paths import get_output_dir, set_output_dir
        import matplotlib.pyplot as plt

        prev = get_output_dir()
        set_output_dir(self.out)
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf), warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)   # fig.show() under Agg
                yield buf
        finally:
            set_output_dir(prev)
            plt.close("all")
            self.log(buf.getvalue())

    def ready(self, i: int) -> bool:
        if i == 0:
            return True
        if self.S is None or not self.S.image_queue:
            return False
        if i <= 3:
            return True
        if i == 4:
            return self.n_labelled() > 0
        if i == 5:
            return "train" in self.done
        return "predict" in self.done

    def n_labelled(self) -> int:
        from pyPetrograph.common.paths import stem_paths

        if self.S is None:
            return 0
        n = 0
        with self.ctx():
            for p in self.S.image_queue:
                if self.S.image_path == p:
                    n += bool(self.S.polygons)
                elif stem_paths(p).polygons.exists() and stem_paths(p).polygons.stat().st_size > 60:
                    n += 1
        return n

    # -- steps ----------------------------------------------------------------------------
    def open(self, paths: List[str], out: str) -> None:
        from pyPetrograph.common.session import Session

        files = [Path(p).expanduser() for p in paths if str(p).strip()]
        missing = [str(p) for p in files if not p.is_file()]
        if not files:
            raise ValueError("Pick at least one photo.")
        if missing:
            raise ValueError("Not found: " + ", ".join(missing))
        self.out = Path(out).expanduser().resolve() if out else _default_out(files[0])
        S = Session()
        S.class_names = dict(DEFAULT_CLASSES)
        S.remember_settings_classes()
        S.current_class = 1
        S.draw_mode = "wand"
        S.wand_threshold = self.wand_threshold
        S.method = "lightgbm"
        S.feature_toggles = dict(self.features_on)
        self.S = S
        with self.ctx():
            S.set_image_queue(files, open_first=True)
        self.done = {"photos"}
        self.train_text, self.train_acc, self.pred = "", None, []
        self.version += 1
        self.status = f"Opened {len(files)} photo(s). Labels are read from / saved to {self.out / 'labels'}"

    def set_classes(self, names: List[str]) -> None:
        S = self.S
        names = [n.strip() for n in names if n and n.strip()]
        if not names:
            raise ValueError("Keep at least one class.")
        if len(set(n.lower() for n in names)) != len(names):
            raise ValueError("Two classes have the same name.")
        old = dict(S.class_names)
        new = {i + 1: n for i, n in enumerate(names)}
        for cid in old:
            if cid not in new:          # removed: drop its outlines on the open photo
                S.remove_class(cid)
        S.class_names = new
        S.remember_settings_classes()
        if S.current_class not in new:
            S.current_class = 1
        self._save()
        self.done.add("classes")
        self.status = "Classes: " + ", ".join(names)

    def open_index(self, i: int) -> None:
        with self.ctx():
            self.S.switch_image(int(i), autosave=False)
        self.version += 1

    def _save(self) -> None:
        with self.ctx():
            self.S.save_all(allow_empty=True)

    def label_op(self, b: Dict[str, Any]) -> str:
        S, op = self.S, b.get("op")
        msg = ""
        if op == "open":
            self.open_index(int(b["i"]))
            return f"Photo {S.stem}"
        if op == "class":
            S.current_class = int(b["cid"])
            return f"Marking: {S.class_names.get(S.current_class)}"
        if op == "threshold":
            self.wand_threshold = S.wand_threshold = float(b["t"])
            return f"Wand tolerance {S.wand_threshold:g}"
        if op == "wand":
            ok = S.add_wand_at(int(round(b["y"])), int(round(b["x"])))
            msg = S.status
            if not ok:
                return msg
        elif op == "poly":
            pts = [(float(x), float(y)) for x, y in b["pts"]]
            if len(pts) < 3:
                return "An outline needs at least 3 points."
            S.add_polygon(pts)
            msg = S.status
        elif op == "undo":
            S.undo()
            msg = S.status
        elif op == "clear":
            S.restart_labels()
            msg = S.status.replace(" (not saved)", "")
        else:
            raise ValueError(f"unknown op {op}")
        self._save()
        self.done.add("label")
        self.done.discard("train")
        return msg

    def label_view(self) -> Dict[str, Any]:
        from pyPetrograph.image_processing.viz import class_color_rgba

        S = self.S
        cols = {cid: "#" + "".join(f"{int(round(c * 255)):02x}" for c in class_color_rgba(cid)[:3])
                for cid in S.class_names}
        polys = []
        for poly, cid in zip(S.polygons, S.class_ids):
            xy = np.asarray(poly.exterior.coords, dtype=float)[:-1]
            polys.append({"cid": int(cid), "flat": [round(v, 1) for v in xy.ravel().tolist()]})
        counts = {str(c): int(sum(1 for k in S.class_ids if int(k) == c)) for c in S.class_names}
        return {"images": [p.stem for p in S.image_queue], "index": S.queue_index,
                "classes": [{"id": c, "name": n, "color": cols[c]} for c, n in sorted(S.class_names.items())],
                "current": S.current_class, "threshold": S.wand_threshold, "polys": polys,
                "counts": counts, "labels_dir": str(self.out / "labels"), "n_labelled": self.n_labelled()}

    def features(self, toggles: Dict[str, bool]) -> None:
        from pyPetrograph.common.notebook_runners import run_feature_preview

        self.features_on = {k: bool(toggles.get(k, False)) for k, _ in FEATURES}
        if not any(self.features_on.values()):
            raise ValueError("Turn on at least one feature.")
        self.S.feature_toggles = dict(self.features_on)
        if self.S.image is not None:
            self._save()
        with self.ctx():
            run_feature_preview(self.S)
        self.done.add("features")
        self.done.discard("train")
        self.version += 1

    def feature_png(self, i: int) -> Path:
        from pyPetrograph.common.paths import stem_paths

        p = self.S.image_queue[i]
        with self.ctx():
            return stem_paths(p).pred.with_name(f"{p.stem}_features.png")

    def train(self) -> None:
        from pyPetrograph.common.notebook_runners import print_train_metrics, run_train_cell

        self.S.feature_toggles = dict(self.features_on)
        with self.ctx():
            res = run_train_cell(self.S, mode="retrain", scope="universal", prompt_load=False)
        with self.ctx() as buf:
            print_train_metrics(res, session=self.S)
        text = buf.getvalue()
        self.train_text = text.strip()
        self.train_acc = res.get("accuracy")
        self.done.add("train")
        self.done.discard("predict")
        self.version += 1

    def predict(self) -> None:
        from pyPetrograph.common.notebook_runners import run_predict_cell

        cur = self.S.queue_index
        with self.ctx():
            res = run_predict_cell(self.S, model_source="universal")
        self.pred = res.get("images", [])
        if 0 <= cur < len(self.S.image_queue):
            with self.ctx():
                self.S.switch_image(cur, autosave=False)
        self.done.add("predict")
        self.version += 1

    def porosity(self, pore: str = "pores"):
        from pyPetrograph.image_processing.fractions import porosity_summary

        with self.ctx():
            tab, stats = porosity_summary(self.S.image_queue, pore_name=pore, verbose=True)
        return tab, stats

    def export_porosity(self, pore: str = "pores") -> Path:
        tab, stats = self.porosity(pore)
        if not len(tab):
            raise ValueError("No porosity values yet: run Predict first.")
        d = self.out / "predictions"
        d.mkdir(parents=True, exist_ok=True)
        path = d / "porosity_summary.csv"
        tab.to_csv(path, index=False)
        self.done.add("porosity")
        return path


def _default_out(first: Path) -> Path:
    """``outputs/`` beside the photo folder (notebook 1 layout: example_data/ and outputs/)."""
    from pyPetrograph.common.paths import get_output_dir, project_root

    root = project_root()
    try:
        first.resolve().relative_to(root)
        return root / "outputs"
    except ValueError:
        return get_output_dir() or (first.resolve().parent.parent / "outputs")


def mount_optical(app, job, err, lock, runner=None):
    """Add the /api/opt/* routes. ``job`` / ``err`` / ``lock`` are the AMICS app's helpers."""
    from fastapi import Body, HTTPException, Query
    from fastapi.responses import Response

    from .routes import png_bytes, read_rgb

    O = OpticalState()
    app.state.opt = O
    cache: Dict[Any, bytes] = {}

    def need(i):
        if not O.ready(i):
            raise HTTPException(409, "open the earlier steps first")

    def user(fn):
        def run(*a):
            try:
                return fn(*a)
            except (ValueError, RuntimeError) as e:
                from .state import UserError
                raise UserError(str(e))
        return run

    @app.get("/api/opt/state")
    def opt_state():
        msg, O.status = O.status, ""
        S = O.S
        return {"steps": [{"name": n, "title": t, "done": n in O.done, "ready": O.ready(i)}
                          for i, (n, t) in enumerate(zip(ORDER, TITLES))],
                "status": msg, "log": O.logs[-80:], "version": O.version,
                "n": len(S.image_queue) if S else 0, "photo": S.stem if S and S.image is not None else None,
                "folder": str(O.out) if O.out else None}

    @app.get("/api/opt/files")
    def opt_files():
        from pyPetrograph.common.paths import project_root

        if O.S is not None:
            return {"paths": [str(p) for p in O.S.image_queue], "out": str(O.out)}
        ex = project_root() / "example_data"
        paths = [str(ex / n) for n in ("AV03_ppl_10x_pic03t3.tif", "AV11_ppl_10x_pic04t4.tif",
                                       "AV12_ppl_10x_pic03t3.tif") if (ex / n).is_file()]
        return {"paths": paths, "out": str(project_root() / "outputs")}

    @app.post("/api/opt/open")
    def opt_open(body: dict = Body(...)):
        def run(j):
            O.open(body.get("paths") or [], body.get("out") or "")
            cache.clear()
        return job("Opening photos", user(run))

    @app.get("/api/opt/classes")
    def opt_classes():
        need(1)
        return O.label_view()

    @app.post("/api/opt/classes")
    def opt_classes_set(body: dict = Body(...)):
        need(1)
        err(lambda: user(O.set_classes)(body.get("names") or []))
        return O.label_view()

    @app.get("/api/opt/label")
    def opt_label():
        need(2)
        with lock:
            return O.label_view()

    @app.post("/api/opt/label/op")
    def opt_label_op(body: dict = Body(...)):
        need(2)
        msg = err(lambda: user(O.label_op)(body))
        cache.clear()
        with lock:
            v = O.label_view()
        v["msg"] = msg
        return v

    def layer(key: str):
        S = O.S
        if S is None:
            return None, False
        if key == "photo":
            return S.image, False
        kind, _, idx = key.partition(":")
        i = int(idx or 0)
        if kind == "feat":
            p = O.feature_png(i)
            return (read_rgb(p) if p.exists() else None), False
        ent = O.pred[i] if 0 <= i < len(O.pred) else None
        if ent is None:
            return None, False
        if kind == "pred":
            p = Path(ent["pred"])
            a = np.asarray(__import__("PIL.Image", fromlist=["Image"]).open(p))
            if a.ndim == 2:
                from pyPetrograph.image_processing.viz import colorize_labels_hsv
                a = colorize_labels_hsv(a)
            return a[..., :3], True
        if kind == "photo":
            return read_rgb(ent["image"]) if Path(ent["image"]).exists() else S.image, False
        if kind == "conf":
            import matplotlib
            c = np.load(ent["confidence"])
            return (matplotlib.colormaps["inferno"](np.clip(c, 0, 1))[..., :3] * 255).astype(np.uint8), False
        return None, False

    @app.get("/api/opt/img/{key}")
    def opt_img(key: str, max: Optional[int] = Query(None), v: str = ""):
        ck = (key, v, max)
        if ck in cache:
            return Response(cache[ck], media_type="image/png")
        with lock:
            arr, nearest = layer(key)
        if arr is None:
            raise HTTPException(404, f"no image {key}")
        b = png_bytes(arr, max, nearest)
        if len(cache) > 30:
            cache.clear()
        cache[ck] = b
        return Response(b, media_type="image/png", headers={"Cache-Control": "no-store"})

    @app.get("/api/opt/imgsize/{key}")
    def opt_imgsize(key: str):
        with lock:
            arr, _ = layer(key)
        if arr is None:
            raise HTTPException(404, f"no image {key}")
        return {"w": int(arr.shape[1]), "h": int(arr.shape[0])}

    @app.get("/api/opt/features")
    def opt_features():
        need(3)
        S = O.S
        imgs = []
        for i, p in enumerate(S.image_queue):
            imgs.append({"name": p.stem, "ready": O.feature_png(i).exists() and "features" in O.done})
        return {"features": [{"key": k, "label": lab, "on": O.features_on.get(k, False)} for k, lab in FEATURES],
                "images": imgs}

    @app.post("/api/opt/features")
    def opt_features_run(body: dict = Body(...)):
        need(3)
        def run(j):
            user(O.features)(body.get("toggles") or {})
            cache.clear()
        return job("Feature preview", run)

    @app.get("/api/opt/train")
    def opt_train():
        need(4)
        return {"text": O.train_text, "accuracy": O.train_acc, "n_labelled": O.n_labelled(),
                "n": len(O.S.image_queue), "features": [lab for k, lab in FEATURES if O.features_on.get(k)]}

    @app.post("/api/opt/train")
    def opt_train_run():
        need(4)
        return job("Training the pixel classifier", lambda j: user(O.train)())

    @app.get("/api/opt/predict")
    def opt_predict():
        need(5)
        return {"images": [{"name": Path(e["image"]).stem,
                            "fractions": e.get("fractions", {})} for e in O.pred]}

    @app.post("/api/opt/predict")
    def opt_predict_run():
        need(5)
        def run(j):
            user(O.predict)()
            cache.clear()
        return job("Predicting every photo", run)

    @app.get("/api/opt/porosity")
    def opt_porosity(pore: str = "pores"):
        need(6)
        with lock:
            tab, stats = O.porosity(pore)
        return {"rows": tab.to_dict(orient="records"), "stats": stats,
                "classes": list(O.S.class_names.values())}

    @app.post("/api/opt/porosity/export")
    def opt_porosity_export(body: dict = Body(default={})):
        need(6)
        def run(j):
            p = user(O.export_porosity)(body.get("pore") or "pores")
            return {"path": str(p)}
        return job("Saving porosity table", run)

    return O

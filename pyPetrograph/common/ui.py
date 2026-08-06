"""Label UI, file pickers, undo stack, and app launch helpers."""
from __future__ import annotations

import json
import subprocess
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path as MplPath
from shapely.geometry import Polygon

from pyPetrograph.common.constants import (
    DEFAULT_CLASSES,
    DEFAULT_WAND_SIMPLIFY,
    DEFAULT_WAND_THRESHOLD,
    LOD_MAX_SIDE,
    PathLike,
    SUPPORTED_EXTENSIONS,
    WAND_SLIDER_MAX,
    WAND_SLIDER_MIN,
)
from pyPetrograph.common.image_io import ensure_rgb_cache
from pyPetrograph.common.paths import cleanup_legacy_outputs, project_root, relpath_display
from pyPetrograph.common.session import Session
from pyPetrograph.image_processing.features import (
    _downsample_for_view,
    max_side_for_target_mp,
    resolve_feature_toggles,
    resolve_view_target_mp,
)
from pyPetrograph.image_processing.viz import class_color_rgba
from pyPetrograph.labeling_ml.polygons import magic_wand_to_polygon, remove_duplicate_polys

class UndoState:
    polygons: List[Polygon]
    class_ids: List[int]
    labels: np.ndarray
    class_names: Dict[int, str]


class UndoStack:
    """General undo stack for labeling / class-list steps."""

    def __init__(self, maxlen: int = 50):
        self._stack: List[UndoState] = []
        self.maxlen = maxlen

    def push(self, state: UndoState) -> None:
        self._stack.append(state)
        if len(self._stack) > self.maxlen:
            self._stack.pop(0)

    def pop(self) -> Optional[UndoState]:
        if not self._stack:
            return None
        return self._stack.pop()

    def clear(self) -> None:
        self._stack.clear()

    def __len__(self) -> int:
        return len(self._stack)


_IMAGE_FILE_FILTER = (
    "Images (*.tif *.tiff *.geotiff *.jpg *.jpeg *.png *.bmp *.webp *.gif);;All (*)"
)


def _run_qt_picker_from_config(config_path: str) -> None:
    """
    Child-process entry: Qt file / yes-no dialogs (never run inside Jupyter kernel).

    Same idea as ``_run_label_gui_from_config`` — keeps the notebook kernel Qt-free
    so Windows/macOS/Linux Jupyter does not hard-crash (e.g. exit 3221226505).
    """
    from PyQt5 import QtWidgets

    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    out_path = Path(cfg["out_path"])
    mode = str(cfg.get("mode", "open_files"))
    result: Dict[str, Any] = {}

    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])

    try:
        if mode == "ask_add_more":
            box = QtWidgets.QMessageBox()
            box.setWindowTitle("Select images")
            box.setText("Add more images from another folder?")
            add_btn = box.addButton("Add more", QtWidgets.QMessageBox.AcceptRole)
            box.addButton("Done", QtWidgets.QMessageBox.RejectRole)
            box.setDefaultButton(add_btn)
            box.exec_()
            result = {"add_more": box.clickedButton() == add_btn}
        else:
            init = str(cfg.get("initial_dir") or ".")
            multi = bool(cfg.get("multi", True))
            title = str(cfg.get("caption") or ("Select image(s)" if multi else "Select file"))
            flt = str(cfg.get("name_filter") or "All (*)")
            if multi:
                paths, _ = QtWidgets.QFileDialog.getOpenFileNames(None, title, init, flt)
                result = {"paths": list(paths)}
            else:
                path, _ = QtWidgets.QFileDialog.getOpenFileName(None, title, init, flt)
                result = {"paths": [path] if path else []}
    except Exception as exc:
        result = {"paths": [], "add_more": False, "error": str(exc)}

    out_path.write_text(json.dumps(result), encoding="utf-8")
    try:
        Path(config_path).unlink(missing_ok=True)
    except Exception:
        pass


def _run_qt_picker_subprocess(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Spawn a short-lived Qt dialog process; return parsed JSON result."""
    import os
    import subprocess
    import sys
    import tempfile

    root = str(Path(__file__).resolve().parent.parent.parent)
    out_tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix="_petrography_picker_out.json", delete=False, encoding="utf-8"
    )
    out_path = out_tmp.name
    out_tmp.close()
    Path(out_path).write_text("{}", encoding="utf-8")

    cfg = dict(cfg)
    cfg["out_path"] = out_path
    in_tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix="_petrography_picker_in.json", delete=False, encoding="utf-8"
    )
    with in_tmp:
        json.dump(cfg, in_tmp, indent=2)
        cfg_path = in_tmp.name

    env = os.environ.copy()
    env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
    # Keep child free of Matplotlib GUI backends; this process is Qt-only dialogs.
    env.pop("MPLBACKEND", None)

    code = (
        "from pyPetrograph import _run_qt_picker_from_config; "
        f"_run_qt_picker_from_config({cfg_path!r})"
    )
    try:
        r = subprocess.run(
            [sys.executable, "-c", code],
            env=env,
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        if r.returncode != 0:
            err = (r.stderr or r.stdout or "").strip() or f"exit {r.returncode}"
            warnings.warn(
                f"File picker subprocess failed ({err}). "
                "Install PyQt5 (conda install -c conda-forge pyqt) "
                "or set paths manually."
            )
            return {}
        return json.loads(Path(out_path).read_text(encoding="utf-8"))
    except Exception as exc:
        warnings.warn(
            f"File picker unavailable ({exc}). "
            "Install PyQt5 (conda install -c conda-forge pyqt) "
            "or set paths manually."
        )
        return {}
    finally:
        for p in (cfg_path, out_path):
            try:
                Path(p).unlink(missing_ok=True)
            except Exception:
                pass


def _pick_files_native(
    initial_dir: str,
    multi: bool = True,
    *,
    caption: Optional[str] = None,
    name_filter: Optional[str] = None,
) -> List[str]:
    """
    OS-agnostic file picker (macOS / Windows / Linux).

    Runs Qt ``QFileDialog`` in a **subprocess** (never inside the Jupyter kernel).
    """
    init = str(Path(initial_dir).expanduser().resolve())
    title = caption or ("Select image(s)" if multi else "Select file")
    flt = name_filter or (_IMAGE_FILE_FILTER if multi else "All (*)")
    result = _run_qt_picker_subprocess(
        {
            "mode": "open_files",
            "initial_dir": init,
            "multi": multi,
            "caption": title,
            "name_filter": flt,
        }
    )
    paths = result.get("paths") or []
    return [str(p) for p in paths if p]


def _ask_add_more_images() -> bool:
    """Ask whether to pick more images (Qt dialog in a subprocess; all OS)."""
    result = _run_qt_picker_subprocess({"mode": "ask_add_more"})
    return bool(result.get("add_more"))


def select_images(
    session: Optional[Session] = None,
    *,
    append: bool = False,
    initial_dir: Optional[PathLike] = None,
) -> Session:
    """
    Desktop file-picker popup to choose images **before** ``launch_app``.

    Multi-select in one folder, then optionally **Add more** from another folder
    (so the queue can span different subfolders). Same Qt dialogs on macOS,
    Windows, and Linux (subprocess — does not crash the Jupyter kernel).

    Parameters
    ----------
    session :
        Existing session, or a new one if None.
    append :
        If True, keep the current queue and add to it.
    initial_dir :
        Where the first dialog opens (default: ``session.image_dir`` or ``.``).

    Returns
    -------
    Session with ``image_queue`` set (first image opened for companion auto-load).
    """
    session = session or Session()
    if initial_dir is not None:
        start = Path(initial_dir).expanduser().resolve()
    elif session.image_dir is not None:
        start = Path(session.image_dir).expanduser().resolve()
    else:
        start = Path(".").resolve()

    picked: List[Path] = list(session.image_queue) if append else []
    seen = {p.resolve() for p in picked}
    current_dir = str(start if start.is_dir() else start.parent)

    while True:
        paths = _pick_files_native(current_dir, multi=True)
        if not paths:
            break
        for p in paths:
            rp = Path(p).resolve()
            if rp not in seen:
                seen.add(rp)
                picked.append(rp)
            current_dir = str(rp.parent)
        print(f"Selected {len(paths)} file(s). Queue total: {len(picked)}.")
        for i, p in enumerate(picked, 1):
            print(f"  {i}. {relpath_display(p)}")
        if not _ask_add_more_images():
            break

    if not picked:
        print("No images in queue. Run select_images again, or set paths manually.")
        return session

    session.set_image_queue(picked, open_first=True)
    # Browse start for any later tools = folder of first image
    session.image_dir = picked[0].parent
    print(f"Ready: {len(picked)} image(s). Next: launch_app(session).")
    return session


class LabelWindow:
    """DMG-07-style one-figure label UI: left control panel + right image axes."""

    def __init__(self, session: Session):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import (
            Button as MplButton,
            PolygonSelector,
            Slider,
            TextBox,
        )

        self.plt = plt
        self.session = session
        self._PolygonSelector = PolygonSelector

        self.fig = plt.figure(figsize=(14, 9))
        try:
            self.fig.canvas.manager.set_window_title("Petrography — label")
        except Exception:
            pass

        # Left controls (above minimap), main image, dedicated minimap (no inset overlap)
        self.ax_panel = self.fig.add_axes([0.015, 0.20, 0.21, 0.76])
        self.ax_panel.set_facecolor("#d8d8d8")
        self.ax_panel.set_xticks([])
        self.ax_panel.set_yticks([])
        for spine in self.ax_panel.spines.values():
            spine.set_visible(False)

        self.ax = self.fig.add_axes([0.26, 0.04, 0.72, 0.92])
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_title("No image — run select_images() in the notebook first")

        self.minimap_ax = self.fig.add_axes([0.015, 0.03, 0.21, 0.15])
        self.minimap_ax.set_facecolor("#222222")
        self.minimap_ax.set_xticks([])
        self.minimap_ax.set_yticks([])
        self.minimap_rect = None
        self._minimap_artist = None

        # Stacked controls (figure coords; left column)
        x0, w = 0.03, 0.18
        y = 0.92
        h_btn = 0.035
        gap = 0.008

        def _btn(label: str, height: float = h_btn):
            nonlocal y
            y -= height
            axb = self.fig.add_axes([x0, y, w, height])
            b = MplButton(axb, label)
            y -= gap
            return b

        def _pair(lab_l: str, lab_r: str, height: float = h_btn):
            nonlocal y
            y -= height
            half = (w - 0.008) / 2.0
            ax_l = self.fig.add_axes([x0, y, half, height])
            ax_r = self.fig.add_axes([x0 + half + 0.008, y, half, height])
            bl = MplButton(ax_l, lab_l)
            br = MplButton(ax_r, lab_r)
            y -= gap
            return bl, br

        def _quad(labs, height: float = h_btn):
            nonlocal y
            half = (w - 0.008) / 2.0
            out = []
            for i in range(0, 4, 2):
                y -= height
                ax_a = self.fig.add_axes([x0, y, half, height])
                ax_b = self.fig.add_axes([x0 + half + 0.008, y, half, height])
                out.append(MplButton(ax_a, labs[i]))
                out.append(MplButton(ax_b, labs[i + 1]))
                y -= gap
            return out

        self.btn_prev, self.btn_next = _pair("◀ Prev", "Next ▶")
        self.btn_save, self.btn_load, self.btn_undo, self.btn_restart = _quad(
            ["Save", "Reload", "Undo", "Restart"]
        )
        self.btn_cycle = _btn("Cycle class")
        self.btn_mode = _btn("Mode: polygon")
        self.btn_fit = _btn("Fit view")

        y -= 0.006
        y -= 0.038
        ax_thr = self.fig.add_axes([x0, y, w, 0.032])
        self.slider_wand = Slider(
            ax_thr,
            "Wand",
            float(WAND_SLIDER_MIN),
            float(WAND_SLIDER_MAX),
            valinit=float(session.wand_threshold),
            valstep=1.0,
        )
        y -= gap + 0.006

        # Class id (read-only label) + editable name
        y -= 0.028
        self.ax_id_label = self.fig.add_axes([x0, y, w, 0.026])
        self.ax_id_label.set_facecolor("#d8d8d8")
        self.ax_id_label.set_xticks([])
        self.ax_id_label.set_yticks([])
        for spine in self.ax_id_label.spines.values():
            spine.set_visible(False)
        self.id_text = self.ax_id_label.text(
            0.0, 0.5, "id: —", transform=self.ax_id_label.transAxes,
            va="center", ha="left", fontsize=9, fontweight="bold",
        )
        y -= gap

        y -= 0.032
        ax_name = self.fig.add_axes([x0, y, w, 0.030])
        cname0 = session.class_names.get(session.current_class, "")
        self.textbox_name = TextBox(ax_name, "name ", initial=str(cname0))
        y -= gap

        self.btn_add = _btn("Add label")

        # Status box stays above dedicated minimap (minimap top ≈ 0.18)
        y -= 0.008
        info_bottom = 0.21
        info_h = max(0.06, y - info_bottom)
        y = info_bottom
        self.ax_info = self.fig.add_axes([x0, y, w, info_h])
        self.ax_info.set_facecolor("#c8c8c8")
        self.ax_info.set_xticks([])
        self.ax_info.set_yticks([])
        for spine in self.ax_info.spines.values():
            spine.set_color("#888888")
        self.info_text = self.ax_info.text(
            0.04,
            0.96,
            "",
            transform=self.ax_info.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            wrap=True,
            family="sans-serif",
        )

        # View / selector state
        self.selector = None
        self.patches: List[Any] = []
        self.image_artist = None
        self.overview = None
        self.pan_origin = None
        self.updating_view = False
        self.mod_pan = False
        self._cids: List[Any] = []
        self._syncing_name = False

        # Wire buttons
        self.btn_prev.on_clicked(lambda _e: self.do_prev())
        self.btn_next.on_clicked(lambda _e: self.do_next())
        self.btn_save.on_clicked(lambda _e: self.do_save())
        self.btn_load.on_clicked(lambda _e: self.do_load())
        self.btn_undo.on_clicked(lambda _e: self.do_undo())
        self.btn_restart.on_clicked(lambda _e: self.do_restart())
        self.btn_cycle.on_clicked(lambda _e: self.do_cycle_class())
        self.btn_mode.on_clicked(lambda _e: self.do_toggle_mode())
        self.btn_fit.on_clicked(lambda _e: self.do_fit())
        self.btn_add.on_clicked(lambda _e: self.do_add_class())
        self.textbox_name.on_submit(self._on_name_submit)
        self.slider_wand.on_changed(lambda v: setattr(self.session, "wand_threshold", float(v)))

        self._connect_canvas()
        self.session.on_change(lambda: (self.refresh_overlays(), self._update_info()))
        self._sync_mode_button()
        self._sync_class_fields()
        self._update_info()

        if self.session.image is not None:
            self._show_image()
        else:
            self._set_status("No image — run select_images() first.")

    # ------------------------------------------------------------------
    # Status / info
    # ------------------------------------------------------------------

    def _set_status(self, msg: str) -> None:
        # Keep short; avoid dumping paths
        self.session.status = (msg or "")[:120]
        self._update_info()
        self.fig.canvas.draw_idle()

    def _current_class_poly_count(self) -> int:
        cid = int(self.session.current_class)
        return sum(1 for c in self.session.class_ids if int(c) == cid)

    def _sync_class_fields(self) -> None:
        s = self.session
        self.id_text.set_text(f"id: {s.current_class}")
        name = s.class_names.get(s.current_class, "")
        self._syncing_name = True
        try:
            self.textbox_name.set_val(str(name))
        finally:
            self._syncing_name = False

    def _on_name_submit(self, text: str) -> None:
        if self._syncing_name:
            return
        name = (text or "").strip()
        if not name:
            return
        self.session.set_class_name(self.session.current_class, name)
        self._set_status(f"Renamed → {name}")

    def _update_info(self) -> None:
        s = self.session
        n = len(s.image_queue)
        i = s.queue_index + 1 if n else 0
        n_poly = self._current_class_poly_count()
        lines = [
            f"image {i}/{n}",
            f"polys (this class): {n_poly}",
            f"mode: {s.draw_mode}",
        ]
        hint = (s.status or "").strip()
        if hint and hint not in ("Ready.",):
            # wrap ~28 chars
            while hint:
                lines.append(hint[:28])
                hint = hint[28:]
        self.info_text.set_text("\n".join(lines[:8]))
        self.fig.canvas.draw_idle()

    def _sync_mode_button(self) -> None:
        mode = self.session.draw_mode
        label = "Mode: wand" if mode == "wand" else "Mode: polygon"
        self.btn_mode.label.set_text(label)

    # ------------------------------------------------------------------
    # Image display / LOD / overlays
    # ------------------------------------------------------------------

    def _show_image(self) -> None:
        if self.session.image is None:
            self._set_status("No image — run select_images() first.")
            return

        self.session.editing_active = True
        img = self.session.image
        h, w = img.shape[:2]

        for patch in self.patches:
            try:
                patch.remove()
            except Exception:
                pass
        self.patches = []
        if self.selector is not None:
            try:
                self.selector.set_active(False)
                self.selector.disconnect_events()
            except Exception:
                pass
            self.selector = None

        self.overview = _downsample_for_view(img, target_mp=self.session.view_target_mp())
        if self.image_artist is None:
            self.image_artist = self.ax.imshow(
                self.overview, extent=(0, w, h, 0), interpolation="nearest"
            )
        else:
            self.image_artist.set_data(self.overview)
            self.image_artist.set_extent((0, w, h, 0))

        self.ax.set_xlim(0, w)
        self.ax.set_ylim(h, 0)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_title("")

        # Dedicated lower-left minimap (never overlaps main view)
        self.minimap_ax.clear()
        self.minimap_ax.set_facecolor("#222222")
        self._minimap_artist = self.minimap_ax.imshow(
            self.overview, extent=(0, w, h, 0), interpolation="nearest"
        )
        self.minimap_ax.set_xlim(0, w)
        self.minimap_ax.set_ylim(h, 0)
        self.minimap_ax.set_xticks([])
        self.minimap_ax.set_yticks([])
        self.minimap_rect = Rectangle(
            (0, 0), w, h, fill=False, edgecolor="yellow", linewidth=1.2
        )
        self.minimap_ax.add_patch(self.minimap_rect)

        def onselect(verts):
            if not self.session.editing_active or self.session.draw_mode != "polygon":
                return
            self.session.add_polygon(verts, class_id=self.session.current_class)
            self.refresh_overlays()
            self._update_info()

        edge = class_color_rgba(self.session.current_class)
        self.selector = self._PolygonSelector(
            self.ax,
            onselect,
            useblit=True,
            props=dict(color=edge[:3], linewidth=1.5),
        )
        self._set_selector_active(True)
        self.refresh_overlays()
        self._sync_mode_button()
        self._sync_class_fields()
        self._update_info()
        self.fig.canvas.draw_idle()

    def refresh_overlays(self) -> None:
        if self.session.image is None:
            return
        for patch in self.patches:
            try:
                patch.remove()
            except Exception:
                pass
        self.patches = []
        for poly, cid in zip(self.session.polygons, self.session.class_ids):
            xs, ys = poly.exterior.xy
            verts = list(zip(xs, ys))
            codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(verts) - 1) + [MplPath.CLOSEPOLY]
            path = MplPath(verts + [verts[0]], codes)
            color = class_color_rgba(int(cid))
            patch = PathPatch(
                path, facecolor="none", edgecolor=color[:3], lw=2.0
            )
            self.ax.add_patch(patch)
            self.patches.append(patch)
        self._update_minimap_rect()
        self.fig.canvas.draw_idle()

    def _update_minimap_rect(self) -> None:
        if self.minimap_rect is None or self.session.image is None:
            return
        h, w = self.session.image.shape[:2]
        x0, x1 = self.ax.get_xlim()
        y1, y0 = self.ax.get_ylim()
        x0, x1 = max(0, min(x0, x1)), min(w, max(x0, x1))
        y0, y1 = max(0, min(y0, y1)), min(h, max(y0, y1))
        self.minimap_rect.set_xy((x0, y0))
        self.minimap_rect.set_width(max(x1 - x0, 1))
        self.minimap_rect.set_height(max(y1 - y0, 1))

    def _update_display_image(self) -> None:
        if self.session.image is None:
            return
        if self.updating_view:
            return
        self.updating_view = True
        try:
            img = self.session.image
            h, w = img.shape[:2]
            x0, x1 = self.ax.get_xlim()
            y1, y0 = self.ax.get_ylim()
            x0, x1 = min(x0, x1), max(x0, x1)
            y0, y1 = min(y0, y1), max(y0, y1)
            view_w = max(x1 - x0, 1.0)
            view_h = max(y1 - y0, 1.0)
            full = abs(x0) < 1 and abs(x1 - w) < 1 and abs(y0) < 1 and abs(y1 - h) < 1
            zoomed = (view_w < 0.9 * w) or (view_h < 0.9 * h)
            if not zoomed or full:
                overview = self.overview
                if overview is None:
                    overview = _downsample_for_view(img, target_mp=self.session.view_target_mp())
                    self.overview = overview
                disp, extent = overview, (0, w, h, 0)
            else:
                pad = 2
                c0 = max(0, int(np.floor(x0)) - pad)
                c1 = min(w, int(np.ceil(x1)) + pad)
                r0 = max(0, int(np.floor(y0)) - pad)
                r1 = min(h, int(np.ceil(y1)) + pad)
                if c1 <= c0 or r1 <= r0:
                    disp, extent = self.overview or img, (0, w, h, 0)
                else:
                    crop = img[r0:r1, c0:c1]
                    ch, cw = crop.shape[:2]
                    max_pix = int(max(self.session.view_target_mp(), 0.5) * 1_000_000)
                    if ch * cw > max_pix:
                        scale = (max_pix / float(ch * cw)) ** 0.5
                        nw = max(1, int(cw * scale))
                        nh = max(1, int(ch * scale))
                        crop = np.asarray(
                            Image.fromarray(crop).resize((nw, nh), Image.Resampling.BILINEAR),
                            dtype=np.uint8,
                        )
                    disp, extent = crop, (c0, c1, r1, r0)
            if self.image_artist is None:
                self.image_artist = self.ax.imshow(disp, extent=extent, interpolation="nearest")
            else:
                self.image_artist.set_data(disp)
                self.image_artist.set_extent(extent)
            self._update_minimap_rect()
            self.fig.canvas.draw_idle()
        finally:
            self.updating_view = False

    def _set_selector_active(self, active: bool) -> None:
        if self.selector is not None:
            self.selector.set_active(
                active and self.session.draw_mode == "polygon" and self.session.editing_active
            )

    # ------------------------------------------------------------------
    # Button actions
    # ------------------------------------------------------------------

    def do_prev(self) -> None:
        if not self.session.image_queue:
            return
        self.session.switch_image(max(0, self.session.queue_index - 1), autosave=False)
        self._show_image()

    def do_next(self) -> None:
        if not self.session.image_queue:
            return
        self.session.switch_image(
            min(len(self.session.image_queue) - 1, self.session.queue_index + 1),
            autosave=False,
        )
        self._show_image()

    def do_save(self) -> None:
        self.session.save_all()
        self._set_status(self.session.status)

    def do_load(self) -> None:
        """Reload saved labels from disk into the scene (undoable)."""
        self.session.load_labels_only()
        self.refresh_overlays()
        self._set_status(self.session.status)

    def do_undo(self) -> None:
        self.session.undo()
        self.refresh_overlays()
        self._update_info()

    def do_restart(self) -> None:
        """Clear current class in the scene only (memory; no disk write)."""
        self.session.restart_labels()
        self._sync_class_fields()
        self.refresh_overlays()
        self._set_status(self.session.status)

    def do_cycle_class(self) -> None:
        ids = sorted(self.session.class_names.keys())
        if not ids:
            return
        try:
            i = ids.index(self.session.current_class)
            self.session.current_class = ids[(i + 1) % len(ids)]
        except ValueError:
            self.session.current_class = ids[0]
        self._sync_class_fields()
        # Refresh draw-tool color for the new class
        if self.selector is not None and self.session.image is not None:
            edge = class_color_rgba(self.session.current_class)[:3]
            try:
                self.selector.to_draw.set_color(edge)
            except Exception:
                pass
            try:
                for artist in getattr(self.selector, "artists", []) or []:
                    artist.set_color(edge)
            except Exception:
                pass
        self._set_status(f"Class → {self.session.current_class}")

    def do_toggle_mode(self) -> None:
        self.session.draw_mode = "wand" if self.session.draw_mode == "polygon" else "polygon"
        self._set_selector_active(True)
        self._sync_mode_button()
        self._set_status(f"Mode → {self.session.draw_mode}")

    def do_fit(self) -> None:
        if self.session.image is None:
            return
        hh, ww = self.session.image.shape[:2]
        self.ax.set_xlim(0, ww)
        self.ax.set_ylim(hh, 0)
        self._update_display_image()

    def do_add_class(self) -> None:
        name = (self.textbox_name.text or "").strip()
        if not name:
            self._set_status("Type a name, then Add label.")
            return
        self.session.add_class(name)
        self._sync_class_fields()
        self._update_info()

    def do_rename_class(self) -> None:
        # Kept for compatibility; name edits go through the textbox submit.
        self._on_name_submit(self.textbox_name.text or "")

    # ------------------------------------------------------------------
    # Canvas events
    # ------------------------------------------------------------------

    def _connect_canvas(self) -> None:
        c = self.fig.canvas
        self._cids = [
            c.mpl_connect("key_press_event", self._on_key),
            c.mpl_connect("key_press_event", self._on_key_mod),
            c.mpl_connect("key_release_event", self._on_key_mod_up),
            c.mpl_connect("scroll_event", self._on_scroll),
            c.mpl_connect("button_press_event", self._on_press),
            c.mpl_connect("motion_notify_event", self._on_motion),
            c.mpl_connect("button_release_event", self._on_release),
        ]
        self.ax.callbacks.connect("xlim_changed", self._on_limit_change)
        self.ax.callbacks.connect("ylim_changed", self._on_limit_change)

    def _on_key(self, event) -> None:
        if event.key in ("enter", "return"):
            if not self.session.editing_active or self.session.draw_mode != "polygon":
                return
            if self.selector is None:
                return
            verts = list(getattr(self.selector, "verts", []) or [])
            if len(verts) >= 3:
                self.session.add_polygon(verts, class_id=self.session.current_class)
                try:
                    self.selector.clear()
                except Exception:
                    pass
                self.refresh_overlays()
                self._update_info()
        # Esc: leave to matplotlib default (new polygon / cancel edit)
        elif event.key in ("n", "N"):
            self.do_next()
        elif event.key in ("p", "P"):
            self.do_prev()
        elif event.key == "s":
            self.do_save()
        elif event.key == "m":
            self.do_toggle_mode()
        elif event.key == "c":
            self.do_cycle_class()

    def _on_key_mod(self, event) -> None:
        if event.key in ("control", "cmd", "meta", "ctrl"):
            self.mod_pan = True

    def _on_key_mod_up(self, event) -> None:
        if event.key in ("control", "cmd", "meta", "ctrl"):
            self.mod_pan = False

    def _on_scroll(self, event) -> None:
        if event.inaxes != self.ax or self.session.image is None:
            return
        if event.xdata is None or event.ydata is None:
            return
        # Gentler steps; clamp so view never goes below ~32 px or above full image
        base = 1.2 if event.button == "up" else 1 / 1.2
        h, w = self.session.image.shape[:2]
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        cx, cy = event.xdata, event.ydata
        nx0 = cx - (cx - x0) / base
        nx1 = cx - (cx - x1) / base
        ny0 = cy - (cy - y0) / base
        ny1 = cy - (cy - y1) / base
        vw = abs(nx1 - nx0)
        vh = abs(ny1 - ny0)
        min_side = 32.0
        if vw < min_side or vh < min_side:
            return
        if vw > w * 1.05 and vh > h * 1.05:
            self.ax.set_xlim(0, w)
            self.ax.set_ylim(h, 0)
        else:
            self.ax.set_xlim(nx0, nx1)
            self.ax.set_ylim(ny0, ny1)
        self._update_display_image()

    def _on_press(self, event) -> None:
        if event.inaxes != self.ax:
            return
        want_pan = (
            self.mod_pan
            or event.button == 2
            or (
                event.key is not None
                and (
                    "control" in str(event.key)
                    or "cmd" in str(event.key)
                    or "meta" in str(event.key)
                )
            )
        )
        if want_pan and event.x is not None and event.y is not None:
            self.pan_origin = (
                float(event.x),
                float(event.y),
                self.ax.get_xlim(),
                self.ax.get_ylim(),
            )
            return
        if (
            self.session.editing_active
            and self.session.draw_mode == "wand"
            and event.button == 1
            and event.xdata is not None
            and event.ydata is not None
        ):
            self.session.add_wand_at(int(round(event.ydata)), int(round(event.xdata)))
            self.refresh_overlays()
            self._update_info()

    def _on_motion(self, event) -> None:
        if self.pan_origin is None or event.x is None or event.y is None:
            return
        ox, oy, (x0, x1), (y0, y1) = self.pan_origin
        try:
            inv = self.ax.transData.inverted()
            d0 = inv.transform((ox, oy))
            d1 = inv.transform((event.x, event.y))
            dx, dy = d1[0] - d0[0], d1[1] - d0[1]
        except Exception:
            return
        self.ax.set_xlim(x0 - dx, x1 - dx)
        self.ax.set_ylim(y0 - dy, y1 - dy)
        self._update_minimap_rect()
        self.fig.canvas.draw_idle()

    def _on_release(self, event) -> None:
        if self.pan_origin is not None:
            self.pan_origin = None
            self._update_display_image()

    def _on_limit_change(self, _ax) -> None:
        if self.updating_view:
            return
        self._update_display_image()

    def run(self) -> None:
        """Block until the window is closed (DMG-07 QuadPicker style)."""
        # No autosave on close — only the Save button writes labels to disk.
        self.plt.show(block=True)


def build_ui(session: Optional[Session] = None):
    """Create LabelWindow; returns (session, win). DMG-07 style (no plt.ion)."""
    session = session or Session()
    win = LabelWindow(session)
    session.open_label_window = lambda: win._show_image()  # type: ignore[attr-defined]
    return session, win


def launch_app(session: Optional[Session] = None, *, wait: bool = False):
    """
    Open the label window in a **separate Python process** (DMG-07 Qt popup style).

    Cursor Jupyter often dies if a GUI runs inside the kernel; the child process
    uses Matplotlib **QtAgg** (PyQt5), same backend idea as DMG-07's
    ``%matplotlib qt`` pickers.

    Saves still go next to your images (Save button only); re-run Label summary afterward.
    """
    import os
    import subprocess
    import sys
    import tempfile

    session = session or Session()
    if not session.image_queue:
        raise RuntimeError(
            "No images in the queue.\n"
            "Run select_images(session) in the notebook first "
            "(file picker; you can Add more from other folders), "
            "then launch_app(session)."
        )
    # Do NOT re-snapshot class defaults here — Settings cell already called
    # remember_settings_classes(). Re-snapshotting after select/open would bake
    # session.json names into Restart defaults.
    root = str(Path(__file__).resolve().parent.parent.parent)
    cfg = _session_launch_config(session)
    cfg["project_root"] = root

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix="_petrography_label.json", delete=False, encoding="utf-8"
    )
    with tmp:
        json.dump(cfg, tmp, indent=2)
        cfg_path = tmp.name

    env = os.environ.copy()
    # Match DMG-07: Qt backend (not Tk — Tk crashes on modern macOS)
    env["MPLBACKEND"] = "QtAgg"
    env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
    # Prevent any accidental tkinter init in the child
    env["PETROGRAPHY_LABEL_GUI"] = "1"

    code = (
        "from pyPetrograph import _run_label_gui_from_config; "
        f"_run_label_gui_from_config({cfg_path!r})"
    )
    cmd = [sys.executable, "-c", code]
    if wait:
        subprocess.call(cmd, env=env, cwd=root)
        print("Label window closed.")
    else:
        subprocess.Popen(cmd, env=env, cwd=root)
        print("Label window opened (Qt popup in a separate process).")
        print("Draw / Save there. Then run Label summary in this notebook.")
    return session


def _session_launch_config(session: Session) -> Dict[str, Any]:
    return {
        "image_dir": str(session.image_dir) if session.image_dir else None,
        "image_queue": [str(p) for p in session.image_queue],
        "queue_index": int(session.queue_index) if session.queue_index >= 0 else 0,
        "class_names": {str(k): v for k, v in session.class_names.items()},
        "default_class_names": {
            str(k): v for k, v in session.default_class_names.items()
        },
        "current_class": int(session.current_class),
        "method": session.method,
        "n_jobs": int(session.n_jobs),
        "preview_dpi": int(session.preview_dpi),
        "figure_dpi": int(session.figure_dpi),
        "load_target_mp": session.load_target_mp,
        "max_side": session.max_side,
        "y_target": float(session.y_target),
        "median_size": int(session.median_size),
        "draw_mode": session.draw_mode,
        "wand_threshold": float(session.wand_threshold),
        "preview_target_mp": session.preview_target_mp,
        "texture_window": int(session.texture_window),
        "lbp_p": int(session.lbp_p),
        "lbp_r": float(session.lbp_r),
        "feature_toggles": dict(session.feature_toggles),
    }


def _session_from_launch_config(cfg: Dict[str, Any]) -> Session:
    s = Session()
    s.class_names = {int(k): v for k, v in cfg.get("class_names", s.class_names).items()}
    if cfg.get("default_class_names"):
        s.default_class_names = {
            int(k): v for k, v in cfg["default_class_names"].items()
        }
    else:
        s.default_class_names = dict(s.class_names)
    s.current_class = int(cfg.get("current_class", s.current_class))
    if cfg.get("image_dir"):
        s.image_dir = Path(cfg["image_dir"])
    s.method = cfg.get("method", s.method)
    s.n_jobs = int(cfg.get("n_jobs", s.n_jobs))
    s.preview_dpi = int(cfg.get("preview_dpi", s.preview_dpi))
    s.figure_dpi = int(cfg.get("figure_dpi", s.figure_dpi))
    s.load_target_mp = cfg.get("load_target_mp", s.load_target_mp)
    s.max_side = cfg.get("max_side", s.max_side)
    s.y_target = float(cfg.get("y_target", s.y_target))
    s.median_size = int(cfg.get("median_size", s.median_size))
    s.draw_mode = cfg.get("draw_mode", s.draw_mode)
    s.wand_threshold = float(cfg.get("wand_threshold", s.wand_threshold))
    if "preview_target_mp" in cfg:
        s.preview_target_mp = cfg.get("preview_target_mp")
    s.texture_window = int(cfg.get("texture_window", s.texture_window))
    s.lbp_p = int(cfg.get("lbp_p", s.lbp_p))
    s.lbp_r = float(cfg.get("lbp_r", s.lbp_r))
    if cfg.get("feature_toggles"):
        s.feature_toggles = resolve_feature_toggles(cfg["feature_toggles"])

    queue = [Path(p) for p in cfg.get("image_queue") or []]
    if queue:
        idx = int(cfg.get("queue_index", 0))
        idx = max(0, min(idx, len(queue) - 1))
        s.set_image_queue(queue, open_first=False)
        s.switch_image(idx, autosave=False)
    return s


def _run_label_gui_from_config(config_path: str) -> None:
    """Child-process entry: QtAgg popup like DMG-07, block until closed."""
    import os
    import sys

    # Backend MUST be set before pyplot / any GUI toolkit init.
    # Never use Tk on macOS (crashes). Match DMG-07: Qt.
    os.environ["MPLBACKEND"] = "QtAgg"
    import matplotlib

    matplotlib.use("QtAgg", force=True)

    try:
        import PyQt5  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "PyQt5 is required for the label window (same as DMG-07).\n"
            "In conda env work:  conda install -c conda-forge pyqt"
        ) from exc

    cfg = json.loads(Path(config_path).read_text())
    root = cfg.get("project_root")
    if root and root not in sys.path:
        sys.path.insert(0, root)

    session = _session_from_launch_config(cfg)
    _session, win = build_ui(session)
    win.run()
    try:
        Path(config_path).unlink(missing_ok=True)
    except Exception:
        pass


def launch_app_inplace(session: Optional[Session] = None):
    """
    Open the label UI **inside** this process (for plain Python / IPython, not Cursor Jupyter).

    Prefer ``launch_app`` from notebooks — it uses a subprocess so the kernel does not crash.
    """
    session, win = build_ui(session)
    win.run()
    return session

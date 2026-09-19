"""Interactive lineup window (subprocess Qt, same idea as LabelWindow)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

from pyPetrograph.align.io import load_scene, save_scene
from pyPetrograph.align.register import auto_register, fit_click_pairs, refine_register
from pyPetrograph.align.scene import Affine2D, Scene
from pyPetrograph.common.constants import DEFAULT_ALIGN_FADE
from pyPetrograph.image_processing.features import _downsample_for_view, resolve_view_target_mp


def _scene_to_cfg(scene: Scene) -> Dict[str, Any]:
    from pyPetrograph.align.io import scene_to_dict

    cfg = scene_to_dict(scene)
    cfg["project_root"] = str(Path(__file__).resolve().parent.parent.parent)
    if scene.json_path is not None:
        cfg["json_path"] = str(scene.json_path)
    return cfg


def _scene_from_cfg(cfg: Dict[str, Any]) -> Scene:
    from pyPetrograph.align.scene import Affine2D, Layer

    scene = Scene(
        working_slot=str(cfg.get("working_slot") or "ppl"),
        point_tolerance_px=float(cfg.get("point_tolerance_px", 8.0)),
    )
    sr = cfg.get("scale_range") or [0.5, 2.0]
    scene.scale_range = (float(sr[0]), float(sr[1]))
    scene.click_pairs = list(cfg.get("click_pairs") or [])
    if cfg.get("json_path"):
        scene.json_path = Path(cfg["json_path"])
    for item in cfg.get("layers") or []:
        angle = item.get("angle_deg")
        scene.layers[str(item["slot"])] = Layer(
            slot=str(item["slot"]),
            path=Path(item["path"]),
            transform=Affine2D.from_dict(item.get("transform") or {}),
            angle_deg=None if angle is None else float(angle),
        )
    return scene


class AlignWindow:
    """Overlay working image + one moving layer; auto first, then drag/scale/re-auto."""

    def __init__(self, scene: Scene, overlay_slot: Optional[str] = None):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button, Slider, TextBox

        self.plt = plt
        self.scene = scene
        slots = [s for s in scene.layers if s != scene.working_slot]
        self.overlay_slot = overlay_slot or (slots[0] if slots else scene.working_slot)
        self.fade = DEFAULT_ALIGN_FADE
        self.click_mode = False
        self._pending_base: Optional[List[float]] = None
        self.drag = None
        self.updating_view = False
        self.overview = None
        self.image_artist = None

        self.fig = plt.figure(figsize=(14, 9))
        try:
            self.fig.canvas.manager.set_window_title("Petrography — align")
        except Exception:
            pass

        self.ax_panel = self.fig.add_axes([0.015, 0.20, 0.21, 0.76])
        self.ax_panel.set_facecolor("#d8d8d8")
        self.ax_panel.set_xticks([])
        self.ax_panel.set_yticks([])
        self.ax = self.fig.add_axes([0.26, 0.04, 0.72, 0.92])
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.minimap_ax = self.fig.add_axes([0.015, 0.03, 0.21, 0.15])
        self.minimap_ax.set_facecolor("#222222")
        self.minimap_ax.set_xticks([])
        self.minimap_ax.set_yticks([])
        self.minimap_rect = None

        x0, w = 0.03, 0.18
        y = 0.92
        h_btn = 0.035
        gap = 0.008

        def _btn(label: str):
            nonlocal y
            y -= h_btn
            b = Button(self.fig.add_axes([x0, y, w, h_btn]), label)
            y -= gap
            return b

        self.btn_auto = _btn("Auto align")
        self.btn_reauto = _btn("Re-auto in view")
        self.btn_save = _btn("Save lineup")
        self.btn_load = _btn("Reload")
        self.btn_cycle = _btn("Cycle overlay")
        self.btn_click = _btn("Click hints: off")
        self.btn_fit = _btn("Fit view")

        y -= 0.01
        y -= 0.032
        self.slider_fade = Slider(
            self.fig.add_axes([x0, y, w, 0.03]), "fade", 0.0, 1.0, valinit=self.fade
        )
        y -= gap + 0.008
        y -= 0.032
        t0 = scene.layers[self.overlay_slot].transform if self.overlay_slot in scene.layers else Affine2D()
        self.box_scale = TextBox(self.fig.add_axes([x0, y, w, 0.03]), "scale ", initial=f"{t0.scale:.4f}")
        y -= gap + 0.032
        self.box_tol = TextBox(
            self.fig.add_axes([x0, y, w, 0.03]),
            "tol px ",
            initial=f"{scene.point_tolerance_px:.1f}",
        )
        y -= gap + 0.032
        self.box_smin = TextBox(
            self.fig.add_axes([x0, y, w, 0.03]),
            "s min ",
            initial=f"{scene.scale_range[0]:.2f}",
        )
        y -= gap + 0.032
        self.box_smax = TextBox(
            self.fig.add_axes([x0, y, w, 0.03]),
            "s max ",
            initial=f"{scene.scale_range[1]:.2f}",
        )
        y -= gap + 0.01
        info_bottom = 0.21
        info_h = max(0.08, y - info_bottom)
        self.ax_info = self.fig.add_axes([x0, info_bottom, w, info_h])
        self.ax_info.set_facecolor("#c8c8c8")
        self.ax_info.set_xticks([])
        self.ax_info.set_yticks([])
        self.info_text = self.ax_info.text(
            0.04, 0.96, "", transform=self.ax_info.transAxes, va="top", ha="left", fontsize=8
        )

        self.btn_auto.on_clicked(lambda _e: self.do_auto())
        self.btn_reauto.on_clicked(lambda _e: self.do_reauto())
        self.btn_save.on_clicked(lambda _e: self.do_save())
        self.btn_load.on_clicked(lambda _e: self.do_load())
        self.btn_cycle.on_clicked(lambda _e: self.do_cycle())
        self.btn_click.on_clicked(lambda _e: self.do_toggle_click())
        self.btn_fit.on_clicked(lambda _e: self.do_fit())
        self.slider_fade.on_changed(self._on_fade)
        self.box_scale.on_submit(self._on_scale)
        self.box_tol.on_submit(self._on_tol)
        self.box_smin.on_submit(self._on_srange)
        self.box_smax.on_submit(self._on_srange)

        self.fig.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.fig.canvas.mpl_connect("button_press_event", self._on_press)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)

        self._show()

    def _source_rgb(self) -> np.ndarray:
        if self.overlay_slot == self.scene.working_slot:
            return self.scene.working().ensure_rgb()
        return self.scene.composite(self.overlay_slot, fade=self.fade)

    def _set_info(self, extra: str = "") -> None:
        t = self.scene.layers[self.overlay_slot].transform
        lines = [
            f"base: {self.scene.working_slot}",
            f"over: {self.overlay_slot}",
            f"s={t.scale:.4f}",
            f"tx={t.tx:.1f} ty={t.ty:.1f}",
            f"rot={t.rotation_deg:.2f}",
            f"tol={self.scene.point_tolerance_px:.1f}px",
            extra[:40] if extra else "wheel=zoom; drag=pan",
            "shift+wheel=scale overlay",
        ]
        self.info_text.set_text("\n".join(lines[:9]))
        self.fig.canvas.draw_idle()

    def _show(self) -> None:
        img = self._source_rgb()
        h, w = img.shape[:2]
        mp = resolve_view_target_mp(None)
        self.overview = _downsample_for_view(img, target_mp=mp)
        if self.image_artist is None:
            self.image_artist = self.ax.imshow(
                self.overview, extent=(0, w, h, 0), interpolation="nearest"
            )
        else:
            self.image_artist.set_data(self.overview)
            self.image_artist.set_extent((0, w, h, 0))
        self.ax.set_xlim(0, w)
        self.ax.set_ylim(h, 0)
        self.minimap_ax.clear()
        self.minimap_ax.imshow(self.overview, extent=(0, w, h, 0), interpolation="nearest")
        self.minimap_ax.set_xlim(0, w)
        self.minimap_ax.set_ylim(h, 0)
        self.minimap_ax.set_xticks([])
        self.minimap_ax.set_yticks([])
        from matplotlib.patches import Rectangle

        self.minimap_rect = Rectangle((0, 0), w, h, fill=False, edgecolor="yellow", linewidth=1.2)
        self.minimap_ax.add_patch(self.minimap_rect)
        self._sync_scale_box()
        self._set_info("ready")
        self.fig.canvas.draw_idle()

    def _refresh_overlay(self) -> None:
        img = self._source_rgb()
        h, w = img.shape[:2]
        mp = resolve_view_target_mp(None)
        self.overview = _downsample_for_view(img, target_mp=mp)
        x0, x1 = self.ax.get_xlim()
        y1, y0 = self.ax.get_ylim()
        self.image_artist.set_data(self.overview)
        self.image_artist.set_extent((0, w, h, 0))
        self.ax.set_xlim(x0, x1)
        self.ax.set_ylim(y1, y0)
        self._sync_scale_box()
        self._set_info()
        self.fig.canvas.draw_idle()

    def _sync_scale_box(self) -> None:
        t = self.scene.layers[self.overlay_slot].transform
        self.box_scale.set_val(f"{t.scale:.4f}")

    def _on_fade(self, val: float) -> None:
        self.fade = float(val)
        self._refresh_overlay()

    def _on_scale(self, text: str) -> None:
        try:
            sc = float(text)
        except ValueError:
            return
        self.scene.layers[self.overlay_slot].transform.scale = sc
        self._refresh_overlay()

    def _on_tol(self, text: str) -> None:
        try:
            self.scene.point_tolerance_px = max(0.0, float(text))
        except ValueError:
            return
        self._set_info("tolerance set")

    def _on_srange(self, _text: str) -> None:
        try:
            lo, hi = float(self.box_smin.text), float(self.box_smax.text)
            if hi > lo > 0:
                self.scene.scale_range = (lo, hi)
        except ValueError:
            return
        self._set_info("scale range set")

    def do_auto(self) -> None:
        if self.overlay_slot == self.scene.working_slot:
            self._set_info("pick an overlay layer")
            return
        self._set_info("auto…")
        self.fig.canvas.draw_idle()
        base = self.scene.working().ensure_rgb()
        mov = self.scene.layers[self.overlay_slot].ensure_rgb()
        cur = self.scene.layers[self.overlay_slot].transform
        fitted = fit_click_pairs(self.scene.click_pairs, tolerance_px=self.scene.point_tolerance_px)
        hint = fitted if fitted is not None else cur
        tform = auto_register(
            base,
            mov,
            scale_hint=hint.scale,
            scale_range=self.scene.scale_range,
            tx_hint=hint.tx,
            ty_hint=hint.ty,
        )
        self.scene.layers[self.overlay_slot].transform = tform
        self._refresh_overlay()
        self._set_info("auto done (tweak if needed)")

    def do_reauto(self) -> None:
        if self.overlay_slot == self.scene.working_slot:
            return
        x0, x1 = self.ax.get_xlim()
        y1, y0 = self.ax.get_ylim()
        r0, r1 = int(min(y0, y1)), int(max(y0, y1))
        c0, c1 = int(min(x0, x1)), int(max(x0, x1))
        self._set_info("re-auto in view…")
        refine_register(self.scene, self.overlay_slot, neighborhood=(r0, r1, c0, c1))
        self._refresh_overlay()
        self._set_info("re-auto done")

    def do_save(self) -> None:
        path = save_scene(self.scene)
        self._set_info(f"saved {path.name}")

    def do_load(self) -> None:
        if self.scene.json_path and self.scene.json_path.exists():
            loaded = load_scene(self.scene.json_path)
            self.scene.layers = loaded.layers
            self.scene.working_slot = loaded.working_slot
            self.scene.point_tolerance_px = loaded.point_tolerance_px
            self.scene.scale_range = loaded.scale_range
            self.scene.click_pairs = loaded.click_pairs
            self._refresh_overlay()
            self._set_info("reloaded")
        else:
            self._set_info("no saved file yet")

    def do_cycle(self) -> None:
        keys = list(self.scene.layers.keys())
        if len(keys) < 2:
            return
        i = keys.index(self.overlay_slot) if self.overlay_slot in keys else 0
        nxt = keys[(i + 1) % len(keys)]
        if nxt == self.scene.working_slot and len(keys) > 1:
            nxt = keys[(i + 2) % len(keys)]
        self.overlay_slot = nxt
        self._refresh_overlay()

    def do_toggle_click(self) -> None:
        self.click_mode = not self.click_mode
        self._pending_base = None
        self.btn_click.label.set_text("Click hints: on" if self.click_mode else "Click hints: off")
        self._set_info("click base, then overlay" if self.click_mode else "")

    def do_fit(self) -> None:
        img = self._source_rgb()
        h, w = img.shape[:2]
        self.ax.set_xlim(0, w)
        self.ax.set_ylim(h, 0)
        self.fig.canvas.draw_idle()

    def _on_scroll(self, event) -> None:
        if event.inaxes != self.ax:
            return
        key = (event.key or "")
        if "shift" in key:
            t = self.scene.layers[self.overlay_slot].transform
            factor = 1.05 if event.button == "up" else 1 / 1.05
            t.scale = float(np.clip(t.scale * factor, 0.05, 20.0))
            self._refresh_overlay()
            return
        x0, x1 = self.ax.get_xlim()
        y1, y0 = self.ax.get_ylim()
        cx, cy = event.xdata, event.ydata
        if cx is None:
            return
        scale = 0.8 if event.button == "up" else 1.25
        self.ax.set_xlim(cx - (cx - x0) * scale, cx + (x1 - cx) * scale)
        self.ax.set_ylim(cy + (y1 - cy) * scale, cy - (cy - y0) * scale)
        self.fig.canvas.draw_idle()

    def _on_press(self, event) -> None:
        if event.inaxes != self.ax or event.xdata is None:
            return
        if self.click_mode and event.button == 1:
            xy = [float(event.xdata), float(event.ydata)]
            if self._pending_base is None:
                self._pending_base = xy
                self._set_info("now click overlay point")
            else:
                self.scene.click_pairs.append({"base": self._pending_base, "moving": xy})
                self._pending_base = None
                fitted = fit_click_pairs(
                    self.scene.click_pairs, tolerance_px=self.scene.point_tolerance_px
                )
                if fitted is not None:
                    self.scene.layers[self.overlay_slot].transform = fitted
                    self._refresh_overlay()
                self._set_info(f"{len(self.scene.click_pairs)} hint pair(s)")
            return
        if event.button == 1:
            self.drag = (event.xdata, event.ydata, "pan")
        elif event.button == 3:
            self.drag = (event.xdata, event.ydata, "move")

    def _on_release(self, event) -> None:
        self.drag = None

    def _on_motion(self, event) -> None:
        if self.drag is None or event.inaxes != self.ax or event.xdata is None:
            return
        x0, y0, mode = self.drag
        dx, dy = float(event.xdata - x0), float(event.ydata - y0)
        if mode == "pan":
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
            self.ax.set_xlim(xlim[0] - dx, xlim[1] - dx)
            self.ax.set_ylim(ylim[0] - dy, ylim[1] - dy)
            self.drag = (event.xdata, event.ydata, "pan")
            self.fig.canvas.draw_idle()
        elif mode == "move":
            t = self.scene.layers[self.overlay_slot].transform
            t.tx += dx
            t.ty += dy
            self.drag = (event.xdata, event.ydata, "move")
            self._refresh_overlay()

    def run(self) -> None:
        self.plt.show(block=True)


def launch_align(scene: Scene, *, overlay_slot: Optional[str] = None, wait: bool = False) -> Scene:
    """Open lineup UI in a subprocess (kernel stays Qt-free). Always call auto in the window."""
    root = str(Path(__file__).resolve().parent.parent.parent)
    cfg = _scene_to_cfg(scene)
    if overlay_slot:
        cfg["overlay_slot"] = overlay_slot
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix="_petrography_align.json", delete=False, encoding="utf-8"
    )
    with tmp:
        json.dump(cfg, tmp, indent=2)
        cfg_path = tmp.name
    env = os.environ.copy()
    env["MPLBACKEND"] = "QtAgg"
    env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
    code = (
        "from pyPetrograph.align.ui import _run_align_from_config; "
        f"_run_align_from_config({cfg_path!r})"
    )
    cmd = [sys.executable, "-c", code]
    if wait:
        subprocess.call(cmd, env=env, cwd=root)
        if scene.json_path and Path(scene.json_path).exists():
            return load_scene(scene.json_path)
        print("Align window closed.")
    else:
        subprocess.Popen(cmd, env=env, cwd=root)
        print("Align window opened (Qt popup in a separate process).")
        print("Auto runs from the Auto align button (always try auto first). Save there.")
    return scene


def _run_align_from_config(config_path: str) -> None:
    os.environ["MPLBACKEND"] = "QtAgg"
    import matplotlib

    matplotlib.use("QtAgg", force=True)
    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    root = cfg.get("project_root")
    if root and root not in sys.path:
        sys.path.insert(0, root)
    scene = _scene_from_cfg(cfg)
    # Always try auto once on first open (even if it looks bad).
    slots = [s for s in scene.layers if s != scene.working_slot]
    overlay = str(cfg.get("overlay_slot") or (slots[0] if slots else scene.working_slot))
    if overlay != scene.working_slot and overlay in scene.layers:
        try:
            t = scene.layers[overlay].transform
            identity = abs(t.scale - 1.0) < 1e-9 and abs(t.tx) < 1e-9 and abs(t.ty) < 1e-9
            if identity:
                refine_register(scene, overlay)
        except Exception:
            pass
    win = AlignWindow(scene, overlay_slot=overlay)
    win.run()
    try:
        Path(config_path).unlink(missing_ok=True)
    except Exception:
        pass

#!/usr/bin/env python3
"""
AMICS pop-up worker. Subprocess only (matplotlib QtAgg; Qt via matplotlib's qt_compat, PySide6 preferred); never import
in the Jupyter kernel and never import ``pyPetrograph`` here (it imports sibling
``bse.py`` and ``file_match.py`` through ``sys.path`` instead).

``--config`` JSON, by ``mode``:

- ``files``: ``start_dir``, ``legend``, ``bse``, ``mineral`` (prefilled paths or null),
  ``auto`` (keys that were auto-filled) -> ``{"legend", "bse", "mineral"}`` or ``{"cancelled": true}``.
- ``points``: ``image`` (stretched gray .npy), ``valid`` (mask .npy), ``points`` (saved),
  ``radius``, ``pct``, ``footer_row`` -> ``{"points", "threshold"}``.
- ``scale``: ``image`` (.npy), ``saved`` (saved scale dict or null), ``default_length``,
  ``default_unit`` -> scale dict with ``confirmed: true``, or ``{"confirmed": false}``.

All modes also take ``title`` and ``out`` (result .json path).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "1")  # sharp windows on high-DPI screens

import matplotlib  # noqa: E402

if not os.environ.get("MPLBACKEND"):
    matplotlib.use("QtAgg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

from bse import dark_mask, threshold_from_points  # noqa: E402
from file_match import find_companions  # noqa: E402

UNIT_TO_UM: Dict[str, float] = {"µm": 1.0, "um": 1.0, "mm": 1000.0, "nm": 0.001}

_DIALOG_UNITS = ["µm", "mm", "nm"]
_IMAGE_FILTER = "Images (*.tif *.tiff *.TIF *.TIFF *.png *.PNG *.jpg *.JPG *.jpeg *.JPEG);;All files (*)"

POINTS_HELP = (
    "Left click: add | Right click: remove nearest | C: clear all | Enter or close: done | "
    "clicks are ignored while zoom/pan is on"
)
SCALE_HELP = (
    "Click 2 points: draw a new line | Right click: undo new line | Enter: type length | "
    "close window: keep current scale"
)


# ---------------------------------------------------------------- figure helpers
def _make_figure(image: np.ndarray, title: str, help_text: str = "", window_title: str = ""):
    image = np.asarray(image)
    h, w = image.shape[:2]
    fig_h = min(max(10.0 * h / max(w, 1), 4.0), 9.0)
    fig, ax = plt.subplots(figsize=(10.0, fig_h))
    if image.ndim == 2:
        kw: Dict[str, Any] = {"cmap": "gray"}
        if image.dtype == np.uint8:
            kw.update(vmin=0, vmax=255)
        ax.imshow(image, interpolation="nearest", **kw)
    else:
        ax.imshow(image, interpolation="nearest")
    ax.set_autoscale_on(False)
    ax.set_title(title, fontsize=10)
    if help_text:
        fig.subplots_adjust(bottom=0.09)
        fig.text(0.5, 0.012, help_text, ha="center", va="bottom", fontsize=9, color="0.2")
    if window_title:
        try:
            fig.canvas.manager.set_window_title(window_title)
        except Exception:
            pass
    return fig, ax


def _tool_active(fig) -> bool:
    tb = getattr(fig.canvas, "toolbar", None)
    return tb is not None and bool(getattr(tb, "mode", ""))


def _set_title(fig, ax, text: str) -> None:
    ax.set_title(text, fontsize=10)
    fig.canvas.draw_idle()


def _bring_to_front(fig) -> None:
    """Show the window on top (otherwise it can open behind the notebook)."""
    try:
        win = fig.canvas.manager.window
        win.show()
        win.raise_()
        win.activateWindow()
    except Exception:
        pass


def _show(fig) -> None:
    _bring_to_front(fig)
    plt.show()


# ---------------------------------------------------------------- points session
def build_points_session(
    image: np.ndarray,
    title: str,
    valid: Optional[np.ndarray] = None,
    points: Sequence[Tuple[float, float]] = (),
    radius: int = 5,
    pct: float = 99.0,
    footer_row: Optional[int] = None,
    window_title: str = "",
):
    """
    Return ``(fig, ax, state)``. ``state['points']`` is a list of ``(x, y)`` (starts with the
    saved ``points``); ``state['threshold']`` is the live threshold (None with no points).
    The blue tint marks pixels darker than the threshold and updates after every change.
    """
    gray = np.asarray(image)
    valid_arr = np.ones(gray.shape[:2], dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    if "c" in plt.rcParams["keymap.back"]:
        plt.rcParams["keymap.back"].remove("c")  # C clears points here

    fig, ax = _make_figure(gray, title, POINTS_HELP, window_title)
    if footer_row is not None:
        ax.axhline(footer_row, color="red", linestyle="--", linewidth=0.8)
    tint = ax.imshow(
        np.zeros(gray.shape[:2], dtype=np.uint8),
        cmap=ListedColormap([(0, 0, 0, 0), (0.1, 0.4, 1.0, 0.55)]),
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    (marks,) = ax.plot([], [], "x", color="red", markersize=9)
    state: Dict[str, Any] = {"points": [(float(x), float(y)) for x, y in points], "threshold": None}

    def refresh():
        pts = state["points"]
        marks.set_data([p[0] for p in pts], [p[1] for p in pts])
        state["threshold"] = None
        status = "no points yet: click dark epoxy / holes"
        if pts:
            try:
                thr = threshold_from_points(gray, pts, valid_arr, radius, pct)
            except ValueError:
                status = f"{len(pts)} points, but no valid pixels near them"
            else:
                dark = dark_mask(gray, thr, valid_arr)
                state["threshold"] = thr
                share = 100.0 * float(dark[valid_arr].mean()) if valid_arr.any() else 0.0
                status = f"threshold {thr:.0f} | dark {share:.0f}% | {len(pts)} points"
                tint.set_data(dark.astype(np.uint8))
        if state["threshold"] is None:
            tint.set_data(np.zeros(gray.shape[:2], dtype=np.uint8))
        ax.set_title(f"{title}\n{status}", fontsize=10)
        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes is not ax or event.xdata is None or event.ydata is None:
            return
        if _tool_active(fig):
            return
        pts = state["points"]
        if event.button == 1:
            pts.append((float(event.xdata), float(event.ydata)))
        elif event.button == 3:
            if not pts:
                return
            d = [(p[0] - event.xdata) ** 2 + (p[1] - event.ydata) ** 2 for p in pts]
            pts.pop(int(np.argmin(d)))
        else:
            return
        refresh()

    def on_key(event):
        if event.key == "enter":
            plt.close(fig)
        elif event.key in ("c", "C"):
            state["points"].clear()
            refresh()

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    refresh()
    return fig, ax, state


# ---------------------------------------------------------------- Qt helpers
_APP: Any = None  # keep a reference: Qt destroys a QApplication that is garbage collected


def _qt_app():
    global _APP
    from matplotlib.backends.qt_compat import QtWidgets

    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv[:1])
    _APP = app
    return app


def _front(widget) -> None:
    widget.show()
    widget.raise_()
    widget.activateWindow()


def _qt_ask_length(default_length: float, default_unit: str) -> Optional[Tuple[float, str]]:
    from matplotlib.backends.qt_compat import QtCore, QtWidgets

    _qt_app()
    dlg = QtWidgets.QDialog()
    dlg.setWindowTitle("Scale length")
    layout = QtWidgets.QVBoxLayout(dlg)
    layout.addWidget(QtWidgets.QLabel("How long is the line you drew?"))
    row = QtWidgets.QHBoxLayout()
    spin = QtWidgets.QDoubleSpinBox()
    spin.setRange(0.0001, 1e9)
    spin.setDecimals(4)
    spin.setValue(float(default_length))
    combo = QtWidgets.QComboBox()
    combo.addItems(_DIALOG_UNITS)
    unit = "µm" if default_unit == "um" else default_unit
    combo.setCurrentIndex(_DIALOG_UNITS.index(unit) if unit in _DIALOG_UNITS else 0)
    row.addWidget(spin)
    row.addWidget(combo)
    layout.addLayout(row)
    buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
    buttons.accepted.connect(dlg.accept)
    buttons.rejected.connect(dlg.reject)
    layout.addWidget(buttons)

    spin.setFocus()
    QtCore.QTimer.singleShot(0, spin.selectAll)
    QtCore.QTimer.singleShot(0, spin.lineEdit().selectAll)
    dlg.raise_()
    dlg.activateWindow()

    if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
        return None
    spin.interpretText()
    return float(spin.value()), combo.currentText()


# ---------------------------------------------------------------- scale session
def build_scale_session(
    image: np.ndarray,
    title: str,
    default_length: float,
    default_unit: str,
    ask_length: Optional[Callable[[float, str], Optional[Tuple[float, str]]]] = None,
    saved: Optional[Dict[str, Any]] = None,
    window_title: str = "",
):
    """
    Return ``(fig, ax, state)``; ``state`` holds ``p1``, ``p2`` (new line), ``result``.

    A ``saved`` scale (with ``p1``/``p2``) is drawn as an orange dashed line. Two clicks draw a
    new red line. Enter asks for the length of the new line (or of the saved line if no new one).
    Closing without Enter leaves ``result`` None (keep the current scale).
    """
    if ask_length is None:
        ask_length = _qt_ask_length
    if saved is not None and not (saved.get("p1") and saved.get("p2")):
        saved = None
    fig, ax = _make_figure(image, title, SCALE_HELP, window_title)
    state: Dict[str, Any] = {"p1": None, "p2": None, "result": None}
    artists: List[Any] = []
    saved_artists: List[Any] = []

    if saved is not None:
        (a, b) = saved["p1"], saved["p2"]
        (ln,) = ax.plot([a[0], b[0]], [a[1], b[1]], "--o", color="orange", linewidth=1.5, markersize=5)
        px_saved = math.hypot(b[0] - a[0], b[1] - a[1])
        txt = ax.text(
            (a[0] + b[0]) / 2,
            (a[1] + b[1]) / 2,
            f" saved: {saved.get('length', '?'):g} {saved.get('unit', '')} = {px_saved:.0f} px"
            if isinstance(saved.get("length"), (int, float))
            else f" saved line: {px_saved:.0f} px",
            color="orange",
            fontsize=9,
            va="bottom",
        )
        saved_artists.extend([ln, txt])

    def clear_artists():
        while artists:
            artists.pop().remove()

    def show_saved(flag: bool):
        for art in saved_artists:
            art.set_visible(flag)

    def on_click(event):
        if event.inaxes is not ax or event.xdata is None or event.ydata is None:
            return
        if _tool_active(fig):
            return
        if event.button == 3:
            clear_artists()
            state["p1"] = state["p2"] = None
            show_saved(True)
            ax.set_title(title, fontsize=10)
            fig.canvas.draw_idle()
            return
        if event.button != 1:
            return
        pt = (float(event.xdata), float(event.ydata))
        if state["p1"] is None or state["p2"] is not None:
            clear_artists()
            show_saved(False)
            state["p1"], state["p2"] = pt, None
            (m,) = ax.plot([pt[0]], [pt[1]], "o", color="red", markersize=6)
            artists.append(m)
            ax.set_title(title, fontsize=10)
        else:
            state["p2"] = pt
            p1 = state["p1"]
            (ln,) = ax.plot([p1[0], pt[0]], [p1[1], pt[1]], "-o", color="red", linewidth=1.5, markersize=6)
            artists.append(ln)
            px = math.hypot(pt[0] - p1[0], pt[1] - p1[1])
            ax.set_title(f"scale line: {px:.1f} px - press Enter to type its length", fontsize=10)
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key != "enter":
            return
        p1, p2 = state["p1"], state["p2"]
        if p1 is None or p2 is None:
            if saved is None:
                _set_title(fig, ax, "click 2 points first")
                return
            p1, p2 = tuple(saved["p1"]), tuple(saved["p2"])
        px = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        if px == 0:
            _set_title(fig, ax, "points are identical")
            return
        dl, du = default_length, default_unit
        if saved is not None and isinstance(saved.get("length"), (int, float)) and saved.get("unit"):
            dl, du = float(saved["length"]), saved["unit"]
        ans = ask_length(dl, du)
        if ans is None:
            return
        length, unit = float(ans[0]), ans[1]
        if not (length > 0) or unit not in UNIT_TO_UM:
            _set_title(fig, ax, "invalid length or unit")
            return
        state["result"] = {
            "p1": [p1[0], p1[1]],
            "p2": [p2[0], p2[1]],
            "px_length": px,
            "length": length,
            "unit": unit,
            "um_per_px": length * UNIT_TO_UM[unit] / px,
            "confirmed": True,
        }
        plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    return fig, ax, state


# ---------------------------------------------------------------- files window
_ROWS = (
    ("legend", "Mineral legend", "the colour key image (colours -> mineral names)"),
    ("bse", "BSE image", "the grey backscatter image"),
    ("mineral", "Mineral map", "the coloured mineral map image"),
)


def run_files_dialog(
    start_dir: str,
    legend: Optional[str] = None,
    bse: Optional[str] = None,
    mineral: Optional[str] = None,
    auto: Sequence[str] = (),
) -> Optional[Dict[str, str]]:
    """One window with three labeled rows (path box + Browse). Returns the three paths or None."""
    from matplotlib.backends.qt_compat import QtWidgets

    _qt_app()
    dlg = QtWidgets.QDialog()
    dlg.setWindowTitle("AMICS: choose the three images")
    dlg.setMinimumWidth(760)
    layout = QtWidgets.QVBoxLayout(dlg)
    layout.addWidget(
        QtWidgets.QLabel(
            "Fill all three rows (type a path or click Browse). Picking the BSE image or the mineral map "
            "fills the other rows from the same folder; you can still change any row."
        )
    )
    grid = QtWidgets.QGridLayout()
    layout.addLayout(grid)

    given = {"legend": legend, "bse": bse, "mineral": mineral}
    auto_keys = set(auto)
    edits: Dict[str, Any] = {}
    status: Dict[str, Any] = {}
    ok_btn_holder: List[Any] = []

    def refresh():
        all_ok = True
        for key, edit in edits.items():
            text = edit.text().strip()
            if text and Path(text).expanduser().is_file():
                tag = " (filled automatically)" if key in auto_keys else ""
                status[key].setText(f"found{tag}")
                status[key].setStyleSheet("color: #2a7a2a")
            else:
                status[key].setText("not chosen" if not text else "file not found")
                status[key].setStyleSheet("color: #b02020")
                all_ok = False
        if ok_btn_holder:
            ok_btn_holder[0].setEnabled(all_ok)

    def set_text(key: str, text: str, is_auto: bool) -> None:
        edits[key].setText(text)
        (auto_keys.add if is_auto else auto_keys.discard)(key)

    def on_edited(key: str) -> None:
        auto_keys.discard(key)
        refresh()

    def browse(key: str, label: str, hint: str) -> None:
        cur = edits[key].text().strip()
        folder = Path(cur).expanduser().parent if cur else Path(start_dir)
        if not folder.is_dir():
            folder = Path(start_dir)
        path, _ = QtWidgets.QFileDialog.getOpenFileName(dlg, f"Select the {label.upper()}: {hint}", str(folder), _IMAGE_FILTER)
        if not path:
            return
        set_text(key, path, False)
        if key in ("bse", "mineral"):
            for other, found in find_companions(path, key).items():
                if found is not None and (other in auto_keys or not edits[other].text().strip()):
                    set_text(other, str(found), True)
        refresh()

    for r, (key, label, hint) in enumerate(_ROWS):
        title = QtWidgets.QLabel(f"<b>{label}</b><br><small>{hint}</small>")
        edit = QtWidgets.QLineEdit(str(given[key]) if given.get(key) else "")
        edit.textEdited.connect(lambda _t, k=key: on_edited(k))
        btn = QtWidgets.QPushButton("Browse...")
        btn.clicked.connect(lambda _c=False, k=key, lb=label, h=hint: browse(k, lb, h))
        note = QtWidgets.QLabel("")
        grid.addWidget(title, 2 * r, 0, 2, 1)
        grid.addWidget(edit, 2 * r, 1)
        grid.addWidget(btn, 2 * r, 2)
        grid.addWidget(note, 2 * r + 1, 1)
        edits[key], status[key] = edit, note
    grid.setColumnStretch(1, 1)

    buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
    buttons.accepted.connect(dlg.accept)
    buttons.rejected.connect(dlg.reject)
    layout.addWidget(buttons)
    ok_btn_holder.append(buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok))
    refresh()
    _front(dlg)

    if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
        return None
    return {k: str(Path(e.text().strip()).expanduser()) for k, e in edits.items()}


# ---------------------------------------------------------------- main
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="AMICS pop-up worker")
    ap.add_argument("--config", required=True)
    args = ap.parse_args(argv)

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    mode = cfg.get("mode")
    title = cfg.get("title", "")

    if mode == "files":
        picked = run_files_dialog(
            cfg.get("start_dir") or ".", cfg.get("legend"), cfg.get("bse"), cfg.get("mineral"), cfg.get("auto") or ()
        )
        result: Dict[str, Any] = {"cancelled": True} if picked is None else picked
    elif mode == "points":
        image = np.load(cfg["image"])
        valid = np.load(cfg["valid"]) if cfg.get("valid") else None
        fig, ax, state = build_points_session(
            image,
            title,
            valid=valid,
            points=[tuple(p) for p in cfg.get("points", [])],
            radius=int(cfg.get("radius", 5)),
            pct=float(cfg.get("pct", 99.0)),
            footer_row=cfg.get("footer_row"),
            window_title=cfg.get("window_title", ""),
        )
        _show(fig)
        result = {
            "points": [[float(x), float(y)] for x, y in state["points"]],
            "threshold": state["threshold"],
        }
    elif mode == "scale":
        image = np.load(cfg["image"])
        fig, ax, state = build_scale_session(
            image,
            title,
            float(cfg.get("default_length", 500.0)),
            cfg.get("default_unit", "µm"),
            saved=cfg.get("saved"),
            window_title=cfg.get("window_title", ""),
        )
        _show(fig)
        result = state["result"] if state["result"] is not None else {"confirmed": False}
    else:
        print(f"unknown mode: {mode!r}", file=sys.stderr)
        return 2

    with open(cfg["out"], "w", encoding="utf-8") as f:
        json.dump(result, f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

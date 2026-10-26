# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""
AMICS module: the steps of notebook 4 (pyPetrograph_4_AMICSworkflow) in order.

  1 Files            cell 'files'        pick_amics_files
  2 Legend           cells 'legend', 'LEGEND_EDIT'   read_legend, plot_legend
  3 BSE & footer     cell 'BSE'          load_bse, detect_footer (red dashed footer line)
  4 Non-grain        cell 'threshold'    click_points (blue tint), threshold.json / .png
  5 Scale (optional) cell 'scale'        draw_scale (saved line orange, new line red), scale.json
  6 Grain segmentation  cell 'grain outlines'  SEG + GrainPlot editing; 'Use this grain map' then
                     runs cells 'sizes' .. 'knolling' in the background (figures / tables cached)
  7 Minerals         cells 'minerals', 'EXTRA_LEGEND', 'overlay', 'vote', 'composition'
  8 Grain size       cells 'GSD', 'GSD by mineral'
  9 Outputs          cells 'CSVs', 'knolling'; nothing is written to the export folder before
                     'Export CSV data sheets' / 'Export all figures'
"""
from __future__ import annotations

import math
import shutil
from pathlib import Path
from typing import List, Optional

import numpy as np
from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox, QLabel, QLineEdit,
    QListWidget, QMessageBox, QPushButton, QSpinBox, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from pyPetrograph.app.amics_pipeline import AmicsSession, AmicsSettings
from pyPetrograph.app.calibrate import UNIT_TO_UM, UNITS
from pyPetrograph.app.shell import AppWindow, Step, row, title

# notebook colours (amics/window_worker.py, bse.plot_threshold_preview)
NON_GRAIN_RGBA = (26, 102, 255, 140)   # (0.1, 0.4, 1.0, 0.55) blue tint
PICK = "red"                           # picks: red "x"
FOOTER = "red"                         # footer: red dashed line
SAVED_SCALE = "orange"                 # saved scale line: orange "--o"
NEW_SCALE = "red"                      # new scale line: red "-o"
GRAIN = "yellow"                       # grain outlines (LineCollection colors="yellow")

VIEW = [("Scroll", "zoom in / out"), ("Drag", "pan"), ("F / dbl-click", "fit the whole image")]
ORDER = ["open", "legend", "footer", "nongrain", "scale", "segment", "minerals", "gsd", "outputs"]
# app defaults (same as notebook 4: MIN_GRAIN_PX = 10, DARK_DROP_FRAC = 0.5)
APP_DEFAULTS = dict(min_grain_px=10, dark_drop_frac=0.5)


def confirm(parent, title_: str, text: str) -> bool:
    """Yes / Cancel dialog (tests replace this)."""
    return QMessageBox.question(parent, title_, text, QMessageBox.Yes | QMessageBox.Cancel,
                                QMessageBox.Cancel) == QMessageBox.Yes


def fig_image(fig) -> np.ndarray:
    """Render a matplotlib Figure (Agg) to an RGB array for the canvas, then close it."""
    import matplotlib.pyplot as plt

    fig.canvas.draw()
    a = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return a


def png_image(path) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"))


def hint(text: str = "") -> QLabel:
    lb = QLabel(text, wordWrap=True)
    lb.setProperty("role", "hint")
    return lb


class AmicsModule:
    def __init__(self, win: AppWindow, settings: Optional[AmicsSettings] = None) -> None:
        import matplotlib

        matplotlib.use("Agg")  # figures are drawn into the canvas, never in their own window
        self.win = win
        self.session = AmicsSession(settings or AmicsSettings(**APP_DEFAULTS), log=win.log)
        self.done = set()  # names of finished steps (ORDER)
        self.version = 0   # bumped each time the downstream results are computed again
        self.steps = [OpenStep(self), LegendStep(self), FooterStep(self), NonGrainStep(self), ScaleStep(self),
                      SegmentStep(self), MineralsStep(self), GsdStep(self), OutputsStep(self)]
        for st in self.steps:
            win.add_step(st)
        win.canvas.clicked.connect(self._click)
        win.canvas.keyPressed.connect(self._key)
        win.step_list.currentRowChanged.connect(self._left_step)
        win.go(0)

    def step(self, cls):
        return next(s for s in self.steps if isinstance(s, cls))

    def _left_step(self, _row) -> None:
        seg = self.step(SegmentStep)
        if self.current() is not seg and seg.add.isChecked():
            seg.add.setChecked(False)

    def current(self) -> Step:
        return self.steps[self.win.step_list.currentRow()]

    def _click(self, x, y, b) -> None:
        st = self.current()
        if hasattr(st, "on_click"):
            st.on_click(x, y, b)

    def _key(self, k) -> None:
        st = self.current()
        if hasattr(st, "on_key"):
            st.on_key(k)

    def finish(self, name: str, go_next: bool = False) -> None:
        """Mark ``name`` done; later steps must be run again (their inputs changed)."""
        i = ORDER.index(name)
        self.done -= set(ORDER[i + 1:])
        self.done.add(name)
        self.win.refresh_steps()
        if go_next:
            self.win.go(i + 1)


class _AStep(Step):
    needs = ""  # ORDER name that must be done before this step opens

    def __init__(self, mod: AmicsModule) -> None:
        super().__init__(mod.win)
        self.mod = mod
        self.s = mod.session
        self.lay.setSpacing(10)

    def ready(self) -> bool:
        return not self.needs or self.needs in self.mod.done

    def complete(self) -> bool:
        return ORDER[self.index()] in self.mod.done

    def finish_layout(self) -> None:
        self.lay.addStretch(1)

    def update_state(self) -> None:
        pass

    def canvas_focus(self) -> None:
        self.win.canvas.setFocus()

    def _near(self, pts, x, y, px: float = 9.0) -> Optional[int]:
        """Index of the pick within ``px`` screen pixels of (x, y), if any."""
        if not pts:
            return None
        k = self.win.canvas.transform().m11() or 1.0
        d = [math.hypot(a - x, b - y) for a, b in pts]
        i = int(np.argmin(d))
        return i if d[i] * k <= px else None

    def footer_line(self, label: bool = True) -> None:
        c, F = self.win.canvas, self.s.footer
        w = self.s.bse_gray.shape[1]
        c.set_lines("footer", [[(0, F), (w, F)]] if F < self.s.bse_gray.shape[0] else [], FOOTER, 1.0, dashed=True)
        if label:
            c.set_label("footer_lbl", 4, F + 2, f"footer row {F}", FOOTER)


# 1 ---------------------------------------------------------------------------
class OpenStep(_AStep):
    title = "Files"

    def __init__(self, mod) -> None:
        super().__init__(mod)
        from pyPetrograph.amics.windows import load_last_files
        from pyPetrograph.common.paths import get_output_dir, project_root

        self.lay.addWidget(title("1  Files", "Pick the BSE image; the legend and mineral map are found beside it when the names match (notebook: pick_amics_files)."))
        last = load_last_files()
        self.edits = {}
        box = QFormLayout()
        for key, label in (("bse", "BSE image"), ("mineral", "Mineral map"), ("legend", "Legend")):
            e = QLineEdit(last.get(key, ""))
            b = QPushButton("…", maximumWidth=32)
            b.clicked.connect(lambda _=False, k=key: self._browse(k))
            self.edits[key] = e
            box.addRow(label, row(e, b))
        self.out = QLineEdit(str(get_output_dir() or (project_root() / "outputs")))
        ob = QPushButton("…", maximumWidth=32)
        ob.clicked.connect(self._browse_out)
        box.addRow("Output folder", row(self.out, ob))
        self.sample = QLineEdit(placeholderText="from the BSE file name")
        box.addRow("Sample", self.sample)
        w = QWidget()
        w.setLayout(box)
        self.lay.addWidget(w)
        self.finish_layout()
        self.set_controls(VIEW + [("…", "choose a file")])
        self.add_primary("Open", self.open)

    def _browse(self, key: str) -> None:
        from pyPetrograph.amics.file_match import complete

        start = self.edits[key].text() or self.edits["bse"].text() or str(Path.home())
        p, _ = QFileDialog.getOpenFileName(self.panel, f"Choose {key}", str(Path(start).parent),
                                           "Images (*.tif *.tiff *.png *.jpg *.jpeg *.bmp)")
        if not p:
            return
        self.edits[key].setText(p)
        cur = {k: (e.text() or None) for k, e in self.edits.items()}
        cur[key] = p
        paths, auto = complete(**{k: (v if k == key else (v if Path(v or "").is_file() else None)) for k, v in cur.items()})
        for k in auto:
            if paths.get(k):
                self.edits[k].setText(str(paths[k]))

    def _browse_out(self) -> None:
        d = QFileDialog.getExistingDirectory(self.panel, "Output folder", self.out.text())
        if d:
            self.out.setText(d)

    def open(self) -> None:
        from pyPetrograph import set_output_dir

        set_output_dir(self.out.text().strip() or None)
        e = self.edits
        try:
            self.s.set_files(e["legend"].text(), e["bse"].text(), e["mineral"].text(), self.sample.text().strip() or None)
        except Exception as exc:
            self.win.status.setText(str(exc))
            return
        self.win.log(f"sample {self.s.sample}")
        self.set_state(f"Opened sample {self.s.sample}", "saved")
        self.mod.finish("open", go_next=True)

    def activate(self) -> None:
        p = self.edits["bse"].text()
        if p and Path(p).is_file():
            from pyPetrograph.amics import load_bse

            try:
                self.win.canvas.set_image(load_bse(p)[0])
            except Exception:
                pass


# 2 ---------------------------------------------------------------------------
class LegendEntryDialog(QDialog):
    """Colour (R, G, B, hex, palette, eyedropper) + mineral name for one legend entry."""

    def __init__(self, step, rgb=(128, 128, 128), name: str = "", row_: Optional[int] = None,
                 on_ok=None, title_: str = "Add legend entry") -> None:
        from PySide6.QtWidgets import QDialogButtonBox, QHBoxLayout

        super().__init__(step.panel)
        self.step, self.row, self.on_ok = step, row_, on_ok
        self.setWindowTitle(title_)
        self.setModal(False)  # the eyedropper needs the viewer
        lay = QVBoxLayout(self)
        lay.addWidget(hint("A legend entry is a mineral name and the colour it has in the legend and the "
                           "mineral map. Pick the colour with R, G, B, hex, the palette, or the eyedropper."))
        self.preview = QLabel(minimumHeight=34)
        lay.addWidget(self.preview)
        f = QFormLayout()
        self.r, self.g, self.b = (QSpinBox(maximum=255) for _ in range(3))
        rgbw = QWidget()
        rl = QHBoxLayout(rgbw)
        rl.setContentsMargins(0, 0, 0, 0)
        for lab, sp in (("R", self.r), ("G", self.g), ("B", self.b)):
            rl.addWidget(QLabel(lab))
            rl.addWidget(sp)
        f.addRow("RGB", rgbw)
        self.hex = QLineEdit(maxLength=7)
        f.addRow("Hex", self.hex)
        self.name = QLineEdit(text=name, placeholderText="mineral name, e.g. Ilmenite")
        f.addRow("Mineral", self.name)
        lay.addLayout(f)
        self.palette_btn = QPushButton("Colour palette…")
        self.palette_btn.clicked.connect(self._palette)
        self.drop_btn = QPushButton("Eyedropper: click the image", checkable=True)
        self.drop_btn.setToolTip("Click a pixel of the legend image or the mineral map in the viewer")
        self.drop_btn.toggled.connect(self._dropper)
        lay.addWidget(row(self.palette_btn, self.drop_btn))
        self.err = QLabel(wordWrap=True)
        self.err.setStyleSheet("color: #ff6b6b")
        lay.addWidget(self.err)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.ok)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        for sp in (self.r, self.g, self.b):
            sp.valueChanged.connect(lambda _: self._sync(from_hex=False))
        self.hex.editingFinished.connect(lambda: self._sync(from_hex=True))
        self.finished.connect(lambda _: self.drop_btn.setChecked(False))
        self.set_rgb(rgb)

    def rgb(self):
        return (self.r.value(), self.g.value(), self.b.value())

    def set_rgb(self, rgb) -> None:
        for sp, v in zip((self.r, self.g, self.b), rgb):
            sp.blockSignals(True)
            sp.setValue(int(v))
            sp.blockSignals(False)
        self._sync(from_hex=False)

    def _sync(self, from_hex: bool) -> None:
        if from_hex:
            c = QColor(self.hex.text().strip())
            if c.isValid():
                self.set_rgb((c.red(), c.green(), c.blue()))
            return
        r, g, b = self.rgb()
        self.hex.setText(f"#{r:02x}{g:02x}{b:02x}")
        txt = "#000" if (0.299 * r + 0.587 * g + 0.114 * b) > 140 else "#fff"
        self.preview.setText(f"  RGB {r}, {g}, {b}")
        self.preview.setStyleSheet(f"background: rgb({r},{g},{b}); color: {txt}; border-radius: 4px")

    def _palette(self) -> None:
        from PySide6.QtWidgets import QColorDialog

        c = QColorDialog.getColor(QColor(*self.rgb()), self, "Legend colour")
        if c.isValid():
            self.set_rgb((c.red(), c.green(), c.blue()))

    def _dropper(self, on: bool) -> None:
        self.step._eyedrop = self if on else None
        self.drop_btn.setText("Eyedropper: on (click the image)" if on else "Eyedropper: click the image")
        if on and self.step.view.currentIndex() == 1:
            self.step.view.setCurrentIndex(0)

    def problem(self) -> str:
        name, rgb = self.name.text().strip(), self.rgb()
        if not name:
            return "Type a mineral name."
        for i, (n, c) in enumerate(self.step.rows_now()):
            if i == self.row:
                continue
            if tuple(c) == rgb:
                return f"RGB {rgb} is already used by '{n}'."
            if n == name:
                return f"'{name}' is already in the legend."
        return ""

    def ok(self) -> bool:
        p = self.problem()
        self.err.setText(p)
        if p:
            return False
        if self.on_ok:
            self.on_ok(self.rgb(), self.name.text().strip(), self.row)
        self.accept()
        return True


class LegendStep(_AStep):
    title = "Legend"
    needs = "open"
    COLS = ["Use", "RGB", "Mineral", ""]

    def __init__(self, mod) -> None:
        super().__init__(mod)
        self.lay.addWidget(title("2  Legend", "Names and colours read from the legend image (OCR). Type over a wrong name, add or delete entries, double-click a swatch to change its colour; untick a mineral to skip it (SKIP_NAMES)."))
        self.view = QComboBox()
        self.view.addItems(["Legend image (as read)", "Legend as used (plot_legend)", "Mineral map"])
        self.view.currentIndexChanged.connect(lambda _: self.show_figure())
        self.lay.addWidget(row(QLabel("Show"), self.view))
        from PySide6.QtWidgets import QAbstractItemView, QHeaderView

        t = self.table = QTableWidget(0, 4)
        t.setHorizontalHeaderLabels(self.COLS)
        t.verticalHeader().hide()
        t.setSelectionBehavior(QTableWidget.SelectRows)
        t.setEditTriggers(QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed | QAbstractItemView.AnyKeyPressed)
        hh = t.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.Fixed)
        hh.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(2, QHeaderView.Stretch)
        hh.setSectionResizeMode(3, QHeaderView.Fixed)
        t.setColumnWidth(0, 52)
        t.setColumnWidth(3, 22)
        t.setMinimumHeight(300)
        t.itemChanged.connect(self._changed)
        t.cellDoubleClicked.connect(self._dbl)
        t.setContextMenuPolicy(Qt.CustomContextMenu)
        t.customContextMenuRequested.connect(self._menu)
        self.lay.addWidget(t)
        self.lay.addWidget(hint("\u270E = edited, + = added entry (italic). Right-click a row: Revert to OCR."))
        self.add_btn = QPushButton("+ Add legend entry")
        self.add_btn.clicked.connect(lambda: self.add_dialog())
        self.del_btn = QPushButton("Delete entry")
        self.del_btn.clicked.connect(self.delete_selected)
        self.lay.addWidget(row(self.add_btn, self.del_btn))
        self.undo_btn = QPushButton("Undo")
        self.undo_btn.clicked.connect(self.undo)
        self.revert_btn = QPushButton("Revert to OCR")
        self.revert_btn.setToolTip("Selected rows back to the name and colour OCR read")
        self.revert_btn.clicked.connect(lambda: self.revert(self._sel_rows()))
        self.lay.addWidget(row(self.undo_btn, self.revert_btn))
        self.report = hint()
        self.lay.addWidget(self.report)
        reset, saved, clear = QPushButton("Reset all to OCR"), QPushButton("Load saved"), QPushButton("Clear saved…")
        reset.setToolTip("Drop every edit: names, colours and ticks as read by OCR")
        reset.clicked.connect(self.reset)
        saved.setToolTip("the legend saved with this sample's session")
        saved.clicked.connect(self.load_saved)
        clear.setProperty("danger", True)
        clear.clicked.connect(self.clear_saved)
        self.lay.addWidget(row(reset, saved))
        again = QPushButton("Read again (OCR)")
        again.clicked.connect(self.read)
        self.lay.addWidget(row(again, clear))
        self.finish_layout()
        self.set_controls(VIEW + [("Type / dbl-click name", "edit the name"), ("Dbl-click swatch", "change the colour"),
                                  ("Tick box", "use / skip this mineral"), ("Del", "delete the selected rows"),
                                  ("Ctrl+Z", "undo")])
        self.add_primary("Use this legend", self.apply)
        self._read_for = None
        self._img = None
        self._eyedrop = None
        self.history = []
        self._dialog = None
        t.itemSelectionChanged.connect(self._sel_changed)
        self._sel_changed()

    # -- table rows ---------------------------------------------------------------
    def _sel_rows(self) -> List[int]:
        return sorted({i.row() for i in self.table.selectedIndexes()})

    def _sel_changed(self) -> None:
        on = bool(self._sel_rows())
        self.del_btn.setEnabled(on)
        self.revert_btn.setEnabled(on)
        self.undo_btn.setEnabled(bool(self.history))

    def rows_state(self):
        """[(rgb, name, use, origin)]; origin = (ocr name, ocr rgb) or None for added rows."""
        out = []
        for i in range(self.table.rowCount()):
            sw, nm = self.table.item(i, 0), self.table.item(i, 2)
            if sw is None or nm is None:
                continue
            org = nm.data(Qt.UserRole)
            out.append((tuple(int(c) for c in sw.data(Qt.UserRole)), nm.text(), sw.checkState() == Qt.Checked,
                        None if org is None else (org[0], tuple(org[1]))))
        return out

    def rows_now(self):
        return [(n.strip(), rgb) for rgb, n, _u, _o in self.rows_state()]

    def _set_rows(self, rows) -> None:
        from PySide6.QtGui import QFont

        t = self.table
        t.blockSignals(True)
        t.setRowCount(len(rows))
        for i, (rgb, name, use, origin) in enumerate(rows):
            sw = QTableWidgetItem()
            sw.setBackground(QColor(*rgb))
            sw.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsUserCheckable)
            sw.setCheckState(Qt.Checked if use else Qt.Unchecked)
            sw.setData(Qt.UserRole, list(rgb))
            sw.setToolTip("tick = use; double-click to change the colour")
            rgb_it = QTableWidgetItem(",".join(map(str, rgb)))
            rgb_it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            rgb_it.setToolTip("double-click to change the colour")
            nm = QTableWidgetItem(name)
            nm.setData(Qt.UserRole, None if origin is None else [origin[0], list(origin[1])])
            if origin is None:
                mark, tip = "+", "added entry (not read by OCR)"
            elif origin[0] != name.strip() or tuple(origin[1]) != tuple(rgb):
                what = []
                if origin[0] != name.strip():
                    what.append(f"OCR name '{origin[0]}'")
                if tuple(origin[1]) != tuple(rgb):
                    what.append(f"OCR colour {tuple(origin[1])}")
                mark, tip = "\u270E", "edited; " + ", ".join(what) + " (right-click: Revert to OCR)"
            else:
                mark, tip = "", "as read by OCR"
            f = QFont(nm.font())
            f.setItalic(bool(mark))
            nm.setFont(f)
            nm.setToolTip(tip)
            mk = QTableWidgetItem(mark)
            mk.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            mk.setToolTip(tip)
            mk.setForeground(QColor("#c98a00"))
            mk.setTextAlignment(Qt.AlignCenter)
            for c, it in enumerate((sw, rgb_it, nm, mk)):
                t.setItem(i, c, it)
        t.blockSignals(False)
        self._last = self.rows_state()
        self._sel_changed()
        self.update_state()
        if self.view.currentIndex() == 1:
            self.show_figure()

    def _push(self) -> None:
        self.history.append(self.rows_state())
        self.history = self.history[-50:]
        self._sel_changed()

    def undo(self) -> None:
        if self.history:
            self._set_rows(self.history.pop())

    def _changed(self, item) -> None:
        if item.column() == 2:          # name typed: re-mark the row; the old table goes on the undo stack
            self.history.append(getattr(self, "_last", []))
            self._set_rows(self.rows_state())
        else:
            self._last = self.rows_state()
            self.update_state()
            if self.view.currentIndex() == 1:
                self.show_figure()

    def _dbl(self, r: int, c: int) -> None:
        if c in (0, 1):
            rgb, name, _u, _o = self.rows_state()[r]
            self.add_dialog(rgb=rgb, name=name, row_=r)

    def _menu(self, pos) -> None:
        from PySide6.QtWidgets import QMenu

        r = self.table.rowAt(pos.y())
        if r < 0:
            return
        m = QMenu(self.table)
        m.addAction("Revert to OCR", lambda: self.revert([r]))
        m.addAction("Change colour…", lambda: self._dbl(r, 0))
        m.addAction("Delete entry", lambda: self.delete_rows([r]))
        m.exec(self.table.viewport().mapToGlobal(pos))

    def on_key(self, k) -> None:
        if k == "delete":
            self.delete_selected()
        elif k == "z":
            self.undo()

    # -- edits ----------------------------------------------------------------------
    def add_dialog(self, rgb=None, name: str = "", row_: Optional[int] = None) -> "LegendEntryDialog":
        if rgb is None:
            used = {c for _n, c in self.rows_now()}
            rgb = next((c for c in ((128, 128, 128), (200, 200, 200), (90, 90, 90)) if c not in used), (127, 127, 127))
        if self._dialog is not None:
            self._dialog.close()
        self._dialog = LegendEntryDialog(self, rgb, name, row_, self._dialog_ok,
                                         "Add legend entry" if row_ is None else f"Change legend entry '{name}'")
        self._dialog.show()
        return self._dialog

    def _dialog_ok(self, rgb, name, row_) -> None:
        self._push()
        rows = self.rows_state()
        if row_ is None:
            rows.append((tuple(rgb), name, True, None))
        else:
            _c, _n, use, org = rows[row_]
            rows[row_] = (tuple(rgb), name, use, org)
        self._set_rows(rows)
        if row_ is None:
            self.table.selectRow(len(rows) - 1)

    def delete_rows(self, idx) -> None:
        idx = set(idx)
        if not idx:
            return
        self._push()
        self._set_rows([r for i, r in enumerate(self.rows_state()) if i not in idx])
        self.win.status.setText(f"deleted {len(idx)} legend entr{'y' if len(idx) == 1 else 'ies'} (Undo brings {'it' if len(idx) == 1 else 'them'} back)")

    def delete_selected(self) -> None:
        self.delete_rows(self._sel_rows())

    def revert(self, idx) -> None:
        rows = self.rows_state()
        idx = [i for i in idx if 0 <= i < len(rows) and rows[i][3] is not None]
        if not idx:
            return
        self._push()
        for i in idx:
            _c, _n, use, org = rows[i]
            rows[i] = (tuple(org[1]), org[0], use, org)
        self._set_rows(rows)

    def on_click(self, x, y, b) -> None:
        """Eyedropper: the pixel colour under the click goes into the open dialog."""
        if self._eyedrop is None or self._img is None:
            return
        h, w = self._img.shape[:2]
        xi, yi = int(x), int(y)
        if 0 <= xi < w and 0 <= yi < h:
            self._eyedrop.set_rgb(tuple(int(v) for v in self._img[yi, xi, :3]))
            self.win.status.setText(f"picked RGB {tuple(int(v) for v in self._img[yi, xi, :3])} at ({xi}, {yi})")

    # -- views ----------------------------------------------------------------------
    def activate(self) -> None:
        if self.s.legend_file is None:
            return
        self.show_figure()
        if self._read_for != self.s.legend_file:
            self.read()

    def show_figure(self) -> None:
        if self.s.legend_file is None:
            return
        i = self.view.currentIndex()
        if i == 0:
            self._img = png_image(self.s.legend_file)
            self.win.canvas.set_image(self._img)
            return
        if i == 2:
            self._img = png_image(self.s.mineral_file) if self.s.mineral_file else None
            if self._img is not None:
                self.win.canvas.set_image(self._img)
            return
        from pyPetrograph.amics import plot_legend

        self._img = None
        legend, skip = self.edited()
        if legend:
            self.win.canvas.set_image(fig_image(plot_legend(legend, [n for n in self.s.s.skip_names if n in legend] + skip)))

    def read(self) -> None:
        self._read_for = self.s.legend_file
        self.win.jobs.run("Reading legend (OCR)", self.s.read_legend, self._read_done)

    def _read_done(self, legend) -> None:
        """OCR result; a legend saved with the session (your edits) is shown instead when there is one."""
        self.report.setText("\n".join(self.s.legend_report[-6:]))
        self.history = []
        sv = self.s.saved()
        if sv.get("legend") and dict(sv["legend"]) != dict(legend):
            self._fill_saved(sv)
            self.win.status.setText("showing your saved legend edits (Reset all to OCR drops them)")
        else:
            self._fill(legend, skip=(sv.get("legend_edit") or {}).get("skip"))
        self.show_figure()

    def _ocr_rows(self):
        return list((self.s.legend_auto or {}).items())

    def _fill(self, legend, skip=None) -> None:
        """Rows from a legend dict; each row is matched to its OCR row (same name, else same colour)."""
        skip = set(self.s.s.skip_names if skip is None else skip)
        ocr = self._ocr_rows()
        by_name = {n: (n, tuple(c)) for n, c in ocr}
        by_rgb = {tuple(c): (n, tuple(c)) for n, c in ocr}
        used, rows = set(), []
        for name, rgb in dict(legend).items():
            rgb = tuple(int(c) for c in rgb)
            org = by_name.get(name) or by_rgb.get(rgb)
            if org is not None and org in used:
                org = None
            if org is not None:
                used.add(org)
            rows.append((rgb, name, name not in skip, org))
        self._set_rows(rows)

    def _fill_saved(self, sv) -> None:
        ed = sv.get("legend_edit") or {}
        self._fill(sv["legend"], skip=ed.get("skip"))

    def reset(self) -> None:
        if self.s.legend_auto:
            self._push()
            self._fill(self.s.legend_auto)

    def load_saved(self) -> None:
        sv = self.s.saved()
        if "legend" not in sv:
            self.win.status.setText("no saved legend for this sample")
            return
        self._push()
        self._fill_saved(sv)
        self.win.status.setText(f"loaded {len(sv['legend'])} legend rows from {Path(sv['folder']).name}/")

    def clear_saved(self) -> None:
        if not confirm(self.panel, "Clear saved legend", "Delete the saved legend (and its edits) for this sample? The table stays as it is."):
            return
        self.s.clear_saved("legend")
        self.update_state()

    def update_state(self) -> None:
        sv = self.s.saved()
        legend, _skip = self.edited()
        n_ed = sum(1 for rgb, n, _u, o in self.rows_state() if o is None or o[0] != n.strip() or tuple(o[1]) != rgb)
        n_del = max(0, len(self._ocr_rows()) - sum(1 for r in self.rows_state() if r[3] is not None))
        ed = f" ({n_ed} edited / added, {n_del} deleted)" if (n_ed or n_del) else ""
        if "legend" in sv and dict(sv["legend"]) == legend:
            self.set_state(f"Saved: {len(legend)} names{ed}", "saved")
        elif "legend" in sv:
            self.set_state(f"Saved legend differs from the table ({len(legend)} names{ed}): not saved yet", "unsaved")
        elif legend:
            self.set_state(f"{len(legend)} names{ed}: not saved yet", "unsaved")
        else:
            self.set_state("")

    def edited(self):
        legend, skip = {}, []
        for rgb, name, use, _o in self.rows_state():
            name = name.strip()
            if not name or name in legend:
                continue
            legend[name] = rgb
            if not use:
                skip.append(name)
        return legend, skip

    def apply(self) -> None:
        legend, skip = self.edited()
        if not legend:
            self.win.status.setText("no legend rows")
            return
        rgbs = list(legend.values())
        if len(set(rgbs)) != len(rgbs):
            self.win.status.setText("two legend entries have the same colour: change one first")
            return
        # LEGEND_EDIT = None when nothing changed (same as the notebook default)
        self.s.set_legend(None if legend == dict(self.s.legend_auto) else legend)
        default = AmicsSettings().skip_names
        if set(skip) != {n for n in default if n in legend}:
            self.s.s.skip_names = [n for n in default if n not in legend] + skip
        self.s.save_legend_edit(legend, skip, self._ocr_rows())
        self.win.jobs.run("Loading BSE", self.s.prepare_bse,
                          lambda _: (self.update_state(), self.mod.finish("legend", go_next=True)))


# 3 ---------------------------------------------------------------------------
class FooterStep(_AStep):
    title = "BSE & footer"
    needs = "legend"

    def __init__(self, mod) -> None:
        super().__init__(mod)
        self.lay.addWidget(title("3  BSE image and footer", "The red dashed line is where the data ends and the scale-bar footer starts (FOOTER_ROW). Rows below it are left out."))
        f = QFormLayout()
        self.row_spin = QSpinBox(maximum=100000)
        self.row_spin.valueChanged.connect(lambda _: self._preview())
        self.detected = hint()
        use = QPushButton("Use detected")
        use.clicked.connect(lambda: self.row_spin.setValue(self.s.detected_footer()))
        f.addRow("Footer row", self.row_spin)
        f.addRow(self.detected, use)
        w = QWidget()
        w.setLayout(f)
        self.lay.addWidget(w)
        self.finish_layout()
        self.set_controls(VIEW + [("Footer row", "move the red dashed line")])
        self.add_primary("Use this footer", self.apply)

    def activate(self) -> None:
        if self.s.bse_gray is None:
            return
        self.win.canvas.set_image(self.s.bse_gray)
        self.row_spin.blockSignals(True)
        self.row_spin.setMaximum(self.s.bse_gray.shape[0])
        self.row_spin.setValue(self.s.footer)
        self.row_spin.blockSignals(False)
        self.detected.setText(f"detected: row {self.s.detected_footer()}")
        self._preview()

    def _preview(self) -> None:
        c, F = self.win.canvas, self.row_spin.value()
        h, w = self.s.bse_gray.shape[:2]
        c.set_lines("footer", [[(0, F), (w, F)]] if F < h else [], FOOTER, 1.0, dashed=True)
        c.set_label("footer_lbl", 4, F + 2, f"{self.s.sample}  footer row {F}", FOOTER)
        changed = F != self.s.footer
        self.set_state(f"Footer row {self.s.footer}" + (f" → {F}: not applied" if changed else ""),
                       "unsaved" if changed else ("saved" if self.complete() else "none"))

    def apply(self) -> None:
        F = self.row_spin.value()
        if F != self.s.footer:
            self.s.set_footer(None if F == self.s.detected_footer() else F)
        self.mod.finish("footer", go_next=True)


# 4 ---------------------------------------------------------------------------
class NonGrainStep(_AStep):
    title = "Non-grain regions"
    needs = "footer"

    def __init__(self, mod) -> None:
        super().__init__(mod)
        self.lay.addWidget(title(
            "4  Non-grain regions",
            "Click on epoxy, holes or empty space between grains. Pixels with a gray level like your picks are tinted "
            "blue and are not counted as grain. A few picks in different places are enough."))
        self.show_tint = QCheckBox("Show blue tint (T)", checked=True)
        self.show_tint.toggled.connect(lambda _: self._preview())
        self.lay.addWidget(self.show_tint)
        g = QGroupBox("Threshold")
        f = QFormLayout(g)
        self.thr = QDoubleSpinBox(decimals=1, maximum=255.0, singleStep=1.0, specialValueText="from picks")
        self.thr.setMinimum(-1)
        self.thr.setValue(-1)
        self.thr.setToolTip("Optional: type a gray level instead of picking (BSE_THRESHOLD)")
        self.thr.setMinimumWidth(120)
        self.thr.valueChanged.connect(lambda _: self._preview())
        self.thr_info = hint()
        f.addRow("Gray level", self.thr)
        f.addRow(self.thr_info)
        self.lay.addWidget(g)
        self.undo_btn, self.reset_btn = QPushButton("Undo last pick"), QPushButton("Reset picks")
        self.undo_btn.clicked.connect(self.undo)
        self.reset_btn.clicked.connect(self.reset)
        self.lay.addWidget(row(self.undo_btn, self.reset_btn))
        load, clear = QPushButton("Load saved"), QPushButton("Clear saved…")
        load.clicked.connect(lambda: self.load_saved(force=True))
        clear.setProperty("danger", True)
        clear.clicked.connect(self.clear_saved)
        self.lay.addWidget(row(load, clear))
        self.finish_layout()
        self.set_controls([
            ("Left click", "add a pick (red x) on epoxy / hole / empty space"),
            ("Click a pick", "remove that pick"),
            ("Right click", "remove the nearest pick"),
            ("Z", "undo last pick"),
            ("C", "reset: clear all picks"),
            ("T", "show / hide the blue tint"),
            ("Enter", "apply"),
            ("Scroll · Drag", "zoom · pan  (F: fit)"),
        ])
        self.add_primary("Apply non-grain regions", self.apply)
        self.points: List = []
        self._filled_for = None

    def load_saved(self, force: bool = False) -> bool:
        """Start from threshold.json (once per sample, or on demand)."""
        key = (str(self.s.bse_file), str(self.s.P.get("folder")))
        if not force and (self._filled_for == key or self.points or self.thr.value() >= 0):
            return False
        self._filled_for = key
        sv = self.s.saved()
        if "threshold" not in sv:
            if force:
                self.win.status.setText("nothing saved for this sample yet")
            self._preview()
            return False
        self.points = list(sv.get("points") or [])
        self.thr.setValue(-1 if self.points else sv["threshold"])
        self._preview()
        return True

    def clear_saved(self) -> None:
        if not confirm(self.panel, "Clear saved non-grain regions",
                       "Delete threshold.json and threshold.png for this sample and clear the picks?"):
            return
        self.s.clear_saved("threshold")
        self.points = []
        self.thr.setValue(-1)
        self.mod.done -= set(ORDER[ORDER.index("nongrain"):])
        self.win.refresh_steps()
        self._preview()

    def activate(self) -> None:
        if self.s.bse_stretch is None:
            return
        self.win.canvas.set_image(self.s.bse_stretch, fit=False)
        self.win.canvas.fit()
        self.load_saved()
        self._preview()
        self.canvas_focus()

    def on_click(self, x, y, b) -> None:
        if b == Qt.LeftButton:
            i = self._near(self.points, x, y)
            if i is None:
                self.points.append((float(x), float(y)))
            else:
                self.points.pop(i)  # click on a pick: deselect it
        elif b == Qt.RightButton and self.points:
            d = [(px - x) ** 2 + (py - y) ** 2 for px, py in self.points]
            self.points.pop(int(np.argmin(d)))
        self._preview()

    def on_key(self, k) -> None:
        if k == "z":
            self.undo()
        elif k == "c":
            self.reset()
        elif k == "t":
            self.show_tint.toggle()
        elif k == "enter":
            self.apply()

    def undo(self) -> None:
        if self.points:
            self.points.pop()
            self._preview()

    def reset(self) -> None:
        self.points = []
        self.thr.setValue(-1)
        self._preview()

    def current_threshold(self) -> Optional[float]:
        if self.thr.value() >= 0:
            return float(self.thr.value())
        if self.points:
            return self.s.threshold_from(self.points)
        return None

    def _preview(self) -> None:
        if self.s.bse_stretch is None:
            return
        c = self.win.canvas
        c.set_crosses("picks", self.points, PICK, 9)
        self.footer_line(label=False)
        try:
            t = self.current_threshold()
        except ValueError:
            t = None
        tint = self.show_tint.isChecked()
        if t is None:
            c.set_mask(~self.s.bse_valid if tint else None, rgba=NON_GRAIN_RGBA)
            self.thr_info.setText("no picks yet")
        else:
            m = self.s.preview_dark(t)
            c.set_mask(m if tint else None, rgba=NON_GRAIN_RGBA)
            self.thr_info.setText(f"gray level {t:.1f}; non-grain (blue) {100 * float(m.mean()):.1f}% of the image")
        self.undo_btn.setEnabled(bool(self.points))
        self.reset_btn.setEnabled(bool(self.points) or self.thr.value() >= 0)
        self.update_state()

    def is_saved(self) -> bool:
        sv = self.s.saved()
        if "threshold" not in sv:
            return False
        if self.thr.value() >= 0:
            return not sv.get("points") and abs(sv["threshold"] - self.thr.value()) < 1e-9
        return [tuple(map(float, p)) for p in sv.get("points") or []] == [tuple(map(float, p)) for p in self.points]

    def update_state(self) -> None:
        sv = self.s.saved()
        n = len(self.points)
        if "threshold" in sv and self.is_saved():
            n_saved = len(sv.get("points") or [])
            how = f"from {n_saved} picks" if n_saved else "typed"
            self.set_state(f"Saved: gray level {sv['threshold']:.1f} {how} "
                           "(threshold.json)" + ("" if self.complete() else ". Press Apply to use it."), "saved")
        elif n or self.thr.value() >= 0:
            self.set_state(f"{n} picks: not applied / not saved yet", "unsaved")
        else:
            self.set_state("No picks yet: click on epoxy or holes", "none")

    def apply(self) -> None:
        thr = self.thr.value() if self.thr.value() >= 0 else None
        try:
            self.s.set_threshold(thr, self.points)
        except ValueError as exc:
            self.win.status.setText(str(exc))
            return
        self.mod.finish("nongrain", go_next=True)
        self.update_state()


# 5 ---------------------------------------------------------------------------
class ScaleStep(_AStep):
    title = "Scale"
    optional = True
    needs = "nongrain"

    def __init__(self, mod) -> None:
        super().__init__(mod)
        st = mod.session.s
        self.lay.addWidget(title(
            "5  Scale (optional)",
            "The view opens on the footer strip with the scale bar. Click both ends of the scale bar, "
            "then enter its length. Orange dashed = saved line, red = new line. Skip to keep the saved "
            "scale (or sizes in pixels when there is none)."))
        self.zoom_btn = QPushButton("Show whole image (B)")
        self.zoom_btn.clicked.connect(self.toggle_zoom)
        self.lay.addWidget(self.zoom_btn)
        g = QGroupBox("Scale bar")
        f = QFormLayout(g)
        self.length = QDoubleSpinBox(decimals=3, maximum=1e9, value=float(st.scale_length))
        self.unit = QComboBox()
        self.unit.addItems(UNITS)
        self.unit.setCurrentText(st.scale_unit if st.scale_unit in UNITS else "µm")
        self.length.valueChanged.connect(lambda _: self._preview())
        self.unit.currentTextChanged.connect(lambda _: self._preview())
        self.um = QLineEdit(placeholderText="optional: type µm per pixel instead")
        self.um.setToolTip("UM_PER_PX: used as is, not saved (as in the notebook)")
        self.um.textChanged.connect(lambda _: self._preview())
        self.info = hint()
        f.addRow("Bar length", self.length)
        f.addRow("Unit", self.unit)
        f.addRow("µm/px", self.um)
        f.addRow(self.info)
        self.lay.addWidget(g)
        self.undo_btn, self.reset_btn = QPushButton("Undo last point"), QPushButton("Reset line")
        self.undo_btn.clicked.connect(self.undo)
        self.reset_btn.clicked.connect(self.reset)
        self.lay.addWidget(row(self.undo_btn, self.reset_btn))
        ocr, clear = QPushButton("Check footer text (OCR)"), QPushButton("Clear saved…")
        ocr.setToolTip("Reads the pixel size printed in the footer, as a cross-check only")
        ocr.clicked.connect(self._ocr)
        clear.setProperty("danger", True)
        clear.clicked.connect(self.clear_saved)
        self.lay.addWidget(row(ocr, clear))
        self.ocr_info = hint()
        self.lay.addWidget(self.ocr_info)
        self.finish_layout()
        self.set_controls([
            ("Left click ×2", "start and end of the scale bar (red line)"),
            ("Click a point", "remove that end point"),
            ("Right click", "remove the last point"),
            ("Z", "undo last point"),
            ("C", "reset: clear the new line"),
            ("B", "footer strip / whole image"),
            ("Enter", "apply"),
            ("Scroll · Drag", "zoom · pan  (F: fit)"),
        ])
        self.add_primary("Apply scale", self.apply)
        self.pts: List = []
        self.saved_scale = None
        self.whole = False

    # -- view -----------------------------------------------------------------
    def activate(self) -> None:
        if self.s.bse_gray is None:
            return
        sc = self.s.saved().get("scale")
        self.saved_scale = dict(sc) if sc else None
        self.win.canvas.set_image(self.s.bse_gray, fit=False)
        self.whole = False
        self._zoom()
        self.canvas_focus()

    def _zoom(self) -> None:
        h, w = self.s.bse_gray.shape[:2]
        F = min(self.s.footer, h)
        if self.whole:
            self.win.canvas.fit()
        else:
            # bottom-left corner: the footer strip plus a little image above it, filling the view
            strip = h - F if F < h else int(h * 0.1)
            rh = min(h, max(80, 4 * strip))
            vp = self.win.canvas.viewport()
            rw = min(w, rh * max(vp.width(), 1) / max(vp.height(), 1))
            self.win.canvas.zoom_to(0, h - rh, rw, h)
        self.zoom_btn.setText("Show footer strip (B)" if self.whole else "Show whole image (B)")
        self._preview()

    def toggle_zoom(self) -> None:
        self.whole = not self.whole
        self._zoom()

    # -- picks ----------------------------------------------------------------
    def on_click(self, x, y, b) -> None:
        if b == Qt.LeftButton:
            i = self._near(self.pts, x, y)
            if i is not None:
                self.pts.pop(i)
            else:
                if len(self.pts) >= 2:
                    self.pts = []
                self.pts.append((float(x), float(y)))
        elif b == Qt.RightButton and self.pts:
            self.pts.pop()
        self._preview()

    def on_key(self, k) -> None:
        {"z": self.undo, "c": self.reset, "b": self.toggle_zoom, "enter": self.apply}.get(k, lambda: None)()

    def undo(self) -> None:
        if self.pts:
            self.pts.pop()
            self._preview()

    def reset(self) -> None:
        self.pts = []
        self.um.setText("")
        self._preview()

    def clear_saved(self) -> None:
        if not confirm(self.panel, "Clear saved scale", "Delete scale.json for this sample? Sizes stay in pixels until you draw a new line."):
            return
        self.s.clear_saved("scale")
        self.saved_scale = None
        self.s.set_scale()
        self.mod.done -= set(ORDER[ORDER.index("scale"):])
        self.win.refresh_steps()
        self._preview()

    def typed(self) -> Optional[float]:
        try:
            v = float(self.um.text().strip())
        except ValueError:
            return None
        return v if v > 0 else None

    def drawn(self):
        if len(self.pts) != 2 or self.length.value() <= 0:
            return None
        try:
            return self.s.scale_from_line(self.pts[0], self.pts[1], float(self.length.value()), self.unit.currentText())
        except ValueError:
            return None

    def _preview(self) -> None:
        if self.s.bse_gray is None:
            return
        c = self.win.canvas
        ss = self.saved_scale
        if ss and ss.get("p1") and ss.get("p2"):
            p1, p2 = tuple(ss["p1"]), tuple(ss["p2"])
            c.set_lines("scale_saved", [[p1, p2]], SAVED_SCALE, 1.5, dashed=True, dots=4)
            c.set_label("scale_saved_lbl", min(p1[0], p2[0]), min(p1[1], p2[1]) - 22,
                        f"saved: {ss.get('length', 0):g} {ss.get('unit', '')}", SAVED_SCALE)
        else:
            c.set_lines("scale_saved", [], SAVED_SCALE)
            c._remove("scale_saved_lbl")
        seg = [self.pts] if len(self.pts) == 2 else ([[self.pts[0], self.pts[0]]] if self.pts else [])
        c.set_lines("scale_new", seg, NEW_SCALE, 1.5, dots=4)
        self.footer_line(label=False)
        t, d = self.typed(), self.drawn()
        if t:
            self.info.setText(f"{t:.4f} µm/px (typed, not saved)")
        elif d:
            self.info.setText(f"{d['length']:g} {d['unit']} = {d['px_length']:.1f} px → {d['um_per_px']:.4f} µm/px")
        elif len(self.pts) == 1:
            self.info.setText("click the other end of the scale bar")
        else:
            self.info.setText("click both ends of the scale bar")
        self.undo_btn.setEnabled(bool(self.pts))
        self.reset_btn.setEnabled(bool(self.pts) or bool(self.um.text()))
        self.update_state()

    def update_state(self) -> None:
        ss, t, d = self.saved_scale, self.typed(), self.drawn()
        if t or d:
            v = t or d["um_per_px"]
            self.set_state(f"New scale {v:.4f} µm/px: not applied" + ("" if t else " / not saved"), "unsaved")
        elif ss:
            used = self.complete() and self.s.scale == ss
            what = (f"{ss['length']:g} {ss.get('unit', '')} line = " if ss.get("length") else "")
            src = "scale.json" if self.s.saved_path("scale_json") else "grains file"
            self.set_state(f"Saved: {what}{float(ss['um_per_px']):.4f} µm/px ({src})"
                           + ("" if used else ". Apply or Skip keeps it."), "saved")
        elif self.complete() and self.s.scale:
            self.set_state(f"Using {self.s.um_per_px:.4f} µm/px", "saved")
        else:
            self.set_state("No scale: sizes will be in pixels", "none")

    def _ocr(self) -> None:
        def done(v):
            cur = self.typed() or (self.drawn() or {}).get("um_per_px") or (self.saved_scale or {}).get("um_per_px")
            txt = f"footer text says {v} µm/px" if v else "footer text: not read"
            if v and cur:
                txt += f" (line: {cur:.4f}, {100 * (cur / v - 1):+.1f}%)"
            self.ocr_info.setText(txt)
        self.win.jobs.run("Reading footer", self.s.footer_ocr, done)

    def apply(self) -> None:
        t, d = self.typed(), self.drawn()
        if t is None and d is None and len(self.pts) == 1:
            self.win.status.setText("click the other end of the scale bar first")
            return
        if t is None and d is None:
            self.skip()
            return
        self.s.set_scale(um_per_px=t, drawn=d if t is None else None)
        if t is None:
            self.saved_scale, self.pts = dict(d), []
        self.mod.finish("scale", go_next=True)

    def skip(self, go_next: bool = True) -> None:
        """Close without a new line: keep the saved scale (notebook behaviour), else none."""
        ss = self.saved_scale if self.saved_scale is not None else self.s.saved().get("scale")
        if ss and ss.get("p1") and ss.get("p2"):
            self.s.set_scale(drawn=dict(ss))
        elif ss:
            self.s.scale = dict(ss)
        else:
            self.s.set_scale()
        self.mod.finish("scale", go_next=go_next)

    def next(self) -> None:
        if not self.complete():
            self.skip()
        else:
            super().next()


# 6 ---------------------------------------------------------------------------
EDIT_CONTROLS = [
    ("Left click", "select a grain (pink); click again: deselect"),
    ("Esc", "deselect all"),
    ("D / Delete", "delete the selected grains"),
    ("M", "merge the selected grains (they must touch)"),
    ("Z", "undo last edit"),
    ("A", "add-grain mode (SAM 2.1)"),
    ("Scroll · Drag", "zoom · pan  (F: fit)"),
]
ADD_CONTROLS = [
    ("Left click", "point inside the new grain (green)"),
    ("Right / ⇧ click", "background point (red)"),
    ("Drag", "box around the grain (blue)"),
    ("Enter", "accept the cyan preview"),
    ("Esc", "clear the prompts"),
    ("A", "leave add mode"),
    ("Scroll", "zoom"),
]


class SegmentStep(_AStep):
    title = "Grain segmentation"
    needs = "nongrain"

    def __init__(self, mod) -> None:
        super().__init__(mod)
        st = mod.session.s
        self.lay.addWidget(title("6  Grain segmentation", "Run SegmentEveryGrain (or load the saved grains), then fix outlines on the image. Yellow = grains. 'Use this grain map' measures the grains and computes minerals, sizes and figures for the next steps."))
        g = QGroupBox("Segmentation")
        f = QFormLayout(g)
        self.sam = QCheckBox("SAM 2.1 refine", checked=st.use_sam2)
        self.min_px = QSpinBox(maximum=100000, value=int(st.min_grain_px))
        self.drop = QDoubleSpinBox(decimals=2, maximum=1.0, singleStep=0.05, value=st.dark_drop_frac)
        self.drop.setToolTip("Drop a grain when this fraction of it is non-grain (DARK_DROP_FRAC)")
        f.addRow(self.sam)
        f.addRow("Min grain (px)", self.min_px)
        f.addRow("Non-grain drop fraction", self.drop)
        load = QPushButton("Load saved grains")
        load.clicked.connect(self.load)
        f.addRow(load)
        self.lay.addWidget(g)
        e = QGroupBox("Edit")
        v = QVBoxLayout(e)
        self.add = QPushButton("Add grain (A)", checkable=True)
        dl, mg, ud = QPushButton("Delete (D)"), QPushButton("Merge (M)"), QPushButton("Undo last edit (Z)")
        dl.clicked.connect(self.delete)
        mg.clicked.connect(self.merge)
        dl.setToolTip("Delete the selected grains (D)")
        mg.setToolTip("Merge the selected grains (M); they must touch")
        self.del_btn, self.merge_btn = dl, mg
        ud.clicked.connect(self.undo)
        self.undo_btn = ud
        self.add.toggled.connect(self._arm_add)
        v.addWidget(row(self.add, ud))
        v.addWidget(row(dl, mg))
        desel, reset = QPushButton("Deselect all (Esc)"), QPushButton("Reset edits")
        desel.clicked.connect(self._clear)
        reset.setToolTip("Back to the grains as segmented / loaded (drops all edits)")
        reset.clicked.connect(self.reset)
        self.reset_btn = reset
        v.addWidget(row(desel, reset))
        self.accept = QPushButton("Accept grain (Enter)")
        self.accept.clicked.connect(self.accept_prompt)
        clr = QPushButton("Clear prompts (Esc)")
        clr.clicked.connect(self.clear_prompts)
        self.prompt_row = row(self.accept, clr)
        v.addWidget(self.prompt_row)
        self.add_hint = hint("Add-grain mode: prompts combine; the cyan preview updates after each one.")
        v.addWidget(self.add_hint)
        self.info = hint("no grains yet")
        v.addWidget(self.info)
        self.lay.addWidget(e)
        clear = QPushButton("Clear saved grains…")
        clear.setProperty("danger", True)
        clear.clicked.connect(self.clear_saved)
        self.lay.addWidget(clear)
        self.finish_layout()
        self.set_controls(EDIT_CONTROLS)
        self.add_primary("Run segmentation", self.run)
        self.use_btn = self.add_primary("Use this grain map", self.apply)
        self.polys: List = []
        self.base: List = []
        self.source = ""
        self.sel: set = set()
        self.history: List = []
        self.edited = False
        self._tree = None
        self._sam_ready = False
        self.prompt_pts: List = []    # [(x, y, positive)]
        self.prompt_box = None        # (x0, y0, x1, y1)
        self.preview = None           # shapely Polygon from SAM
        self.win.canvas.boxDragged.connect(self.on_box)
        self._show_add_ui(False)
        self.update_state()

    def _show_add_ui(self, on: bool) -> None:
        for w in (self.prompt_row, self.add_hint):
            w.setVisible(on)
        self.accept.setEnabled(on and self.preview is not None)
        self.set_controls(ADD_CONTROLS if on else EDIT_CONTROLS)

    def _sync_settings(self) -> None:
        st = self.s.s
        st.use_sam2, st.min_grain_px, st.dark_drop_frac = self.sam.isChecked(), float(self.min_px.value()), float(self.drop.value())

    def activate(self) -> None:
        if self.s.bse_stretch is None:
            return
        if "scale" not in self.mod.done:  # scale step not visited: keep the saved scale, as the notebook does
            self.mod.step(ScaleStep).skip(go_next=False)
        self.win.canvas.set_image(self.s.bse_stretch, fit=False)
        self.win.canvas.fit()
        self.redraw()
        self.canvas_focus()

    def run(self) -> None:
        if self.edited and not confirm(self.panel, "Run segmentation", "Discard your unsaved grain edits and segment again?"):
            return
        self._sync_settings()
        self._sam_ready = False
        prog = self.win.segProgress.emit  # thread-safe: queued to the GUI thread
        self.win.jobs.run("Segmenting (models stay loaded for the next run)",
                          lambda: self.s.segment(self.win.backend, progress=prog),
                          lambda p: self._got(p, "segmented"))

    def load(self) -> None:
        if not self.s.P or self.s.saved_path("grains") is None:
            self.win.status.setText("no saved grains for this sample")
            return
        self.win.jobs.run("Loading grains", self.s.load_saved_grains, lambda p: self._got(p, "saved"))

    def _got(self, polys, source: str = "") -> None:
        self.polys, self.sel, self.history, self.edited = list(polys), set(), [], False
        self.base = list(polys)
        self.source = source or self.source
        self._tree = None
        self.redraw()
        self.win.log(f"{len(self.polys)} grains")

    def reset(self) -> None:
        if not self.history:
            return
        if not confirm(self.panel, "Reset edits", f"Drop all {len(self.history)} edits and go back to the {self.source} grains?"):
            return
        self._got(self.base)

    def clear_saved(self) -> None:
        p = self.s.saved_path("grains") if self.s.P else None
        if p is None:
            self.win.status.setText("no saved grains for this sample")
            return
        if not confirm(self.panel, "Clear saved grains", f"Delete {Path(p).name}? Grains shown now stay until you leave."):
            return
        self.s.clear_saved("grains")
        self.mod.done -= set(ORDER[ORDER.index("segment"):])
        self.win.refresh_steps()
        self.update_state()

    def update_state(self) -> None:
        p = self.s.saved_path("grains") if self.s.P else None
        on_disk = p is not None
        n = len(self.polys)
        self.use_btn.setEnabled(bool(n))
        self.del_btn.setEnabled(bool(self.sel))
        self.merge_btn.setEnabled(len(self.sel) >= 2)
        self.undo_btn.setEnabled(bool(self.history))
        self.reset_btn.setEnabled(bool(self.history))
        if not n:
            self.set_state("No grains yet: Run segmentation" + (" or Load saved grains" if on_disk else ""), "none")
        elif self.edited:
            self.set_state(f"{n} grains, {len(self.history)} edits not saved: 'Use this grain map' saves them", "unsaved")
        elif on_disk and self.source in ("saved", "segmented", "measured"):
            self.set_state(f"Saved: {n} grains ({Path(p).name})" + ("" if self.complete() else ". 'Use this grain map' measures them."), "saved")
        else:
            self.set_state(f"{n} grains: not saved yet", "unsaved")

    def tree(self):
        if self._tree is None:
            from shapely.strtree import STRtree

            self._tree = STRtree(self.polys) if self.polys else None
        return self._tree

    def redraw(self) -> None:
        c = self.win.canvas
        c.set_polygons("grains", [p for i, p in enumerate(self.polys) if i not in self.sel], GRAIN, 1.0)
        c.set_polygons("selected", [self.polys[i] for i in sorted(self.sel)], "#ff375f", 2.0, fill="#55ff375f", z=3)
        self.info.setText(f"{len(self.polys)} grains, {len(self.sel)} selected" + (" (edited)" if self.edited else ""))
        self.update_state()

    def _push(self) -> None:
        self.history.append(list(self.polys))
        self.edited = True
        self._tree = None

    def on_click(self, x, y, b) -> None:
        from shapely.geometry import Point

        if self.add.isChecked():
            from PySide6.QtWidgets import QApplication

            neg = b == Qt.RightButton or bool(QApplication.keyboardModifiers() & Qt.ShiftModifier)
            if b in (Qt.LeftButton, Qt.RightButton):
                self.add_point(x, y, not neg)
            return
        t = self.tree()
        if t is None:
            return
        pt = Point(x, y)
        hits = [int(i) for i in t.query(pt, predicate="intersects")]
        if not hits:
            return
        i = min(hits, key=lambda k: self.polys[k].area)
        self.sel.symmetric_difference_update({i})
        self.redraw()

    def on_key(self, k) -> None:
        if self.add.isChecked() and k in ("enter", "esc"):
            (self.accept_prompt if k == "enter" else self.clear_prompts)()
            return
        {"d": self.delete, "m": self.merge, "z": self.undo, "esc": self._clear,
         "a": lambda: self.add.setChecked(not self.add.isChecked())}.get(k, lambda: None)()

    def _clear(self) -> None:
        self.sel.clear()
        self.redraw()

    def delete(self) -> None:
        if not self.sel:
            return
        self._push()
        self.polys = [p for i, p in enumerate(self.polys) if i not in self.sel]
        self.sel.clear()
        self.redraw()

    def merge(self) -> None:
        """GrainPlot merge: union of the selection; rejected if they do not touch."""
        import shapely
        from shapely.geometry import Polygon

        if len(self.sel) < 2:
            return
        u = shapely.unary_union([self.polys[i] for i in self.sel])
        if u.geom_type != "Polygon":
            self.win.status.setText("selected grains do not touch: not merged")
            self._clear()
            return
        self._push()
        self.polys = [p for i, p in enumerate(self.polys) if i not in self.sel] + [Polygon(u.exterior.coords)]
        self.sel.clear()
        self.redraw()

    def undo(self) -> None:
        if self.history:
            self.polys = self.history.pop()
            self.sel.clear()
            self._tree = None
            self.edited = bool(self.history)
            self.redraw()

    def _arm_add(self, on: bool) -> None:
        self.win.canvas.set_box_mode(on)
        if not on:
            self.clear_prompts()
        self._show_add_ui(on)
        if not on or self._sam_ready:
            return
        self._sync_settings()

        def prep():
            w = self.s.weights(sam2=True)
            self.win.backend.request("set_image", image=str(self.s.seg_image_png()), sam=str(w["sam"]))
            return True

        def ok(_):
            self._sam_ready = True
            self.win.status.setText("SAM ready: click inside a grain, shift/right-click background, or drag a box")

        self.win.jobs.run("Preparing SAM 2.1 for click-to-add", prep, ok, lambda e: self.add.setChecked(False))

    # -- SAM prompts ------------------------------------------------------------
    def add_point(self, x, y, positive: bool = True) -> None:
        self.prompt_pts.append((float(x), float(y), bool(positive)))
        self._run_prompt()

    def on_box(self, x0, y0, x1, y1) -> None:
        if self.add.isChecked() and self.mod.current() is self:
            self.prompt_box = (float(x0), float(y0), float(x1), float(y1))
            self._run_prompt()

    def prompt_request(self) -> dict:
        return {
            "points": [[x, y] for x, y, _ in self.prompt_pts],
            "labels": [p for _, _, p in self.prompt_pts],
            "box": list(self.prompt_box) if self.prompt_box else None,
        }

    def _draw_prompts(self) -> None:
        from shapely.geometry import box as sbox

        c = self.win.canvas
        c.set_points("pos", [(x, y) for x, y, p in self.prompt_pts if p], "#30d158", 5)
        c.set_points("neg", [(x, y) for x, y, p in self.prompt_pts if not p], "#ff453a", 5)
        c.set_polygons("pbox", [sbox(*self.prompt_box)] if self.prompt_box else [], "#64d2ff", 1.5, z=4)
        c.set_polygons("preview", [self.preview] if self.preview is not None else [], "#5ac8fa", 2.0,
                       fill="#335ac8fa", z=4)
        self._show_add_ui(self.add.isChecked())

    def _run_prompt(self) -> None:
        from shapely.geometry import Polygon

        self._draw_prompts()
        if not self._sam_ready or (not self.prompt_pts and not self.prompt_box):
            return
        if self.win.jobs.busy:  # a SAM call is running: run again with all prompts when it ends
            self._prompt_pending = True
            return
        self._prompt_pending = False
        req = self.prompt_request()

        def done(msg):
            if getattr(self, "_prompt_pending", False):
                self._prompt_pending = False
                self._run_prompt()
                return
            coords = msg.get("coords")
            self.preview = Polygon(coords) if coords and len(coords) >= 3 else None
            if self.preview is None:
                self.win.status.setText("SAM found no grain for these prompts")
            self._draw_prompts()

        self.win.jobs.run("SAM", lambda: self.win.backend.request("prompt", **req), done)

    def accept_prompt(self) -> None:
        if self.preview is None:
            return
        self._push()
        self.polys.append(self.preview)
        self.clear_prompts()
        self.redraw()

    def clear_prompts(self) -> None:
        self.prompt_pts, self.prompt_box, self.preview = [], None, None
        self._prompt_pending = False
        self._draw_prompts()

    def apply(self) -> None:
        if not self.polys:
            self.win.status.setText("no grains")
            return
        self._sync_settings()

        if not self.s.extra_legend:  # hand-named map colours saved earlier
            self.s.set_extra_legend(self.s.saved().get("extra_legend") or {})

        def work():
            if self.edited:
                self.s.save_edits(self.win.backend, self.polys)
            else:
                self.s.grain_polys = list(self.polys)
            self.s.finalize_grains()
            return self.s.compute_downstream()

        def done(_):
            self.mod.version += 1
            self._got(self.s.grain_polys, "measured")
            # move on only if the user is still here; if they went elsewhere meanwhile, just mark it done
            here = self.mod.current() is self
            self.mod.finish("segment", go_next=here)
            if not here:
                cur = self.mod.current()
                if cur is not None and cur.ready():
                    cur.activate()  # show the new results where the user is

        def failed(msg):
            self.win.status.setText(f"'Use this grain map' failed: {msg}")
            self.win.log(f"'Use this grain map' failed: {msg}")
            self.win.refresh_steps()

        self.win.jobs.run("Measuring grains, voting minerals, drawing figures", work, done, failed)


# 7 ---------------------------------------------------------------------------
STATUS = {  # minerals table: icon, colour, tooltip
    "ok": ("\u2713", "#1f9d3a", "identical colour in the map"),
    "reassigned": ("\u25B2", "#c98a00", "reassigned / approximated: the map colour differs from the legend"),
    "missing": ("\u2717", "#d70015", "not found in the map"),
    "unlisted": ("\u2717", "#d70015", "map colour no legend name could take: counted as Unknown"),
}
REASSIGNED_BG = "#ffe9a8"
MINERAL_FIGURES = [("Composition", "composition"), ("Overlay", "overlay"),
                   ("Mixed grains", "mixed"), ("Unmapped grains", "unmapped")]


def rgb_text(rgb) -> str:
    return ",".join(str(int(v)) for v in rgb)


def short_count(n: int) -> str:
    """1234 -> '1.2k', 1090000 -> '1.09M'."""
    n = int(n)
    if n < 1000:
        return str(n)
    if n < 1_000_000:
        return f"{n / 1000:.3g}k"
    return f"{n / 1_000_000:.3g}M"


def swatch(rgb, size: int = 11):
    from PySide6.QtGui import QIcon, QPixmap

    pm = QPixmap(size, size)
    pm.fill(QColor(*rgb))
    return QIcon(pm)


class _AfterGrains(_AStep):
    """Steps that only display what 'Use this grain map' computed."""
    needs = "segment"

    def complete(self) -> bool:
        return "segment" in self.mod.done

    def fresh(self) -> bool:
        """True once per new computation (refill tables / figure lists)."""
        if getattr(self, "_version", None) == self.mod.version:
            return False
        self._version = self.mod.version
        return True


class MineralsStep(_AfterGrains):
    title = "Minerals"

    def __init__(self, mod) -> None:
        super().__init__(mod)
        self.lay.addWidget(title("7  Minerals", "Each grain takes the mineral under most of its pixels. The table compares every legend colour with the colours in the mineral map."))
        self.view = QComboBox()
        self.view.addItems([n for n, _ in MINERAL_FIGURES])
        self.view.currentIndexChanged.connect(lambda _: self.show_figure())
        self.lay.addWidget(row(QLabel("Figure:"), self.view))
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["", "Mineral", "Legend colour", "Colour in map", "Pixels"])
        self.table.verticalHeader().hide()
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        from PySide6.QtWidgets import QHeaderView

        f = self.table.font()
        f.setPointSizeF(max(9.0, f.pointSizeF() - 1.5))
        self.table.setFont(f)
        self.table.setIconSize(QSize(11, 11))
        self.table.setTextElideMode(Qt.ElideRight)
        self.table.setWordWrap(False)
        self.table.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        hh = self.table.horizontalHeader()
        hh.setMinimumSectionSize(20)
        hh.setStretchLastSection(False)
        hh.setSectionResizeMode(0, QHeaderView.Fixed)
        self.table.setColumnWidth(0, 22)
        hh.setSectionResizeMode(1, QHeaderView.Stretch)     # long names elide (full name in the tooltip)
        for c in (2, 3, 4):
            hh.setSectionResizeMode(c, QHeaderView.ResizeToContents)
        self.table.setMinimumHeight(260)
        self.table.itemSelectionChanged.connect(self._sel_changed)
        self.lay.addWidget(self.table)
        self.lay.addWidget(hint("\u2713 identical colour   \u25B2 reassigned / approximated (highlighted)   "
                                "\u2717 not found or not matchable"))
        self.add_btn = QPushButton("+ Add map colour not in legend")
        self.add_btn.clicked.connect(self._add_dialog)
        self.lay.addWidget(self.add_btn)
        self.lay.addWidget(hint("Give a map colour that has no legend entry a mineral name (EXTRA_LEGEND); "
                                "the grains are counted again."))
        self.rm_btn = QPushButton("Remove added colour")
        self.rm_btn.clicked.connect(self.remove_selected)
        clear = QPushButton("Clear saved…")
        clear.setProperty("danger", True)
        clear.setToolTip("Delete the saved added colours (extra_legend.json)")
        clear.clicked.connect(self.clear_saved)
        self.lay.addWidget(row(self.rm_btn, clear))
        self.summary = hint()
        self.lay.addWidget(self.summary)
        self.finish_layout()
        self.set_controls(VIEW + [("Figure:", "pick the figure"), ("Click a row", "select it (+ Add uses its colour)")])
        self.rows: List[dict] = []
        self._sel_changed()

    def activate(self) -> None:
        if not self.complete():
            self.set_state("Not computed yet: press 'Use this grain map' in step 6", "none")
            return
        if self.fresh():
            self.fill()
        self.show_figure()

    def fill(self) -> None:
        from PySide6.QtGui import QBrush, QFont

        self.rows = self.s.mineral_table()
        t = self.table
        t.setRowCount(0)
        bold = QFont()
        bold.setBold(True)
        for r in self.rows:
            i = t.rowCount()
            t.insertRow(i)
            icon, col, tip = STATUS[r["status"]]
            st = QTableWidgetItem(icon)
            st.setForeground(QBrush(QColor(col)))
            st.setFont(bold)
            st.setTextAlignment(Qt.AlignCenter)
            name = r["mineral"] + ("  (added)" if r["extra"] else "")
            nm = QTableWidgetItem(name)
            nm.setToolTip(f"{name}: {tip}")
            cells = [st, nm]
            leg = QTableWidgetItem("" if r["legend_rgb"] is None else rgb_text(r["legend_rgb"]))
            if r["legend_rgb"] is not None:
                leg.setIcon(swatch(r["legend_rgb"]))
            maps = r["map_rgbs"]
            other = [c for c in maps if c != r["legend_rgb"]]
            shown = (other or maps or [None])[0] if r["status"] == "reassigned" else (maps[0] if maps else None)
            mp = QTableWidgetItem("" if shown is None else rgb_text(shown) + (f" +{len(maps) - 1}" if len(maps) > 1 else ""))
            if shown is not None:
                mp.setIcon(swatch(shown))
            mp.setToolTip("map colours: " + "; ".join(", ".join(map(str, c)) for c in maps) if maps else "not in the map")
            px = QTableWidgetItem(f"{short_count(r['pixels'])} ({r['pct']:.1f}%)" if r["pixels"] else "0")
            px.setToolTip(f"{r['pixels']:,d} px ({r['pct']:.2f}% of the mapped pixels)")
            px.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            cells += [leg, mp, px]
            for c, it in enumerate(cells):
                it.setToolTip(it.toolTip() or tip)
                if r["status"] == "reassigned":
                    it.setBackground(QBrush(QColor(REASSIGNED_BG)))
                    if c:
                        it.setForeground(QBrush(QColor("#1c1c1e")))
                t.setItem(i, c, it)
        n = {k: sum(r["status"] == k for r in self.rows) for k in STATUS}
        mapped = len(self.s.mapped) if self.s.mapped is not None else 0
        unm = int((self.s.grains["mineral"] == "Unmapped").sum()) if self.s.grains is not None and "mineral" in self.s.grains else 0
        self.summary.setText(f"{mapped} grains with a mineral, {unm} unmapped (left out). "
                             f"Added colours: {len(self.s.extra_legend)}.")
        self.set_state(f"\u2713 {n['ok']}   \u25B2 {n['reassigned']}   \u2717 {n['missing'] + n['unlisted']}  "
                       "· computed, not exported yet", "saved")

    def show_figure(self) -> None:
        key = dict(MINERAL_FIGURES).get(self.view.currentText())
        p = self.s.C.get(key) if self.s.C else None
        if p is not None and Path(p).exists():
            self.win.canvas.set_image(png_image(p))

    def selected_row(self) -> Optional[dict]:
        ix = self.table.currentRow()
        return self.rows[ix] if 0 <= ix < len(self.rows) and self.table.selectedItems() else None

    def _sel_changed(self) -> None:
        r = self.selected_row()
        self.rm_btn.setEnabled(bool(r and r["extra"]))

    @staticmethod
    def parse_rgb(text: str):
        t = text.strip()
        if t.startswith("#") and len(t) == 7:
            return tuple(int(t[i:i + 2], 16) for i in (1, 3, 5))
        parts = [p for p in t.replace(",", " ").replace("(", " ").replace(")", " ").split() if p]
        if len(parts) == 3 and all(p.isdigit() and 0 <= int(p) <= 255 for p in parts):
            return tuple(int(p) for p in parts)
        return None

    def suggest_rgb(self):
        r = self.selected_row()
        if r and r["map_rgbs"]:
            return r["map_rgbs"][-1]
        unl = [r for r in self.rows if r["status"] == "unlisted"]
        return unl[0]["map_rgbs"][0] if unl else (0, 0, 0)

    def _add_dialog(self) -> None:
        from PySide6.QtWidgets import QDialog, QDialogButtonBox

        d = QDialog(self.panel)
        d.setWindowTitle("Add map colour not in legend")
        f = QFormLayout(d)
        f.addRow(hint("A colour in the mineral map with no legend entry gets this mineral name."))
        name = QLineEdit(placeholderText="mineral name")
        col = QLineEdit(", ".join(map(str, self.suggest_rgb())))
        f.addRow("Mineral", name)
        f.addRow("Map colour (R, G, B or #rrggbb)", col)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(d.accept)
        bb.rejected.connect(d.reject)
        f.addRow(bb)
        if d.exec() != QDialog.Accepted:
            return
        rgb = self.parse_rgb(col.text())
        if not name.text().strip() or rgb is None:
            self.win.status.setText("need a mineral name and a colour (R, G, B 0–255 or #rrggbb)")
            return
        self.add_extra(name.text().strip(), rgb)

    def add_extra(self, name: str, rgb) -> None:
        self._recompute({**self.s.extra_legend, str(name): tuple(int(c) for c in rgb)})

    def remove_selected(self) -> None:
        r = self.selected_row()
        if r and r["extra"]:
            self._recompute({k: v for k, v in self.s.extra_legend.items() if k != r["mineral"]})

    def clear_saved(self) -> None:
        if not confirm(self.panel, "Clear saved added colours", "Delete the saved added map colours (extra_legend.json) and count again without them?"):
            return
        self.s.clear_saved("extra")
        if self.s.extra_legend and self.complete():
            self._recompute({})
        else:
            self.s.set_extra_legend({})

    def _recompute(self, extra) -> None:
        self.s.set_extra_legend(extra)
        if not self.complete():
            return

        def done(_):
            self.mod.version += 1
            self.mod.finish("segment")
            self.activate()

        self.win.jobs.run("Counting minerals again", self.s.compute_downstream, done)


# 8 ---------------------------------------------------------------------------
class GsdStep(_AfterGrains):
    title = "Grain size"

    def __init__(self, mod) -> None:
        super().__init__(mod)
        self.lay.addWidget(title("8  Grain size", "Grain-size curves from the grain map of step 6: all grains, or one mineral."))
        self.view = QComboBox()
        self.view.currentIndexChanged.connect(lambda _: self.show_figure())
        self.lay.addWidget(row(QLabel("Figure:"), self.view))
        self.summary = QLabel(wordWrap=True)
        self.lay.addWidget(self.summary)
        self.finish_layout()
        self.set_controls(VIEW + [("Figure:", "all grains or one mineral")])
        self.add_primary("Replot grain size curves", self.replot)
        self.figs = {}

    def activate(self) -> None:
        if not self.complete():
            self.set_state("Not computed yet: press 'Use this grain map' in step 6", "none")
            return
        if self.fresh():
            self.fill()
        self.show_figure()

    def fill(self) -> None:
        self.figs = self.s.gsd_figures()
        cur = self.view.currentText()
        self.view.blockSignals(True)
        self.view.clear()
        self.view.addItems(list(self.figs))
        if cur in self.figs:
            self.view.setCurrentText(cur)
        self.view.blockSignals(False)
        g = self.s.grains
        big = int((g["area_px"] >= self.s.s.min_grain_px).sum())
        unit = "µm" if self.s.size_unit == "um" else "px"
        self.summary.setText(f"{len(g)} grains measured; {big} with area ≥ {self.s.s.min_grain_px:g} px in the curves "
                             f"(minerals with ≥ {self.s.s.min_grains_curve} grains get their own curve).\n"
                             f"Sizes in {unit}" + (f" ({self.s.um_per_px:.4f} µm/px)." if self.s.um_per_px else " (no scale)."))
        self.set_state(f"{len(self.figs)} curves ready · not exported yet", "saved")

    def show_figure(self) -> None:
        p = self.figs.get(self.view.currentText())
        if p and Path(p).exists():
            self.win.canvas.set_image(png_image(p))

    def replot(self) -> None:
        if not self.complete():
            return

        def done(_):
            self._version = None
            self.activate()

        self.win.jobs.run("Replotting grain size curves", self.s.replot_gsd, done)


# 9 ---------------------------------------------------------------------------
class OutputsStep(_AfterGrains):
    title = "Outputs"

    def __init__(self, mod) -> None:
        super().__init__(mod)
        self.lay.addWidget(title("9  Outputs", "The knolling diagram is ready below. Nothing is written to the export folder until you press an Export button; the files have the same names as notebook 4's."))
        self.summary = QLabel(wordWrap=True)
        self.lay.addWidget(self.summary)
        self.lay.addWidget(hint("Exported files (click a figure to view it):"))
        self.files = QListWidget()
        self.files.setMinimumHeight(180)
        self.files.currentTextChanged.connect(self._show)
        self.lay.addWidget(self.files)
        self.finish_layout()
        self.set_controls(VIEW + [("Click a file", "view an exported figure")])
        self.csv_btn = self.add_primary("Export CSV data sheets", self.export_data)
        self.fig_btn = self.add_primary("Export all figures", self.export_figures)
        self.replot_btn = QPushButton("Replot grain knolling diagram")
        self.replot_btn.clicked.connect(self.replot)
        self.open_btn = QPushButton("Open export folder")
        self.open_btn.setToolTip("Open the export folder in Finder / Explorer (created if needed)")
        self.open_btn.clicked.connect(self._open)
        self.primary_box.addWidget(row(self.replot_btn, self.open_btn))
        self.exported = {"data": [], "figures": []}

    def complete(self) -> bool:
        return "outputs" in self.mod.done

    def ready(self) -> bool:
        return "segment" in self.mod.done

    def activate(self) -> None:
        on = "segment" in self.mod.done
        for b in (self.csv_btn, self.fig_btn, self.replot_btn):
            b.setEnabled(on)
        if not on:
            self.set_state("Not computed yet: press 'Use this grain map' in step 6", "none")
            return
        if self.fresh():
            self.exported = {"data": [], "figures": []}
            self.files.clear()
        self.show_knolling()
        C = self.s.C
        n_csv = len(list(C["folder"].glob("*.csv")))
        n_png = len(list(C["folder"].glob("*.png")))
        self.summary.setText(f"Ready to export to {self.s.P['folder']}:\n"
                             f"• CSV data sheets: {n_csv} tables (counts + one per mineral) and the grains / minerals GeoJSON, legend / threshold JSON\n"
                             f"• Figures: {n_png} PNGs (legend, threshold, overlay, unmapped, mixed, composition, grain size, knolling)")
        self.update_state()

    def update_state(self) -> None:
        d, f = self.exported["data"], self.exported["figures"]
        if d or f:
            self.set_state(f"Exported {len(d)} data files and {len(f)} figures to {self.s.P['folder'].name}/", "saved")
        else:
            self.set_state("Nothing exported yet", "none")

    def show_knolling(self) -> None:
        k = self.s.knoll
        if k is not None:
            self.win.canvas.set_image(k if k.ndim == 2 else k[..., :3])

    def _exported(self, kind: str, files) -> None:
        self.exported[kind] = list(files)
        self.files.clear()
        for p in self.exported["data"] + self.exported["figures"]:
            self.files.addItem(Path(p).name)
        self.mod.done.add("outputs")
        self.win.refresh_steps()
        self.update_state()

    def export_data(self) -> None:
        self.win.jobs.run("Exporting CSV data sheets", self.s.export_data, lambda f: self._exported("data", f))

    def export_figures(self) -> None:
        self.win.jobs.run("Exporting figures", self.s.export_figures, lambda f: self._exported("figures", f))

    def replot(self) -> None:
        self.win.jobs.run("Replotting the knolling diagram", self.s.replot_knolling, lambda _: self.show_knolling())

    def _show(self, name: str) -> None:
        p = self.s.P.get("folder", Path(".")) / name
        if p.suffix.lower() in (".png", ".jpg") and p.exists():
            self.win.canvas.set_image(png_image(p))

    def export_folder(self) -> Optional[Path]:
        """The sample's export folder, created if it does not exist yet."""
        if not self.s.P.get("folder"):
            return None
        d = Path(self.s.P["folder"])
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _open(self) -> None:
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices

        d = self.export_folder()
        if d is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(d)))

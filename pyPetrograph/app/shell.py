# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""
Shared app shell: one window, step list (left), canvas (center), slim parameter
panel (right), status bar with a busy indicator and a log. Modules (AMICS now;
others later) provide the steps.
"""
from __future__ import annotations

import sys
from typing import List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QApplication, QDockWidget, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QMainWindow,
    QPlainTextEdit, QProgressBar, QPushButton, QScrollArea, QSplitter, QStackedWidget, QVBoxLayout,
    QWidget,
)

from pyPetrograph.app.backend import SegBackend
from pyPetrograph.app.jobs import JobRunner
from pyPetrograph.app.viewer import ImageView

STYLE = """
QListWidget#steps { border: none; padding: 8px 0; background: palette(base); font-size: 14px; outline: 0; }
QListWidget#steps::item { padding: 9px 14px; border-radius: 6px; margin: 1px 8px; }
QListWidget#steps::item:selected { background: palette(highlight); color: palette(highlighted-text); }
QListWidget#steps::item:disabled { color: palette(placeholder-text); }
QWidget#panel { background: palette(base); }
QLabel[role="title"] { font-size: 17px; font-weight: 600; }
QLabel[role="hint"] { color: palette(placeholder-text); }
QPushButton[primary="true"] {
    font-weight: 600; font-size: 14px; padding: 8px 14px; border-radius: 7px;
    background: #0a84ff; color: white; border: 1px solid #0a6fd6;
}
QPushButton[primary="true"]:hover { background: #2a95ff; }
QPushButton[primary="true"]:pressed { background: #0a6fd6; }
QPushButton[primary="true"]:disabled { background: #9cc7f5; color: #eef5ff; border-color: #9cc7f5; }
QPushButton[danger="true"] {
    color: white; background: #c4161c; border: 1px solid #9e1015; border-radius: 6px;
    padding: 4px 10px; font-weight: 600;
}
QPushButton[danger="true"]:hover { background: #db2a30; }
QPushButton[danger="true"]:pressed { background: #9e1015; }
QPushButton[danger="true"]:disabled { background: #e8b4b6; color: #fff5f5; border-color: #e8b4b6; }
QProgressBar#segprogress { min-height: 18px; border: 1px solid palette(mid); border-radius: 5px; text-align: center; }
QProgressBar#segprogress::chunk { background: #0a84ff; border-radius: 4px; }
QWidget#footer { border-top: 1px solid palette(mid); }
QGroupBox#controls { font-weight: 600; border: 1px solid palette(mid); border-radius: 6px; margin-top: 14px; padding: 6px 4px 4px 4px; }
QGroupBox#controls::title { subcontrol-origin: margin; left: 8px; padding: 0 3px; }
QLabel[role="key"] { font-weight: 600; font-family: Menlo, Consolas, monospace; }
QLabel[role="state"] { padding: 6px 8px; border-radius: 6px; }
QLabel[state="saved"] { background: rgba(48, 209, 88, 0.18); }
QLabel[state="unsaved"] { background: rgba(255, 159, 10, 0.22); }
QLabel[state="none"] { background: rgba(142, 142, 147, 0.15); }
"""

MARK = {"done": "\u2713", "current": "\u25B6", "todo": "\u25CB"}  # ✓ ▶ ○


def _mmss(sec: float) -> str:
    sec = max(0, int(round(sec)))
    return f"{sec // 3600}:{sec % 3600 // 60:02d}:{sec % 60:02d}" if sec >= 3600 else f"{sec // 60}:{sec % 60:02d}"


def progress_text(d: dict, elapsed: float) -> str:
    """'U-Net: 120 / 560 tiles (21%) · patch 2/4 · 0:42 elapsed · ETA 2:31' from a server progress dict."""
    frac = float(d.get("frac") or 0.0)
    pct = int(frac * 100)
    stage = d.get("stage", "")
    if stage == "merge":
        head = f"merge: non-grain cleanup and saving ({pct}%)"
    else:
        head = f"{stage}: {d.get('done', 0)} / {d.get('total', 0)} {d.get('unit', '')} ({pct}%)"
        if int(d.get("patches") or 0) > 1:
            head += f" · patch {d.get('patch')}/{d.get('patches')}"
    eta = f" · ETA {_mmss(elapsed * (1 - frac) / frac)}" if 0.02 <= frac < 1 else ""
    return f"{head} · {_mmss(elapsed)} elapsed{eta}"


class Step:
    """
    One entry in the step list. Right panel layout, top to bottom:
    ``self.lay`` (title + options; the only part that scrolls), the Controls box,
    the saved / unsaved state line, and a pinned footer with the step's primary
    buttons plus Back / Next. ``activate`` fills the canvas.
    """

    title = ""
    optional = False

    def __init__(self, win: "AppWindow") -> None:
        self.win = win
        self.panel = QWidget(objectName="panel")
        outer = QVBoxLayout(self.panel)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        self.scroll = QScrollArea(objectName="panel")
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QScrollArea.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        body = QWidget(objectName="panel")
        self.lay = QVBoxLayout(body)
        self.lay.setContentsMargins(14, 14, 14, 8)
        self.lay.setAlignment(Qt.AlignTop)
        self.scroll.setWidget(body)
        outer.addWidget(self.scroll, 1)

        fixed = QWidget(objectName="footer")
        self.fixed = QVBoxLayout(fixed)
        self.fixed.setContentsMargins(12, 6, 12, 10)
        self.controls_box = QGroupBox("Controls", objectName="controls")
        self.controls_box.setLayout(QGridLayout())
        self.controls_box.layout().setContentsMargins(8, 6, 8, 6)
        self.controls_box.layout().setHorizontalSpacing(10)
        self.controls_box.layout().setVerticalSpacing(2)
        self.controls_box.hide()
        self.fixed.addWidget(self.controls_box)
        self.state_lbl = QLabel("", wordWrap=True)
        self.state_lbl.setProperty("role", "state")
        self.state_lbl.hide()
        self.fixed.addWidget(self.state_lbl)
        self.primary_box = QVBoxLayout()
        self.primary_box.setSpacing(10)
        self.fixed.addLayout(self.primary_box)
        nav = QHBoxLayout()
        self.back_btn = QPushButton("\u2039 Back")
        self.next_btn = QPushButton("Next \u203A")
        self.next_btn.setProperty("primary", True)
        self.back_btn.clicked.connect(lambda: self.win.go(self.index() - 1))
        self.next_btn.clicked.connect(self.next)
        nav.addWidget(self.back_btn)
        nav.addStretch(1)
        nav.addWidget(self.next_btn)
        self.fixed.addLayout(nav)
        outer.addWidget(fixed, 0)
        self.primaries: List[QPushButton] = []

    # -- layout helpers ---------------------------------------------------------
    def add_primary(self, text: str, slot) -> QPushButton:
        b = QPushButton(text)
        b.setProperty("primary", True)
        b.clicked.connect(slot)
        self.primary_box.addWidget(b)
        self.primaries.append(b)
        return b

    def set_controls(self, rows) -> None:
        """``rows``: (input, what it does) pairs, shown in the always-visible Controls box."""
        g = self.controls_box.layout()
        while g.count():
            w = g.takeAt(0).widget()
            w.hide()
            w.setParent(None)
            w.deleteLater()
        for r, (key, what) in enumerate(rows):
            k = QLabel(key)
            k.setProperty("role", "key")
            g.addWidget(k, r, 0, Qt.AlignTop)
            g.addWidget(QLabel(what, wordWrap=True), r, 1)
        g.setColumnStretch(1, 1)
        self.controls_box.setVisible(bool(rows))

    def set_state(self, text: str, state: str = "none") -> None:
        """``state``: 'saved' (green), 'unsaved' (orange) or 'none' (gray)."""
        self.state_lbl.setText(text)
        self.state_lbl.setProperty("state", state)
        self.state_lbl.style().unpolish(self.state_lbl)
        self.state_lbl.style().polish(self.state_lbl)
        self.state_lbl.setVisible(bool(text))

    # -- flow -------------------------------------------------------------------
    def index(self) -> int:
        return self.win.steps.index(self)

    def activate(self) -> None:  # canvas content for this step
        pass

    def ready(self) -> bool:
        """Can the user open this step (its inputs exist)?"""
        return True

    def complete(self) -> bool:
        """Is this step's required input done (enables Next)?"""
        return True

    def next(self) -> None:
        self.win.go(self.index() + 1)

    def update_nav(self) -> None:
        i = self.index()
        last = i == len(self.win.steps) - 1
        self.back_btn.setEnabled(i > 0)
        self.next_btn.setVisible(not last)
        done = self.complete()
        self.next_btn.setText("Skip \u203A" if self.optional and not done else "Next \u203A")
        self.next_btn.setEnabled((done or self.optional) and not last and self.win.steps[i + 1].ready())


class AppWindow(QMainWindow):
    logged = Signal(str)
    segProgress = Signal(object)  # progress dicts from the SEG server (any thread)

    def __init__(self, backend: Optional[SegBackend] = None) -> None:
        super().__init__()
        self.setWindowTitle("pyPetrograph")
        self.resize(1400, 880)
        self.logged.connect(self._append_log)
        self.backend = backend or SegBackend(log=self.log)
        self.jobs = JobRunner(self)
        self.jobs.busyChanged.connect(self._busy)
        self.canvas = ImageView()
        self.steps: List[Step] = []

        self.step_list = QListWidget(objectName="steps")
        self.step_list.setFixedWidth(190)
        self.step_list.currentRowChanged.connect(self._show_step)
        self.panels = QStackedWidget(objectName="panel")
        self.panels.setFixedWidth(430)

        split = QSplitter(Qt.Horizontal)
        split.addWidget(self.step_list)
        split.addWidget(self.canvas)
        split.addWidget(self.panels)
        split.setStretchFactor(1, 1)
        split.setChildrenCollapsible(False)
        self.setCentralWidget(split)

        self.status = QLabel("Ready")
        self.progress = QProgressBar(maximumWidth=140, maximum=0, visible=False, textVisible=False)
        self.seg_bar = QProgressBar(objectName="segprogress", minimumWidth=560, maximum=1000, visible=False)
        self.seg_bar.setFormat("")
        self.segProgress.connect(self._seg_progress)
        self._t0 = None
        log_btn = QPushButton("Log", flat=True)
        sb = self.statusBar()
        sb.addWidget(self.status, 1)
        sb.addPermanentWidget(self.seg_bar)
        sb.addPermanentWidget(self.progress)
        sb.addPermanentWidget(log_btn)
        self.log_view = QPlainTextEdit(readOnly=True)
        self.log_view.setMaximumBlockCount(5000)
        self.log_dock = QDockWidget("Log", self)
        self.log_dock.setWidget(self.log_view)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.log_dock)
        self.log_dock.hide()
        log_btn.clicked.connect(lambda: self.log_dock.setVisible(not self.log_dock.isVisible()))
        fit = QAction("Fit image", self, shortcut=QKeySequence("Ctrl+0"), triggered=self.canvas.fit)
        self.addAction(fit)

    # -- steps -----------------------------------------------------------------
    def add_step(self, step: Step) -> None:
        self.steps.append(step)
        item = QListWidgetItem("")
        self.step_list.addItem(item)
        self.panels.addWidget(step.panel)

    def step_status(self, i: int) -> str:
        if i == self.step_list.currentRow():
            return "current"
        return "done" if self.steps[i].complete() else "todo"

    def refresh_steps(self) -> None:
        for i, st in enumerate(self.steps):
            it = self.step_list.item(i)
            flags = it.flags()
            it.setFlags(flags | Qt.ItemIsEnabled if st.ready() else flags & ~Qt.ItemIsEnabled)
            opt = "  (optional)" if st.optional else ""
            it.setText(f"{MARK[self.step_status(i)]}  {i + 1}  {st.title}{opt}")
            it.setToolTip({"done": "done", "current": "current step", "todo": "to do"}[self.step_status(i)])
            st.update_nav()

    def go(self, index: int) -> None:
        """Open step ``index``; a step whose inputs are not ready is refused (no jumping ahead)."""
        self.refresh_steps()
        if not (0 <= index < len(self.steps)):
            return
        if index > self.step_list.currentRow() and not self.steps[index].ready():
            self.status.setText(f"'{self.steps[index].title}' needs the earlier steps first")
            return
        self.step_list.setCurrentRow(index)

    def _show_step(self, i: int) -> None:
        if 0 <= i < len(self.steps):
            self.panels.setCurrentIndex(i)
            self.steps[i].activate()
            self.refresh_steps()

    # -- status / log ----------------------------------------------------------
    def log(self, text: str) -> None:
        self.logged.emit(str(text))  # thread-safe

    def _append_log(self, text: str) -> None:
        self.log_view.appendPlainText(text)
        last = text.strip().splitlines()[-1] if text.strip() else ""
        if last and not self.jobs.busy:
            self.status.setText(last[:160])

    def _seg_progress(self, d) -> None:
        import time

        if not self.jobs.busy:  # a late message after the job ended
            return
        if self._t0 is None:
            self._t0 = time.monotonic()
        self.progress.setVisible(False)
        self.seg_bar.setVisible(True)
        self.seg_bar.setValue(int(1000 * float(d.get("frac") or 0)))
        txt = progress_text(d, time.monotonic() - self._t0)
        self.seg_bar.setFormat(txt)
        self.status.setText("Segmenting …")
        self.last_progress = txt

    def _busy(self, on: bool, label: str) -> None:
        import time

        self.progress.setVisible(on)
        self.seg_bar.setVisible(False)
        self._t0 = time.monotonic() if on else None
        if on:
            self.status.setText(label + " …")
        elif label:
            self.status.setText("Error: " + label)
            self.log("ERROR " + label)
        elif self.status.text().endswith(" …"):
            self.status.setText(self.status.text()[:-2] + " – done")
        for st in self.steps:
            st.panel.setEnabled(not on)
        if not on:
            self.refresh_steps()

    def closeEvent(self, ev) -> None:
        self.backend.close()
        super().closeEvent(ev)


def _own_plugins() -> None:
    """Use PySide6's own Qt plugins (a conda ``bin/qt.conf`` for another Qt would point elsewhere)."""
    import os

    import PySide6

    plug = os.path.join(os.path.dirname(PySide6.__file__), "Qt", "plugins")
    if os.path.isdir(plug):
        os.environ.setdefault("QT_PLUGIN_PATH", plug)
        os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", os.path.join(plug, "platforms"))


def make_app(argv=None) -> QApplication:
    if QApplication.instance() is None:
        _own_plugins()
    app = QApplication.instance() or QApplication(list(argv or sys.argv))
    app.setApplicationName("pyPetrograph")
    if sys.platform != "darwin":
        app.setStyle("Fusion")  # follows the OS light / dark scheme (Qt >= 6.5)
    app.setStyleSheet(STYLE)
    return app


def title(text: str, hint: str = "") -> QWidget:
    w = QWidget()
    lay = QVBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 6)
    t = QLabel(text)
    t.setProperty("role", "title")
    lay.addWidget(t)
    if hint:
        h = QLabel(hint, wordWrap=True)
        h.setProperty("role", "hint")
        lay.addWidget(h)
    return w


def row(*widgets) -> QWidget:
    w = QWidget()
    lay = QHBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)
    for x in widgets:
        lay.addWidget(x)
    return w

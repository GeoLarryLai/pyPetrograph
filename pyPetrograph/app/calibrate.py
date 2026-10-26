# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Shared scale calibration box: draw a line on the canvas + type its length, or type µm/px."""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QLabel, QLineEdit, QPushButton, QWidget,
)

from pyPetrograph.app.viewer import ImageView

UNITS = ["µm", "mm", "nm"]
UNIT_TO_UM = {"µm": 1.0, "um": 1.0, "mm": 1000.0, "nm": 0.001}


class ScaleBox(QGroupBox):
    """``scale()`` -> ``{"um_per_px": ...}`` (typed) or the drawn-line dict, or ``{}``."""

    changed = Signal()

    def __init__(self, canvas: ImageView, default_length: float = 500, default_unit: str = "µm") -> None:
        super().__init__("Scale")
        self.canvas = canvas
        self.pts: List[Tuple[float, float]] = []
        self.drawing = False
        f = QFormLayout(self)
        self.um = QLineEdit(placeholderText="µm per pixel (blank = use line)")
        self.um.textChanged.connect(lambda _: self._update())
        self.draw_btn = QPushButton("Draw scale line", checkable=True)
        self.draw_btn.toggled.connect(self._toggle)
        self.length = QDoubleSpinBox(decimals=3, maximum=1e9, value=float(default_length))
        self.unit = QComboBox()
        self.unit.addItems(UNITS)
        self.unit.setCurrentText(default_unit if default_unit in UNITS else "µm")
        self.length.valueChanged.connect(lambda _: self._update())
        self.unit.currentTextChanged.connect(lambda _: self._update())
        self.info = QLabel("no scale: sizes stay in pixels", wordWrap=True)
        self.info.setProperty("role", "hint")
        f.addRow("µm/px", self.um)
        f.addRow(self.draw_btn)
        f.addRow("Line length", self.length)
        f.addRow("Unit", self.unit)
        f.addRow(self.info)

    def _toggle(self, on: bool) -> None:
        self.drawing = on
        if on:
            self.pts = []
            self.canvas.set_points("scale", [])

    def on_click(self, x: float, y: float, button) -> bool:
        """Canvas click while drawing; returns True when consumed."""
        if not self.drawing or button != Qt.LeftButton:
            return False
        if len(self.pts) >= 2:
            self.pts = []
        self.pts.append((x, y))
        self.canvas.set_points("scale", self.pts, color="#0a84ff", radius=4, line=True)
        if len(self.pts) == 2:
            self.draw_btn.setChecked(False)
        self._update()
        return True

    def redraw(self) -> None:
        self.canvas.set_points("scale", self.pts, color="#0a84ff", radius=4, line=True)

    def line_px(self) -> float:
        if len(self.pts) != 2:
            return 0.0
        (x1, y1), (x2, y2) = self.pts
        return math.hypot(x2 - x1, y2 - y1)

    def typed(self) -> Optional[float]:
        try:
            v = float(self.um.text().strip())
        except ValueError:
            return None
        return v if v > 0 else None

    def drawn(self) -> Optional[Dict[str, Any]]:
        px = self.line_px()
        if px <= 0 or self.length.value() <= 0:
            return None
        p1, p2 = self.pts
        u = self.unit.currentText()
        return {"p1": [p1[0], p1[1]], "p2": [p2[0], p2[1]], "px_length": px,
                "length": float(self.length.value()), "unit": u,
                "um_per_px": float(self.length.value()) * UNIT_TO_UM[u] / px}

    def _update(self) -> None:
        t, d = self.typed(), self.drawn()
        if t:
            self.info.setText(f"{t:.4f} µm/px (typed)")
        elif d:
            self.info.setText(f"{d['length']:g} {d['unit']} = {d['px_length']:.1f} px → {d['um_per_px']:.4f} µm/px")
        else:
            self.info.setText("no scale: sizes stay in pixels")
        self.changed.emit()

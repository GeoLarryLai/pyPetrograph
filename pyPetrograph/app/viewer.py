# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Fast image canvas: pan (drag), zoom (wheel / pinch), fit (double-click or F), overlays."""
from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QImage, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import QGraphicsPixmapItem, QGraphicsScene, QGraphicsView


def to_qimage(arr: np.ndarray) -> QImage:
    """uint8 gray, RGB or RGBA array -> QImage (copied, so the array may be freed)."""
    a = np.ascontiguousarray(arr)
    if a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8)
    h, w = a.shape[:2]
    if a.ndim == 2:
        return QImage(a.data, w, h, w, QImage.Format_Grayscale8).copy()
    if a.shape[2] == 3:
        return QImage(a.data, w, h, 3 * w, QImage.Format_RGB888).copy()
    return QImage(a.data, w, h, 4 * w, QImage.Format_RGBA8888).copy()


def _rings(geom) -> Iterable[np.ndarray]:
    if geom is None or getattr(geom, "is_empty", True):
        return
    if geom.geom_type == "Polygon":
        yield np.asarray(geom.exterior.coords)
        return
    for part in getattr(geom, "geoms", ()):
        yield from _rings(part)


def polygons_path(polys: Sequence) -> QPainterPath:
    path = QPainterPath()
    for g in polys:
        for ring in _rings(g):
            if len(ring) < 2:
                continue
            path.moveTo(QPointF(ring[0, 0], ring[0, 1]))
            for x, y in ring[1:, :2]:
                path.lineTo(QPointF(x, y))
            path.closeSubpath()
    return path


class ImageView(QGraphicsView):
    """``clicked(x, y, button)`` in image pixels; a drag pans instead of clicking."""

    clicked = Signal(float, float, object)
    boxDragged = Signal(float, float, float, float)  # x0, y0, x1, y1 in image pixels (box mode only)
    keyPressed = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setViewportUpdateMode(QGraphicsView.SmartViewportUpdate)
        self.setFrameShape(QGraphicsView.NoFrame)
        self.setBackgroundBrush(self.palette().window())
        self._img = QGraphicsPixmapItem()
        self._img.setTransformationMode(Qt.SmoothTransformation)
        self._mask = QGraphicsPixmapItem()
        self._mask.setZValue(1)
        self.scene().addItem(self._img)
        self.scene().addItem(self._mask)
        self._layers = {}
        self._marks = {}
        self.setFocusPolicy(Qt.StrongFocus)
        self._press = None
        self._shape: Tuple[int, int] = (0, 0)
        self._box_mode = False
        self._band = None

    def set_box_mode(self, on: bool) -> None:
        """Left-drag draws a rectangle (``boxDragged``) instead of panning; wheel still zooms."""
        self._box_mode = bool(on)
        self.setDragMode(QGraphicsView.NoDrag if on else QGraphicsView.ScrollHandDrag)

    def _clip(self, p):
        h, w = self._shape
        return min(max(p.x(), 0.0), w - 1.0), min(max(p.y(), 0.0), h - 1.0)

    # -- content -------------------------------------------------------------
    def set_image(self, arr: Optional[np.ndarray], fit: bool = True) -> None:
        self.clear_overlays()
        if arr is None:
            self._img.setPixmap(QPixmap())
            self._shape = (0, 0)
            return
        new_shape = tuple(arr.shape[:2])
        self._img.setPixmap(QPixmap.fromImage(to_qimage(arr)))
        self.scene().setSceneRect(QRectF(0, 0, arr.shape[1], arr.shape[0]))
        if fit or new_shape != self._shape:
            self._shape = new_shape
            self.fit()

    def set_mask(self, mask: Optional[np.ndarray], rgba=(230, 60, 60, 110)) -> None:
        if mask is None:
            self._mask.setPixmap(QPixmap())
            return
        m = np.asarray(mask, dtype=bool)
        over = np.zeros(m.shape + (4,), np.uint8)
        over[m] = rgba
        self._mask.setPixmap(QPixmap.fromImage(to_qimage(over)))

    def set_polygons(self, name: str, polys: Sequence, color: str = "#ffd60a", width: float = 1.2,
                     fill: Optional[str] = None, z: float = 2) -> None:
        self._set_path(name, polygons_path(polys), color, width, fill, z)

    def set_points(self, name: str, pts: Sequence[Tuple[float, float]], color: str = "#30d158",
                   radius: float = 6, line: bool = False) -> None:
        path = QPainterPath()
        for x, y in pts:
            path.addEllipse(QPointF(x, y), radius, radius)
        if line and len(pts) == 2:
            path.moveTo(QPointF(*pts[0]))
            path.lineTo(QPointF(*pts[1]))
        self._set_path(name, path, color, 2.0, None, 5)

    def set_crosses(self, name: str, pts: Sequence[Tuple[float, float]], color: str = "red",
                    size: float = 9) -> None:
        """matplotlib ``marker="x"``: a cross of ``size`` screen pixels at each point."""
        self._marks[name] = (list(pts), color, size)
        self._draw_marks(name)

    def _draw_marks(self, name: str) -> None:
        pts, color, size = self._marks[name]
        k = self.transform().m11() or 1.0
        r = size / 2.0 / k
        path = QPainterPath()
        for x, y in pts:
            path.moveTo(QPointF(x - r, y - r))
            path.lineTo(QPointF(x + r, y + r))
            path.moveTo(QPointF(x - r, y + r))
            path.lineTo(QPointF(x + r, y - r))
        self._set_path(name, path, color, 2.0, None, 5)

    def set_lines(self, name: str, segs: Sequence[Sequence[Tuple[float, float]]], color: str = "red",
                  width: float = 1.5, dashed: bool = False, dots: float = 0, z: float = 4) -> None:
        """Polylines (``"-o"`` / ``"--o"`` style when ``dots`` > 0: end markers of that radius)."""
        path = QPainterPath()
        for seg in segs:
            seg = list(seg)
            if len(seg) >= 2:
                path.moveTo(QPointF(*seg[0]))
                for p in seg[1:]:
                    path.lineTo(QPointF(*p))
        self._set_path(name, path, color, width, None, z, Qt.DashLine if dashed else Qt.SolidLine)
        if dots:
            self._marks.pop(name + ":dots", None)
            k = self.transform().m11() or 1.0
            d = QPainterPath()
            for seg in segs:
                for x, y in seg:
                    d.addEllipse(QPointF(x, y), dots / k, dots / k)
            self._set_path(name + ":dots", d, color, 1.0, color, z)
        else:
            self._remove(name + ":dots")

    def set_label(self, name: str, x: float, y: float, text: str, color: str = "red") -> None:
        self._remove(name)
        item = self.scene().addSimpleText(text)
        item.setBrush(QBrush(QColor(color)))
        item.setFlag(item.GraphicsItemFlag.ItemIgnoresTransformations, True)
        item.setPos(QPointF(x, y))
        item.setZValue(6)
        self._layers[name] = item

    def _remove(self, name: str) -> None:
        old = self._layers.pop(name, None)
        if old is not None:
            self.scene().removeItem(old)

    def layer_names(self):
        return set(self._layers)

    def zoom_to(self, x0: float, y0: float, x1: float, y1: float) -> None:
        self.fitInView(QRectF(x0, y0, x1 - x0, y1 - y0), Qt.KeepAspectRatio)
        self._rescaled()

    def _rescaled(self) -> None:
        for name in list(self._marks):
            if name in self._layers:
                self._draw_marks(name)

    def _set_path(self, name, path, color, width, fill, z, style=Qt.SolidLine) -> None:
        old = self._layers.pop(name, None)
        if old is not None:
            self.scene().removeItem(old)
        pen = QPen(QColor(color), width, style)
        pen.setCosmetic(True)
        item = self.scene().addPath(path, pen, QBrush(QColor(fill)) if fill else QBrush(Qt.NoBrush))
        item.setZValue(z)
        self._layers[name] = item

    def clear_overlays(self) -> None:
        for item in self._layers.values():
            self.scene().removeItem(item)
        self._layers.clear()
        self._marks.clear()
        self.set_mask(None)

    def fit(self) -> None:
        if self._shape != (0, 0):
            self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)
            self._rescaled()

    # -- input ---------------------------------------------------------------
    def wheelEvent(self, ev) -> None:
        f = 1.0015 ** ev.angleDelta().y()
        self.scale(f, f)
        self._rescaled()

    def mouseDoubleClickEvent(self, ev) -> None:
        self.fit()

    def mousePressEvent(self, ev) -> None:
        self._press = ev.position()
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev) -> None:
        super().mouseMoveEvent(ev)
        if self._box_mode and self._press is not None and ev.buttons() & Qt.LeftButton:
            a, b = self.mapToScene(self._press.toPoint()), self.mapToScene(ev.position().toPoint())
            if self._band is None:
                pen = QPen(QColor("#64d2ff"), 1.5, Qt.DashLine)
                pen.setCosmetic(True)
                self._band = self.scene().addRect(QRectF(), pen)
                self._band.setZValue(6)
            self._band.setRect(QRectF(a, b).normalized())

    def mouseReleaseEvent(self, ev) -> None:
        super().mouseReleaseEvent(ev)
        if self._band is not None:
            self.scene().removeItem(self._band)
            self._band = None
            if self._press is not None and (ev.position() - self._press).manhattanLength() >= 5:
                x0, y0 = self._clip(self.mapToScene(self._press.toPoint()))
                x1, y1 = self._clip(self.mapToScene(ev.position().toPoint()))
                self._press = None
                self.boxDragged.emit(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
                return
        if self._press is not None and (ev.position() - self._press).manhattanLength() < 5:
            p = self.mapToScene(ev.position().toPoint())
            if 0 <= p.x() < self._shape[1] and 0 <= p.y() < self._shape[0]:
                self.clicked.emit(float(p.x()), float(p.y()), ev.button())
        self._press = None
        self.viewport().setCursor(Qt.ArrowCursor)

    def keyPressEvent(self, ev) -> None:
        if ev.key() == Qt.Key_F:
            self.fit()
            return
        k = ev.key()
        if k in (Qt.Key_Return, Qt.Key_Enter):
            self.keyPressed.emit("enter")
        elif k == Qt.Key_Escape:
            self.keyPressed.emit("esc")
        elif k in (Qt.Key_Delete, Qt.Key_Backspace):
            self.keyPressed.emit("delete")
        elif Qt.Key_A <= k <= Qt.Key_Z:  # also with Ctrl / Cmd held (Cmd+Z = undo)
            self.keyPressed.emit(chr(k).lower())
        elif ev.text():
            self.keyPressed.emit(ev.text().lower())
        super().keyPressEvent(ev)

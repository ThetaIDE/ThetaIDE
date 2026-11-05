"""Parallel coordinates with value-based brushes that survive live snapshot updates."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass

from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QLinearGradient, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import QWidget

from ..theme import theme_color

MISSING = object()


def numeric(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def category_key(value):
    return json.dumps(value, sort_keys=True)


def display(value):
    return f"{value:.4g}" if numeric(value) else str(value)


@dataclass
class Axis:
    key: tuple[str, str]
    title: str
    low: float = 0
    high: float = 1
    categories: tuple = ()

    def __post_init__(self):
        self._positions = {category_key(value): index / (len(self.categories) - 1) if len(self.categories) > 1 else 0.5
                           for index, value in enumerate(self.categories)}

    def value(self, trial):
        if self.key[0] == "objective":
            value = trial.get("value", MISSING)
            return value if numeric(value) else MISSING
        return trial.get("parameters", {}).get(self.key[1], MISSING)

    def fraction(self, value):
        if value is MISSING:
            return None
        if self.categories:
            return self._positions.get(category_key(value))
        if not numeric(value):
            return None
        return (value - self.low) / (self.high - self.low) if self.high != self.low else 0.5

    def accepts(self, trial, brush):
        value = self.value(trial)
        if isinstance(brush, set):
            return value is not MISSING and category_key(value) in brush
        return numeric(value) and brush[0] <= value <= brush[1]


class ParallelCoordinates(QWidget):
    selection_changed = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(340)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAccessibleName("Parallel coordinates: drag on axes to filter trials; Escape clears selections")
        self.study_key = None
        self.axes = []
        self.trials = []
        self.brushes = {}
        self.selected_ids = set()
        self.direction = "maximize"
        self._drag = None

    def set_study(self, study, key=None):
        if key != self.study_key:
            self.brushes.clear()
            self._drag = None
        self.study_key = key
        study = study or {}
        self.trials = study.get("trials", [])
        self.direction = study.get("direction", "maximize")
        self.axes = []
        names = sorted({name for trial in self.trials for name in trial.get("parameters", {})})
        for name in names:
            values = [trial["parameters"][name] for trial in self.trials if name in trial.get("parameters", {})]
            finite = [value for value in values if numeric(value)]
            if finite and all(isinstance(value, (float, int)) and not isinstance(value, bool) for value in values):
                self.axes.append(Axis(("parameter", name), name, min(finite), max(finite)))
            else:
                categories = tuple(value for _, value in sorted({category_key(value): value for value in values}.items()))
                self.axes.append(Axis(("parameter", name), name, categories=categories))
        if self.trials:
            values = [trial["value"] for trial in self.trials if numeric(trial.get("value"))]
            self.axes.append(Axis(("objective", "value"), study.get("objective_metric") or "Objective",
                                  min(values) if values else 0, max(values) if values else 1))
        keys = {axis.key for axis in self.axes}
        self.brushes = {key: brush for key, brush in self.brushes.items() if key in keys}
        if self._drag and self._drag[0] not in keys:
            self._drag = None
        self.selected_ids = self.matching_ids()
        self.setMinimumWidth(max(500, len(self.axes) * 170))
        self.update()

    def matching_ids(self):
        return {trial["id"] for trial in self.trials
                if all(axis.key not in self.brushes or axis.accepts(trial, self.brushes[axis.key]) for axis in self.axes)}

    def set_selected_ids(self, ids):
        self.selected_ids = set(ids)
        self.update()

    def clear_brushes(self):
        self.brushes.clear()
        self._drag = None
        self.selection_changed.emit()
        self.update()

    def plot_area(self):
        return QRectF(85, 85, max(1, self.width() - 170), max(1, self.height() - 155))

    def axis_x(self, index):
        area = self.plot_area()
        return area.left() + index * area.width() / max(1, len(self.axes) - 1)

    def _axis_at(self, point):
        area = self.plot_area()
        if not area.top() - 6 <= point.y() <= area.bottom() + 6:
            return None
        return next((i for i in range(len(self.axes)) if abs(self.axis_x(i) - point.x()) <= 16), None)

    def _fraction_at(self, y):
        area = self.plot_area()
        return max(0, min(1, (area.bottom() - y) / area.height()))

    def _brush_range(self, index, start, end):
        axis = self.axes[index]
        low, high = sorted((self._fraction_at(start), self._fraction_at(end)))
        if axis.categories:
            selected = {category_key(value) for value in axis.categories if low <= axis.fraction(value) <= high}
            if abs(start - end) < 4:
                selected = {category_key(min(axis.categories, key=lambda value: abs(axis.fraction(value) - low)))}
            self.brushes[axis.key] = selected
        else:
            span = axis.high - axis.low
            self.brushes[axis.key] = (axis.low + low * span, axis.low + high * span)
        self.selection_changed.emit()
        self.update()

    def mousePressEvent(self, event):
        index = self._axis_at(event.position())
        if index is not None and event.button() == Qt.MouseButton.RightButton:
            self.brushes.pop(self.axes[index].key, None)
            self.selection_changed.emit()
            self.update()
        elif index is not None and event.button() == Qt.MouseButton.LeftButton:
            self.setFocus()
            self._drag = (self.axes[index].key, event.position().y())
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag:
            key, start = self._drag
            index = next(i for i, axis in enumerate(self.axes) if axis.key == key)
            self._brush_range(index, start, event.position().y())
        else:
            index = self._axis_at(event.position())
            self.setCursor(Qt.CursorShape.CrossCursor if index is not None else Qt.CursorShape.ArrowCursor)
            if index is not None:
                self.setToolTip(f"{self.axes[index].title}\nDrag to select a range. Right-click to clear this axis.")
            else:
                self.setToolTip("")

    def mouseReleaseEvent(self, event):
        if self._drag and event.button() == Qt.MouseButton.LeftButton:
            key, start = self._drag
            index = next(i for i, axis in enumerate(self.axes) if axis.key == key)
            if abs(start - event.position().y()) < 4 and not self.axes[index].categories:
                self.brushes.pop(self.axes[index].key, None)
                self.selection_changed.emit()
                self.update()
            else:
                self._brush_range(index, start, event.position().y())
            self._drag = None
        else:
            super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.clear_brushes()
        else:
            super().keyPressEvent(event)

    def _color(self, trial):
        fraction = self.axes[-1].fraction(trial.get("value", MISSING))
        if fraction is None:
            return QColor(theme_color("muted"))
        if self.direction == "minimize":
            fraction = 1 - fraction
        low, high = QColor(theme_color("secondary")), QColor(theme_color("primary"))
        return QColor(*(round(a + (b - a) * fraction) for a, b in zip(low.getRgb()[:3], high.getRgb()[:3])))

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(theme_color("base")))
        painter.setPen(QColor(theme_color("muted")))
        if not self.axes:
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Trial parameters will appear here as the sweep runs.")
            return
        area = self.plot_area()
        # Draw rejected trials first so matching trials remain in front.
        for selected in (False, True):
            for trial in self.trials:
                if (trial["id"] in self.selected_ids) != selected:
                    continue
                color = self._color(trial)
                color.setAlpha(170 if selected else 25)
                painter.setPen(QPen(color, 1.4 if selected else 1))
                path = QPainterPath()
                connected = False
                for index, axis in enumerate(self.axes):
                    fraction = axis.fraction(axis.value(trial))
                    if fraction is None:
                        connected = False
                        continue
                    point = QPointF(self.axis_x(index), area.bottom() - fraction * area.height())
                    if connected:
                        path.lineTo(point)
                    else:
                        path.moveTo(point)
                    connected = True
                painter.drawPath(path)
        for index, axis in enumerate(self.axes):
            x = self.axis_x(index)
            painter.setPen(QPen(QColor(theme_color("muted")), 1))
            painter.drawLine(QPointF(x, area.top()), QPointF(x, area.bottom()))
            painter.setPen(QColor(theme_color("text")))
            title = ("Objective\n" if axis.key[0] == "objective" else "") + axis.title
            painter.drawText(QRectF(x - 78, 8, 156, 65), Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap, title)
            ticks = [(axis.fraction(value), display(value)) for value in axis.categories] if axis.categories else [
                (fraction, display(axis.low + fraction * (axis.high - axis.low)))
                for fraction in ([0.5] if axis.low == axis.high else [0, .25, .5, .75, 1])]
            stride = max(1, math.ceil(len(ticks) / max(2, area.height() // 24)))
            for fraction, text in ticks[::stride]:
                y = area.bottom() - fraction * area.height()
                painter.setPen(QColor(theme_color("muted")))
                painter.drawLine(QPointF(x - 4, y), QPointF(x + 4, y))
                painter.drawText(QRectF(x + 9, y - 10, 66, 20), Qt.AlignmentFlag.AlignVCenter,
                                 painter.fontMetrics().elidedText(text, Qt.TextElideMode.ElideRight, 66))
            brush = self.brushes.get(axis.key)
            if brush is not None:
                if isinstance(brush, set):
                    fractions = [axis.fraction(json.loads(value)) for value in brush]
                elif axis.categories:
                    fractions = [axis.fraction(value) for value in axis.categories
                                 if numeric(value) and brush[0] <= value <= brush[1]]
                else:
                    fractions = [axis.fraction(brush[0]), axis.fraction(brush[1])]
                fractions = [fraction for fraction in fractions if fraction is not None]
                if fractions:
                    top, bottom = max(0, min(1, max(fractions))), max(0, min(1, min(fractions)))
                    bounds = QRectF(x - 8, area.bottom() - top * area.height() - 4, 16, (top - bottom) * area.height() + 8)
                    fill = QColor(theme_color("primary"))
                    fill.setAlpha(65)
                    painter.fillRect(bounds, fill)
                    painter.setPen(QColor(theme_color("primary")))
                    painter.drawRect(bounds)
        gradient = QLinearGradient(85, 0, 205, 0)
        gradient.setColorAt(0, QColor(theme_color("secondary")))
        gradient.setColorAt(1, QColor(theme_color("primary")))
        painter.fillRect(QRectF(85, self.height() - 36, 120, 8), gradient)
        painter.setPen(QColor(theme_color("muted")))
        painter.drawText(QRectF(85, self.height() - 23, 250, 20), "Worse → better objective")

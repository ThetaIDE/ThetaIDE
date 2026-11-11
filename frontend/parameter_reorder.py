"""Reorder existing parameter layouts without recreating their editors.

Labels start a drag. Editors keep their normal mouse gestures, and a drag
belongs to exactly one sibling list for its entire life.
"""
import json
from dataclasses import dataclass

from PyQt6 import sip
from PyQt6.QtCore import QEvent, QObject, QPoint, QRect, QRectF, Qt, QTimer
from PyQt6.QtGui import QColor, QCursor, QLinearGradient, QPainter, QPainterPath, QPen, QPixmap
from PyQt6.QtWidgets import QApplication, QFormLayout, QFrame, QLabel, QLayout, QWidget

from .theme import theme_color


def _object(item):
    if item is None:
        return None
    return item.widget() or item.layout()


def _widgets(obj):
    if isinstance(obj, QWidget):
        return [obj]
    if isinstance(obj, QLayout):
        return [widget for i in range(obj.count()) for widget in _widgets(_object(obj.itemAt(i)))]
    return []


def _hierarchy(obj):
    return isinstance(obj, QWidget) and obj.objectName().startswith("hierarchyBlock_")


def _group_caption(block):
    if not _hierarchy(block) or not block.form_layout.rowCount():
        return None
    field = _object(block.form_layout.itemAt(0, QFormLayout.ItemRole.FieldRole))
    if isinstance(field, QLayout) and field.property("parameterPinned"):
        return next((w for w in _widgets(field) if isinstance(w, QLabel)), None)
    return None


@dataclass(eq=False)
class ParameterGroup:
    # Each entry is a physical form row, or a layout/widget in a column.
    parts: list
    caption: QWidget | None
    key: str = ""

    def widgets(self):
        return [w for part in self.parts for obj in (part if isinstance(part, tuple) else (part,))
                for w in _widgets(obj)]

    def rect(self, host):
        rect = QRect()
        for widget in self.widgets():
            if not widget.isHidden() or widget.sizePolicy().retainSizeWhenHidden():
                rect = rect.united(QRect(widget.mapTo(host, QPoint()), widget.size()))
        return rect


def form_groups(form):
    """Keep a header/selector and its following child block together."""
    groups = []
    for index in range(form.rowCount()):
        caption = _object(form.itemAt(index, QFormLayout.ItemRole.LabelRole))
        field = _object(form.itemAt(index, QFormLayout.ItemRole.FieldRole))
        if caption is None and field is None:
            field = _object(form.itemAt(index, QFormLayout.ItemRole.SpanningRole))
        part = (caption, field)
        own_caption = _group_caption(field)
        if (caption is None or isinstance(caption, QLabel) and not caption.text()) and _hierarchy(field) and groups and own_caption is None:
            groups[-1].parts.append(part)
        else:
            handle = own_caption or caption
            if (handle is None or isinstance(handle, QLabel) and not handle.text()) and isinstance(field, QLayout):
                handle = next((w for w in _widgets(field) if isinstance(w, QLabel)), None)
            if handle is not None and handle.objectName() in ("cardTitle", "heading"):
                handle = None
            if isinstance(field, QLayout) and field.property("parameterPinned"):
                handle = None
            groups.append(ParameterGroup([part], handle))
    return groups


class FormRows:
    def __init__(self, form):
        self.layout = form
        self.groups = form_groups(form)

    def apply(self):
        # takeRow transfers ownership; removeRow would destroy live editors.
        while self.layout.rowCount():
            self.layout.takeRow(0)
        for group in self.groups:
            for caption, field in group.parts:
                if caption is None:
                    self.layout.addRow(field)
                else:
                    self.layout.addRow(caption, field)
        self.layout.activate()


class ColumnRows:
    """Two-column presentation with logical siblings and optional shared rows."""
    def __init__(self, grid, columns, groups):
        self.layout, self.columns, self.groups = grid, columns, groups

    def apply(self):
        if not self.columns:
            # Use physical rows so a later sibling can fill the cell beside a
            # child block, without waiting for the whole parent group to end.
            while self.layout.count():
                self.layout.takeAt(0)
            rows = [0, 0]
            for group in self.groups:
                column = 0 if rows[0] <= rows[1] else 1
                for part in group.parts:
                    if isinstance(part, QWidget):
                        self.layout.addWidget(part, rows[column], column, Qt.AlignmentFlag.AlignTop)
                    else:
                        self.layout.addLayout(part, rows[column], column, Qt.AlignmentFlag.AlignTop)
                    rows[column] += 1
            self.layout.setVerticalSpacing(8)
            self.layout.activate()
            return
        for column in self.columns:
            while column.count():
                column.takeAt(0)
        heights = [0, 0]
        for group in self.groups:
            column = 0 if heights[0] <= heights[1] else 1
            for part in group.parts:
                if isinstance(part, QWidget):
                    self.columns[column].addWidget(part)
                else:
                    self.columns[column].addLayout(part)
                heights[column] += max(1, part.sizeHint().height())
        for column in self.columns:
            column.addStretch()
        self.layout.activate()


class ListRows:
    """Shared method parameters retain their specialized horizontal rows."""
    def __init__(self, layout):
        self.layout = layout
        self.groups = []
        for i in range(layout.count()):
            obj = _object(layout.itemAt(i))
            if obj is None:
                continue
            own_caption = _group_caption(obj)
            if _hierarchy(obj) and own_caption is None and self.groups and self.groups[-1].caption is not None:
                self.groups[-1].parts.append(obj)
                continue
            caption = None
            if i > 0:
                if isinstance(obj, QLayout):
                    caption = next((w for w in _widgets(obj) if isinstance(w, QLabel)), None)
                elif _hierarchy(obj):
                    caption = own_caption
            self.groups.append(ParameterGroup([obj], caption))

    def apply(self):
        while self.layout.count():
            self.layout.takeAt(0)
        for group in self.groups:
            for part in group.parts:
                if isinstance(part, QWidget):
                    self.layout.addWidget(part)
                else:
                    self.layout.addLayout(part)
        self.layout.activate()


class FlowRows(ListRows):
    """A wrapping list of self-contained parameter controls."""

    def __init__(self, layout):
        self.layout = layout
        self.groups = []
        for i in range(layout.count()):
            widget = layout.itemAt(i).widget()
            caption = next(iter(widget.findChildren(QLabel)), None)
            self.groups.append(ParameterGroup([widget], caption))


class GridRows:
    """Label/editor pairs in the experiment's two-column budget grid."""
    def __init__(self, grid):
        self.layout = grid
        self.groups = []
        pairs = {}
        children = {}
        for i in range(grid.count()):
            item = grid.itemAt(i)
            row, col, _, span = grid.getItemPosition(i)
            obj = _object(item)
            if span == 2 and _hierarchy(obj):
                children[(row - 1, col)] = obj
            elif span == 1:
                pairs[(row, col)] = obj
        for (row, col), caption in sorted(pairs.items()):
            if col % 2 == 0 and isinstance(caption, QLabel) and (row, col + 1) in pairs:
                parts = [(caption, pairs[(row, col + 1)])]
                if (row, col) in children:
                    parts.append((None, children[(row, col)]))
                self.groups.append(ParameterGroup(parts, caption))

    def apply(self):
        while self.layout.count():
            self.layout.takeAt(0)
        row = 0
        for start in range(0, len(self.groups), 2):
            pair = self.groups[start:start + 2]
            for column, group in enumerate(pair):
                caption, field = group.parts[0]
                self.layout.addWidget(caption, row, column * 2)
                if isinstance(field, QWidget):
                    self.layout.addWidget(field, row, column * 2 + 1)
                else:
                    self.layout.addLayout(field, row, column * 2 + 1)
                for _, child in group.parts[1:]:
                    self.layout.addWidget(child, row + 1, column * 2, 1, 2)
            row += 2 if any(len(group.parts) > 1 for group in pair) else 1
        self.layout.activate()


class _DragCard(QWidget):
    def __init__(self, host, rect):
        super().__init__(host, Qt.WindowType.ToolTip | Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        # Preserve GUI scale and width; large nested groups trail off vertically.
        size = rect.size()
        size.setHeight(min(size.height(), 100))
        self.resize(size)
        ratio = host.devicePixelRatioF()
        self.pixmap = QPixmap(size * ratio)
        self.pixmap.setDevicePixelRatio(ratio)
        self.pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(self.pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(QColor(theme_color("surface")))
        rounded = QPainterPath()
        rounded.addRoundedRect(QRectF(self.rect()), 6, 6)
        painter.setClipPath(rounded)
        painter.fillPath(rounded, QColor(theme_color("surface")))
        painter.drawPixmap(0, 0, host.grab(QRect(rect.topLeft(), size)))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor(theme_color("border")), 1))
        painter.drawRoundedRect(self.rect().adjusted(0, 0, -1, -1), 6, 6)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_DestinationIn)
        fade = QLinearGradient(0, 0, self.width(), 0)
        fade.setColorAt(0, QColor(0, 0, 0, 235))
        fade.setColorAt(0.55, QColor(0, 0, 0, 220))
        fade.setColorAt(1, QColor(0, 0, 0, 0))
        painter.fillRect(self.rect(), fade)
        if rect.height() > size.height():
            fade = QLinearGradient(0, 0, 0, self.height())
            fade.setColorAt(0, QColor(0, 0, 0, 255))
            fade.setColorAt(0.35, QColor(0, 0, 0, 255))
            fade.setColorAt(1, QColor(0, 0, 0, 0))
            painter.fillRect(self.rect(), fade)
        painter.end()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawPixmap(0, 0, self.pixmap)


class ParameterReorder(QObject):
    def __init__(self, rows, host, scroll, order, save):
        super().__init__(host)
        self.rows, self.host, self.scroll, self.save = rows, host, scroll, save
        self.dragged = None
        self.pressed = None
        self.preview = None
        self.placeholder = None
        self.original = None
        self.hidden = []
        self._timer = QTimer(self)
        self._timer.setInterval(30)
        self._timer.timeout.connect(self._autoscroll)
        counts = {}
        for group in rows.groups:
            caption = group.caption
            if caption is None:
                continue
            text = caption.text() if isinstance(caption, QLabel) else next(
                (w.text() for w in caption.findChildren(QLabel)), "")
            counts[text] = counts.get(text, 0) + 1
            group.key = json.dumps([text, counts[text]])
            caption.installEventFilter(self)
            for child in caption.findChildren(QLabel):
                child.installEventFilter(self)
        self.order = list(order)
        if order:
            movable = sorted((g for g in rows.groups if g.caption is not None),
                             key=lambda g: order.index(g.key) if g.key in order else len(order))
            iterator = iter(movable)
            rows.groups[:] = [next(iterator) if g.caption is not None else g for g in rows.groups]
            rows.apply()

    def eventFilter(self, watched, event):
        kind = event.type()
        if self.dragged is not None:
            if kind == QEvent.Type.MouseMove:
                self.move(event.globalPosition().toPoint())
                return True
            if kind == QEvent.Type.MouseButtonRelease and event.button() == Qt.MouseButton.LeftButton:
                self.finish(self._valid_position(event.globalPosition().toPoint()))
                return True
            if kind == QEvent.Type.KeyPress and event.key() == Qt.Key.Key_Escape:
                self.finish(False)
                return True
            if kind == QEvent.Type.ApplicationDeactivate or (
                kind in (QEvent.Type.WindowDeactivate, QEvent.Type.Hide) and watched in (self.host, self.host.window())
            ):
                self.finish(False)
        elif kind == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
            group = next((g for g in self.rows.groups if g.caption is watched or
                          g.caption is not None and g.caption.isAncestorOf(watched)), None)
            if group and len(self.rows.groups) > 1 and watched.isEnabled():
                self.pressed = (group, event.globalPosition().toPoint())
        elif kind == QEvent.Type.MouseMove and self.pressed:
            if not event.buttons() & Qt.MouseButton.LeftButton:
                self.pressed = None
            elif (event.globalPosition().toPoint() - self.pressed[1]).manhattanLength() >= QApplication.startDragDistance():
                group, origin = self.pressed
                self.begin(group, origin)
                self.move(event.globalPosition().toPoint())
                return True
        elif kind == QEvent.Type.MouseButtonRelease:
            self.pressed = None
        return False

    def begin(self, group, position):
        self.pressed = None
        self.dragged = group
        self.original = list(self.rows.groups)
        rect = group.rect(self.host)
        self.offset = self.host.mapFromGlobal(position) - rect.topLeft()
        self.offset.setY(min(self.offset.y(), 40))
        self.preview = _DragCard(self.host, rect)
        for widget in group.widgets():
            policy = widget.sizePolicy()
            self.hidden.append((widget, widget.isHidden(), policy.retainSizeWhenHidden()))
            policy.setRetainSizeWhenHidden(True)
            widget.setSizePolicy(policy)
            widget.hide()
        self.placeholder = QFrame(self.host)
        self.placeholder.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.placeholder.setObjectName("parameterDropSpace")
        self.placeholder.setStyleSheet(
            f"#parameterDropSpace {{ border: 1px dashed {theme_color('border')}; border-radius: 5px; "
            f"background: {theme_color('base')}; }}")
        self.placeholder.setGeometry(rect)
        self.placeholder.show()
        QApplication.instance().installEventFilter(self)
        self.host.installEventFilter(self)
        # Keep receiving the release when the pointer leaves the app window.
        self.host.grabMouse()
        self.preview.show()
        QApplication.setOverrideCursor(Qt.CursorShape.ClosedHandCursor)
        self._timer.start()

    def _valid_position(self, position):
        if not self.scroll.viewport().rect().contains(self.scroll.viewport().mapFromGlobal(position)):
            return False
        point = self.host.mapFromGlobal(position)
        bounds = QRect()
        for group in self.rows.groups:
            bounds = bounds.united(group.rect(self.host))
        return bounds.adjusted(-8, -8, 8, 8).contains(point)

    def move(self, position):
        self.preview.move(position - self.offset)
        if not self._valid_position(position):
            return
        point = self.host.mapFromGlobal(position)
        target = next((group for group in self.rows.groups if group.rect(self.host).contains(point)), None)
        if target is None or target is self.dragged or target.caption is None:
            return
        movable = [g for g in self.rows.groups if g.caption is not None]
        source_index, target_index = movable.index(self.dragged), movable.index(target)
        movable.pop(source_index)
        movable.insert(target_index, self.dragged)
        iterator = iter(movable)
        self.rows.groups[:] = [next(iterator) if g.caption is not None else g for g in self.rows.groups]
        self.rows.apply()
        self.placeholder.setGeometry(self.dragged.rect(self.host))

    def _autoscroll(self):
        if self.dragged is None:
            return
        position = QCursor.pos()
        viewport = self.scroll.viewport()
        point = viewport.mapFromGlobal(position)
        if not viewport.rect().contains(point):
            return
        delta = -12 if point.y() < 36 else 12 if point.y() > viewport.height() - 36 else 0
        if delta:
            bar = self.scroll.verticalScrollBar()
            bar.setValue(bar.value() + delta)
            self.move(position)

    def finish(self, accepted):
        QApplication.instance().removeEventFilter(self)
        if QWidget.mouseGrabber() is self.host:
            self.host.releaseMouse()
        self._timer.stop()
        QApplication.restoreOverrideCursor()
        if not accepted:
            self.rows.groups[:] = self.original
            self.rows.apply()
        self.preview.close()
        self.preview.deleteLater()
        self.placeholder.deleteLater()
        self.placeholder.hide()
        for widget, hidden, retain in self.hidden:
            if not sip.isdeleted(widget):
                policy = widget.sizePolicy()
                policy.setRetainSizeWhenHidden(retain)
                widget.setSizePolicy(policy)
                widget.setVisible(not hidden)
        self.hidden.clear()
        changed = self.rows.groups != self.original
        self.dragged = self.preview = self.original = None
        if accepted and changed:
            visible = [group.key for group in self.rows.groups if group.key]
            # Retain absent optional parameters at their old positions.
            iterator = iter(visible)
            order = [next(iterator) if key in visible else key for key in self.order]
            order.extend(key for key in visible if key not in order)
            self.order = order
            self.save(order)

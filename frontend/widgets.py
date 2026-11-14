import math
import re
from PyQt6.QtCore import Qt, QEvent, QRect, QRectF, QSize, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QDoubleValidator, QPainter, QPen, QSyntaxHighlighter, QTextCharFormat, QTextOption
from PyQt6.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QHBoxLayout, QFrame, QAbstractButton,
    QComboBox, QDoubleSpinBox, QSpinBox, QTableWidgetItem, QScrollArea,
    QToolButton, QMenu, QInputDialog, QAbstractSpinBox, QTextEdit, QSizePolicy, QLineEdit, QLayout,
)
from .theme import theme_color
from .charts.chart import Chart, format_tick, nice_step  # noqa: F401  (moved; re-exported)


class _WheelNeedsFocus:
    """Only edit the value when the widget has been clicked into.

    Qt gives spin boxes and combo boxes WheelFocus by default, so they take
    focus from a passing wheel event and then consume it. Scrolling a long form
    silently rewrites every field the pointer crosses. Ignoring the event
    instead lets it reach the scroll area, which is what the user meant.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def wheelEvent(self, event):
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()


class SpinBox(_WheelNeedsFocus, QSpinBox):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)


class IntegerField(QLineEdit):
    """Exact Python integer editor when a value/range exceeds Qt's 32-bit spin box."""

    valueChanged = pyqtSignal(object)

    def __init__(self, value=0, minimum=None, maximum=None, parent=None):
        super().__init__(parent)
        self._minimum, self._maximum = minimum, maximum
        self._value = int(value)
        self.setText(str(self._value))
        self.setToolTip("Enter a whole number; large integers are preserved exactly.")
        self.editingFinished.connect(self._commit)

    def value(self):
        return self._value

    def setValue(self, value):
        value = int(value)
        changed = value != self._value
        self._value = value
        self.setText(str(value))
        if changed:
            self.valueChanged.emit(value)

    def _commit(self):
        text = self.text().strip()
        try:
            if not re.fullmatch(r"[+-]?[0-9]+", text): raise ValueError('Expected integer')
            value = int(text)
            if self._minimum is not None and value < self._minimum: raise ValueError('Below minimum')
            if self._maximum is not None and value > self._maximum: raise ValueError('Above maximum')
        except ValueError:
            self.setText(str(self._value))
            return
        self.setValue(value)


def integer_field(value, *, minimum=None, maximum=None):
    """Use ordinary spin controls when safe, otherwise an exact integer field."""
    low, high = -(2**31), 2**31 - 1
    if (not low <= value <= high or (minimum is not None and not low <= minimum <= high)
            or (maximum is not None and not low <= maximum <= high)
            or (minimum is not None and value < minimum) or (maximum is not None and value > maximum)):
        return IntegerField(value, minimum, maximum)
    widget = SpinBox()
    widget.setRange(low if minimum is None else minimum, high if maximum is None else maximum)
    widget.setValue(value)
    return widget


class DoubleSpinBox(_WheelNeedsFocus, QDoubleSpinBox):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)


class ScientificDoubleSpinBox(DoubleSpinBox):
    """Compact, round-trip float editing, including scientific notation."""

    def __init__(self, value=0.0, parent=None):
        super().__init__(parent)
        self.setDecimals(323)
        limit = float.fromhex("0x1.fffffffffffffp+1023")
        self.setRange(-limit, limit)
        self._validator = QDoubleValidator(self)
        self._validator.setNotation(QDoubleValidator.Notation.ScientificNotation)
        self.setValue(value)

    def textFromValue(self, value):
        return str(value)

    def valueFromText(self, text):
        try:
            return float(text)
        except ValueError:
            return self.value()

    def validate(self, text, position):
        if not hasattr(self, "_validator"):
            return super().validate(text, position)
        return self._validator.validate(text, position)


class FlowLayout(QLayout):
    """Keep compact controls together, wrapping complete controls as width changes."""

    def __init__(self, parent=None, *, spacing=12, vertical_alignment=Qt.AlignmentFlag.AlignTop):
        super().__init__(parent)
        self._items = []
        self._vertical_alignment = vertical_alignment
        self.setContentsMargins(0, 0, 0, 0)
        self.setSpacing(spacing)

    def addItem(self, item):
        self._items.append(item)
        self.invalidate()

    def count(self):
        return len(self._items)

    def itemAt(self, index):
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index):
        if 0 <= index < len(self._items):
            item = self._items.pop(index)
            self.invalidate()
            return item
        return None

    def expandingDirections(self):
        return Qt.Orientation(0)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self._arrange(QRect(0, 0, width, 0), apply=False)

    def minimumSize(self):
        size = QSize(0, 0)
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        margins = self.contentsMargins()
        return size + QSize(margins.left() + margins.right(), margins.top() + margins.bottom())

    def sizeHint(self):
        return self.minimumSize()

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._arrange(rect, apply=True)

    def _arrange(self, rect, *, apply):
        margins = self.contentsMargins()
        area = rect.adjusted(margins.left(), margins.top(), -margins.right(), -margins.bottom())
        x, y, height = area.x(), area.y(), 0
        line = []

        def place_line():
            if not apply:
                return
            for item, left, size in line:
                offset = (height - size.height()) // 2 if self._vertical_alignment == Qt.AlignmentFlag.AlignVCenter else 0
                item.setGeometry(QRect(left, y + offset, size.width(), size.height()))

        for item in self._items:
            if item.isEmpty():
                continue
            size = item.sizeHint()
            if x > area.x() and x + size.width() > area.x() + area.width():
                place_line()
                line = []
                x = area.x()
                y += height + self.spacing()
                height = 0
            line.append((item, x, size))
            x += size.width() + self.spacing()
            height = max(height, size.height())
        place_line()
        return y + height - rect.y() + margins.bottom()


class WrappingFlowWidget(QWidget):
    """A flow row that reserves enough height inside nested forms and grids."""

    def __init__(self, parent=None, *, spacing=12, vertical_alignment=Qt.AlignmentFlag.AlignTop, fit_contents=False):
        super().__init__(parent)
        self._fit_contents = fit_contents
        FlowLayout(self, spacing=spacing, vertical_alignment=vertical_alignment)
        policy = QSizePolicy(QSizePolicy.Policy.Maximum if fit_contents else QSizePolicy.Policy.Expanding,
                             QSizePolicy.Policy.Preferred)
        policy.setHeightForWidth(True)
        self.setSizePolicy(policy)

    def sizeHint(self):
        if not self._fit_contents:
            return super().sizeHint()
        layout = self.layout()
        items = [layout.itemAt(i) for i in range(layout.count()) if not layout.itemAt(i).isEmpty()]
        margins = layout.contentsMargins()
        width = sum(item.sizeHint().width() for item in items) + max(0, len(items) - 1) * layout.spacing()
        width += margins.left() + margins.right()
        return QSize(width, layout.heightForWidth(width))

    def _update_height(self):
        layout = self.layout()
        if layout is not None:
            height = layout.heightForWidth(self.width())
            if self.minimumHeight() != height:
                self.setMinimumHeight(height)
                self.updateGeometry()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_height()

    def event(self, event):
        result = super().event(event)
        if event.type() == QEvent.Type.LayoutRequest:
            self._update_height()
        return result


class ComboBox(_WheelNeedsFocus, QComboBox):
    pass


class ExpandingTextEdit(QTextEdit):
    """Plain text form editor that grows with wrapped lines instead of scrolling."""

    contentHeightChanged = pyqtSignal(int)

    def __init__(self, text="", parent=None):
        super().__init__(parent)
        self._adjusting_height = False
        self._height_layout_timer = QTimer(self)
        self._height_layout_timer.setSingleShot(True)
        self._height_layout_timer.timeout.connect(self._relayout_parent)
        self.setProperty("parameterText", True)
        self.setAcceptRichText(False)
        self.setTabChangesFocus(True)
        self.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        self.setWordWrapMode(QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.document().setDocumentMargin(2)
        self.document().documentLayout().documentSizeChanged.connect(self._adjust_height)
        self.setPlainText(text)
        self._adjust_height()

    def _adjust_height(self, *_):
        if self._adjusting_height:
            return
        self._adjusting_height = True
        try:
            content = self.document().documentLayout().documentSize().height()
            chrome = self.height() - self.viewport().height()
            height = math.ceil(max(content, self.fontMetrics().lineSpacing()) + chrome)
            if self.height() != height:
                self.setFixedHeight(height)
                self.contentHeightChanged.emit(height)
                # Cocoa can defer the containing form's LayoutRequest past
                # the next paint. Commit this height before sibling regions
                # are painted, so growing text never covers their editors.
                self.updateGeometry()
                self._height_layout_timer.start(0)
        finally:
            self._adjusting_height = False

    def _relayout_parent(self):
        parent = self.parentWidget()
        layout = QWidget.layout(parent) if parent is not None else None
        if layout is not None:
            layout.invalidate()
            layout.activate()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._adjust_height()


class HierarchicalComboBox(ComboBox):
    """Keep full path values while browsing slash-separated items in submenus."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.popup_menu = QMenu(self)

    def _populate_menu(self):
        self.popup_menu.clear()
        folders = {(): self.popup_menu}
        for index in range(self.count()):
            parts = self.itemText(index).split("/")
            parent = ()
            for part in parts[:-1]:
                path = (*parent, part)
                if path not in folders:
                    folders[path] = folders[parent].addMenu(part)
                parent = path
            action = folders[parent].addAction(parts[-1] or "(none)")
            action.setData(self.itemData(index))
            action.setCheckable(True)
            action.setChecked(index == self.currentIndex())
            action.triggered.connect(lambda _checked=False, i=index: self.setCurrentIndex(i))

    def showPopup(self):
        self._populate_menu()
        try:
            self.popup_menu.exec(self.mapToGlobal(self.rect().bottomLeft()))
        finally:
            # Reset QComboBox's popup state after closing the custom menu.
            super().hidePopup()

    def hidePopup(self):
        self.popup_menu.close()
        super().hidePopup()


class SmoothScrollArea(QScrollArea):
    """QScrollArea that respects pixelDelta for high-precision, smooth trackpad and mouse scrolling."""

    def viewportEvent(self, event):
        if event.type() == event.Type.Wheel:
            pixel_delta = event.pixelDelta()
            if not pixel_delta.isNull():
                handled = False
                if pixel_delta.y() != 0 and self.verticalScrollBar().maximum() > 0:
                    bar = self.verticalScrollBar()
                    bar.setValue(bar.value() - pixel_delta.y())
                    handled = True
                if pixel_delta.x() != 0 and self.horizontalScrollBar().maximum() > 0:
                    hbar = self.horizontalScrollBar()
                    hbar.setValue(hbar.value() - pixel_delta.x())
                    handled = True
                if handled:
                    event.accept()
                    return True
            angle_delta = event.angleDelta()
            if not angle_delta.isNull():
                handled = False
                if angle_delta.y() != 0 and self.verticalScrollBar().maximum() > 0:
                    bar = self.verticalScrollBar()
                    step = bar.singleStep() if bar.singleStep() > 0 else 20
                    delta = int(round(angle_delta.y() / 120.0 * step * 2))
                    bar.setValue(bar.value() - delta)
                    handled = True
                if angle_delta.x() != 0 and self.horizontalScrollBar().maximum() > 0:
                    hbar = self.horizontalScrollBar()
                    step = hbar.singleStep() if hbar.singleStep() > 0 else 20
                    delta = int(round(angle_delta.x() / 120.0 * step * 2))
                    hbar.setValue(hbar.value() - delta)
                    handled = True
                if handled:
                    event.accept()
                    return True
        return super().viewportEvent(event)

    def wheelEvent(self, event):
        pixel_delta = event.pixelDelta()
        if not pixel_delta.isNull():
            handled = False
            if pixel_delta.y() != 0 and self.verticalScrollBar().maximum() > 0:
                bar = self.verticalScrollBar()
                bar.setValue(bar.value() - pixel_delta.y())
                handled = True
            if pixel_delta.x() != 0 and self.horizontalScrollBar().maximum() > 0:
                hbar = self.horizontalScrollBar()
                hbar.setValue(hbar.value() - pixel_delta.x())
                handled = True
            if handled:
                event.accept()
                return
        angle_delta = event.angleDelta()
        if not angle_delta.isNull():
            handled = False
            if angle_delta.y() != 0 and self.verticalScrollBar().maximum() > 0:
                bar = self.verticalScrollBar()
                step = bar.singleStep() if bar.singleStep() > 0 else 20
                delta = int(round(angle_delta.y() / 120.0 * step * 2))
                bar.setValue(bar.value() - delta)
                handled = True
            if angle_delta.x() != 0 and self.horizontalScrollBar().maximum() > 0:
                hbar = self.horizontalScrollBar()
                step = hbar.singleStep() if hbar.singleStep() > 0 else 20
                delta = int(round(angle_delta.x() / 120.0 * step * 2))
                hbar.setValue(hbar.value() - delta)
                handled = True
            if handled:
                event.accept()
                return
        super().wheelEvent(event)


def label(text, kind=None, wrap=False):
    """A QLabel styled by `kind` (its object name). wrap=True lets long text flow onto more lines
    instead of setting a minimum width for its whole layout."""
    result = QLabel(text)
    if kind:
        result.setObjectName(kind)
    result.setWordWrap(wrap)
    return result


class SortableItem(QTableWidgetItem):
    """Table cell that sorts by a value (number, timestamp or text) rather than by its displayed text.

    Cells without a value (shown as "—") sort last in either direction.
    """
    def __init__(self, text, key):
        super().__init__(text)
        self.key = key

    def __lt__(self, other):
        if not isinstance(other, SortableItem):
            return super().__lt__(other)
        if self.key is None and other.key is None:
            return False
        if self.key is None or other.key is None:
            table = self.tableWidget()
            descending = (table is not None
                          and table.horizontalHeader().sortIndicatorOrder() == Qt.SortOrder.DescendingOrder)
            return descending if self.key is None else not descending
        return self.key < other.key


class ElidedLabel(QLabel):
    """A one-line label that ends in … when it is given less width than its text needs."""

    def minimumSizeHint(self):
        hint = super().minimumSizeHint()
        hint.setWidth(min(hint.width(), 60))
        return hint

    def paintEvent(self, event):
        painter = QPainter(self)
        text = self.fontMetrics().elidedText(self.text(), Qt.TextElideMode.ElideRight, self.contentsRect().width())
        painter.setPen(self.palette().color(self.foregroundRole()))
        painter.drawText(self.contentsRect(), int(self.alignment() | Qt.AlignmentFlag.AlignVCenter), text)


class MetricCard(QFrame):
    def __init__(self, title, subtitle):
        super().__init__()
        self.setObjectName("card")
        layout = QVBoxLayout(self)
        self.title = label(title, "muted")
        layout.addWidget(self.title)
        self.value = label("—", "value")
        layout.addWidget(self.value)
        self.subtitle = label(subtitle, "muted")
        layout.addWidget(self.subtitle)


class YamlHighlighter(QSyntaxHighlighter):
    def highlightBlock(self, text):
        for pattern, color in ((r'^\s*[\w_]+(?=:)', '#83a598'),
                               (r'\b\d+(?:\.\d+)?\b', '#d3869b'),
                               (r'"[^"\n]*"', '#b8bb26'), (r'#.*$', '#928374')):
            fmt = QTextCharFormat()
            fmt.setForeground(QColor(theme_color(color)))
            for match in re.finditer(pattern, text):
                self.setFormat(match.start(), len(match.group()), fmt)


class ToggleSlider(QAbstractButton):
    """Instant toggle slider switch with theme-aware pill track and knob."""

    def __init__(self, checked=True, parent=None):
        super().__init__(parent)
        self.setCheckable(True)
        self.setChecked(checked)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedSize(40, 22)

    def sizeHint(self):
        return QSize(40, 22)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(0.5, 0.5, self.width() - 1, self.height() - 1)
        radius = rect.height() / 2
        is_on = self.isChecked()

        track_color = QColor(theme_color("primary" if is_on else "raised"))
        border_color = QColor(theme_color("primary" if is_on else "border"))
        if not self.isEnabled():
            track_color.setAlpha(120)
            border_color.setAlpha(120)
        painter.setPen(QPen(border_color, 1))
        painter.setBrush(track_color)
        painter.drawRoundedRect(rect, radius, radius)

        thumb_diameter = rect.height() - 6
        thumb_y = 3.0
        thumb_x = (rect.width() - thumb_diameter - 3.0) if is_on else 3.0
        knob_color = QColor(theme_color("base" if is_on else "muted"))
        if not self.isEnabled():
            knob_color.setAlpha(150)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(knob_color)
        painter.drawEllipse(QRectF(thumb_x, thumb_y, thumb_diameter, thumb_diameter))
        painter.end()


class TagCloudWidget(QWidget):
    """Interactive tag cloud / chip editor for lists of strings (e.g. compatible_paradigms)."""
    tagsChanged = pyqtSignal(list)

    def __init__(self, tags: list[str] | None = None, available_options: list[str] | None = None, parent=None):
        super().__init__(parent)
        self._tags = [str(t) for t in tags] if tags else []
        self._available_options = [str(opt) for opt in available_options] if available_options else []
        self._init_ui()

    def _init_ui(self):
        self.main_layout = QHBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(6)
        self._render_chips()

    def set_available_options(self, options: list[str]):
        self._available_options = [str(opt) for opt in options] if options else []
        self._render_chips()

    def set_tags(self, tags: list[str]):
        self._tags = [str(t) for t in tags] if tags else []
        self._render_chips()

    def get_tags(self) -> list[str]:
        return list(self._tags)

    def _render_chips(self):
        while self.main_layout.count():
            item = self.main_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        for tag in self._tags:
            chip = QFrame()
            chip.setObjectName("tagChip")
            chip.setStyleSheet(f"""
                QFrame#tagChip {{
                    background-color: {theme_color('surface')};
                    border: 1px solid {theme_color('border')};
                    border-radius: 12px;
                    padding: 2px 6px;
                }}
            """)
            c_layout = QHBoxLayout(chip)
            c_layout.setContentsMargins(6, 2, 4, 2)
            c_layout.setSpacing(4)
            lbl = QLabel(tag)
            lbl.setStyleSheet(f"color: {theme_color('text')}; font-size: 12px; font-weight: 500;")
            c_layout.addWidget(lbl)

            btn_del = QToolButton()
            btn_del.setText("✕")
            btn_del.setCursor(Qt.CursorShape.PointingHandCursor)
            btn_del.setFixedSize(16, 16)
            btn_del.setStyleSheet(f"""
                QToolButton {{
                    background: transparent;
                    border: none;
                    color: {theme_color('muted')};
                    font-size: 11px;
                    padding: 0;
                }}
                QToolButton:hover {{
                    color: {theme_color('accent')};
                }}
            """)
            btn_del.clicked.connect(lambda _, t=tag: self._remove_tag(t))
            c_layout.addWidget(btn_del)
            self.main_layout.addWidget(chip)

        # Add button dropdown
        btn_add = QToolButton()
        btn_add.setText("+ Add Paradigm ▾")
        btn_add.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_add.setStyleSheet(f"""
            QToolButton {{
                background-color: transparent;
                border: 1px dashed {theme_color('border')};
                border-radius: 12px;
                color: {theme_color('muted')};
                padding: 3px 10px;
                font-size: 11px;
            }}
            QToolButton:hover {{
                border-color: {theme_color('accent')};
                color: {theme_color('text')};
                background-color: {theme_color('raised')};
            }}
            QToolButton::menu-indicator {{
                image: none;
                width: 0px;
            }}
        """)
        menu = QMenu(btn_add)
        menu.setStyleSheet(f"""
            QMenu {{
                background-color: {theme_color('panel')};
                border: 1px solid {theme_color('border')};
                border-radius: 4px;
                padding: 4px 0px;
            }}
            QMenu::item {{
                padding: 5px 18px 5px 12px;
                color: {theme_color('text')};
                font-size: 12px;
            }}
            QMenu::item:selected {{
                background-color: {theme_color('raised')};
                color: {theme_color('accent')};
            }}
        """)
        candidates = [opt for opt in self._available_options if opt not in self._tags]
        for cand in candidates:
            act = menu.addAction(cand)
            act.triggered.connect(lambda _, c=cand: self._add_tag(c))

        if candidates:
            menu.addSeparator()

        act_custom = menu.addAction("+ Custom...")
        act_custom.triggered.connect(self._prompt_custom_tag)

        btn_add.setMenu(menu)
        btn_add.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.main_layout.addWidget(btn_add)
        self.main_layout.addStretch()

    def _add_tag(self, tag: str):
        if tag and tag not in self._tags:
            self._tags.append(tag)
            self._render_chips()
            self.tagsChanged.emit(self._tags)

    def _remove_tag(self, tag: str):
        if tag in self._tags:
            self._tags.remove(tag)
            self._render_chips()
            self.tagsChanged.emit(self._tags)

    def _prompt_custom_tag(self):
        val, ok = QInputDialog.getText(self, "Add Tag", "Tag / Paradigm Name:")
        if ok and val.strip():
            self._add_tag(val.strip())


class LayerStackWidget(QWidget):
    """Interactive visual layer stack editor for network layers (hidden_sizes, hidden_dims).

    Renders as: [ 256 ✕ ] → [ 128 ✕ ] → [ 64 ✕ ]  [+ Layer]
    """
    layersChanged = pyqtSignal(list)

    def __init__(self, layers: list[int] | None = None, parent=None):
        super().__init__(parent)
        self._layers = []
        if layers:
            for x in layers:
                try:
                    self._layers.append(int(x))
                except (ValueError, TypeError):
                    pass
        if not self._layers:
            self._layers = [64, 64]
        self._init_ui()

    def _init_ui(self):
        self.main_layout = QHBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(6)
        self._render_blocks()

    def set_layers(self, layers: list[int]):
        self._layers = []
        if layers:
            for x in layers:
                try:
                    self._layers.append(int(x))
                except (ValueError, TypeError):
                    pass
        self._render_blocks()

    def get_layers(self) -> list[int]:
        return list(self._layers)

    def _render_blocks(self):
        while self.main_layout.count():
            item = self.main_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        for idx, size in enumerate(self._layers):
            if idx > 0:
                arrow = QLabel("→")
                arrow.setStyleSheet(f"color: {theme_color('muted')}; font-size: 14px; font-weight: bold;")
                self.main_layout.addWidget(arrow)

            block = QFrame()
            block.setObjectName("layerBlock")
            block.setStyleSheet(f"""
                QFrame#layerBlock {{
                    background-color: {theme_color('surface')};
                    border: 1px solid {theme_color('border')};
                    border-radius: 6px;
                    padding: 2px 6px;
                }}
            """)
            b_layout = QHBoxLayout(block)
            b_layout.setContentsMargins(4, 2, 4, 2)
            b_layout.setSpacing(4)

            spin = SpinBox()
            spin.setRange(1, 16384)
            spin.setValue(int(size))
            spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
            spin.setFixedWidth(54)
            spin.setStyleSheet(f"""
                QSpinBox {{
                    background: transparent;
                    border: none;
                    color: {theme_color('text')};
                    font-size: 12px;
                    font-weight: bold;
                    selection-background-color: {theme_color('raised')};
                }}
            """)
            spin.valueChanged.connect(lambda val, i=idx: self._on_layer_val_changed(i, val))
            b_layout.addWidget(spin)

            btn_del = QToolButton()
            btn_del.setText("✕")
            btn_del.setCursor(Qt.CursorShape.PointingHandCursor)
            btn_del.setFixedSize(14, 14)
            btn_del.setStyleSheet(f"""
                QToolButton {{
                    background: transparent;
                    border: none;
                    color: {theme_color('muted')};
                    font-size: 10px;
                    padding: 0;
                }}
                QToolButton:hover {{
                    color: {theme_color('accent')};
                }}
            """)
            btn_del.clicked.connect(lambda _, i=idx: self._remove_layer(i))
            b_layout.addWidget(btn_del)

            self.main_layout.addWidget(block)

        # Add Layer Button
        btn_add = QToolButton()
        btn_add.setText("+ Layer")
        btn_add.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_add.setStyleSheet(f"""
            QToolButton {{
                background-color: transparent;
                border: 1px dashed {theme_color('border')};
                border-radius: 6px;
                color: {theme_color('muted')};
                padding: 4px 8px;
                font-size: 11px;
                font-weight: 500;
            }}
            QToolButton:hover {{
                border-color: {theme_color('accent')};
                color: {theme_color('text')};
                background-color: {theme_color('raised')};
            }}
        """)
        btn_add.clicked.connect(self._add_layer)
        self.main_layout.addWidget(btn_add)
        self.main_layout.addStretch()

    def _on_layer_val_changed(self, idx: int, val: int):
        if 0 <= idx < len(self._layers) and self._layers[idx] != val:
            self._layers[idx] = val
            self.layersChanged.emit(self._layers)

    def _remove_layer(self, idx: int):
        if 0 <= idx < len(self._layers):
            self._layers.pop(idx)
            self._render_blocks()
            self.layersChanged.emit(self._layers)

    def _add_layer(self):
        new_val = self._layers[-1] if self._layers else 64
        self._layers.append(new_val)
        self._render_blocks()
        self.layersChanged.emit(self._layers)


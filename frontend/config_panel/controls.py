"""Shared parameter presentation. Section builders supply values and capabilities."""

import json
import weakref
from typing import Any

from PyQt6 import sip
from PyQt6.QtCore import QByteArray, QEvent, QMimeData, QObject, QPoint, QRect, QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QCursor, QDrag, QIcon, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QComboBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLayout,
    QLineEdit,
    QMenu,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..sidetabs import svg_icon
from ..theme import theme_color
from ..widgets import ComboBox as BaseComboBox
from ..widgets import label

CONTENT_WIDTH = 960
NUMBER_FIELD_WIDTH = 140
TEXT_FIELD_WIDTH = 220
HIERARCHY_INDENT = 12
PARAM_LABEL_WIDTH = 180
PARAM_FIELD_WIDTH = 220

SUBTLE_ICON_COLORS = {
    (QIcon.Mode.Normal, QIcon.State.Off): "muted",
    (QIcon.Mode.Active, QIcon.State.Off): "text",
    (QIcon.Mode.Selected, QIcon.State.Off): "accent",
    (QIcon.Mode.Normal, QIcon.State.On): "accent",
}


class ComboBox(BaseComboBox):
    """Open editable selectors from their text field as well as the box."""

    def setEditable(self, editable):
        super().setEditable(editable)
        if editable and self.lineEdit() is not None:
            self.lineEdit().installEventFilter(self)

    def eventFilter(self, watched, event):
        if watched is self.lineEdit() and event.type() == QEvent.Type.MouseButtonRelease:
            if event.button() == Qt.MouseButton.LeftButton:
                QTimer.singleShot(0, self.showPopup)
        return super().eventFilter(watched, event)


def _make_subtle_tool_button(icon_name: str, tooltip: str, size: int = 24, icon_size: int = 14) -> QToolButton:
    """Create a subtle icon-only tool button (trash, close, plus) without prominent borders."""
    btn = QToolButton()
    btn.setIcon(svg_icon(icon_name, SUBTLE_ICON_COLORS))
    btn.setIconSize(QSize(icon_size, icon_size))
    btn.setFixedSize(size, size)
    btn.setToolTip(tooltip)
    btn.setAccessibleName(tooltip)
    btn.setProperty("parameterRemove", icon_name == "close")
    btn.setStyleSheet("""
        QToolButton {
            background: transparent;
            border: none;
            border-radius: 3px;
            padding: 0px;
        }
        QToolButton:hover {
            background: #3c3836;
        }
    """)
    return btn


class _RowHoverReveal(QObject):
    """Reveal a control when the pointer is over its parameter row."""

    def __init__(self, button: QToolButton, row_layout: QHBoxLayout, *, reserve_space=True):
        super().__init__(button)
        self._button = weakref.ref(button)
        self._row = weakref.ref(row_layout)
        self._targets = [
            weakref.ref(item.widget())
            for item in (row_layout.itemAt(i) for i in range(row_layout.count()))
            if item is not None and item.widget() is not None and not isinstance(item.widget(), QToolButton)
        ]
        self._parent = None
        self._floating = not reserve_space
        self._refresh_queued = False
        policy = button.sizePolicy()
        policy.setRetainSizeWhenHidden(reserve_space)
        button.setSizePolicy(policy)
        button.hide()
        button.installEventFilter(self)
        for ref in self._targets:
            widget = ref()
            if widget is not None:
                widget.setMouseTracking(True)
                widget.installEventFilter(self)
        QTimer.singleShot(0, self._attach_parent)

    def _attach_parent(self):
        button = self._button()
        if button is None or sip.isdeleted(button):
            return
        parent = button.parentWidget()
        if parent is None:
            return
        self._parent = weakref.ref(parent)
        parent.setMouseTracking(True)
        parent.installEventFilter(self)
        for widget in parent.findChildren(QWidget, options=Qt.FindChildOption.FindDirectChildrenOnly):
            widget.setMouseTracking(True)
            widget.installEventFilter(self)
        if self._floating:
            row = self._row()
            if row is None or sip.isdeleted(row):
                return
            # Floating hover controls never contribute to size hints or wrapping.
            # Lift them above the entry/card so its bounds do not clip the action.
            row.removeWidget(button)
            button.setParent(parent.window())
            parent.window().setMouseTracking(True)
            parent.destroyed.connect(self._retire_button)
            row.destroyed.connect(self._retire_button)
            button.hide()

    def _retire_button(self):
        """Floating controls leave the host tree when their owning row disappears."""
        button = self._button()
        if button is not None and not sip.isdeleted(button):
            button.hide()
            button.setParent(None)
            button.deleteLater()

    def _position_overlay(self):
        button, row = self._button(), self._row()
        parent = self._parent() if self._parent else None
        if (
            button is None
            or sip.isdeleted(button)
            or row is None
            or sip.isdeleted(row)
            or parent is None
            or sip.isdeleted(parent)
        ):
            return
        anchor = QRect()
        for i in range(row.count()):
            widget = row.itemAt(i).widget()
            if widget is not None and (not widget.isHidden() or widget.sizePolicy().retainSizeWhenHidden()):
                anchor = anchor.united(QRect(widget.mapTo(parent, QPoint()), widget.size()))
        if anchor.isEmpty():
            return
        host = button.parentWidget()
        origin = parent.mapTo(
            host, QPoint(anchor.right() + row.spacing() + 1, anchor.center().y() - button.height() // 2)
        )
        button.move(min(origin.x(), max(0, host.width() - button.width())), origin.y())
        button.raise_()

    def _queue_refresh(self):
        if self._refresh_queued:
            return
        self._refresh_queued = True
        QTimer.singleShot(0, self._refresh)

    def _refresh(self):
        self._refresh_queued = False
        self._update_visibility()

    def _update_visibility(self):
        parent = self._parent() if self._parent else None
        button = self._button()
        targets = [ref() for ref in self._targets]
        targets = [
            w
            for w in targets
            if w is not None
            and not sip.isdeleted(w)
            and parent is not None
            and not sip.isdeleted(parent)
            and w.isVisibleTo(parent)
        ]
        if parent is None or sip.isdeleted(parent) or button is None or sip.isdeleted(button):
            return
        if not parent.isVisible():
            self._set_visible(False)
            return
        if not targets:
            self._set_visible(False)
            return
        pos = parent.mapFromGlobal(QCursor.pos())
        row_top = min(widget.mapTo(parent, widget.rect().topLeft()).y() for widget in targets)
        row_bottom = max(widget.mapTo(parent, widget.rect().bottomRight()).y() for widget in targets)
        row = self._row()
        if row is None or sip.isdeleted(row):
            return
        # A card can host several columns sharing the same vertical position.
        # Use the owning field layout's horizontal bounds, including its label.
        bounds = QRect(row.geometry())
        if bounds.isEmpty():
            # Taking an overlay out of a freshly built layout invalidates its
            # cached rectangle. Live editor geometry is already available;
            # do not force a synchronous relayout from the hover handler.
            for widget in targets:
                bounds = bounds.united(QRect(widget.mapTo(parent, QPoint()), widget.size()))
        caption_ref = getattr(row, "_parameter_caption", None)
        caption = caption_ref() if caption_ref else None
        if caption is not None and not sip.isdeleted(caption):
            bounds = bounds.united(QRect(caption.mapTo(parent, QPoint()), caption.size()))
        bounds.setTop(row_top)
        bounds.setBottom(row_bottom)
        focused = QApplication.focusWidget()
        has_focus = focused is button or any(
            w is focused or focused is not None and w.isAncestorOf(focused) for w in targets
        )
        over_extension = False
        if self._floating and button.isVisible():
            action_origin = parent.mapFromGlobal(button.mapToGlobal(QPoint()))
            # Keep the gap traversable as well as the button itself.
            extension = bounds.united(QRect(action_origin, button.size()))
            over_extension = extension.contains(pos)
        self._set_visible(has_focus or over_extension or parent.rect().contains(pos) and bounds.contains(pos))

    def _set_visible(self, visible):
        button = self._button()
        row = self._row()
        if button is None or sip.isdeleted(button):
            return
        if row is None or sip.isdeleted(row) or (not self._floating and row.indexOf(button) < 0):
            button.hide()
            return
        if self._floating and visible:
            if self._parent is None:
                return
            self._position_overlay()
        if button.isHidden() == (not visible):
            return
        if self._floating:
            app = QApplication.instance()
            if visible:
                app.installEventFilter(self)
            else:
                app.removeEventFilter(self)
        button.setVisible(visible)

    def eventFilter(self, watched, event):
        parent = self._parent() if self._parent else None
        owns_geometry = watched is parent or any(watched is ref() for ref in self._targets)
        if self._floating and event.type() == QEvent.Type.Wheel:
            self._set_visible(False)
        if self._floating and owns_geometry and event.type() == QEvent.Type.Hide:
            self._set_visible(False)
        if (
            self._floating
            and owns_geometry
            and event.type() in (QEvent.Type.Move, QEvent.Type.Resize, QEvent.Type.Hide, QEvent.Type.LayoutRequest)
        ):
            self._queue_refresh()
        if event.type() == QEvent.Type.FocusOut:
            QTimer.singleShot(0, self._update_visibility)
        if event.type() == QEvent.Type.FocusIn and isinstance(watched, QWidget):
            targets = [ref() for ref in self._targets]
            if any(
                w is not None and not sip.isdeleted(w) and w.isVisible() and (w is watched or w.isAncestorOf(watched))
                for w in targets
            ):
                self._set_visible(True)
        if event.type() in (QEvent.Type.Enter, QEvent.Type.Leave, QEvent.Type.MouseMove):
            self._update_visibility()
        return False


def _reveal_remove_on_row_hover(row_layout: QHBoxLayout, button: QToolButton, *, reserve_space=True) -> None:
    """Keep parameter rows tidy until a row control is hovered."""
    if button.property("parameterHoverBound"):
        return
    button.setProperty("parameterHoverBound", True)
    _RowHoverReveal(button, row_layout, reserve_space=reserve_space)


def configure_editor(widget, *, compact=False, field_width=None):
    """Apply ordinary geometry where an editor enters a row, never after render."""
    if widget.property("parameterCompact"):
        return
    widget.setProperty("parameterManaged", True)
    widget.setProperty("parameterColumn", compact)
    editor = getattr(widget, "_parameter_editor", None)
    if editor is not None:
        configure_editor(editor, compact=compact, field_width=field_width)
    if isinstance(widget, (QComboBox, QAbstractSpinBox, QLineEdit)):
        width = field_width or (NUMBER_FIELD_WIDTH if compact else PARAM_FIELD_WIDTH)
        if widget.property("parameterExpands"):
            widget.setMinimumWidth(0)
            widget.setMaximumWidth(16777215)
        else:
            widget.setMinimumWidth(NUMBER_FIELD_WIDTH if compact else width)
            widget.setMaximumWidth(width)
        widget.setSizePolicy(QSizePolicy.Policy.Expanding, widget.sizePolicy().verticalPolicy())


class ParameterRow(QHBoxLayout):
    """An editor and stable action slots; rich editors can expand explicitly."""

    def __init__(self, widget=None, *, add=None, remove=None, expand=False, extra_actions=()):
        super().__init__()
        self.editor = widget
        self.setContentsMargins(0, 0, 0, 0)
        self.setSpacing(6)
        if widget is None:
            return
        if expand:
            widget.setProperty("parameterExpands", True)
        configure_editor(widget)
        self.addWidget(widget, 1 if expand else 0)
        for action in (add, *extra_actions, remove):
            if action is not None:
                self.addWidget(action)
        if remove is not None:
            _reveal_remove_on_row_hover(self, remove)
        if not expand:
            self.addStretch()

    def addWidget(self, widget, *args):
        if self.editor is None and not isinstance(widget, (QLabel, QToolButton)):
            self.editor = widget
        configure_editor(widget)
        super().addWidget(widget, *args)
        if hasattr(widget, "contentHeightChanged"):
            widget.contentHeightChanged.connect(self._invalidate_geometry)
        if isinstance(widget, QToolButton) and widget.property("parameterRemove"):
            _reveal_remove_on_row_hover(self, widget)

    def _invalidate_geometry(self, *_):
        layout = self
        while isinstance(layout, QLayout):
            layout.invalidate()
            layout = layout.parent()
        if isinstance(layout, QWidget):
            layout.updateGeometry()


def _parameter_row(widget, **kwargs):
    return ParameterRow(widget, **kwargs)


def _parameter_title(title):
    """Create the shared title used by configuration parameter boxes."""
    title_widget = label(title, "heading")
    title_widget.setStyleSheet("font-size: 23px; font-weight: 600;")
    return title_widget


def _collapsible_parameter_label(title, children, *, expanded=True, on_toggled=None):
    """Shared disclosure control, before the label and outside the child block."""
    container = QWidget()
    container.setProperty("parameterLabel", True)
    container.setStyleSheet('QWidget[parameterLabel="true"] { background: transparent; }')
    container.setFixedWidth(PARAM_LABEL_WIDTH)
    row = QHBoxLayout(container)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(4)
    button = _make_subtle_tool_button(
        "bin_expanded" if expanded else "bin_collapsed", "Expand or collapse " + title, size=20, icon_size=10
    )
    button.setCheckable(True)
    button.setChecked(expanded)
    button.setProperty("parameterDisclosure", True)
    button.setAccessibleName("Expand or collapse " + title)

    def toggle(checked):
        button.setIcon(svg_icon("bin_expanded" if checked else "bin_collapsed", SUBTLE_ICON_COLORS))
        children.setVisible(checked)
        if on_toggled is not None:
            on_toggled(checked)

    button.toggled.connect(toggle)
    if not expanded:
        children.hide()
    row.addWidget(button)
    caption = ParameterLabel(title)
    row.addWidget(caption, 1)
    return container


def _parameter_header(title, *, add=None, remove=None, heading=True, leading=None):
    """Every parameter container puts its adder immediately after its title."""
    row = QHBoxLayout()
    row.setSpacing(8)
    if leading is not None:
        leading.setProperty("parameterHeaderLeading", True)
        row.addWidget(leading)
    title_widget = _parameter_title(title) if heading else label(title, "muted")
    row.caption = title_widget
    row.addWidget(title_widget)
    if add is not None:
        row.addWidget(add)
    row.addStretch()
    if remove is not None:
        row.addWidget(remove)
        _reveal_remove_on_row_hover(row, remove)
    return row


class ParameterLabel(QLabel):
    """Clip long labels; reveal overflow above the editor on label hover only."""

    def setWordWrap(self, enabled):
        super().setWordWrap(False)

    def hasHeightForWidth(self):
        return False

    def sizeHint(self):
        width = self.maximumWidth() if self.maximumWidth() < 16777215 else PARAM_LABEL_WIDTH
        return QSize(width, super().sizeHint().height())

    def _hide_overflow(self):
        overlay = getattr(self, "_hover_overlay", None)
        if overlay is not None and not sip.isdeleted(overlay):
            overlay.hide()

    def enterEvent(self, event):
        super().enterEvent(event)
        text_width = self.fontMetrics().horizontalAdvance(self.text())
        if text_width <= self.contentsRect().width():
            return
        host = self.window()
        overlay = getattr(self, "_hover_overlay", None)
        if overlay is None or sip.isdeleted(overlay):
            overlay = QLabel(host)
            overlay.setObjectName("parameterLabelOverflow")
            overlay.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
            self.destroyed.connect(overlay.deleteLater)
            self._hover_overlay = overlay
        overlay.setText(self.text())
        overlay.setFont(self.font())
        overlay.setAlignment(self.alignment())
        overlay.setStyleSheet(
            f"QLabel {{ background: {theme_color('surface')}; color: {self.palette().color(self.foregroundRole()).name()}; border: none; padding: 0px; }}"
        )
        origin = self.mapTo(host, self.contentsRect().topLeft())
        overlay.setGeometry(
            origin.x(), origin.y(), min(text_width + 2, host.width() - origin.x()), self.contentsRect().height()
        )
        overlay.show()
        overlay.raise_()

    def leaveEvent(self, event):
        self._hide_overflow()
        super().leaveEvent(event)

    def hideEvent(self, event):
        self._hide_overflow()
        super().hideEvent(event)

    def resizeEvent(self, event):
        self._hide_overflow()
        super().resizeEvent(event)

    def mousePressEvent(self, event):
        self._hide_overflow()
        super().mousePressEvent(event)


class ParameterForm(QFormLayout):
    """Common geometry for parameter lists; groups span both form columns."""

    def __init__(self, parent=None, *, compact=False, field_width=None):
        super().__init__(parent)
        self.compact = compact
        self.field_width = field_width
        self._declared_groups = None
        self.setContentsMargins(0, 0, 0, 0)
        self.setVerticalSpacing(8)
        self.setHorizontalSpacing(10)
        self.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

    def addRow(self, *items):
        if len(items) == 2 and isinstance(items[0], str) and not items[0]:
            items = (items[1],)
        if len(items) == 2 and isinstance(items[0], str):
            caption = ParameterLabel(items[0])
            caption.setWordWrap(True)
            caption.setAccessibleDescription(items[0])
            items = (caption, items[1])
        super().addRow(*items)
        if len(items) == 2:
            item = self.itemAt(self.rowCount() - 1, QFormLayout.ItemRole.LabelRole)
            if item and isinstance(item.widget(), QLabel):
                item.widget().setFixedWidth(120 if self.compact else PARAM_LABEL_WIDTH)
                item.widget().setWordWrap(True)
            self._configure(items[1])
            if isinstance(items[1], QHBoxLayout) and item and item.widget():
                items[1]._parameter_caption = weakref.ref(item.widget())
            editor = getattr(items[1], "editor", items[1])
            if isinstance(editor, QWidget) and not editor.accessibleName() and item and item.widget():
                editor.setAccessibleName(item.widget().text())
        elif len(items) == 1:
            self._configure(items[0])

    def _configure(self, item):
        if isinstance(item, QWidget):
            form = getattr(item, "form_layout", None)
            if isinstance(form, ParameterForm):
                form.set_compact(self.compact, field_width=self.field_width)
            else:
                configure_editor(item, compact=self.compact, field_width=self.field_width)
        elif isinstance(item, QLayout):
            for index in range(item.count()):
                child = item.itemAt(index)
                self._configure(child.widget() or child.layout())

    def set_compact(self, compact, *, field_width=None):
        self.compact = compact
        self.field_width = field_width
        for index in range(self.rowCount()):
            caption = self.itemAt(index, QFormLayout.ItemRole.LabelRole)
            if caption and caption.widget():
                caption.widget().setFixedWidth(120 if compact else PARAM_LABEL_WIDTH)
            field = self.itemAt(index, QFormLayout.ItemRole.FieldRole)
            if field is None:
                field = self.itemAt(index, QFormLayout.ItemRole.SpanningRole)
            if field:
                self._configure(field.widget() or field.layout())

    def add_parameter(self, identity, caption, field, *, children=None):
        """Declare one logical subtree, including child membership, at construction."""
        from ..parameter_reorder import ParameterGroup

        if self._declared_groups is None:
            from ..parameter_reorder import form_groups

            self._declared_groups = form_groups(self)
        if isinstance(caption, str):
            caption = ParameterLabel(caption)
            caption.setAccessibleDescription(caption.text())
        parts = [(caption, field)]
        if caption is None:
            self.addRow(field)
        else:
            self.addRow(caption, field)
        if children is not None:
            self.addRow(children)
            parts.append((None, children))
        group = ParameterGroup(parts, caption, key=json.dumps(list(identity)))
        self._declared_groups.append(group)
        return group

    def add_children(self, identity, children):
        """Attach children by declared identity, independently of row adjacency."""
        key = json.dumps(list(identity))
        group = next((g for g in self._declared_groups or [] if g.key == key), None)
        if group is None:
            return self.add_parameter((*identity, "children"), None, children)
        self.addRow(children)
        group.parts.append((None, children))
        return group


class IndentedHierarchyBlock(QFrame):
    """Indented container with a subtle vertical guide line denoting hierarchy in Theta-IDE."""

    def __init__(self, parent=None, level: int = 1):
        super().__init__(parent)
        self.setObjectName(f"hierarchyBlock_lvl{level}")
        border_color = theme_color("border")
        self.setStyleSheet(f"""
            QFrame#hierarchyBlock_lvl{level} {{
                border-left: 2px solid {border_color};
                margin-left: {HIERARCHY_INDENT}px;
                padding-left: 10px;
                background: transparent;
                border-top: none;
                border-right: none;
                border-bottom: none;
            }}
        """)
        self.form_layout = ParameterForm(self)
        self.form_layout.setContentsMargins(0, 4, 0, 6)
        self.form_layout.setVerticalSpacing(6)
        self.form_layout.setHorizontalSpacing(10)
        self.form_layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)

    def addRow(self, label_or_widget, widget=None):
        if widget is not None:
            self.form_layout.addRow(label_or_widget, widget)
        else:
            self.form_layout.addRow(label_or_widget)


class ConfigBox(QFrame):
    """Themed card container representing a section of configuration."""

    def __init__(self, title=None, subtitle=None, parent=None, *, section=None):
        super().__init__(parent)
        self.setObjectName("card")
        self.setProperty("configBox", True)
        self.setProperty("configSection", section)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(16, 14, 16, 14)
        self.layout.setSpacing(10)
        self.title_label = None
        self.subtitle_label = None

    def add_widget(self, widget):
        self.layout.addWidget(widget)

    def add_layout(self, sub_layout):
        self.layout.addLayout(sub_layout)

    def set_header(self, title, *, add=None, remove=None, leading=None):
        header = _parameter_header(title, add=add, remove=remove, leading=leading)
        self.title_label = header.itemAt(1 if leading is not None else 0).widget()
        self.add_layout(header)
        return header

    def add_parameter_body(self, *, enabled=True):
        body = QWidget()
        body.setProperty("configBoxBody", True)
        layout = QVBoxLayout(body)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        body.setEnabled(enabled)
        self.add_widget(body)
        return body


CATEGORICAL_CHOICES = {
    "activation": ["relu", "gelu", "tanh", "elu", "silu", "sigmoid", "leaky_relu", "identity"],
    "activation_fn": ["relu", "gelu", "tanh", "elu", "silu", "sigmoid", "leaky_relu", "identity"],
    "act_fn": ["relu", "gelu", "tanh", "elu", "silu", "sigmoid", "leaky_relu", "identity"],
    "optimizer": ["adam", "adamw", "sgd", "rmsprop"],
    "opt": ["adam", "adamw", "sgd", "rmsprop"],
    "mode": ["min", "max"],
    "direction": ["min", "max"],
}


def _make_categorical_combo(key: str, current_val: Any, on_change) -> ComboBox:
    combo = ComboBox()
    options = list(CATEGORICAL_CHOICES.get(key.lower(), []))
    str_val = str(current_val) if current_val is not None else ""
    if str_val and str_val not in options:
        options.insert(0, str_val)
    for opt in options:
        combo.addItem(opt)
    if str_val in options:
        combo.setCurrentText(str_val)
    combo.setEditable(True)
    combo.setMaximumWidth(TEXT_FIELD_WIDTH)
    combo.currentTextChanged.connect(on_change)
    return combo


def _fit_inline_control(control):
    """Size compact editors from the displayed text, including uncommitted digits."""
    control.setProperty("parameterCompact", True)
    editor = control.lineEdit() if isinstance(control, QAbstractSpinBox) else control
    if isinstance(control, QAbstractSpinBox):
        # The spin box already supplies its frame and padding. Its internal
        # line edit must not inherit another padded input box from the theme.
        editor.setStyleSheet("QLineEdit { background: transparent; border: none; padding: 0px; min-height: 0px; }")

    control.ensurePolished()
    editor.ensurePolished()

    def resize(*_):
        text = control.currentText() if isinstance(control, QComboBox) else editor.displayText()
        padding = 38 if isinstance(control, QComboBox) else 24
        width = max(48, editor.fontMetrics().horizontalAdvance(text or "Value") + padding)
        if control.minimumWidth() == width and control.maximumWidth() == width:
            return
        if isinstance(control, QAbstractSpinBox):
            # Qt reformats numeric text on resize. Preserve exactly what the
            # user is typing, including leading decimal zeros and the caret.
            cursor = editor.cursorPosition()
            selection = editor.selectionStart()
            selected = editor.selectedText()
            blocked = editor.blockSignals(True)
            try:
                control.setFixedWidth(width)
                editor.setText(text)
                if selection >= 0:
                    editor.setSelection(selection, len(selected))
                else:
                    editor.setCursorPosition(cursor)
            finally:
                editor.blockSignals(blocked)
        else:
            control.setFixedWidth(width)
        control.updateGeometry()

    if isinstance(control, QComboBox):
        control.currentTextChanged.connect(resize)
    else:
        # Numeric line edits emit while Qt is still updating their value.
        # Reflow on the next event turn, after that edit has completed.
        timer = QTimer(control)
        timer.setSingleShot(True)
        timer.timeout.connect(resize)
        editor.textChanged.connect(lambda *_: timer.start(0))
    resize()


class _SweepTypePicker(ComboBox):
    cancelled = pyqtSignal()

    def hidePopup(self):
        super().hidePopup()
        QTimer.singleShot(0, self._cancel_if_unselected)

    def focusOutEvent(self, event):
        super().focusOutEvent(event)
        QTimer.singleShot(0, self._cancel_if_unselected)

    def _cancel_if_unselected(self):
        if self.currentData() is None and not self.view().isVisible():
            self.cancelled.emit()


def _make_param_adder_button(
    tooltip: str,
    candidates: list[str],
    on_chosen,
    parent_widget: QWidget | None = None,
    allow_custom: bool = True,
) -> QToolButton:
    """Create a subtle plus button that opens a focused dropdown menu for selecting a parameter."""
    btn = _make_subtle_tool_button("plus", tooltip, size=22, icon_size=12)
    btn.setProperty("parameterAdd", True)
    btn.setStyleSheet("""
        QToolButton {
            background: transparent;
            border: 1px dashed #504945;
            border-radius: 3px;
            padding: 0px;
        }
        QToolButton:hover {
            background: #3c3836;
            border-color: #a89984;
        }
        QToolButton::menu-indicator {
            image: none;
            width: 0px;
        }
    """)
    menu = QMenu(btn)
    menu.setStyleSheet("""
        QMenu {
            background-color: #282828;
            border: 1px solid #504945;
            border-radius: 4px;
            padding: 4px 0px;
        }
        QMenu::item {
            padding: 5px 18px 5px 12px;
            color: #ebdbb2;
            font-size: 12px;
        }
        QMenu::item:selected {
            background-color: #3c3836;
            color: #fbf1c7;
        }
        QMenu::separator {
            height: 1px;
            background: #3c3836;
            margin: 4px 6px;
        }
    """)
    for cand in candidates:
        act = menu.addAction(cand)
        act.triggered.connect(lambda _, c=cand: on_chosen(c))

    if allow_custom:
        if candidates:
            menu.addSeparator()
        act_custom = menu.addAction("+ Custom...")

        def _prompt_custom():
            val, ok = QInputDialog.getText(parent_widget or btn, "Add Parameter", "Parameter name:")
            if ok and val.strip():
                on_chosen(val.strip())

        act_custom.triggered.connect(_prompt_custom)

    btn.setMenu(menu)
    btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
    btn.setEnabled(not menu.isEmpty())
    return btn


class _SectionMoveControls(QWidget):
    """Show compact reorder controls while a section title is hovered."""

    def __init__(self, title, box, move):
        super().__init__(box)
        self.title = title
        self.setObjectName("sectionMoveControls")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        for glyph, direction, tooltip in (("⌃", -1, "Move section up"), ("⌄", 1, "Move section down")):
            button = QToolButton(self)
            button.setText(glyph)
            button.setToolTip(tooltip)
            button.setFixedSize(18, 12)
            button.setStyleSheet(
                "QToolButton { border: none; padding: 0; color: #a89984; } QToolButton:hover { color: #fbf1c7; }"
            )
            button.clicked.connect(lambda _, d=direction: move(d))
            layout.addWidget(button)
            button.installEventFilter(self)
        self.setFixedSize(18, 24)
        title.installEventFilter(self)
        self.installEventFilter(self)
        self.hide()

    def eventFilter(self, watched, event):
        if event.type() == QEvent.Type.Enter:
            self.show()
        elif event.type() == QEvent.Type.Leave:
            QTimer.singleShot(0, self._hide_unless_hovered)
        return False

    def _hide_unless_hovered(self):
        pos = QCursor.pos()
        if not self.title.rect().contains(self.title.mapFromGlobal(pos)) and not self.rect().contains(
            self.mapFromGlobal(pos)
        ):
            self.hide()


class TitleLabel(QLabel):
    """Clickable heading label that emits clicked and double_clicked on click."""

    clicked = pyqtSignal()
    double_clicked = pyqtSignal()

    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.setObjectName("heading")
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.rect().contains(event.position().toPoint()):
            self.clicked.emit()
            self.double_clicked.emit()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
            self.double_clicked.emit()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)


class InlineTitleEditor(QLineEdit):
    """Line editor supporting cancellation on Escape key press."""

    escape_pressed = pyqtSignal()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.escape_pressed.emit()
            event.accept()
            return
        super().keyPressEvent(event)


class MethodDragHandle(QToolButton):
    """Grip icon button to initiate dragging and reordering of method cards."""

    def __init__(self, method_name: str, parent=None):
        super().__init__(parent)
        self.method_name = method_name
        self.setIcon(svg_icon("grip", SUBTLE_ICON_COLORS))
        self.setIconSize(QSize(14, 14))
        self.setFixedSize(22, 22)
        self.setCursor(Qt.CursorShape.SizeAllCursor)
        self.setToolTip("Drag to reorder method")
        self.setStyleSheet("""
            QToolButton {
                background: transparent;
                border: none;
                border-radius: 3px;
                padding: 0;
            }
            QToolButton:hover {
                background: #3c3836;
            }
        """)
        self._drag_start_pos = None

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start_pos = event.pos()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if (event.buttons() & Qt.MouseButton.LeftButton) and self._drag_start_pos is not None:
            distance = (event.pos() - self._drag_start_pos).manhattanLength()
            if distance >= QApplication.startDragDistance():
                self._start_drag()
                self._drag_start_pos = None
                return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self._drag_start_pos = None
        super().mouseReleaseEvent(event)

    def _start_drag(self):
        drag = QDrag(self)
        mime = QMimeData()
        mime.setData("application/x-theta-method", QByteArray(self.method_name.encode("utf-8")))
        drag.setMimeData(mime)

        p = self.parent()
        while p and not isinstance(p, (ConfigBox, QFrame)):
            p = p.parent()
        target_widget = p or self
        pixmap = target_widget.grab()
        ghost = QPixmap(pixmap.size())
        ghost.fill(Qt.GlobalColor.transparent)
        painter = QPainter(ghost)
        painter.setOpacity(0.65)
        painter.drawPixmap(0, 0, pixmap)
        painter.end()

        drag.setPixmap(ghost)
        drag.setHotSpot(self._drag_start_pos or QPoint(pixmap.width() // 2, 15))
        drag.exec(Qt.DropAction.MoveAction)


class MethodCardsContainer(QWidget):
    """Container holding method cards arranged in 2 columns that accepts drops and displays drop indicator line."""

    def __init__(self, viewer, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        self.grid = QGridLayout(self)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setHorizontalSpacing(14)
        self.grid.setVerticalSpacing(14)
        self.grid.setColumnStretch(0, 1)
        self.grid.setColumnStretch(1, 1)
        self.setAcceptDrops(True)
        self._drop_index = None
        self._cards: list[tuple[str, QWidget]] = []

    def add_method_card(self, method_name: str, card: QWidget):
        card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        idx = len(self._cards)
        row = idx // 2
        col = idx % 2
        self._cards.append((method_name, card))
        self.grid.addWidget(card, row, col)

    def dragEnterEvent(self, event):
        if event.mimeData().hasFormat("application/x-theta-method"):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasFormat("application/x-theta-method"):
            pos = event.position().toPoint()
            idx = self._calculate_drop_index(pos)
            if idx != self._drop_index:
                self._drop_index = idx
                self.update()
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        self._drop_index = None
        self.update()
        event.accept()

    def dropEvent(self, event):
        if event.mimeData().hasFormat("application/x-theta-method"):
            method_name = bytes(event.mimeData().data("application/x-theta-method")).decode("utf-8")
            target_idx = self._drop_index
            self._drop_index = None
            self.update()
            if target_idx is not None:
                self.viewer.reorder_method(method_name, target_idx)
            event.acceptProposedAction()
        else:
            event.ignore()

    def _calculate_drop_index(self, pos: QPoint) -> int:
        if not self._cards:
            return 0
        for i, (_, card) in enumerate(self._cards):
            geom = card.geometry()
            if pos.y() < geom.top():
                return i
            if geom.top() <= pos.y() <= geom.bottom():
                if pos.x() < geom.center().x():
                    return i
                col = i % 2
                if col == 1 or i == len(self._cards) - 1:
                    if pos.x() >= geom.center().x():
                        return i + 1
        return len(self._cards)

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._drop_index is not None and self._cards:
            painter = QPainter(self)
            try:
                acc_color = QColor(theme_color("accent"))
            except Exception:
                acc_color = QColor("#fabd2f")
            pen = QPen(acc_color, 3)
            painter.setPen(pen)
            if self._drop_index < len(self._cards):
                card = self._cards[self._drop_index][1]
                g = card.geometry()
                painter.drawLine(g.left() - 4, g.top() + 4, g.left() - 4, g.bottom() - 4)
            else:
                card = self._cards[-1][1]
                g = card.geometry()
                painter.drawLine(g.right() + 4, g.top() + 4, g.right() + 4, g.bottom() - 4)
            painter.end()

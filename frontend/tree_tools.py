"""What the Experiment tab's tree (config_tree.py) and the Results run selector (results/explorer.py) share.

One source for their look and their header behaviour: the themed 30 × 30 header buttons, the bold group
headings with a rule under them, the folder icon, and the search / collapse / expand actions.
"""

from __future__ import annotations

import json
import re

from PyQt6.QtCore import QByteArray, QMimeData, QPoint, QRect, QSettings, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QDrag, QFont, QIcon, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (QApplication, QLayout, QMenu, QStyledItemDelegate, QStyleOptionViewItem,
                             QToolButton, QWidget, QWidgetItem)

from .sidetabs import svg_icon
from .theme import theme_color

TOOL_ICON_COLORS = {
    (QIcon.Mode.Normal, QIcon.State.Off): "text",
    (QIcon.Mode.Active, QIcon.State.Off): "accent",
    (QIcon.Mode.Selected, QIcon.State.Off): "accent",
    (QIcon.Mode.Disabled, QIcon.State.Off): "disabled",
    (QIcon.Mode.Normal, QIcon.State.On): "accent",
    (QIcon.Mode.Active, QIcon.State.On): "accent",
    (QIcon.Mode.Selected, QIcon.State.On): "accent",
    (QIcon.Mode.Disabled, QIcon.State.On): "disabled",
}


def group_heading_font(base: QFont) -> QFont:
    """The bold, larger font of a group heading."""
    font = QFont(base)
    font.setPointSize(max(base.pointSize() + 1, 13) if base.pointSize() > 0 else 13)
    font.setBold(True)
    return font


def folder_icon() -> QIcon:
    return svg_icon("folder", TOOL_ICON_COLORS)


def style_tool_button(button: QToolButton, icon_name: str, tooltip: str = "") -> QToolButton:
    """Make `button` a 30 × 30 header button with a themed SVG icon (refresh_tool_icons recolors it)."""
    button.setProperty("svg_icon", icon_name)
    button.setIcon(svg_icon(icon_name, TOOL_ICON_COLORS))
    button.setIconSize(QSize(16, 16))
    button.setFixedSize(30, 30)
    button.setStyleSheet("padding: 0;")
    if tooltip:
        button.setToolTip(tooltip)
        button.setAccessibleName(tooltip)
    return button


def square_tool_button(icon_name: str, tooltip: str = "") -> QToolButton:
    return style_tool_button(QToolButton(), icon_name, tooltip)


def refresh_tool_icons(widget) -> None:
    """Re-render every header button under `widget` in the current theme's colors."""
    for button in widget.findChildren(QToolButton):
        name = button.property("svg_icon")
        if name:
            button.setIcon(svg_icon(name, TOOL_ICON_COLORS))


class GroupItemDelegate(QStyledItemDelegate):
    """Draws group rows as headings: larger bold text, a subtle rule under them, more room above."""

    def is_group(self, index) -> bool:
        """Whether `index` is a group heading (the experiment tree marks them in its item data)."""
        data = index.data(Qt.ItemDataRole.UserRole)
        return isinstance(data, dict) and bool(data.get("is_group"))

    def initStyleOption(self, option, index):
        super().initStyleOption(option, index)
        if self.is_group(index):
            option.font = group_heading_font(option.font)

    def paint(self, painter, option, index):
        super().paint(painter, option, index)
        if self.is_group(index):
            painter.save()
            painter.setPen(QPen(QColor("#504945"), 1))
            y = option.rect.bottom() - 1
            painter.drawLine(option.rect.left() + 4, y, option.rect.right() - 4, y)
            painter.restore()

    def sizeHint(self, option, index):
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        hint = super().sizeHint(opt, index)
        if self.is_group(index):
            return QSize(hint.width(), hint.height() + 10)
        return hint


class TreeHeaderActions:
    """Search, collapse all and expand all, for a widget with `tree`, `search` (a line edit) and
    `btn_search` (a checkable button). Call `watch_search()` once both exist: Esc then closes the search."""

    def watch_search(self):
        self.search.hide()
        self.search.installEventFilter(self)
        self.btn_search.setCheckable(True)
        self.btn_search.setChecked(False)
        self.btn_search.clicked.connect(self.toggle_search)

    def toggle_search(self, checked: bool | None = None):
        """Show or hide the filter; hiding it clears the filter."""
        if checked is None:
            checked = self.search.isHidden()
        self.btn_search.setChecked(checked)
        self.search.setVisible(checked)
        if checked:
            self.search.setFocus()
            self.search.selectAll()
        else:
            self.search.clear()

    def eventFilter(self, obj, event):
        if obj is self.search and event.type() == event.Type.KeyPress and event.key() == Qt.Key.Key_Escape:
            self.toggle_search(False)
            return True
        return super().eventFilter(obj, event)

    def collapse_all(self):
        """Collapse every node of the tree."""
        self.tree.collapseAll()

    def expand_all(self):
        """Expand every node of the tree."""
        self.tree.expandAll()


TREE_BUTTON_MIME = "application/x-theta-tree-button"


class DraggableToolButton(QToolButton):
    """Square header tool button with drag-and-drop reordering support."""

    def __init__(self, button_id: str, parent=None):
        super().__init__(parent)
        self.button_id = button_id
        self._drag_start_pos = None
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start_pos = event.pos()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if (event.buttons() & Qt.MouseButton.LeftButton) and self._drag_start_pos is not None:
            dist = (event.pos() - self._drag_start_pos).manhattanLength()
            if dist >= QApplication.startDragDistance():
                self._start_drag()
                self._drag_start_pos = None
                return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self._drag_start_pos = None
        super().mouseReleaseEvent(event)

    def contextMenuEvent(self, event):
        p = self.parentWidget()
        if p is not None and hasattr(p, "contextMenuEvent"):
            p.contextMenuEvent(event)
        else:
            super().contextMenuEvent(event)

    def _start_drag(self):
        self.setDown(False)
        drag = QDrag(self)
        mime = QMimeData()
        mime.setData(TREE_BUTTON_MIME, QByteArray(self.button_id.encode("utf-8")))
        drag.setMimeData(mime)

        pixmap = self.grab()
        ghost = QPixmap(pixmap.size())
        ghost.fill(Qt.GlobalColor.transparent)
        painter = QPainter(ghost)
        painter.setOpacity(0.75)
        painter.drawPixmap(0, 0, pixmap)
        painter.end()

        drag.setPixmap(ghost)
        drag.setHotSpot(self._drag_start_pos or QPoint(pixmap.width() // 2, pixmap.height() // 2))
        toolbar = self.parentWidget()
        if isinstance(toolbar, TreeHeaderWidget):
            toolbar.begin_drag()
        try:
            drag.exec(Qt.DropAction.MoveAction)
        finally:
            if isinstance(toolbar, TreeHeaderWidget):
                toolbar.end_drag()


class TreeHeaderLayout(QLayout):
    """Layout managing two rows of toolbar buttons with left and right groups per row."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._items: list[QWidgetItem] = []
        self.groups: dict[str, list[DraggableToolButton]] = {
            "row1_left": [],
            "row1_right": [],
            "row2_left": [],
            "row2_right": [],
        }

    def addItem(self, item):
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int):
        if 0 <= index < len(self._items):
            return self._items[index]
        return None

    def takeAt(self, index: int):
        if 0 <= index < len(self._items):
            return self._items.pop(index)
        return None

    def row_count(self) -> int:
        return max((int(match.group(1)) for key, buttons in self.groups.items()
                    if buttons and (match := re.fullmatch(r"row([1-9][0-9]*)_(?:left|right)", key))), default=1)

    def sizeHint(self) -> QSize:
        return self.minimumSize()

    def minimumSize(self) -> QSize:
        width = max((30 * count + 4 * max(0, count - 1)
                     for row in range(1, self.row_count() + 1)
                     if (count := sum(len(self.groups.get(f"row{row}_{side}", []))
                                      for side in ("left", "right")))), default=30)
        return QSize(width, self.row_count() * 34 - 4)

    def setGeometry(self, rect: QRect):
        super().setGeometry(rect)
        for row in range(1, self.row_count() + 1):
            y = rect.top() + (row - 1) * 34
            left = self.groups.get(f"row{row}_left", [])
            right = self.groups.get(f"row{row}_right", [])
            for index, button in enumerate(left):
                button.setGeometry(QRect(rect.left() + index * 34, y, 30, 30))
            x = rect.right() - (len(right) * 34 - 4) + 1
            for index, button in enumerate(right):
                button.setGeometry(QRect(x + index * 34, y, 30, 30))


class TreeHeaderWidget(QWidget):
    """Header toolbar widget with two rows and 4 draggable/reorderable button groups."""

    layout_changed = pyqtSignal()

    def __init__(self, tree_widget: "ConfigTreeWidget", parent=None):
        super().__init__(parent)
        self.tree_widget = tree_widget
        self.setAcceptDrops(True)
        self.header_layout = TreeHeaderLayout(self)
        self._drop_info = None  # (group_key, target_index, (kind, rect))
        self._dragging = False

    def set_groups(self, groups: dict[str, list[DraggableToolButton]]):
        # Remove unoccupied rows while retaining each occupied row's left/right alignment.
        rows = sorted({int(m.group(1)) for key, buttons in groups.items()
                       if buttons and (m := re.fullmatch(r"row([1-9][0-9]*)_(?:left|right)", key))})
        normalized = {}
        for new_row, old_row in enumerate(rows, 1):
            for side in ("left", "right"):
                normalized[f"row{new_row}_{side}"] = list(groups.get(f"row{old_row}_{side}", []))
        # Keep the original two-row keys available to callers, even if a row is empty.
        for row in range(1, max(2, len(rows)) + 1):
            for side in ("left", "right"):
                normalized.setdefault(f"row{row}_{side}", [])
        self.header_layout.groups = normalized
        self.header_layout._items.clear()
        for buttons in normalized.values():
            for button in buttons:
                self.header_layout.addWidget(button)
        self.header_layout.invalidate()
        self._update_size()
        self.header_layout.setGeometry(self.contentsRect())
        self.update()

    def _update_size(self):
        height = self.header_layout.minimumSize().height() + (34 if self._dragging else 0)
        self.setFixedHeight(height)
        self.updateGeometry()

    def begin_drag(self):
        self._dragging = True
        self._update_size()

    def end_drag(self):
        self._dragging = False
        self._drop_info = None
        self._update_size()
        self.update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.header_layout.setGeometry(self.contentsRect())

    def sizeHint(self) -> QSize:
        size = self.header_layout.sizeHint()
        return QSize(size.width(), size.height() + (34 if self._dragging else 0))

    def minimumSizeHint(self) -> QSize:
        return self.sizeHint()

    def _calc_drop_info(self, pos: QPoint):
        w = self.width()
        row = min(max(1, pos.y() // 34 + 1), self.header_layout.row_count() + 1)
        side = "left" if pos.x() <= w / 2 else "right"
        group_key = f"row{row}_{side}"
        buttons = self.header_layout.groups.get(group_key, [])
        y = (row - 1) * 34
        if not buttons:
            return (group_key, 0, ("box", QRect(0 if side == "left" else w - 30, y, 30, 30)))
        for idx, btn in enumerate(buttons):
            geom = btn.geometry()
            if pos.x() < geom.center().x():
                return (group_key, idx, ("line", QRect(max(0, geom.left() - 2), geom.top() + 2, 2, 26)))
        geom = buttons[-1].geometry()
        return (group_key, len(buttons), ("line", QRect(min(w - 2, geom.right() + 2), geom.top() + 2, 2, 26)))

    def dragEnterEvent(self, event):
        if event.mimeData().hasFormat(TREE_BUTTON_MIME):
            event.acceptProposedAction()
            self._drop_info = self._calc_drop_info(event.position().toPoint())
            self.update()

    def dragMoveEvent(self, event):
        if event.mimeData().hasFormat(TREE_BUTTON_MIME):
            event.acceptProposedAction()
            self._drop_info = self._calc_drop_info(event.position().toPoint())
            self.update()

    def dragLeaveEvent(self, event):
        self._drop_info = None
        self.update()

    def dropEvent(self, event):
        if event.mimeData().hasFormat(TREE_BUTTON_MIME):
            event.acceptProposedAction()
            button_id = bytes(event.mimeData().data(TREE_BUTTON_MIME)).decode("utf-8")
            if self._drop_info:
                target_group, target_idx, _ = self._drop_info
                self.move_button(button_id, target_group, target_idx)
            self._drop_info = None
            self.update()

    def move_button(self, button_id: str, target_group: str, target_idx: int):
        src_group = None
        src_idx = None
        target_btn = None
        for g_name, b_list in self.header_layout.groups.items():
            for idx, b in enumerate(b_list):
                if b.button_id == button_id:
                    src_group = g_name
                    src_idx = idx
                    target_btn = b
                    break
            if src_group:
                break

        match = re.fullmatch(r"row([1-9][0-9]*)_(?:left|right)", target_group)
        if not target_btn or not match or int(match.group(1)) > self.header_layout.row_count() + 1:
            return
        self.header_layout.groups.setdefault(target_group, [])

        if src_group == target_group:
            if src_idx == target_idx or target_idx == src_idx + 1:
                return
            self.header_layout.groups[src_group].pop(src_idx)
            if target_idx > src_idx:
                target_idx -= 1
            self.header_layout.groups[target_group].insert(target_idx, target_btn)
        else:
            self.header_layout.groups[src_group].pop(src_idx)
            self.header_layout.groups[target_group].insert(target_idx, target_btn)

        self.set_groups(self.header_layout.groups)
        self.layout_changed.emit()

    def contextMenuEvent(self, event):
        menu = QMenu(self)
        reset_act = menu.addAction("Reset Toolbar Layout")
        reset_act.triggered.connect(self.tree_widget.reset_toolbar_layout)
        menu.exec(event.globalPos())

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._drop_info:
            painter = QPainter(self)
            try:
                acc_color = QColor(theme_color("accent"))
            except Exception:
                acc_color = QColor("#fabd2f")
            pen = QPen(acc_color, 2)
            painter.setPen(pen)
            kind, geom = self._drop_info[2]
            if kind == "line":
                painter.fillRect(geom, acc_color)
            elif kind == "box":
                pen.setStyle(Qt.PenStyle.DashLine)
                painter.setPen(pen)
                painter.drawRect(geom.adjusted(1, 1, -1, -1))
            painter.end()



class TreeToolbarPersistence:
    """Persist and reset the shared toolbar groups for each tree mode."""

    def _load_toolbar_groups(self) -> dict[str, list[DraggableToolButton]]:
        defaults = self._default_toolbar_groups()
        all_buttons = {b.button_id: b for g in defaults.values() for b in g}

        saved_state = None
        if self.settings_manager:
            try:
                saved_state = self.settings_manager.get("filetree", f"buttons_{self.mode}", default=None)
            except Exception:
                saved_state = None
        if not saved_state:
            settings = QSettings("ThetaIDE", "ThetaIDE")
            raw = settings.value(f"filetree_buttons_{self.mode}", None)
            if raw and isinstance(raw, str):
                try:
                    saved_state = json.loads(raw)
                except Exception:
                    saved_state = None

        if not isinstance(saved_state, dict):
            return defaults

        loaded_groups = {key: [] for key in defaults}
        placed = set()
        for g_name, b_ids in saved_state.items():
            if not re.fullmatch(r"row([1-9][0-9]*)_(?:left|right)", g_name) or not isinstance(b_ids, list):
                continue
            # A valid layout can never need more rows than it has buttons.
            if int(g_name.split("_")[0][3:]) > len(all_buttons):
                continue
            loaded_groups.setdefault(g_name, [])
            for bid in b_ids:
                if isinstance(bid, str) and bid in all_buttons and bid not in placed:
                    loaded_groups[g_name].append(all_buttons[bid])
                    placed.add(bid)

        # Append any newly added / missing buttons to their default location
        for g_name, b_list in defaults.items():
            for b in b_list:
                if b.button_id not in placed:
                    loaded_groups[g_name].append(b)
                    placed.add(b.button_id)

        return loaded_groups

    def _save_toolbar_layout(self):
        state = {
            g_name: [b.button_id for b in btn_list]
            for g_name, btn_list in self.header_widget.header_layout.groups.items()
        }
        if self.settings_manager:
            try:
                self.settings_manager.set("filetree", f"buttons_{self.mode}", state)
            except Exception:
                pass
        try:
            settings = QSettings("ThetaIDE", "ThetaIDE")
            settings.setValue(f"filetree_buttons_{self.mode}", json.dumps(state))
        except Exception:
            pass

    def reset_toolbar_layout(self):
        """Reset the header toolbar buttons to their default 2-row layout."""
        self.header_widget.set_groups(self._default_toolbar_groups())
        self._save_toolbar_layout()


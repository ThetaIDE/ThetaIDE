"""Discussion experiments, saved runs and figures, using the shared tree tools and Results catalog."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QIcon, QKeySequence, QShortcut
from PyQt6.QtWidgets import QAbstractItemView, QLineEdit, QMessageBox, QTreeWidget, QTreeWidgetItem, QVBoxLayout

from ..sidetabs import svg_icon
from ..tree_tools import (TOOL_ICON_COLORS, DraggableToolButton, FileTreeWidget, GroupItemDelegate,
                          folder_icon, group_heading_font,
                          refresh_tool_icons, style_tool_button)
from . import model

class DiscussionTree(FileTreeWidget):
    """Experiment tree with figure leaves.

    ``figure_selected`` carries the selected figure file, or None when the selection is not a
    figure; ``experiment_selected`` carries the experiment of the selection ("" for a group or
    folder).

    Figure leaves have a context menu (Rename…, Duplicate, Delete…), F2 and Delete.
    ``rename_requested`` asks the panel to rename a figure; ``about_to_modify`` is emitted
    before a figure's files are duplicated or deleted, so pending discussion text is saved first.
    """

    figure_selected = pyqtSignal(object)
    experiment_selected = pyqtSignal(str)
    rename_requested = pyqtSignal(object)
    about_to_modify = pyqtSignal(object)
    refresh_requested = pyqtSignal()

    mode = "discussion"

    def __init__(self, paths, parent=None, settings_manager=None):
        super().__init__(parent)
        self.paths = paths
        self.settings_manager = settings_manager
        self.runs = []
        self._init_ui()
        self.populate()

    # ── Toolbar ──────────────────────────────────────────────────────────────

    def _tool_button(self, icon_name, tooltip, button_id):
        button = style_tool_button(DraggableToolButton(button_id), icon_name, tooltip)
        return button

    def _toolbar_groups(self):
        return {
            "row1_left": [self.btn_refresh, self.btn_collapse, self.btn_expand, self.btn_search],
            "row1_right": [],
            "row2_left": [],
            "row2_right": [],
        }

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.btn_refresh = self._tool_button("reload", "Reload experiments, runs and figures from disk", "reload")
        self.btn_refresh.clicked.connect(self.refresh)
        self.btn_collapse = self._tool_button("collapse_all", "Collapse all folders in the tree", "collapse_all")
        self.btn_collapse.clicked.connect(self.collapse_all)
        self.btn_expand = self._tool_button("expand_all", "Expand all folders in the tree", "expand_all")
        self.btn_expand.clicked.connect(self.expand_all)
        self.btn_search = self._tool_button("search", "Filter experiments, runs and figures", "search")

        self._init_tree_toolbar(layout)

        self.search = QLineEdit()
        self.search.setPlaceholderText("Filter experiments, runs and figures…")
        self.search.textChanged.connect(self.filter_tree)
        self.watch_search()
        layout.addWidget(self.search)

        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setIndentation(20)
        self.tree.setItemDelegate(GroupItemDelegate(self.tree))
        self.tree.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.tree.currentItemChanged.connect(self._current_changed)
        self._connect_file_tree(self.tree)
        shortcut = QShortcut(QKeySequence(Qt.Key.Key_F2), self.tree)
        shortcut.setContext(Qt.ShortcutContext.WidgetShortcut)
        shortcut.activated.connect(self._rename_selected)
        layout.addWidget(self.tree, 1)

    def refresh_icons(self):
        refresh_tool_icons(self)
        self.header_widget.update()
        self._visit(lambda item: self._set_icon(item, item.data(0, Qt.ItemDataRole.UserRole) or {}))

    # ── Contents ─────────────────────────────────────────────────────────────

    def _visit(self, function, parent=None):
        parent = parent or self.tree.invisibleRootItem()
        for index in range(parent.childCount()):
            child = parent.child(index)
            function(child)
            self._visit(function, child)

    def _expanded(self) -> set[str]:
        expanded = set()
        self._visit(lambda item: expanded.add(item.data(0, Qt.ItemDataRole.UserRole)["rel"])
                    if item.isExpanded() else None)
        return expanded

    def _set_icon(self, item, data):
        kind = data.get("type")
        if kind in ("folder", "experiment", "run"):
            item.setIcon(0, folder_icon())
        elif kind == "figure":
            item.setIcon(0, svg_icon("plots", TOOL_ICON_COLORS))
        else:
            item.setIcon(0, QIcon())

    def _item(self, text, data):
        item = QTreeWidgetItem([text])
        item.setData(0, Qt.ItemDataRole.UserRole, data)
        item.setToolTip(0, data["rel"])
        self._set_icon(item, data)
        if data.get("is_group"):
            font = group_heading_font(self.tree.font())
            item.setFont(0, font)
            item.setData(0, Qt.ItemDataRole.FontRole, font)
        return item

    def populate(self, select: Path | None = None):
        """Rebuild from disk, keeping expansion and selection (or selecting ``select``)."""
        first = self.tree.topLevelItemCount() == 0
        expanded = self._expanded()
        current = select or self.selected_figure() or self._data().get("rel")
        blocked = self.tree.blockSignals(True)
        try:
            self.tree.clear()
            for node in model.experiment_tree(self.paths, self.runs):
                self._add_node(self.tree.invisibleRootItem(), node)
            self._visit(lambda item: item.setExpanded(
                item.data(0, Qt.ItemDataRole.UserRole)["rel"] in expanded
                or (first and bool(item.data(0, Qt.ItemDataRole.UserRole).get("is_group")))))
            if self.search.text():
                self.filter_tree(self.search.text())
        finally:
            self.tree.blockSignals(blocked)
        item = self._find(current) if current else None
        if item is not None:
            if select is not None:
                if item.isHidden():
                    self.search.clear()
                parent = item.parent()
                while parent is not None:
                    parent.setExpanded(True)
                    parent = parent.parent()
            self.tree.setCurrentItem(item)  # emits the selection again
        elif current:
            self._current_changed(None, True)  # the selection is gone
        self._update_delete_action()

    def set_runs(self, entries):
        """Use the same catalog entries and saved-run identities as Results."""
        self.runs = list(entries)
        self.populate()

    def refresh(self):
        self.populate()
        self.refresh_requested.emit()

    def _add_node(self, parent, node):
        data = {"type": node.kind, "rel": node.rel, "is_group": node.kind == "group"}
        item = self._item(node.name, data)
        for child in node.children:
            self._add_node(item, child)
        if node.kind in ("experiment", "run"):
            for path in model.figures(self.paths, node.rel):
                figure = self._item(path.stem, {"type": "figure", "rel": f"{node.rel}/{path.stem}",
                                                "path": str(path), "experiment": node.rel})
                item.addChild(figure)
        if item.childCount():
            parent.addChild(item)

    def _find(self, target):
        """The item of a figure path, or of a group, folder or experiment rel."""
        found = []
        key = ("path", str(target)) if isinstance(target, Path) else ("rel", str(target))

        def check(item):
            data = item.data(0, Qt.ItemDataRole.UserRole) or {}
            if not found and data.get(key[0]) == key[1] and (key[0] == "path" or data.get("type") != "figure"):
                found.append(item)

        self._visit(check)
        return found[0] if found else None

    def filter_tree(self, text):
        query = text.strip().lower()

        def match(item):
            data = item.data(0, Qt.ItemDataRole.UserRole) or {}
            own = query in item.text(0).lower() or query in data.get("rel", "").lower()
            children = [match(item.child(i)) for i in range(item.childCount())]
            visible = own or any(children) or not query
            item.setHidden(not visible)
            if query and visible:
                item.setExpanded(True)
            return visible

        for index in range(self.tree.topLevelItemCount()):
            match(self.tree.topLevelItem(index))
        self._update_delete_action()

    # ── Selection ────────────────────────────────────────────────────────────

    def _data(self):
        item = self.tree.currentItem()
        return (item.data(0, Qt.ItemDataRole.UserRole) or {}) if item is not None else {}

    def selected_figure(self) -> Path | None:
        data = self._data()
        return Path(data["path"]) if data.get("type") == "figure" else None

    def selected_experiment(self) -> str:
        data = self._data()
        if data.get("type") == "figure":
            return data["experiment"]
        return data["rel"] if data.get("type") in ("experiment", "run") else ""

    def select_figure(self, path: Path):
        item = self._find(Path(path))
        if item is not None:
            self.tree.setCurrentItem(item)

    def _current_changed(self, current, previous):
        if previous is None and current is None:
            return
        self.experiment_selected.emit(self.selected_experiment())
        self.figure_selected.emit(self.selected_figure())

    # ── Adding figures ───────────────────────────────────────────────────────

    def add_figure(self, experiment: str, capability: str) -> Path:
        path = model.create_figure(self.paths, experiment, capability)
        self.populate(select=path)
        return path

    # ── Managing figures ─────────────────────────────────────────────────────

    def _figure_at(self, item) -> Path | None:
        data = (item.data(0, Qt.ItemDataRole.UserRole) or {}) if item is not None else {}
        return Path(data["path"]) if data.get("type") == "figure" else None

    def _add_context_actions(self, menu, item, _items):
        path = self._figure_at(item)
        if path is not None:
            menu.addAction("Rename…", lambda: self.rename_requested.emit(path))
            menu.addAction("Duplicate", lambda: self.duplicate_figure(path))

    def deletion_targets(self, items):
        if not items:
            return []
        item = items[0]  # Discussion uses single selection.
        path = self._figure_at(item)
        if path is not None:
            return [(None, [path])]

        paths = []

        def collect(child):
            path = self._figure_at(child)
            if path is not None:
                paths.append(path)

        self._visit(collect, item)
        if not paths:
            return []
        folder = item.data(0, Qt.ItemDataRole.UserRole)["rel"]
        return [(folder, paths)]

    def _prompt_delete_targets(self, targets):
        folder, paths = targets[0]
        if folder is None:
            self.prompt_delete(paths[0])
            return
        count = len(paths)
        figures = "figure" if count == 1 else "figures"
        answer = QMessageBox.question(
            self, "Delete discussion folder",
            f"Delete '{folder}' and all {count} {figures} and associated discussion text inside it?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
        if answer != QMessageBox.StandardButton.Yes:
            return
        errors = self._delete_figures(paths)
        if errors:
            QMessageBox.warning(self, "Delete failed", "\n".join(f"{path.name}: {error}" for path, error in errors))

    def _rename_selected(self):
        path = self.selected_figure()
        if path is not None:
            self.rename_requested.emit(path)

    def duplicate_figure(self, path: Path) -> Path:
        self.about_to_modify.emit(Path(path))
        copy = model.duplicate_figure(path)
        self.populate(select=copy)
        return copy

    def prompt_delete(self, path: Path) -> bool:
        path = Path(path)
        text = " and its discussion" if model.discussion_path(path).is_file() else ""
        answer = QMessageBox.question(
            self, "Delete figure", f"Delete the figure '{path.stem}'{text}?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
        if answer != QMessageBox.StandardButton.Yes:
            return False
        try:
            self.delete_figure(path)
        except OSError as exc:
            QMessageBox.warning(self, "Delete failed", str(exc))
            return False
        return True

    def delete_figure(self, path: Path):
        errors = self._delete_figures([Path(path)])
        if errors:
            raise errors[0][1]

    def _delete_figures(self, paths: list[Path]) -> list[tuple[Path, OSError]]:
        errors = []
        selected = self.selected_figure()
        for path in paths:
            self.about_to_modify.emit(path)
            try:
                model.delete_figure(path)
            except OSError as exc:
                errors.append((path, exc))
        if selected is not None and selected in paths and not selected.is_file():
            item = self._find(selected)
            parent = item.parent() if item is not None else None
            if parent is not None:
                self.tree.setCurrentItem(parent)
        self.populate()
        return errors

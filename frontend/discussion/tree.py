"""Discussion experiments, saved runs and figures, using the shared tree tools and Results catalog."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (QAbstractItemView, QApplication, QLineEdit, QMenu, QTreeWidget, QTreeWidgetItem,
                             QVBoxLayout, QWidget)

from ..config_panel.controls import mark_action
from ..sidetabs import svg_icon
from ..tree_tools import (TOOL_ICON_COLORS, DraggableToolButton, GroupItemDelegate, TreeHeaderActions,
                          TreeHeaderWidget, TreeToolbarPersistence, folder_icon, group_heading_font,
                          refresh_tool_icons, style_tool_button)
from . import model

NO_CANDIDATES = "No compatible plot components"


class DiscussionTree(TreeToolbarPersistence, TreeHeaderActions, QWidget):
    """Experiment tree with figure leaves.

    ``figure_selected`` carries the selected figure file, or None when the selection is not a
    figure; ``experiment_selected`` carries the experiment of the selection ("" for a group or
    folder). ``regenerate_requested`` asks for the selected figure to be rebuilt.
    """

    figure_selected = pyqtSignal(object)
    experiment_selected = pyqtSignal(str)
    regenerate_requested = pyqtSignal()
    refresh_requested = pyqtSignal()

    mode = "discussion"

    def __init__(self, paths, parent=None, settings_manager=None, candidates=None):
        super().__init__(parent)
        self.paths = paths
        self.settings_manager = settings_manager
        self.runs = []
        # experiment rel -> FigureCandidate list; composes the experiment by default.
        self.candidates = candidates or (lambda rel: model.figure_candidates(self.paths, rel))
        self._init_ui()
        self.populate()

    # ── Toolbar ──────────────────────────────────────────────────────────────

    def _tool_button(self, icon_name, tooltip, button_id):
        button = style_tool_button(DraggableToolButton(button_id), icon_name, tooltip)
        return button

    def _default_toolbar_groups(self):
        return {
            "row1_left": [self.btn_refresh, self.btn_collapse, self.btn_expand, self.btn_search],
            "row1_right": [self.btn_add, self.btn_regenerate],
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
        self.btn_add = self._tool_button("plus", "Add a figure from a compatible plot component", "plus")
        self.btn_add.clicked.connect(self.open_add_menu)
        self.btn_regenerate = self._tool_button("regenerate", "Regenerate the figure (picks up new runs)",
                                                "regenerate")
        self.btn_regenerate.clicked.connect(self.regenerate_requested)

        self.header_widget = TreeHeaderWidget(self)
        self.header_widget.set_groups(self._load_toolbar_groups())
        self.header_widget.layout_changed.connect(self._save_toolbar_layout)
        layout.addWidget(self.header_widget)

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
        layout.addWidget(self.tree, 1)
        self._update_actions()

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

    def _item(self, parent, text, data):
        item = QTreeWidgetItem(parent, [text])
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
                self._add_node(self.tree, node)
            if select is not None:
                expanded |= self._ancestors(model.figure_experiment(self.paths, select))
            self._visit(lambda item: item.setExpanded(
                item.data(0, Qt.ItemDataRole.UserRole)["rel"] in expanded
                or (first and bool(item.data(0, Qt.ItemDataRole.UserRole).get("is_group")))))
            if self.search.text():
                self.filter_tree(self.search.text())
        finally:
            self.tree.blockSignals(blocked)
        item = self._find(current) if current else None
        if item is not None:
            self.tree.setCurrentItem(item)  # emits the selection again
        elif current:
            self._current_changed(None, True)  # the selection is gone
        self._update_actions()

    def set_runs(self, entries):
        """Use the same catalog entries and saved-run identities as Results."""
        self.runs = list(entries)
        self.populate()

    def refresh(self):
        self.populate()
        self.refresh_requested.emit()

    @staticmethod
    def _ancestors(rel: str) -> set[str]:
        parts = rel.split("/")
        return {"/".join(parts[:i]) for i in range(1, len(parts) + 1)}

    def _add_node(self, parent, node):
        data = {"type": node.kind, "rel": node.rel, "is_group": node.kind == "group"}
        item = self._item(parent, node.name, data)
        for child in node.children:
            self._add_node(item, child)
        if node.kind in ("experiment", "run"):
            for path in model.figures(self.paths, node.rel):
                self._item(item, path.stem, {"type": "figure", "rel": f"{node.rel}/{path.stem}",
                                             "path": str(path), "experiment": node.rel})
        return item

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

    def _update_actions(self):
        self.btn_add.setEnabled(bool(self.selected_experiment()))
        self.btn_regenerate.setEnabled(self.selected_figure() is not None)

    def _current_changed(self, current, previous):
        self._update_actions()
        if previous is None and current is None:
            return
        self.experiment_selected.emit(self.selected_experiment())
        self.figure_selected.emit(self.selected_figure())

    # ── Adding figures ───────────────────────────────────────────────────────

    def build_add_menu(self) -> QMenu:
        """The compatible plot components of the selected experiment: blocked ones omitted,
        warned ones marked. Choosing one creates the figure and selects it."""
        menu = QMenu(self)
        experiment = self.selected_experiment()
        if not experiment:
            return menu
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            candidates = self.candidates(experiment)
        except Exception as exc:  # the experiment does not compose
            action = menu.addAction(f"Cannot read {experiment}: {exc}")
            action.setEnabled(False)
            return menu
        finally:
            QApplication.restoreOverrideCursor()
        offered = [c for c in candidates if not c.blocked]
        for candidate in offered:
            action = menu.addAction(candidate.selector)
            action.setData(candidate.selector)
            mark_action(action, candidate.warnings)
            if not action.toolTip() or action.toolTip() == action.text():
                action.setToolTip(f"{candidate.selector} ({candidate.package.id})")
            action.triggered.connect(lambda _=False, s=candidate.selector, e=experiment: self.add_figure(e, s))
        if not offered:
            menu.addAction(NO_CANDIDATES).setEnabled(False)
        return menu

    def open_add_menu(self):
        menu = self.build_add_menu()
        if not menu.isEmpty():
            menu.exec(self.btn_add.mapToGlobal(self.btn_add.rect().bottomLeft()))
        menu.deleteLater()

    def add_figure(self, experiment: str, capability: str) -> Path:
        path = model.create_figure(self.paths, experiment, capability)
        self.populate(select=path)
        return path

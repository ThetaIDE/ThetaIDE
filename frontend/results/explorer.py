"""Run explorer: a checkable tree of group → experiment → agent → version.

It looks and behaves like the Experiment tab's tree, from the same code (frontend/tree_tools.py): the
square reload / collapse all / expand all / search buttons (Esc closes the search), bold group headings
with a rule under them, and folders for experiments. Groups are headings only; experiments, agents and
versions are ticked.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QLineEdit, QMenu, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget

from ..tree_tools import (DraggableToolButton, GroupItemDelegate, TreeHeaderActions, TreeHeaderWidget,
                          TreeToolbarPersistence, folder_icon, refresh_tool_icons, style_tool_button)
from .data import Manifest, Source

_KIND, _DATA = Qt.ItemDataRole.UserRole, Qt.ItemDataRole.UserRole + 1
_CHECKABLE = Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
_PARENT = _CHECKABLE | Qt.ItemFlag.ItemIsAutoTristate
_HEADING = Qt.ItemFlag.ItemIsEnabled  # a group: a heading, not ticked itself


class _RunGroupDelegate(GroupItemDelegate):
    """The experiment tree's group heading look, for this tree's group rows."""

    def is_group(self, index) -> bool:
        return index.data(_KIND) == "group"


class RunExplorer(TreeToolbarPersistence, TreeHeaderActions, QWidget):
    """Tick experiments, agents or versions; every ticked leaf is a series source.

    An agent is a leaf (its newest version) until its manifest shows more than one version;
    then each version becomes a child the user can tick.
    """

    selection_changed = pyqtSignal(list)  # list[Source]
    manifest_wanted = pyqtSignal(str, str)  # group, experiment_id
    refresh_requested = pyqtSignal()
    archive_requested = pyqtSignal(str, str)
    delete_requested = pyqtSignal(str, str)
    archive_group_requested = pyqtSignal(str)
    delete_group_requested = pyqtSignal(str)

    def __init__(self, parent=None, settings_manager=None):
        super().__init__(parent)
        self.settings_manager = settings_manager
        self.mode = "results"
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        def button(icon, tooltip):
            return style_tool_button(DraggableToolButton(icon), icon, tooltip)

        self.btn_refresh = button("reload", "Re-read results/ from disk")
        self.btn_refresh.clicked.connect(self.refresh_requested)
        self.btn_collapse = button("collapse_all", "Collapse all groups")
        self.btn_collapse.clicked.connect(self.collapse_all)
        self.btn_expand = button("expand_all", "Expand all groups and experiments")
        self.btn_expand.clicked.connect(self.expand_all)
        self.btn_search = button("search", "Filter runs")
        self.btn_archive = button("archive", "Archive selected run")
        self.btn_delete = button("trash", "Delete selected run")
        self.btn_archive.clicked.connect(lambda: self._request_selected(self.archive_requested))
        self.btn_delete.clicked.connect(lambda: self._request_selected(self.delete_requested))
        self.header_widget = TreeHeaderWidget(self)
        self.header_widget.set_groups(self._load_toolbar_groups())
        self.header_widget.layout_changed.connect(self._save_toolbar_layout)
        self.header_layout = self.header_widget.header_layout
        layout.addWidget(self.header_widget)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Filter runs…")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self.apply_filter)
        layout.addWidget(self.search)
        self.watch_search()  # hidden until the search button shows it, as in the Experiment tab
        self.tree = QTreeWidget()
        self.tree.setItemDelegate(_RunGroupDelegate(self.tree))
        self.tree.setHeaderHidden(True)
        self.tree.setIndentation(14)
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._context_menu)
        self.tree.itemChanged.connect(self._item_changed)
        self.tree.itemExpanded.connect(self._item_expanded)
        self.tree.itemCollapsed.connect(self._item_collapsed)
        layout.addWidget(self.tree, 1)
        self._collapsed_groups: set[str] = set()  # groups start expanded, as in the Experiment tab
        self._emitting = True

    def _default_toolbar_groups(self):
        return {
            "row1_left": [self.btn_refresh, self.btn_collapse, self.btn_expand, self.btn_search],
            "row1_right": [],
            "row2_left": [self.btn_archive, self.btn_delete],
            "row2_right": [],
        }

    # ── Theme ────────────────────────────────────────────────────────────────

    def refresh_icons(self):
        """Re-render the header and folder icons in the current theme's colors."""
        refresh_tool_icons(self)
        for item in self._items():
            if item.data(0, _KIND) == "experiment":
                item.setIcon(0, folder_icon())

    # ── Building ─────────────────────────────────────────────────────────────

    def set_runs(self, entries: list[dict]):
        """(Re)build from catalog.list_runs items, keeping what was ticked and expanded."""
        checked = set(self.checked_sources())
        expanded = {item.data(0, _DATA) for item in self._items() if item.isExpanded()}
        self._emitting = False
        self.tree.clear()
        groups: dict[str, QTreeWidgetItem] = {}
        for entry in sorted(entries, key=lambda e: (e["group"], e["experiment_id"])):
            group, exp = entry["group"], entry["experiment_id"]
            agents = [a["name"] for a in entry.get("agents") or [] if a.get("name")]
            if not agents:
                continue
            if group not in groups:
                groups[group] = self._item(self.tree.invisibleRootItem(), group, "group", group, _HEADING)
            exp_item = self._item(groups[group], exp, "experiment", f"{group}/{exp}", _PARENT)
            exp_item.setToolTip(0, f"results/logs/{group}/{exp}")
            exp_item.setIcon(0, folder_icon())
            for agent in agents:
                self._item(exp_item, agent, "agent", Source(group, exp, agent), _CHECKABLE)
        for item in self._items():
            group = item.data(0, _KIND) == "group"
            if (group and item.data(0, _DATA) not in self._collapsed_groups) or item.data(0, _DATA) in expanded:
                item.setExpanded(True)
        for item in self._leaves():
            if item.data(0, _DATA) in checked:
                item.setCheckState(0, Qt.CheckState.Checked)
        self.apply_filter(self.search.text())
        self._emitting = True
        self._emit()

    def add_manifest(self, manifest: Manifest):
        """Turn each multi-version agent of this run into a parent with one child per version."""
        self._emitting = False
        for item in self._items():
            source = item.data(0, _DATA)
            if item.data(0, _KIND) != "agent" or source.run_key != manifest.run_key or item.childCount():
                continue
            versions = manifest.agents.get(source.agent) or []
            if len(versions) < 2:
                continue
            was_checked = item.checkState(0) == Qt.CheckState.Checked
            item.setFlags(_PARENT)
            for version in versions:
                child = self._item(
                    item,
                    version,
                    "version",
                    Source(source.group, source.experiment_id, source.agent, version),
                    _CHECKABLE,
                )
                # A ticked agent meant "its newest version": keep exactly that ticked.
                if was_checked and version == versions[-1]:
                    child.setCheckState(0, Qt.CheckState.Checked)
        self._emitting = True
        self._emit()

    def _item(self, parent, text, kind, data, flags):
        item = QTreeWidgetItem(parent, [text])
        item.setData(0, _KIND, kind)
        item.setData(0, _DATA, data)
        item.setFlags(flags)
        if flags & Qt.ItemFlag.ItemIsUserCheckable:  # a group heading has no checkbox
            item.setCheckState(0, Qt.CheckState.Unchecked)
        return item

    # ── Reading ──────────────────────────────────────────────────────────────

    def _items(self):
        stack = [self.tree.topLevelItem(i) for i in range(self.tree.topLevelItemCount())]
        while stack:
            item = stack.pop()
            yield item
            stack.extend(item.child(i) for i in range(item.childCount()))

    def _leaves(self):
        return [item for item in self._items() if item.data(0, _KIND) in ("agent", "version") and not item.childCount()]

    def checked_sources(self) -> list[Source]:
        """Ticked series sources, in tree order."""
        leaves = [item for item in self._leaves() if item.checkState(0) == Qt.CheckState.Checked]
        order = {id(item): i for i, item in enumerate(self._items_in_order())}
        return [item.data(0, _DATA) for item in sorted(leaves, key=lambda it: order[id(it)])]

    def _items_in_order(self):
        def walk(item):
            yield item
            for i in range(item.childCount()):
                yield from walk(item.child(i))

        for i in range(self.tree.topLevelItemCount()):
            yield from walk(self.tree.topLevelItem(i))

    # ── Events ───────────────────────────────────────────────────────────────

    def _run_for_item(self, item):
        while item and item.data(0, _KIND) not in ("experiment", "group"):
            item = item.parent()
        if item and item.data(0, _KIND) == "experiment":
            return tuple(item.data(0, _DATA).split("/", 1))
        return None

    def _request_selected(self, signal):
        run = self._run_for_item(self.tree.currentItem())
        if run:
            signal.emit(*run)

    def _context_menu(self, position):
        item = self.tree.itemAt(position)
        if not item:
            return
        menu = QMenu(self.tree)
        if item.data(0, _KIND) == "group":
            group = item.data(0, _DATA)
            menu.addAction("Archive group…", lambda: self.archive_group_requested.emit(group))
            menu.addAction("Delete group…", lambda: self.delete_group_requested.emit(group))
        else:
            run = self._run_for_item(item)
            if not run:
                return
            group, experiment = run
            menu.addAction("Archive run…", lambda: self.archive_requested.emit(group, experiment))
            menu.addAction("Delete run…", lambda: self.delete_requested.emit(group, experiment))
        menu.exec(self.tree.viewport().mapToGlobal(position))

    def _item_changed(self, item, _column):
        if item.data(0, _KIND) == "agent" and item.checkState(0) != Qt.CheckState.Unchecked:
            source = item.data(0, _DATA)
            self.manifest_wanted.emit(source.group, source.experiment_id)
        self._emit()

    def _item_collapsed(self, item):
        if item.data(0, _KIND) == "group" and self._emitting:
            self._collapsed_groups.add(item.data(0, _DATA))

    def _item_expanded(self, item):
        if item.data(0, _KIND) == "group":
            self._collapsed_groups.discard(item.data(0, _DATA))
        if item.data(0, _KIND) == "experiment":
            group, exp = item.data(0, _DATA).split("/", 1)
            self.manifest_wanted.emit(group, exp)

    def _emit(self):
        if self._emitting:
            self.selection_changed.emit(self.checked_sources())

    def apply_filter(self, text):
        """Show experiments whose group/name contains `text`, and the groups that hold them."""
        needle = text.strip().lower()
        for g in range(self.tree.topLevelItemCount()):
            group = self.tree.topLevelItem(g)
            any_visible = False
            for e in range(group.childCount()):
                exp = group.child(e)
                visible = not needle or needle in exp.data(0, _DATA).lower()
                exp.setHidden(not visible)
                any_visible |= visible
            group.setHidden(not any_visible)

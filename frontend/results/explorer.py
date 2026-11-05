"""Run explorer: a tree of group → experiment → run → agent → version.

It looks and behaves like the Experiment tab's tree, from the same code (frontend/tree_tools.py): the
square reload / collapse all / expand all / search buttons (Esc closes the search), bold group headings
with a rule under them, and folders for experiments. Runs, agents and versions can be selected.
"""

from __future__ import annotations

import re

from PyQt6.QtCore import QEvent, QRect, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter
from PyQt6.QtWidgets import QLineEdit, QMenu, QStyleOptionViewItem, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget

from ..tree_tools import (DraggableToolButton, GroupItemDelegate, TreeHeaderActions, TreeHeaderWidget,
                          TreeToolbarPersistence, folder_icon, refresh_tool_icons, style_tool_button)
from .data import Manifest, Source

_KIND, _DATA = Qt.ItemDataRole.UserRole, Qt.ItemDataRole.UserRole + 1
_CHECKABLE = Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
_PARENT = _CHECKABLE
_HEADING = Qt.ItemFlag.ItemIsEnabled  # a group: a heading, not ticked itself
_RUN_SUFFIX = re.compile(r"_\d{8}-\d{6}(?:-\d+)?$")


class _RunGroupDelegate(GroupItemDelegate):
    """The experiment tree's group heading look, for this tree's group rows."""

    def is_group(self, index) -> bool:
        return index.data(_KIND) == "group"

    def initStyleOption(self, option, index):
        super().initStyleOption(option, index)
        if index.flags() & Qt.ItemFlag.ItemIsUserCheckable:
            option.features &= ~QStyleOptionViewItem.ViewItemFeature.HasCheckIndicator

    @staticmethod
    def _switch_rect(option):
        return QRect(option.rect.right() - 39, option.rect.center().y() - 8, 32, 16)

    def paint(self, painter, option, index):
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        checkable = bool(index.flags() & Qt.ItemFlag.ItemIsUserCheckable)
        if checkable:
            opt.features &= ~QStyleOptionViewItem.ViewItemFeature.HasCheckIndicator
            opt.rect.setRight(opt.rect.right() - 44)
        super().paint(painter, opt, index)
        if not checkable:
            return
        rect = self._switch_rect(option)
        state = index.data(Qt.ItemDataRole.CheckStateRole)
        on = state != Qt.CheckState.Unchecked.value
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(option.palette.highlight() if on else option.palette.mid())
        painter.drawRoundedRect(rect, 8, 8)
        painter.setBrush(QColor("white"))
        x = (rect.center().x() - 6 if state == Qt.CheckState.PartiallyChecked.value
             else rect.right() - 13 if on else rect.left() + 2)
        painter.drawEllipse(x, rect.top() + 2, 12, 12)
        painter.restore()

    def editorEvent(self, event, model, option, index):
        if not index.flags() & Qt.ItemFlag.ItemIsUserCheckable:
            return False
        if event.type() == QEvent.Type.MouseButtonRelease:
            if event.button() != Qt.MouseButton.LeftButton or not self._switch_rect(option).contains(event.pos()):
                return False
        elif event.type() == QEvent.Type.KeyPress:
            if event.key() not in (Qt.Key.Key_Space, Qt.Key.Key_Select):
                return False
        else:
            return False
        state = index.data(Qt.ItemDataRole.CheckStateRole)
        new_state = Qt.CheckState.Unchecked if state == Qt.CheckState.Checked.value else Qt.CheckState.Checked
        return model.setData(index, new_state.value, Qt.ItemDataRole.CheckStateRole)


class RunExplorer(TreeToolbarPersistence, TreeHeaderActions, QWidget):
    """Select runs, agents or versions; every selected leaf is a series source.

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
        self.tree.setIndentation(20)
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._context_menu)
        self.tree.itemChanged.connect(self._item_changed)
        self.tree.itemExpanded.connect(self._item_expanded)
        self.tree.itemCollapsed.connect(self._item_collapsed)
        layout.addWidget(self.tree, 1)
        self._collapsed_groups: set[str] = set()  # groups start expanded, as in the Experiment tab
        self._emitting = True
        self._updating = False
        self._states = {}
        self._restore_sources: set[Source] = set()

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
            if item.data(0, _KIND) in ("experiment", "run"):
                item.setIcon(0, folder_icon())

    # ── Building ─────────────────────────────────────────────────────────────

    def set_runs(self, entries: list[dict]):
        """(Re)build from catalog.list_runs items, keeping what was ticked and expanded."""
        checked = set(self.checked_sources())
        self._restore_sources = {source for source in checked if source.version is not None}
        expanded = {(item.data(0, _KIND), item.data(0, _DATA)) for item in self._items() if item.isExpanded()}
        self._emitting = False
        self._states.clear()
        self.tree.clear()
        groups: dict[str, QTreeWidgetItem] = {}
        experiments: dict[tuple[str, str], QTreeWidgetItem] = {}
        for entry in sorted(entries, key=lambda e: (e["group"], e["experiment_id"])):
            group, exp = entry["group"], entry["experiment_id"]
            agents = [a["name"] for a in entry.get("agents") or [] if a.get("name")]
            if not agents and not entry.get("has_run_logs"):
                continue
            if not agents:
                agents = [""]  # A run can fail before any agent writes metrics.
            if group not in groups:
                groups[group] = self._item(self.tree.invisibleRootItem(), group, "group", group, _HEADING)
            name = _RUN_SUFFIX.sub("", exp)
            key = (group, name)
            if key not in experiments:
                exp_item = self._item(groups[group], name, "experiment", key,
                                      Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                exp_item.setIcon(0, folder_icon())
                experiments[key] = exp_item
            run_item = self._item(experiments[key], exp, "run", f"{group}/{exp}", _PARENT)
            run_item.setToolTip(0, f"results/logs/{group}/{exp}")
            run_item.setIcon(0, folder_icon())
            for agent in agents:
                self._item(run_item, agent or "Run files", "agent", Source(group, exp, agent), _CHECKABLE)
        for item in self._items():
            group = item.data(0, _KIND) == "group"
            if (group and item.data(0, _DATA) not in self._collapsed_groups) or (item.data(0, _KIND), item.data(0, _DATA)) in expanded:
                item.setExpanded(True)
        for item in self._leaves():
            source = item.data(0, _DATA)
            if source in checked or any(s.run_key == source.run_key and s.agent == source.agent for s in self._restore_sources):
                item.setCheckState(0, Qt.CheckState.Checked)
        for item in self._items():
            if item.data(0, _KIND) == "run":
                self._sync_parent(item)
        self.apply_filter(self.search.text())
        self._emitting = True
        self._emit()
        for source in self._restore_sources.copy():
            self.manifest_wanted.emit(source.group, source.experiment_id)

    def select_run(self, group: str, experiment: str) -> bool:
        """Select all methods of one run, replacing the current comparison."""
        target = next((item for item in self._items()
                       if item.data(0, _KIND) == "run" and item.data(0, _DATA) == f"{group}/{experiment}"), None)
        if target is None:
            return False
        self._updating = True
        for item in self._items():
            if item.data(0, _KIND) == "run":
                for i in range(item.childCount()):
                    self._set_method(item.child(i), item is target)
                self._sync_parent(item)
        self._updating = False
        target.setExpanded(True)
        target.parent().setExpanded(True)
        target.parent().parent().setExpanded(True)
        self.manifest_wanted.emit(group, experiment)
        self._emit()
        return True

    def add_manifest(self, manifest: Manifest):
        """Turn each multi-version agent of this run into a parent with one child per version."""
        self._emitting = False
        self._updating = True
        for item in self._items():
            source = item.data(0, _DATA)
            if item.data(0, _KIND) != "agent" or source.run_key != manifest.run_key or item.childCount():
                continue
            versions = manifest.agents.get(source.agent) or []
            if len(versions) < 2:
                continue
            was_checked = item.checkState(0) != Qt.CheckState.Unchecked
            restored = {s.version for s in self._restore_sources if s.run_key == source.run_key and s.agent == source.agent}
            item.setFlags(_PARENT)
            details = manifest.version_details.get(source.agent, {})
            seeds = {}
            for version in versions:
                parent = details.get(version, {}).get("trial_version")
                if parent:
                    seeds.setdefault(parent, []).append(version)
            parents = list(dict.fromkeys([details.get(v, {}).get("trial_version") or v for v in versions]))
            for version in parents:
                seed_versions = seeds.get(version, [])
                info = details.get(seed_versions[0], {}) if seed_versions else details.get(version, {})
                title = f"{info.get('study')} · trial {info['trial_number']}" if seed_versions else version
                child = self._item(
                    item,
                    title,
                    "version",
                    Source(source.group, source.experiment_id, source.agent, version),
                    _CHECKABLE,
                )
                child.setToolTip(0, version)
                # A ticked agent meant "its newest version": keep exactly that ticked.
                selected = version in restored or (was_checked and not restored and version == parents[-1])
                for seed_version in sorted(seed_versions, key=lambda v: details[v].get("seed") or 0):
                    seed = details[seed_version].get("seed")
                    leaf = self._item(child, f"Seed {seed}", "version",
                                      Source(source.group, source.experiment_id, source.agent, seed_version,
                                             f"{title} · seed {seed}", version), _CHECKABLE)
                    leaf.setToolTip(0, seed_version)
                    if selected or seed_version in restored:
                        leaf.setCheckState(0, Qt.CheckState.Checked)
                if seed_versions:
                    self._sync_parent(child)
                elif selected:
                    child.setCheckState(0, Qt.CheckState.Checked)
            self._sync_parent(item)
            self._sync_parent(item.parent())
        self._restore_sources = {s for s in self._restore_sources if s.run_key != manifest.run_key}
        self._updating = False
        self._emitting = True
        self._emit()

    def select_version(self, run_key, agent, version):
        target = next((item for item in self._items() if item.data(0, _KIND) == "version"
                       and (source := item.data(0, _DATA)).run_key == run_key
                       and source.agent == agent and source.version == version), None)
        if target is None:
            return False
        self._updating = True
        for leaf in self._leaves():
            leaf.setCheckState(0, Qt.CheckState.Unchecked)
        self._set_version(target, True)
        # Children precede parents so every ancestor reflects the new selection.
        for item in reversed(list(self._items_in_order())):
            self._sync_parent(item)
        self._updating = False
        parent = target
        while parent:
            parent.setExpanded(True)
            parent = parent.parent()
        self.tree.setCurrentItem(target)
        self._emit()
        return True

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
        while item and item.data(0, _KIND) not in ("run", "experiment", "group"):
            item = item.parent()
        if item and item.data(0, _KIND) == "run":
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
        if item.data(0, _KIND) not in ("run", "agent", "version"):
            return
        state = item.checkState(0)
        previous = self._states.get(id(item), Qt.CheckState.Unchecked)
        self._states[id(item)] = state
        if not self._emitting or self._updating or state == previous:
            return
        self._updating = True
        enabled = item.checkState(0) != Qt.CheckState.Unchecked
        if item.data(0, _KIND) == "run":
            for i in range(item.childCount()):
                self._set_method(item.child(i), enabled)
            self._sync_parent(item)
        elif item.data(0, _KIND) == "agent":
            self._set_method(item, enabled)
            self._sync_parent(item.parent())
        else:
            self._set_version(item, enabled)
            parent = item.parent()
            while parent:
                self._sync_parent(parent)
                parent = parent.parent()
        self._updating = False
        if enabled:
            run = self._run_for_item(item)
            if run:
                self.manifest_wanted.emit(*run)
        self._emit()

    def _set_method(self, item, enabled):
        if item.childCount():
            for i in range(item.childCount()):
                self._set_version(item.child(i), enabled and i == item.childCount() - 1)
        item.setCheckState(0, Qt.CheckState.Checked if enabled else Qt.CheckState.Unchecked)

    def _set_version(self, item, enabled):
        for i in range(item.childCount()):
            self._set_version(item.child(i), enabled)
        item.setCheckState(0, Qt.CheckState.Checked if enabled else Qt.CheckState.Unchecked)

    def _sync_parent(self, item):
        if not item or not item.childCount():
            return
        states = [item.child(i).checkState(0) for i in range(item.childCount())]
        # A method is enabled when any version is selected; a run is enabled when all methods are.
        if item.data(0, _KIND) == "agent":
            state = Qt.CheckState.Checked if any(s != Qt.CheckState.Unchecked for s in states) else Qt.CheckState.Unchecked
        elif all(s == Qt.CheckState.Checked for s in states):
            state = Qt.CheckState.Checked
        elif all(s == Qt.CheckState.Unchecked for s in states):
            state = Qt.CheckState.Unchecked
        else:
            state = Qt.CheckState.PartiallyChecked
        item.setCheckState(0, state)

    def _item_collapsed(self, item):
        if item.data(0, _KIND) == "group" and self._emitting:
            self._collapsed_groups.add(item.data(0, _DATA))

    def _item_expanded(self, item):
        if item.data(0, _KIND) == "group":
            self._collapsed_groups.discard(item.data(0, _DATA))
        if item.data(0, _KIND) == "run":
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
                matches_experiment = not needle or needle in "/".join(exp.data(0, _DATA)).lower()
                visible = False
                for r in range(exp.childCount()):
                    run = exp.child(r)
                    run_visible = matches_experiment or needle in run.data(0, _DATA).lower()
                    run.setHidden(not run_visible)
                    visible |= run_visible
                exp.setHidden(not visible)
                any_visible |= visible
            group.setHidden(not any_visible)

"""Run explorer: a tree of group → experiment → run → agent → version.

It looks and behaves like the Experiment tab's tree, from the same code (frontend/tree_tools.py): the
square reload / collapse all / expand all / search buttons (Esc closes the search), bold group headings
with a rule under them, and folders for experiments. Runs, agents and versions can be selected.
"""

from __future__ import annotations

from pathlib import PurePosixPath

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QAbstractItemView, QLineEdit, QMenu, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget

from ..tree_tools import (DraggableToolButton, GroupItemDelegate, TreeHeaderActions, TreeHeaderWidget,
                          TreeToolbarPersistence, folder_icon, refresh_tool_icons, style_tool_button)
from .data import Manifest, Source
from src.app.pipeline.result_paths import RUN_SUFFIX, SAVED_RUN_NAME, result_experiment_id

_KIND, _DATA, _TIME = Qt.ItemDataRole.UserRole, Qt.ItemDataRole.UserRole + 1, Qt.ItemDataRole.UserRole + 2
_CHECKABLE = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
_PARENT = _CHECKABLE
_HEADING = _CHECKABLE


class _RunGroupDelegate(GroupItemDelegate):
    """The experiment tree's group heading look, for this tree's group rows."""

    def is_group(self, index) -> bool:
        return index.data(_KIND) == "group"



class RunExplorer(TreeToolbarPersistence, TreeHeaderActions, QWidget):
    """Select a run, agent or version to display its results.
    """

    selection_changed = pyqtSignal(list)  # list[Source]
    manifest_wanted = pyqtSignal(str, str)  # group, experiment_id
    refresh_requested = pyqtSignal()
    archive_requested = pyqtSignal(str, str)
    delete_requested = pyqtSignal(str, str)
    delete_selection_requested = pyqtSignal(list)  # (group, run or None)
    archive_group_requested = pyqtSignal(str)
    delete_group_requested = pyqtSignal(str)
    rename_requested = pyqtSignal(str, str, str)  # kind, group, experiment/run ID (empty for groups)

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
        self.btn_delete = button("trash", "Delete selected folders")
        self.btn_archive.clicked.connect(lambda: self._request_selected(self.archive_requested))
        self.btn_delete.clicked.connect(self._request_delete_selection)
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
        self.tree.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.tree.setIndentation(20)
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._context_menu)
        self.tree.currentItemChanged.connect(self._current_changed)
        self.tree.itemExpanded.connect(self._item_expanded)
        self.tree.itemCollapsed.connect(self._item_collapsed)
        layout.addWidget(self.tree, 1)
        self._collapsed_groups: set[str] = set()  # groups start expanded, as in the Experiment tab
        self._emitting = True
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
        """(Re)build from catalog.list_runs items, keeping the selection and expanded rows."""
        current = self.tree.currentItem()
        selected = (current.data(0, _KIND), current.data(0, _DATA)) if current else None
        expanded = {(item.data(0, _KIND), item.data(0, _DATA)) for item in self._items() if item.isExpanded()}
        self._emitting = False
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
            name = (PurePosixPath(exp).parent.as_posix() if (entry.get("config") or {}).get("saved_run")
                    else result_experiment_id(exp))
            key = (group, name)
            if key not in experiments:
                exp_item = self._item(groups[group], PurePosixPath(name).name, "experiment", key,
                                      Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                exp_item.setToolTip(0, f"results/logs/{group}/{name}")
                exp_item.setIcon(0, folder_icon())
                experiments[key] = exp_item
            run_item = self._item(experiments[key], PurePosixPath(exp).name, "run", f"{group}/{exp}", _PARENT)
            times = [a.get("started") for a in entry.get("agents") or [] if isinstance(a.get("started"), str)]
            suffix = RUN_SUFFIX.search(exp)
            stamp = SAVED_RUN_NAME.fullmatch(PurePosixPath(exp).name)
            fallback_time = (stamp.group()[:15].replace("-", "T") if stamp else
                             suffix.group()[1:16].replace("-", "T") if suffix else "")
            run_item.setData(0, _TIME, max(times, default=fallback_time))
            note = f"results/logs/{group}/{exp}"
            if entry.get('incomplete_results'):
                run_item.setText(0, run_item.text(0) + ' · incomplete results')
                note += '\nJob ended before result publication completed. Showing the downloaded partial results.'
            run_item.setToolTip(0, note)
            run_item.setIcon(0, folder_icon())
            for agent in agents:
                self._item(run_item, agent or "Run files", "agent", Source(group, exp, agent), _CHECKABLE)
        for item in self._items():
            group = item.data(0, _KIND) == "group"
            if (group and item.data(0, _DATA) not in self._collapsed_groups) or (item.data(0, _KIND), item.data(0, _DATA)) in expanded:
                item.setExpanded(True)
        self.apply_filter(self.search.text())
        target = next((item for item in self._items() if (item.data(0, _KIND), item.data(0, _DATA)) == selected), None)
        # Version rows arrive asynchronously with the manifest.
        self._restore_sources = {selected[1]} if selected and selected[0] == "version" else set()
        if self._restore_sources:
            source = selected[1]
            target = next((item for item in self._items() if item.data(0, _KIND) == "agent"
                           and item.data(0, _DATA).run_key == source.run_key
                           and item.data(0, _DATA).agent == source.agent), None)
        self.tree.setCurrentItem(target)
        self._emitting = True
        self._emit()
        for source in self._restore_sources.copy():
            self.manifest_wanted.emit(source.group, source.experiment_id)

    def select_run(self, group: str, experiment: str) -> bool:
        """Select all methods of one run."""
        target = next((item for item in self._items()
                       if item.data(0, _KIND) == "run" and item.data(0, _DATA) == f"{group}/{experiment}"), None)
        if target is None:
            return False
        self.tree.setCurrentItem(target)
        target.setExpanded(True)
        target.parent().setExpanded(True)
        target.parent().parent().setExpanded(True)
        self.manifest_wanted.emit(group, experiment)
        return True

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
                for seed_version in sorted(seed_versions, key=lambda v: details[v].get("seed") or 0):
                    seed = details[seed_version].get("seed")
                    leaf = self._item(child, f"Seed {seed}", "version",
                                      Source(source.group, source.experiment_id, source.agent, seed_version,
                                             f"{title} · seed {seed}", version), _CHECKABLE)
                    leaf.setToolTip(0, seed_version)
        restored = next((s for s in self._restore_sources if s.run_key == manifest.run_key), None)
        if restored:
            target = next((item for item in self._items() if item.data(0, _KIND) == "version"
                           and item.data(0, _DATA) == restored), None)
            if target:
                self.tree.setCurrentItem(target)
        self._restore_sources = {s for s in self._restore_sources if s.run_key != manifest.run_key}
        self._emitting = True
        self._emit()

    def select_version(self, run_key, agent, version):
        target = next((item for item in self._items() if item.data(0, _KIND) == "version"
                       and (source := item.data(0, _DATA)).run_key == run_key
                       and source.agent == agent and source.version == version), None)
        if target is None:
            return False
        parent = target
        while parent:
            parent.setExpanded(True)
            parent = parent.parent()
        self.tree.setCurrentItem(target)
        return True

    def _item(self, parent, text, kind, data, flags):
        item = QTreeWidgetItem(parent, [text])
        item.setData(0, _KIND, kind)
        item.setData(0, _DATA, data)
        item.setFlags(flags)
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
        """Sources represented by the current item, using latest descendants for folders."""
        item = self.tree.currentItem()
        if not item:
            return []
        kind = item.data(0, _KIND)
        if kind in ("group", "experiment"):
            runs = [child for child in self._descendants(item) if child.data(0, _KIND) == "run"]
            if not runs:
                return []
            item = max(runs, key=lambda run: (str(run.data(0, _TIME) or "").replace("-", "").replace(":", ""), run.data(0, _DATA)))
            kind = "run"
        if kind == "run":
            return [source for i in range(item.childCount()) for source in self._latest_sources(item.child(i))]
        if kind == "agent":
            return self._latest_sources(item)
        if kind == "version":
            return [leaf.data(0, _DATA) for leaf in self._descendants(item) if not leaf.childCount()] if item.childCount() else [item.data(0, _DATA)]
        return []

    def _descendants(self, item):
        for i in range(item.childCount()):
            child = item.child(i)
            yield child
            yield from self._descendants(child)

    def _latest_sources(self, item):
        if not item.childCount():
            return [item.data(0, _DATA)]
        latest = item.child(item.childCount() - 1)
        return ([leaf.data(0, _DATA) for leaf in self._descendants(latest) if not leaf.childCount()]
                if latest.childCount() else [latest.data(0, _DATA)])

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

    def deletion_targets(self):
        """Expand experiment folders and remove overlapping group/run targets."""
        items = self.tree.selectedItems()
        groups = {item.data(0, _DATA) for item in items if item.data(0, _KIND) == "group"}
        runs = set()
        for item in items:
            if item.data(0, _KIND) == "experiment":
                runs.update(self._run_for_item(child) for child in self._descendants(item)
                            if child.data(0, _KIND) == "run")
            else:
                run = self._run_for_item(item)
                if run:
                    runs.add(run)
        return ([(group, None) for group in sorted(groups)]
                + [run for run in sorted(runs) if run[0] not in groups])

    def _request_delete_selection(self):
        targets = self.deletion_targets()
        if targets:
            self.delete_selection_requested.emit(targets)

    def _context_menu(self, position):
        item = self.tree.itemAt(position)
        if not item:
            return
        # Right-clicking any selected row must preserve the entire selection.
        if not item.isSelected():
            self.tree.setCurrentItem(item)
        menu = QMenu(self.tree)
        if len(self.tree.selectedItems()) == 1:
            kind, identity = item.data(0, _KIND), item.data(0, _DATA)
            if kind in ("group", "experiment", "run"):
                group, experiment = ((identity, "") if kind == "group" else
                                     identity if kind == "experiment" else identity.split("/", 1))
                menu.addAction("Rename…", lambda: self.rename_requested.emit(kind, group, experiment))
            if item.data(0, _KIND) == "group":
                group = item.data(0, _DATA)
                menu.addAction("Archive group…", lambda: self.archive_group_requested.emit(group))
            else:
                run = self._run_for_item(item)
                if run:
                    menu.addAction("Archive run…", lambda: self.archive_requested.emit(*run))
        if self.deletion_targets():
            menu.addAction("Delete selected folders…", self._request_delete_selection)
        menu.exec(self.tree.viewport().mapToGlobal(position))

    def _current_changed(self, item, _previous):
        if not self._emitting:
            return
        self._restore_sources.clear()
        sources = self.checked_sources()
        if sources:
            self.manifest_wanted.emit(sources[0].group, sources[0].experiment_id)
        self._emit()

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

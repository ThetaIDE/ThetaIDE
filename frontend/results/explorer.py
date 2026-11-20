"""Run explorer: a tree of group → folders → experiment → run → agent → version.

Folders mirror the experiment's subdirectories below its group (BlendRL/Cartpole/LearnFast shows
BlendRL → Cartpole → LearnFast).

It looks and behaves like the Experiment tab's tree, from the same code (frontend/tree_tools.py): the
square reload / collapse all / expand all / search buttons (Esc closes the search), bold group headings
with a rule under them, and folders for experiments. Runs, agents and versions can be selected. Its sync
button, like the Experiment tab's, shows only for a run that ran on a site other than local.
"""

from __future__ import annotations

from pathlib import PurePosixPath

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QAbstractItemView, QLineEdit, QTreeWidget, QTreeWidgetItem, QVBoxLayout

from ..tree_tools import (DraggableToolButton, FileTreeWidget, GroupItemDelegate, TOOL_ICON_COLORS,
                          refresh_tool_icons, style_tool_button, svg_icon)
from .data import Manifest, Source
from src.app.pipeline.result_paths import (
    LEGACY_TRIAL_VERSION, SAVED_RUN_NAME, SEED_NAME, TRIAL_NAME, result_experiment_id, saved_run_label,
)


def _results_label() -> str:
    """The workspace-relative results folder, as shown in tooltips."""
    from src.app.workspace import WorkspacePaths

    return WorkspacePaths.for_process().results_relative

_KIND, _DATA, _TIME = Qt.ItemDataRole.UserRole, Qt.ItemDataRole.UserRole + 1, Qt.ItemDataRole.UserRole + 2
_VERSIONS = Qt.ItemDataRole.UserRole + 3
_CHECKABLE = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
_PARENT = _CHECKABLE
_HEADING = _CHECKABLE
_ROW_ICONS = {"folder": "folder", "experiment": "config", "run": "folder", "agent": "gear"}


class _RunGroupDelegate(GroupItemDelegate):
    """The experiment tree's group heading look, for this tree's group rows."""

    def is_group(self, index) -> bool:
        return index.data(_KIND) == "group"



class RunExplorer(FileTreeWidget):
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
    add_figure_requested = pyqtSignal(str)  # group/experiment or group/run ID
    delete_toolbar_group = "row2_left"

    def __init__(self, parent=None, settings_manager=None):
        super().__init__(parent)
        self.settings_manager = settings_manager
        self.mode = "results"
        self.runs: list[dict] = []
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        def button(icon, tooltip):
            return style_tool_button(DraggableToolButton(icon), icon, tooltip)

        self.btn_refresh = button("reload", "Re-read results from disk")
        self.btn_refresh.clicked.connect(self.refresh_requested)
        self.btn_collapse = button("collapse_all", "Collapse all groups")
        self.btn_collapse.clicked.connect(self.collapse_all)
        self.btn_expand = button("expand_all", "Expand all groups and experiments")
        self.btn_expand.clicked.connect(self.expand_all)
        self.btn_search = button("search", "Filter runs")
        self.btn_add_figure = button("plus", "Add figure in Discussion")
        self.btn_add_figure.setEnabled(False)
        self.btn_add_figure.clicked.connect(self._request_add_figure)
        self.btn_archive = button("archive", "Archive selected run")
        self.btn_sync = button("sync", "Sync site data")  # driven by the panel's RunSiteSyncControl
        self.btn_sync.hide()
        self.btn_archive.clicked.connect(lambda: self._request_selected(self.archive_requested))
        self._init_tree_toolbar(layout)
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
        self._connect_file_tree(self.tree)
        self.tree.currentItemChanged.connect(self._current_changed)
        self.tree.itemExpanded.connect(self._item_expanded)
        self.tree.itemCollapsed.connect(self._item_collapsed)
        layout.addWidget(self.tree, 1)
        self._collapsed_groups: set[str] = set()  # groups start expanded, as in the Experiment tab
        self._emitting = True
        self._restore_sources: set[Source] = set()

    def _toolbar_groups(self):
        return {
            "row1_left": [self.btn_refresh, self.btn_collapse, self.btn_expand, self.btn_search],
            "row1_right": [self.btn_add_figure],
            "row2_left": [self.btn_archive],
            "row2_right": [self.btn_sync],
        }

    # ── Theme ────────────────────────────────────────────────────────────────

    def refresh_icons(self):
        """Re-render the header and folder icons in the current theme's colors."""
        refresh_tool_icons(self)
        for item in self._items():
            self._set_row_icon(item)

    @staticmethod
    def _set_row_icon(item):
        kind = item.data(0, _KIND)
        icon = "file" if kind == "agent" and not item.data(0, _DATA).agent else _ROW_ICONS.get(kind)
        if icon:
            item.setIcon(0, svg_icon(icon, TOOL_ICON_COLORS))

    # ── Building ─────────────────────────────────────────────────────────────

    def set_runs(self, entries: list[dict]):
        """(Re)build from catalog.list_runs items, keeping the selection and expanded rows."""
        self.runs = entries
        current = self.tree.currentItem()
        selected = (current.data(0, _KIND), current.data(0, _DATA)) if current else None
        expanded = {(item.data(0, _KIND), item.data(0, _DATA)) for item in self._items() if item.isExpanded()}
        self._emitting = False
        self.tree.clear()
        groups: dict[str, QTreeWidgetItem] = {}
        # One row per (group, path): an experiment whose folder also holds other experiments is both.
        experiments: dict[tuple[str, str], QTreeWidgetItem] = {}

        def folder(group: str, path: PurePosixPath) -> QTreeWidgetItem:
            """The item that holds `path`'s children: the group itself, or a folder below it."""
            if path == PurePosixPath("."):
                return groups[group]
            key = (group, path.as_posix())
            if key not in experiments:
                item = self._item(folder(group, path.parent), path.name, "folder", key,
                                  Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                item.setToolTip(0, f"{_results_label()}/logs/{group}/{path.as_posix()}")
                experiments[key] = item
            return experiments[key]

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
                exp_label = PurePosixPath(name).name
                exp_item = self._item(folder(group, PurePosixPath(name).parent), exp_label, "experiment", key,
                                      Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                exp_item.setToolTip(0, f"{_results_label()}/logs/{group}/{name}")
                experiments[key] = exp_item
            experiments[key].setData(0, _KIND, "experiment")  # a folder row that turns out to be an experiment too
            self._set_row_icon(experiments[key])
            run_label = "Run" if exp == name else saved_run_label(exp)
            run_item = self._item(experiments[key], run_label, "run", f"{group}/{exp}", _PARENT)
            times = [a.get("started") for a in entry.get("agents") or [] if isinstance(a.get("started"), str)]
            stamp = SAVED_RUN_NAME.fullmatch(PurePosixPath(exp).name)
            fallback_time = stamp.group()[:15].replace("-", "T") if stamp and stamp.group()[8:9] == "-" else ""
            run_item.setData(0, _TIME, max(times, default=fallback_time))
            note = f"{_results_label()}/logs/{group}/{exp}"
            if not entry.get("finished"):
                note += "\nIn progress — see Monitor for live monitoring. Generated files are available here."
            if entry.get('incomplete_results'):
                note += '\nJob ended before result publication completed. Showing the downloaded partial results.'
            run_item.setToolTip(0, note)
            for agent in agents:
                method = self._item(run_item, agent or "Run files", "agent", Source(group, exp, agent), _CHECKABLE)
                if agent:
                    method.setChildIndicatorPolicy(QTreeWidgetItem.ChildIndicatorPolicy.ShowIndicator)
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
        parent = target
        while parent:
            parent.setExpanded(True)
            parent = parent.parent()
        self.manifest_wanted.emit(group, experiment)
        return True

    def add_manifest(self, manifest: Manifest):
        """Show trials and their seed runs below each method."""
        self._emitting = False
        current = self.tree.currentItem()
        selected = current.data(0, _DATA) if current and current.data(0, _KIND) == "version" else None
        for item in [row for row in self._items() if row.data(0, _KIND) == "agent"]:
            source = item.data(0, _DATA)
            if source.run_key != manifest.run_key:
                continue
            versions = manifest.agents.get(source.agent) or []
            details = manifest.version_details.get(source.agent, {})
            signature = (tuple(versions), details)
            if signature == item.data(0, _VERSIONS):
                continue
            item.setData(0, _VERSIONS, signature)
            expanded = {child.data(0, _DATA) for child in self._descendants(item) if child.isExpanded()}
            item.takeChildren()
            item.setChildIndicatorPolicy(QTreeWidgetItem.ChildIndicatorPolicy.DontShowIndicatorWhenChildless)
            branches = {}
            for version in versions:
                info = details.get(version, {})
                trial, _, seed_name = version.partition("/")
                number, seed = info.get("trial_number"), info.get("seed")
                if match := TRIAL_NAME.fullmatch(trial):
                    number = int(match[1])
                    if SEED_NAME.fullmatch(seed_name):
                        seed = int(seed_name[5:])
                elif match := LEGACY_TRIAL_VERSION.fullmatch(info.get("trial_version") or version):
                    number = int(match[1])
                key = trial if seed_name else info.get("trial_version") or version
                if seed is not None and number is None and key == version:
                    key = "trial_0"
                title = (f"trial_{number}" if number is not None else
                         "trial_0" if seed is not None else version)
                _, seeds = branches.setdefault(key, (title, []))
                if seed is not None:
                    seeds.append((seed, version))
                    branches[key] = (title, seeds)
            if len(versions) < 2 and not any(TRIAL_NAME.fullmatch(title) for title, _ in branches.values()):
                continue
            for version, (title, seed_versions) in branches.items():
                child = self._item(
                    item,
                    title,
                    "version",
                    Source(source.group, source.experiment_id, source.agent, version),
                    _CHECKABLE,
                )
                child.setToolTip(0, version)
                for seed, seed_version in sorted(seed_versions):
                    leaf = self._item(child, f"seed_{seed}", "version",
                                      Source(source.group, source.experiment_id, source.agent, seed_version,
                                             f"{title} / seed_{seed}", version), _CHECKABLE)
                    leaf.setToolTip(0, seed_version)
                if child.data(0, _DATA) in expanded:
                    child.setExpanded(True)
        restored = next((s for s in self._restore_sources if s.run_key == manifest.run_key), selected)
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
        self._set_row_icon(item)
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
        if kind in ("group", "folder", "experiment"):
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
        while item and item.data(0, _KIND) not in ("run", "experiment", "folder", "group"):
            item = item.parent()
        if item and item.data(0, _KIND) == "run":
            return tuple(item.data(0, _DATA).split("/", 1))
        return None

    def _request_selected(self, signal):
        run = self._run_for_item(self.tree.currentItem())
        if run:
            signal.emit(*run)

    def _figure_target(self) -> str:
        item = self.tree.currentItem()
        if item is not None and item.data(0, _KIND) == "experiment":
            return "/".join(item.data(0, _DATA))
        run = self._run_for_item(item)
        return "/".join(run) if run else ""

    def _request_add_figure(self):
        target = self._figure_target()
        if target:
            self.add_figure_requested.emit(target)

    def deletion_targets(self, items=None):
        """Expand folders and experiments into their runs, and remove overlapping group/run targets."""
        if items is None:
            items = self.tree.selectedItems()
        groups = {item.data(0, _DATA) for item in items if item.data(0, _KIND) == "group"}
        runs = set()
        for item in items:
            if item.data(0, _KIND) in ("folder", "experiment"):
                runs.update(self._run_for_item(child) for child in self._descendants(item)
                            if child.data(0, _KIND) == "run")
            else:
                run = self._run_for_item(item)
                if run:
                    runs.add(run)
        return ([(group, None) for group in sorted(groups)]
                + [run for run in sorted(runs) if run[0] not in groups])

    def _prompt_delete_targets(self, targets):
        self.delete_selection_requested.emit(targets)

    def _add_context_actions(self, menu, item, items):
        if len(items) == 1:
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
        if item.data(0, _KIND) == "agent":
            source = item.data(0, _DATA)
            self.manifest_wanted.emit(source.group, source.experiment_id)

    def _emit(self):
        if self._emitting:
            self.btn_add_figure.setEnabled(bool(self._figure_target()))
            self.selection_changed.emit(self.checked_sources())

    def apply_filter(self, text):
        """Show experiments whose group/path contains `text`, and the folders and groups that hold them."""
        needle = text.strip().lower()

        def show(item, matched: bool) -> bool:
            kind, data = item.data(0, _KIND), item.data(0, _DATA)
            if kind == "run":
                visible = matched or needle in data.lower()
            else:
                matched = matched or (kind != "group" and needle in "/".join(data).lower())
                visible = False
                for i in range(item.childCount()):
                    visible |= show(item.child(i), matched)
            item.setHidden(not visible)
            return visible

        for g in range(self.tree.topLevelItemCount()):
            show(self.tree.topLevelItem(g), not needle)
        self._update_delete_action()

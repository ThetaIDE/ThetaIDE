"""Results panel: tick runs on the left; see every metric, value and file they produced on the right."""

from __future__ import annotations

import html
from pathlib import PurePosixPath

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QMessageBox,
    QSizePolicy,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..widgets import ElidedLabel, label
from .artifacts import ArtifactBrowser
from .config_diff import run_config_path
from .data import Source, chart_points, summary_rows, thin
from .explorer import RunExplorer
from .inspector import PointInspector, format_value
from .loader import ResultsLoader
from .metrics_catalog import AXIS_COLUMNS
from .plot_grid import PlotGrid, chart_title
from .summary import SummaryView
from src.app.pipeline.result_paths import result_experiment_id

PALETTE = (
    "#b8bb26",
    "#83a598",
    "#fabd2f",
    "#d3869b",
    "#8ec07c",
    "#fe8019",
    "#fb4934",
    "#689d6a",
    "#458588",
    "#d79921",
    "#b16286",
    "#98971a",
)
# Artifact tabs: the kinds each lists, and what it says when the ticked runs have none.
ARTIFACT_TABS = {
    "Files": (None, "Select runs to list every file they produced."),
    "Config": (("config",), "The selected runs have no saved experiment YAML or resolved config."),
}
TIME_COLUMNS = ("training_time_seconds", "time/total")  # a run's training time, for the Overview cards


def duration(seconds: float) -> str:
    if seconds >= 3600:
        return f"{int(seconds // 3600)}h {int(seconds % 3600 // 60):02d}m"
    if seconds >= 60:
        return f"{int(seconds // 60)}m {int(seconds % 60):02d}s"
    return f"{seconds:.1f}s"


class ResultsPanel(QWidget):
    """Everything the ticked runs produced.

    Monitors: one chart per metric, against the first step column the runs logged; running experiments
    expose generated files while live monitoring stays in Monitor.
    Diagnostics: saved rich monitor snapshots, shown only for runs that produced them.
    Config and Files: artifacts in the runs' manifests, previewed by kind. The run tree's sync button
    pulls a run that ran on a site other than local.

    `backend` needs get(path, callback); `settings` (optional) is the app's settings manager, which
    keeps the run tree's header layout.
    `load_config(run_key)` (optional) puts a run's saved builder config into the experiment builder and
    returns None, or returns why it could not.
    """

    def __init__(self, backend, settings=None, load_config=None, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.load_config = load_config
        self.curves: dict[str, list] = {}  # metric -> [(label, chart points, color)], as drawn
        self.loader = ResultsLoader(backend, self)
        self.sources: list[Source] = []
        self.axis: str | None = None  # the step column every chart is drawn against

        self.explorer = RunExplorer(settings_manager=settings)
        self.explorer.selection_changed.connect(self._selection_changed)
        self.explorer.manifest_wanted.connect(self.loader.load_manifest)
        self.explorer.refresh_requested.connect(self.refresh)
        self.explorer.archive_requested.connect(self._archive_run)
        self.explorer.delete_requested.connect(self._delete_run)
        self.explorer.delete_selection_requested.connect(self._delete_selection)
        self.explorer.archive_group_requested.connect(self._archive_group)
        self.explorer.delete_group_requested.connect(self._delete_group)
        self.explorer.rename_requested.connect(self._rename_results)
        from ..site_sync import RunSiteSyncControl
        self.site_sync = RunSiteSyncControl(self.explorer.btn_sync, self)
        self.site_sync.synced.connect(lambda _: self.refresh())
        self.remotes: dict[str, dict] = {}  # run key -> its site execution record
        self.tabs = QTabWidget()
        # Hidden pages participate in QTabWidget's minimumSizeHint: a wide page could otherwise
        # consume the entire splitter width and leave the explorer handle with no legal travel.
        self.tabs.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        self.tabs.setMinimumWidth(360)
        self.tabs.setDocumentMode(True)
        self.tab_corner = QWidget()  # buttons a layout puts beside the tabs (the Control rail's Export)
        QHBoxLayout(self.tab_corner).setContentsMargins(0, 0, 4, 0)
        self.tabs.setCornerWidget(self.tab_corner)
        self.curves_page = QWidget()
        QVBoxLayout(self.curves_page).setContentsMargins(8, 6, 0, 0)
        self.curves_body: QWidget | None = None
        self.legend = label("", "muted")
        self.legend.setWordWrap(True)
        self.legend.setTextFormat(Qt.TextFormat.RichText)
        self.grid = PlotGrid(show_actions=False)
        self.grid.point_hovered.connect(self._point_hovered)
        self.inspector = PointInspector()
        self._parking = QWidget(self)  # holds the widgets the current layout leaves out; never shown
        self._parking.hide()
        self.tabs.addTab(self.curves_page, "Monitors")
        self.diagnostics_page = QWidget()
        QVBoxLayout(self.diagnostics_page).setContentsMargins(0, 0, 0, 0)
        self.diagnostics_view = None
        self.tabs.addTab(self.diagnostics_page, "Diagnostics")
        self.tabs.setTabVisible(self.tabs.indexOf(self.diagnostics_page), False)
        self.summary = SummaryView()
        self.summary.changed.connect(self.update_summary)
        self.summary.setParent(self._parking)
        self.sweeps_page = QWidget()
        QVBoxLayout(self.sweeps_page).setContentsMargins(0, 0, 0, 0)
        self.sweeps_view = None
        self.tabs.addTab(self.sweeps_page, "Sweeps")
        self.tabs.setTabVisible(self.tabs.indexOf(self.sweeps_page), False)
        # Artifact tabs start empty and build their browser when first opened: each one is hundreds of
        # widgets, and the app restyles every widget when its theme is applied.
        self.browsers: dict[str, ArtifactBrowser] = {}
        self._pages: dict[str, QWidget] = {}
        for name in ARTIFACT_TABS:
            page = QWidget()
            QVBoxLayout(page).setContentsMargins(0, 0, 0, 0)
            self._pages[name] = page
            if name == "Files":
                self.tabs.insertTab(0, page, name)
                self.tabs.setCurrentWidget(page)
            else:
                self.tabs.addTab(page, name)
        self.tabs.currentChanged.connect(self._tab_opened)
        # The selected run's name above the tabs, as Monitor titles its run; the full paths are on the tooltip.
        self.run_title = ElidedLabel("")
        self.run_title.setObjectName("heading")
        self.run_title.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.run_title.hide()

        self.root = QVBoxLayout(self)
        self.root.setContentsMargins(18, 14, 18, 12)
        self.root.setSpacing(10)
        self.frame: QWidget | None = None
        self.status = label("", "muted")
        self.status.setParent(self._parking)
        self._compose()

        self.loader.runs_listed.connect(self._runs_listed)
        self.loader.manifest_loaded.connect(self.explorer.add_manifest)
        self.loader.manifest_loaded.connect(lambda _manifest: self.update_artifacts())
        self.loader.table_loaded.connect(lambda _source: self.redraw())
        self.loader.failed.connect(lambda message: self.status.setText(f"Could not load: {message}"))

        self.unfinished: set[str] = set()
        self.run_count = 0
        self._listing: tuple = ()
        self._pending_open_run = None
        self._pending_log = None
        self.browser("Files")
        self.redraw()
        self.refresh_timer = QTimer(self)
        self.refresh_timer.setInterval(4000)
        self.refresh_timer.timeout.connect(self._poll_generated_results)
        self.refresh_timer.start()

    # ── Layouts ──────────────────────────────────────────────────────────────

    def _compose(self):
        """Arrange the run explorer and grouped metric charts."""
        self.frame = QWidget()
        frame = QVBoxLayout(self.frame)
        frame.setContentsMargins(0, 0, 0, 0)
        self.curves_body = QWidget()
        body = QVBoxLayout(self.curves_body)
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(8)
        self.curves_page.layout().addWidget(self.curves_body)
        self.root.insertWidget(0, self.frame, 1)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.setHandleWidth(8)
        split.setChildrenCollapsible(False)
        split.addWidget(self.explorer)
        results = QWidget()
        results.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        results_layout = QVBoxLayout(results)
        results_layout.setContentsMargins(0, 0, 0, 0)
        results_layout.setSpacing(6)
        results_layout.addWidget(self.run_title)
        results_layout.addWidget(self.tabs, 1)
        split.addWidget(results)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([260, 900])
        frame.addWidget(split, 1)
        body.addWidget(self.legend)
        body.addWidget(self.grid, 1)

    # ── Data flow ────────────────────────────────────────────────────────────

    def refresh(self):
        """Re-list runs and drop cached data, so new and still-training runs are re-read."""
        if self.diagnostics_view is not None:
            self.diagnostics_view.reset()
        self.loader.clear()
        self.status.setText("Listing runs…")
        self.loader.list_runs()
        # The tree keeps its ticks when the run list is unchanged, so nothing else re-reads what is
        # ticked: without this the next redraw (e.g. changing Versions) would find no data at all.
        for source in self.sources:
            if source.agent and source.run_key not in self.unfinished:
                self.loader.load_table(source)
            self.loader.load_manifest(source.group, source.experiment_id)

    def _poll_generated_results(self):
        if not self.isVisible():
            return
        self.loader.list_runs()
        for key in self.run_keys():
            if key in self.unfinished:
                self.loader.manifests.pop(key, None)
                self.loader.load_manifest(*key.split("/", 1))

    def _rename_results(self, kind, group, experiment):
        old_name = group if kind == "group" else PurePosixPath(experiment).name
        new_name, accepted = QInputDialog.getText(self, f"Rename {kind}", "New name:", text=old_name)
        if not accepted or new_name == old_name:
            return
        selected = self.run_keys()

        def finished(data, error):
            if error:
                QMessageBox.warning(self, "Rename failed", str(error))
                return
            renamed = data["runs"]
            target = next((renamed[key] for key in selected if key in renamed), None)
            if target:
                self._pending_open_run = tuple(target.split("/", 1))
            self.sources = []
            self.refresh()

        self.loader.backend.rename_results(kind, group, experiment, new_name, finished)

    def _archive_run(self, group, experiment):
        path, _ = QFileDialog.getSaveFileName(self, "Archive run", f"{group}-{experiment}.zip", "ZIP archives (*.zip)")
        if not path:
            return
        if not path.lower().endswith(".zip"):
            path += ".zip"
        self.status.setText(f"Archiving {group}/{experiment}…")
        self.loader.backend.archive_run(group, experiment, path,
                                        lambda _, error: self._run_action_done("Archive", error, path))

    def _delete_selection(self, targets):
        if not targets:
            return
        names = [f"{group}/{run}" if run else f"{group} (all runs)" for group, run in targets]
        live = any((f"{group}/{run}" in self.unfinished if run else
                    any(key.startswith(f"{group}/") for key in self.unfinished))
                   for group, run in targets)
        warning = "\nSome selected runs may still be training." if live else ""
        answer = QMessageBox.question(
            self, "Delete selected folders",
            "Permanently delete these results and their files across results folders?\n\n"
            + "\n".join(names) + warning,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self.status.setText("Deleting selected results…")
        pending = len(targets)
        errors = []

        def done(_, error):
            nonlocal pending
            if error:
                errors.append(str(error))
            pending -= 1
            if pending == 0:
                self._run_action_done("Delete", "\n".join(errors) or None)

        for group, run in targets:
            if run is None:
                self.loader.backend.delete_group(group, done)
            else:
                self.loader.backend.delete_run(group, run, done)

    def _delete_run(self, group, experiment):
        live = f"{group}/{experiment}" in self.unfinished
        warning = " This run may still be training." if live else ""
        answer = QMessageBox.question(
            self, "Delete run", f"Permanently delete {group}/{experiment} and its files across results folders?{warning}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self.status.setText(f"Deleting {group}/{experiment}…")
        self.loader.backend.delete_run(group, experiment,
                                       lambda _, error: self._run_action_done("Delete", error))

    def _archive_group(self, group):
        path, _ = QFileDialog.getSaveFileName(self, "Archive group", f"{group}.zip", "ZIP archives (*.zip)")
        if not path:
            return
        if not path.lower().endswith(".zip"):
            path += ".zip"
        self.status.setText(f"Archiving group {group}…")
        self.loader.backend.archive_group(group, path,
                                          lambda _, error: self._run_action_done("Archive", error, path))

    def _delete_group(self, group):
        live = any(key.startswith(f"{group}/") for key in self.unfinished)
        warning = " Some runs in this group may still be training." if live else ""
        answer = QMessageBox.question(
            self, "Delete group",
            f"Permanently delete group {group} and all its runs and files across results folders?{warning}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self.status.setText(f"Deleting group {group}…")
        self.loader.backend.delete_group(group, lambda _, error: self._run_action_done("Delete", error))

    def _run_action_done(self, action, error, path=None):
        if error:
            QMessageBox.warning(self, f"{action} failed", str(error))
            self.refresh()
        elif action == "Delete":
            self.loader.clear()
            self.loader.list_runs()
        else:
            self.status.setText(f"Archived to {path}")

    def set_offline(self):
        self.status.setText(
            "Training pipeline unavailable: results can't be read. Run ThetaIDE from the project venv "
            '(pip install -e ".[gui]" then python -m frontend).'
        )

    def open_run(self, group, experiment, failure=False, job_id=None):
        """Show a run on the Curves tab once its catalog listing arrives."""
        self._pending_open_run = (group, experiment)
        self._pending_log = (f"{group}/{experiment}", job_id) if failure else None
        self.tabs.setCurrentWidget(self._pages["Files"] if failure else self.curves_page)
        self.refresh()

    def _runs_listed(self, entries):
        listing = tuple(
            sorted(
                (e["group"], e["experiment_id"], tuple(a.get("name") for a in e.get("agents") or []),
                 bool(e.get("has_run_logs")), bool(e.get("incomplete_results")),
                 tuple((name, tuple(versions)) for name, versions in sorted(e.get("completed_versions", {}).items())))
                + (bool(e.get("finished")),)
                for e in entries
            )
        )
        old_unfinished = self.unfinished
        self.unfinished = {f"{e['group']}/{e['experiment_id']}" for e in entries if not e.get("finished")}
        for source in self.sources:
            if source.run_key in old_unfinished - self.unfinished:
                self.loader.load_table(source)
                self.loader.manifests.pop(source.run_key, None)
                self.loader.load_manifest(source.group, source.experiment_id)
        self.remotes = {f"{e['group']}/{e['experiment_id']}": e["remote"] for e in entries if e.get("remote")}
        self._update_site_sync()
        if listing != self._listing:  # rebuild only when run identity or completion changes
            self._listing = listing
            self.explorer.set_runs(entries)
        if self._pending_open_run and self.explorer.select_run(*self._pending_open_run):
            self._pending_open_run = None
        self.run_count = sum(1 for e in entries if e.get("agents"))

    def _selection_changed(self, sources):
        self.sources = sources
        self._update_title()
        self._update_site_sync()
        for source in sources:
            if source.agent and source.run_key not in self.unfinished:
                self.loader.load_table(source)
            self.loader.load_manifest(source.group, source.experiment_id)
        self.redraw()
        self.update_artifacts()

    def _update_title(self):
        """Title the selected run "group / experiment/run", as Monitor does; several runs share one line."""
        keys = self.run_keys()
        if not keys:
            self.run_title.hide()
            return
        runs = [tuple(key.split("/", 1)) for key in keys]
        if len(runs) == 1:
            text = " / ".join(runs[0])
        else:
            experiments = {(group, result_experiment_id(run)) for group, run in runs}
            where = " / ".join(next(iter(experiments))) + " · " if len(experiments) == 1 else ""
            text = f"{where}{len(runs)} runs"
        from .explorer import _results_label
        self.run_title.setText(text)
        self.run_title.setToolTip("\n".join(f"{_results_label()}/logs/{key}" for key in keys))
        self.run_title.show()

    def _update_site_sync(self):
        """Offer the tree's sync button for one selected run that ran on a site other than local."""
        keys = self.run_keys()
        self.site_sync.set_remote(self.remotes.get(keys[0]) if len(keys) == 1 else None)

    def run_keys(self) -> list[str]:
        """Selected runs, each once (several agents or versions can share a run)."""
        return list(dict.fromkeys(source.run_key for source in self.sources))

    # ── Artifacts ────────────────────────────────────────────────────────────

    def browser(self, name: str) -> ArtifactBrowser:
        """The artifact browser of one tab, built on first use."""
        if name not in self.browsers:
            kinds, empty = ARTIFACT_TABS[name]
            browser = self.browsers[name] = ArtifactBrowser(self.loader, kinds, empty)
            if name == "Config":
                browser.tree.hide()
                browser.title.hide()
                browser.detail.hide()
                browser.save_button.hide()
            layout = self._pages[name].layout()
            layout.addWidget(browser)
            self.update_artifacts()
        return self.browsers[name]

    def _tab_opened(self, index):
        name = self.tabs.tabText(index)
        if name == "Diagnostics" and self.diagnostics_view is None:
            from .diagnostics import SavedDiagnostics
            self.diagnostics_view = SavedDiagnostics(self.loader)
            self.diagnostics_page.layout().addWidget(self.diagnostics_view)
        if self.diagnostics_view is not None:
            self.diagnostics_view.set_sources(self.sources, self.diagnostic_manifests(), active=name == "Diagnostics")
        if name == "Sweeps":
            if self.sweeps_view is None:
                from .sweeps import SweepResultsView
                self.sweeps_view = SweepResultsView(self.loader, self._open_sweep_metrics)
                self.sweeps_page.layout().addWidget(self.sweeps_view)
            self.sweeps_view.set_manifests(self.manifests())
        if name in ARTIFACT_TABS:
            self.browser(name)

    def manifests(self):
        """Loaded manifests of the ticked runs."""
        return [self.loader.manifests[k] for k in self.run_keys() if k in self.loader.manifests]

    def diagnostic_manifests(self):
        """Loaded manifests of the ticked, finished runs that saved rich monitor snapshots."""
        return [m for m in self.manifests() if m.run_key not in self.unfinished
                and any(a.get("kind") == "monitor_snapshot" or "/monitors/" in a.get("path", "")
                        for a in m.artifacts)]

    def update_artifacts(self):
        manifests = self.manifests()
        diagnostic_runs = self.diagnostic_manifests()
        self.tabs.setTabVisible(self.tabs.indexOf(self.diagnostics_page), bool(diagnostic_runs))
        has_sweeps = any(a.get("path", "").endswith("/tuning_summary.json") for m in manifests for a in m.artifacts)
        self.tabs.setTabVisible(self.tabs.indexOf(self.sweeps_page), has_sweeps)
        if self.diagnostics_view is not None:
            self.diagnostics_view.set_sources(self.sources, diagnostic_runs,
                active=self.tabs.currentWidget() is self.diagnostics_page)
        if self.sweeps_view is not None:
            self.sweeps_view.set_manifests(manifests)
        for name, browser in self.browsers.items():
            if name == "Config":
                from .data import Manifest
                configs = []
                for m in manifests:
                    prefix = f"logs/{m.run_key}/experiment"
                    recipes = [a for a in m.artifacts if a["path"] == f"{prefix}.yaml" or
                               (a["path"].startswith(f"{prefix}_") and a["path"].endswith(".yaml"))]
                    # Older runs only have the resolved config saved by training.
                    files = recipes or [a for a in m.artifacts if a["path"] == run_config_path(m.run_key)]
                    configs.append(Manifest(m.group, m.experiment_id, m.agents, files))
                browser.set_manifests(configs)
                if browser.current is None and (files := browser.artifacts()):
                    browser.select(files[0][0], files[0][1]["path"])
            else:
                browser.set_manifests(manifests)
                if name == "Files" and self._pending_log:
                    key, job_id = self._pending_log
                    def log_priority(artifact):
                        path = artifact['path']
                        name = PurePosixPath(path).name
                        if name == 'pipeline.err': priority = 0
                        elif job_id and name == f'{job_id}.stderr.log': priority = 1
                        elif name == 'scheduler.err.preview.log' or path.endswith('.err'): priority = 2
                        elif name == 'pipeline.log': priority = 3
                        elif job_id and name == f'{job_id}.stdout.log': priority = 4
                        elif name == 'scheduler.out.preview.log': priority = 5
                        else: return None
                        return (not bool(artifact.get('size')), priority, path)
                    candidates = [a for run, a in browser.artifacts()
                                  if run == key and log_priority(a) is not None]
                    candidates.sort(key=log_priority)
                    if candidates and browser.select(key, candidates[0]["path"]):
                        self._pending_log = None

    def _open_sweep_metrics(self, run_key, path):
        from pathlib import PurePosixPath
        parts = PurePosixPath(path.removeprefix(f"logs/{run_key}/")).parts
        manifest = self.loader.manifests.get(run_key)
        if manifest and len(parts) == 3:
            agent, version, _ = parts
            details = manifest.version_details.get(agent, {})
            if any(info.get("trial_version") == version for info in details.values()):
                if self.explorer.select_version(run_key, agent, version):
                    self.tabs.setCurrentWidget(self.curves_page)
                    return
        browser = self.browser("Files")
        if browser.select(run_key, path):
            self.tabs.setCurrentWidget(self._pages["Files"])
        else:
            self.status.setText("The selected trial's metrics file is not available in this run.")

    # ── Point inspector ──────────────────────────────────────────────────────

    def _point_hovered(self, x):
        if self.axis:
            self.inspector.show_point(self.axis, x, self.loaded_tables(), None)

    # ── Drawing ──────────────────────────────────────────────────────────────

    def loaded_tables(self):
        return [(s, self.loader.tables[s]) for s in self.sources if s in self.loader.tables and s.run_key not in self.unfinished]

    def all_metrics(self) -> list[str]:
        """Every metric any ticked run logged."""
        seen: dict[str, None] = {}
        for _, table in self.loaded_tables():
            for metric in table.metrics():
                seen.setdefault(metric)
        return list(seen)

    def curve_series(self) -> dict[str, list]:
        """Per metric, every plotted source: (label, chart points, color)."""
        tables = dict(self.loaded_tables())
        sources = [s for s in self.sources if s in tables]
        colors = {s: PALETTE[i % len(PALETTE)] for i, s in enumerate(sources)}
        return {metric: [(s.label, chart_points(points, metric), colors[s]) for s in sources
                         if (points := tables[s].series(metric, self.axis))]
                for metric in self.all_metrics()}

    def redraw(self):
        tables = self.loaded_tables()
        self.axis = next((axis for axis in AXIS_COLUMNS if any(axis in t.columns for _, t in tables)), None)
        metrics = self.all_metrics()
        if not self.sources:
            empty = "Tick one or more runs on the left to plot every metric they logged."
        elif all(s.run_key in self.unfinished for s in self.sources):
            empty = "In progress — see Monitor for live monitoring. Generated files are available in Files."
        elif not tables:
            empty = "Loading metrics…"
        else:
            empty = "The ticked runs have no numeric metrics."
        self.grid.configure(collapsible=True)
        self.grid.set_metrics(metrics, empty)
        self.curves = self.curve_series()
        for metric, chart in self.grid.charts().items():
            chart.set_metric(metric, chart_title(metric))
            # Draw thinned copies; inspector and Summary use the full data.
            chart.set_series([(name, thin(points, metric), color) for name, points, color in self.curves.get(metric, [])])
        drawn = list(dict.fromkeys((name, color) for series in self.curves.values() for name, _, color in series))
        self.legend.setText("   ".join(f'<span style="color:{color}">■</span> {html.escape(name)}' for name, color in drawn))
        for metric, card in self.grid.cards.items():
            card.set_value(*self.latest(metric))
        self.update_summary()

    def latest(self, metric: str) -> tuple[str, str | None]:
        """(value, color) of the last drawn point of the first series."""
        series = self.curves.get(metric, [])
        if not series or not series[0][1]:
            return "", None
        _name, points, color = series[0]
        return format_value(metric, points[-1][metric]), color

    # ── Summary ──────────────────────────────────────────────────────────────

    def summary_entries(self) -> list[tuple[str, str, list]]:
        """(source label, metric, raw points) as plotted."""
        tables = dict(self.loaded_tables())
        chosen = self.summary.selected_metric()
        return [(s.label, metric, points) for s in self.sources if s in tables
                for metric in ([chosen] if chosen else self.all_metrics())
                if (points := tables[s].series(metric, self.axis))]

    def update_summary(self):
        self.summary.set_metrics(self.all_metrics())
        entries = self.summary_entries()
        rows = summary_rows(entries, self.summary.threshold_value())
        chosen = self.summary.selected_metric()
        absent = sorted(set([chosen] if chosen else self.all_metrics()) - {metric for _, metric, _ in entries})
        # e.g. a value logged once on a row without a step count: say so rather than drop it silently
        note = f"no points on this x-axis for {', '.join(absent)} (see Files)" if absent and self.sources else ""
        self.summary.show_rows(rows, self.axis, note)

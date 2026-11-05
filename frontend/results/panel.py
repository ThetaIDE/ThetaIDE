"""Results panel: tick runs on the left; see every metric, value and file they produced on the right."""

from __future__ import annotations

import html

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QMenu,
    QMessageBox,
    QPushButton,
    QSlider,
    QSplitter,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..widgets import label
from .artifacts import ArtifactBrowser
from .config_diff import run_config_path
from .data import (
    COMBINE_MODES,
    SPREAD,
    Source,
    aggregate,
    chart_points,
    combined_points,
    series_groups,
    smooth,
    summary_rows,
    thin,
    to_csv,
)
from .explorer import RunExplorer
from .executions import ExecutionsView
from .inspector import PointInspector, format_value
from .loader import ResultsLoader
from .metrics_catalog import AXIS_COLUMNS, describe
from .plot_grid import PlotGrid, chart_title
from .summary import SummaryView

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
    "Reports": (("report",), "The ticked runs have no reports yet."),
    "Config": (("config",), "The selected runs have no saved experiment YAML or resolved config."),
}
AXIS_LABELS = {
    "transitions": "Environment steps",
    "step": "Logged step",
    "epoch": "Epoch",
    "time/total": "Wall time (s)",
    "training_time_seconds": "Training time (s)",
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

    Curves: one chart per metric;
    an agent's versions (repeated runs) can be combined into a mean ± std or ± SEM band.
    Summary: final, best, range, mean and area under the curve per series and metric, plus when a
    threshold was first reached. Export writes the visible curves or the summary as CSV, or the charts as PNG.
    Reports, Config and Files: artifacts in the runs'
    manifests, previewed by kind.

    `backend` needs get(path, callback); `settings` (optional) needs get(*keys, default=) and
    set(*keys, value), and keeps hidden plots across sessions under [results] hidden_plots.
    `load_config(run_key)` (optional) puts a run's saved builder config into the experiment builder and
    returns None, or returns why it could not.
    """

    def __init__(self, backend, settings=None, load_config=None, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.load_config = load_config
        self.curves: dict[str, list] = {}  # metric -> [(label, chart points, color, combined)], as drawn
        self.loader = ResultsLoader(backend, self)
        self.sources: list[Source] = []
        self.hidden: set[str] = set(self._setting("hidden_plots", []))
        self.smoothing = 0.0

        # Controls, run tree, tabs, charts and inspector.
        self.plots_button = QToolButton()
        self.plots_button.setText("Plots ▾")
        self.plots_button.setToolTip("Show or hide individual plots")
        self.plots_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.plots_menu = QMenu(self.plots_button)
        self.plots_menu.aboutToShow.connect(self._fill_plots_menu)
        self.plots_button.setMenu(self.plots_menu)
        self.axis_select = QComboBox()
        self.axis_select.setToolTip("What the horizontal axis of every chart measures")
        self.axis_select.currentIndexChanged.connect(lambda _: self.redraw())
        self.smoothing_label = label("Smoothing: 0%", "muted")
        self.smoothing_slider = QSlider(Qt.Orientation.Horizontal)
        self.smoothing_slider.setRange(0, 95)
        self.smoothing_slider.setFixedWidth(110)
        self.smoothing_slider.setToolTip("Exponential moving average smoothing (0% = raw data)")
        self.smoothing_slider.valueChanged.connect(self._smoothing_changed)
        self.combine_select = QComboBox()
        for mode, text in COMBINE_MODES.items():
            self.combine_select.addItem(text, mode)
        self.combine_select.setToolTip(
            "Combine the ticked versions (repeated runs, e.g. seeds) of each agent into one mean curve with a band"
        )
        self.combine_select.currentIndexChanged.connect(lambda _: self.redraw())
        self.primary_select = QComboBox()
        self.primary_select.addItem("Nearest", None)
        self.primary_select.setToolTip(
            "The series whose logged steps hovering snaps to (drawn on top and thicker). "
            "Nearest snaps to whichever series is closest to the mouse."
        )
        self.primary_select.currentIndexChanged.connect(lambda _: self.redraw())
        self.export_button = QToolButton()
        self.export_button.setText("Export ▾")
        self.export_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        export_menu = QMenu(self.export_button)
        export_menu.addAction("Visible curves as CSV…", lambda: self._export("curves"))
        export_menu.addAction("Summary as CSV…", lambda: self._export("summary"))
        export_menu.addAction("Charts as PNG…", lambda: self._export("png"))
        self.export_button.setMenu(export_menu)
        self.refresh_button = QPushButton("↻  Refresh")
        self.refresh_button.setToolTip("Re-read results/ through the backend")
        self.refresh_button.clicked.connect(self.refresh)

        self.explorer = RunExplorer(settings_manager=settings)
        self.explorer.selection_changed.connect(self._selection_changed)
        self.explorer.manifest_wanted.connect(self.loader.load_manifest)
        self.explorer.refresh_requested.connect(self.refresh)
        self.explorer.archive_requested.connect(self._archive_run)
        self.explorer.delete_requested.connect(self._delete_run)
        self.explorer.archive_group_requested.connect(self._archive_group)
        self.explorer.delete_group_requested.connect(self._delete_group)
        self.tabs = QTabWidget()
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
        self.grid = PlotGrid()
        self.grid.hide_requested.connect(lambda metric: self.set_hidden(metric, True))
        self.grid.point_hovered.connect(self._point_hovered)
        self.inspector = PointInspector()
        self._parking = QWidget(self)  # holds the widgets the current layout leaves out; never shown
        self._parking.hide()
        self.tabs.addTab(self.curves_page, "Curves")
        self.executions_view = ExecutionsView(backend, self.open_run)
        self.tabs.addTab(self.executions_view, "Executions")
        self.summary = SummaryView()
        self.summary.changed.connect(self.update_summary)
        self.tabs.addTab(self.summary, "Summary")
        self.sweeps_page = QWidget()
        QVBoxLayout(self.sweeps_page).setContentsMargins(0, 0, 0, 0)
        self.sweeps_view = None
        self.tabs.addTab(self.sweeps_page, "Sweeps")
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

    # ── Settings ─────────────────────────────────────────────────────────────

    def _setting(self, key, default):
        if self.settings is None:
            return default
        value = self.settings.get("results", key, default=default)
        return value if isinstance(value, type(default)) else default

    def set_hidden(self, metric: str, hidden: bool):
        (self.hidden.add if hidden else self.hidden.discard)(metric)
        if self.settings is not None:
            self.settings.set("results", "hidden_plots", sorted(self.hidden))
        self.redraw()

    def show_all(self):
        self.hidden.clear()
        if self.settings is not None:
            self.settings.set("results", "hidden_plots", [])
        self.redraw()

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
        split.setChildrenCollapsible(False)
        split.addWidget(self.explorer)
        split.addWidget(self.tabs)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([260, 900])
        frame.addWidget(split, 1)
        controls = QHBoxLayout()
        for widget in (self.plots_button, self.axis_select, self.smoothing_label, self.smoothing_slider,
                       self.combine_select, self.primary_select, self.export_button, self.refresh_button):
            controls.addWidget(widget)
        controls.addStretch()
        body.addLayout(controls)
        body.addWidget(self.legend)
        body.addWidget(self.grid, 1)

    def set_primary(self, name: str):
        """Make series `name` primary, or go back to Nearest when it already is."""
        index = self.primary_select.findData(name)
        self.primary_select.setCurrentIndex(0 if name == self.primary_series() or index < 0 else index)

    # ── Data flow ────────────────────────────────────────────────────────────

    def refresh(self):
        """Re-list runs and drop cached data, so new and still-training runs are re-read."""
        self.loader.clear()
        self.status.setText("Listing runs…")
        self.loader.list_runs()
        # The tree keeps its ticks when the run list is unchanged, so nothing else re-reads what is
        # ticked: without this the next redraw (e.g. changing Versions) would find no data at all.
        for source in self.sources:
            if source.agent:
                self.loader.load_table(source)
            self.loader.load_manifest(source.group, source.experiment_id)

    def _archive_run(self, group, experiment):
        path, _ = QFileDialog.getSaveFileName(self, "Archive run", f"{group}-{experiment}.zip", "ZIP archives (*.zip)")
        if not path:
            return
        if not path.lower().endswith(".zip"):
            path += ".zip"
        self.status.setText(f"Archiving {group}/{experiment}…")
        self.loader.backend.archive_run(group, experiment, path,
                                        lambda _, error: self._run_action_done("Archive", error, path))

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
                 bool(e.get("has_run_logs")),
                 tuple((name, tuple(versions)) for name, versions in sorted(e.get("completed_versions", {}).items())))
                for e in entries if e.get("finished")
            )
        )
        if listing != self._listing:  # rebuild the tree only when runs or agents changed, not on every re-list
            self._listing = listing
            self.explorer.set_runs([e for e in entries if e.get("finished")])
        if self._pending_open_run and self.explorer.select_run(*self._pending_open_run):
            self._pending_open_run = None
        self.unfinished = {f"{e['group']}/{e['experiment_id']}" for e in entries if e.get("in_progress")}
        self.run_count = sum(1 for e in entries if e.get("agents"))

    def _selection_changed(self, sources):
        self.sources = sources
        for source in sources:
            if source.agent:
                self.loader.load_table(source)
            self.loader.load_manifest(source.group, source.experiment_id)
        self.redraw()
        self.update_artifacts()

    def run_keys(self) -> list[str]:
        """Ticked runs, in tree order, each once (several agents or versions share a run)."""
        return list(dict.fromkeys(source.run_key for source in self.sources))

    # ── Artifacts ────────────────────────────────────────────────────────────

    def browser(self, name: str) -> ArtifactBrowser:
        """The artifact browser of one tab, built on first use."""
        if name not in self.browsers:
            kinds, empty = ARTIFACT_TABS[name]
            browser = self.browsers[name] = ArtifactBrowser(self.loader, kinds, empty)
            layout = self._pages[name].layout()
            layout.addWidget(browser)
            self.update_artifacts()
        return self.browsers[name]

    def _tab_opened(self, index):
        name = self.tabs.tabText(index)
        if name == "Sweeps":
            if self.sweeps_view is None:
                from .sweeps import SweepResultsView
                self.sweeps_view = SweepResultsView(self.loader, self._open_sweep_metrics)
                self.sweeps_page.layout().addWidget(self.sweeps_view)
            self.sweeps_view.set_manifests(self.manifests())
        if name == "Executions":
            self.executions_view.refresh()
        if name in ARTIFACT_TABS:
            self.browser(name)

    def manifests(self):
        """Loaded manifests of the ticked runs."""
        return [self.loader.manifests[k] for k in self.run_keys() if k in self.loader.manifests]

    def update_artifacts(self):
        manifests = self.manifests()
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
                    candidates = [a for run, a in browser.artifacts() if run == key and job_id and
                                  a["path"].endswith((f"/{job_id}.stderr.log", f"/{job_id}.stdout.log"))]
                    candidates.sort(key=lambda a: (not bool(a.get("size")), not a["path"].endswith(".stderr.log")))
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
        axis = self.axis_select.currentData()
        if not axis:
            return
        tables = self.loaded_tables()
        primary = self.primary_series()
        if primary:  # the primary series' runs come first: the hovered x is one of their logged steps
            members = next((m for name, m in self.plotted_groups() if name == primary), [])
            tables = sorted(tables, key=lambda entry: entry[0] not in members)
        self.inspector.show_point(axis, x, tables, primary)

    def primary_series(self) -> str | None:
        """The legend label of the series hovering snaps to, or None to snap to the nearest series."""
        return self.primary_select.currentData()

    def _sync_primary(self, names: list[str]) -> str | None:
        """Offer every plotted series as the primary one, keeping the choice while it is still plotted."""
        current = self.primary_series()
        self.primary_select.blockSignals(True)
        self.primary_select.clear()
        self.primary_select.addItem("Nearest", None)
        for name in names:
            self.primary_select.addItem(name, name)
        self.primary_select.setCurrentIndex(names.index(current) + 1 if current in names else 0)
        self.primary_select.blockSignals(False)
        return self.primary_series()

    def _smoothing_changed(self, value):
        self.smoothing = value / 100.0
        self.smoothing_label.setText(f"Smoothing: {value}%")
        self.redraw()

    # ── Drawing ──────────────────────────────────────────────────────────────

    def loaded_tables(self):
        return [(s, self.loader.tables[s]) for s in self.sources if s in self.loader.tables]

    def all_metrics(self) -> list[str]:
        """Every metric any ticked run logged, including hidden ones."""
        seen: dict[str, None] = {}
        for _, table in self.loaded_tables():
            for metric in table.metrics():
                seen.setdefault(metric)
        return list(seen)

    def _sync_axes(self, tables):
        available = [axis for axis in AXIS_COLUMNS if any(axis in t.columns for _, t in tables)]
        current = self.axis_select.currentData()
        self.axis_select.blockSignals(True)
        self.axis_select.clear()
        for axis in available:
            self.axis_select.addItem(AXIS_LABELS.get(axis, axis), axis)
        if current in available:
            self.axis_select.setCurrentIndex(available.index(current))
        self.axis_select.blockSignals(False)
        return self.axis_select.currentData()

    def combine_mode(self) -> str | None:
        """None (separate curves), "std" or "sem"."""
        return self.combine_select.currentData()

    def plotted_groups(self) -> list[tuple[str, list[Source]]]:
        """Series as plotted: (legend label, loaded sources it covers), combining versions if asked."""
        tables = dict(self.loaded_tables())
        groups = series_groups(self.sources, self.combine_mode() is not None)
        return [(name, kept) for name, members in groups if (kept := [s for s in members if s in tables])]

    def curve_series(self, axis) -> dict[str, list]:
        """Per metric, every plotted series: (label, chart points, color, combined)."""
        tables = dict(self.loaded_tables())
        groups = self.plotted_groups()
        colors = {name: PALETTE[i % len(PALETTE)] for i, (name, _) in enumerate(groups)}
        curves: dict[str, list] = {}
        for metric in self.all_metrics():
            series = []
            for name, members in groups:
                parts = [p for s in members if (p := smooth(tables[s].series(metric, axis), self.smoothing))]
                if not parts:
                    continue
                if len(members) == 1:
                    series.append((name, chart_points(parts[0], metric), colors[name], False))
                else:
                    series.append((name, combined_points(parts, metric, self.combine_mode()), colors[name], True))
            curves[metric] = series
        return curves

    def redraw(self):
        tables = self.loaded_tables()
        axis = self._sync_axes(tables)
        metrics = self.all_metrics()
        visible = [m for m in metrics if m not in self.hidden]
        if not self.sources:
            empty = "Tick one or more runs on the left to plot every metric they logged."
        elif not tables:
            empty = "Loading metrics…"
        elif not metrics:
            empty = "The ticked runs have no numeric metrics."
        else:
            empty = f"All {len(metrics)} plots are hidden. Use Plots ▾ to show them."
        self.grid.configure(collapsible=True)
        self.grid.set_metrics(visible, empty)
        self.curves = self.curve_series(axis)
        groups = self.plotted_groups()
        primary = self._sync_primary([name for name, _ in groups])
        large = set()
        for metric, chart in self.grid.charts().items():
            series = self.curves.get(metric, [])
            chart.band = SPREAD if any(combined for *_, combined in series) else None
            chart.primary = primary
            chart.hover_all = metric in large  # large charts list every series' value at the hovered step
            chart.set_metric(metric, chart_title(metric))
            # Draw thinned copies; inspector, Summary and exports use the full data.
            chart.set_series([(name, thin(points, metric), color) for name, points, color, _ in series])
        self.legend.setText(
            "   ".join(
                f'<span style="color:{PALETTE[i % len(PALETTE)]}">■</span> {html.escape(name)}'
                for i, (name, _) in enumerate(groups)
            )
        )
        hidden_note = f" · {len(metrics) - len(visible)} hidden" if len(visible) < len(metrics) else ""
        self.plots_button.setText(f"Plots ({len(visible)}/{len(metrics)}{hidden_note}) ▾" if metrics else "Plots ▾")
        self._update_layout_widgets(axis, metrics, visible, groups, primary)
        self.update_summary()

    def latest(self, metric: str, primary: str | None) -> tuple[str, str | None]:
        """(value, color) of the last drawn point of the primary series, or of the first series."""
        series = self.curves.get(metric, [])
        chosen = next((entry for entry in series if entry[0] == primary), series[0] if series else None)
        if chosen is None or not chosen[1]:
            return "", None
        return format_value(metric, chosen[1][-1][metric]), chosen[2]

    def _update_layout_widgets(self, axis, metrics, visible, groups, primary):
        for metric, card in self.grid.cards.items():
            card.set_value(*self.latest(metric, primary))

    # ── Summary ──────────────────────────────────────────────────────────────

    def summary_entries(self, axis) -> list[tuple[str, str, list]]:
        """(series label, metric, raw points) as plotted, but unsmoothed: combined series use their mean."""
        tables = dict(self.loaded_tables())
        chosen = self.summary.selected_metric()
        entries = []
        for name, members in self.plotted_groups():
            for metric in [chosen] if chosen else self.all_metrics():
                parts = [p for s in members if (p := tables[s].series(metric, axis))]
                if not parts:
                    continue
                points = parts[0] if len(members) == 1 else [(a.x, a.mean) for a in aggregate(parts)]
                entries.append((name, metric, points))
        return entries

    def update_summary(self):
        axis = self.axis_select.currentData()
        self.summary.set_metrics(self.all_metrics())
        entries = self.summary_entries(axis)
        rows = summary_rows(entries, self.summary.threshold_value())
        notes = ["combined series are summarized by their mean curve"] if self.combine_mode() else []
        chosen = self.summary.selected_metric()
        absent = sorted(set([chosen] if chosen else self.all_metrics()) - {metric for _, metric, _ in entries})
        if absent and self.sources:
            # e.g. a value logged once on a row without a step count: say so rather than drop it silently
            notes.append(f"no points on this x-axis for {', '.join(absent)} (pick another x-axis, or see Files)")
        self.summary.show_rows(rows, axis, " · ".join(notes))

    # ── Export ───────────────────────────────────────────────────────────────

    def curves_csv(self) -> str:
        """Every point of every visible chart, one row per point, as currently drawn."""
        axis = self.axis_select.currentData() or "x"
        mode = self.combine_mode()
        rows = []
        for metric in self.all_metrics():
            if metric in self.hidden:
                continue
            for name, points, _color, combined in self.curves.get(metric, []):
                for p in points:
                    rows.append([name, metric, p["step"], p[metric]] + ([p.get(SPREAD), p.get("n")] if mode else []))
        headers = ["series", "metric", axis, "value"] + ([mode, "runs"] if mode else [])
        return to_csv(headers, rows)

    def summary_csv(self) -> str:
        return to_csv(self.summary.headers, self.summary.rows)

    def export_to(self, kind: str, path: str) -> bool:
        """Write curves or summary CSV, or the charts as PNG, to `path`."""
        try:
            if kind == "png":
                if not self.grid.body.grab().save(path):
                    raise OSError("could not write the image")
            else:
                text = self.curves_csv() if kind == "curves" else self.summary_csv()
                with open(path, "w", encoding="utf-8", newline="") as f:
                    f.write(text)
        except OSError as exc:
            self.status.setText(f"Could not export: {exc}")
            return False
        self.status.setText(f"Exported to {path}")
        return True

    def _export(self, kind: str):
        names = {"curves": "curves.csv", "summary": "summary.csv", "png": "charts.png"}
        filters = "PNG image (*.png)" if kind == "png" else "CSV (*.csv)"
        path, _ = QFileDialog.getSaveFileName(self, "Export", names[kind], filters)
        if path:
            self.export_to(kind, path)

    def _fill_plots_menu(self):
        self.plots_menu.clear()
        metrics = self.all_metrics()
        if not metrics:
            self.plots_menu.addAction("Tick runs to list their plots").setEnabled(False)
            return
        self.plots_menu.addAction("Show all plots", self.show_all)
        self.plots_menu.addSeparator()
        for metric in metrics:
            action = self.plots_menu.addAction(chart_title(metric))
            action.setCheckable(True)
            action.setChecked(metric not in self.hidden)
            action.toggled.connect(lambda on, m=metric: self.set_hidden(m, not on))

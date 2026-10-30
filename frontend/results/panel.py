"""Results panel: tick runs on the left; see every metric, value and file they produced on the right."""

from __future__ import annotations

import html

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGridLayout,
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
from .config_diff import ConfigDiff, run_config_path
from .data import (
    COMBINE_MODES,
    SPREAD,
    Source,
    aggregate,
    chart_points,
    combined_points,
    first_reaching,
    series_groups,
    smooth,
    summarize,
    summary_rows,
    thin,
    to_csv,
)
from .explorer import RunExplorer
from .inspector import PointInspector, format_value
from .layouts import (
    DEFAULT_LAYOUT,
    LAYOUTS,
    Filmstrip,
    KpiStrip,
    PlotsList,
    SeriesChips,
    headline_metric,
)
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
    "Tables": (("table", "metrics"), "The ticked runs have no tables."),
    "Figures": (("figure",), "The ticked runs have no figures yet."),
    "Reports": (("report",), "The ticked runs have no reports yet."),
    "Config": (("config", "metadata"), "The ticked runs have no config files."),
    "Files": (None, "Tick runs to list every file they produced."),
}
POLL_MS = 2000  # how often ticked runs that are still training fetch their new rows
RELIST_EVERY = 15  # polls between re-listing runs (~30 s): learns which runs finished or started
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

    Curves: one chart per metric, with a point inspector showing every value at the hovered step;
    an agent's versions (repeated runs) can be combined into a mean ± std or ± SEM band.
    Summary: final, best, range, mean and area under the curve per series and metric, plus when a
    threshold was first reached. Export writes the visible curves or the summary as CSV, or the charts as PNG.
    Tables, Figures, Reports, Config (with a cross-run diff) and Files: every artifact in the runs'
    manifests, previewed by kind.

    `backend` needs get(path, callback); `settings` (optional) needs get(*keys, default=) and
    set(*keys, value), and keeps hidden plots and the chosen layout (one of LAYOUTS) across sessions under
    [results] hidden_plots and layout.
    `load_config(run_key)` (optional) puts a run's saved builder config into the experiment builder and
    returns None, or returns why it could not.
    """

    def __init__(self, backend, settings=None, load_config=None, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.load_config = load_config
        self.curves: dict[str, list] = {}  # metric -> [(label, chart points, color, combined)], as drawn
        self.load_button = None
        self.loader = ResultsLoader(backend, self)
        self.sources: list[Source] = []
        self.hidden: set[str] = set(self._setting("hidden_plots", []))
        self.smoothing = 0.0

        # Controls, run tree, tabs, charts and inspector are built once; each layout (see layouts.py)
        # only arranges them, so switching layouts keeps every tick, choice and loaded table.
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
            "The series whose logged steps hovering snaps to (drawn on top, thicker, and listed first in the "
            "inspector). Nearest snaps to whichever series is closest to the mouse."
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
        self.kpis = KpiStrip()
        self.kpis.clicked.connect(self.set_primary)
        self.chips = SeriesChips()
        self.chips.clicked.connect(self.set_primary)
        self.filmstrip = Filmstrip()
        self.filmstrip.chosen.connect(self.set_focus)
        self.focus_metric: str | None = None
        self.plots_list = PlotsList()
        # Deferred: hiding a plot redraws, which rebuilds this list from inside its own signal.
        self.plots_list.visibility_changed.connect(
            lambda metric, on: QTimer.singleShot(0, lambda: self.set_hidden(metric, not on))
        )
        self._parking = QWidget(self)  # holds the widgets the current layout leaves out; never shown
        self._parking.hide()
        self.tabs.addTab(self.curves_page, "Curves")
        self.summary = SummaryView()
        self.summary.changed.connect(self.update_summary)
        self.tabs.addTab(self.summary, "Summary")
        # Artifact tabs start empty and build their browser when first opened: each one is hundreds of
        # widgets, and the app restyles every widget when its theme is applied.
        self.browsers: dict[str, ArtifactBrowser] = {}
        self.config_diff: ConfigDiff | None = None
        self._pages: dict[str, QWidget] = {}
        for name in ARTIFACT_TABS:
            page = QWidget()
            QVBoxLayout(page).setContentsMargins(0, 0, 0, 0)
            self._pages[name] = page
            self.tabs.addTab(page, name)
        self.tabs.currentChanged.connect(self._tab_opened)

        self.root = QVBoxLayout(self)
        self.root.setContentsMargins(18, 14, 18, 12)
        self.root.setSpacing(10)
        self.frame: QWidget | None = None
        self.status = label("", "muted")
        self.root.addWidget(self.status)
        layout_name = self._setting("layout", DEFAULT_LAYOUT)
        self.layout_name = layout_name if layout_name in LAYOUTS else DEFAULT_LAYOUT
        self._compose()

        self.loader.runs_listed.connect(self._runs_listed)
        self.loader.manifest_loaded.connect(self.explorer.add_manifest)
        self.loader.manifest_loaded.connect(lambda _manifest: self.update_artifacts())
        self.loader.file_loaded.connect(self._file_loaded)
        self.loader.table_loaded.connect(lambda _source: self.redraw())
        self.loader.failed.connect(lambda message: self.status.setText(f"Could not load: {message}"))

        # Live updates: while the panel is on screen, ticked runs that are still training are
        # polled for new rows, and the run list is refreshed now and then.
        self.unfinished: set[str] = set()  # run keys without a recorded end time
        self.run_count = 0
        self._listing: tuple = ()
        self._polls = 0
        self.poll_timer = QTimer(self)
        self.poll_timer.setInterval(POLL_MS)
        self.poll_timer.timeout.connect(self.poll)
        self.poll_timer.start()
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

    def set_layout(self, name: str):
        """Switch to one of LAYOUTS and remember it under [results] layout."""
        if name not in LAYOUTS or name == self.layout_name:
            return
        self.layout_name = name
        if self.settings is not None:
            self.settings.set("results", "layout", name)
        self._compose()
        self.redraw()

    def show_all(self):
        self.hidden.clear()
        if self.settings is not None:
            self.settings.set("results", "hidden_plots", [])
        self.redraw()

    # ── Layouts ──────────────────────────────────────────────────────────────

    def _shared_widgets(self) -> list[QWidget]:
        return [
            self.plots_button,
            self.axis_select,
            self.smoothing_label,
            self.smoothing_slider,
            self.combine_select,
            self.primary_select,
            self.export_button,
            self.refresh_button,
            self.explorer,
            self.tabs,
            self.legend,
            self.grid,
            self.inspector,
            self.kpis,
            self.chips,
            self.filmstrip,
            self.plots_list,
        ]

    def _compose(self):
        """Arrange the shared widgets as `self.layout_name` says, dropping the previous arrangement."""
        for widget in self._shared_widgets():
            widget.setParent(self._parking)
        for old in (self.frame, self.curves_body):
            if old is not None:
                old.setParent(None)
                old.deleteLater()
        self.frame = QWidget()
        frame = QVBoxLayout(self.frame)
        frame.setContentsMargins(0, 0, 0, 0)
        frame.setSpacing(10)
        self.curves_body = QWidget()
        body = QVBoxLayout(self.curves_body)
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(8)
        self.curves_page.layout().addWidget(self.curves_body)
        self.root.insertWidget(0, self.frame, 1)
        name = self.layout_name
        self.inspector.set_transposed(name == "rail")
        self.tab_corner.setVisible(name == "rail")

        if name == "rail":
            split = QSplitter(Qt.Orientation.Horizontal)
            split.addWidget(self._rail())
            split.addWidget(self.tabs)
            split.setSizes([280, 900])
            frame.addWidget(split, 1)
            self.tab_corner.layout().addWidget(self.export_button)
            body.addWidget(self.legend)
            charts = QSplitter(Qt.Orientation.Vertical)
            charts.addWidget(self.grid)
            charts.addWidget(self.inspector)
            charts.setSizes([620, 180])
            body.addWidget(charts, 1)
            return

        frame.addLayout(self._toolbar())
        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(self.explorer)
        split.addWidget(self.tabs)
        split.setSizes([260, 900])
        frame.addWidget(split, 1)
        if name == "quiet":
            body.addWidget(self.legend)
            charts = QSplitter(Qt.Orientation.Horizontal)
            charts.addWidget(self.grid)
            charts.addWidget(self.inspector)
            charts.setSizes([760, 300])
            body.addWidget(charts, 1)
        elif name == "focus":
            body.addWidget(self.chips)
            main = QWidget()
            main_layout = QVBoxLayout(main)
            main_layout.setContentsMargins(0, 0, 0, 0)
            main_layout.setSpacing(8)
            main_layout.addWidget(self.grid, 1)
            main_layout.addWidget(self.filmstrip)
            charts = QSplitter(Qt.Orientation.Horizontal)
            charts.addWidget(main)
            charts.addWidget(self.inspector)
            charts.setSizes([760, 300])
            body.addWidget(charts, 1)
        else:  # overview: the series cards head the grid, above its large chart
            charts = QSplitter(Qt.Orientation.Vertical)
            charts.addWidget(self.grid)
            charts.addWidget(self.inspector)
            charts.setSizes([640, 180])
            body.addWidget(charts, 1)

    def _toolbar(self) -> QHBoxLayout:
        """One row of controls: view settings on the left, actions on the right."""
        row = QHBoxLayout()
        row.setSpacing(8)
        row.addWidget(label("X-axis", "muted"))
        row.addWidget(self.axis_select)
        row.addSpacing(10)
        row.addWidget(self.smoothing_label)
        row.addWidget(self.smoothing_slider)
        row.addSpacing(10)
        row.addWidget(label("Versions", "muted"))
        row.addWidget(self.combine_select)
        row.addSpacing(10)
        row.addWidget(label("Primary", "muted"))
        row.addWidget(self.primary_select)
        row.addStretch()
        for widget in (self.plots_button, self.export_button, self.refresh_button):
            row.addWidget(widget)
        return row

    def _rail(self) -> QWidget:
        """The Control rail: runs, view settings and a checkbox per plot, top to bottom."""
        rail = QWidget()
        rail.setObjectName("resultsRail")
        layout = QVBoxLayout(rail)
        layout.setContentsMargins(0, 0, 8, 0)
        layout.setSpacing(8)
        layout.addWidget(label("RUNS", "eyebrow"))
        layout.addWidget(self.explorer, 3)
        layout.addWidget(label("VIEW", "eyebrow"))
        form = QGridLayout()
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(6)
        rows = (("X-axis", self.axis_select), ("Versions", self.combine_select), ("Primary", self.primary_select))
        for i, (caption, widget) in enumerate(rows):
            form.addWidget(label(caption, "muted"), i, 0)
            form.addWidget(widget, i, 1)
        form.addWidget(self.smoothing_label, len(rows), 0)
        form.addWidget(self.smoothing_slider, len(rows), 1)
        form.setColumnStretch(1, 1)
        layout.addLayout(form)
        layout.addWidget(label("PLOTS", "eyebrow"))
        layout.addWidget(self.plots_list, 2)
        layout.addWidget(self.refresh_button)
        return rail

    def _grid_options(self, visible: list[str]) -> dict:
        name = self.layout_name
        if name == "rail":
            return {"columns": 3, "flat": True}
        if name == "focus":
            if self.focus_metric not in visible:
                self.focus_metric = headline_metric(visible)
            return {"columns": 1, "focus": self.focus_metric}
        if name == "overview":
            return {"collapsible": True, "hero": headline_metric(visible), "header": self.kpis}
        return {"collapsible": True}

    def set_focus(self, metric: str):
        """Show `metric` as the Focus layout's large chart."""
        self.focus_metric = metric
        self.redraw()

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

    def _runs_listed(self, entries):
        listing = tuple(
            sorted(
                (e["group"], e["experiment_id"], tuple(a.get("name") for a in e.get("agents") or [])) for e in entries
            )
        )
        if listing != self._listing:  # rebuild the tree only when runs or agents changed, not on every re-list
            self._listing = listing
            self.explorer.set_runs(entries)
        self.unfinished = {
            f"{e['group']}/{e['experiment_id']}" for e in entries if e.get("agents") and not e.get("finished")
        }
        self.run_count = sum(1 for e in entries if e.get("agents"))
        self._show_status()

    def live_sources(self) -> list[Source]:
        """Ticked sources whose run is still training (no recorded end time)."""
        return [s for s in self.sources if s.run_key in self.unfinished]

    def _show_status(self):
        live = len({s.run_key for s in self.live_sources()})
        following = f" · following {live} run{'s' * (live != 1)} still training" if live else ""
        self.status.setText(
            f"{self.run_count} run{'s' * (self.run_count != 1)} in results/logs. "
            f"Tick runs to plot every metric they logged.{following}"
        )

    def poll(self):
        """One live-update tick: new rows for ticked runs still training; now and then, re-list runs."""
        if not self.isVisible():
            return
        for source in self.live_sources():
            self.loader.poll_table(source)
        self._polls += 1
        if self._polls % RELIST_EVERY == 0:
            self.loader.list_runs()

    def _selection_changed(self, sources):
        self.sources = sources
        self._show_status()
        for source in sources:
            self.loader.load_table(source)
            self.loader.load_manifest(source.group, source.experiment_id)
        self.redraw()
        self.update_artifacts()

    def run_keys(self) -> list[str]:
        """Ticked runs, in tree order, each once (several agents or versions share a run)."""
        return list(dict.fromkeys(source.run_key for source in self.sources))

    # ── Artifacts ────────────────────────────────────────────────────────────

    def browser(self, name: str) -> ArtifactBrowser:
        """The artifact browser of one tab ("Tables", "Figures", …), built on first use."""
        if name not in self.browsers:
            kinds, empty = ARTIFACT_TABS[name]
            browser = self.browsers[name] = ArtifactBrowser(self.loader, kinds, empty)
            layout = self._pages[name].layout()
            if name == "Config":
                self.config_diff = ConfigDiff()
                top = QWidget()
                top_layout = QVBoxLayout(top)
                top_layout.setContentsMargins(0, 0, 0, 0)
                actions = QHBoxLayout()
                actions.addStretch()
                self.load_button = QPushButton("Load in experiment builder")
                self.load_button.setToolTip("Put the ticked run's saved builder settings into the Experiment tab")
                self.load_button.clicked.connect(self.load_in_builder)
                actions.addWidget(self.load_button)
                top_layout.addLayout(actions)
                top_layout.addWidget(self.config_diff, 1)
                split = QSplitter(Qt.Orientation.Vertical)
                split.addWidget(top)
                split.addWidget(browser)
                split.setSizes([300, 400])
                layout.addWidget(split)
            else:
                layout.addWidget(browser)
            self.update_artifacts()
        return self.browsers[name]

    def _tab_opened(self, index):
        name = self.tabs.tabText(index)
        if name in ARTIFACT_TABS:
            self.browser(name)

    def manifests(self):
        """Loaded manifests of the ticked runs."""
        return [self.loader.manifests[k] for k in self.run_keys() if k in self.loader.manifests]

    def load_in_builder(self):
        """Load the one ticked run's saved builder config into the experiment builder."""
        runs = self.run_keys()
        if len(runs) != 1:
            self.status.setText("Tick exactly one run to load its saved config into the experiment builder.")
            return
        problem = self.load_config(runs[0]) if self.load_config else "The experiment builder is not available here."
        self.status.setText(problem or f"Loaded the saved config of {runs[0]} into the experiment builder.")

    def update_artifacts(self):
        if self.load_button is not None:
            self.load_button.setEnabled(len(self.run_keys()) == 1)
        manifests = self.manifests()
        for browser in self.browsers.values():
            browser.set_manifests(manifests)
        if self.config_diff is None:
            return
        for manifest in manifests:
            path = run_config_path(manifest.run_key)
            if any(a["path"] == path for a in manifest.artifacts):
                self.loader.load_file(manifest.run_key, path)
        self._show_config_diff()

    def _file_loaded(self, run_key, path):
        if path == run_config_path(run_key):
            self._show_config_diff()

    def _show_config_diff(self):
        if self.config_diff is None:
            return
        configs, missing = {}, []
        for key in self.run_keys():
            manifest = self.loader.manifests.get(key)
            if manifest is None:
                continue  # still loading
            data = self.loader.files.get((key, run_config_path(key)))
            if data is not None:
                configs[key] = data
            elif not any(a["path"] == run_config_path(key) for a in manifest.artifacts):
                missing.append(key)
        self.config_diff.show_configs(configs, missing)

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
        self.grid.configure(**self._grid_options(visible))
        self.grid.set_metrics(visible, empty)
        self.curves = self.curve_series(axis)
        groups = self.plotted_groups()
        primary = self._sync_primary([name for name, _ in groups])
        large = {self.grid.hero, self.grid.focus}
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
        """Fill what only some layouts show: card values, series cards, chips, filmstrip, plot checklist."""
        name = self.layout_name
        colors = {series: PALETTE[i % len(PALETTE)] for i, (series, _) in enumerate(groups)}
        for metric, card in self.grid.cards.items():
            card.set_value(*self.latest(metric, primary) if name in ("quiet", "overview") else ("",))
        if name == "overview":
            hero = self.grid.hero
            self.kpis.set_cards(self.kpi_cards(axis, hero, groups, colors) if hero else [], primary)
        elif name == "focus":
            focus = self.grid.focus
            chips = []
            for series, _ in groups:
                drawn = next((p for n, p, *_ in self.curves.get(focus, []) if n == series), None)
                chips.append((series, colors[series], format_value(focus, drawn[-1][focus]) if drawn else ""))
            self.chips.set_chips(chips, primary)
            tiles = [
                (
                    metric,
                    describe(metric).label,
                    [(n, thin(p, metric, 120), color) for n, p, color, _ in self.curves.get(metric, [])],
                    self.latest(metric, primary)[0],
                )
                for metric in visible
            ]
            self.filmstrip.set_tiles(tiles, focus, primary)
        elif name == "rail":
            self.plots_list.set_metrics(metrics, self.hidden)

    def kpi_cards(self, axis, metric, groups, colors) -> list[tuple[str, str, list[tuple[str, str]]]]:
        """The Overview's headline numbers per series: best, final, when it got there, training time.

        "Reaches … at" replaces "Best at" while the Summary tab has a threshold for this metric.
        """
        tables = dict(self.loaded_tables())
        higher = describe(metric).higher_is_better
        threshold = self.summary.threshold_value() if self.summary.selected_metric() == metric else None
        cards = []
        for series, members in groups:
            parts = [p for s in members if (p := tables[s].series(metric, axis))]
            points = parts[0] if len(parts) == 1 else [(a.x, a.mean) for a in aggregate(parts)]
            stats = []
            summary = summarize(points, higher)
            if summary is not None:
                stats.append(("Best" if higher is not None else "Max", format_value(metric, summary.best)))
                stats.append(("Final", format_value(metric, summary.final)))
                if threshold is not None:
                    at = first_reaching(points, threshold, higher)
                    stats.append((f"Reaches {threshold:g} at", "not yet" if at is None else f"{at:,.0f}"))
                else:
                    stats.append(("Best at", f"{summary.best_x:,.0f}"))
            seconds = [t for s in members if (t := self.training_time(tables[s])) is not None]
            if seconds:
                stats.append(("Training time", duration(max(seconds))))
            cards.append((series, colors[series], stats))
        return cards

    @staticmethod
    def training_time(table) -> float | None:
        """The last training time a run logged, in seconds."""
        for column in TIME_COLUMNS:
            values = [v for v in table.columns.get(column, []) if v is not None]
            if values:
                return values[-1]
        return None

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
            notes.append(f"no points on this x-axis for {', '.join(absent)} (pick another x-axis, or see Tables)")
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

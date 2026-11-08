"""Training monitor panel: live training curves for the selected run.

The panel owns its widgets and drawing. Run state (the run list, the selected and active runs,
the queue) stays on the host window, which the rest of the app also reads.
"""
from __future__ import annotations

from ..method_labels import monitor_method_label, parent_trial_label

from typing import TYPE_CHECKING
from html import escape

from PyQt6.QtCore import QEvent, Qt, QTimer
from PyQt6.QtWidgets import (
    QFrame,
    QApplication,
    QPlainTextEdit,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ..model import LIVE_STATUSES, METRIC_COLUMNS, available_metrics
from ..results.metrics_catalog import describe, group_sort_key, metric_group
from ..queue_panel import QueuePanel
from ..theme import theme_color
from ..widgets import ElidedLabel, label
from .live_view import PALETTE, ChartsView, FeedView, FilesView, LogView, StatusStrip, SystemView, latest_problem
from .pipeline_strip import PipelineStrip, pipeline_segments
from .runs_sidebar import RunsSidebar

if TYPE_CHECKING:
    from ..app import Window

# Second-chart metrics with hand-written text: (card title, chart title, card subtitle, number format).
# Any other metric a run logged is offered too, described by the metrics catalog.
SECOND_METRICS = {
    "loss": ("TRAINING LOSS", "Total loss", "Combined training objective", ".4f"),
    "value_loss": ("VALUE LOSS", "Value loss", "value-function error · critic MSE loss", ".3f"),
    "policy_loss": ("POLICY LOSS", "Policy loss", "Policy optimization objective", ".4f"),
    "entropy": ("POLICY ENTROPY", "Policy entropy", "how random actions are · ln 2 ≈ 0.693 is uniform", ".3f"),
    "approx_kl": ("APPROX. KL", "Approximate KL per update", "size of each policy update", ".5f"),
}
SECTION_COLUMNS = {"losses": 4}  # losses are small multiples, four to a row; other sections have two
SIDEBAR_MIN_WIDTH = 250
LIVE_REFRESH_MS = 400  # the live views redraw at most this often while records stream in
# Shown in the reward chart, so not offered for the second one.
_REWARD_KEYS = {"reward", "reward_std"}


#: Run-record key for each metrics.csv column, so a declared monitor_metric can be
#: looked up in the points metric_points() builds (eval/reward is stored as "reward").
_RECORD_KEY = {column: key for key, column in METRIC_COLUMNS.items()}


def headline_metric(run, config):
    """The metric this run is judged by, as a run-record key, and its description.

    Taken from the environment's `monitor_metric`, which is already the number
    the backend selects `best_model.ckpt` on. The headline card used to be
    episode reward whatever the run was, so a supervised run - which has no
    environment, no episodes and no reward - showed a dash above an empty chart
    while the value it was actually scored by sat in the second slot.

    Falls back to reward, so a run whose environment declares nothing, or whose
    environment is no longer on disk, looks exactly as it did before.
    """
    declared = None
    env_name = (config or {}).get("env")
    if env_name:
        from ..config_viewer import config_tree  # local import: keeps this module light

        tree = config_tree()
        env = tree.environments.get(env_name) if tree else None
        if env is not None:
            declared = env.raw.get("monitor_metric")
    if not declared:
        return "reward", describe("eval/reward")
    key = _RECORD_KEY.get(str(declared), str(declared))
    return key, describe(str(declared))


def measures_environment_steps(run):
    """Whether this run's progress is counted in steps through an environment.

    Asked of the paradigm rather than hardcoded. One that declares agents has a
    policy acting in an environment, so a step budget is its real extent.
    Supervised and unsupervised declare none and train by epochs: for them the
    budget is a number the run never approaches, and using it as the chart's
    x-extent squeezes the whole curve into the first few percent.

    Runs recorded before the paradigm was stored, and those found on disk, keep
    the previous behaviour.
    """
    paradigm = run.get("paradigm")
    if not paradigm:
        return True
    from ..config_viewer import config_tree  # local import: keeps this module light

    tree = config_tree()
    if tree is None or paradigm not in tree.paradigms:
        return True
    p_obj = tree.paradigms.get(paradigm)
    if p_obj and hasattr(p_obj, "field_enabled") and p_obj.field_enabled("eval_episodes"):
        return True
    if paradigm in ("online_rl", "offline_rl", "rl"):
        return True
    return bool(tree.agents_for(paradigm))


def run_setup_label(run, config):
    """The "environment / methods" half of a run's caption.

    Derived from the run itself. Simulated runs have no environment or method.
    """
    if run.get("simulated"):
        return "Simulated run"
    env = (config or {}).get("env") or "?"
    methods = (run.get("backend") or {}).get("agents") or ["?"]
    return f"{env}  /  {' + '.join(methods).upper()}"


def second_metric_choices(run) -> list[str]:
    """Every metric this run logged except reward: the hand-described ones first, then the rest by group."""
    available = available_metrics(run) - _REWARD_KEYS
    known = [key for key in SECOND_METRICS if key in available]
    others = sorted(available - set(SECOND_METRICS), key=lambda n: (group_sort_key(metric_group(n)), n))
    return known + others or ["loss"]


def second_metric_text(key: str) -> tuple[str, str, str, str]:
    """(card title, chart title, card subtitle, number format) for any metric key."""
    if key in SECOND_METRICS:
        return SECOND_METRICS[key]
    info = describe(key)
    return info.label.upper(), info.label, info.description or key, info.fmt


def run_display_name(run) -> str:
    return (run.get("display_name") or (run.get("backend") or {}).get("experiment_id")
            or (run.get("config") or {}).get("name") or run.get("id", "Experiment"))


class MonitorPanel(QWidget):
    """The Training monitor tab."""

    # Widgets the host window also exposes under the same names (existing code and tests use them).
    EXPOSED = (
        "run_title", "run_caption",
    )
    # The views besides the workspace, opened from the status row in its place.
    VIEW_NAMES = ("System", "Logs", "Feed", "Files")

    def __init__(self, host: Window):
        super().__init__()
        self.host = host
        # The runs list lives in the Plots tab (the window places it there): its ticks choose what the Plot
        # Viewer draws. This panel shows only the current or most recent run (host.selected).
        self.sidebar = RunsSidebar()
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(16, 0, 0, 0)
        layout.setSpacing(12)

        self.content = content
        # One status row: badge, the run's name and description, its numbers, and the other views.
        self.status_strip = StatusStrip()
        # Long names give way first, ending in … (the full text is on the tooltip), so the row fits.
        self.run_title = ElidedLabel("Your next experiment")
        self.run_title.setObjectName("heading")
        self.run_caption = ElidedLabel("Configure an experiment, then launch training.")
        self.run_caption.setObjectName("muted")
        for widget in (self.run_title, self.run_caption):
            widget.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        title_block = QWidget()
        title_layout = QVBoxLayout(title_block)
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.setSpacing(2)
        title_row = QHBoxLayout()
        title_row.setContentsMargins(0, 0, 0, 0)
        title_row.setSpacing(0)
        title_row.addWidget(self.run_title)
        self.run_title.hide()
        self.group_button = QPushButton()
        self.experiment_button = QPushButton()
        for button, tooltip in ((self.group_button, "Choose an experiment group"),
                                (self.experiment_button, "Choose an experiment to monitor")):
            button.setObjectName("monitorMethodTitle")
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setToolTip(tooltip)
            button.setMenu(QMenu(button))
            button.menu().setObjectName("monitorTitleMenu")
            button.menu().aboutToShow.connect(self._refresh_run_selectors)
            title_row.addWidget(button)
        self.method_button = QPushButton()
        self.method_button.setObjectName("monitorMethodTitle")
        self.method_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.method_button.setToolTip("Choose a method to monitor")
        self.method_button.setMenu(QMenu(self.method_button))
        self.method_button.menu().setObjectName("monitorTitleMenu")
        self.method_button.hide()
        title_row.addWidget(self.method_button)
        self.trial_button = QPushButton()
        self.trial_button.setObjectName("monitorMethodTitle")
        self.trial_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.trial_button.setToolTip("Choose the sweep overview or a trial")
        self.trial_button.setMenu(QMenu(self.trial_button))
        self.trial_button.menu().setObjectName("monitorTitleMenu")
        self.trial_button.hide()
        title_row.addWidget(self.trial_button)
        title_row.addStretch()
        title_layout.addLayout(title_row)
        self.method_legend = QLabel()
        self.method_legend.setObjectName("muted")
        self.method_legend.setTextFormat(Qt.TextFormat.RichText)
        self.method_legend.setWordWrap(True)
        title_layout.addWidget(self.method_legend)
        title_layout.addWidget(self.run_caption)
        self.status_strip.add_title(title_block)
        self.status_strip.badge.hide()
        self.run_caption.hide()
        view_buttons = QWidget()
        view_row = QHBoxLayout(view_buttons)
        view_row.setContentsMargins(8, 0, 0, 0)
        view_row.setSpacing(4)
        self.view_buttons: dict[str, QPushButton] = {}
        for name in self.VIEW_NAMES:
            button = QPushButton(name)
            button.setObjectName("monitorViewButton")
            button.setCheckable(True)
            button.setToolTip(f"Show {name} in place of the charts; click again to go back")
            button.clicked.connect(lambda checked, n=name: self.show_view(n if checked else "Overview"))
            view_row.addWidget(button)
            self.view_buttons[name] = button
        self.yaml_button = QPushButton("Config")
        self.yaml_button.setObjectName("monitorViewButton")
        self.yaml_button.setCheckable(True)
        self.yaml_button.setToolTip("Show the saved configuration for the selected run")
        self.yaml_button.clicked.connect(self.show_run_yaml)
        view_row.addWidget(self.yaml_button)
        self.queue_button = QPushButton("Queue")
        self.queue_button.setObjectName("monitorViewButton")
        self.queue_button.setCheckable(True)
        self.queue_button.clicked.connect(self._toggle_queue_overlay)
        view_row.addWidget(self.queue_button)
        self.status_strip.add_trailing(view_buttons)
        layout.addWidget(self.status_strip)

        # The run's methods in training order; clicking one shows only its curves.
        self.pipeline_strip = PipelineStrip(content)
        self.pipeline_strip.segment_clicked.connect(self.focus_agent)
        self.pipeline_strip.hide()
        self.queue_overlay = QFrame(content)
        self.queue_overlay.setObjectName("card")
        self.queue_overlay.setFrameShape(QFrame.Shape.StyledPanel)
        self.queue_layout = QVBoxLayout(self.queue_overlay)
        self.queue_layout.setContentsMargins(0, 0, 0, 0)
        self.queue_preview = QueuePanel(self._open_queue_job, split_methods=True,
                                        lives_for_job=self._queue_lives_for_job,
                                        on_cancel=getattr(getattr(host, "jobs", None), "cancel", None))
        self.queue_preview.structure_changed.connect(self._place_queue_overlay)
        self.queue_layout.addWidget(self.queue_preview)
        self.queue_overlay.hide()
        if hasattr(host, "jobs"):
            host.jobs.queue_changed.connect(self._queue_changed)
        self.yaml_overlay = QFrame(content)
        self.yaml_overlay.setObjectName("card")
        yaml_layout = QVBoxLayout(self.yaml_overlay)
        yaml_layout.setContentsMargins(14, 12, 14, 12)
        self.yaml_viewer = QPlainTextEdit()
        self.yaml_viewer.setReadOnly(True)
        yaml_layout.addWidget(self.yaml_viewer)
        self.yaml_overlay.hide()
        QApplication.instance().installEventFilter(self)
        self.focused_agent: str | None = None
        self._focused_sweep: str | None = None
        self._focus_run_id: str | None = None  # the run whose method is focused

        # One grouped chart for each logged metric.
        self.charts_view = ChartsView(
                                      group_columns=SECTION_COLUMNS, show_caption=False)
        self.overview_scroll = self.charts_view.grid
        self.overview_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.overview_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.system_view = SystemView()
        self.log_view = LogView()
        self.feed_view = FeedView()
        self.files_view = FilesView()
        from .sweep_view import SweepView
        self.sweep_view = SweepView()
        self.sweep_view.trial_selected.connect(self._open_sweep_trial)
        self.views = QStackedWidget()
        self._views = {"Overview": self.charts_view, "System": self.system_view, "Logs": self.log_view,
                       "Feed": self.feed_view, "Files": self.files_view, "Sweep": self.sweep_view}
        for view in self._views.values():
            self.views.addWidget(view)
        self.views.currentChanged.connect(lambda _: self.refresh_live())
        layout.addWidget(self.views, 1)

        # Bottom row: the newest warning (with the log), or what the run is waiting for; plugin buttons.
        bottom = QHBoxLayout()
        self.problem_bar = QFrame()
        self.problem_bar.setObjectName("monitorProblemBar")
        problem_row = QHBoxLayout(self.problem_bar)
        problem_row.setContentsMargins(10, 4, 4, 4)
        self.problem_text = label("", "muted")
        self.problem_text.setWordWrap(True)
        problem_row.addWidget(self.problem_text, 1)
        self.open_log = QPushButton("Open log")
        self.open_log.setToolTip("Show the pipeline's output")
        self.open_log.clicked.connect(lambda: self.show_view("Logs"))
        problem_row.addWidget(self.open_log)
        self.problem_bar.hide()
        bottom.addWidget(self.problem_bar, 1)
        if hasattr(host, "extension_slots") and host.extension_slots is not None:
            from ..plugins.slots import ExtensionSlotWidget
            slot = host.extension_slots.get_slot("monitor.footer")
            if slot:
                self.footer_slot_widget = ExtensionSlotWidget(slot)
                bottom.addWidget(self.footer_slot_widget)
        layout.addLayout(bottom)
        self.status_note: str | None = None  # what the run is waiting for, when that is worth saying

        # Records can arrive many times a second: redraw the live views at most a few times a second,
        # and tick the status strip ("updated 3 s ago") once a second.
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(LIVE_REFRESH_MS)
        self._refresh_timer.timeout.connect(self.refresh_live)
        self._clock = QTimer(self)
        self._clock.setInterval(1000)
        self._clock.timeout.connect(self._tick_status)
        self._clock.start()
        self._log_run_id = None
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 18, 20, 14)
        self.sidebar.setMinimumWidth(SIDEBAR_MIN_WIDTH)
        root.addWidget(content)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._place_queue_overlay()
        self._place_yaml_overlay()

    def _place_yaml_overlay(self):
        if not hasattr(self, "yaml_overlay"):
            return
        width = min(560, max(1, self.content.width() - 20))
        anchor = self.yaml_button.mapTo(self.content, self.yaml_button.rect().bottomLeft())
        top = anchor.y() + 8
        self.yaml_overlay.setGeometry(max(0, self.content.width() - width), top, width,
                                      max(1, self.content.height() - top - 10))

    def _place_queue_overlay(self):
        if not hasattr(self, "queue_overlay"):
            return
        width = max(1, min(self.content.width() - 20, self.content.width() * 2 // 3))
        anchor = self.queue_button.mapTo(self.content, self.queue_button.rect().bottomLeft())
        top = anchor.y() + 8
        available_height = max(1, self.content.height() - top - 10)
        height = min(available_height, self.queue_preview.content_height(width))
        self.queue_overlay.setGeometry(max(0, self.content.width() - width), top, width, height)

    def _toggle_queue_overlay(self, checked):
        if checked:
            self._refresh_queue_overlay()
            self._place_queue_overlay()
            self.queue_overlay.show()
            self.queue_overlay.raise_()
        else:
            self.queue_overlay.hide()

    def eventFilter(self, watched, event):
        # Native clicks reach the QWindow before Qt dispatches them to a widget.
        # Dismissing at that stage hides the card before it can receive the click.
        if event.type() == QEvent.Type.MouseButtonPress and isinstance(watched, QWidget):
            widget = watched
            ancestor = widget
            while ancestor is not None and ancestor not in (self.queue_button, self.yaml_button):
                ancestor = ancestor.parentWidget()
            if ancestor is not None and (self.queue_overlay.isVisible() or self.yaml_overlay.isVisible()):
                self.queue_button.setChecked(False)
                self.yaml_button.setChecked(False)
                self.queue_overlay.hide()
                self.yaml_overlay.hide()
                return True
            for overlay, button in ((self.queue_overlay, self.queue_button), (self.yaml_overlay, self.yaml_button)):
                if not overlay.isVisible():
                    continue
                target = widget
                while target is not None and target not in (overlay, button):
                    target = target.parentWidget()
                if target is None:
                    button.setChecked(False)
                    overlay.hide()
        return super().eventFilter(watched, event)

    def _refresh_queue_overlay(self):
        data = self.host.jobs.queue_state()
        jobs = []
        for job in [*data.get("active", []), *[dict(item, queue_root=True) for item in data.get("queued", [])]]:
            job = dict(job)
            run = self.host.run_for_job(job.get("job_id"))
            if not (job.get("sweep_parent") or job.get("sweep_trial")):
                job["description"] = run.get("description", "") if run else ""
            jobs.append(job)
        self.queue_preview.set_data(dict(data, active=jobs))
        self.queue_preview.empty.setText("No running or queued jobs")
        self._place_queue_overlay()

    def _queue_lives_for_job(self, job_id):
        run = self.host.run_for_job(job_id)
        return self.host.live_for_run(run, load=False) if run else {}

    def _queue_changed(self, _data):
        if self.queue_overlay.isVisible():
            self._refresh_queue_overlay()

    def _open_queue_job(self, job_id, method):
        self.host.open_job(job_id, method)
        self.queue_button.setChecked(False)
        self.queue_overlay.hide()

    def show_run_yaml(self, checked=True):
        self.yaml_button.setChecked(checked)
        if not checked:
            self.yaml_overlay.hide()
            return
        self._refresh_run_yaml()
        self._place_yaml_overlay()
        self.yaml_overlay.show()
        self.yaml_overlay.raise_()

    def _refresh_run_yaml(self):
        run = self.host.selected
        folder = self.host.run_folder(run) if run else None
        path = folder / "config.yaml" if folder else None
        try:
            text = path.read_text(encoding="utf-8") if path else ""
        except OSError:
            text = ""
        self.yaml_viewer.setPlainText(text if text else "Saved configuration is not available yet for this run.")

    def show_view(self, name: str):
        """Show one of the views ("Overview" is the charts workspace) and mark its button."""
        self.views.setCurrentWidget(self._views[name])
        for each, button in self.view_buttons.items():
            button.setChecked(each == name)

    # ── Run navigation ───────────────────────────────────────────────────────

    def set_runs(self, runs, selected=None):
        """Refill the runs list (its ticks are kept)."""
        self.sidebar.set_runs(runs)
        self._refresh_run_selectors()

    def _monitor_runs(self):
        runs = [run for run in getattr(self.host, "runs", [])
                if run.get("status") in (*LIVE_STATUSES, "queued")]
        selected = self.host.selected
        if (selected and selected.get("status") in (*LIVE_STATUSES, "queued")
                and not any(run.get("id") == selected.get("id") for run in runs)):
            runs.append(selected)
        return runs

    def _current_run(self):
        selected = self.host.selected
        if selected and selected.get("status") in (*LIVE_STATUSES, "queued"):
            return selected
        return next(iter(self._monitor_runs()), None)

    def _refresh_run_selectors(self):
        selected = self._current_run()
        if not selected:
            self.group_button.hide()
            self.experiment_button.hide()
            self.run_title.show()
            return
        self.run_title.hide()
        self.group_button.show()
        self.experiment_button.show()
        group_for = lambda run: (run.get("backend") or {}).get("group") or "Ungrouped"
        group = group_for(selected)
        self.group_button.setText(group)
        self.experiment_button.setText(f" / {run_display_name(selected)}")
        runs = self._monitor_runs()
        menu = self.group_button.menu()
        if not menu.isVisible():
            menu.clear()
        for name in (() if menu.isVisible() else dict.fromkeys(group_for(run) for run in runs)):
            target = selected if name == group else next(run for run in runs if group_for(run) == name)
            action = menu.addAction(name)
            action.setCheckable(True)
            action.setChecked(name == group)
            action.triggered.connect(lambda checked=False, r=target: self._select_monitor_run(r))
        menu = self.experiment_button.menu()
        if not menu.isVisible():
            menu.clear()
        for run in (() if menu.isVisible() else runs):
            if group_for(run) != group:
                continue
            action = menu.addAction(f"{run_display_name(run)} · {run.get('status', '')}")
            action.setCheckable(True)
            action.setChecked(run.get("id") == selected.get("id"))
            action.triggered.connect(lambda checked=False, r=run: self._select_monitor_run(r))

    def _select_monitor_run(self, run):
        self.host.select_run(run)
        self.focus_agent("")
        self.show_view("Overview")
        self._refresh_run_selectors()

    def show_selected(self, run):
        """`run` is now the one shown here."""
        self._refresh_run_selectors()
        if run and run.get("id") != self._log_run_id:
            self._log_run_id = None  # its output may have grown while another run was shown: read it again
        if self.yaml_overlay.isVisible():
            self._refresh_run_yaml()

    def update_run_row(self, run):
        """A run's metrics or status changed: refresh its sidebar row."""
        self.sidebar.update_run(run)

    def show_hardware(self, hw):
        """Keep the System navigation label stable as hardware samples arrive."""
        button = self.view_buttons["System"]
        button.setText("System")
        button.setToolTip("View CPU, memory and GPU memory over time")

    def _switch_to(self, run_id):
        run = self.host.by_id(run_id)
        if run is not None and run is not self.host.selected:
            self.host.select_run(run)

    def render(self):
        host = self.host
        run = self._current_run()
        if not run:
            self.refresh_live()
            return
        self._refresh_run_selectors()
        config = run["config"]
        live = not run["simulated"]
        title = run_display_name(run)
        group = (run.get("backend") or {}).get("group")
        self.run_title.setText(f"{group} / {title}" if group else title)
        if live:
            where = f"results/logs/{run['backend']['group']}/{run['backend']['experiment_id']}/"
            stored = f"{where}  ·  results/jobs/jobs.db"
        else:
            where, stored = "", "Simulated demo run (in-memory)  ·  results/jobs/jobs.db"
        about = f"{run_setup_label(run, config)}\n{stored}"
        self.run_title.setToolTip(about)
        self.run_caption.setToolTip(about)

        note = None  # only what the run is waiting for: where its files are is on the title's tooltip
        if live:
            if run["status"] == "queued":
                position = run.get("queue_position")
                ahead = f"position {position + 1}" if position is not None else "waiting"
                note = (f"In the job queue ({ahead}). Queued experiments wait for an available slot; "
                        "this view follows it when it starts.")
                if not host.queue_running:
                    note += " The queue is paused: use Start or pause queue (Ctrl+Shift+R)."
            elif run["status"] == "running" and run.get("phase") == "plotting":
                note = f"Training finished — the pipeline is generating plots in results/plots/{run['backend']['group']}/…"
            elif run["status"] == "incomplete":
                note = (f"Metrics from {where} · this run was started outside the app and has no recorded end "
                        "time: it may still be training, or it stopped early.")
        self.status_note = note
        self.sidebar.update_run(run)
        self.live_changed()

    # ── Live views ───────────────────────────────────────────────────────────

    def live_changed(self):
        """New records (or another run selected): redraw the live views soon, once for many changes."""
        if not self._refresh_timer.isActive():
            self._refresh_timer.start()

    def _run_is_live(self, run) -> bool:
        return bool(run) and not run.get("simulated") and run.get("status") in LIVE_STATUSES

    def refresh_live(self):
        """Draw the status strip and the open live view for the selected run."""
        run = self._current_run()
        if not run:
            self._refresh_run_selectors()
            self.run_title.setText("No live training")
            self.method_legend.clear()
            self.method_legend.hide()
            self.method_button.hide()
            self.trial_button.hide()
            self.problem_bar.hide()
            self.focused_agent = self.pipeline_strip.focused = None
            self.pipeline_strip.segments = []
            self.status_strip.show_run("", {}, None, False)
            self.charts_view.show_lives({})
            self.system_view.show_lives({})
            self.feed_view.show_lives({})
            self.files_view.show_lives({}, None)
            self.log_view.set_lines([], "Completed training is available in Results.")
            self.sweep_view.set_studies([])
            self.sweep_view.set_lives({})
            self._log_run_id = None
            return
        from .sweep_view import study_summaries
        jobs = getattr(self.host, "jobs", None)
        project_root = getattr(jobs, "project_root", None)
        lookup = getattr(jobs, "job", None)
        source = getattr(jobs, "_sweep_summaries", None)
        active_job = lookup((run.get("backend") or {}).get("job_id")) if callable(lookup) else None
        studies = (source(active_job) if active_job is not None and callable(source) else
                   study_summaries(project_root, run) if project_root is not None else [])
        studies = [(method, study) for method, study in studies
                   if study.get("stop_reason", "running") == "running"]
        dashboard_studies = studies
        view = self.views.currentWidget()
        lives = self.host.live_for_run(run, load=True)  # the Overview charts every metric
        if run.get("simulated"):
            from ..live_run import LiveRun
            demo = LiveRun()
            demo.rows = [{("eval/reward" if k == "reward" else "eval/reward_std" if k == "reward_std" else k): v
                          for k, v in point.items() if v is not None} for point in run.get("metrics", [])]
            lives = {run_display_name(run): demo}
        live = self._run_is_live(run)
        trial_states = {
            f"{method} · {study['study']} · trial {trial['id']}": trial.get("state")
            for method, study in studies for trial in study.get("trials", [])
        }
        lives = {name: item for name, item in lives.items()
                 if (item.start or item.rows) and not item.is_finished()
                 and trial_states.get(parent_trial_label(name), "running") == "running"}
        # Optuna also calls a sampled, resource-waiting trial "running".
        # Only a worker start or its actual telemetry admits it to this monitor.
        studies = [(method, {**study, "trials": [
            trial for trial in study.get("trials", [])
            if trial.get("state") == "running" and (
                trial.get("execution_started")
                or any(parent_trial_label(name) == f"{method} · {study['study']} · trial {trial['id']}" for name in lives))]})
            for method, study in studies]
        studies = [(method, study) for method, study in studies if study["trials"]]
        if view is self.sweep_view:
            active_keys = {(method, study['study']) for method, study in studies}
            dashboard_studies = [(method, study) for method, study in dashboard_studies
                                 if (method, study['study']) in active_keys]
            self.sweep_view.set_studies(dashboard_studies)
            if self._focused_sweep:
                for index, (method, study) in enumerate(dashboard_studies):
                    if f"{method}:{study['study']}" == self._focused_sweep:
                        self.sweep_view.selector.setCurrentIndex(index)
                        break
            self.sweep_view.set_lives(lives)
        if self.focused_agent and (run["id"] != self._focus_run_id or self.focused_agent not in lives):
            self.focused_agent = self.pipeline_strip.focused = None
        self.status_strip.show_run(run.get("status", ""), lives, None, live)
        methods = list(lives)
        if not self.focused_agent:
            self.focused_agent = methods[0] if methods else None
            self._focus_run_id = run["id"]
        menu = self.method_button.menu()
        trial_menu = self.trial_button.menu()
        if not menu.isVisible():
            menu.clear()
        if not trial_menu.isVisible():
            trial_menu.clear()
        sweep_methods = {}
        for method, study in studies:
            sweep_methods.setdefault(method, []).append(study)
        trial_methods = {
            f"{method} · {study['study']} · trial {trial['id']}": method
            for method, study in studies for trial in study.get("trials", [])
        }
        selected_method = trial_methods.get(parent_trial_label(self.focused_agent or ""), self.focused_agent or "")
        selected_study = None
        if self._focused_sweep and view is self.sweep_view:
            for method, study in studies:
                if f"{method}:{study['study']}" == self._focused_sweep:
                    selected_method, selected_study = method, study
                    break
        method_names = list(dict.fromkeys(
            [trial_methods.get(parent_trial_label(name), name) for name in methods] + list(sweep_methods)))
        if selected_method not in method_names:
            selected_method = method_names[0] if method_names else ""
        for name in (() if menu.isVisible() else method_names):
            action = menu.addAction(str(name))
            action.setCheckable(True)
            action.setChecked(name == selected_method)
            if name in sweep_methods:
                study = sweep_methods[name][0]
                key = f"{name}:{study['study']}"
                action.triggered.connect(lambda checked=False, k=key: self.focus_sweep(k))
            else:
                action.triggered.connect(lambda checked=False, n=name: self._open_sweep_trial(n))
        method_studies = sweep_methods.get(selected_method, [])
        for study in (() if trial_menu.isVisible() else method_studies):
            key = f"{selected_method}:{study['study']}"
            action = trial_menu.addAction("Sweep")
            action.setCheckable(True)
            action.setChecked(study is selected_study)
            action.triggered.connect(lambda checked=False, k=key: self.focus_sweep(k))
            for trial in study.get("trials", []):
                name = f"{selected_method} · {study['study']} · trial {trial['id']}"
                if not live or trial.get("state") != "running":
                    continue
                for source in lives:
                    if parent_trial_label(source) != name or lives[source].finish:
                        continue
                    suffix = " / Seed " + source.split(" · seed ", 1)[1] if " · seed " in source else ""
                    action = trial_menu.addAction(f"Trial {trial['id']}" + suffix)
                    action.setCheckable(True)
                    action.setChecked(source == self.focused_agent and selected_study is None)
                    action.triggered.connect(lambda checked=False, n=source: self._open_sweep_trial(n))
        trial_label = "Sweep" if selected_study is not None else (
            "Trial " + self.focused_agent.split(" · trial ", 1)[1]
            if self.focused_agent and " · trial " in self.focused_agent else "Sweep")
        self.method_button.setText(f" / {selected_method}" if selected_method else "")
        self.method_button.setVisible(bool(method_names))
        self.trial_button.setText(f" / {trial_label}")
        self.trial_button.setVisible(bool(method_studies))
        shown = self.shown_lives(lives)
        colors = {name: PALETTE[i % len(PALETTE)] for i, name in enumerate(methods)}
        self.method_legend.setText(" &nbsp; ".join(
            f'<span style="color:{theme_color(colors[name])}">●</span> {escape(monitor_method_label(name))}'
            for name in shown
        ))
        self.method_legend.setVisible(bool(shown) and view is not self.sweep_view)
        self.pipeline_strip.segments = pipeline_segments(list((run.get("backend") or {}).get("agents") or []), lives)
        # The bottom bar: the newest warning or error (with the log), else what the run is waiting for.
        problem = latest_problem(lives)
        if problem is not None:
            agent, event = problem
            mark = "✖" if event.get("level") == "error" else "⚠"
            self.problem_text.setText(f"{mark}  {agent}: {event.get('message', '')}")
        elif self.status_note:
            self.problem_text.setText(f"●  {self.status_note}")
        self.open_log.setVisible(problem is not None)
        self.problem_bar.setVisible(problem is not None or bool(self.status_note))
        if view is self.charts_view:
            view.show_lives(shown, colors, live, comparing=False)
        elif view is self.system_view:
            view.show_lives(shown, colors)
        elif view is self.feed_view:
            self.feed_view.show_lives(lives)
        elif view is self.files_view:
            self.files_view.show_lives(lives, self.host.run_folder(run))
        elif view is self.log_view and self._log_run_id != run["id"]:
            self._log_run_id = run["id"]
            lines, empty = self.host.log_lines_for_run(run)
            self.log_view.set_lines(lines, empty)

    def shown_lives(self, lives):
        """The shown run's agents, with the focused one drawn last.

        Runs ticked in the runs list are not drawn here: they are compared in the Plots tab.
        """
        if self.focused_agent and " · trial " in self.focused_agent:
            return {self.focused_agent: lives[self.focused_agent]} if self.focused_agent in lives else {}
        return {name: lives[name] for name in lives if name != self.focused_agent} | (
            {self.focused_agent: lives[self.focused_agent]} if self.focused_agent in lives else {})

    def focus_agent(self, agent: str):
        """Draw the selected method over its siblings."""
        self.focused_agent = agent or None
        self._focused_sweep = None
        self._focus_run_id = (self._current_run() or {}).get("id")
        self.refresh_live()

    def focus_sweep(self, key: str):
        """Open the dashboard for the selected sweep job."""
        self._focused_sweep = key
        self._focus_run_id = (self._current_run() or {}).get("id")
        self.show_view("Sweep")
        self.refresh_live()

    def _open_sweep_trial(self, name):
        lookup = getattr(self.host, "live_for_run", None)
        if callable(lookup):
            lives = lookup(self._current_run(), load=True)
            if name not in lives:
                name = next((source for source in lives if parent_trial_label(source) == name), name)
        self.focus_agent(name)
        self.show_view("Overview")

    def append_log(self, run, lines):
        """New output of `run`: shown at once if its log is the one loaded (otherwise read when opened)."""
        if self._log_run_id == run.get("id"):
            self.log_view.append(lines)

    def _tick_status(self):
        run = self._current_run()
        if self.isVisible():
            self.refresh_live()

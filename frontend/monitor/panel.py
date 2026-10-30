"""Training monitor panel: live and historic curves for the selected run.

The panel owns its widgets and drawing. Run state (the run list, the selected and active runs,
the pinned baseline, the queue) stays on the host window, which the rest of the app also reads.
"""
from __future__ import annotations

from typing import TYPE_CHECKING
from html import escape

from PyQt6.QtCore import QEvent, Qt, QTimer
from PyQt6.QtWidgets import (
    QFrame,
    QApplication,
    QPlainTextEdit,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ..charts import Chart
from ..model import LIVE_STATUSES, METRIC_COLUMNS, available_metrics, metric_points
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
OVERVIEW_REWARD_HEIGHT = 300  # px: the Overview's charts never shrink below these; the tab scrolls instead
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
    return run["config"]["name"] if run["simulated"] else run["backend"]["experiment_id"]


class MonitorPanel(QWidget):
    """The Training monitor tab."""

    # Widgets the host window also exposes under the same names (existing code and tests use them).
    EXPOSED = (
        "smoothing_label", "smoothing_slider", "btn_pin_baseline", "run_title", "run_caption", "reward_chart",
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
        title_layout.addWidget(self.run_title)
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
                                        lives_for_job=self._queue_lives_for_job)
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
        self._focus_run_id: str | None = None  # the run whose method is focused

        # Sections of charts, led by the run’s headline metric.
        self.reward_chart = Chart("reward", "Episode reward  ·  mean ± 1 std over evaluation episodes",
                                  band="reward_std")
        self.reward_chart.setMinimumHeight(OVERVIEW_REWARD_HEIGHT)
        self.reward_card_frame = QFrame()  # framed like the charts beside it
        self.reward_card_frame.setObjectName("resultsChartCard")
        frame_layout = QVBoxLayout(self.reward_card_frame)
        frame_layout.setContentsMargins(0, 0, 0, 0)
        frame_layout.addWidget(self.reward_chart)
        # Smoothing and the baseline pin sit on the first section's heading row.
        toolbar = QWidget()
        tools = QHBoxLayout(toolbar)
        tools.setContentsMargins(0, 0, 0, 0)
        tools.setSpacing(10)
        self.smoothing_label = label("Smoothing: 0%", "muted")
        tools.addWidget(self.smoothing_label)
        self.smoothing_slider = QSlider(Qt.Orientation.Horizontal)
        self.smoothing_slider.setRange(0, 95)
        self.smoothing_slider.setValue(0)
        self.smoothing_slider.setFixedWidth(110)
        self.smoothing_slider.setToolTip("Exponential moving average smoothing for training curves (0% = raw)")
        self.smoothing_slider.valueChanged.connect(self.on_smoothing_changed)
        tools.addWidget(self.smoothing_slider)
        self.btn_pin_baseline = QPushButton("Pin as baseline")
        self.btn_pin_baseline.setCheckable(True)
        self.btn_pin_baseline.setToolTip("Pin this run to overlay as a dashed baseline curve on other runs")
        self.btn_pin_baseline.clicked.connect(self.toggle_pin_baseline)
        tools.addWidget(self.btn_pin_baseline)
        self.charts_view = ChartsView(lead=("eval", self.reward_card_frame), toolbar=toolbar,
                                      group_columns=SECTION_COLUMNS, show_caption=False)
        self.overview_scroll = self.charts_view.grid
        self.overview_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.overview_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.system_view = SystemView()
        self.log_view = LogView()
        self.feed_view = FeedView()
        self.files_view = FilesView()
        self.views = QStackedWidget()
        self._views = {"Overview": self.charts_view, "System": self.system_view, "Logs": self.log_view,
                       "Feed": self.feed_view, "Files": self.files_view}
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
        width = min(850, max(1, self.content.width() - 20))
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
        if event.type() == QEvent.Type.MouseButtonPress:
            widget = watched if isinstance(watched, QWidget) else None
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
        for job in [*data.get("active", []), *data.get("queued", [])]:
            job = dict(job)
            run = self.host.run_for_job(job.get("job_id"))
            job["description"] = run.get("description", "") if run else ""
            jobs.append(job)
        self.queue_preview.set_data({"active": jobs})
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

    def show_selected(self, run):
        """`run` is now the one shown here."""
        if run and run.get("id") != self._log_run_id:
            self._log_run_id = None  # its output may have grown while another run was shown: read it again
        if self.yaml_overlay.isVisible():
            self._refresh_run_yaml()

    def update_run_row(self, run):
        """A run's metrics or status changed: refresh its sidebar row."""
        self.sidebar.update_run(run)

    def toggle_pin_baseline(self):
        host = self.host
        if not host.selected:
            return
        host.pinned_baseline_run = None if host.pinned_baseline_run is host.selected else host.selected
        self.render()

    # ── Charts ───────────────────────────────────────────────────────────────

    def on_smoothing_changed(self, value):
        self.smoothing_label.setText(f"Smoothing: {value}%")
        factor = value / 100.0
        self.reward_chart.set_smoothing(factor)
        self.charts_view.set_smoothing(factor)

    def show_hardware(self, hw):
        """Live CPU and memory use of the training process (and its children, when reported), on the System button."""
        children = int(hw.get("children") or 0)
        processes = f" across {children + 1} processes" if children else ""
        parts = [f"CPU {hw['cpu_percent']:.1f}%{processes}", f"Process RAM {hw.get('memory_mb', 0):.0f} MB"]
        gpu = sum(v for k, v in hw.items() if k.startswith("gpu") and k.endswith("_mem_allocated_mb"))
        if gpu:
            parts.append(f"GPU memory {gpu:.0f} MB")
        if hw.get("host_cpu_percent") is not None:
            parts.append(f"Machine CPU {hw['host_cpu_percent']:.0f}%")
        button = self.view_buttons["System"]
        button.setText(f"System · CPU {hw['cpu_percent']:.0f}%")
        button.setToolTip("  •  ".join(parts) + "\nClick for CPU, memory and GPU memory over time")


    def _switch_to(self, run_id):
        run = self.host.by_id(run_id)
        if run is not None and run is not self.host.selected:
            self.host.select_run(run)

    def render(self):
        host = self.host
        run = host.selected
        if not run:
            return
        config = run["config"]
        metrics = run["metrics"]
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

        # Update Pin Baseline button
        baseline = host.pinned_baseline_run
        if baseline is None:
            self.btn_pin_baseline.setChecked(False)
            self.btn_pin_baseline.setText("Pin as baseline")
            self.btn_pin_baseline.setToolTip("Pin this run's curve to overlay as a dashed baseline when inspecting other runs")
        elif baseline is run:
            self.btn_pin_baseline.setChecked(True)
            self.btn_pin_baseline.setText("Pinned baseline")
            self.btn_pin_baseline.setToolTip("This run is currently pinned as the baseline. Click to unpin.")
        else:
            self.btn_pin_baseline.setChecked(False)
            p_name = run_display_name(baseline)
            label_name = (p_name[:12] + "…") if len(p_name) > 12 else p_name
            self.btn_pin_baseline.setText(f"Replace baseline ({label_name})")
            self.btn_pin_baseline.setToolTip(f"Baseline '{p_name}' is pinned. Click to replace it with this run.")

        step = max((m["step"] for m in metrics), default=0)
        requested = config["total_timesteps"]
        # Records from older runs may not include an effective budget; use the steps actually run.
        budget = (run["backend"].get("effective_timesteps") or max(step, requested)) if live else requested
        head_key, head_info = headline_metric(run, config)
        # The headline chart leads its metric's section and draws that metric, so the grid leaves it out.
        head_column = METRIC_COLUMNS.get(head_key, head_key)
        self.charts_view.exclude = {head_column} | ({"eval/reward_std"} if head_key == "reward" else set())
        lead_group = metric_group(head_column)
        if self.charts_view.grid.lead is None or self.charts_view.grid.lead[0] != lead_group:
            self.charts_view.grid.lead = (lead_group, self.reward_card_frame)
        env_steps = measures_environment_steps(run)
        note = None  # only what the run is waiting for: where its files are is on the title's tooltip
        if live:
            if run["status"] == "queued":
                position = run.get("queue_position")
                ahead = f"position {position + 1}" if position is not None else "waiting"
                note = (f"In the job queue ({ahead}). Queued experiments train one at a time; "
                        "this view follows it when it starts.")
                if not host.queue_running:
                    note += " The queue is paused: use Start or pause queue (Ctrl+Shift+R)."
            elif run["status"] == "starting" or (run["status"] == "running" and not metrics):
                note = "Starting the pipeline — waiting for the first metrics."
            elif run["status"] == "running" and run.get("phase") == "plotting":
                note = f"Training finished — the pipeline is generating plots in results/plots/{run['backend']['group']}/…"
            elif run["status"] == "incomplete":
                note = (f"Metrics from {where} · this run was started outside the app and has no recorded end "
                        "time: it may still be training, or it stopped early.")
        self.status_note = note
        # Fit the data when the budget is not what this run is measured against,
        # or the curve lands in the leftmost few percent of the chart.
        xmax = budget if (live and env_steps) else None
        has_spread = head_key == "reward" and any(m.get("reward_std") is not None for m in metrics)
        self.reward_chart.set_metric(
            head_key,
            "Episode reward  ·  mean ± 1 std over evaluation episodes" if has_spread else head_info.label,
        )
        # Drawn like the charts beside it: while the run trains, its y-axis only grows and its newest point is marked.
        training = self._run_is_live(run)
        self.reward_chart.hold_y = training
        self.reward_chart.show_future = training and xmax is not None
        self.reward_chart.live_series = {title} if training else set()
        self.reward_chart.set_series([(title, metrics, "#b8bb26")], xmax)

        # Baseline overlay
        if baseline and baseline is not run:
            b_title = f"{run_display_name(baseline)} (baseline)"
            self.reward_chart.set_baseline_series([(b_title, baseline["metrics"], "#d3869b")])
        else:
            self.reward_chart.set_baseline_series([])
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
        run = self.host.selected
        if not run:
            return
        view = self.views.currentWidget()
        lives = self.host.live_for_run(run, load=True)  # the Overview charts every metric, so it reads them all
        live = self._run_is_live(run)
        if self.focused_agent and run["id"] != self._focus_run_id:  # another run: its methods are others
            self.focused_agent = self.pipeline_strip.focused = None
        self.status_strip.show_run(run.get("status", ""), lives, None, live)
        methods = list(dict.fromkeys([*((run.get("backend") or {}).get("agents") or []), *lives]))
        colors = {name: PALETTE[i % len(PALETTE)] for i, name in enumerate(methods)}
        self.method_legend.setText(" &nbsp; ".join(
            f'<span style="color:{theme_color(colors[name])}">●</span> {escape(str(name))}' for name in methods
        ))
        self.method_legend.setVisible(bool(methods))
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
            shown = self.shown_lives(lives)
            head_key = self.reward_chart.metric
            series = [(name, metric_points(method.rows), colors[name]) for name, method in shown.items()]
            series = [entry for entry in series if any(point.get(head_key) is not None for point in entry[1])]
            if series:
                self.reward_chart.live_series = {name for name, method in shown.items()
                                                 if live and not method.is_finished()}
                self.reward_chart.set_series(series, self.reward_chart.xmax)
            elif lives:
                self.reward_chart.set_series([], self.reward_chart.xmax)
            view.show_lives(shown, colors, live, comparing=False)
        elif view is self.system_view:
            view.show_lives(self.shown_lives(lives), colors)
        elif view is self.feed_view:
            self.feed_view.show_lives(lives)
        elif view is self.files_view:
            self.files_view.show_lives(lives, self.host.run_folder(run))
        elif view is self.log_view and self._log_run_id != run["id"]:
            self._log_run_id = run["id"]
            lines, empty = self.host.log_lines_for_run(run)
            self.log_view.set_lines(lines, empty)

    def shown_lives(self, lives):
        """The shown run's agents, or only the one whose pipeline-strip segment was clicked.

        Runs ticked in the runs list are not drawn here: they are compared in the Plots tab.
        """
        if self.focused_agent in lives:
            return {self.focused_agent: lives[self.focused_agent]}
        return lives

    def focus_agent(self, agent: str):
        """Show only `agent`'s curves of the selected run ("" shows every agent again)."""
        self.focused_agent = agent or None
        self._focus_run_id = (self.host.selected or {}).get("id")
        self.refresh_live()

    def append_log(self, run, lines):
        """New output of `run`: shown at once if its log is the one loaded (otherwise read when opened)."""
        if self._log_run_id == run.get("id"):
            self.log_view.append(lines)

    def _tick_status(self):
        run = self.host.selected
        if self.isVisible() and self._run_is_live(run):
            self.status_strip.show_run(run.get("status", ""), self.host.live_for_run(run, load=False), None, True)

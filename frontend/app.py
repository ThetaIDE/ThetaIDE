"""Spyder-inspired workspace for configuring and analyzing machine-learning experiments."""
import argparse
import base64
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

if sys.platform == "darwin" and "QTWEBENGINE_CHROMIUM_FLAGS" not in os.environ:
    os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = "--disable-gpu"

from PyQt6.QtCore import QByteArray, Qt, QTimer, QUrl
from PyQt6.QtGui import QAction, QDesktopServices, QFont, QIcon, QKeySequence
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QDockWidget,
    QHBoxLayout,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QStyle,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from src.app.services import PROJECT_ROOT
from src.app.services.config_service import LaunchError

from . import wheel_guard
from .about import AboutDialog
from .app_icon import (
    DESKTOP_FILE_NAME,
    ICON_PATH,
    app_icon,
    install_desktop_entry,
    register_windows_app_identity,
    set_windows_app_id,
)
from .components_panel import ComponentsPanel
from .config_model import config_fingerprint
from .config_tab_manager import ConfigTabManager
from .config_tree import ConfigTreeWidget
from .config_viewer import ConfigViewer
from .export_dialog import ExportExperimentDialog
from .hub import HubClient, HubDialog
from .job_manager import JobManager
from .live_run import LiveRun, LiveRunCache
from .local_backend import LocalBackend
from .model import (
    BASE_EXPERIMENT,
    FINAL_STATUSES,
    LIVE_STATUSES,
    Config,
    Store,
    disk_run,
    metric_points,
    metric_points_from_api,
    new_run,
    sample,
)
from .monitor import MonitorPanel
from .plots import PlotViewer
from .plugins import CommandRegistry, ExtensionSlotRegistry, PluginManager
from .queue_panel import QueuePanel
from .results.panel import ResultsPanel
from .run_notifier import RunNotifier
from .settings import SettingsManager
from .sidetabs import SideTabs, svg_icon
from .terminal import TerminalPanel
from .theme import STYLE, ThemeManager, theme_color
from .theme_builder import ThemeBuilder
from .titlebar import apply_title_bar
from .widgets import ToggleSlider, YamlHighlighter, label
from .workflows import WorkflowsPanel

REPO_ROOT = Path(__file__).resolve().parent.parent  # holds venv/, in/, results/
PERSIST_SECONDS = 5.0  # how often a training run's growing record is saved to the run store

# Metrics the monitor's second chart can show: key -> (card title, chart title, subtitle, value format)


# Panes the default and restored sidebar leave hidden. Workflows can draw a pipeline but not run it yet.
HIDDEN_BY_DEFAULT = {"workflows"}

def normalize_menu_shortcut(shortcut_str: str) -> str:
    """Normalize user-entered shortcut string for Qt QKeySequence.

    Replaces 'cmd' and 'command' with 'Ctrl' so that macOS command shortcuts
    are properly parsed by QKeySequence on all platforms.
    """
    if not shortcut_str:
        return ""
    import re
    return re.sub(r"(?i)\b(cmd|command)\b", "Ctrl", str(shortcut_str).strip())


def launch_buttons(window):
    """Buttons that launch the open experiment now: the experiment tree's and the Queue tab's."""
    panel = getattr(window, "queue_panel", None)
    return [b for b in (window.start_button, panel and panel.launch_button) if b is not None]


class Window(QMainWindow):
    def __init__(self, data_dir, progress=None):
        """`progress(message, fraction)` reports startup steps (the splash screen shows them)."""
        step = progress or (lambda message, fraction: None)
        super().__init__()
        self.setWindowTitle("ThetaIDE")
        self.resize(1480, 940)
        self.setMinimumSize(1080, 720)
        self.setDockNestingEnabled(True)
        self.store = Store(data_dir)

        # --- Unified settings (must be first; everything else reads from it) ---
        step("Loading settings…", 0.02)
        self.settings_manager = SettingsManager(Path(data_dir))
        self.settings_manager.changed.connect(self._on_settings_changed)
        self.zoom_factor = self.settings_manager.zoom
        self.zoom_slider = None
        self.zoom_percent_label = None

        self.theme_manager = ThemeManager(
            Path(data_dir) / ".appearance.json",
            self,
            initial_theme=self.settings_manager.theme,
            initial_zoom=self.zoom_factor,
        )
        self.pane_sliders = {}
        self.plugin_sliders = {}
        data_p = Path(data_dir).resolve()
        if data_p.name == "runs" and data_p.parent.name == ".thetaide":
            workspace_dir = data_p.parent.parent
            ide_data_dir = data_p.parent
        elif data_p.name == ".thetaide":
            workspace_dir = data_p.parent
            ide_data_dir = data_p
        else:
            workspace_dir = data_p.parent
            ide_data_dir = data_p

        self.extension_slots = ExtensionSlotRegistry(parent=self)
        self.command_registry = CommandRegistry(parent=self)

        step("Discovering plugins…", 0.04)
        self.plugin_manager = PluginManager(
            self,
            [Path(__file__).parent / "plugins" / "core", Path(__file__).parent / "plugins", ide_data_dir / "plugins"],
            settings_manager=self.settings_manager,
        )
        self.plugin_manager.discover()
        self.hub_client = HubClient(
            workspace_dir=workspace_dir,
            data_dir=ide_data_dir,
            on_change_callback=self._on_hub_component_changed,
            settings_manager=self.settings_manager,
        )
        step("Loading run history…", 0.06)
        saved, errors = self.store.load()
        self.runs = saved
        # catalog.list_runs() entries by (group, experiment_id): every results/logs folder, however it was launched.
        self.disk_entries = {}
        self.backend_connected = False  # True once the training pipeline imports and composes in-process
        self.active_runs: list[dict] = []  # runs training now (several when the queue runs N at once)
        self.selected = None
        self.pinned_baseline_run = None
        self.timer = QTimer(self)
        self.timer.setInterval(350)
        self.timer.timeout.connect(self.tick)
        # Configs compose in-process and training runs as a QProcess: no server to start.
        self.backend = LocalBackend(self)
        self.jobs = JobManager(store_path=Path(data_dir) / "jobs" / "jobs.db", parent=self)
        self.jobs.job_changed.connect(self.on_job_changed)
        self.jobs.max_parallel = max(1, int(self.settings_manager.get("queue", "max_parallel", default=1) or 1))
        self.jobs.job_output.connect(self.on_job_output)
        self.jobs.job_metrics.connect(self.on_job_metrics)
        self.jobs.job_records.connect(self.on_job_records)
        # (job_id, agent) -> LiveRun: every record a training run reported, with its speed, ETA and events
        self.live_runs: dict[tuple[str, str], LiveRun] = {}
        self._persisted_at: dict[str, float] = {}  # job_id -> when its run was last saved while training
        self.disk_lives = LiveRunCache()  # LiveRuns of runs not followed in this session, read from results/logs
        self.jobs.job_hardware.connect(self.on_job_hardware)
        self.jobs.queue_changed.connect(self.on_queue_changed)
        self.tray = None  # TrayController, set up by main(): closing while training hides to the tray
        self.run_notifier = RunNotifier(self)  # toast or system notification and a chime when a run ends
        self._quitting = False  # an explicit Quit (menu, tray, reload) closes for real
        self._shutting_down = False  # set once closeEvent commits to closing
        self.schema = None
        self.schema_pending = False
        self.compose_serial = 0
        self.compose_timer = QTimer(self)
        self.compose_timer.setSingleShot(True)
        self.compose_timer.setInterval(250)
        self.compose_timer.timeout.connect(self.request_compose)
        self.compose_valid = False
        self.compose_error = None
        self.issued_ids = set()
        self.queue_running = False
        self.active_config_path = None
        self.active_config_rel_path = None
        self.current_experiment = None
        # Created before anything that talks to the backend: its replies arrive on the event loop, which
        # the splash progress steps below spin, so a reply can land before the rest of the window exists.
        self.state_label = label("Idle", "muted")
        self.backend_label = label("", "muted")
        for status in (self.state_label, self.backend_label):
            status.setContentsMargins(12, 0, 12, 0)
        self.pipeline_state = "loading"  # "loading", "ready" or "unavailable"
        self.show_pipeline_state()
        self.fixed_fields = {}
        self.fields = {}
        step("Building the workspace…", 0.08)
        self.make_center()
        step("Starting plugins…", 0.56)
        self.plugin_manager.initialize_plugins()
        step("Loading the default experiment…", 0.58)
        self.init_default_experiment()
        self.tabs.tabOrderChanged.connect(lambda _: self.save_layout())
        step("Restoring layout…", 0.66)
        self.load_layout()
        self.restore_session()
        if hasattr(self, "settings_manager"):
            self.tabs.set_auto_hide(bool(self.settings_manager.get("sidebar", "auto_hide", default=False)))
        self.tabs.autoHideChanged.connect(self.on_sidebar_auto_hide_changed)
        self.make_menus()
        self.statusBar().setSizeGripEnabled(False)
        self.theme_manager.changed.connect(self.theme_changed)
        self.theme_manager.committed.connect(self.persist_theme_choice)
        step("Applying theme…", 0.68)
        self.theme_changed()
        self.statusBar().addPermanentWidget(self.state_label)
        self.statusBar().addPermanentWidget(self.backend_label)
        step("Loading runs…", 0.79)
        self.refresh_runs()
        self.update_config()
        if self.runs:
            self.select_run(self.runs[0])
        self.log("ThetaIDE ready. Launch trains on this machine with run_pipeline.py; "
                 "Queue → Start simulated demo needs no training environment.")
        self.request_schema(apply_defaults=True)
        self.reconnect_live_run()
        for error in errors:
            self.log(error)
        if self.theme_manager.error:
            self.log(self.theme_manager.error)
        step("Finishing up…", 0.8)
        self.set_zoom(self.zoom_factor, persist=False)
        self.default_layout = self.saveState()

        # --- Hotkey and Action-key Navigation Manager ---
        from .hotkeys import HotkeyManager
        self.hotkey_manager = HotkeyManager(self, self.settings_manager)

    def button(self, text, callback, primary=False):
        button = QPushButton(text)
        if primary:
            button.setObjectName("primary")
        button.clicked.connect(callback)
        return button

    def make_center(self):
        self.tabs = SideTabs()  # its gear opens Settings
        self.setCentralWidget(self.tabs)

        # 1. Components Panel
        self.components_panel = ComponentsPanel(hub_client=self.hub_client, log_fn=self.log, parent=self)
        self.tabs.addTab(self.components_panel, "Components", "components", "Components", tab_id="components")

        # 2. Experiment (Config) Panel
        config_panel = QWidget()
        config_layout = QVBoxLayout(config_panel)
        config_layout.setContentsMargins(18, 14, 18, 14)
        config_layout.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.config_splitter = splitter

        # 1. Config Tree (mirrors in/config/experiment/) - extends to top!
        self.config_tree = ConfigTreeWidget(mode="experiments", settings_manager=getattr(self, "settings_manager", None))
        self.config_tree.setMinimumWidth(350)
        self.tree = self.config_tree.tree  # backwards compatibility alias
        self.config_tree.file_selected.connect(self.on_config_file_selected)
        self.config_tree.file_opened.connect(self.on_config_file_opened)
        self.config_tree.file_deleted.connect(self.on_config_file_deleted)
        self.config_tree.dir_deleted.connect(self.on_config_dir_deleted)
        splitter.addWidget(self.config_tree)

        # Square buttons located at the top of the filetree next to 'New' and 'Copy'
        self.btn_save = self.config_tree.btn_save
        self.start_button = self.config_tree.btn_launch
        self.btn_launch = self.config_tree.btn_launch
        self.btn_export = self.config_tree.btn_export
        self.btn_toggle_yaml = self.config_tree.btn_toggle_yaml
        self.queue_button = None
        self.stop_button = None

        if self.start_button is not None:
            self.start_button.clicked.connect(self.launch_training)
        if self.btn_save is not None:
            self.btn_save.clicked.connect(self.save_current_config)
        if self.btn_export is not None:
            self.btn_export.clicked.connect(self.export_config)
        if self.btn_toggle_yaml is not None:
            self.btn_toggle_yaml.clicked.connect(lambda: self.toggle_yaml_preview())

        # 2. Right Side Section (contains viewer / preview)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        preview_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.preview_splitter = preview_splitter

        # Multi-tabbed Config Tab Manager (replaces single ConfigViewer)
        self.config_tab_manager = ConfigTabManager(self.settings_manager, self)
        self.config_tab_manager.auto_save = True
        self.config_tab_manager.btn_save = self.btn_save
        self.config_tab_manager.config_changed.connect(self.update_config)
        self.config_tab_manager.save_requested.connect(self.on_config_saved)
        self.config_tab_manager.dirty_state_changed.connect(self.config_tree.set_file_dirty)
        self.config_tab_manager.file_renamed.connect(self.on_config_file_renamed)
        self.config_tab_manager.current_tab_changed.connect(self.on_config_tab_switched)
        self.config_tree.dir_renamed.connect(self.config_tab_manager.on_dir_renamed)
        preview_splitter.addWidget(self.config_tab_manager)

        # Preview Panel (Hydra YAML - hidden by default!)
        preview_panel = QWidget()
        preview_layout = QVBoxLayout(preview_panel)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.setSpacing(6)

        prev_header = QHBoxLayout()
        prev_header.setSpacing(8)
        prev_header.addWidget(label("RESOLVED HYDRA YAML", "eyebrow"))
        prev_header.addStretch()
        btn_close_prev = QToolButton()
        btn_close_prev.setText("✕")
        btn_close_prev.setToolTip("Close preview and expand config viewer")
        btn_close_prev.clicked.connect(lambda: self.toggle_yaml_preview(False))
        prev_header.addWidget(btn_close_prev)
        preview_layout.addLayout(prev_header)

        self.preview_status = label("Resolved config • waiting for backend", "muted")
        preview_layout.addWidget(self.preview_status)
        self.preview_errors = label("", "configError")
        self.preview_errors.setWordWrap(True)
        self.preview_errors.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.preview_errors.hide()
        preview_layout.addWidget(self.preview_errors)
        self.preview = QPlainTextEdit()
        self.preview.setReadOnly(True)
        self.highlighter = YamlHighlighter(self.preview.document())
        preview_layout.addWidget(self.preview, 1)

        self.builder_status = label("", "muted")
        self.builder_status.setWordWrap(True)
        preview_layout.addWidget(self.builder_status)

        self.preview_panel = preview_panel
        self.preview_panel.hide()  # Hidden by default!
        preview_splitter.addWidget(preview_panel)

        preview_splitter.setSizes([1000, 0])
        right_layout.addWidget(preview_splitter, 1)

        splitter.addWidget(right_panel)
        splitter.setSizes([350, 1000])
        config_layout.addWidget(splitter, 1)
        self.config_panel = config_panel
        self.tabs.addTab(self.config_panel, "Experiment", "config", "Experiment", tab_id="config")

        # 3. Workflows Panel (String Diagrams)
        self.workflows_panel = WorkflowsPanel(log_fn=self.log, parent=self)
        self.tabs.addTab(self.workflows_panel, "Workflows", "workflow", "Workflows", tab_id="workflows")

        # 4. Training Monitor Panel
        self.monitor = MonitorPanel(self)
        for name in MonitorPanel.EXPOSED:  # existing code and tests reach these through the window
            setattr(self, name, getattr(self.monitor, name))
        self.monitor_panel = self.monitor
        self.tabs.addTab(self.monitor_panel, "Training monitor", "monitor", "Monitor", tab_id="monitor")

        # 4. Results Browser Panel: every metric of the ticked runs
        self.results_view = ResultsPanel(self.backend, self.settings_manager, load_config=self.load_results_config)
        self.results_panel = self.results_view
        self.tabs.addTab(self.results_panel, "Results browser", "results", "Results", tab_id="results")

        # 5. Plot Viewer
        self.plot_viewer = PlotViewer()
        # The runs list: the runs ticked in it are plotted together here. Past runs are viewed in this tab;
        # the Training monitor shows only the current or most recent run.
        self.plot_viewer.add_runs_list(self.monitor.sidebar)
        self.monitor.sidebar.compare_changed.connect(self.plot_ticked_runs)
        self.tabs.addTab(self.plot_viewer, "Plot Viewer", "plots", "Plots", tab_id="plots")

        # 5b. Figure workspace: compose several plots into one figure for a paper.
        # Separate from the Plot Viewer above, which views runs; this assembles
        # and exports a finished figure.
        from .figures.view import FigureWorkspace

        self.figure_workspace = FigureWorkspace()
        self.tabs.addTab(self.figure_workspace, "Figure workspace", "export", "Figures", tab_id="figures")

        # 6. Job Queue Panel
        self.queue_panel = QueuePanel(self.move_queued, self.remove_queued, self.open_job, self.set_queue_running,
                                      on_launch=self.launch_training, on_add=self.add_to_queue,
                                      on_stop=self.stop_run, on_demo=self.start_demo,
                                      on_parallel=self.set_max_parallel)
        self.queue_panel.set_parallel(self.jobs.max_parallel)
        self.queue_button = self.queue_panel.add_button
        self.stop_button = self.queue_panel.stop_button
        self.tabs.addTab(self.queue_panel, "Job queue", "queue", "Queue", tab_id="queue")

        # 8. Terminal Panel
        command, cwd, env = self.terminal_config()
        self.terminal_panel = TerminalPanel(cwd=cwd, parent=self, command=command, env=env)
        self.tabs.addTab(self.terminal_panel, "Terminal", "terminal", "Terminal", tab_id="terminal")

        # 9. Console Panel
        self.console_panel = self.make_console()
        self.tabs.addTab(self.console_panel, "Console", "console", "Console", tab_id="console")

        self.settings_panel = self.make_settings()
        self.tabs.set_settings_widget(self.settings_panel)
        self.tabs.currentChanged.connect(self.tab_changed)

    def dock(self, title, name, widget, area):
        dock = QDockWidget(title, self)
        dock.setObjectName(name)
        dock.setWidget(widget)
        self.addDockWidget(area, dock)
        return dock

    def init_default_experiment(self):
        default_exp = None
        exp_dir = self.config_tree.root_dir / "experiment"
        candidates = sorted(
            p for p in exp_dir.rglob("*.yaml")
            if not p.name.startswith("_") and p.is_file() and "/methods/" not in p.as_posix()
        ) if exp_dir.is_dir() else []
        if candidates:
            default_exp = str(candidates[0].relative_to(self.config_tree.root_dir))
        if default_exp:
            self.config_tree.select_file(default_exp)

    def toggle_yaml_preview(self, checked=None):
        if checked is None:
            checked = self.btn_toggle_yaml.isChecked()
        else:
            self.btn_toggle_yaml.setChecked(checked)
        self.preview_panel.setVisible(checked)
        if hasattr(self, "preview_splitter"):
            if checked:
                self.preview_splitter.setSizes([600, 420])
                self.request_compose()
            else:
                self.preview_splitter.setSizes([1000, 0])

    @property
    def config_viewer(self) -> ConfigViewer:
        """Backwards-compatibility proxy returning the active ConfigViewer."""
        if hasattr(self, "config_tab_manager") and self.config_tab_manager is not None:
            return self.config_tab_manager.current_viewer()
        return None

    def on_config_file_selected(self, file_path, rel_path, pinned=False):
        if hasattr(self, "config_tab_manager"):
            self.config_tab_manager.open_file(file_path, rel_path, pinned=pinned)

    def on_config_file_opened(self, file_path, rel_path):
        if hasattr(self, "config_tab_manager"):
            self.config_tab_manager.open_file(file_path, rel_path, pinned=True)

    def on_config_file_deleted(self, rel_path):
        norm = Path(rel_path).as_posix()
        if hasattr(self, "config_tab_manager"):
            self.config_tab_manager.close_tabs_matching(lambda p: Path(p).as_posix() == norm)

    def on_config_dir_deleted(self, rel_path):
        norm = Path(rel_path).as_posix()
        if hasattr(self, "config_tab_manager"):
            self.config_tab_manager.close_tabs_matching(
                lambda p: Path(p).as_posix() == norm or Path(p).as_posix().startswith(norm + "/")
            )

    def close_active_tab(self):
        if hasattr(self, "config_tab_manager") and self.config_tab_manager is not None:
            self.config_tab_manager.close_current_tab()

    def on_config_tab_switched(self, rel_path, viewer):
        if not viewer or not rel_path:
            self.active_config_path = None
            self.active_config_rel_path = None
            self.current_experiment = None
            for button in launch_buttons(self):
                button.setEnabled(False)
            if self.queue_button is not None:
                self.queue_button.setEnabled(False)
            if hasattr(self, "queue_panel"):
                self.queue_panel.set_experiment(None)
            return

        self.active_config_path = viewer.current_path
        self.active_config_rel_path = rel_path
        if self.btn_save is not None:
            self.btn_save.setEnabled(viewer.is_dirty)

        norm_rel = Path(rel_path).as_posix()
        if norm_rel.startswith("experiment/") and not Path(rel_path).name.startswith("_"):
            exp_name = Path(norm_rel).relative_to("experiment").with_suffix("").as_posix()
            self.current_experiment = exp_name
        else:
            self.current_experiment = None
        self.update_launch_state()

        if hasattr(self, "config_tree") and self.config_tree is not None:
            self.config_tree.select_file(rel_path, emit_signal=False)

        if Path(rel_path).name.startswith("_") or not norm_rel.startswith("experiment/"):
            self.compose_valid = False
            self.update_launch_state()
            if getattr(self, "preview_status", None) is not None:
                self.preview_status.setText("Template configuration • not directly executable")
            if getattr(self, "preview_errors", None) is not None:
                self.preview_errors.hide()
            if getattr(self, "builder_status", None) is not None:
                self.builder_status.setObjectName("muted")
                self.builder_status.setText("Base template: provides group defaults inherited by experiments in this directory.")
                self.builder_status.setStyle(self.builder_status.style())
            if getattr(self, "preview", None) is not None:
                if getattr(viewer, "current_path", None) and viewer.current_path.is_file():
                    self.preview.setPlainText(f"# {viewer.current_path.name} (Base Template)\n# Experiments in this group inherit these settings:\n\n" + viewer.current_path.read_text(encoding="utf-8"))
                else:
                    self.preview.setPlainText("# Base Template\n# Group defaults inherited by experiments.")
            if hasattr(self, "plugin_manager"):
                self.plugin_manager.notify_experiment_changed(self.current_experiment, self.active_config_path)
            return

        self.request_compose()
        if hasattr(self, "plugin_manager"):
            self.plugin_manager.notify_experiment_changed(self.current_experiment, self.active_config_path)

    def duplicate_experiment(self):
        self.config_tree.prompt_duplicate()

    def save_current_config(self):
        viewer = self.config_viewer
        if viewer:
            ok = viewer.save_to_disk()
            if ok and viewer.current_rel_path:
                self.statusBar().showMessage(f"Saved {viewer.current_rel_path}", 4000)

    def on_config_saved(self):
        viewer = self.config_viewer
        if viewer and viewer.current_rel_path:
            self.statusBar().showMessage(f"Saved {viewer.current_rel_path}", 4000)
        self.compose_timer.start()  # debounced: an edit's config_changed has usually queued one already

    def on_config_file_renamed(self, old_rel, new_rel):
        viewer = self.config_viewer
        if viewer:
            self.active_config_path = viewer.current_path
        self.active_config_rel_path = new_rel

        norm_rel = Path(new_rel).as_posix()
        if norm_rel.startswith("experiment/") and not Path(new_rel).name.startswith("_"):
            exp_name = Path(norm_rel).relative_to("experiment").with_suffix("").as_posix()
            self.current_experiment = exp_name
            if self.start_button is not None:
                self.start_button.setToolTip(f"Train {exp_name} through the backend (F5)")
        else:
            self.current_experiment = None

        self.config_tree.current_rel_path = new_rel
        self.config_tree.populate()
        self.statusBar().showMessage(f"Renamed to {new_rel}", 4000)
        self.request_compose()
        if hasattr(self, "plugin_manager"):
            self.plugin_manager.notify_experiment_changed(self.current_experiment, self.active_config_path)

    def make_console(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(18, 16, 18, 14)
        layout.setSpacing(10)

        header = QHBoxLayout()
        header.setSpacing(8)
        header.addWidget(label("CONSOLE / SYSTEM OUTPUT", "eyebrow"))
        header.addStretch()
        for cmd_name, desc in (
            ("status", "Show active training and run summary"),
            ("config", "Show pipeline config command"),
            ("help", "List available console commands"),
        ):
            btn = QPushButton(f"› {cmd_name}")
            btn.setToolTip(desc)
            btn.clicked.connect(lambda _, c=cmd_name: self.run_quick_command(c))
            header.addWidget(btn)
        header.addWidget(self.button("Clear output", self.console_clear))
        layout.addLayout(header)

        self.console = QPlainTextEdit()
        self.console.setReadOnly(True)
        self.console.setMaximumBlockCount(3000)
        layout.addWidget(self.console, 1)

        self.command = QLineEdit()
        self.command.setPlaceholderText("Console › help, status, config, clear (see Terminal tab for system shell)")
        self.command.returnPressed.connect(self.console_command)
        layout.addWidget(self.command)
        return panel

    def console_clear(self):
        if hasattr(self, "console"):
            self.console.clear()

    def make_settings(self):
        from .settings_view import SettingsView
        self.settings_view = SettingsView(self)
        return self.settings_view


    def settings_theme_selected(self, theme_name):
        if theme_name and theme_name != self.theme_manager.active["name"]:
            theme = self.theme_manager.themes().get(theme_name)
            if theme:
                self.select_theme(theme)

    def terminal_config(self):
        """(argv or None for auto-detect, start folder, extra env) from the [terminal] settings."""
        from .terminal.pty_session import parse_shell_command, venv_environment

        sm = getattr(self, "settings_manager", None)
        if sm is None:
            return None, str(Path.cwd()), {}
        command = parse_shell_command(sm.terminal_shell)
        if sm.terminal_shell and command is None:
            self.log(f"[terminal] Shell '{sm.terminal_shell}' not found; using the default shell.")
        cwd = sm.terminal_cwd
        if not cwd or not Path(cwd).expanduser().is_dir():
            cwd = str(Path.cwd())
        env = venv_environment(REPO_ROOT / "venv") if sm.terminal_activate_venv else {}
        return command, str(Path(cwd).expanduser()), env

    def _on_settings_changed(self):
        """Called when settings.toml changes on disk (or when the app mutates it).

        Applies any differences to the live UI — theme, sidebar visibility, etc.
        Designed to be idempotent so redundant calls are harmless.
        """
        if not hasattr(self, "settings_manager"):
            return

        if hasattr(self, "tabs"):
            self.tabs.set_settings_button_logo(self.settings_manager.settings_button_logo)

        # --- Terminal: shell, folder and venv apply from the next new shell ---
        if hasattr(self, "terminal_panel"):
            command, cwd, env = self.terminal_config()
            self.terminal_panel.configure(command=command, cwd=cwd, env=env)

        # --- Theme hot-reload ---
        new_theme = self.settings_manager.theme
        if new_theme != self.theme_manager.active.get("name"):
            if not self.theme_manager.select_by_name(new_theme):
                self.log(f"[settings] Unknown theme '{new_theme}' in settings.toml; keeping current theme.")
            else:
                # Sync the dropdown in the Settings panel if it's already built
                if hasattr(self, "settings_theme_select"):
                    self.settings_theme_select.blockSignals(True)
                    self.settings_theme_select.setCurrentText(new_theme)
                    self.settings_theme_select.blockSignals(False)

        # --- Sidebar visibility hot-reload ---
        known_set = set(self.settings_manager.sidebar_order)
        visible_set = set(self.settings_manager.sidebar_visible)
        if known_set and hasattr(self, "tabs"):
            for pid in self.tabs.tab_order:
                if pid not in known_set:
                    continue  # plugin tab not in settings — leave it alone
                should_be = pid in visible_set
                if self.tabs.is_tab_visible(pid) != should_be:
                    self.tabs.set_tab_visible(pid, should_be)
                    if hasattr(self, "pane_sliders") and pid in self.pane_sliders:
                        slider = self.pane_sliders[pid]
                        slider.blockSignals(True)
                        slider.setChecked(should_be)
                        slider.blockSignals(False)
            if hasattr(self, "pane_sliders"):
                self.update_pane_sliders_state()

        # --- Hotkeys hot-reload ---
        if hasattr(self, "settings_action_key_combo"):
            cur = self.settings_manager.action_key.lower()
            idx = self.settings_action_key_combo.findData(cur)
            if idx >= 0:
                if self.settings_action_key_combo.currentIndex() != idx:
                    self.settings_action_key_combo.blockSignals(True)
                    self.settings_action_key_combo.setCurrentIndex(idx)
                    self.settings_action_key_combo.blockSignals(False)
            else:
                self.settings_action_key_combo.blockSignals(True)
                self.settings_action_key_combo.addItem(cur.replace("_", "+").upper(), cur)
                self.settings_action_key_combo.setCurrentIndex(self.settings_action_key_combo.count() - 1)
                self.settings_action_key_combo.blockSignals(False)

        # --- Zoom factor hot-reload ---
        new_zoom = self.settings_manager.zoom
        if abs(new_zoom - getattr(self, "zoom_factor", 1.0)) > 1e-4:
            self.set_zoom(new_zoom, persist=False)

        # --- Menu Shortcuts hot-reload ---
        if hasattr(self, "menu_actions"):
            for aid, action in self.menu_actions.items():
                default_sc = action.property("_default_sc") or ""
                sc = self.settings_manager.get("shortcuts", aid, default=default_sc)
                if sc is None:
                    sc = default_sc
                norm = normalize_menu_shortcut(sc)
                if norm:
                    seq = QKeySequence(norm)
                    if aid == "zoom_in":
                        shortcuts = [seq]
                        if norm.endswith("+"):
                            shortcuts.append(QKeySequence(norm[:-1] + "="))
                        elif norm.endswith("="):
                            shortcuts.append(QKeySequence(norm[:-1] + "+"))
                        action.setShortcuts(shortcuts)
                    else:
                        action.setShortcut(seq)
                else:
                    action.setShortcut(QKeySequence())

        # --- Settings view sync ---
        if hasattr(self, "settings_view") and hasattr(self.settings_view, "sync_from_settings"):
            self.settings_view.sync_from_settings()

    def _on_action_key_changed(self, index):
        if not hasattr(self, "settings_action_key_combo"):
            return
        val = self.settings_action_key_combo.itemData(index)
        if val:
            self.settings_manager.set("hotkeys", "action_key", val)
            self.statusBar().showMessage(f"Action Key set to {val.replace('_', ' ').title()}", 3000)

    def switch_to_pane(self, target: str | int) -> bool:
        """Switch to a pane by name or numeric index [0, 1, 2, ...]."""
        if hasattr(self, "hotkey_manager"):
            if isinstance(target, int):
                return self.hotkey_manager.switch_to_pane_by_index(target)
            return self.hotkey_manager.switch_to_pane(str(target))
        return False

    def on_sidebar_auto_hide_changed(self, enabled):
        """Keep the menu item and the Settings switch in step, and remember the choice."""
        for control in (getattr(self, "auto_hide_action", None), getattr(self, "auto_hide_slider", None)):
            if control is not None and control.isChecked() != enabled:
                control.blockSignals(True)
                control.setChecked(enabled)
                control.blockSignals(False)
        if hasattr(self, "settings_manager"):
            self.settings_manager.set("sidebar", "auto_hide", enabled)
        self.statusBar().showMessage(
            "Sidebar auto-hide on: hover the left edge to show it." if enabled else "Sidebar docked.", 4000)

    def on_pane_slider_toggled(self, pane_id, checked):
        # Count the other panes, so the guard holds whether or not this slider has already flipped
        others_visible = sum(1 for pid, s in self.pane_sliders.items() if pid != pane_id and s.isChecked())
        if not checked and others_visible == 0:
            slider = self.pane_sliders.get(pane_id)
            if slider:
                slider.blockSignals(True)
                slider.setChecked(True)
                slider.blockSignals(False)
            self.statusBar().showMessage("At least one panel must remain visible in the sidebar.", 3000)
            return

        self.tabs.set_tab_visible(pane_id, checked)
        self.update_pane_sliders_state()
        self.save_layout()

    def on_plugin_slider_toggled(self, plugin_id, checked):
        if not hasattr(self, "plugin_manager"):
            return
        if checked:
            success = self.plugin_manager.enable_plugin(plugin_id)
            if success:
                name = self.plugin_manager.manifests[plugin_id].name
                self.statusBar().showMessage(f"Activated plugin: {name}", 4000)
            else:
                slider = self.plugin_sliders.get(plugin_id)
                if slider:
                    slider.blockSignals(True)
                    slider.setChecked(False)
                    slider.blockSignals(False)
                self.statusBar().showMessage(f"Failed to activate plugin: {plugin_id}", 4000)
        else:
            self.plugin_manager.disable_plugin(plugin_id)
            manifest = self.plugin_manager.manifests.get(plugin_id)
            name = manifest.name if manifest else plugin_id
            self.statusBar().showMessage(f"Deactivated plugin: {name}", 4000)

        # Extension slots update live, but rebuild the open config editor too
        # so plugin contributions are immediately reflected in its method rows.
        viewer = getattr(self, "config_viewer", None)
        if viewer is not None and hasattr(viewer, "_render_boxes"):
            viewer._render_boxes()
        self.save_layout()

    def _clear_layout(self, layout):
        if not layout:
            return
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                self._clear_layout(item.layout())

    def _create_plugin_row(self, pid: str, manifest, is_core: bool = False):
        row = QHBoxLayout()
        row.setSpacing(10)

        info_layout = QVBoxLayout()
        info_layout.setSpacing(1)
        badge = "  ·  Core" if is_core else ""
        title_lbl = label(f"{manifest.name}  ·  v{manifest.version}{badge}")
        title_lbl.setStyleSheet("font-weight: 500;")
        info_layout.addWidget(title_lbl)
        if manifest.description:
            desc_lbl = label(manifest.description, "muted")
            info_layout.addWidget(desc_lbl)
        row.addLayout(info_layout, 1)

        is_enabled = self.plugin_manager.is_plugin_enabled(pid)
        slider = ToggleSlider(checked=is_enabled)
        slider.setToolTip(f"Enable or disable {manifest.name}")
        slider.setAccessibleName(f"Toggle {manifest.name} plugin")
        slider.toggled.connect(lambda chk, p=pid: self.on_plugin_slider_toggled(p, chk))
        self.plugin_sliders[pid] = slider
        row.addWidget(slider)

        instance = self.plugin_manager.instances.get(pid)
        has_settings = False
        if instance and hasattr(instance, "get_settings_widget"):
            has_settings = True
        elif hasattr(manifest, "extra") and isinstance(manifest.extra, dict) and manifest.extra.get("has_settings", False):
            has_settings = True

        if has_settings:
            btn_settings = QToolButton()
            btn_settings.setIcon(svg_icon("settings", {(QIcon.Mode.Normal, QIcon.State.Off): "text"}))
            btn_settings.setToolTip(f"{manifest.name} Settings")
            btn_settings.setFixedSize(30, 30)
            btn_settings.setCursor(Qt.CursorShape.PointingHandCursor)
            btn_settings.clicked.connect(lambda _, p=pid: self.open_plugin_settings(p))
            row.addWidget(btn_settings)

        if not is_core:
            btn_trash = QToolButton()
            trash_icon_path = Path(__file__).parent / "icons" / "trash.svg"
            if trash_icon_path.exists():
                btn_trash.setIcon(QIcon(str(trash_icon_path)))
            else:
                btn_trash.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_TrashIcon))
            btn_trash.setToolTip(f"Uninstall {manifest.name}")
            btn_trash.setFixedSize(28, 28)
            btn_trash.setCursor(Qt.CursorShape.PointingHandCursor)
            btn_trash.setStyleSheet(
                "QToolButton { border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 4px; background: rgba(255, 255, 255, 0.04); } "
                "QToolButton:hover { background: rgba(251, 73, 52, 0.2); border-color: #fb4934; }"
            )
            btn_trash.clicked.connect(lambda _, p=pid: self.uninstall_plugin_from_settings(p))
            row.addWidget(btn_trash)

        return row

    def refresh_plugins_ui(self):
        self.plugin_sliders = {}

        has_core = hasattr(self, "core_plugins_grid") and self.core_plugins_grid is not None
        has_community = hasattr(self, "plugins_grid") and self.plugins_grid is not None

        if not has_core and not has_community:
            return

        if has_core:
            self._clear_layout(self.core_plugins_grid)
        if has_community:
            self._clear_layout(self.plugins_grid)

        core_manifests = {}
        community_manifests = {}

        if hasattr(self, "plugin_manager") and self.plugin_manager.manifests:
            for pid, manifest in self.plugin_manager.manifests.items():
                kind = getattr(manifest, "kind", None)
                if not kind and hasattr(manifest, "extra") and isinstance(manifest.extra, dict):
                    kind = manifest.extra.get("kind", "plugin")
                if kind is not None and kind != "plugin":
                    continue

                if getattr(manifest, "is_core", False):
                    core_manifests[pid] = manifest
                else:
                    community_manifests[pid] = manifest

        # 1. Render Core Plugins
        if has_core:
            if core_manifests:
                for pid, manifest in core_manifests.items():
                    row = self._create_plugin_row(pid, manifest, is_core=True)
                    self.core_plugins_grid.addLayout(row)
            else:
                self.core_plugins_grid.addWidget(label("No core plugins installed.", "muted"))

        # 2. Render Community Plugins
        if has_community:
            if community_manifests:
                for pid, manifest in community_manifests.items():
                    row = self._create_plugin_row(pid, manifest, is_core=False)
                    self.plugins_grid.addLayout(row)
            else:
                self.plugins_grid.addWidget(label("No community plugins installed.", "muted"))

    def uninstall_plugin_from_settings(self, plugin_id: str):
        if not hasattr(self, "plugin_manager"):
            return
        manifest = self.plugin_manager.manifests.get(plugin_id)
        name = manifest.name if manifest else plugin_id
        reply = QMessageBox.question(
            self,
            "Uninstall Plugin",
            f"Are you sure you want to uninstall '{name}'?\nThis will remove its files and disable the plugin.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self.plugin_manager.uninstall_plugin(plugin_id)

        # Let the Hub refresh its installed-status badge if it's open.
        if hasattr(self, "hub_client"):
            self.hub_client.refresh_installed_status()

        self.refresh_plugins_ui()
        self.save_layout()
        self.statusBar().showMessage(f"Uninstalled plugin: {name}", 4000)

    def open_plugin_settings(self, plugin_id: str):
        from PyQt6.QtWidgets import QDialogButtonBox, QMessageBox, QVBoxLayout
        
        instance = self.plugin_manager.instances.get(plugin_id)
        if not instance:
            QMessageBox.information(self, "Plugin Disabled", "Please enable the plugin first to configure its settings.")
            return
            
        context = self.plugin_manager.contexts.get(plugin_id)
        settings_widget = None
        if hasattr(instance, "get_settings_widget"):
            settings_widget = instance.get_settings_widget(context)
            
        if not settings_widget:
            QMessageBox.information(self, "No Settings", f"The plugin '{plugin_id}' has no configurable settings.")
            return
            
        dialog = QDialog(self)
        manifest = self.plugin_manager.manifests.get(plugin_id)
        name = manifest.name if manifest else plugin_id
        dialog.setWindowTitle(f"{name} Settings")
        dialog.setMinimumWidth(400)
        
        layout = QVBoxLayout(dialog)
        layout.addWidget(settings_widget)
        
        btn_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        btn_box.accepted.connect(dialog.accept)
        layout.addWidget(btn_box)
        
        dialog.exec()

    def _on_hub_component_changed(self, component_id: str, action: str):
        from .config_viewer import invalidate_config_tree
        invalidate_config_tree()

        comp = self.hub_client.get_component(component_id) if hasattr(self, "hub_client") else None

        # Determine target category folder in in/config/ (e.g. "agent", "model", "env", "experiment")
        target_folder = None
        target_config_rel = None
        if comp:
            folder_map = {
                "method": "agent",
                "model": "model",
                "env": "env",
                "experiment": "experiment",
            }
            target_folder = folder_map.get(comp.kind)
            if target_folder:
                target_config_rel = f"{target_folder}/{comp.id}.yaml"

        # Refresh components panel and config tree so added or removed YAMLs update immediately without collapsing folders
        if hasattr(self, "components_panel") and self.components_panel is not None:
            self.components_panel.reload_components(
                target_rel_path=target_config_rel if action == "install" else None,
                ensure_expanded=target_folder,
            )
        if hasattr(self, "config_tree") and self.config_tree is not None:
            self.config_tree.populate(ensure_expanded=target_folder)
        if hasattr(self, "config_tab_manager") and self.config_tab_manager is not None:
            self.config_tab_manager.refresh_component_choices()

        # Non-plugin components (methods, models, envs) are filesystem assets only — no plugin UI to update.
        if comp and comp.kind != "plugin":
            self.statusBar().showMessage(f"Hub {action} finished: {component_id}", 4000)
            return

        if not hasattr(self, "plugin_manager"):
            return

        if action == "uninstall":
            # Resolve the plugin id from the component's target_path (e.g. "plugins/my_plugin" → "my_plugin").
            # Deactivate any running instance before discover() prunes its manifest.
            pid = Path(comp.target_path).name if comp and comp.target_path else component_id
            if pid in self.plugin_manager.instances:
                self.plugin_manager.disable_plugin(pid)
            self.plugin_manager.manifests.pop(pid, None)
            self.plugin_manager.enabled_states.pop(pid, None)
            self.plugin_manager.save_states()

        # Sync manifests with the current filesystem state.
        self.plugin_manager.discover()

        if action == "install":
            # Auto-enable if the user had it enabled before (e.g. re-installing).
            pid = Path(comp.target_path).name if comp and comp.target_path else component_id
            if pid in self.plugin_manager.manifests:
                if self.plugin_manager.is_plugin_enabled(pid) and pid not in self.plugin_manager.instances:
                    self.plugin_manager.enable_plugin(pid)

        self.refresh_plugins_ui()
        self.save_layout()
        self.statusBar().showMessage(f"Hub {action} finished: {component_id}", 4000)

    def open_hub(self, initial_kind: str | None = None, mode: str = "plugins"):
        if hasattr(self, "hub_client"):
            dialog = HubDialog(self.hub_client, initial_kind=initial_kind, mode=mode, parent=self)
            dialog.exec()

    def update_pane_sliders_state(self):
        visible_sliders = [s for s in self.pane_sliders.values() if s.isChecked()]
        if len(visible_sliders) == 1:
            visible_sliders[0].setEnabled(False)
            visible_sliders[0].setToolTip("At least one panel must remain visible in the sidebar")
        else:
            for s in self.pane_sliders.values():
                s.setEnabled(True)
                s.setToolTip("Show or hide this panel in the sidebar")

    def reset_sidebar_layout(self):
        default_order = [
            "components",
            "config",
            "workflows",
            "monitor",
            "results",
            "plots",
            "tensorboard",
            "queue",
            "terminal",
            "console",
        ]
        self.tabs.clear_bins(emit=False)
        self.tabs.apply_tab_order(default_order)
        # Every tab, not only those with a switch: a plugin's tab must come back too.
        for pid in list(self.tabs.tabs):
            self.tabs.set_tab_visible(pid, pid not in HIDDEN_BY_DEFAULT)
        for pid, slider in self.pane_sliders.items():
            slider.blockSignals(True)
            slider.setChecked(pid not in HIDDEN_BY_DEFAULT)
            slider.setEnabled(True)
            slider.blockSignals(False)
        self.save_layout()
        self.statusBar().showMessage("Restored default sidebar panels and order.", 4000)

    def save_layout(self):
        if not hasattr(self, "settings_manager") or self._shutting_down:
            # Shutdown removes each plugin's tab. That isn't the user's layout, and each save
            # rebuilt the open experiment three times, flashing its boxes as windows.
            return
        try:
            order = list(self.tabs.tab_order)
            visible = [pid for pid in order if self.tabs.is_tab_visible(pid)]
            self.settings_manager.set("sidebar", "order", order)
            self.settings_manager.set("sidebar", "visible", visible)
            self.settings_manager.set("sidebar", "bins", self.tabs.bins_state())
        except Exception as exc:
            self.log(f"Failed to save sidebar layout: {exc}")

    def load_layout(self):
        if not hasattr(self, "settings_manager"):
            return
        try:
            order = self.settings_manager.sidebar_order
            visible_set = set(self.settings_manager.sidebar_visible)
            known_set = set(order)  # tabs settings.toml explicitly knows about
            bins = self.settings_manager.get("sidebar", "bins", default=[])
            self.tabs.set_bins(bins if isinstance(bins, list) else [])
            if order:
                self.tabs.apply_tab_order(order)
            for pid in self.tabs.tab_order:
                if pid not in known_set:
                    # Plugin-added tab not yet in settings — leave it visible
                    # (the user hasn't made a preference yet; save_layout will
                    # persist it the next time the layout changes)
                    continue
                is_visible = pid in visible_set
                self.tabs.set_tab_visible(pid, is_visible)
                if hasattr(self, "pane_sliders") and pid in self.pane_sliders:
                    slider = self.pane_sliders[pid]
                    slider.blockSignals(True)
                    slider.setChecked(is_visible)
                    slider.blockSignals(False)
            if hasattr(self, "pane_sliders"):
                self.update_pane_sliders_state()
        except Exception as exc:
            self.log(f"Failed to load sidebar layout: {exc}")

    def save_session(self):
        """Persist the user's current workspace view for the next launch."""
        if not hasattr(self, "settings_manager") or getattr(self, "_restoring_session", False):
            return
        try:
            pane_id = next(
                (pid for pid, entry in self.tabs.tabs.items()
                 if entry.get("widget") is self.tabs.currentWidget()),
                "config",
            )
            manager = getattr(self, "config_tab_manager", None)
            open_configs = list(manager._tabs) if manager else []
            current_viewer = manager.current_viewer() if manager else None
            active_config = current_viewer.current_rel_path if current_viewer else ""
            session = {
                "active_pane": pane_id,
                "open_configs": open_configs,
                "active_config": active_config or "",
                "config_splitter": self.config_splitter.sizes(),
                "preview_splitter": self.preview_splitter.sizes(),
                "preview_visible": not self.preview_panel.isHidden(),
                "window_geometry": bytes(self.saveGeometry().toBase64()).decode("ascii"),
            }
            self.settings_manager.persist_many("session", session)
        except Exception as exc:
            self.log(f"Failed to save workspace session: {exc}")

    def restore_session(self):
        """Restore the last open pane, config tabs, and editor split positions."""
        sm = getattr(self, "settings_manager", None)
        if sm is None:
            return
        self._restoring_session = True
        try:
            saved_geometry = sm.get("session", "window_geometry", default="")
            if isinstance(saved_geometry, str) and saved_geometry:
                self.restoreGeometry(QByteArray.fromBase64(saved_geometry.encode("ascii")))
            manager = self.config_tab_manager
            saved_configs = sm.get("session", "open_configs", default=[])
            if isinstance(saved_configs, list):
                # init_default_experiment opens a fallback tab; replace it when a saved
                # session has valid files, while gracefully skipping files that moved.
                paths = [str(p) for p in saved_configs if isinstance(p, str)]
                valid = [p for p in paths if (self.config_tree.root_dir / p).is_file()]
                if valid:
                    manager.close_tabs_matching(lambda _path: True)
                    for rel_path in valid:
                        manager.open_file(self.config_tree.root_dir / rel_path, rel_path, pinned=True)
                    active = sm.get("session", "active_config", default="")
                    if active in manager._tabs:
                        manager.setCurrentWidget(manager._tabs[active])

            config_sizes = sm.get("session", "config_splitter", default=[])
            if isinstance(config_sizes, list) and len(config_sizes) == 2:
                self.config_splitter.setSizes([max(0, int(v)) for v in config_sizes])
            preview_sizes = sm.get("session", "preview_splitter", default=[])
            if isinstance(preview_sizes, list) and len(preview_sizes) == 2:
                self.preview_splitter.setSizes([max(0, int(v)) for v in preview_sizes])
            preview_visible = bool(sm.get("session", "preview_visible", default=False))
            self.preview_panel.setVisible(preview_visible)
            self.btn_toggle_yaml.setChecked(preview_visible)

            pane_id = sm.get("session", "active_pane", default="config")
            if pane_id in self.tabs.tabs:
                self.tabs.setCurrentWidget(self.tabs.tabs[pane_id]["widget"])
        except Exception as exc:
            self.log(f"Failed to restore workspace session: {exc}")
        finally:
            self._restoring_session = False

    def test_backend_connection(self):
        """Settings → Training pipeline: check the pipeline composes in this Python environment."""
        if hasattr(self, "settings_backend_status"):
            self.settings_backend_status.setText("Status: Checking the training pipeline…")
            self.settings_backend_status.setStyleSheet("")
            self.backend.config_schema(self.backend_health_callback)

    def backend_health_callback(self, data, error):
        if not hasattr(self, "settings_backend_status"):
            return
        self._backend_ok = not error
        if error:
            self.settings_backend_status.setText(f"Status: Unavailable ({error})")
        else:
            self.settings_backend_status.setText(f"Status: Ready · composes {data['base_experiment']} in-process")
        self.color_backend_status()

    def color_backend_status(self):
        """Colour the pipeline status by the last check's result, in the current theme."""
        ok = getattr(self, "_backend_ok", None)
        if ok is not None and hasattr(self, "settings_backend_status"):
            self.settings_backend_status.setStyleSheet(f"color: {theme_color('primary' if ok else 'number')};")

    def run_quick_command(self, command):
        self.tabs.setCurrentWidget(self.console_panel)
        if hasattr(self, "command"):
            self.command.setText(command)
        self.console_command(command)

    def make_menus(self):
        self.menu_actions = {}
        file_menu = self.menuBar().addMenu("File")
        for aid, title, default_sc, callback in (
            ("new_experiment", "New experiment", "Ctrl+N", self.new_experiment),
            ("scaffold_plugin", "Scaffold New Component Plugin…", "Ctrl+Shift+P", self.scaffold_plugin),
            ("reload_components", "Reload Component Code", "Ctrl+Alt+R", self.reload_components),  # Ctrl+Shift+R is the queue's
            ("export_config", "Export draft YAML…", "Ctrl+Shift+S", self.export_config),
            ("save_config", "Save configuration", "Ctrl+S", self.save_current_config),
            ("close_tab", "Close tab", "Ctrl+W", self.close_active_tab),
            ("quit", "Quit", "Ctrl+Q", self.quit_app),
        ):
            action = QAction(title, self)
            action.setProperty("_default_sc", default_sc)
            sc = self.settings_manager.get("shortcuts", aid, default=default_sc) if hasattr(self, "settings_manager") else default_sc
            sc_norm = normalize_menu_shortcut(sc)
            if sc_norm:
                action.setShortcut(QKeySequence(sc_norm))
            action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
            action.triggered.connect(callback)
            file_menu.addAction(action)
            self.addAction(action)
            self.menu_actions[aid] = action

        # Training is started and stopped from the Queue tab; its shortcuts stay window-wide.
        for aid, title, default_sc, callback in (
            ("launch_training", "Launch training", "F5", self.launch_training),
            ("stop_run", "Stop", "Shift+F5", self.stop_run),
            ("add_to_queue", "Add to queue", "Ctrl+Shift+Q", self.add_to_queue),
            ("toggle_queue", "Start or pause queue", "Ctrl+Shift+R",
             lambda: self.set_queue_running(not self.queue_running)),
            ("start_demo", "Start simulated demo", "Ctrl+F5", self.start_demo),
        ):
            action = QAction(title, self)
            action.setProperty("_default_sc", default_sc)
            sc = self.settings_manager.get("shortcuts", aid, default=default_sc) if hasattr(self, "settings_manager") else default_sc
            sc_norm = normalize_menu_shortcut(sc)
            if sc_norm:
                action.setShortcut(QKeySequence(sc_norm))
            action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
            action.triggered.connect(callback)
            self.addAction(action)
            self.menu_actions[aid] = action
        view_menu = self.menuBar().addMenu("View")
        self.themes_menu = view_menu.addMenu("Themes")
        self.themes_menu.aboutToShow.connect(self.populate_themes_menu)
        view_menu.addAction("Theme builder…", self.show_theme_builder)
        view_menu.addSeparator()

        for aid, title, default_sc, callback in (
            ("zoom_in", "Zoom In", "Ctrl++", self.zoom_in),
            ("zoom_out", "Zoom Out", "Ctrl+-", self.zoom_out),
            ("reset_zoom", "Reset Zoom", "Ctrl+0", self.reset_zoom),
            ("open_launcher", "Open launcher…", "Ctrl+Space", self.toggle_launcher),
        ):
            action = QAction(title, self)
            action.setProperty("_default_sc", default_sc)
            sc = self.settings_manager.get("shortcuts", aid, default=default_sc) if hasattr(self, "settings_manager") else default_sc
            sc_norm = normalize_menu_shortcut(sc)
            if sc_norm:
                seq = QKeySequence(sc_norm)
                if aid == "zoom_in":
                    shortcuts = [seq]
                    if sc_norm.endswith("+"):
                        shortcuts.append(QKeySequence(sc_norm[:-1] + "="))
                    elif sc_norm.endswith("="):
                        shortcuts.append(QKeySequence(sc_norm[:-1] + "+"))
                    action.setShortcuts(shortcuts)
                else:
                    action.setShortcut(seq)
            action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
            action.triggered.connect(callback)
            view_menu.addAction(action)
            self.addAction(action)
            self.menu_actions[aid] = action

        view_menu.addSeparator()
        view_menu.addAction("Components", lambda: self.tabs.setCurrentWidget(self.components_panel))
        view_menu.addAction("Experiment config", lambda: self.tabs.setCurrentWidget(self.config_panel))
        view_menu.addAction("Workflows", lambda: self.tabs.setCurrentWidget(self.workflows_panel))
        view_menu.addAction("Training monitor", lambda: self.tabs.setCurrentWidget(self.monitor_panel))
        view_menu.addAction("Results browser", lambda: self.tabs.setCurrentWidget(self.results_panel))
        view_menu.addAction("Plot Viewer", lambda: self.tabs.setCurrentWidget(self.plot_viewer))
        if hasattr(self, "tabs") and "tensorboard" in self.tabs.tabs:
            view_menu.addAction("TensorBoard", self.show_tensorboard)
        view_menu.addAction("Job queue", lambda: self.tabs.setCurrentWidget(self.queue_panel))
        view_menu.addAction("Terminal", lambda: self.tabs.setCurrentWidget(self.terminal_panel))
        view_menu.addAction("Console", lambda: self.tabs.setCurrentWidget(self.console_panel))
        view_menu.addAction("Settings", self.show_settings)
        view_menu.addAction("Restore default layout", lambda: self.restoreState(self.default_layout))
        self.auto_hide_action = QAction("Auto-hide sidebar", self, checkable=True)
        self.auto_hide_action.setChecked(self.tabs.auto_hide)
        self.auto_hide_action.toggled.connect(lambda on: self.tabs.set_auto_hide(on))
        view_menu.addAction(self.auto_hide_action)
        view_menu.addSeparator()
        view_menu.addAction("Preferences: Open Settings File", self.open_settings_file)
        help_menu = self.menuBar().addMenu("Help")
        action = QAction("Documentation", self)
        action.setProperty("_default_sc", "F1")
        sc = self.settings_manager.get("shortcuts", "open_docs", default="F1") if hasattr(self, "settings_manager") else "F1"
        sc_norm = normalize_menu_shortcut(sc)
        if sc_norm:
            action.setShortcut(QKeySequence(sc_norm))
        action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        action.triggered.connect(self.show_docs)
        help_menu.addAction(action)
        self.addAction(action)
        self.menu_actions["open_docs"] = action

    def open_settings_file(self):
        """Open the workspace settings.toml in the user's default text editor."""
        path = self.settings_manager.workspace_settings_path
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))
        self.statusBar().showMessage(f"Opened {path.name} — changes apply automatically on save.", 6000)

    def toggle_launcher(self):
        """Open or close the launcher (frontend/launcher.py): jump to any panel, plugin, setting or command."""
        if getattr(self, "launcher", None) is None:
            from .launcher import Launcher

            self.launcher = Launcher(self)
        self.launcher.toggle()

    def zoom_in(self):
        """Zoom in (increase interface and text size by 10%)."""
        self.set_zoom(self.zoom_factor + 0.1)

    def zoom_out(self):
        """Zoom out (decrease interface and text size by 10%)."""
        self.set_zoom(self.zoom_factor - 0.1)

    def reset_zoom(self):
        """Reset interface and text zoom to 100% (default size)."""
        self.set_zoom(1.0)

    def set_zoom(self, factor: float, persist: bool = True):
        """Set zoom scale factor between 0.6 (60%) and 2.4 (240%)."""
        factor = round(max(0.6, min(2.4, float(factor))), 2)
        if abs(getattr(self, "zoom_factor", 1.0) - factor) < 1e-4 and hasattr(self, "_zoom_initialized"):
            return
        self._zoom_initialized = True
        self.zoom_factor = factor

        # 1. Update in-memory setting immediately so getter returns current zoom
        if hasattr(self, "settings_manager"):
            self.settings_manager._data.setdefault("appearance", {})["zoom"] = factor
            if persist:
                if not hasattr(self, "_zoom_persist_timer"):
                    self._zoom_persist_timer = QTimer(self)
                    self._zoom_persist_timer.setSingleShot(True)
                    self._zoom_persist_timer.setInterval(250)
                    self._zoom_persist_timer.timeout.connect(self._persist_zoom)
                self._zoom_persist_timer.start()

        # 2. Update stylesheet via theme manager (instant CSS update without theme reload cascade)
        if hasattr(self, "theme_manager"):
            self.theme_manager.set_zoom(factor, target_widget=self)

        # 3. Update global application font (with base size caching)
        app = QApplication.instance()
        if app:
            font = app.font()
            base_pt = getattr(app, "_base_font_point_size", None)
            if base_pt is None:
                base_pt = font.pointSizeF() if font.pointSizeF() > 0 else 10.0
                app._base_font_point_size = base_pt
            new_pt = max(6.0, base_pt * factor)
            if abs(font.pointSizeF() - new_pt) > 0.05:
                font.setPointSizeF(new_pt)
                app.setFont(font)

        # 4. Update terminal font size (default 13px)
        if hasattr(self, "terminal_panel"):
            term_size = max(8, round(13 * factor))
            self.terminal_panel.set_font_size(term_size)

        # 5. Refresh Settings view styles if present and visible
        if hasattr(self, "settings_view") and hasattr(self.settings_view, "refresh_styles"):
            if self.settings_view.isVisible():
                self.settings_view.refresh_styles()
            else:
                self._settings_styles_stale = True  # restyled when the Settings tab is next shown

        # 6. Update appearance UI zoom controls if built
        if hasattr(self, "zoom_slider") and self.zoom_slider:
            self.zoom_slider.blockSignals(True)
            self.zoom_slider.setValue(int(round(factor * 100)))
            self.zoom_slider.blockSignals(False)
        if hasattr(self, "zoom_percent_label") and self.zoom_percent_label:
            self.zoom_percent_label.setText(f"{int(round(factor * 100))}%")

        # 7. Status bar confirmation, for zooms the user makes. Not while starting up: a message shown
        # before the window is visible doesn't hide the status bar's normal widgets (QStatusBar only
        # hides visible ones), so for its 1.5 s it was drawn on top of the theme label.
        if self.isVisible() and hasattr(self, "statusBar") and self.statusBar():
            self.statusBar().showMessage(f"Zoom: {int(round(factor * 100))}%", 1500)

    def _persist_zoom(self):
        """Save debounced zoom factor to settings.toml without triggering redundant reload cascade."""
        if hasattr(self, "settings_manager") and hasattr(self.settings_manager, "persist_key"):
            self.settings_manager.persist_key("appearance", "zoom", self.zoom_factor)

    def show_tensorboard(self):
        if hasattr(self, "tabs") and "tensorboard" in self.tabs.tabs:
            self.tabs.setCurrentWidget(self.tabs.tabs["tensorboard"]["widget"])

    def tab_changed(self, index):
        if not getattr(self, "_restoring_session", False) and index >= 0:
            pane_id = next(
                (pid for pid, entry in self.tabs.tabs.items()
                 if entry.get("widget") is self.tabs.widget(index)),
                None,
            )
            if pane_id:
                self.settings_manager.persist_key("session", "active_pane", pane_id)
        # 1. If leaving components panel, auto-save any unsaved edits
        if hasattr(self, "components_panel") and self.components_panel is not None:
            if hasattr(self.components_panel, "viewer") and self.components_panel.viewer and self.components_panel.viewer.is_dirty:
                self.components_panel.save_current()

        # Refresh the config panels only when something under in/config changed since they last
        # loaded (the Components panel, a hub install, an edit in the terminal). Reloading on every
        # switch rebuilt the tree and re-rendered the open form each time (~0.5s).
        if self._configs_changed_since(self):
            from .config_viewer import invalidate_config_tree
            invalidate_config_tree()  # other panels read the shared tree too

        w = self.tabs.widget(index)
        if hasattr(w, "activate") and callable(w.activate):
            w.activate()

        # If switching to config_panel (Experiment panel), refresh the active viewer so dropdowns update
        if w is getattr(self, "config_panel", None):
            if hasattr(self, "config_tab_manager"):
                active_v = self.config_tab_manager.current_viewer()
                if active_v and hasattr(active_v, "reload_data") and self._configs_changed_since(active_v):
                    active_v.reload_data(force_tree_refresh=True)

        # If switching to components_panel, refresh viewer
        if w is getattr(self, "components_panel", None):
            viewer = getattr(self.components_panel, "viewer", None)
            if hasattr(viewer, "reload_data") and self._configs_changed_since(viewer):
                viewer.reload_data(force_tree_refresh=True)

        if self.tabs.widget(index) is self.queue_panel:
            self.on_queue_changed(self.jobs.queue_state())
        if hasattr(self, "terminal_panel") and self.tabs.widget(index) is self.terminal_panel:
            self.terminal_panel.terminal.focus_terminal()
            self.terminal_panel.terminal.fit_terminal()
        if hasattr(self, "console_panel") and self.tabs.widget(index) is self.console_panel:
            self.command.setFocus()
        if hasattr(self, "settings_panel") and self.tabs.widget(index) is self.settings_panel:
            self.test_backend_connection()
            if hasattr(self, "settings_view"):
                # Theme changes restyle it immediately; only a zoom made while it was hidden is pending.
                if getattr(self, "_settings_styles_stale", False):
                    self._settings_styles_stale = False
                    self.settings_view.refresh_styles()

    def _configs_changed_since(self, holder) -> bool:
        """True if in/config changed since `holder` last checked; remembers the new state on it."""
        fingerprint = config_fingerprint(self.config_tree.root_dir)
        changed = getattr(holder, "_config_fingerprint", None) != fingerprint
        holder._config_fingerprint = fingerprint
        return changed

    def show_settings(self, tab_id=None):
        """Open Settings, on `tab_id` (e.g. "hotkeys") if given."""
        if hasattr(self, "settings_panel"):
            self.tabs.setCurrentWidget(self.settings_panel)
            if tab_id and hasattr(self, "settings_view"):
                self.settings_view.show_tab(tab_id)

    def show_docs(self):
        """Open the Documentation tab (F1), showing it if hidden and switching its plugin on if it is off."""
        if "documentation" in self.tabs.tabs:
            self.hotkey_manager.switch_to_pane("documentation")
        elif "documentation" in getattr(getattr(self, "plugin_manager", None), "manifests", {}):
            from .launcher import _enable_plugin
            _enable_plugin(self, "documentation")
        else:
            self.statusBar().showMessage("The Documentation plugin isn't installed.", 4000)

    def theme_changed(self):
        self.show_pipeline_state()
        self.highlighter.rehighlight()
        self.tabs.set_settings_button_logo(self.settings_manager.settings_button_logo)
        self.tabs.refresh_icons()
        for tree in (getattr(self, "config_tree", None),
                     getattr(getattr(self, "components_panel", None), "components_tree", None)):
            if tree is not None:
                tree.refresh_icons()
        if hasattr(self, "results_view"):
            self.results_view.explorer.refresh_icons()
        if hasattr(self, "settings_theme_select"):
            self.settings_theme_select.blockSignals(True)
            self.settings_theme_select.setCurrentText(self.theme_manager.active["name"])
            self.settings_theme_select.blockSignals(False)
        if hasattr(self, "settings_backend_status"):
            # Recolour only: re-running the check would recompose the whole pipeline config.
            self.color_backend_status()
        if hasattr(self, "terminal_panel"):
            self.terminal_panel.apply_theme(self.theme_manager.active)
        if hasattr(self, "settings_view"):
            self.settings_view.refresh_styles()
        if self.isVisible():  # before the first show, showEvent colours the title bar
            apply_title_bar(self, self.theme_manager.active["colors"])

    def showEvent(self, event):
        super().showEvent(event)
        apply_title_bar(self, self.theme_manager.active["colors"])  # the native window exists once shown

    def populate_themes_menu(self):
        self.themes_menu.clear()
        for name, theme in self.theme_manager.themes().items():
            action = self.themes_menu.addAction(name)
            action.setCheckable(True)
            action.setChecked(name == self.theme_manager.active["name"])
            action.triggered.connect(lambda _, palette=theme: self.select_theme(palette))
        self.themes_menu.addSeparator()
        self.themes_menu.addAction("Theme builder…", self.show_theme_builder)

    def persist_theme_choice(self, theme_name):
        """Record a chosen theme in settings.toml. Every way of picking a theme (View menu, Settings
        dropdown, theme builder) ends here, so a later settings reload can't revert to a stale name."""
        if hasattr(self, "settings_manager") and self.settings_manager.theme != theme_name:
            self.settings_manager.set("appearance", "theme", theme_name)

    def select_theme(self, theme):
        try:
            self.theme_manager.commit(theme)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Could not save theme", str(exc))

    def show_theme_builder(self):
        dialog = ThemeBuilder(self.theme_manager, self)
        dialog.exec()
        dialog.deleteLater()

    def config(self):
        if hasattr(self, "config_viewer") and getattr(self.config_viewer, "raw_data", None):
            d = self.config_viewer.raw_data
            exp_id = d.get("experiment_id") or (self.config_viewer.current_path.stem if getattr(self.config_viewer, "current_path", None) else "default_experiment")
            seed = int(d.get("seed") if d.get("seed") is not None else 42)
            steps = int(d.get("total_timesteps") or 10000)
            tb = bool(d.get("tensorboard", False))
            methods = d.get("methods", {})
            method_name, method = next(iter(methods.items()), ("method", {})) if isinstance(methods, dict) else ("method", {})
            method = method if isinstance(method, dict) else {}
            lr = float(method.get("lr") or 0.0003)
            batch = int(method.get("batch_size") or 64)
            gamma = float(method.get("gamma") or 0.99)
            return Config(name=str(exp_id), method_name=str(method_name), agent=str(method.get("agent") or ""),
                          model=str(method.get("model") or ""), seed=seed, total_timesteps=steps, lr=lr,
                          batch_size=batch, gamma=gamma, tensorboard=tb)
        return Config()

    def update_config(self, *_):
        self.compose_timer.start()

    def request_compose(self):
        self.compose_serial += 1
        serial, config = self.compose_serial, self.config()
        exp_target = getattr(self, "current_experiment", None)
        if not exp_target:
            if hasattr(self, "config_tree") and (self.config_tree.root_dir / f"experiment/{BASE_EXPERIMENT}.yaml").exists():
                exp_target = BASE_EXPERIMENT
            else:
                self.preview.setPlainText("# No experiment currently selected.\n# Select an experiment from the tree or click 'New Experiment' to create one.")
                self.preview_status.setText("No experiment selected")
                self.compose_valid = False
                self.compose_error = None
                self.update_launch_state()
                return
        overrides = self.config_viewer.get_overrides() if hasattr(self, "config_viewer") else config.overrides()
        # A tab switch or edit starts a new validation request. Keep the launch
        # controls disabled until that exact config has been composed; otherwise
        # the previous experiment's valid state can leave Run active while an
        # invalid recipe is still being checked.
        self.compose_valid = False
        self.compose_error = None
        self.composed_config = None
        self.update_launch_state()
        self.backend.compose(exp_target, overrides,
                             lambda data, error: self.compose_finished(serial, config, data, error))

    def compose_finished(self, serial, config, data, error):
        if serial != self.compose_serial:
            return  # superseded by a newer edit
        if error or not isinstance(data, dict):
            self.compose_error = str(error or "Invalid pipeline response")
            self.compose_valid = False
            # Do not let the launch guard inspect a previous experiment's
            # resolved configuration after a failed refresh.
            self.composed_config = None
            self.update_launch_state()
            self.set_backend_state(False, error or "Invalid pipeline response")
            self.preview_status.setText("Pipeline unavailable • local recipe draft, not validated")
            self.preview_errors.hide()
            self.builder_status.setObjectName("muted")
            self.builder_status.setText("Training pipeline unavailable: this config has not been validated.")
            self.builder_status.setStyle(self.builder_status.style())
            self.preview.setPlainText(config.recipe_yaml())
            return
        self.set_backend_state(True)
        if self.schema is None and not self.schema_pending:
            self.request_schema(apply_defaults=False)
        header = ["# Resolved exactly as the command line would:",
                  "#   " + " ".join(data.get("argv", []))]
        header += [f"# Notice: {notice}" for notice in data.get("notices", [])]
        sections = ["\n".join(header)]
        if data.get("methods"):
            sections.append("# Methods — the settings each method actually trains with\nmethods:\n"
                            + "".join(f"  {line}\n" for line in data["methods_yaml"].splitlines()).rstrip())
        sections.append("# Full composed config — agent.* holds defaults that the method settings above override\n"
                        + data["config_yaml"])
        self.preview.setPlainText("\n\n".join(sections))
        self.compose_valid = bool(data.get("valid", False))
        # The launched run is described by the experiment, not by the builder. Without
        # this the run record keeps generic placeholder values instead of the
        # environment and method declared by the composed experiment.
        self.composed_config = data.get("config") if isinstance(data.get("config"), dict) else None
        if self.composed_config is not None:
            composed = self.composed_config
            env = composed.get("env")
            self.composed_facts = {
                "env": env.get("name") if isinstance(env, dict) else env,
                "seed": composed.get("seed"),
                "total_timesteps": composed.get("total_timesteps"),
                "paradigm": composed.get("paradigm"),
            }
        if data["valid"]:
            self.compose_error = None
            self.preview_status.setText("Resolved config • ✓ valid for the training pipeline")
            self.preview_errors.hide()
            self.builder_status.setObjectName("configOk")
            self.builder_status.setText("✓ Config validated by the training pipeline")
        else:
            messages = [f"[{e['stage']}] {e['message']}" for e in data["errors"]]
            self.compose_error = messages[0].splitlines()[0] if messages else "Configuration validation failed"
            self.preview_status.setText(f"Resolved config • ✗ {len(messages)} error(s)")
            self.preview_errors.setText("\n".join(messages))
            self.preview_errors.show()
            self.builder_status.setObjectName("configError")
            missing = self.check_missing_components()
            if missing:
                self.builder_status.setText(f"✗ Missing component: {', '.join(missing)}")
            else:
                self.builder_status.setText("✗ " + messages[0].splitlines()[0])
        self.update_launch_state()
        self.builder_status.setStyle(self.builder_status.style())

    def set_backend_state(self, connected, error=None):
        was_connected, self.backend_connected = self.backend_connected, connected
        if connected and not was_connected:
            self.request_disk_runs()  # list runs on disk, including ones started outside the app
        if not connected and hasattr(self, "results_view"):
            self.results_view.set_offline()
        self.pipeline_state = "ready" if connected else "unavailable"
        self.show_pipeline_state()
        if connected:
            self.backend_label.setToolTip(f"Configs compose in-process and training runs with {sys.executable}.")
        else:
            self.backend_label.setToolTip(f"{error}\nRun ThetaIDE from the project venv: "
                                          'pip install -e ".[gui]" then python -m frontend')

    def show_pipeline_state(self):
        """The status bar's pipeline indicator: a dot in the theme's colour for the state, then its name."""
        text, color = {"ready": ("Pipeline ready", "primary"), "unavailable": ("Pipeline unavailable", "number"),
                       "loading": ("Loading pipeline…", "muted")}[self.pipeline_state]
        self.backend_label.setText(f'<span style="color: {theme_color(color)}">●</span>&nbsp; {text}')

    def request_schema(self, apply_defaults):
        self.schema_pending = True
        self.backend.config_schema(lambda data, error: self.schema_loaded(data, error, apply_defaults))

    def schema_loaded(self, data, error, apply_defaults):
        self.schema_pending = False
        if error:
            self.set_backend_state(False, error)
            self.log(f"Training pipeline unavailable ({error}). Run ThetaIDE from the project venv: "
                     'pip install -e ".[gui]" then python -m frontend')
            return
        self.schema = data
        self.update_config()

    def set_config(self, config):
        if hasattr(self, "config_viewer"):
            d = self.config_viewer.raw_data
            if isinstance(config, dict):
                d.clear()
                d.update(config)
            else:
                d["experiment_id"] = getattr(config, "name", "experiment")
                d["seed"] = getattr(config, "seed", 42)
                d["total_timesteps"] = getattr(config, "total_timesteps", 10000)
                d["tensorboard"] = getattr(config, "tensorboard", False)
                methods = d.setdefault("methods", {})
                if isinstance(methods, dict):
                    method_name = getattr(config, "method_name", "method")
                    method = next(iter(methods.values()), None)
                    if not methods:
                        method = methods.setdefault(method_name, {})
                    if isinstance(method, dict):
                        for field in ("agent", "model", "lr", "batch_size", "gamma"):
                            value = getattr(config, field, None)
                            if value not in (None, ""):
                                method[field] = value
            self.config_viewer._render_boxes()
        self.update_config()

    def new_experiment(self):
        if hasattr(self, "config_tree"):
            self.config_tree.prompt_new_in_group()
        else:
            self.set_config(Config())
            self.tabs.setCurrentWidget(self.config_panel)

    def persist(self, run):
        # A run found in results/logs is only listed until something of the user's (such as notes)
        # is attached; saving it adopts it into the run store like a run launched from the app.
        transient = run.pop("transient", False)
        try:
            self.store.save(run)
            return True
        except OSError as exc:
            if transient:
                run["transient"] = True
            self.log(f"SAVE FAILED: {exc}")
            self.statusBar().showMessage(f"Could not save run: {exc}", 12000)
            return False

    def check_missing_components(self) -> list[str]:
        """Check if current experiment references any missing agent, model, or environment."""
        if not getattr(self, "current_experiment", None):
            return []

        missing = []

        # 1. Check from compose errors if any
        compose_err = getattr(self, "compose_error", "") or ""
        import re
        find_matches = re.findall(r"Could not find '(agent|model|env|paradigms)/([^']+)'", compose_err)
        for cat, name in find_matches:
            missing.append(f"{cat}: {name}")
        err_matches = re.findall(r"missing component[s]?: ([^.]+)", compose_err, re.IGNORECASE)
        for m in err_matches:
            for item in m.split(","):
                item_clean = item.strip()
                if item_clean:
                    missing.append(item_clean)

        ct = getattr(self, "config_tree", None)
        tree = getattr(ct, "tree_model", None) if ct is not None else None

        # 2. Check from current viewer or experiment file
        data = None
        viewer = getattr(self, "config_viewer", None)
        if viewer and getattr(viewer, "raw_data", None) and isinstance(viewer.raw_data, dict):
            data = viewer.raw_data
            if tree is not None:
                env_val = getattr(viewer, "_get_explicit_env", lambda: None)() or getattr(viewer, "_get_base_env", lambda: None)()
                if env_val and hasattr(tree, "environments"):
                    env_exists = (
                        env_val in tree.environments
                        or any(getattr(e, "name", None) == env_val for e in tree.environments.values())
                        or (getattr(tree, "config_root", Path("in/config")) / "env" / f"{env_val}.yaml").is_file()
                    )
                    if not env_exists:
                        missing.append(f"environment: {env_val}")

                methods = data.get("methods", {})
                if isinstance(methods, dict):
                    for m_name, m_spec in methods.items():
                        if not isinstance(m_spec, dict):
                            continue
                        ag = m_spec.get("agent")
                        if isinstance(ag, dict):
                            ag = ag.get("algorithm") or ag.get("name")
                        if ag and hasattr(tree, "agents"):
                            ag_str = str(ag)
                            ag_exists = (
                                ag_str in tree.agents
                                or any(getattr(a, "name", None) == ag_str or getattr(a, "algorithm", None) == ag_str for a in tree.agents.values())
                                or (getattr(tree, "config_root", Path("in/config")) / "agent" / f"{ag_str}.yaml").is_file()
                            )
                            if not ag_exists:
                                missing.append(f"agent: {ag_str}")

                        mo = m_spec.get("model")
                        if isinstance(mo, dict):
                            mo = next(iter(mo)) if mo else None
                        if mo and hasattr(tree, "models"):
                            mo_str = str(mo)
                            mo_stem = mo_str.split("/")[-1]
                            mo_exists = (
                                mo_str in tree.models
                                or mo_stem in tree.models
                                or (getattr(tree, "config_root", Path("in/config")) / "model" / f"{mo_str}.yaml").is_file()
                                or (getattr(tree, "config_root", Path("in/config")) / "model" / "basics" / f"{mo_stem}.yaml").is_file()
                            )
                            if not mo_exists:
                                missing.append(f"model: {mo_str}")

        if not data and hasattr(self, "active_config_path") and self.active_config_path:
            try:
                import yaml
                data = yaml.safe_load(Path(self.active_config_path).read_text(encoding="utf-8"))
            except Exception:
                data = None
        if not data:
            exp_target = getattr(self, "current_experiment", None)
            if exp_target:
                p = Path("in/config/experiment") / f"{exp_target}.yaml"
                if p.is_file():
                    try:
                        import yaml
                        data = yaml.safe_load(p.read_text(encoding="utf-8"))
                    except Exception:
                        data = None

        # A child experiment often inherits `env` from its group's _base.yaml.
        # Raw editor data therefore is not itself launchable.  Prefer the
        # exact config that the backend just composed and validated, falling
        # back to raw data only while composition is pending.
        composed_config = getattr(self, "composed_config", None)
        component_config = composed_config if isinstance(composed_config, dict) else data
        if component_config:
            try:
                from src.app.pipeline.validation import find_missing_components
                missing.extend(find_missing_components(component_config))
            except Exception:
                pass

        return list(dict.fromkeys(missing))

    @property
    def active(self):
        """The active run you are looking at: the shown run if it is training, else the newest active one."""
        if self.selected is not None and any(run is self.selected for run in self.active_runs):
            return self.selected
        return self.active_runs[-1] if self.active_runs else None

    @active.setter
    def active(self, run):
        """`None` clears every active run; a run is added to them (tests and old callers set it directly)."""
        if run is None:
            self.active_runs = []
        elif not any(r is run for r in self.active_runs):
            self.active_runs = [*self.active_runs, run]

    def launch_capacity(self) -> int:
        """How many runs may train at once (the Queue tab's "Run up to N at once")."""
        return max(1, int(getattr(self.jobs, "max_parallel", 1) or 1))

    def update_launch_state(self):
        training = len(self.active_runs)
        idle = not training
        full = training >= Window.launch_capacity(self)
        has_experiment = bool(getattr(self, "current_experiment", None))
        missing_components = Window.check_missing_components(self) if has_experiment else []
        if not isinstance(missing_components, list):
            missing_components = []
        ready = not full and self.compose_valid and has_experiment and not missing_components
        for button in launch_buttons(self):  # the experiment tree's and the Queue tab's
            button.setEnabled(ready)
        if self.queue_button is not None:
            self.queue_button.setEnabled(self.compose_valid and has_experiment and not missing_components)
        if self.stop_button is not None:
            self.stop_button.setEnabled(not idle)
        if full and training == 1:
            tip = "Cannot run: A run is currently active. Stop it, or let more run at once in the Queue tab."
            btn_text = "Running..."
        elif full:
            tip = (f"Cannot run: {training} runs are training, the most at once. Stop one, or let more run at "
                   "once in the Queue tab.")
            btn_text = "Running..."
        elif not has_experiment:
            rel = getattr(self, "active_config_rel_path", "") or ""
            if rel and Path(rel).name.startswith("_"):
                tip = "Cannot run: Base templates (_base.yaml) cannot be run directly. Select an experiment."
            else:
                tip = "Cannot run: Select an experiment configuration from the experiment tree."
            btn_text = "No experiment"
        elif missing_components:
            missing_str = ", ".join(missing_components)
            tip = f"Cannot run: Missing component ({missing_str}). Install it via Component Hub."
            btn_text = f"Missing component: {missing_components[0]}"
            if getattr(self, "builder_status", None) is not None:
                self.builder_status.setObjectName("configError")
                self.builder_status.setText(f"✗ Missing component: {missing_str}")
                self.builder_status.setStyle(self.builder_status.style())
        elif not self.compose_valid:
            err = getattr(self, "compose_error", None) or "This configuration has not passed pipeline validation."
            tip = f"Cannot run: {err}"
            btn_text = "Validation failed"
        else:
            exp_name = getattr(self, "current_experiment", "experiment")
            tip = f"Train {exp_name} through the backend (F5)"
            btn_text = "Launch training"
        for button in launch_buttons(self):
            button.setToolTip(tip)
        # Only the experiment tree's button says why in its text; the Queue tab's stays "Launch now"
        # (a missing-component reason does not fit it) and says why in its tooltip.
        if self.start_button is not None and hasattr(self.start_button, "setText"):
            self.start_button.setText(btn_text)
        if hasattr(self, "queue_panel"):
            self.queue_panel.set_experiment(getattr(self, "current_experiment", None),
                                            getattr(self, "active_config_rel_path", "") or "")
            self.queue_panel.stop_button.setToolTip("Stop the active run (Shift+F5)" if not idle else "No run is active")
            self.queue_panel.add_button.setToolTip(
                tip.replace("Cannot run", "Cannot queue") if not ready and not full
                else f"Add {getattr(self, 'current_experiment', 'the experiment')} to the queue (Ctrl+Shift+Q)")
        if getattr(self, "state_label", None) is not None:
            run = self.active
            if run is None:
                self.state_label.setText("Idle")
            elif training > 1:
                self.state_label.setText(f"TRAINING   •   {training} runs  ")
            elif run.get("simulated"):
                self.state_label.setText(f"Demo · {run['status']}")
            else:
                experiment_id = run.get("backend", {}).get("experiment_id", "")
                self.state_label.setText(f"{run['status'].capitalize()} · {experiment_id}" if experiment_id
                                         else run["status"].capitalize())

    def activate(self, run):
        self.active = run  # added to the active runs
        if run not in self.runs:
            self.runs.insert(0, run)
        self.select_run(run)
        self.refresh_runs()
        self.tabs.setCurrentWidget(self.monitor_panel)
        self.update_launch_state()

    # ── Live training (run_pipeline.py as a QProcess, reported through signals) ──

    def launch_training(self):
        if len(self.active_runs) >= self.launch_capacity():
            self.statusBar().showMessage(
                "As many runs are training as may run at once. Stop one, or raise \"Run up to\" in the Queue tab.",
                5000)
            return
        missing = self.check_missing_components()
        if missing:
            self.statusBar().showMessage(f"Cannot run: Missing component ({', '.join(missing)})", 6000)
            return
        if not self.compose_valid:
            self.statusBar().showMessage("Fix config errors before launching.", 5000)
            return
        if hasattr(self, "config_viewer") and self.config_viewer.is_dirty:
            self.config_viewer.save_to_disk()

        config = self.config()
        exp_target = getattr(self, "current_experiment", None)
        if not exp_target:
            if hasattr(self, "config_tree") and (self.config_tree.root_dir / f"experiment/{BASE_EXPERIMENT}.yaml").exists():
                exp_target = BASE_EXPERIMENT
            else:
                self.statusBar().showMessage("Please select or create an experiment to launch.", 5000)
                return
        is_recover = False
        if hasattr(self, "config_viewer") and getattr(self.config_viewer, "raw_data", None):
            is_recover = bool(self.config_viewer.raw_data.get("recover", False))
        experiment_id = self.unique_experiment_id(config.name, is_recover=is_recover)
        overrides = self.config_viewer.get_overrides() if hasattr(self, "config_viewer") else config.overrides()
        overrides = [o for o in overrides if not o.startswith("++experiment_id=")] + [f"++experiment_id='{experiment_id}'"]
        for button in launch_buttons(self):
            button.setEnabled(False)
        self.log(f"Launching {experiment_id}:\n  python run_pipeline.py {exp_target} {' '.join(overrides)}")
        try:
            job = self.jobs.launch(exp_target, overrides)
        except LaunchError as exc:
            self.launch_finished(config, None, str(exc))
            return
        self.launch_finished(config, job, None)

    def config_for_run(self, config):
        """`config` with whatever the composed experiment actually declares.

        self.config() is the experiment builder's record. An experiment launched
        from the tree may not have populated it, so use facts from composition.
        """
        from dataclasses import replace

        facts = getattr(self, "composed_facts", None) or {}
        known = {key: value for key, value in facts.items()
                 if key != "paradigm" and value is not None}
        return replace(config, **known) if known else config

    def launch_finished(self, config, job, error):
        if error:
            self.log(f"Launch failed: {error}")
            self.statusBar().showMessage("Launch failed; see the console.", 8000)
            self.update_launch_state()
            return
        run = new_run(self.config_for_run(config), simulated=False)
        run["paradigm"] = (getattr(self, "composed_facts", None) or {}).get("paradigm")
        run.update(status="starting", log_since=0, metrics_since=0, metrics_byte=0, backend={
            key: job.get(key) for key in ("job_id", "experiment", "group", "experiment_id", "agents", "total_timesteps",
                                          "effective_timesteps")})
        if not self.persist(run):
            return
        self.activate(run)
        self.log(f"Job {job['job_id']} started. Metrics: results/logs/{job['group']}/{job['experiment_id']}/")
        self.on_job_changed(job)  # it may have started (or failed to start) before the run existed

    def run_for_job(self, job_id):
        return next((r for r in self.runs if not r["simulated"] and r["backend"].get("job_id") == job_id), None)

    def reconnect_live_run(self):
        """Runs left live or queued by the last session: training ended with that session (QProcess jobs
        belong to the app), so mark them from the job record. Queued jobs are kept and wait for Start."""
        changed = False
        for run in self.runs:
            if run["simulated"] or run["status"] not in (*LIVE_STATUSES, "queued"):
                continue
            job = self.jobs.job(run["backend"].get("job_id", ""))
            status = job["status"] if job else "interrupted"
            if status in ("pending", "running"):
                status = "interrupted"
            if run["status"] != status:
                run["status"] = status
                self.persist(run)
                changed = True
        if changed:
            self.log("Runs that were training when ThetaIDE last closed are marked interrupted; "
                     "their metrics on disk are kept.")
            self.refresh_runs()

    def on_job_output(self, job_id, lines):
        run = self.run_for_job(job_id)
        if run is not None and run is self.selected and hasattr(self, "monitor"):
            self.monitor.append_log(run, lines)
        if run is None or not any(r is run for r in self.active_runs):
            return
        prefix = f"{run['backend'].get('experiment_id', '')} " if len(self.active_runs) > 1 else ""
        for line in lines:
            self.log(f"│ {prefix}{line}")
            if "Auto-Generating" in line:
                run["phase"] = "plotting"

    def on_job_metrics(self, job_id, agent, rows, reset):
        run = self.run_for_job(job_id)
        if run is None:
            return
        if reset:
            run["metrics"] = []
        if rows:
            run["metrics"].extend(metric_points(rows))
        if self.selected is run:
            self.render_run()
        elif hasattr(self, "monitor"):
            self.monitor.update_run_row(run)  # its sidebar sparkline
        # Rows now arrive as they are logged (the telemetry stream), not in CSV batches: save a training
        # run every few seconds rather than on every row. finish() saves it once more at the end.
        now = time.monotonic()
        if rows and run["status"] == "running" and now - self._persisted_at.get(job_id, 0) >= PERSIST_SECONDS:
            self._persisted_at[job_id] = now
            self.persist(run)

    def on_job_records(self, job_id, agent, records, reset):
        """Keep each agent's LiveRun current, and echo the active run's events to the console."""
        live = self.live_runs.get((job_id, agent))
        if live is None:
            job = self.jobs.job(job_id) or {}
            budget = job.get("effective_timesteps") or job.get("total_timesteps")
            live = self.live_runs[(job_id, agent)] = LiveRun(budget=budget)
        events = live.add(records, reset)
        run = self.run_for_job(job_id)
        if run is not None and run is self.selected and hasattr(self, "monitor"):
            self.monitor.live_changed()
        if run is not None and any(r is run for r in self.active_runs):
            marks = {"warning": "⚠", "error": "✖"}
            for event in events:
                self.log(f"{marks.get(event.get('level'), '◆')} {agent}: {event.get('message', '')}")

    def live_runs_for_job(self, job_id) -> dict[str, LiveRun]:
        """agent -> LiveRun for one job."""
        return {agent: live for (job, agent), live in self.live_runs.items() if job == job_id}

    def live_for_run(self, run, load=True) -> dict[str, LiveRun]:
        """agent -> LiveRun for a run: followed in this session, or read from results/logs (when `load`)."""
        if not run or run.get("simulated"):
            return {}
        backend = run.get("backend") or {}
        followed = self.live_runs_for_job(backend.get("job_id"))
        if followed or not (backend.get("group") and backend.get("experiment_id")):
            return followed
        agents = backend.get("agents") or []
        if not load:
            return self.disk_lives.cached(self.jobs.logs_dir, backend["group"], backend["experiment_id"], agents)
        return self.disk_lives.load(self.jobs.logs_dir, backend["group"], backend["experiment_id"], agents)

    def log_lines_for_run(self, run) -> tuple[list[str], str]:
        """(the pipeline output of a run, what to say when there is none)."""
        if not run or run.get("simulated"):
            return [], "Simulated demo runs have no pipeline output."
        job_id = (run.get("backend") or {}).get("job_id")
        if not job_id:
            return [], "This run was started outside ThetaIDE (command line or Slurm): its output was not captured."
        lines = self.jobs.log_lines(job_id)
        if lines:
            return lines, ""
        path = self.jobs.jobs_dir / f"{job_id}.log"
        try:
            return path.read_text(encoding="utf-8", errors="replace").splitlines(), ""
        except OSError:
            return [], f"No output was saved for this run ({path.name} is missing)."

    def run_folder(self, run):
        backend = (run or {}).get("backend") or {}
        if not (backend.get("group") and backend.get("experiment_id")):
            return None
        return self.jobs.logs_dir / backend["group"] / backend["experiment_id"]

    def on_job_hardware(self, job_id, hardware):
        run = self.run_for_job(job_id)
        if run is not None and run is self.selected and hardware.get("cpu_percent") is not None:
            self.monitor.show_hardware(hardware)

    def on_job_changed(self, job):
        run = self.run_for_job(job["job_id"])
        if run is None:
            return
        status = {"pending": "starting"}.get(job.get("status"), job.get("status"))
        if run["status"] == "queued" and status in ("starting", "running"):
            self.log(f"Queue: {run['backend']['experiment_id']} started.")
            run["status"] = status
            self.persist(run)
            self.activate(run)
            return
        if any(r is run for r in self.active_runs):
            if status in FINAL_STATUSES:
                code = job.get("returncode")
                detail = f" (exit code {code})" if code not in (None, 0) else ""
                if job.get("error"):
                    self.log(f"Job error: {job['error']}")
                self.finish(status, detail, run=run)
            elif run["status"] != status:
                run["status"] = status
                self.update_launch_state()
                self.refresh_runs()
        elif run["status"] != status and status in (*FINAL_STATUSES, "queued"):
            run["status"] = status  # e.g. removed from the queue, or finished while another run was followed
            self.persist(run)
            self.refresh_runs()
            self.render_run()

    # ── Job queue ────────────────────────────────────────────────────────────

    def unique_experiment_id(self, name, is_recover=False):
        """<name>_<date>-<time>, with a counter if several jobs are created within the same second.
        When recovering a checkpoint (is_recover=True), preserves the exact experiment name."""
        if is_recover:
            return name
        group = getattr(self, "current_experiment", "").split("/")[0] if getattr(self, "current_experiment", None) else None
        exists_on_disk = False
        if group:
            from src.app.services import LOGS_DIR
            if (LOGS_DIR / group / name).exists():
                exists_on_disk = True
        if name in self.issued_ids or exists_on_disk:
            base = f"{name}_{datetime.now():%Y%m%d-%H%M%S}"
            experiment_id, n = base, 2
            while experiment_id in self.issued_ids:
                experiment_id, n = f"{base}-{n}", n + 1
            self.issued_ids.add(experiment_id)
            return experiment_id
        self.issued_ids.add(name)
        return name

    def add_to_queue(self):
        missing = self.check_missing_components()
        if missing:
            self.statusBar().showMessage(f"Cannot queue: Missing component ({', '.join(missing)})", 6000)
            return
        if not self.compose_valid:
            self.statusBar().showMessage("Fix config errors before queueing.", 5000)
            return
        if hasattr(self, "config_viewer") and self.config_viewer.is_dirty:
            self.config_viewer.save_to_disk()

        config = self.config()
        exp_target = getattr(self, "current_experiment", None)
        if not exp_target:
            if hasattr(self, "config_tree") and (self.config_tree.root_dir / f"experiment/{BASE_EXPERIMENT}.yaml").exists():
                exp_target = BASE_EXPERIMENT
            else:
                self.statusBar().showMessage("Please select or create an experiment to queue.", 5000)
                return
        is_recover = False
        if hasattr(self, "config_viewer") and getattr(self.config_viewer, "raw_data", None):
            is_recover = bool(self.config_viewer.raw_data.get("recover", False))
        experiment_id = self.unique_experiment_id(config.name, is_recover=is_recover)
        overrides = self.config_viewer.get_overrides() if hasattr(self, "config_viewer") else config.overrides()
        overrides = [o for o in overrides if not o.startswith("++experiment_id=")] + [f"++experiment_id='{experiment_id}'"]
        if self.queue_button is not None:
            self.queue_button.setEnabled(False)
        self.log(f"Queuing {experiment_id}:\n  python run_pipeline.py {exp_target} {' '.join(overrides)}")
        try:
            job = self.jobs.launch(exp_target, overrides, queue=True)
        except LaunchError as exc:
            self.queued(config, None, str(exc))
            return
        self.queued(config, job, None)

    def queued(self, config, job, error):
        if error:
            self.log(f"Could not add to queue: {error}")
            self.statusBar().showMessage("Could not add to queue; see the console.", 8000)
            return
        run = new_run(config, simulated=False)
        run.update(status="queued", log_since=0, metrics_since=0, queue_position=job.get("position"), backend={
            key: job.get(key) for key in ("job_id", "experiment", "group", "experiment_id", "agents", "total_timesteps",
                                          "effective_timesteps")})
        if not self.persist(run):
            return
        self.runs.insert(0, run)
        self.refresh_runs()
        self.queue_running = bool(job.get("queue_running"))
        place = f"position {job['position'] + 1}"
        hint = ("it runs after the jobs ahead of it" if self.queue_running
                else "the queue is paused; press Start queue in the Queue tab to begin")
        self.log(f"Queued {job['experiment_id']} ({place}): {hint}.")
        self.statusBar().showMessage(f"Added {job['experiment_id']} to the queue ({place}); {hint}.", 8000)
        if self.queue_button is not None:
            self.queue_button.setEnabled(self.compose_valid)
        self.on_queue_changed(self.jobs.queue_state())

    def on_queue_changed(self, data):
        if self.queue_running != data["running"]:
            self.queue_running = data["running"]
            self.render_run()
        self.queue_panel.set_data(data)
        positions = {job["job_id"]: job["position"] for job in data.get("queued", [])}
        changed = False
        for run in [r for r in self.runs if r["status"] == "queued"]:
            position = positions.get(run["backend"]["job_id"])
            if position is not None and run.get("queue_position") != position:
                run["queue_position"] = position
                changed = True
        if changed:
            self.refresh_runs()
            self.render_run()

    def set_max_parallel(self, count):
        """Queue tab → Run up to N at once: saved, and a running queue starts more jobs if it now can."""
        self.jobs.set_max_parallel(count)
        if hasattr(self, "queue_panel"):
            self.queue_panel.set_parallel(self.jobs.max_parallel)
        if hasattr(self, "settings_manager"):
            self.settings_manager.set("queue", "max_parallel", int(count))
        self.update_launch_state()

    def set_queue_running(self, running):
        action = "start" if running else "pause"
        data = self.jobs.start_queue() if running else self.jobs.pause_queue()
        self.queue_toggled(action, data, None)

    def queue_toggled(self, action, data, error):
        if error:
            self.log(f"Could not {action} the queue: {error}")
            return
        if action == "start" and not data["running"]:
            self.statusBar().showMessage("The queue is empty; add experiments first.", 5000)
        else:
            self.log("Queue started: jobs will train one after another." if data["running"] else
                     "Queue paused: no new job will start; one already training keeps running.")

    def move_queued(self, job_id, position):
        try:
            self.jobs.move(job_id, position)
        except LaunchError as exc:
            self.log(f"Move failed: {exc}")

    def remove_queued(self, job_id):
        try:
            self.jobs.cancel(job_id)
        except KeyError:
            self.log("Remove failed: that job no longer exists.")

    def plot_ticked_runs(self, run_ids):
        """The runs list's ticks changed: draw those runs in the Plot Viewer (loading their curves if needed)."""
        for run_id in run_ids:
            run = self.by_id(run_id)
            if run is not None:
                self.load_disk_metrics(run)  # command-line runs load their curves on first use
        self.plot_viewer.set_compared(run_ids, self.monitor.sidebar.colors())

    def open_job(self, job_id):
        """Queue → double-click (or Open in Plot Viewer): plot that job's run in the Plot Viewer."""
        run = next((r for r in self.runs if not r["simulated"] and r["backend"].get("job_id") == job_id), None)
        if run is None:
            self.statusBar().showMessage("That job has no run record in this workspace.", 5000)
            return
        self.monitor.sidebar.tick(run["id"])
        self.tabs.setCurrentWidget(self.plot_viewer)

    def view_run(self, run):
        """Open a run from its completion toast in the Plot Viewer."""
        if run["id"] not in self.monitor.sidebar.compared:
            self.monitor.sidebar.tick(run["id"])
        self.tabs.setCurrentWidget(self.plot_viewer)

    def latest_run(self):
        """The run the Training monitor shows: the one training now, else the most recent one."""
        return self.active or (self.runs[0] if self.runs else None)

    # ── Simulated demo (no training environment needed) ──────────────────────

    def demo_run(self):
        return next((run for run in self.active_runs if run["simulated"]), None)

    def start_demo(self):
        if self.demo_run() is not None or len(self.active_runs) >= self.launch_capacity():
            self.statusBar().showMessage("A run is already active. Stop it before starting another.", 4000)
            return
        run = new_run(self.config())
        if not self.persist(run):
            return
        self.activate(run)
        self.log(f"[demo] Started {run['config']['name']} · seed {run['config']['seed']} · {run['id']}")
        self.timer.start()

    def tick(self):
        run = self.demo_run()
        if run is None:
            return
        metric = sample(Config(**run["config"]), len(run["metrics"]) + 1)
        run["metrics"].append(metric)
        self.plot_viewer.update_runs(self.runs)
        if self.selected is run:
            self.render_run()
        if len(run["metrics"]) % 10 == 0:
            self.log(f"[demo] step={metric['step']:>6}  reward={metric['reward']:.2f}  loss={metric['loss']:.4f}")
            self.persist(run)
        if len(run["metrics"]) >= 60:
            self.finish("completed", run=run)

    # ── Shared lifecycle ─────────────────────────────────────────────────────

    def stop_run(self):
        run = self.active
        if not run:
            return
        if run["simulated"]:
            self.finish("stopped", run=run)
            return
        if self.stop_button is not None:
            self.stop_button.setEnabled(False)
        self.log(f"Stopping {run['backend']['experiment_id']}: terminating the pipeline and its training processes…")
        try:
            self.jobs.cancel(run["backend"]["job_id"])  # job_changed reports "cancelled" and finishes the run
        except KeyError:
            self.finish("stopped", run=run)

    def finish(self, status, detail="", run=None):
        """`run` (by default the active run you are looking at) ended with `status`."""
        run = run if run is not None else self.active
        if run is None:
            return
        if run["simulated"]:
            self.timer.stop()
        run["status"] = status
        self.persist(run)
        prefix = "[demo] " if run["simulated"] else ""
        self.log(f"{prefix}{status.capitalize()}{detail} · {run['id']} · config and metrics saved locally")
        if status == "failed":
            self.statusBar().showMessage("Training failed; the console shows the pipeline output.", 10000)
        self.active_runs = [r for r in self.active_runs if r is not run]
        self.run_notifier.run_finished(run, status)
        self.update_launch_state()
        self.refresh_runs()
        if not run["simulated"] and self.backend_connected:
            self.request_disk_runs()
        self.render_run()

    def select_run(self, run):
        self.selected = run
        if hasattr(self, "monitor"):
            self.monitor.show_selected(run)
        self.render_run()
        self.backfill_metrics(run)
        self.load_disk_metrics(run)

    # ── Training monitor (widgets and drawing live in MonitorPanel) ──────────

    def render_run(self):
        if hasattr(self, "monitor"):
            self.monitor.render()

    def toggle_pin_baseline(self):
        self.monitor.toggle_pin_baseline()

    def backfill_metrics(self, run):
        """Fetch the full metrics of a finished trained run recorded before all metrics were kept."""
        if run["simulated"] or run["status"] not in FINAL_STATUSES or run.get("metrics_backfilled"):
            return
        if not run["metrics"] or any("entropy" in m for m in run["metrics"]):
            return
        backend = run["backend"]
        agents = backend.get("agents") or []
        if not agents:
            return
        agent = agents[0]
        self.backend.run_metrics(backend["group"], backend["experiment_id"], agent,
                                 lambda data, error: self.backfill_received(run, data, error))

    def backfill_received(self, run, data, error):
        run["metrics_backfilled"] = True  # try once; the results folder may have been deleted
        if error:
            return
        points = metric_points_from_api(data["metrics"])
        if points:
            run["metrics"] = points
            self.persist(run)
            if self.selected is run:
                self.render_run()

    def refresh_runs(self, *_):
        self.plot_viewer.update_runs(self.runs)
        if hasattr(self, "settings_runs_count_label"):
            self.settings_runs_count_label.setText(f"Total run records:  {len(self.runs)} runs")
        # Sync the Monitor's run selector
        if hasattr(self, "monitor"):
            self.monitor.set_runs(self.runs)

    def request_disk_runs(self):
        """List every results/logs folder (catalog.list_runs on the background thread)."""
        self.backend.list_runs(self.disk_runs_received)
        if hasattr(self, "results_view"):
            self.results_view.refresh()

    def disk_runs_received(self, data, error):
        if error:
            self.log(f"Could not list runs in results/logs: {error}")
            return
        self.disk_entries = {(entry["group"], entry["experiment_id"]): entry for entry in data.get("runs", [])}
        linked = {(run["backend"]["group"], run["backend"]["experiment_id"])
                  for run in self.runs if not run["simulated"] and not run.get("transient")}
        previous = {run["id"]: run for run in self.runs if run.get("transient")}
        self.runs = [run for run in self.runs if not run.get("transient")]
        for key, entry in self.disk_entries.items():
            if key in linked:
                continue  # the app already has a record for this run
            fresh = disk_run(entry)
            run = previous.get(fresh["id"])
            if run is None:
                run = fresh
            else:  # keep the same record (and any metrics already loaded) so selections stay valid
                run.update({k: v for k, v in fresh.items() if k not in ("metrics", "notes")})
            self.runs.append(run)
        self.runs.sort(key=lambda r: r.get("created") or "", reverse=True)
        if self.selected is not None and not any(run is self.selected for run in self.runs):
            self.selected = None  # its folder was deleted
        self.refresh_runs()
        if self.selected is None and self.latest_run() is not None:
            self.select_run(self.latest_run())

    def load_disk_metrics(self, run):
        """Fetch the metrics of a run found in results/logs (the app has none recorded for it)."""
        if run.get("origin") != "cli" or run["metrics"] or run.get("metrics_requested"):
            return
        agents = run["backend"].get("agents") or []
        if not agents:
            return
        run["metrics_requested"] = True
        self.backend.run_metrics(run["backend"]["group"], run["backend"]["experiment_id"], agents[0],
                                 lambda data, error: self.disk_metrics_received(run, data, error))

    def disk_metrics_received(self, run, data, error):
        if error:
            run["metrics_requested"] = False  # allow a retry on the next selection
            self.log(f"Could not load metrics for {run['backend']['experiment_id']}: {error}")
            return
        run["metrics"] = metric_points_from_api(data["metrics"])
        if not run.get("transient"):
            self.persist(run)
        if self.selected is run:
            self.render_run()
        self.refresh_runs()

    def by_id(self, run_id):
        return next((run for run in self.runs if run["id"] == run_id), None)

    def view_selected_plot(self):
        if self.selected:
            self.monitor.sidebar.tick(self.selected["id"])  # the plot draws the ticked runs
        self.tabs.setCurrentWidget(self.plot_viewer)

    def tree_selected(self, item, _):
        pass

    def load_run_config(self, run):
        """Put a run record's builder config into the experiment builder; return why not, or None."""
        cfg_dict = None
        group = run.get("backend", {}).get("group")
        exp_id = run.get("backend", {}).get("experiment_id")
        if group and exp_id:
            from src.app.services import LOGS_DIR
            saved_cfg = LOGS_DIR / group / exp_id / "config.yaml"
            if saved_cfg.is_file():
                try:
                    cfg_dict = yaml.safe_load(saved_cfg.read_text(encoding="utf-8"))
                except Exception:
                    cfg_dict = None
        if cfg_dict is None and hasattr(self, "store"):
            store_cfg = self.store.root / run.get("id", "") / "config.yaml"
            if store_cfg.is_file():
                try:
                    cfg_dict = yaml.safe_load(store_cfg.read_text(encoding="utf-8"))
                except Exception:
                    cfg_dict = None

        if isinstance(cfg_dict, dict):
            self.set_config(cfg_dict)
            self.tabs.setCurrentWidget(self.config_panel)
            self.log(f"Loaded exact configuration from {run['id']}. Launch to create a new run.")
            return None

        if "config" in run and isinstance(run["config"], dict):
            self.set_config(Config(**run["config"]))
            self.tabs.setCurrentWidget(self.config_panel)
            self.log(f"Loaded configuration from {run['id']}. Launch to create a new run.")
            return None

        return "Could not load saved configuration for this run."

    def load_results_config(self, run_key):
        """The Results panel's Load in experiment builder: find the run record for a results/logs folder."""
        group, experiment_id = run_key.split("/", 1)
        run = next((r for r in self.runs if not r["simulated"] and r["backend"].get("group") == group
                    and r["backend"].get("experiment_id") == experiment_id), None)
        if run is None:
            return ("No builder config is saved for this run (only runs Theta-IDE launched or listed have one). "
                    "Its full saved config is in the Config tab.")
        return self.load_run_config(run)

    def export_config(self):
        config = self.config()
        exp_target = getattr(self, "current_experiment", None)
        if not exp_target and hasattr(self, "config_viewer") and getattr(self.config_viewer, "current_path", None):
            cur = self.config_viewer.current_path
            try:
                exp_target = str(cur.relative_to(PROJECT_ROOT / "in" / "config" / "experiment"))
            except Exception:
                exp_target = cur.stem
        if not exp_target:
            exp_target = getattr(config, "name", "experiment")

        raw_yaml = None
        if hasattr(self, "config_viewer") and getattr(self.config_viewer, "raw_data", None):
            raw_yaml = yaml.safe_dump(self.config_viewer.raw_data, sort_keys=False)
        elif hasattr(config, "recipe_yaml"):
            raw_yaml = config.recipe_yaml()

        dialog = ExportExperimentDialog(
            experiment_id=str(exp_target),
            workspace_root=PROJECT_ROOT,
            raw_yaml_text=raw_yaml,
            parent=self,
        )
        if dialog.exec():
            if dialog.export_result and dialog.export_result.success:
                res = dialog.export_result
                self.log(f"Exported Theta Hub component package ({len(res.files_included)} files) to {res.archive_path}.")
                QMessageBox.information(
                    self,
                    "Export Successful",
                    f"Successfully exported experiment component package:\n\n{res.archive_path}\n\nBundled {len(res.files_included)} files (manifest, configs, rules, models).",
                )
            else:
                self.log(f"Exported configuration to {dialog.edit_dest.text()}.")

    def scaffold_plugin(self):
        from .scaffold_dialog import ScaffoldPluginDialog
        dialog = ScaffoldPluginDialog(workspace_root=PROJECT_ROOT, parent=self)
        if dialog.exec() and dialog.scaffold_result and dialog.scaffold_result.success:
            from .config_viewer import invalidate_config_tree
            invalidate_config_tree()
            res = dialog.scaffold_result
            self.log(f"Scaffolded {res.component_id} component plugin at {res.target_dir} ({len(res.created_files)} files).")
            try:
                if hasattr(self, "components_panel") and hasattr(self.components_panel, "components_tree"):
                    self.components_panel.components_tree.populate()
            except Exception as exc:
                self.log(f"Warning during tree refresh: {exc}")
            QMessageBox.information(
                self,
                "Scaffolding Successful",
                f"Successfully scaffolded {res.component_id} component plugin!\n\nTarget directory: {res.target_dir}\nCreated {len(res.created_files)} files.",
            )

    def reload_components(self):
        from src.app.core.hot_reload import reload_component_modules
        reloaded = reload_component_modules()
        msg = f"Hot-reloaded {len(reloaded)} component modules live."
        self.log(msg)
        if hasattr(self, "components_panel") and hasattr(self.components_panel, "components_tree"):
            self.components_panel.components_tree.refresh()
        QMessageBox.information(
            self,
            "Hot-Reload Complete",
            f"{msg}\n\nRefreshed model, agent, and environment registries.",
        )

    def log(self, message):
        if hasattr(self, "console"):
            self.console.appendPlainText(message)

    def console_command(self, cmd_override=None):
        if cmd_override is not None:
            command = cmd_override.strip()
        elif hasattr(self, "command"):
            command = self.command.text().strip()
            self.command.clear()
        else:
            command = ""
        self.log(f"theta › {command}")
        if command == "clear":
            self.console.clear()
        elif command == "help":
            self.log("help    Show commands\nstatus  Show the active run\nconfig  Show the equivalent run_pipeline command\nclear   Clear output\n(Use the Terminal tab for an interactive shell with tmux support.)")
        elif command == "status":
            states = [f"simulated demo {run['status']}" if run["simulated"] else
                      f"{run['status']} · {run['backend']['experiment_id']} · job {run['backend']['job_id']}"
                      for run in self.active_runs]
            self.log(f"Training: {' | '.join(states) or 'idle'} · {len(self.runs)} records")
        elif command == "config":
            self.log(self.config().command())
        elif command:
            self.log(f"Unknown command '{command}'. Type help or switch to the Terminal tab.")

    def quit_app(self):
        """Quit ThetaIDE (File → Quit, the tray's Quit): stops training after confirming."""
        self._quitting = True
        if not self.close():
            self._quitting = False  # the user chose to keep training
            return
        if self.tray is not None:
            self.tray.remove()
        QApplication.quit()

    def closeEvent(self, event):
        if not self._quitting and self.tray is not None and self.tray.keeps_running():
            # Training runs inside this app, so keep the app alive in the tray instead of stopping it
            event.ignore()
            self.tray.hide_to_tray()
            return
        if self.jobs.is_busy():
            # Training runs as a child of this window (QProcess), so closing ends it.
            running = ", ".join(job["experiment_id"] for job in self.jobs.active_jobs())
            answer = QMessageBox.question(
                self, "Training is running",
                f"Closing ThetaIDE stops {running}. Metrics written so far stay in results/logs." + "\n\nClose anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
        self._shutting_down = True
        self.save_session()
        for viewer in self.findChildren(ConfigViewer):
            viewer.flush_pending_save()  # edits typed in the last moment before closing
        for run in list(self.active_runs):
            if run["simulated"]:
                self.finish("stopped", run=run)
            else:
                self.persist(run)
        self.jobs.shutdown()
        if hasattr(self, "plugin_manager") and self.plugin_manager is not None:
            self.plugin_manager.shutdown()
        if hasattr(self, "tensorboard_server") and self.tensorboard_server is not None:
            self.tensorboard_server.stop()
        if hasattr(self, "reloader") and self.reloader is not None:
            self.reloader.stop()
        if hasattr(self, "terminal_panel"):
            for term in self.terminal_panel.terminals:
                term.close()
        if hasattr(self, "hotkey_manager"):
            self.hotkey_manager.cleanup()
        if hasattr(self, "_zoom_persist_timer") and self._zoom_persist_timer.isActive():
            self._zoom_persist_timer.stop()
            self._persist_zoom()
        event.accept()
        if self.tray is not None:
            # With a tray the app doesn't quit when its last window closes, so a real close ends it here
            self.tray.remove()
            QApplication.quit()

    def reload_frontend(self):
        """Reload the frontend process cleanly."""
        from .terminal_reloader import reload_frontend
        reload_frontend(window=self, app=QApplication.instance())


def main():
    parser = argparse.ArgumentParser(description="ThetaIDE PyQt frontend proof of concept")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent.parent / ".thetaide" / "runs")
    parser.add_argument("--install-desktop-entry", action="store_true",
                        help="Linux: add ThetaIDE (with its icon) to the application menu and exit")
    args = parser.parse_args()
    if args.install_desktop_entry:
        entry, icon = install_desktop_entry()
        print(f"Installed {entry}\nInstalled {icon}")
        return 0
    set_windows_app_id()  # before any window: the taskbar then shows ThetaIDE's icon, not python.exe's
    if sys.platform == "darwin":
        try:
            import ctypes
            libc = ctypes.CDLL(None)
            if hasattr(libc, "setprogname"):
                libc.setprogname(b"ThetaIDE")
        except Exception:
            pass

    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True)
    QApplication.setApplicationName("ThetaIDE")
    QApplication.setApplicationDisplayName("ThetaIDE")
    QApplication.setDesktopFileName(DESKTOP_FILE_NAME)
    QApplication.setOrganizationName("ThetaIDE")
    QApplication.setOrganizationDomain("thetaide.org")

    app = QApplication(["ThetaIDE"] + sys.argv[1:])
    app.setApplicationName("ThetaIDE")
    app.setApplicationDisplayName("ThetaIDE")
    app.setStyle("Fusion")
    if sys.platform == "darwin":
        app.setFont(QFont(".AppleSystemUIFont", 10))
    else:
        app.setFont(QFont("Segoe UI", 10))
    app.setStyleSheet(STYLE)
    app.wheel_guard = wheel_guard.install(app)

    if ICON_PATH.is_file():
        app.setWindowIcon(app_icon())  # every window's title bar, the taskbar and (macOS) the Dock

    from .splash import SplashScreen, splash_theme

    splash = SplashScreen(splash_theme(args.data_dir))
    splash.show()
    app.processEvents()  # paint it now: the event loop doesn't run until the window is built
    try:
        window = Window(args.data_dir, progress=splash.set_progress)
    except OSError as exc:
        splash.close()
        QMessageBox.critical(None, "Workspace unavailable", f"Could not open local run storage:\n{exc}")
        return 1
    splash.set_progress("Opening window…", 0.88)
    window.show()
    splash.finish(window)

    from .tray import TrayController

    window.tray = TrayController(window)
    if window.tray.available():
        # The app outlives its window while training in the tray; quit_app() ends it.
        app.setQuitOnLastWindowClosed(False)
    app.aboutToQuit.connect(window.jobs.shutdown)  # any exit path (Ctrl+C included) stops training cleanly

    from .terminal_reloader import TerminalReloader

    reloader = TerminalReloader(window=window, app=app)
    window.reloader = reloader
    app.aboutToQuit.connect(reloader.stop)

    # Clean POSIX signal handling: allow Ctrl+C (SIGINT) and SIGTERM to quit QApplication cleanly
    import signal

    def _sig_handler(*_):
        reloader.stop()
        app.quit()

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)
    sigint_timer = QTimer()
    sigint_timer.start(250)
    sigint_timer.timeout.connect(lambda: None)

    if sys.stdin.isatty():
        sys.stdout.write(
            "\033[90m[Theta-IDE] Ready · Press \033[1;36mCtrl+R\033[0m\033[90m in this terminal to reload frontend\033[0m\r\n"
        )
        sys.stdout.flush()

    reloader.start()

    try:
        return app.exec()
    finally:
        reloader.stop()

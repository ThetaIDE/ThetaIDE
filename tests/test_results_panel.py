"""Results panel (frontend/results): explorer, one chart per metric, hiding plots, axes, smoothing."""

from urllib.parse import urlencode
import os
import sys
import tempfile
from pathlib import Path

import pytest

pytest.importorskip("PyQt6")

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PyQt6.QtCore import Qt  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True)
app = QApplication.instance() or QApplication(sys.argv[:1])

from frontend.results.panel import ResultsPanel  # noqa: E402

PPO_ROWS = [  # Lightning PPO: eval and loss values on separate rows
    {"epoch": "", "eval/reward": "20", "losses/total_loss": "", "time/total": "1", "transitions": "1000"},
    {"epoch": "0", "eval/reward": "", "losses/total_loss": "0.5", "time/total": "", "transitions": "1000"},
    {"epoch": "", "eval/reward": "60", "losses/total_loss": "", "time/total": "2", "transitions": "2000"},
    {"epoch": "1", "eval/reward": "", "losses/total_loss": "0.25", "time/total": "", "transitions": "2000"},
]
SB3_ROWS = [  # SB3 through ContractLogger: its own loss names
    {"step": "1", "transitions": "1000", "eval/reward": "15", "losses/train/value_loss": "3.5"},
    {"step": "2", "transitions": "2000", "eval/reward": "35", "losses/train/value_loss": "2.0"},
]
RUNS = [
    {"group": "cartpole", "experiment_id": "ppo_run", "agents": [{"name": "ppo"}], "finished": True},
    {"group": "sb3", "experiment_id": "sb3_run", "agents": [{"name": "ppo"}], "finished": True},
    {"group": "sb3", "experiment_id": "never_logged", "agents": [], "finished": True},  # nothing to plot: not listed
]
OLD_PPO_ROWS = [{"eval/reward": "5", "transitions": "1000"}]


def _png() -> bytes:
    from PyQt6.QtCore import QBuffer, QIODevice
    from PyQt6.QtGui import QColor, QImage

    image = QImage(8, 6, QImage.Format.Format_RGB32)
    image.fill(QColor("#83a598"))
    buffer = QBuffer()
    buffer.open(QIODevice.OpenModeFlag.WriteOnly)
    image.save(buffer, "PNG")
    return bytes(buffer.data())


# Every file each run produced, as results/-relative path -> (kind, bytes); the manifest lists these.
FILES = {
    "cartpole/ppo_run": {
        "logs/cartpole/ppo_run/config.yaml": ("config", b"seed: 1\nenv:\n  name: cartpole\nlr: 0.001\n"),
        "logs/cartpole/ppo_run/ppo/version_1/metrics.csv": ("metrics", b"transitions,eval/reward\n1000,20\n2000,60\n"),
        "plots/cartpole/ppo_run/time_report.csv": ("table", b"Method,Avg_Time_Seconds,Runs\nppo,14.8,1\niql,9.5,2\n"),
        "plots/cartpole/ppo_run/losses/ppo/losses_total_loss.png": ("figure", None),  # filled with _png() below
        "plots/cartpole/ppo_run/hyperparameters_report.md": ("report", b"# Hyperparameters\n\n| lr | 0.001 |\n"),
        "checkpoints/cartpole/ppo_run/ppo/best_model.ckpt": ("checkpoint", b"\x00weights"),
    },
    "sb3/sb3_run": {
        "logs/sb3/sb3_run/config.yaml": ("config", b"seed: 2\nenv:\n  name: cartpole\nlr: 0.001\n"),
    },
}


def _manifest(run_key, versions):
    group, exp = run_key.split("/")
    artifacts = [
        {"path": path, "kind": kind, "size": len(data or b"x"), "modified": 1.0, "agent": None, "version": None}
        for path, (kind, data) in FILES.get(run_key, {}).items()
    ]
    return {"group": group, "experiment_id": exp, "agents": {"ppo": versions}, "artifacts": artifacts}


class LocalBackendCalls:
    """LocalBackend's results methods for a fake that answers the old API paths (GET /api/runs/...).

    The fakes keep recording requests as those paths, so tests can check what was asked for and with which
    version/since, the same as when the Results panel talked to the FastAPI server.
    """

    def list_runs(self, callback, *, completed_only=False):
        self.get("/api/runs", callback)

    def run_manifest(self, group, experiment_id, callback, *, completed_only=False):
        self.get(f"/api/runs/{group}/{experiment_id}/manifest", callback)

    def run_metrics(self, group, experiment_id, agent, callback, version=None, since=0, *, completed_only=False):
        params = {key: value for key, value in (("version", version), ("since", since)) if value}
        query = f"?{urlencode(params)}" if params else ""
        self.get(f"/api/runs/{group}/{experiment_id}/{agent}/metrics{query}", callback)

    def run_file(self, group, experiment_id, path, callback):
        self.get_bytes(f"/api/runs/{group}/{experiment_id}/files/{path}", callback)


class FakeBackend(LocalBackendCalls):
    """Answers like the real API, synchronously. `offline` makes every request fail."""

    def __init__(self):
        self.requests = []
        self.offline = False

    def get(self, path, callback):
        self.requests.append(path)
        if self.offline:
            callback(None, "Connection refused")
        elif path == "/api/runs":
            callback({"runs": RUNS}, None)
        elif path == "/api/runs/cartpole/ppo_run/manifest":
            callback(_manifest("cartpole/ppo_run", ["version_0", "version_1"]), None)
        elif path.endswith("/manifest"):
            group, exp = path.split("/")[3:5]
            callback(_manifest(f"{group}/{exp}", ["version_0"]), None)
        elif path == "/api/runs/cartpole/ppo_run/ppo/metrics?version=version_0":
            callback({"metrics": OLD_PPO_ROWS, "columns": ["eval/reward", "transitions"]}, None)
        elif path.startswith("/api/runs/cartpole/ppo_run/ppo/metrics"):
            callback({"metrics": PPO_ROWS, "columns": list(PPO_ROWS[0])}, None)
        elif path.startswith("/api/runs/sb3/sb3_run/ppo/metrics"):
            callback({"metrics": SB3_ROWS, "columns": list(SB3_ROWS[0])}, None)
        else:
            callback(None, f"HTTP 404: {path}")

    def get_bytes(self, path, callback):
        self.requests.append(path)
        group, exp, rest = path.removeprefix("/api/runs/").split("/", 2)
        artifact = rest.removeprefix("files/")
        kind, data = FILES.get(f"{group}/{exp}", {}).get(artifact, (None, None))
        if kind == "figure":
            data = _png()
        callback(data, None) if data is not None else callback(None, f"HTTP 404: {path}")


class FakeSettings:
    def __init__(self, data=None):
        self.data = data or {}

    def get(self, *keys, default=None):
        node = self.data
        for key in keys:
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node

    def set(self, *keys_and_value):
        *keys, value = keys_and_value
        node = self.data
        for key in keys[:-1]:
            node = node.setdefault(key, {})
        node[keys[-1]] = value


def tree_items(panel):
    from frontend.results.explorer import _DATA, _KIND
    tree = panel.explorer.tree
    stack = [tree.topLevelItem(i) for i in range(tree.topLevelItemCount())]
    items = {}
    while stack:
        item = stack.pop()
        items[item.text(0)] = item
        if item.data(0, _KIND) == "run":
            items[item.data(0, _DATA).rsplit("/", 1)[-1]] = item
        stack.extend(item.child(i) for i in range(item.childCount()))
    return items


def make_panel(settings=None):
    backend = FakeBackend()
    panel = ResultsPanel(backend, settings)
    panel.refresh()
    return panel, backend


def test_artifact_context_menu_renames_and_refreshes_all_browsers(monkeypatch):
    from pathlib import PurePosixPath
    from PyQt6.QtCore import QPoint
    from PyQt6.QtWidgets import QInputDialog, QMenu

    run_key = "cartpole/ppo_run"
    old_path = f"plots/{run_key}/hyperparameters_report.md"
    new_path = f"plots/{run_key}/renamed.md"
    monkeypatch.setitem(FILES, run_key, dict(FILES[run_key]))
    panel, backend = make_panel()
    panel.explorer.select_run("cartpole", "ppo_run")
    files, reports = panel.browser("Files"), panel.browser("Reports")
    panel.update_artifacts()
    assert files.select(run_key, old_path)
    assert reports.select(run_key, old_path)
    assert (run_key, old_path) in panel.loader.files
    calls = []

    def rename(group, experiment, path, name, callback):
        calls.append((group, experiment, path, name))
        destination = str(PurePosixPath(path).with_name(name))
        FILES[run_key][destination] = FILES[run_key].pop(path)
        callback(destination, None)

    monkeypatch.setattr(backend, "rename_run_file", rename, raising=False)
    monkeypatch.setattr(QInputDialog, "getText", lambda *args, **kwargs: ("renamed.md", True))
    item = files.tree.currentItem()
    assert files.select(run_key, f"logs/{run_key}/config.yaml")
    monkeypatch.setattr(files.tree, "itemAt", lambda position: item)

    def choose_rename(menu, position):
        assert [action.text() for action in menu.actions()] == ["Rename…", "Save a copy…"]
        menu.actions()[0].trigger()

    monkeypatch.setattr(QMenu, "exec", choose_rename)
    files._context_menu(QPoint())
    assert calls == [("cartpole", "ppo_run", old_path, "renamed.md")]
    assert (run_key, old_path) not in panel.loader.files
    for browser in (files, reports):
        assert browser.current[1]["path"] == new_path
        assert browser.tree.currentItem().text(0) == "renamed.md"
        assert "Hyperparameters" in browser.viewers["text"].browser.toPlainText()
        assert not browser.select(run_key, old_path)
    panel.close()


def test_artifact_rename_cancel_and_failure_preserve_current_file(monkeypatch):
    from PyQt6.QtWidgets import QInputDialog, QMessageBox

    panel, backend = make_panel()
    panel.explorer.select_run("cartpole", "ppo_run")
    browser = panel.browser("Files")
    panel.update_artifacts()
    path = "plots/cartpole/ppo_run/hyperparameters_report.md"
    assert browser.select("cartpole/ppo_run", path)
    calls, errors = [], []

    def rename(*args):
        calls.append(args[:-1])
        args[-1](None, "A file with that name already exists")

    monkeypatch.setattr(backend, "rename_run_file", rename, raising=False)
    monkeypatch.setattr(QMessageBox, "warning", lambda parent, title, message: errors.append(message))
    monkeypatch.setattr(QInputDialog, "getText", lambda *args, **kwargs: ("existing.md", False))
    browser._ask_rename(*browser.current)
    assert not calls
    monkeypatch.setattr(QInputDialog, "getText", lambda *args, **kwargs: ("existing.md", True))
    browser._ask_rename(*browser.current)
    assert len(calls) == 1 and errors == ["A file with that name already exists"]
    assert browser.current[1]["path"] == path
    assert browser.select("cartpole/ppo_run", path)
    panel.close()


@pytest.mark.parametrize("kind", ["group", "experiment", "run"])
def test_main_results_context_menu_renames_on_disk(tmp_path, monkeypatch, kind, workspace_results):
    import json
    from PyQt6.QtCore import QPoint
    from PyQt6.QtWidgets import QInputDialog, QMenu
    from src.app.services import catalog
    from src.app.services.result_rename import rename_results
    from frontend.results.explorer import _DATA, _KIND

    for run in ("run", "run_20261110-120000"):
        logs = workspace_results / "logs/group" / run
        version = logs / "agent/version_0"
        version.mkdir(parents=True)
        (logs / "config.yaml").write_text("seed: 1")
        (version / "metrics.csv").write_text("transitions,eval/reward\n1,10\n")
        (version / "runtime.json").write_text(json.dumps({"end_time_iso": "2026-11-10T12:00:00Z"}))

    def answer(fn, callback, *args, **kwargs):
        try:
            data = fn(*args, **kwargs)
        except Exception as exc:
            callback(None, str(exc))
        else:
            callback(data, None)

    class Backend:
        def list_runs(self, callback, **kwargs):
            callback({"runs": catalog.list_runs(**kwargs)}, None)

        def run_manifest(self, group, run, callback, **kwargs):
            answer(catalog.run_manifest, callback, group, run, **kwargs)

        def run_metrics(self, group, run, agent, callback, **kwargs):
            answer(catalog.run_metrics, callback, group, run, agent, **kwargs)

        def run_file(self, group, run, path, callback):
            answer(catalog.read_run_file, callback, group, run, path)

        def rename_results(self, kind, group, run, name, callback):
            answer(rename_results, callback, kind, group, run, name)

    panel = ResultsPanel(Backend())
    panel.refresh()
    target = next(item for item in panel.explorer._items() if item.data(0, _KIND) == kind
                  and (kind != "run" or item.data(0, _DATA) == "group/run"))
    monkeypatch.setattr(panel.explorer.tree, "itemAt", lambda position: target)
    prompts = []

    def name(*args, **kwargs):
        prompts.append(args[1])
        return "renamed", True

    monkeypatch.setattr(QInputDialog, "getText", name)

    def choose_rename(menu, position):
        action = next(action for action in menu.actions() if action.text() == "Rename…")
        action.trigger()

    monkeypatch.setattr(QMenu, "exec", choose_rename)
    panel.explorer._context_menu(QPoint())
    assert prompts == [f"Rename {kind}"]
    renamed_key = "renamed/run" if kind == "group" else "group/renamed"
    assert renamed_key in {f"{entry['group']}/{entry['experiment_id']}" for entry in catalog.list_runs()}
    assert panel.explorer.tree.currentItem() is not None
    assert panel.run_keys()[0].startswith("renamed/" if kind == "group" else "group/renamed")
    browser = panel.browser("Files")
    panel.update_artifacts()
    assert browser.artifacts()
    assert all((workspace_results / artifact["path"]).is_file() for _, artifact in browser.artifacts())
    assert "eval/reward" in panel.grid.cards
    panel.close()


@pytest.mark.parametrize("tree_name", ["runs", "files"])
def test_results_tree_splitters_resize_without_snapping_closed(tree_name):
    panel, _ = make_panel()
    panel.resize(1800, 800)
    panel.show()
    app.processEvents()
    tree = panel.explorer if tree_name == "runs" else panel.browser("Files").tree
    splitter = tree.parentWidget()
    try:
        splitter.moveSplitter(240, 1)
        app.processEvents()
        assert abs(tree.width() - 240) <= 2
        splitter.moveSplitter(320, 1)
        app.processEvents()
        assert abs(tree.width() - 320) <= 2
        splitter.moveSplitter(0, 1)
        app.processEvents()
        assert tree.width() > 0
        splitter.moveSplitter(280, 1)
        app.processEvents()
        assert abs(tree.width() - 280) <= 2
    finally:
        panel.close()
        panel.deleteLater()


def tick(panel, *names):
    items = tree_items(panel)
    for name in names:
        items[name].setCheckState(0, Qt.CheckState.Checked)


def test_results_run_tree_drag_after_visiting_curves():
    from PyQt6.QtCore import QPoint
    from PyQt6.QtTest import QTest

    panel, _ = make_panel()
    panel.resize(1100, 800)
    panel.show()
    app.processEvents()
    panel.explorer.select_run("cartpole", "ppo_run")
    panel.tabs.setCurrentWidget(panel.curves_page)
    app.processEvents()
    panel.tabs.setCurrentWidget(panel._pages["Files"])
    app.processEvents()
    splitter = panel.explorer.parentWidget()
    try:
        for target in (400, 240, 480, 280):
            handle = splitter.handle(1)
            QTest.mousePress(handle, Qt.MouseButton.LeftButton, pos=handle.rect().center())
            destination = splitter.mapToGlobal(QPoint(target + handle.width() // 2, handle.height() // 2))
            QTest.mouseMove(handle, handle.mapFromGlobal(destination))
            QTest.mouseRelease(handle, Qt.MouseButton.LeftButton, pos=handle.mapFromGlobal(destination))
            app.processEvents()
            assert abs(panel.explorer.width() - target) <= 2
    finally:
        panel.close()
        panel.deleteLater()


@pytest.mark.parametrize("tab", ["Files", "Reports", "Config"])
def test_results_preview_content_does_not_pin_tree_width(tab):
    panel, _ = make_panel()
    panel.resize(1100, 800)
    browser = panel.browser(tab)
    panel.tabs.setCurrentWidget(panel._pages[tab])
    panel.show()
    app.processEvents()
    splitter = browser.tree.parentWidget()
    try:
        # An unbroken filename in the info preview used to raise the minimum
        # width of both nested splitters, preventing either tree from resizing.
        browser.info.show_info({"path": "checkpoints/" + "weights_" * 40 + ".ckpt"})
        app.processEvents()
        if tab == "Config":
            assert browser.tree.isHidden() and browser.title.isHidden()
            return
        for target in (400, 200, 320):
            splitter.moveSplitter(target, 1)
            app.processEvents()
            assert abs(browser.tree.width() - target) <= 2
        assert panel.width() == 1100
    finally:
        panel.close()
        panel.deleteLater()


def series_names(panel, metric):
    return [title for title, _, _ in panel.grid.cards[metric].chart.series]


def test_explorer_lists_runs_that_logged_metrics():
    panel, _ = make_panel()
    items = tree_items(panel)
    assert {"cartpole", "sb3", "ppo_run", "sb3_run"} <= set(items)
    assert "never_logged" not in items
    assert panel.root.indexOf(panel.status) == -1
    assert panel.grid.cards == {}  # nothing ticked yet


def test_offline_backend_explains_itself():
    backend = FakeBackend()
    backend.offline = True
    panel = ResultsPanel(backend)
    panel.refresh()
    assert "Could not load" in panel.status.text()
    panel.set_offline()
    assert "Training pipeline unavailable" in panel.status.text()


def test_window_hosts_the_panel_as_the_results_tab():
    from frontend.app import Window

    with tempfile.TemporaryDirectory() as data_dir:
        window = Window(Path(data_dir))
        try:
            assert window.results_panel is window.results_view  # the Run list tab is gone
            assert not hasattr(window, "table")
        finally:
            window.close()


# ── Phase 4: every data point ────────────────────────────────────────────────


def browser_select(panel, tab, suffix):
    browser = panel.browser(tab)
    run_key, artifact = next((k, a) for k, a in browser.artifacts() if a["path"].endswith(suffix))
    assert browser.select(run_key, artifact["path"])
    return browser


# ── Phase 5: comparisons ─────────────────────────────────────────────────────


def set_combine(panel, mode):
    panel.combine_select.setCurrentIndex(panel.combine_select.findData(mode))


def set_primary(panel, name):
    panel.primary_select.setCurrentIndex(panel.primary_select.findData(name))


def test_chart_hover_snaps_only_to_the_primary_series():
    from PyQt6.QtCore import QEvent, QPointF
    from PyQt6.QtGui import QMouseEvent

    from frontend.charts.chart import Chart

    chart = Chart("m", "M")
    chart.resize(400, 300)
    chart.set_series(
        [
            ("a", [{"step": 0, "m": 1.0}, {"step": 450, "m": 2.0}, {"step": 1000, "m": 3.0}], "primary"),
            ("b", [{"step": 500, "m": 5.0}], "primary"),
        ],
        xmax=1000,
    )
    hovered = []
    chart.point_hovered.connect(hovered.append)

    def hover_at(step):
        x = 52 + (chart.width() - 74) * step / 1000
        position = QPointF(x, 150)
        chart.mouseMoveEvent(
            QMouseEvent(
                QEvent.Type.MouseMove,
                position,
                Qt.MouseButton.NoButton,
                Qt.MouseButton.NoButton,
                Qt.KeyboardModifier.NoModifier,
            )
        )

    hover_at(490)
    assert hovered[-1] == 500.0  # Nearest: b's point is closest to the mouse
    chart.primary = "a"
    chart.hover_point = None
    hover_at(490)
    assert hovered[-1] == 450.0  # primary a: snaps to a's own logged step


def grid_widgets(panel):
    """Cards and headers the plot grid currently lays out (folded or unfocused cards are left out)."""
    layout = panel.grid.grid
    return [layout.itemAt(i).widget() for i in range(layout.count())]


def test_a_quick_relayout_does_not_open_the_empty_label_as_a_window():
    """Metrics arriving right after "Loading metrics…" must not leave that label floating in its own window.

    Adding a widget to a visible layout queues a call that shows it. If the grid was laid out again before
    that call ran, the call used to show the detached, parentless label as a top-level window.
    """
    from frontend.results.plot_grid import PlotGrid

    grid = PlotGrid()
    grid.show()
    app.processEvents()
    grid.set_metrics([], "Loading metrics…")  # queues the label's show…
    grid.set_metrics(["eval/reward"])  # …and the metrics arrive before it runs
    app.processEvents()
    floating = [w for w in app.topLevelWidgets() if w.isVisible() and w is not grid and w.parent() is None
                and getattr(w, "text", lambda: "")() == "Loading metrics…"]
    assert floating == []
    assert not grid.empty.isVisible() and grid.cards["eval/reward"].isVisible()
    grid.set_metrics([], "Nothing to plot")  # and the label still shows when it is placed again
    app.processEvents()
    assert grid.empty.isVisible() and grid.empty.parent() is grid.body
    grid.close()
    grid.deleteLater()


def test_the_run_selector_shares_the_experiment_trees_header_behaviour():
    """Search, Esc, collapse and expand come from the same code as the Experiment tab's tree (tree_tools)."""
    from PyQt6.QtCore import QEvent
    from PyQt6.QtGui import QKeyEvent

    from frontend.config_tree import ConfigTreeWidget
    from frontend.results.explorer import RunExplorer
    from frontend.tree_tools import GroupItemDelegate, TreeHeaderActions

    explorer = RunExplorer()
    assert isinstance(explorer, TreeHeaderActions) and issubclass(ConfigTreeWidget, TreeHeaderActions)
    assert isinstance(explorer.tree.itemDelegate(), GroupItemDelegate)
    explorer.show()
    assert explorer.search.isHidden()
    explorer.btn_search.click()
    assert explorer.search.isVisible() and explorer.btn_search.isChecked()
    explorer.search.setText("cart")
    QApplication.sendEvent(explorer.search, QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier))
    assert explorer.search.isHidden() and not explorer.btn_search.isChecked() and explorer.search.text() == ""
    explorer.close()
    explorer.deleteLater()


def test_open_run_selects_its_curves_after_listing():
    panel, _ = make_panel()
    tick(panel, "sb3_run")
    panel.tabs.setCurrentWidget(panel.summary)
    panel.open_run("cartpole", "ppo_run")
    assert panel.tabs.currentWidget() is panel.curves_page
    assert panel.run_keys() == ["cartpole/ppo_run"]
    assert "eval/reward" in panel.grid.cards


def test_failed_run_without_metrics_opens_error_log(monkeypatch):
    run_key = "sb3/never_logged"
    path = f"logs/{run_key}/job-1.stderr.log"
    monkeypatch.setitem(FILES, run_key, {path: ("log", b"traceback here")})
    monkeypatch.setitem(RUNS[2], "has_run_logs", True)
    panel, backend = make_panel()
    panel.open_run("sb3", "never_logged", failure=True, job_id="job-1")
    browser = panel.browser("Files")
    assert panel.tabs.currentWidget() is panel._pages["Files"]
    assert browser.current[1]["path"] == path
    assert "traceback here" in browser.viewers["text"].browser.toPlainText()
    assert not any("never_logged//metrics" in request for request in backend.requests)
    panel.close()


def test_saved_sweep_results_load_after_restart_and_open_recorded_trial_metrics(monkeypatch):
    import json

    key = "cartpole/ppo_run"
    base = f"checkpoints/{key}/ppo"
    summary = {"study": "search", "direction": "maximize", "objective_metric": "eval/reward",
               "best_trial_id": 7, "trials": [{"id": 7, "state": "complete", "value": 90,
               "parameters": {"lr": 0.01}, "duration_seconds": 5, "log_version": "version_1",
               "intermediate_values": {"1": 10, "2": 90}}]}
    files = dict(FILES[key])
    files[f"{base}/tuning_summary.json"] = ("metadata", json.dumps(summary).encode())
    files[f"{base}/sweeps/search/tuning_summary.json"] = ("metadata", json.dumps(summary).encode())
    files[f"{base}/sweeps/older/tuning_summary.json"] = ("metadata", json.dumps(dict(summary, study="older")).encode())
    monkeypatch.setitem(FILES, key, files)
    for _ in range(2):  # fresh panels have no live monitor state or in-memory study data
        panel, backend = make_panel()
        panel.explorer.select_run("cartpole", "ppo_run")
        assert panel.sweeps_view is None
        panel.tabs.setCurrentWidget(panel.sweeps_page)
        view = panel.sweeps_view
        assert view.selector.count() == 2  # the mirrored latest snapshot is deduplicated
        index = next(i for i, (_, data) in enumerate(view.studies) if data["study"] == "search")
        view.selector.setCurrentIndex(index)
        assert view.table.rowCount() == 1
        assert "90" in view.summary.text()
        assert view.chart_type.findText("Parameter: lr") >= 0
        view.chart_type.setCurrentText("Intermediate values")
        assert view.plot.series[0][1] == [(1.0, 10.0), (2.0, 90.0)]
        view.table.selectRow(0)
        assert view.open_button.isEnabled()
        view.open_button.click()
        assert panel.tabs.currentWidget() is panel._pages["Files"]
        assert panel.browser("Files").current[1]["path"] == f"logs/{key}/ppo/version_1/metrics.csv"
        panel.explorer.select_run("sb3", "sb3_run")
        assert view.selector.count() == 0
        assert view.table.rowCount() == 0
        assert not view.open_button.isEnabled()
        panel.close()



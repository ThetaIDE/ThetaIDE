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

from frontend.results.data import Source  # noqa: E402
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
    tree = panel.explorer.tree
    stack = [tree.topLevelItem(i) for i in range(tree.topLevelItemCount())]
    items = {}
    while stack:
        item = stack.pop()
        items[item.text(0)] = item
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


def test_ticking_runs_plots_every_metric_they_logged():
    panel, _ = make_panel()
    tick(panel, "ppo_run", "sb3_run")  # ticking an experiment ticks its agents
    # One chart per metric across both engines, nothing whitelisted; reward is shared by both runs.
    assert set(panel.grid.cards) == {"eval/reward", "losses/total_loss", "losses/train/value_loss", "time/total"}
    assert series_names(panel, "eval/reward") == ["ppo_run / ppo (version_1)", "sb3_run / ppo"]
    assert series_names(panel, "losses/train/value_loss") == ["sb3_run / ppo"]
    reward = dict((t, p) for t, p, _ in panel.grid.cards["eval/reward"].chart.series)
    assert reward["sb3_run / ppo"] == [{"step": 1000.0, "eval/reward": 15.0}, {"step": 2000.0, "eval/reward": 35.0}]
    assert "sb3_run / ppo" in panel.legend.text()


def test_multi_version_agents_offer_each_version_with_the_newest_ticked():
    panel, _ = make_panel()
    tick(panel, "ppo_run")
    items = tree_items(panel)
    assert items["version_1"].checkState(0) == Qt.CheckState.Checked
    assert items["version_0"].checkState(0) == Qt.CheckState.Unchecked
    assert panel.sources == [Source("cartpole", "ppo_run", "ppo", "version_1")]
    items["version_0"].setCheckState(0, Qt.CheckState.Checked)  # overlay the older version too
    assert series_names(panel, "eval/reward") == ["ppo_run / ppo (version_0)", "ppo_run / ppo (version_1)"]


def test_hiding_plots_is_remembered_in_settings():
    settings = FakeSettings()
    panel, _ = make_panel(settings)
    tick(panel, "sb3_run")
    panel.grid.cards["losses/train/value_loss"].hide_button.click()
    assert "losses/train/value_loss" not in panel.grid.cards
    assert settings.data == {"results": {"hidden_plots": ["losses/train/value_loss"]}}
    assert "1 hidden" in panel.plots_button.text()
    # A new panel (e.g. after restarting the app) keeps it hidden; Show all brings it back.
    again, _ = make_panel(settings)
    tick(again, "sb3_run")
    assert set(again.grid.cards) == {"eval/reward"}
    again.show_all()
    assert set(again.grid.cards) == {"eval/reward", "losses/train/value_loss"}
    assert settings.data["results"]["hidden_plots"] == []


def test_plots_menu_lists_every_metric_with_its_visibility():
    panel, _ = make_panel(FakeSettings({"results": {"hidden_plots": ["eval/reward"]}}))
    tick(panel, "sb3_run")
    panel._fill_plots_menu()
    actions = {a.text(): a for a in panel.plots_menu.actions() if a.isCheckable()}
    assert actions["Episode reward  ·  eval/reward"].isChecked() is False
    assert actions["Value loss  ·  losses/train/value_loss"].isChecked() is True
    actions["Episode reward  ·  eval/reward"].setChecked(True)  # show it again from the menu
    assert "eval/reward" in panel.grid.cards


def test_x_axis_choice_and_smoothing():
    panel, _ = make_panel()
    tick(panel, "ppo_run")
    axes = [panel.axis_select.itemData(i) for i in range(panel.axis_select.count())]
    assert axes == ["transitions", "epoch", "time/total"]
    panel.axis_select.setCurrentIndex(axes.index("epoch"))
    ((_, loss, _),) = panel.grid.cards["losses/total_loss"].chart.series
    assert [p["step"] for p in loss] == [0.0, 1.0]
    panel.smoothing_slider.setValue(50)
    ((_, loss, _),) = panel.grid.cards["losses/total_loss"].chart.series
    assert [p["losses/total_loss"] for p in loss] == [0.5, 0.375]


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


def test_point_inspector_shows_every_metric_of_every_run():
    panel, _ = make_panel()
    tick(panel, "ppo_run", "sb3_run")
    panel.grid.cards["eval/reward"].chart.point_hovered.emit(2000.0)
    values = {s.experiment_id: v for s, v in panel.inspector.values.items()}
    # Lightning wrote reward and loss on separate rows; the inspector merges them for the step.
    assert values["ppo_run"] == {
        "transitions": 2000.0,
        "eval/reward": 60.0,
        "time/total": 2.0,
        "epoch": 1.0,
        "losses/total_loss": 0.25,
    }
    assert values["sb3_run"]["losses/train/value_loss"] == 2.0
    assert panel.inspector.table.rowCount() == 7  # every metric of both runs, including the axes
    # Between logged points, each run shows its nearest one and the caption says so.
    panel.grid.point_hovered.emit(1400.0)
    assert {s.experiment_id: v["transitions"] for s, v in panel.inspector.values.items()} == {
        "ppo_run": 1000.0,
        "sb3_run": 1000.0,
    }
    assert "nearest logged point" in panel.inspector.caption.text()


def test_files_tab_shows_every_table_row_and_plots_numeric_columns():
    panel, _ = make_panel()
    tick(panel, "ppo_run")
    names = [a["path"].rsplit("/", 1)[1] for _, a in panel.browser("Files").artifacts()]
    assert {"metrics.csv", "time_report.csv"} <= set(names)
    browser = browser_select(panel, "Files", "time_report.csv")
    viewer = browser.viewers["table"]
    assert browser.stack.currentWidget() is viewer
    assert (viewer.table.rowCount(), viewer.table.columnCount()) == (2, 3)
    assert viewer.table.item(1, 0).text() == "iql"
    assert viewer.plot.isVisibleTo(browser) and list(viewer.columns) == ["Avg_Time_Seconds", "Runs"]
    assert "2 rows × 3 columns" in browser.detail.text()


def test_figures_reports_and_text_are_previewed():
    panel, _ = make_panel()
    tick(panel, "ppo_run")
    figures = browser_select(panel, "Files", "losses_total_loss.png")
    assert figures.stack.currentWidget() is figures.viewers["figure"] and "8 × 6 px" in figures.detail.text()
    assert figures.title.text() == "plots/losses/ppo/losses_total_loss.png"  # path inside the run's folders
    reports = browser_select(panel, "Reports", "hyperparameters_report.md")
    assert "Hyperparameters" in reports.viewers["text"].browser.toPlainText()


def test_config_tab_shows_saved_yaml_for_selected_runs():
    panel, _ = make_panel()
    tick(panel, "ppo_run")
    browser = panel.browser("Config")
    assert "seed: 1" in browser.viewers["text"].browser.toPlainText()
    tick(panel, "sb3_run")
    assert len(browser.artifacts()) == 2
    assert browser.select("sb3/sb3_run", "logs/sb3/sb3_run/config.yaml")
    assert "seed: 2" in browser.viewers["text"].browser.toPlainText()


def test_config_tab_prefers_saved_experiment_yaml(monkeypatch):
    key = "logs/cartpole/ppo_run/experiment.yaml"
    monkeypatch.setitem(FILES["cartpole/ppo_run"], key, ("config", b"seed: 42\n"))
    panel, _ = make_panel()
    tick(panel, "ppo_run")
    browser = panel.browser("Config")
    assert [a["path"] for _, a in browser.artifacts()] == [key]
    assert "seed: 42" in browser.viewers["text"].browser.toPlainText()


def test_files_tab_reaches_every_artifact_and_saves_copies(tmp_path):
    panel, _ = make_panel()
    tick(panel, "ppo_run", "sb3_run")
    listed = {a["path"] for _, a in panel.browser("Files").artifacts()}
    assert listed == {path for files in FILES.values() for path in files}  # every file of both runs
    browser = browser_select(panel, "Files", "best_model.ckpt")
    assert browser.stack.currentWidget() is browser.info
    assert "No preview for this kind of file" in browser.info.text.text()
    target = tmp_path / "copy.ckpt"
    browser.save_copy(str(target))
    assert target.read_bytes() == b"\x00weights"


def test_files_tab_shows_directories_and_preserves_expansion():
    panel, _ = make_panel()
    tick(panel, "ppo_run")
    browser = panel.browser("Files")
    assert browser.tree.columnCount() == 1 and browser.tree.isHeaderHidden()
    assert browser.select("cartpole/ppo_run", "plots/cartpole/ppo_run/losses/ppo/losses_total_loss.png")
    file_item = browser.tree.currentItem()
    assert file_item.text(0) == "losses_total_loss.png"
    assert [file_item.parent().text(0), file_item.parent().parent().text(0),
            file_item.parent().parent().parent().text(0)] == ["ppo", "losses", "plots"]
    browser.set_manifests(panel.manifests())
    file_item = browser.tree.currentItem()
    assert file_item and file_item.parent().isExpanded()
    browser.tree.setCurrentItem(file_item.parent())
    assert browser.current is None and not browser.save_button.isEnabled()


def test_artifact_tabs_are_built_when_first_opened():
    panel, _ = make_panel()
    tick(panel, "ppo_run")
    names = [panel.tabs.tabText(i) for i in range(panel.tabs.count())]
    assert names[0] == "Files" and "Figures" not in names and "Tables" not in names
    assert list(panel.browsers) == ["Files"]
    panel.tabs.setCurrentIndex(names.index("Reports"))
    assert list(panel.browsers) == ["Files", "Reports"]
    assert [a["path"].rsplit("/", 1)[1] for _, a in panel.browsers["Reports"].artifacts()] == ["hyperparameters_report.md"]


def test_files_too_large_to_preview_are_not_downloaded():
    panel, backend = make_panel()
    tick(panel, "ppo_run")
    browser = panel.browser("Files")
    run_key, artifact = next((k, a) for k, a in browser.artifacts() if a["path"].endswith("time_report.csv"))
    artifact["size"] = 10**9
    before = len(backend.requests)
    browser.select(run_key, artifact["path"])
    assert "Too large to preview" in browser.info.text.text() and len(backend.requests) == before


# ── Phase 5: comparisons ─────────────────────────────────────────────────────


def set_combine(panel, mode):
    panel.combine_select.setCurrentIndex(panel.combine_select.findData(mode))


def test_combining_versions_draws_one_mean_curve_with_a_band():
    panel, _ = make_panel()
    tick(panel, "ppo_run")
    tree_items(panel)["version_0"].setCheckState(0, Qt.CheckState.Checked)  # both versions ticked
    assert series_names(panel, "eval/reward") == ["ppo_run / ppo (version_0)", "ppo_run / ppo (version_1)"]
    set_combine(panel, "std")
    chart = panel.grid.cards["eval/reward"].chart
    ((name, points, _),) = chart.series
    assert name == "ppo_run / ppo (mean of 2)" and chart.band == "__spread"
    first, second = points
    # version_0 reward 5 and version_1 reward 20 at step 1000: mean 12.5, sample std 10.61.
    assert (first["step"], first["eval/reward"], first["n"]) == (1000.0, 12.5, 2)
    assert first["__spread"] == pytest.approx(10.6066, abs=1e-3)
    assert (second["eval/reward"], second["n"], second["__spread"]) == (60.0, 1, 0.0)  # only version_1 here
    set_combine(panel, "sem")
    ((_, points, _),) = chart.series
    assert points[0]["__spread"] == pytest.approx(7.5)  # std / sqrt(2)
    set_combine(panel, None)
    assert chart.band is None and len(chart.series) == 2


def test_refresh_keeps_ticked_runs_plotted():
    # Regression: Refresh dropped the loaded data but kept the ticks, so the next redraw (e.g. choosing
    # Mean ± std) emptied every chart for good.
    panel, _ = make_panel()
    tick(panel, "ppo_run", "sb3_run")
    panel.refresh()
    for mode in ("std", None, "sem", None):
        set_combine(panel, mode)
        assert len(panel.grid.cards["eval/reward"].chart.series) == 2, mode


def set_primary(panel, name):
    panel.primary_select.setCurrentIndex(panel.primary_select.findData(name))


def test_primary_series_decides_which_steps_hovering_reads():
    panel, _ = make_panel()
    tick(panel, "ppo_run", "sb3_run")
    options = [panel.primary_select.itemText(i) for i in range(panel.primary_select.count())]
    assert options == ["Nearest", "ppo_run / ppo (version_1)", "sb3_run / ppo"]
    set_primary(panel, "sb3_run / ppo")
    chart = panel.grid.cards["eval/reward"].chart
    assert chart.primary == "sb3_run / ppo"
    assert [name for name, _, _ in chart._draw_order()][-1] == "sb3_run / ppo"  # drawn on top
    chart.point_hovered.emit(2000.0)
    # The primary series' runs come first in the inspector, and the caption names it.
    assert next(iter(panel.inspector.values)).experiment_id == "sb3_run"
    assert "a step of sb3_run / ppo" in panel.inspector.caption.text()
    panel.redraw()
    assert panel.primary_series() == "sb3_run / ppo"  # kept while still plotted
    tree_items(panel)["sb3_run"].setCheckState(0, Qt.CheckState.Unchecked)
    assert panel.primary_series() is None and chart.primary is None  # back to Nearest once unticked


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


def test_grouped_layout_has_one_chart_per_metric_and_ignores_old_layout_settings():
    for old_layout in ("overview", "quiet", "rail", "focus"):
        panel, _ = make_panel(FakeSettings({"results": {"layout": old_layout}}))
        tick(panel, "ppo_run", "sb3_run")
        assert panel.grid.hero is None and panel.grid.focus is None
        assert panel.grid.collapsible
        reward = panel.grid.cards["eval/reward"]
        assert grid_widgets(panel).count(reward) == 1
        assert series_names(panel, "eval/reward") == ["ppo_run / ppo (version_1)", "sb3_run / ppo"]
        panel.grid.toggle_group("losses")
        assert panel.grid.cards["losses/total_loss"] not in grid_widgets(panel)


def test_summary_tab_lists_every_series_and_metric():
    panel, _ = make_panel()
    tick(panel, "sb3_run")
    rows = {row[1]: row for row in panel.summary.rows}
    reward = rows["eval/reward"]
    # Series, Metric, Final, Best, Best at, Min, Max, Mean, Area under curve, Points
    assert reward == ["sb3_run / ppo", "eval/reward", 35.0, 35.0, 2000.0, 15.0, 35.0, 25.0, 25000.0, 2]
    loss = rows["losses/train/value_loss"]
    assert (loss[3], loss[4]) == (2.0, 2000.0)  # a loss: lower is better
    assert panel.summary.table.rowCount() == 2


def test_summary_threshold_reports_when_each_series_got_there():
    panel, _ = make_panel()
    tick(panel, "ppo_run", "sb3_run")
    summary = panel.summary
    assert not summary.threshold.isEnabled()  # thresholds need one metric
    summary.metric_select.setCurrentIndex(summary.metric_select.findData("eval/reward"))
    summary.threshold.setText("30")
    assert summary.headers[-1] == "Reaches threshold at"
    reached = {row[0]: row[-1] for row in summary.rows}
    assert reached == {"ppo_run / ppo (version_1)": 2000.0, "sb3_run / ppo": 2000.0}
    summary.threshold.setText("100")
    assert all(row[-1] is None for row in summary.rows)  # nobody got there


def test_exports_write_curves_summary_and_charts(tmp_path):
    panel, _ = make_panel()
    tick(panel, "sb3_run")
    panel.set_hidden("losses/train/value_loss", True)  # hidden charts are not exported
    assert panel.export_to("curves", str(tmp_path / "curves.csv"))
    assert (tmp_path / "curves.csv").read_text(encoding="utf-8").splitlines() == [
        "series,metric,transitions,value",
        "sb3_run / ppo,eval/reward,1000.0,15.0",
        "sb3_run / ppo,eval/reward,2000.0,35.0",
    ]
    assert panel.export_to("summary", str(tmp_path / "summary.csv"))
    summary = (tmp_path / "summary.csv").read_text(encoding="utf-8").splitlines()
    assert summary[0].startswith("Series,Metric,Final,Best") and len(summary) == 3
    panel.resize(900, 700)
    assert panel.export_to("png", str(tmp_path / "charts.png"))
    assert (tmp_path / "charts.png").read_bytes()[:4] == bytes([0x89]) + b"PNG"


def test_combined_curves_export_their_spread(tmp_path):
    panel, _ = make_panel()
    tick(panel, "ppo_run")
    tree_items(panel)["version_0"].setCheckState(0, Qt.CheckState.Checked)
    set_combine(panel, "sem")
    lines = panel.curves_csv().splitlines()
    assert lines[0] == "series,metric,transitions,value,sem,runs"
    assert lines[1] == "ppo_run / ppo (mean of 2),eval/reward,1000.0,12.5,7.5,2"


def test_window_loads_results_runs_into_the_builder_only_when_it_has_their_config():
    from frontend.app import Window

    with tempfile.TemporaryDirectory() as data_dir:
        window = Window(Path(data_dir))
        try:
            assert "No builder config" in window.load_results_config("cartpole/unknown_run")
            from frontend.model import Config, new_run

            run = new_run(Config(name="synthetic_run"), simulated=False)
            window.runs.append(run)
            window.refresh_runs()
            run.update(simulated=False, backend={"group": "thetaide", "experiment_id": "exp_1", "job_id": "j"})
            assert window.load_results_config("thetaide/exp_1") is None
            assert window.tabs.currentWidget() is window.config_panel
        finally:
            window.close()


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


def test_run_action_buttons_target_the_selected_run():
    from frontend.results.explorer import RunExplorer

    explorer = RunExplorer()
    explorer.set_runs([{"group": "group", "experiment_id": "run", "agents": [{"name": "agent"}]}])
    experiment = explorer.tree.topLevelItem(0).child(0)
    explorer.tree.setCurrentItem(experiment.child(0).child(0))
    calls = []
    explorer.archive_requested.connect(lambda group, run: calls.append(("archive", group, run)))
    explorer.delete_requested.connect(lambda group, run: calls.append(("delete", group, run)))
    explorer.btn_archive.click()
    explorer.btn_delete.click()
    assert calls == [("archive", "group", "run"), ("delete", "group", "run")]
    explorer.tree.setCurrentItem(None)
    explorer.btn_delete.click()
    assert len(calls) == 2
    explorer.deleteLater()


def test_run_selector_groups_timestamped_reruns_and_filters_them():
    from frontend.results.explorer import RunExplorer

    explorer = RunExplorer()
    explorer.set_runs([
        {"group": "group", "experiment_id": name, "agents": [{"name": "agent"}]}
        for name in ("cartpole", "cartpole_20261028-165620", "cartpole_20261028-165620-2", "other")
    ])
    group = explorer.tree.topLevelItem(0)
    experiment = group.child(0)
    assert experiment.text(0) == "cartpole"
    assert [experiment.child(i).text(0) for i in range(experiment.childCount())] == [
        "cartpole", "cartpole_20261028-165620", "cartpole_20261028-165620-2"
    ]
    assert not experiment.flags() & Qt.ItemFlag.ItemIsUserCheckable
    for i in range(experiment.childCount()):
        experiment.child(i).setCheckState(0, Qt.CheckState.Checked)
    assert {source.experiment_id for source in explorer.checked_sources()} == {
        "cartpole", "cartpole_20261028-165620", "cartpole_20261028-165620-2"
    }
    explorer.apply_filter("165620-2")
    assert not experiment.isHidden() and experiment.child(0).isHidden()
    assert not experiment.child(2).isHidden() and group.child(1).isHidden()
    explorer.deleteLater()


def test_run_selector_switches_are_right_aligned_and_clickable():
    from PyQt6.QtCore import QPoint
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QStyleOptionViewItem
    from frontend.results.explorer import RunExplorer

    explorer = RunExplorer()
    explorer.resize(400, 300)
    explorer.set_runs([{"group": "group", "experiment_id": "run", "agents": [{"name": "agent"}]}])
    explorer.show()
    group = explorer.tree.topLevelItem(0)
    experiment = group.child(0)
    experiment.setExpanded(True)
    run = experiment.child(0)
    option = QStyleOptionViewItem()
    explorer.tree.itemDelegate().initStyleOption(option, explorer.tree.indexFromItem(run))
    assert not option.features & QStyleOptionViewItem.ViewItemFeature.HasCheckIndicator
    run.setExpanded(True)
    QApplication.processEvents()
    rect = explorer.tree.visualItemRect(run)
    QTest.mouseClick(explorer.tree.viewport(), Qt.MouseButton.LeftButton,
                     pos=QPoint(rect.right() - 20, rect.center().y()))
    assert run.checkState(0) == Qt.CheckState.Checked
    assert len(explorer.checked_sources()) == 1
    QTest.mouseClick(explorer.tree.viewport(), Qt.MouseButton.LeftButton,
                     pos=QPoint(rect.right() - 20, rect.center().y()))
    assert not explorer.checked_sources()
    explorer.close()
    explorer.deleteLater()


def test_run_switch_selects_all_methods_but_only_latest_versions():
    from frontend.results.data import Manifest
    from frontend.results.explorer import RunExplorer

    explorer = RunExplorer()
    explorer.set_runs([{"group": "g", "experiment_id": "run", "agents": [{"name": "a"}, {"name": "b"}]}])
    explorer.add_manifest(Manifest("g", "run", {"a": ["version_0", "version_1"],
                                                 "b": ["version_0", "version_1"]}, []))
    run = explorer.tree.topLevelItem(0).child(0).child(0)
    run.setCheckState(0, Qt.CheckState.Checked)
    assert {(s.agent, s.version) for s in explorer.checked_sources()} == {("a", "version_1"), ("b", "version_1")}
    run.child(0).child(0).setCheckState(0, Qt.CheckState.Checked)
    run.setCheckState(0, Qt.CheckState.Unchecked)
    assert not explorer.checked_sources()
    assert all(run.child(i).child(j).checkState(0) == Qt.CheckState.Unchecked for i in range(2) for j in range(2))
    run.setCheckState(0, Qt.CheckState.Checked)
    assert {(s.agent, s.version) for s in explorer.checked_sources()} == {("a", "version_1"), ("b", "version_1")}
    run.child(0).setCheckState(0, Qt.CheckState.Unchecked)
    assert {(s.agent, s.version) for s in explorer.checked_sources()} == {("b", "version_1")}
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


def test_saved_sweeps_distinguish_runs_with_same_method_and_study_and_handle_bad_snapshots(monkeypatch):
    import json

    for key in ("cartpole/ppo_run", "sb3/sb3_run"):
        files = dict(FILES[key])
        data = {"study": "search", "direction": "maximize", "trials": [{"id": 0, "state": "complete", "value": 1}]}
        files[f"checkpoints/{key}/ppo/tuning_summary.json"] = ("metadata", json.dumps(data).encode())
        files[f"checkpoints/{key}/broken/tuning_summary.json"] = ("metadata", b"{broken")
        monkeypatch.setitem(FILES, key, files)
    panel, _ = make_panel()
    tick(panel, "ppo_run", "sb3_run")
    panel.tabs.setCurrentWidget(panel.sweeps_page)
    view = panel.sweeps_view
    assert {method for method, study in view.studies} == {"cartpole/ppo_run / ppo", "sb3/sb3_run / ppo"}
    view.table.selectRow(0)
    assert not view.open_button.isEnabled()  # legacy summaries never open another trial's newest version
    panel.close()


def test_saved_sweep_results_refresh_reloads_snapshot(monkeypatch):
    import json

    key = "cartpole/ppo_run"
    path = f"checkpoints/{key}/ppo/tuning_summary.json"
    files = dict(FILES[key])
    study = {"study": "search", "direction": "maximize", "best_trial_id": 0,
             "trials": [{"id": 0, "state": "complete", "value": 10}]}
    files[path] = ("metadata", json.dumps(study).encode())
    monkeypatch.setitem(FILES, key, files)
    panel, _ = make_panel()
    tick(panel, "ppo_run")
    panel.tabs.setCurrentWidget(panel.sweeps_page)
    assert "Best 10" in panel.sweeps_view.summary.text()
    study["trials"][0]["value"] = 20
    files[path] = ("metadata", json.dumps(study).encode())
    panel.refresh()
    assert "Best 20" in panel.sweeps_view.summary.text()
    assert panel.sweeps_view.selector.count() == 1
    panel.close()


def test_saved_importance_chart_stays_static_until_results_are_refreshed(monkeypatch):
    import json
    from PyQt6.QtCore import QTimer

    key = "cartpole/ppo_run"
    path = f"checkpoints/{key}/ppo/tuning_summary.json"
    files = dict(FILES[key])
    study = {"study": "search", "direction": "maximize", "trials": [],
             "parameter_importance": {"learning_rate": 0.75, "batch_size": 0.25},
             "parameter_importance_info": {"completed_trials": 10, "evaluator": "fANOVA", "status": "ready"}}
    files[path] = ("metadata", json.dumps(study).encode())
    monkeypatch.setitem(FILES, key, files)
    panel, backend = make_panel()
    tick(panel, "ppo_run")
    panel.tabs.setCurrentWidget(panel.sweeps_page)
    chart = panel.sweeps_view.importance_view
    assert chart.scores == [("learning_rate", 0.75), ("batch_size", 0.25)]
    assert not panel.findChildren(QTimer)
    before = list(backend.requests)
    study["parameter_importance"] = {"learning_rate": 0.2, "batch_size": 0.8}
    files[path] = ("metadata", json.dumps(study).encode())
    app.processEvents()
    assert backend.requests == before
    assert chart.scores[0] == ("learning_rate", 0.75)
    panel.refresh()
    assert chart.scores == [("batch_size", 0.8), ("learning_rate", 0.2)]
    panel.close()


def test_saved_sweeps_link_parallel_filters_to_history_without_live_polling(monkeypatch):
    import json
    from PyQt6.QtCore import QTimer

    key = "cartpole/ppo_run"
    files = dict(FILES[key])
    study = {"study": "search", "direction": "maximize", "trials": [
        {"id": 0, "state": "complete", "value": 10, "parameters": {"lr": .01, "optimizer": "adam"}},
        {"id": 1, "state": "complete", "value": 50, "parameters": {"lr": .02, "optimizer": "sgd"}},
        {"id": 2, "state": "complete", "value": 90, "parameters": {"lr": .03, "optimizer": "adam"}},
    ]}
    files[f"checkpoints/{key}/ppo/tuning_summary.json"] = ("metadata", json.dumps(study).encode())
    monkeypatch.setitem(FILES, key, files)
    panel, backend = make_panel()
    tick(panel, "ppo_run")
    panel.tabs.setCurrentWidget(panel.sweeps_page)
    view = panel.sweeps_view
    before = list(backend.requests)
    view.parallel_plot.brushes[("parameter", "lr")] = (0, .025)
    view.parallel_plot.selection_changed.emit()
    view.filter.setText("adam")
    assert view.parallel_plot.selected_ids == {0}
    assert view.plot.series[0][1] == [(0, 10)]
    assert view.selection_count.text() == "1 / 3 trials"
    assert not panel.findChildren(QTimer)
    assert backend.requests == before
    view.clear_filters.click()
    assert view.parallel_plot.selected_ids == {0, 1, 2}
    assert view.plot.series[0][1] == [(0, 10), (1, 50), (2, 90)]
    panel.close()


def test_trial_toggle_loads_each_saved_seed_curve_and_preserves_selection(tmp_path, monkeypatch):
    import json
    from src.app.services import catalog

    logs = tmp_path / "results/logs"
    monkeypatch.setattr(catalog, "LOGS_DIR", logs)
    trial_version = "version_sweep_search_trial_35"
    run_dir = logs / "group/run"

    def write(path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(data)

    write(run_dir / f"ppo/{trial_version}/metrics.csv", "step,objective/learning_curve\n0,0.4472875\n")
    trials = {"study": "search", "stop_reason": "trial_limit", "best_trial_id": 35,
              "trials": [{"id": 35, "state": "complete", "value": .4472875, "log_version": trial_version}]}
    write(logs.parent / "checkpoints/group/run/ppo/tuning_summary.json", json.dumps(trials))
    runs = []
    for index in range(7):
        version = f"{trial_version}_sample_seed_{index}"
        runs.append({"experiment_id": "run", "method": "ppo", "agent": "ppo", "version": version,
                     "training_seed": 1234679 + index, "status": "completed" if index < 6 else "running"})
        write(run_dir / f"ppo/{version}/metrics.csv",
              f"transitions,eval/reward,losses/policy\n0,10,1\n16384,{100 + index},0.5\n32768,{300 + index},0.1\n")
        # An end timestamp must not publish a seed still marked running in its execution record.
        write(run_dir / f"ppo/{version}/runtime.json", json.dumps({"completed_at": "2026-11-04"}))
    write(logs.parent / "executions/group/run/execution/manifest.json", json.dumps({
        "parent": {"kind": "trial", "log_version": trial_version, "study": "search", "trial_number": 35},
        "runs": runs}))

    class Backend:
        def list_runs(self, callback, **kwargs):
            callback({"runs": catalog.list_runs(**kwargs)}, None)

        def run_manifest(self, group, experiment, callback, **kwargs):
            callback(catalog.run_manifest(group, experiment, **kwargs), None)

        def run_metrics(self, group, experiment, agent, callback, **kwargs):
            callback(catalog.run_metrics(group, experiment, agent, **kwargs), None)

        def run_file(self, group, experiment, path, callback):
            callback(catalog.read_run_file(group, experiment, path), None)

    panel = ResultsPanel(Backend())
    panel.refresh()
    panel.explorer.select_run("group", "run")
    assert len(panel.sources) == 6
    assert set(panel.grid.cards) == {"eval/reward", "losses/policy"}
    assert len(panel.grid.cards["eval/reward"].chart.series) == 6
    assert all(len(points) == 3 for _, points, _ in panel.grid.cards["eval/reward"].chart.series)
    items = tree_items(panel)
    trial = items["search · trial 35"]
    assert trial.childCount() == 6
    assert "Seed 1234685" not in items
    trial.setCheckState(0, Qt.CheckState.Unchecked)
    assert panel.sources == []
    trial.setCheckState(0, Qt.CheckState.Checked)
    assert len(panel.sources) == 6
    items["Seed 1234679"].setCheckState(0, Qt.CheckState.Unchecked)
    assert len(panel.sources) == 5
    assert trial.checkState(0) == Qt.CheckState.PartiallyChecked
    panel.refresh()
    assert len(panel.sources) == 5
    panel.tabs.setCurrentWidget(panel.sweeps_page)
    panel.sweeps_view.table.selectRow(0)
    panel.sweeps_view.open_button.click()
    assert panel.tabs.currentWidget() is panel.curves_page
    assert len(panel.sources) == 6
    assert all("trial 35 · seed" in source.label for source in panel.sources)
    panel.combine_select.setCurrentIndex(1)
    series = panel.grid.cards["eval/reward"].chart.series
    assert len(series) == 1
    assert all(point["n"] == 6 for point in series[0][1])
    assert "trial 35" in series[0][0]
    panel.close()

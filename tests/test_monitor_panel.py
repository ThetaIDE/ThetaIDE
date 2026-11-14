"""Training monitor panel (frontend/monitor): extracted from the main window, metric-agnostic."""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("PyQt6")

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PyQt6.QtCore import QObject, Qt, pyqtSignal  # noqa: E402
from PyQt6.QtTest import QTest  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True)
app = QApplication.instance() or QApplication(sys.argv[:1])

from frontend.app import Window  # noqa: E402
from frontend.model import disk_run, metric_points  # noqa: E402
from frontend.monitor import MonitorPanel, second_metric_choices, second_metric_text  # noqa: E402

# What the SB3 runner logs through ContractLogger: SB3's loss names, prefixed with losses/.
SB3_ROWS = [{"step": float(i), "transitions": float(i * 1000), "eval/reward": 10.0 + 18 * i,
             "losses/train/value_loss": 40.0 / i, "losses/train/policy_gradient_loss": -0.01 * i}
            for i in range(1, 5)]


def sb3_run():
    run = disk_run({
        "group": "sb3", "experiment_id": "cartpole_sb3_ppo", "metadata": {},
        "config": {"env": "cartpole", "seed": 1, "total_timesteps": 4000, "methods": {"ppo": {"agent": "sb3/ppo"}}},
        "agents": [{"name": "ppo", "started": "2026-10-08T10:00:00+00:00", "finished": "2026-10-08T10:02:00+00:00",
                    "training_time_seconds": 120.0, "timesteps": 4000, "best_reward": 80.0, "latest_reward": 80.0}],
        "finished": True,
    })
    run["metrics"] = metric_points(SB3_ROWS)
    return run


def test_second_metric_choices_offer_every_logged_metric():
    run = {"metrics": metric_points(SB3_ROWS)}
    # Reward has its own chart; every other metric is offered, not only the PPO ones.
    assert second_metric_choices(run) == ["losses/train/policy_gradient_loss", "losses/train/value_loss"]
    ppo = {"metrics": [{"step": 1, "reward": 1.0, "loss": 0.5, "entropy": 0.6, "time/train": 2.0}]}
    assert second_metric_choices(ppo) == ["loss", "entropy", "time/train"]  # hand-described first
    assert second_metric_choices({"metrics": [{"step": 1, "reward": 1.0}]}) == ["loss"]  # fallback


def test_second_metric_text_uses_the_catalog_for_unknown_keys():
    assert second_metric_text("loss")[1] == "Total loss"
    card, chart, _, fmt = second_metric_text("losses/train/value_loss")
    assert (card, chart, fmt) == ("VALUE LOSS", "Value loss", ".4g")
    assert second_metric_text("custom/success_rate")[1] == "Success rate"


def test_config_preview_is_nonmodal_and_follows_selected_run(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "config.yaml").write_text("method: ppo\n")
    (second / "config.yaml").write_text("method: dqn\n")
    host = SimpleNamespace(selected={"id": "first", "status": "running"}, run_folder=lambda run: tmp_path / run["id"])
    monitor = MonitorPanel(host)
    monitor.resize(900, 700)
    monitor.show()
    assert monitor.status_strip.badge.isHidden()
    assert monitor.run_caption.isHidden()
    QTest.mouseClick(monitor.yaml_button, Qt.MouseButton.LeftButton)
    assert monitor.yaml_overlay.isVisible()
    assert monitor.yaml_viewer.toPlainText().endswith("method: ppo\n")
    assert "Saved experiment configuration" in monitor.yaml_viewer.toPlainText()
    host.selected = {"id": "second", "status": "running"}
    monitor.show_selected(host.selected)
    assert monitor.yaml_viewer.toPlainText().endswith("method: dqn\n")
    QTest.mouseClick(monitor, Qt.MouseButton.LeftButton)
    assert not monitor.yaml_overlay.isVisible()
    assert not monitor.yaml_button.isChecked()
    monitor.close()


def test_config_preview_follows_streaming_run_trial_and_seed(tmp_path):
    import yaml
    from frontend.live_run import LiveRun

    (tmp_path / "config.yaml").write_text("model: wrong-experiment-file\n")
    active = {"id": "active", "status": "running"}
    first, second = LiveRun(), LiveRun()
    for live, seed in ((first, 11), (second, 22)):
        live.add([{"type": "start", "run": {"pid": seed, "host": "worker", "version": f"seed_{seed}"},
                   "config": {"seed": seed, "agent": {"policy": "MlpPolicy"},
                              "model": {"name": "blendrl-dev"}}}])
    first_name = "BlendRL · study · trial 20 · seed 0"
    second_name = "BlendRL · study · trial 20 · seed 1"
    lives = {first_name: first, second_name: second}
    host = SimpleNamespace(selected={"id": "old", "status": "completed"}, runs=[active],
                           run_folder=lambda run: tmp_path,
                           live_for_run=lambda run, load=True: lives if run is active else {})
    monitor = MonitorPanel(host)
    monitor.show()
    monitor.show_run_yaml()
    assert yaml.safe_load(monitor.yaml_viewer.toPlainText()) == first.config
    assert first_name in monitor.yaml_viewer.toPlainText()
    monitor.focus_agent(second_name)
    text = monitor.yaml_viewer.toPlainText()
    assert yaml.safe_load(text) == second.config
    assert second_name in text and "Pid: 22" in text
    second.add([{"type": "event", "kind": "model", "data": {
        "policy_class": "theta.ThetaActorCriticPolicy", "model_class": "blendrl.BlenderActorCritic",
        "model_components": ["symbolic.NSFReasoner", "neural.DNNModel"]}}])
    monitor.refresh_live()
    text = monitor.yaml_viewer.toPlainText()
    assert "Runtime policy_class: theta.ThetaActorCriticPolicy" in text
    assert "Runtime model_class: blendrl.BlenderActorCritic" in text
    assert "symbolic.NSFReasoner" in text
    assert yaml.safe_load(text) == second.config
    monitor.close()


def test_config_preview_does_not_replace_missing_process_config_with_experiment(tmp_path):
    from frontend.live_run import LiveRun

    (tmp_path / "config.yaml").write_text("model: misleading-parent-config\n")
    live = LiveRun()
    live.add([{"type": "metrics", "step": 1, "data": {"loss": 1}}])
    host = SimpleNamespace(selected={"id": "active", "status": "running"},
                           run_folder=lambda run: tmp_path, live_for_run=lambda run, load=True: {"worker": live})
    monitor = MonitorPanel(host)
    monitor.show_run_yaml()
    text = monitor.yaml_viewer.toPlainText()
    assert "not been reported by this process" in text
    assert "misleading-parent-config" not in text
    monitor.close()


def test_config_preview_uses_process_from_focused_sweep(tuner_monitor_workspace):
    import yaml
    from frontend.live_run import LiveRun

    first, second = LiveRun(), LiveRun()
    first.add([{"type": "start", "config": {"model": "first"}}])
    second.add([{"type": "start", "config": {"model": "second"}}])
    lives = {"A · study-a · trial 1": first, "B · study-b · trial 2": second}
    paths, config = tuner_monitor_workspace
    host = SimpleNamespace(workspace_paths=paths, selected={"id": "active", "status": "running", "config": config},
                           live_for_run=lambda run, load=True: lives)
    monitor = MonitorPanel(host)
    monitor.refresh_live()
    monitor.focused_agent = next(iter(lives))
    monitor._focus_run_id = "active"
    monitor._focused_sweep = "B:study-b"
    monitor.views.setCurrentWidget(monitor.sweep_view)
    monitor.show_run_yaml()
    assert yaml.safe_load(monitor.yaml_viewer.toPlainText()) == second.config
    monitor.close()


def test_queue_preview_uses_process_cards_for_each_method():
    class Jobs(QObject):
        queue_changed = pyqtSignal(dict)

        def queue_state(self):
            return {"active": [{"job_id": "one", "experiment_id": "experiment", "status": "running",
                                "agents": ["ppo", "dqn"], "run_in_parallel": False}],
                    "queued": [{"job_id": "two", "status": "queued", "agents": ["sac"]}]}

    opened = []
    jobs = Jobs()
    live = SimpleNamespace(start=None, events=[], rows=[{}], finish=None, budget=100,
                           progress=lambda: 40, rate=lambda: 2, artifacts=[])
    host = SimpleNamespace(jobs=jobs, selected=None, run_for_job=lambda job_id: {"description": "My run"},
                           live_for_run=lambda run, load=False: {"ppo": live},
                           open_job=lambda job_id, method: opened.append((job_id, method)))
    monitor = MonitorPanel(host)
    monitor.resize(1000, 700)
    monitor.show()
    QTest.mouseClick(monitor.queue_button, Qt.MouseButton.LeftButton)
    assert list(monitor.queue_preview.cards) == ["one:ppo", "one:dqn", "two:experiment", "two:sac"]
    assert monitor.queue_preview.cards["one:ppo"].description.text() == "My run"
    assert monitor.queue_preview.cards["two:sac"].elapsed.text() == "Waiting"
    assert monitor.queue_preview.cards["one:ppo"].progress_fraction == 0.4
    assert monitor.queue_preview.cards["one:ppo"].speed.text() == "Speed: 2 steps/s"
    assert monitor.queue_preview.cards["one:ppo"].eta.text() == "Time left: 30s"
    assert monitor.queue_preview.cards["one:dqn"].speed.text() == "Speed: —"
    assert monitor.queue_overlay.width() == monitor.content.width() * 2 // 3
    assert monitor.queue_preview.cards["one:dqn"].progress_fraction is None
    assert monitor.queue_overlay.height() < monitor.content.height()
    live.progress = lambda: 75
    monitor.queue_preview._update_times()
    assert monitor.queue_preview.cards["one:ppo"].progress_fraction == 0.75
    jobs.queue_changed.emit({})
    assert list(monitor.queue_preview.cards) == ["one:ppo", "one:dqn", "two:experiment", "two:sac"]
    QTest.mouseClick(monitor.queue_preview.cards["one:dqn"], Qt.MouseButton.LeftButton)
    assert opened == [("one", "dqn")]
    assert not monitor.queue_overlay.isVisible()
    QTest.mouseClick(monitor.queue_button, Qt.MouseButton.LeftButton)
    QTest.mouseClick(monitor, Qt.MouseButton.LeftButton)
    assert not monitor.queue_overlay.isVisible()
    monitor.close()


def test_queue_and_config_buttons_dismiss_open_menu_without_reopening():
    class Jobs(QObject):
        queue_changed = pyqtSignal(dict)

        def queue_state(self):
            return {"active": [], "queued": []}

    host = SimpleNamespace(jobs=Jobs(), selected=None, run_for_job=lambda _: None)
    monitor = MonitorPanel(host)
    monitor.resize(1000, 700)
    monitor.show()
    for opener, closer, overlay in ((monitor.queue_button, monitor.queue_button, monitor.queue_overlay),
                                    (monitor.queue_button, monitor.yaml_button, monitor.queue_overlay),
                                    (monitor.yaml_button, monitor.yaml_button, monitor.yaml_overlay),
                                    (monitor.yaml_button, monitor.queue_button, monitor.yaml_overlay)):
        QTest.mouseClick(opener, Qt.MouseButton.LeftButton)
        assert overlay.isVisible()
        QTest.mouseClick(closer, Qt.MouseButton.LeftButton)
        assert not monitor.queue_overlay.isVisible()
        assert not monitor.yaml_overlay.isVisible()
        assert not monitor.queue_button.isChecked()
        assert not monitor.yaml_button.isChecked()
    monitor.close()


def test_queue_card_receives_click_dispatched_through_window():
    class Jobs(QObject):
        queue_changed = pyqtSignal(dict)

        def queue_state(self):
            return {"active": [{"job_id": "job", "status": "running", "agents": ["ppo"]}]}

    opened = []
    host = SimpleNamespace(jobs=Jobs(), selected=None, run_for_job=lambda _: None,
                           open_job=lambda *args: opened.append(args))
    monitor = MonitorPanel(host)
    monitor.resize(1000, 700)
    monitor.show()
    app.processEvents()
    button_position = monitor.queue_button.mapTo(monitor, monitor.queue_button.rect().center())
    QTest.mouseClick(monitor.windowHandle(), Qt.MouseButton.LeftButton, pos=button_position)
    app.processEvents()
    assert monitor.queue_overlay.isVisible()
    card = monitor.queue_preview.cards["job:ppo"]
    position = card.mapTo(monitor, card.rect().center())
    QTest.mouseClick(monitor.windowHandle(), Qt.MouseButton.LeftButton, pos=position)
    assert opened == [("job", "ppo")]
    assert not monitor.queue_overlay.isVisible()
    QTest.mouseClick(monitor.windowHandle(), Qt.MouseButton.LeftButton, pos=button_position)
    assert monitor.queue_overlay.isVisible()
    QTest.mouseClick(monitor.windowHandle(), Qt.MouseButton.LeftButton, pos=monitor.rect().topLeft())
    assert not monitor.queue_overlay.isVisible()
    assert not monitor.queue_button.isChecked()
    monitor.close()


class TestMonitorPanel(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.window = Window(Path(self.temp_dir.name))

    def tearDown(self):
        self.window.close()
        self.temp_dir.cleanup()

    def test_window_exposes_the_panel_widgets_under_their_old_names(self):
        monitor = self.window.monitor
        self.assertIsInstance(monitor, MonitorPanel)
        self.assertIs(self.window.monitor_panel, monitor)
        for name in MonitorPanel.EXPOSED:
            self.assertIs(getattr(self.window, name), getattr(monitor, name), name)

    def test_runner_metrics_use_generic_catalog_labels(self):
        run = sb3_run()
        self.window.runs.insert(0, run)
        self.window.refresh_runs()
        self.window.select_run(run)
        monitor = self.window.monitor
        self.assertIsNone(monitor.charts_view.grid.header)
        self.assertNotIn("eval", monitor.charts_view.grid.collapsed)


def test_last_finished_run_opens_results_only_after_all_jobs_finish():
    from unittest.mock import Mock

    run = {"backend": {"group": "group", "experiment_id": "last"}}
    host = SimpleNamespace(active_runs=[{}], jobs=SimpleNamespace(is_busy=Mock(return_value=False)),
                           _quitting=False, _set_monitor_active=Mock(), results_view=SimpleNamespace(open_run=Mock()),
                           tabs=SimpleNamespace(setCurrentWidget=Mock()), results_panel=object())
    Window._open_last_run_results(host, run)
    host.results_view.open_run.assert_not_called()
    host.active_runs.clear()
    host.jobs.is_busy.return_value = True
    Window._open_last_run_results(host, run)
    host.results_view.open_run.assert_not_called()
    host.jobs.is_busy.return_value = False
    Window._open_last_run_results(host, run)
    host._set_monitor_active.assert_called_once_with(False)
    host.results_view.open_run.assert_called_once_with("group", "last")
    host.tabs.setCurrentWidget.assert_called_once_with(host.results_panel)


def test_monitor_header_and_legend_follow_run_and_method_colors(monkeypatch):
    from frontend.live_run import LiveRun
    from frontend.monitor.live_view import PALETTE
    from frontend.theme import BUILTINS, theme_color

    monkeypatch.setattr(app, "theta_theme", BUILTINS["Nord"], raising=False)
    run = sb3_run()
    run["status"] = "running"
    run["backend"]["agents"] = ["ppo", "dqn"]
    dqn = LiveRun()
    dqn.rows = [{"step": 1, "eval/reward": 42.0}]
    lives = {"dqn": dqn}
    host = SimpleNamespace(selected=run, queue_running=False,
                           live_for_run=lambda run, load=True: lives)
    monitor = MonitorPanel(host)
    monitor.render()
    monitor.refresh_live()
    assert monitor.run_title.text() == "sb3 / cartpole_sb3_ppo"
    assert "ppo" not in monitor.method_legend.text() and "dqn" in monitor.method_legend.text()
    assert theme_color(PALETTE[0]) in monitor.method_legend.text()
    assert monitor.charts_view._series_colors["dqn"] == PALETTE[0]
    reward = monitor.charts_view.grid.cards["eval/reward"].chart
    assert reward.series[0][0] == "dqn"
    assert theme_color(reward.series[0][2]) == theme_color(PALETTE[0])
    assert not monitor.charts_view.grid.lead_hidden
    other = LiveRun()
    other.rows = [{"step": 1, "eval/reward": 10.0}]
    lives["ppo · study · trial 1"] = dqn
    lives["ppo · study · trial 2"] = other
    monitor.focus_agent("ppo · study · trial 2")
    assert "Trial 2" in monitor.method_legend.text()
    assert "Trial 1" not in monitor.method_legend.text()
    assert "dqn" not in monitor.method_legend.text()
    assert list(monitor.shown_lives(lives)) == ["ppo · study · trial 2"]
    monitor.focus_agent("dqn")
    assert monitor.charts_view._series_colors["dqn"] == PALETTE[0]
    monitor.close()


def test_method_and_trial_dropdowns_route_multiple_sweeps(tmp_path, monkeypatch, tuner_monitor_workspace):
    from unittest.mock import Mock
    from frontend.live_run import LiveRun
    from src.app.pipeline import tuning_monitor as sweep_view

    study = {"study": "search", "trials": [{"id": 1, "state": "running"},
                                             {"id": 2, "state": "complete"}]}
    other = {"study": "other", "trials": [{"id": 0, "state": "running", "execution_started": 1}]}
    monkeypatch.setattr(sweep_view, "study_summaries", lambda *_, **kw: [("ppo", study), ("dqn", other)])
    active = "ppo · search · trial 1"
    completed = "ppo · search · trial 2"
    lives = {name: LiveRun() for name in (active, completed)}
    lives[active].start = {"type": "start"}
    host = SimpleNamespace(selected=sb3_run(), live_for_run=lambda *a, **kw: lives,
                           jobs=SimpleNamespace(project_root=tmp_path, queue_changed=Mock()))
    host.selected["status"] = "running"
    host.workspace_paths, host.selected['config'] = tuner_monitor_workspace
    monitor = MonitorPanel(host)
    monitor.refresh_live()
    assert "Sweep" not in monitor.view_buttons
    method_menu = monitor.method_button.menu()
    method_menu.popup(monitor.mapToGlobal(monitor.rect().center()))
    app.processEvents()
    actions = {action.text(): action for action in method_menu.actions()}
    monitor.refresh_live()
    assert method_menu.isVisible()
    assert method_menu.actions() == list(actions.values())
    method_menu.hide()
    assert set(actions) == {"ppo", "dqn"}
    assert monitor.method_button.text() == " / ppo"
    assert monitor.trial_button.text() == " / Trial 1"
    assert [a.text() for a in monitor.trial_button.menu().actions()] == ["Sweep", "Trial 1"]
    assert "ppo / Trial 1" in monitor.method_legend.text()
    assert "Trial 2" not in monitor.method_legend.text()
    actions["dqn"].trigger()
    assert monitor.views.currentWidget() is monitor.sweep_view
    assert monitor.sweep_view.selector.currentData() == ("dqn", "other")
    actions = {action.text(): action for action in monitor.method_button.menu().actions()}
    actions["ppo"].trigger()
    assert monitor.method_button.text() == " / ppo"
    assert monitor.trial_button.text() == " / Sweep"
    trials = {a.text(): a for a in monitor.trial_button.menu().actions()}
    assert "Trial 2" not in trials
    trials["Trial 1"].trigger()
    assert monitor.views.currentWidget() is monitor.charts_view
    assert monitor.focused_agent == active
    study["trials"][0]["state"] = "pruned"
    monitor.refresh_live()
    assert "Trial 1" not in [action.text() for action in monitor.trial_button.menu().actions()]
    assert monitor.method_legend.text() == ""
    assert monitor.method_legend.isHidden()
    study["trials"][0]["state"] = "running"
    lives[active].finish = {"status": "failed"}
    monitor.refresh_live()
    assert "Trial 1" not in [action.text() for action in monitor.trial_button.menu().actions()]
    assert "ppo / Trial 1" not in monitor.method_legend.text()
    assert active in lives  # history remains available from the sweep dashboard
    lives[active].finish = None
    monitor.refresh_live()
    assert "ppo / Trial 1" in monitor.method_legend.text()
    host.selected["status"] = "cancelled"
    monitor.refresh_live()
    assert monitor.method_legend.text() == ""  # parent can stop before trial telemetry finishes
    assert monitor.method_legend.isHidden()
    monitor.close()


def test_sweep_queue_cards_keep_their_study_identity():
    from frontend.queue_panel import QueuePanel

    opened = []
    panel = QueuePanel(lambda *args: opened.append(args))
    panel.set_data({"active": [{"job_id": "job", "sweep_parent": True, "sweep_key": "ppo:search"},
                                {"job_id": "job", "sweep_parent": True, "sweep_key": "dqn:other"}]})
    QTest.mouseClick(panel.cards["job:dqn:other"], Qt.MouseButton.LeftButton)
    assert opened == [("job", "sweep:dqn:other")]
    panel.close()


@pytest.mark.parametrize("method, expected", [("ppo", "Overview"),
                                               ("ppo · search · trial 1", "Overview"),
                                               ("sweep:ppo:search", "Sweep"), ("", "Overview")])
def test_open_queue_job_switches_to_its_live_monitor(method, expected):
    from unittest.mock import Mock

    run = {"id": "run"}
    monitor = SimpleNamespace(focus_agent=Mock(), focus_sweep=Mock(), show_view=Mock())
    host = SimpleNamespace(run_for_job=lambda _: run, select_run=Mock(), monitor=monitor,
                           _set_monitor_active=Mock(), tabs=SimpleNamespace(setCurrentWidget=Mock()),
                           monitor_panel=object())
    Window.open_job(host, "job", method)
    host.select_run.assert_called_once_with(run)
    if expected == "Sweep":
        monitor.focus_sweep.assert_called_once_with("ppo:search")
    else:
        monitor.focus_agent.assert_called_once_with(method)
        monitor.show_view.assert_called_once_with("Overview")
    host.tabs.setCurrentWidget.assert_called_once_with(host.monitor_panel)


def test_monitor_title_selects_groups_and_running_or_queued_experiments():
    from copy import deepcopy

    first = sb3_run()
    first.update(id="first", status="running")
    second, third = deepcopy(first), deepcopy(first)
    second.update(id="second", status="queued")
    second["backend"]["experiment_id"] = "second"
    third["id"] = "third"
    third["backend"].update(group="other", experiment_id="third")
    host = SimpleNamespace(selected=first, runs=[first, second, third], live_for_run=lambda *a, **kw: {})
    monitor = MonitorPanel(host)

    def select(run):
        host.selected = run
        monitor.show_selected(run)

    host.select_run = select
    monitor.show_selected(first)
    assert monitor.group_button.text() == "sb3"
    menu = monitor.experiment_button.menu()
    assert len(menu.actions()) == 2
    menu.popup(monitor.mapToGlobal(monitor.rect().center()))
    app.processEvents()
    actions = menu.actions()
    monitor.set_runs(host.runs)
    assert menu.isVisible()
    assert menu.actions() == actions
    menu.hide()
    actions[1].trigger()
    assert host.selected is second
    assert monitor.experiment_button.text() == " / second"
    monitor.group_button.menu().actions()[1].trigger()
    assert host.selected is third
    assert monitor.group_button.text() == "other"
    assert len(monitor.experiment_button.menu().actions()) == 1
    second["status"] = "completed"
    monitor.set_runs(host.runs)
    monitor.group_button.menu().actions()[0].trigger()
    assert host.selected is first
    assert len(monitor.experiment_button.menu().actions()) == 1
    monitor.close()


def test_play_queues_before_starting_and_opens_the_queued_monitor():
    from unittest.mock import Mock

    calls = []
    job = {"job_id": "new"}
    host = SimpleNamespace(
        check_missing_components=lambda: [], compose_valid=True,
        config=lambda: SimpleNamespace(name="new", overrides=lambda: []), current_experiment="g/new",
        jobs=SimpleNamespace(launch=Mock(side_effect=lambda *a, **kw: calls.append(("launch", kw)) or job)),
        log=lambda *a: None, queued=lambda *a: calls.append(("record",)) or {"id": "new"},
        open_job=lambda *a: calls.append(("monitor",)),
        set_queue_running=lambda value: calls.append(("start", value)),
        active_runs=[{"id": "already_running"}], launch_capacity=lambda: 1)
    host.add_to_queue = lambda **kw: Window.add_to_queue(host, **kw)
    Window.launch_training(host)
    assert calls == [("launch", {"queue": True, "overwrite": True}), ("record",), ("monitor",), ("start", True)]


def test_agents_queue_labelled_variants_with_extra_overrides(monkeypatch):
    """A variant gets its own results folder and records its overrides; a refusal is raised when strict."""
    from unittest.mock import Mock

    from src.app.services.config_service import LaunchError

    launches, persisted = [], []
    def host(valid=True):
        return SimpleNamespace(
            check_missing_components=lambda: [], compose_valid=valid, compose_error=None if valid else "[compose] bad key",
            config=lambda: SimpleNamespace(name="default_exp", overrides=lambda: []), current_experiment="g/cartpole",
            jobs=SimpleNamespace(launch=Mock(side_effect=lambda *a, **kw: launches.append((a, kw)) or {"job_id": "j"})),
            log=lambda *a: None, queued=lambda *a: {"id": "r1"}, persist=lambda run: persisted.append(dict(run)) or True,
            open_job=lambda *a: None, set_queue_running=lambda value: None,
            unique_experiment_id=lambda name, is_recover=False: name)

    run = Window.queue_current(host(), start=True, extra_overrides=["++model.use_edit=true"], label="edit on", strict=True)
    (experiment, overrides), options = launches[0]
    assert experiment == "g/cartpole"
    assert overrides == ["++model.use_edit=true", "++experiment_id='cartpole-edit-on'"]
    assert options == {"queue": True, "overwrite": False}  # never replaces earlier results
    assert run["overrides"] == ["++model.use_edit=true"] and run["label"] == "edit on"
    assert persisted[-1]["label"] == "edit on"

    with pytest.raises(LaunchError, match="bad key"):
        Window.queue_current(host(valid=False), extra_overrides=["x=1"], label="off", strict=True)
    assert len(launches) == 1


def test_play_stays_enabled_when_training_capacity_is_full(monkeypatch):
    from unittest.mock import Mock
    from PyQt6.QtWidgets import QPushButton

    monkeypatch.setattr(Window, "check_missing_components", lambda _: [])
    run = {"status": "running", "backend": {"experiment_id": "first"}}
    button = QPushButton()
    host = SimpleNamespace(active_runs=[run], active=run, compose_valid=True,
                           current_experiment="g/second", start_button=button, state_label=None)
    Window.update_launch_state(host)
    assert button.isEnabled()
    assert "queue" in button.toolTip()
    host.compose_valid = False
    Window.update_launch_state(host)
    assert not button.isEnabled()


@pytest.mark.parametrize("capacity", [1, 2])
def test_queued_experiments_wait_for_slots_and_start_in_order(tmp_path, monkeypatch, capacity):
    from frontend.runner_manager import JobManager

    manager = JobManager(store_path=tmp_path / "jobs.db", project_root=tmp_path,
                         logs_dir=tmp_path / "logs")
    started = []

    def start(job):
        job["status"] = "running"
        started.append(job["experiment_id"])

    monkeypatch.setattr(manager, "_start", start)
    monkeypatch.setattr(manager, "_scan_metrics", lambda *_: None)
    monkeypatch.setattr(manager, "_unwatch", lambda *_: None)
    manager.max_parallel = capacity
    jobs = []
    for name in ("first", "second", "third"):
        plan = {"experiment": f"g/{name}", "group": "g", "experiment_id": name,
                "agents": [], "total_timesteps": 10, "effective_timesteps": 10, "run_in_parallel": False}
        jobs.append(manager.launch(f"g/{name}", queue=True, _plan=plan))
        manager.start_queue()
    assert started == ["first", "second", "third"][:capacity]
    assert len(manager.queue_order) == 3 - capacity
    manager._finish(jobs[0]["job_id"], 0)
    assert started == ["first", "second", "third"][:capacity + 1]


@pytest.mark.parametrize("job_bucket", ["active", "queued"])
def test_queue_updates_restore_hidden_monitor_even_when_job_state_is_unchanged(job_bucket):
    from unittest.mock import Mock
    from PyQt6.QtWidgets import QCheckBox, QWidget
    from frontend.sidetabs import SideTabs

    tabs = SideTabs()
    config, monitor = QWidget(), QWidget()
    tabs.addTab(config, "Config", tab_id="config")
    tabs.addTab(monitor, "Monitor", tab_id="monitor")
    slider = QCheckBox()
    host = SimpleNamespace(tabs=tabs, pane_sliders={"monitor": slider}, active_runs=[], runs=[],
                           _queue_has_active_jobs=None, queue_running=True,
                           run_for_job=lambda _: None, update_pane_sliders_state=Mock())
    host._set_monitor_active = lambda active: Window._set_monitor_active(host, active)
    data = {"running": True, "active": [], "queued": []}
    data[job_bucket] = [{"job_id": "job", "position": 0}]
    Window.on_queue_changed(host, data)
    assert tabs.is_tab_visible("monitor") and slider.isChecked()
    # Reproduce restored settings hiding the entry while its page is open.
    tabs.set_tab_visible("monitor", False)
    slider.setChecked(False)
    tabs.setCurrentWidget(monitor)
    Window.on_queue_changed(host, data)
    assert tabs.currentWidget() is monitor
    assert tabs.is_tab_visible("monitor") and slider.isChecked()
    tabs.close()


def test_loading_saved_layout_keeps_monitor_visible_during_active_work():
    from unittest.mock import Mock
    from PyQt6.QtWidgets import QCheckBox, QWidget
    from frontend.sidetabs import SideTabs

    tabs = SideTabs()
    tabs.addTab(QWidget(), "Config", tab_id="config")
    tabs.addTab(QWidget(), "Monitor", tab_id="monitor")
    slider = QCheckBox()
    host = SimpleNamespace(tabs=tabs, pane_sliders={"monitor": slider}, _queue_has_active_jobs=True,
                           settings_manager=SimpleNamespace(sidebar_order=["config", "monitor"],
                                                            sidebar_visible=["config"], get=lambda *a, **kw: []),
                           update_pane_sliders_state=Mock(), log=Mock())
    Window.load_layout(host)
    assert tabs.is_tab_visible("monitor") and slider.isChecked()
    host.log.assert_not_called()
    tabs.close()

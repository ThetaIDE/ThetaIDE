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
    host = SimpleNamespace(selected={"id": "first"}, run_folder=lambda run: tmp_path / run["id"])
    monitor = MonitorPanel(host)
    monitor.resize(900, 700)
    monitor.show()
    assert monitor.status_strip.badge.isHidden()
    assert monitor.run_caption.isHidden()
    QTest.mouseClick(monitor.yaml_button, Qt.MouseButton.LeftButton)
    assert monitor.yaml_overlay.isVisible()
    assert monitor.yaml_viewer.toPlainText() == "method: ppo\n"
    host.selected = {"id": "second"}
    monitor.show_selected(host.selected)
    assert monitor.yaml_viewer.toPlainText() == "method: dqn\n"
    QTest.mouseClick(monitor, Qt.MouseButton.LeftButton)
    assert not monitor.yaml_overlay.isVisible()
    assert not monitor.yaml_button.isChecked()
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
                           progress=lambda: 40, artifacts=[])
    host = SimpleNamespace(jobs=jobs, selected=None, run_for_job=lambda job_id: {"description": "My run"},
                           live_for_run=lambda run, load=False: {"ppo": live},
                           open_job=lambda job_id, method: opened.append((job_id, method)))
    monitor = MonitorPanel(host)
    monitor.resize(1000, 700)
    monitor.show()
    QTest.mouseClick(monitor.queue_button, Qt.MouseButton.LeftButton)
    assert list(monitor.queue_preview.cards) == ["one:ppo", "one:dqn", "two:sac"]
    assert monitor.queue_preview.cards["one:ppo"].description.text() == "My run"
    assert monitor.queue_preview.cards["two:sac"].elapsed.text() == "Waiting"
    assert monitor.queue_preview.cards["one:ppo"].progress_fraction == 0.4
    assert monitor.queue_preview.cards["one:dqn"].progress_fraction is None
    assert monitor.queue_overlay.height() < monitor.content.height()
    live.progress = lambda: 75
    monitor.queue_preview._update_times()
    assert monitor.queue_preview.cards["one:ppo"].progress_fraction == 0.75
    jobs.queue_changed.emit({})
    assert list(monitor.queue_preview.cards) == ["one:ppo", "one:dqn", "two:sac"]
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
    run["backend"]["agents"] = ["ppo", "dqn"]
    dqn = LiveRun()
    dqn.rows = [{"step": 1, "eval/reward": 42.0}]
    lives = {"dqn": dqn}
    host = SimpleNamespace(selected=run, pinned_baseline_run=None, queue_running=False,
                           live_for_run=lambda run, load=True: lives)
    monitor = MonitorPanel(host)
    monitor.render()
    monitor.refresh_live()
    assert monitor.run_title.text() == "sb3 / cartpole_sb3_ppo"
    assert "ppo" in monitor.method_legend.text() and "dqn" in monitor.method_legend.text()
    assert theme_color(PALETTE[1]) in monitor.method_legend.text()
    assert monitor.charts_view._series_colors["dqn"] == PALETTE[1]
    assert monitor.reward_chart.series[0][0] == "dqn"
    assert theme_color(monitor.reward_chart.series[0][2]) == theme_color(PALETTE[1])
    assert not monitor.charts_view.grid.lead_hidden
    monitor.focus_agent("dqn")
    assert monitor.charts_view._series_colors["dqn"] == PALETTE[1]
    monitor.close()

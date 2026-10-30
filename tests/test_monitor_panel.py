"""Training monitor panel (frontend/monitor): extracted from the main window, metric-agnostic."""
import os
import sys
import tempfile
import unittest
from pathlib import Path

import pytest

pytest.importorskip("PyQt6")

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PyQt6.QtCore import Qt  # noqa: E402
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

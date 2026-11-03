import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
import pytest

pytest.importorskip("PyQt6")

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication
QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True)
app = QApplication.instance() or QApplication(sys.argv[:1])

from frontend.app import Window
from frontend.model import Config, new_run, sample


class TestRunNavigator(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp_dir.name)
        self.window = Window(self.data_dir)
        initial = new_run(Config(name="initial_run", env="test_env"))
        initial["metrics"] = [
            {"step": 0, "reward": 10.0, "entropy": 0.69},
            {"step": 100, "reward": 50.0, "entropy": 0.50},
        ]
        self.window.runs.append(initial)
        self.window.refresh_runs()
        self.window.select_run(initial)

    def tearDown(self):
        self.window.close()
        self.temp_dir.cleanup()

    def test_the_runs_list_lists_every_run_and_clicking_one_plots_it(self):
        """The runs list plots runs; the monitor keeps showing the most recent one."""
        sidebar = self.window.monitor.sidebar
        self.assertGreaterEqual(len(sidebar.items), 1)
        self.assertEqual(set(sidebar.items), {run["id"] for run in self.window.runs})
        self.assertIs(self.window.selected, self.window.latest_run())
        shown = self.window.selected
        extra = new_run(Config(name="extra_run", seed=99))
        self.window.runs.append(extra)
        self.window.refresh_runs()
        item = sidebar.items[extra["id"]]
        sidebar.tree.itemPressed.emit(item, 0)
        sidebar.tree.itemClicked.emit(item, 0)
        self.assertEqual(sidebar.compared, [extra["id"]])
        self.assertIs(self.window.selected, shown)
        sidebar.tree.itemPressed.emit(item, 0)  # a second click unticks it
        sidebar.tree.itemClicked.emit(item, 0)
        self.assertEqual(sidebar.compared, [])

    def test_the_run_title_says_where_the_run_is_stored(self):
        """Hovering the run's name tells where its records are kept."""
        self.window.render_run()
        self.assertIn("jobs.db", self.window.run_title.toolTip())

    def test_job_signals_update_monitor(self):
        """Generic job signals (output, metrics, hardware) update the monitor."""
        run = new_run(Config(name="tests_cartpole", seed=42), simulated=False)
        run["backend"] = {
            "job_id": "test-job-123",
            "group": "tests",
            "experiment_id": "cartpole_test",
            "agents": ["test_method"],
            "total_timesteps": 640,
        }
        self.window.runs.insert(0, run)
        self.window.active = run
        self.window.select_run(run)

        rows = [
            {"eval/reward": 78.5, "eval/reward_std": 12.0, "step": 0.0, "transitions": 0.0},
            {"losses/total_loss": 32.1, "losses/entropy": 0.61, "step": 0.0, "transitions": 512.0},
        ]
        self.window.on_job_output("test-job-123", ["Training in progress..."])
        self.window.on_job_metrics("test-job-123", "test_method", rows, False)
        self.window.on_job_hardware("test-job-123", {"cpu_percent": 15.2, "memory_mb": 210.5})
        self.assertIn("Training in progress...", self.window.console.toPlainText())
        self.assertEqual(len(run["metrics"]), 2)
        self.assertEqual(run["metrics"][0]["reward"], 78.5)
        self.assertIn("15.2%", self.window.monitor.view_buttons["System"].toolTip())


if __name__ == "__main__":
    unittest.main()

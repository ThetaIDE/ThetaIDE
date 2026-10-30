"""Unit tests for Experiment Parameter CRUD in ThetaIDE:
- Removing parameters like total_timesteps, seed, eval_episodes to rely on group defaults
- Adding parameters back via "+ Add Option ▾" and ensuring they land in the right spot in the GUI
- Adding arbitrary custom parameters and editing/removing them
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

os.environ["QT_QPA_PLATFORM"] = "offscreen"

try:
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QApplication, QCheckBox, QComboBox, QLabel, QLineEdit

    from frontend.config_viewer import ConfigViewer, SpinBox
    HAS_PYQT6 = True
    app = QApplication.instance() or QApplication(sys.argv[:1])
except ImportError:
    HAS_PYQT6 = False


@unittest.skipIf(not HAS_PYQT6, "PyQt6 not installed in current environment")
class TestExperimentParamsCrud(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.file_path = Path(self.temp_dir.name) / "test_exp.yaml"
        self.file_path.write_text(
            "experiment_id: test_exp\n"
            "paradigm: online_rl\n"
            "seed: 123\n"
            "total_timesteps: 25000\n"
            "eval_episodes: 50\n"
            "tensorboard: true\n"
            "methods:\n"
            "  ppo:\n"
            "    agent: ppo\n"
            "    model: dnn\n",
            encoding="utf-8"
        )
        self.viewer = ConfigViewer()
        self.viewer.auto_save = True
        self.viewer.load_file(self.file_path, "experiment/test_exp.yaml")

    def tearDown(self):
        self.viewer.close()
        self.temp_dir.cleanup()

    def test_timestamp_toggle_defaults_off_and_persists(self):
        self.assertFalse(self.viewer.chk_timestamp_experiment.isChecked())
        self.viewer.chk_timestamp_experiment.setChecked(True)
        self.assertTrue(yaml.safe_load(self.file_path.read_text())["timestamp_experiment"])
        self.viewer._render_boxes()
        self.assertTrue(self.viewer.chk_timestamp_experiment.isChecked())
        self.viewer.chk_timestamp_experiment.setChecked(False)
        self.assertFalse(yaml.safe_load(self.file_path.read_text())["timestamp_experiment"])

    def test_remove_total_timesteps_relies_on_default(self):
        """Removing total_timesteps pops it from config to rely on group base."""
        self.assertIn("total_timesteps", self.viewer.raw_data)
        self.assertEqual(self.viewer.raw_data["total_timesteps"], 25000)

        # Simulate clicking the remove button for total_timesteps
        self.viewer.raw_data.pop("total_timesteps", None)
        self.viewer._mark_dirty()
        self.viewer._render_boxes()

        self.assertNotIn("total_timesteps", self.viewer.raw_data)
        disk_data = yaml.safe_load(self.file_path.read_text(encoding="utf-8"))
        self.assertNotIn("total_timesteps", disk_data)

    def test_remove_seed_relies_on_default(self):
        """Removing seed pops it from config to rely on group base."""
        self.assertIn("seed", self.viewer.raw_data)
        self.assertEqual(self.viewer.raw_data["seed"], 123)

        # Simulate clicking remove button for seed
        self.viewer.raw_data.pop("seed", None)
        self.viewer._mark_dirty()
        self.viewer._render_boxes()

        self.assertNotIn("seed", self.viewer.raw_data)
        disk_data = yaml.safe_load(self.file_path.read_text(encoding="utf-8"))
        self.assertNotIn("seed", disk_data)

    def test_remove_eval_episodes_relies_on_default(self):
        """Removing eval_episodes pops it from config to rely on group base."""
        self.assertIn("eval_episodes", self.viewer.raw_data)
        self.assertEqual(self.viewer.raw_data["eval_episodes"], 50)

        # Simulate clicking remove button for eval_episodes
        self.viewer.raw_data.pop("eval_episodes", None)
        self.viewer._mark_dirty()
        self.viewer._render_boxes()

        self.assertNotIn("eval_episodes", self.viewer.raw_data)
        disk_data = yaml.safe_load(self.file_path.read_text(encoding="utf-8"))
        self.assertNotIn("eval_episodes", disk_data)

    def test_add_option_places_params_in_right_spot(self):
        """Adding options via _add_experiment_param adds them and positions them in the GUI."""
        # Start without total_timesteps and without seed
        self.viewer.raw_data.pop("total_timesteps", None)
        self.viewer.raw_data.pop("seed", None)
        self.viewer._mark_dirty()
        self.viewer._render_boxes()

        self.assertNotIn("total_timesteps", self.viewer.raw_data)
        self.assertNotIn("seed", self.viewer.raw_data)

        # Add total_timesteps via Add Option
        self.viewer._add_experiment_param("total_timesteps", 50000)
        self.assertIn("total_timesteps", self.viewer.raw_data)
        self.assertEqual(self.viewer.raw_data["total_timesteps"], 50000)
        self.assertEqual(self.viewer.spin_timesteps.value(), 50000)

        # Add seed via Add Option
        self.viewer._add_experiment_param("seed", 777)
        self.assertIn("seed", self.viewer.raw_data)
        self.assertEqual(self.viewer.raw_data["seed"], 777)
        self.assertEqual(self.viewer.spin_seed.value(), 777)

    def test_add_custom_parameter(self):
        """Custom parameters can be added, edited, and removed."""
        self.viewer._add_experiment_param("eval_freq_steps", 5000)
        self.assertIn("eval_freq_steps", self.viewer.raw_data)
        self.assertEqual(self.viewer.raw_data["eval_freq_steps"], 5000)

        disk_data = yaml.safe_load(self.file_path.read_text(encoding="utf-8"))
        self.assertEqual(disk_data["eval_freq_steps"], 5000)

        # Now remove custom parameter
        self.viewer.raw_data.pop("eval_freq_steps", None)
        self.viewer._mark_dirty()
        self.viewer._render_boxes()

        self.assertNotIn("eval_freq_steps", self.viewer.raw_data)
        disk_data_after = yaml.safe_load(self.file_path.read_text(encoding="utf-8"))
        self.assertNotIn("eval_freq_steps", disk_data_after)

    def test_checkpoints_are_every_n_timesteps_or_a_total_count(self):
        """The Checkpoints row edits one key; switching its mode swaps it for the other, value converted,
        without rebuilding the page (which used to throw the scroll position to the bottom)."""
        self.viewer._add_experiment_param("num_checkpoints", 10)
        self.assertEqual(self.viewer.field_widgets["num_checkpoints"].value(), 10)
        mode = next(
            c for c in self.viewer.findChildren(QComboBox)
            if c.count() == 2 and c.itemText(0) == "Every N timesteps"
        )
        self.viewer.resize(900, 300)
        bar = self.viewer.scroll.verticalScrollBar()
        bar.setValue(bar.maximum() // 2)
        before = bar.value()
        mode.setCurrentIndex(mode.findData("checkpoint_every_steps"))
        self.assertNotIn("num_checkpoints", self.viewer.raw_data)
        self.assertEqual(self.viewer.raw_data["checkpoint_every_steps"], 2500)  # 25,000 timesteps / 10
        self.assertEqual(self.viewer.field_widgets["checkpoint_every_steps"].value(), 2500)
        self.assertEqual(bar.value(), before)
        self.viewer.field_widgets["checkpoint_every_steps"].setValue(5000)
        mode.setCurrentIndex(mode.findData("num_checkpoints"))
        disk_data = yaml.safe_load(self.file_path.read_text(encoding="utf-8"))
        self.assertEqual(disk_data["num_checkpoints"], 5)
        self.assertNotIn("checkpoint_every_steps", disk_data)

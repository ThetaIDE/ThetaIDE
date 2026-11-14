"""Tests that generic experiment settings render and round-trip in the editor."""
import os
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

os.environ["QT_QPA_PLATFORM"] = "offscreen"

try:
    from PyQt6 import QtWebEngineWidgets  # noqa: F401
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QApplication, QLabel, QLineEdit

    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True)
    app = QApplication.instance() or QApplication(sys.argv[:1])
    from frontend.config_viewer import FIXED_BOX_KEYS, ConfigViewer, parse_list_field
    from frontend.widgets import ToggleSlider

    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False

def _write_sample(config_root):
    path = Path(config_root) / "experiment" / "fixture_supervised" / "sample.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# @package _global_\n"
        "paradigm: supervised\n"
        "experiment_id: fixture_sample\n"
        "custom_component:\n"
        "  n_splits: 3\n"
        "  use_feature: false\n"
        "  horizons: [1, 6, 12]\n"
        "  target_model: fixture_mlp\n"
        "  nested_probe:\n"
        "    encoder:\n"
        "      width: 64\n"
        "custom_list: [fixture_plot]\n"
        "custom_flag: true\n"
        "sweep: true\n",
        encoding="utf-8",
    )
    return path


# ---------------------------------------------------------------------------
# parse_list_field: no Qt needed
# ---------------------------------------------------------------------------


@unittest.skipIf(not HAS_PYQT6, "PyQt6 not installed in current environment")
class TestParseListField(unittest.TestCase):
    def test_names_stay_strings(self):
        self.assertEqual(parse_list_field("ep_dl_sweep, ep_agreement"), ["ep_dl_sweep", "ep_agreement"])

    def test_numbers_stay_numbers(self):
        """Numeric list values must not become strings in the config."""
        self.assertEqual(parse_list_field("1, 6, 12"), [1, 6, 12])
        self.assertTrue(all(isinstance(v, int) for v in parse_list_field("1, 6, 12")))

    def test_mixed_scalars_keep_their_types(self):
        self.assertEqual(parse_list_field("true, 0.5, x"), [True, 0.5, "x"])

    def test_blank_entries_are_dropped(self):
        self.assertEqual(parse_list_field(" a ,, b "), ["a", "b"])

    def test_empty_text_is_an_empty_list(self):
        self.assertEqual(parse_list_field(""), [])

    def test_unparseable_text_is_kept_verbatim(self):
        self.assertEqual(parse_list_field("[unclosed"), ["[unclosed"])


# ---------------------------------------------------------------------------
# Which keys belong to which renderer
# ---------------------------------------------------------------------------


@unittest.skipIf(not HAS_PYQT6, "PyQt6 not installed in current environment")
class TestKeyPartition(unittest.TestCase):
    def test_the_standard_rl_keys_are_claimed_by_the_fixed_boxes(self):
        for key in ("seed", "total_timesteps", "eval_episodes", "methods", "env"):
            self.assertIn(key, FIXED_BOX_KEYS)

    def test_paradigm_specific_keys_are_not_claimed(self):
        """These are exactly what used to be invisible."""
        for key in ("custom_component", "eval_protocol", "plots", "consolidate", "partition", "sweep"):
            self.assertNotIn(key, FIXED_BOX_KEYS)

    def test_keys_the_panel_overhaul_owns_are_claimed(self):
        """The sweep box edits hydra.sweeper, the dataset row edits dataset_path
        and the Checkpoint Settings box edits checkpoint, so this pass must leave
        them alone or the same setting appears twice."""
        for key in ("hydra", "dataset_path", "offline_datasets", "experiment_id", "checkpoint"):
            self.assertIn(key, FIXED_BOX_KEYS)


def _headings(viewer):
    return [w.text() for w in viewer.findChildren(QLabel) if w.objectName() == "cardTitle"]


def _open(path, config_root):
    rel = path.relative_to(config_root).as_posix()
    viewer = ConfigViewer()
    viewer.load_file(path, rel)
    return viewer


@unittest.skipIf(not HAS_PYQT6, "PyQt6 not installed in current environment")
class TestSupervisedConfigurationIsVisible(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.sample = _write_sample(self.root)

    def test_custom_component_configuration_gets_a_heading(self):
        viewer = _open(self.sample, self.root)
        self.addCleanup(viewer.close)
        self.assertIn("custom_component", _headings(viewer))

    def test_every_complex_section_outside_the_fixed_boxes_gets_a_heading(self):
        path = self.sample
        raw = yaml.safe_load(path.read_text())
        expected = {k for k, v in raw.items() if k not in FIXED_BOX_KEYS and isinstance(v, (dict, list))}
        viewer = _open(path, self.root)
        self.addCleanup(viewer.close)
        self.assertTrue(expected <= set(_headings(viewer)))

    def test_a_scalar_outside_the_fixed_boxes_appears_once(self):
        viewer = _open(self.sample, self.root)
        self.addCleanup(viewer.close)
        self.assertNotIn("Additional Settings", _headings(viewer))
        self.assertEqual(sum(w.text() == "custom_flag" for w in viewer.findChildren(QLabel)), 1)

    def test_the_settings_are_editable_not_just_displayed(self):
        viewer = _open(self.sample, self.root)
        self.addCleanup(viewer.close)
        texts = {w.text() for w in viewer.findChildren(QLineEdit)}
        self.assertIn("fixture_mlp", texts, "target_model should be in an editable field")

    def test_a_boolean_setting_renders_as_a_toggle_slider(self):
        """A spin box would write 0/1 back over true/false; booleans render as toggle sliders."""
        viewer = _open(self.sample, self.root)
        self.addCleanup(viewer.close)
        self.assertTrue(viewer.findChildren(ToggleSlider))

    def test_no_section_is_rendered_twice(self):
        """The fixed boxes and this pass must partition the keys between them."""
        viewer = _open(self.sample, self.root)
        self.addCleanup(viewer.close)
        headings = _headings(viewer)
        self.assertEqual(len(headings), len(set(headings)), f"duplicates in {headings}")

    def test_a_section_the_fixed_boxes_own_is_left_to_them(self):
        """Hydra-owned settings stay out of the generic section renderer."""
        viewer = _open(self.sample, self.root)
        self.addCleanup(viewer.close)
        viewer.raw_data["hydra"] = {"sweeper": {"name": "fixture_sweeper"}}
        viewer._render_boxes()
        headings = _headings(viewer)
        self.assertNotIn("hydra", headings)


# ---------------------------------------------------------------------------
# Round trip
# ---------------------------------------------------------------------------


@unittest.skipIf(not HAS_PYQT6, "PyQt6 not installed in current environment")
class TestRoundTrip(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.tmp = _write_sample(self.root)
        self.before = yaml.safe_load(self.tmp.read_text())
        self.viewer = _open(self.tmp, self.root)
        self.addCleanup(self.viewer.close)

    def _save_and_reload(self):
        self.viewer.save_to_disk()
        return yaml.safe_load(self.tmp.read_text())

    def test_an_integer_stays_an_integer(self):
        """A stringified tau_min or n_splits would reach the backend as text."""
        self.viewer._on_nested_edited("custom_component", "n_splits", 4)
        value = self._save_and_reload()["custom_component"]["n_splits"]
        self.assertEqual(value, 4)
        self.assertIsInstance(value, int)

    def test_a_boolean_stays_a_boolean(self):
        self.viewer._on_nested_edited("custom_component", "use_feature", True)
        value = self._save_and_reload()["custom_component"]["use_feature"]
        self.assertIs(value, True)

    def test_a_list_of_numbers_stays_a_list_of_numbers(self):
        self.viewer._on_nested_edited("custom_component", "horizons", [1, 6, 12])
        value = self._save_and_reload()["custom_component"]["horizons"]
        self.assertEqual(value, [1, 6, 12])
        self.assertTrue(all(isinstance(v, int) for v in value))

    def test_a_top_level_list_round_trips(self):
        self.viewer._on_field_edited("custom_list", ["fixture_plot", "fixture_report"])
        self.assertEqual(self._save_and_reload()["custom_list"], ["fixture_plot", "fixture_report"])

    def test_untouched_settings_are_preserved(self):
        self.viewer._on_nested_edited("custom_component", "n_splits", 4)
        after = self._save_and_reload()["custom_component"]
        for key, value in self.before["custom_component"].items():
            if key != "n_splits":
                with self.subTest(setting=key):
                    self.assertEqual(after[key], value)

    def test_no_key_is_lost(self):
        self.viewer._on_nested_edited("custom_component", "n_splits", 4)
        self.assertTrue(set(self.before) <= set(self._save_and_reload()))

    def test_the_package_directive_survives(self):
        """It is line 1 of all 67 configs and yaml.safe_dump does not keep comments."""
        self.viewer._on_nested_edited("custom_component", "n_splits", 4)
        self.viewer.save_to_disk()
        self.assertTrue(self.tmp.read_text().startswith("# @package _global_"))

    def test_editing_a_list_through_its_widget_writes_a_list(self):
        """The widget holds text; what lands in the config must not be a string."""
        fields = [w for w in self.viewer.findChildren(QLineEdit) if w.text() == "fixture_plot"]
        self.assertTrue(fields, "expected the custom list editor")
        fields[0].setText("fixture_plot, fixture_report")
        fields[0].editingFinished.emit()
        self.assertEqual(self._save_and_reload()["custom_list"], ["fixture_plot", "fixture_report"])

    def test_deeply_nested_experiment_setting_is_rendered_and_round_trips(self):
        """A component may put parameters below more than one grouping level."""
        self.viewer.raw_data["custom_component"]["nested_probe"] = {"encoder": {"width": 64}}
        self.viewer._render_boxes()
        fields = [field for field in self.viewer.findChildren(QLineEdit) if field.text() == "64"]
        self.assertTrue(fields, "expected a writable field for custom_component.nested_probe.encoder.width")
        fields[0].setText("128")
        saved = self._save_and_reload()
        self.assertEqual(saved["custom_component"]["nested_probe"]["encoder"]["width"], 128)


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------------------
# Keeping a group valid when its paradigm is chosen
# ---------------------------------------------------------------------------


@unittest.skipIf(not HAS_PYQT6, "PyQt6 not installed in current environment")
class TestParadigmChangeKeepsTheConfigValid(unittest.TestCase):
    """Inline group creation writes `paradigm: null` and the paradigm is picked
    afterwards, so the base has to be reconciled at that point. Without it the
    group inherits intervals_count: 4 and eval_episodes: 100 from config.yaml,
    which every non-RL paradigm rejects before training starts."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dir = Path(self.temp_dir.name) / "experiment" / "_ParadigmProbe"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.base = self.dir / "_base.yaml"
        self.base.write_text(
            "# @package _global_\ndefaults:\n  - override /env: mimic\n\n"
            "paradigm: null\n\nseed: 1\nsave_dataset: false\nrecover: false\n"
        )
        self.viewer = ConfigViewer()
        self.viewer.load_file(self.base, "experiment/_ParadigmProbe/_base.yaml")

    def tearDown(self):
        self.viewer.close()
        self.temp_dir.cleanup()

    def _pick(self, paradigm):
        self.viewer._on_paradigm_changed(paradigm)
        self.viewer.save_to_disk()
        return yaml.safe_load(self.base.read_text())

    def test_supervised_pins_the_fields_it_forbids(self):
        data = self._pick("supervised")
        self.assertNotIn("intervals_count", data)
        self.assertEqual(data["eval_episodes"], 0)

    def test_unsupervised_pins_them_too(self):
        data = self._pick("unsupervised")
        self.assertNotIn("intervals_count", data)
        self.assertEqual(data["eval_episodes"], 0)

    def test_offline_rl_pins_them_too(self):
        """offline_rl forbids both as well; mimic/_base.yaml is the reference."""
        data = self._pick("offline_rl")
        self.assertNotIn("intervals_count", data)
        self.assertEqual(data["eval_episodes"], 0)

    def test_online_rl_leaves_them_to_the_defaults(self):
        self._pick("supervised")
        data = self._pick("online_rl")
        self.assertNotIn("intervals_count", data)
        self.assertNotIn("eval_episodes", data)

    def test_switching_back_does_not_clobber_a_deliberate_value(self):
        """Only the pin this wrote is cleared, not a value the user chose."""
        self._pick("supervised")
        self.viewer._on_field_edited("eval_episodes", 7)
        data = self._pick("online_rl")
        self.assertEqual(data["eval_episodes"], 7)

    def test_a_group_pinned_this_way_satisfies_the_backend(self):
        """Non-online bases pin the values required by their paradigm rules."""
        data = self._pick("supervised")
        self.assertEqual(data["eval_episodes"], 0)
        self.assertNotIn("intervals_count", data)

    def test_an_empty_methods_block_is_not_written(self):
        """`methods: {}` satisfies non_empty no better than having none, and an
        absent block lets the group base supply one."""
        self.viewer.raw_data["methods"] = {}
        self.assertNotIn("methods", self._pick("supervised"))

    def test_universal_parameters_are_kept_on_a_group_base(self):
        """methods.params holds settings every method inherits, so a block that
        holds only params is a real one. This used to be dropped, back when
        params was incidental rather than something the panel lets you create."""
        self.viewer.raw_data["methods"] = {"params": {"lr": 0.0003}}
        self.assertEqual(self._pick("supervised")["methods"], {"params": {"lr": 0.0003}})

    def test_an_empty_params_entry_is_still_dropped(self):
        self.viewer.raw_data["methods"] = {"params": {}}
        self.assertNotIn("methods", self._pick("supervised"))

    def test_a_group_base_keeps_its_default_method(self):
        """A group base may declare one default method, which new experiments
        inherit; picking the paradigm and saving must not strip it. (Bases used to
        drop every concrete method, which deleted that default on the first save.)"""
        self.viewer.raw_data["methods"] = {"m1": {"model": "lstm"}}
        self.assertEqual(self._pick("supervised")["methods"], {"m1": {"model": "lstm"}})

    def test_an_experiment_keeps_its_concrete_methods(self):
        """The same save path must not strip a method from an experiment file."""
        experiment = self.dir / "run.yaml"
        experiment.write_text(
            "# @package _global_\ndefaults:\n  - _ParadigmProbe/_base\n\n"
            "experiment_id: run\nmethods:\n  m1:\n    model: lstm\n"
        )
        viewer = ConfigViewer()
        viewer.load_file(experiment, "experiment/_ParadigmProbe/run.yaml")
        viewer._on_field_edited("seed", 7)
        viewer.save_to_disk()
        self.assertEqual(yaml.safe_load(experiment.read_text())["methods"], {"m1": {"model": "lstm"}})

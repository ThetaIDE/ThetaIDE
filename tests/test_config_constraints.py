"""Tests for paradigm settings and compatibility requirements driving the ConfigViewer form.

Covers the wiring between frontend/config_model.py and the experiment pane:
a paradigm enables or disables its own fields, and agent choices follow the
algorithms' declared requirements, so an invalid experiment cannot be
constructed in the UI at all.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "offscreen"

try:
    from PyQt6.QtWidgets import QApplication, QComboBox

    app = QApplication.instance() or QApplication(sys.argv[:1])
    from frontend.config_model import ConfigTree
    from frontend.config_viewer import ConfigViewer
    from frontend.config_panel.catalogs import config_tree, invalidate_config_tree
    from frontend.config_panel.catalogs import set_config_tree

    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False


_SHARED_TMP = None
_MOCK_TREE = None


def setUpModule():
    global _SHARED_TMP, _MOCK_TREE
    if not HAS_PYQT6:
        return

    _SHARED_TMP = tempfile.TemporaryDirectory()
    root = Path(_SHARED_TMP.name) / "src/usr/config"

    # Paradigms
    (root / "paradigms").mkdir(parents=True, exist_ok=True)
    (root / "paradigms" / "online_rl.yaml").write_text("allows_eval_episodes: true\n", encoding="utf-8")
    (root / "paradigms" / "offline_rl.yaml").write_text("description: offline\n", encoding="utf-8")

    # Agents: a package whose algorithms declare the paradigms they run in.
    import json
    from src.app.workspace import WorkspacePaths
    package = WorkspacePaths(Path(_SHARED_TMP.name)).user("algorithm") / "fixture-agents"
    (package / "configs/agent").mkdir(parents=True)
    (package / "agents.py").write_text("class Agent:\n    pass\n", encoding="utf-8")
    agents = {"ppo": "online_rl", "cql": "offline_rl", "iql": "offline_rl", "cew": "offline_rl"}
    for name in agents:
        (package / f"configs/agent/{name}.yaml").write_text(f"algorithm: {name}\n", encoding="utf-8")
    (package / ".theta_component.json").write_text(json.dumps({
        "id": "fixture-agents", "name": "Fixture agents", "kind": "method", "version": "1.0.0",
        "storage_version": 3, "api_version": 1,
        "capabilities": [{"id": name, "kind": "algorithm", "entry_point": "agents.py:Agent",
                          "requires": [{"level": "block", "config": "paradigm", "in": [paradigm]}]}
                         for name, paradigm in agents.items()],
        "payload": {"implementations": ["agents.py"], "resources": [],
                    "configs": {f"agent/{name}.yaml": f"configs/agent/{name}.yaml" for name in agents}}}),
        encoding="utf-8")

    # Models
    (root / "model").mkdir(parents=True, exist_ok=True)
    (root / "model" / "dnn.yaml").write_text(
        "architecture: dnn\nhidden_sizes: [64, 64]\nactivation: relu\n",
        encoding="utf-8",
    )

    # Environments
    (root / "env").mkdir(parents=True, exist_ok=True)
    (root / "env" / "cartpole.yaml").write_text(
        "description: cartpole\n",
        encoding="utf-8",
    )
    (root / "env" / "mimic.yaml").write_text("description: mimic\n", encoding="utf-8")
    (root / "env" / "pyrenees.yaml").write_text("description: pyrenees\n", encoding="utf-8")

    # Experiments and groups
    exp_dir = root / "experiment"
    (exp_dir / "cartpole").mkdir(parents=True, exist_ok=True)
    (exp_dir / "cartpole" / "_base.yaml").write_text(
        "paradigm: online_rl\nenv: cartpole\neval_episodes: 10\n",
        encoding="utf-8",
    )
    (exp_dir / "cartpole" / "demo.yaml").write_text(
        "experiment_id: cp_demo\ntotal_timesteps: 10000\n",
        encoding="utf-8",
    )

    (exp_dir / "mimic").mkdir(parents=True, exist_ok=True)
    (exp_dir / "mimic" / "_base.yaml").write_text(
        "paradigm: offline_rl\nenv: mimic\neval_episodes: 0\n",
        encoding="utf-8",
    )
    (exp_dir / "mimic" / "demo.yaml").write_text(
        "experiment_id: mimic_demo\n",
        encoding="utf-8",
    )

    _MOCK_TREE = ConfigTree(root)
    set_config_tree(_MOCK_TREE)


def tearDownModule():
    global _SHARED_TMP, _MOCK_TREE
    invalidate_config_tree()
    if _SHARED_TMP is not None:
        _SHARED_TMP.cleanup()
        _SHARED_TMP = None
    _MOCK_TREE = None


def an_experiment_in(group):
    """Any experiment in `group`, as (path, rel_path)."""
    if _SHARED_TMP is not None:
        directory = Path(_SHARED_TMP.name) / "src/usr/config/experiment" / group
        for path in sorted(directory.glob("*.yaml")):
            if not path.name.startswith("_"):
                return path, f"experiment/{group}/{path.name}"
    raise AssertionError(f"no experiment configs left in {directory}")


ONLINE_EXPERIMENT = (
    "paradigm: online_rl\n"
    "experiment_id: cp_demo\n"
    "seed: 42\n"
    "total_timesteps: 5000\n"
    "eval_episodes: 10\n"
    "methods:\n"
    "  ppo:\n"
    "    agent: ppo\n"
    "    model: dnn\n"
)

OFFLINE_EXPERIMENT = (
    "paradigm: offline_rl\n"
    "experiment_id: mimic_demo\n"
    "seed: 7\n"
    "total_timesteps: 5000\n"
    "methods:\n"
    "  cql:\n"
    "    agent: cql\n"
    "    model: dnn\n"
)


@unittest.skipIf(not HAS_PYQT6, "PyQt6 not installed in current environment")
class ConfigViewerConstraintTest(unittest.TestCase):
    """Base fixture: writes an experiment file and loads it into a viewer."""

    def load(self, text):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        path = Path(self.temp_dir.name) / "exp.yaml"
        path.write_text(text, encoding="utf-8")
        viewer = ConfigViewer()
        self.addCleanup(viewer.close)
        viewer.load_file(path, "experiment/exp.yaml")
        return viewer

    def agent_widget(self, viewer):
        for row in viewer.findChildren(QComboBox):
            if row.objectName().startswith("methodAgent_"):
                return row
        return None


@unittest.skipIf(not HAS_PYQT6, "PyQt6 not installed in current environment")
class TestConfigTreeAvailable(unittest.TestCase):
    def test_config_tree_is_discoverable(self):
        self.assertIsNotNone(config_tree(), "frontend must find src/usr/config from a checkout")

    def test_config_tree_is_cached(self):
        self.assertIs(config_tree(), config_tree())


@unittest.skipIf(not HAS_PYQT6, "PyQt6 not installed in current environment")
class TestParadigmSelector(ConfigViewerConstraintTest):
    def test_paradigm_combo_is_created(self):
        viewer = self.load(ONLINE_EXPERIMENT)
        self.assertIn("paradigm", viewer.field_widgets)
        self.assertIsInstance(viewer.combo_paradigm, QComboBox)

    def test_paradigm_combo_reflects_file(self):
        viewer = self.load(OFFLINE_EXPERIMENT)
        self.assertEqual(viewer.combo_paradigm.currentText(), "offline_rl")

    def test_paradigm_combo_lists_every_paradigm(self):
        viewer = self.load(ONLINE_EXPERIMENT)
        listed = {viewer.combo_paradigm.itemText(i) for i in range(viewer.combo_paradigm.count())}
        self.assertEqual(listed - {"— Select Paradigm —"}, set(config_tree().paradigms))
        self.assertIn("— Select Paradigm —", listed)

    def test_changing_paradigm_updates_raw_data(self):
        viewer = self.load(ONLINE_EXPERIMENT)
        viewer.combo_paradigm.setCurrentText("offline_rl")
        self.assertEqual(viewer.raw_data["paradigm"], "offline_rl")

    def test_changing_paradigm_marks_dirty(self):
        viewer = self.load(ONLINE_EXPERIMENT)
        self.assertFalse(viewer.is_dirty)
        viewer.combo_paradigm.setCurrentText("offline_rl")
        self.assertTrue(viewer.is_dirty)


@unittest.skipIf(not HAS_PYQT6, "PyQt6 not installed in current environment")
class TestEnvironmentFiltering(ConfigViewerConstraintTest):
    def test_online_offers_environments(self):
        viewer = self.load(ONLINE_EXPERIMENT)
        offered = {viewer.combo_env.itemText(i) for i in range(viewer.combo_env.count())}
        offered.discard(ConfigViewer.INHERIT)
        self.assertTrue(offered)
        self.assertIn("cartpole", offered)

    def test_offline_offers_environments(self):
        viewer = self.load(OFFLINE_EXPERIMENT)
        offered = {viewer.combo_env.itemText(i) for i in range(viewer.combo_env.count())}
        offered.discard(ConfigViewer.INHERIT)
        self.assertTrue(offered)
        self.assertIn("mimic", offered)

    def test_env_defaults_to_inherit_when_file_omits_it(self):
        viewer = self.load(ONLINE_EXPERIMENT)
        self.assertEqual(viewer.combo_env.currentText(), ConfigViewer.INHERIT)

    def test_inherit_is_not_written_into_config(self):
        viewer = self.load(ONLINE_EXPERIMENT)
        self.assertNotIn("env", viewer.raw_data)

    def test_selecting_env_records_it(self):
        viewer = self.load(ONLINE_EXPERIMENT)
        viewer.combo_env.setCurrentText("cartpole")
        self.assertEqual(viewer._get_explicit_env(), "cartpole")
        self.assertNotIn("env", viewer.raw_data)

    def test_switching_paradigm_drops_incompatible_env(self):
        viewer = self.load(ONLINE_EXPERIMENT)
        viewer.combo_env.setCurrentText("cartpole")
        self.assertEqual(viewer._get_explicit_env(), "cartpole")
        viewer.combo_paradigm.setCurrentText("offline_rl")
        self.assertEqual(viewer._get_explicit_env(), "cartpole")
        from unittest.mock import patch
        from frontend.config_model import Paradigm
        with patch.object(Paradigm, "permits_environment", return_value=False):
            viewer.combo_paradigm.setCurrentText("online_rl")
            self.assertIsNone(viewer._get_explicit_env())
            self.assertNotIn("env", viewer.raw_data)


@unittest.skipIf(not HAS_PYQT6, "PyQt6 not installed in current environment")
class TestAgentFiltering(ConfigViewerConstraintTest):
    def test_agent_is_a_dropdown_not_free_text(self):
        viewer = self.load(ONLINE_EXPERIMENT)
        self.assertIsNotNone(self.agent_widget(viewer))

    def test_online_agent_choices_exclude_offline_algorithms(self):
        viewer = self.load(ONLINE_EXPERIMENT)
        combo = self.agent_widget(viewer)
        offered = {combo.itemText(i) for i in range(combo.count())}
        self.assertIn("ppo", offered)
        self.assertFalse(offered & {"cql", "iql", "cew"})

    def test_offline_agent_choices_exclude_ppo(self):
        viewer = self.load(OFFLINE_EXPERIMENT)
        combo = self.agent_widget(viewer)
        offered = {combo.itemText(i) for i in range(combo.count())}
        self.assertIn("cql", offered)
        self.assertNotIn("ppo", offered)

    def test_off_list_agent_is_preserved_not_silently_rewritten(self):
        """An existing file with a mismatched agent must not be quietly changed."""
        viewer = self.load(ONLINE_EXPERIMENT.replace("agent: ppo", "agent: cql"))
        combo = self.agent_widget(viewer)
        self.assertEqual(combo.currentText(), "cql")
        self.assertEqual(viewer.raw_data["methods"]["ppo"]["agent"], "cql")

    def test_off_list_agent_explains_itself(self):
        from PyQt6.QtWidgets import QLabel
        viewer = self.load(ONLINE_EXPERIMENT.replace("agent: ppo", "agent: cql"))
        status = [label.text() for label in viewer.findChildren(QLabel)
                  if label.objectName() == "compatibilityStatus"]
        self.assertTrue(any("Incompatible" in text and "paradigm" in text for text in status), status)


@unittest.skipIf(not HAS_PYQT6, "PyQt6 not installed in current environment")
class TestForbiddenFields(ConfigViewerConstraintTest):
    def test_eval_episodes_enabled_for_online(self):
        viewer = self.load(ONLINE_EXPERIMENT)
        self.assertTrue(viewer.spin_eval_ep.isEnabled())

    def test_eval_episodes_disabled_for_offline(self):
        viewer = self.load(OFFLINE_EXPERIMENT)
        self.assertFalse(viewer.spin_eval_ep.isEnabled())

    def test_disabled_field_explains_why(self):
        viewer = self.load(OFFLINE_EXPERIMENT)
        self.assertIn("eval_episodes", viewer.spin_eval_ep.toolTip())

    def test_switching_paradigm_disables_the_field(self):
        viewer = self.load(ONLINE_EXPERIMENT)
        self.assertTrue(viewer.spin_eval_ep.isEnabled())
        viewer.combo_paradigm.setCurrentText("offline_rl")
        self.assertFalse(viewer.spin_eval_ep.isEnabled())


@unittest.skipIf(not HAS_PYQT6, "PyQt6 not installed in current environment")
class TestOverrides(ConfigViewerConstraintTest):
    def test_paradigm_is_emitted(self):
        viewer = self.load(ONLINE_EXPERIMENT)
        self.assertIn("paradigm=online_rl", viewer.get_overrides())

    def test_selected_env_is_emitted(self):
        viewer = self.load(ONLINE_EXPERIMENT)
        viewer.combo_env.setCurrentText("cartpole")
        self.assertIn("env=cartpole", viewer.get_overrides())

    def test_inherited_env_is_not_emitted(self):
        viewer = self.load(ONLINE_EXPERIMENT)
        self.assertFalse(any(o.startswith("env=") for o in viewer.get_overrides()))

    def test_existing_override_behaviour_is_unchanged(self):
        viewer = self.load(ONLINE_EXPERIMENT)
        overrides = viewer.get_overrides()
        self.assertIn("++experiment_id='cp_demo'", overrides)
        self.assertIn("seed=42", overrides)
        self.assertIn("total_timesteps=5000", overrides)


@unittest.skipIf(not HAS_PYQT6, "PyQt6 not installed in current environment")
class TestInheritedValuesAreVisible(unittest.TestCase):
    """A disabled field showed the widget default, which is not the value the run
    uses, and "(inherit from base)" never said what was inherited."""

    def viewer(self, group):
        path, rel = an_experiment_in(group)
        v = ConfigViewer()
        self.addCleanup(v.close)
        v.load_file(path, rel)
        return v

    def test_inherit_option_names_the_environment(self):
        self.assertEqual(self.viewer("mimic").combo_env.itemText(0), "mimic")

    def test_inherit_option_keeps_the_sentinel_in_item_data(self):
        self.assertEqual(self.viewer("mimic").combo_env.itemData(0), ConfigViewer.INHERIT)

    def test_env_selection_is_none_while_inherited(self):
        self.assertIsNone(self.viewer("mimic").env_selection())

    def test_env_selection_reports_an_explicit_choice(self):
        viewer = self.viewer("mimic")
        viewer.combo_env.setCurrentText("pyrenees")
        self.assertEqual(viewer.env_selection(), "pyrenees")

    def test_disabled_field_shows_the_value_the_base_pins(self):
        viewer = self.viewer("mimic")
        self.assertEqual(viewer.spin_eval_ep.value(), 0)

    def test_disabled_field_says_where_the_value_came_from(self):
        viewer = self.viewer("mimic")
        self.assertIn("offline_rl", viewer.spin_eval_ep.suffix())

    def test_enabled_fields_carry_no_suffix(self):
        viewer = self.viewer("cartpole")
        self.assertEqual(viewer.spin_eval_ep.suffix(), "")


@unittest.skipIf(not HAS_PYQT6, "PyQt6 not installed in current environment")
class TestTreeToolbar(unittest.TestCase):
    """The glyphs used before did not render in the theme font, and the title
    elided to "EXPERIM" with five buttons beside it."""

    def setUp(self):
        from frontend.config_tree import ConfigTreeWidget

        self.widget = ConfigTreeWidget(root_dir=Path(_SHARED_TMP.name) / "src/usr/config", mode="experiments")
        self.addCleanup(self.widget.close)

    def test_icon_buttons_have_icons_not_unrenderable_glyphs(self):
        for name in ("btn_refresh", "btn_collapse", "btn_expand", "btn_search"):
            button = getattr(self.widget, name)
            self.assertFalse(button.icon().isNull(), f"{name} has no icon")
            self.assertEqual(button.text(), "", f"{name} still carries glyph text")

    def test_every_toolbar_button_explains_itself(self):
        for name in ("btn_refresh", "btn_collapse", "btn_expand", "btn_search", "btn_new", "btn_duplicate"):
            self.assertTrue(getattr(self.widget, name).toolTip(), f"{name} has no tooltip")

    def test_toolbar_is_at_top_of_tree(self):
        first_item = self.widget.layout().itemAt(0)
        self.assertIsNotNone(first_item.widget())
        self.assertIs(first_item.widget(), self.widget.header_widget)

    def test_search_bar_hidden_by_default_and_toggled_by_magnifying_glass(self):
        # Search bar is hidden initially
        self.assertTrue(self.widget.search.isHidden())
        self.assertFalse(self.widget.btn_search.isChecked())

        # Click magnifying glass button to show search bar
        self.widget.btn_search.click()
        self.assertFalse(self.widget.search.isHidden())
        self.assertTrue(self.widget.btn_search.isChecked())

        # Click again to hide search bar
        self.widget.btn_search.click()
        self.assertTrue(self.widget.search.isHidden())
        self.assertFalse(self.widget.btn_search.isChecked())


@unittest.skipIf(not HAS_PYQT6, "PyQt6 not installed in current environment")
class TestScrollDoesNotEditFields(unittest.TestCase):
    """Qt gives these widgets WheelFocus, so scrolling a long form used to
    rewrite every field the pointer crossed."""

    def setUp(self):
        from pathlib import Path

        from PyQt6.QtCore import Qt

        self.Qt = Qt
        self.viewer = ConfigViewer()
        self.addCleanup(self.viewer.close)
        self.viewer.load_file(
            *an_experiment_in("cartpole")
        )
        self.viewer.show()
        QApplication.processEvents()

    def wheel(self, widget):
        from PyQt6.QtCore import QPoint, QPointF
        from PyQt6.QtGui import QWheelEvent

        event = QWheelEvent(
            QPointF(5, 5), QPointF(5, 5), QPoint(0, -120), QPoint(0, -120),
            self.Qt.MouseButton.NoButton, self.Qt.KeyboardModifier.NoModifier,
            self.Qt.ScrollPhase.NoScrollPhase, False,
        )
        QApplication.sendEvent(widget, event)
        return event.isAccepted()

    def test_scrolling_an_unfocused_field_leaves_it_alone(self):
        box = self.viewer.spin_timesteps
        box.clearFocus()
        QApplication.processEvents()
        before = box.value()
        self.wheel(box)
        self.assertEqual(box.value(), before)

    def test_unfocused_wheel_is_left_for_the_scroll_area(self):
        box = self.viewer.spin_timesteps
        box.clearFocus()
        QApplication.processEvents()
        self.assertFalse(self.wheel(box), "the field swallowed the wheel event")

    def test_scrolling_a_focused_field_still_edits_it(self):
        box = self.viewer.spin_timesteps
        box.setFocus(self.Qt.FocusReason.MouseFocusReason)
        QApplication.processEvents()
        if not box.hasFocus():
            self.skipTest("The headless macOS window server did not grant widget focus.")
        before = box.value()
        self.wheel(box)
        self.assertNotEqual(box.value(), before)

    def test_combo_boxes_behave_the_same(self):
        combo = self.viewer.combo_paradigm
        combo.clearFocus()
        QApplication.processEvents()
        before = combo.currentText()
        self.wheel(combo)
        self.assertEqual(combo.currentText(), before)

    def test_fields_do_not_take_focus_from_a_passing_wheel(self):
        for name in ("spin_timesteps", "spin_seed", "combo_paradigm"):
            widget = getattr(self.viewer, name)
            self.assertEqual(widget.focusPolicy(), self.Qt.FocusPolicy.StrongFocus, name)


@unittest.skipIf(not HAS_PYQT6, "PyQt6 not installed in current environment")
class TestSavePreservesHydraDirectives(unittest.TestCase):
    """yaml.safe_dump drops comments, and "# @package _global_" is a Hydra
    directive rather than one. Losing it stopped the file's keys being applied
    globally, so a single edit through the panel made the experiment invalid:
    "requires 'methods' to satisfy rule 'non_empty'". All 67 configs carry it."""

    def setUp(self):
        import tempfile
        from pathlib import Path

        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "exp.yaml"
        self.path.write_text(
            "# @package _global_\n"
            "# a second note\n"
            "defaults:\n  - mimic/_base\n\n"
            "experiment_id: demo\nseed: 42\n"
            "methods:\n  cql_dnn:\n    agent: cql\n    model: dnn\n"
        )
        self.viewer = ConfigViewer()
        self.addCleanup(self.viewer.close)
        self.viewer.load_file(self.path, "experiment/mimic/exp.yaml")

    def test_package_directive_survives_a_save(self):
        self.viewer.spin_seed.setValue(43)
        self.viewer.save_to_disk()
        self.assertTrue(self.path.read_text().startswith("# @package _global_"))

    def test_the_whole_leading_comment_block_survives(self):
        self.viewer.save_to_disk()
        self.assertIn("# a second note", self.path.read_text())

    def test_the_edit_is_still_written(self):
        import yaml

        self.viewer.spin_seed.setValue(43)
        self.viewer.save_to_disk()
        self.assertEqual(yaml.safe_load(self.path.read_text())["seed"], 43)

    def test_methods_survive_a_save(self):
        import yaml

        self.viewer.save_to_disk()
        self.assertIn("cql_dnn", yaml.safe_load(self.path.read_text())["methods"])

    def test_repeated_saves_do_not_stack_the_preamble(self):
        for _ in range(3):
            self.viewer.save_to_disk()
        self.assertEqual(self.path.read_text().count("# @package _global_"), 1)

    def test_a_file_without_a_preamble_is_unharmed(self):
        from pathlib import Path

        plain = Path(self.tmp.name) / "plain.yaml"
        plain.write_text("seed: 1\n")
        viewer = ConfigViewer()
        self.addCleanup(viewer.close)
        viewer.load_file(plain, "experiment/mimic/plain.yaml")
        viewer.save_to_disk()
        self.assertFalse(plain.read_text().startswith("#"))


if __name__ == "__main__":
    unittest.main()

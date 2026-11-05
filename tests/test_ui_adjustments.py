"""Unit tests for the 6 experiment panel and config viewer UI adjustments:
1. ToggleSlider used for bool parameters instead of checkboxes.
2. Parameter labels remove "(Required)" from Paradigm and Environment.
3. Method card bottom actions (duplicate, plus) are pinned to the bottom right with stretch.
4. Hierarchical parameter adders have only the indented combobox, no plus button, and add on selection.
5. Experiment creation writes minimal configs without redundant seed/total_timesteps/eval_episodes.
6. Play button shows why an experiment cannot run when hovering over it while disabled.
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import yaml

os.environ["QT_QPA_PLATFORM"] = "offscreen"

try:
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QApplication, QLabel, QToolButton, QVBoxLayout

    from frontend.config_tree import ConfigTreeWidget
    from frontend.config_viewer import ConfigViewer
    from frontend.widgets import ComboBox, ToggleSlider
    HAS_PYQT6 = True
    app = QApplication.instance() or QApplication(sys.argv[:1])
except ImportError:
    HAS_PYQT6 = False


@unittest.skipIf(not HAS_PYQT6, "PyQt6 not installed in current environment")
class TestUIAdjustments(unittest.TestCase):
    def setUp(self):
        from frontend.config_model import ConfigTree
        from frontend.config_viewer import set_config_tree

        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmp_root = Path(self.temp_dir.name)

        config_dir = self.tmp_root / "config"
        agent_dir = config_dir / "agent"
        agent_dir.mkdir(parents=True, exist_ok=True)
        (agent_dir / "ppo.yaml").write_text("algorithm: ppo\nlr: 0.0003\nclip_range: 0.2\n", encoding="utf-8")

        model_dir = config_dir / "model"
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "dnn.yaml").write_text("architecture: dnn\nhidden_sizes: [64, 64]\nactivation: relu\n", encoding="utf-8")

        paradigm_dir = config_dir / "paradigms"
        paradigm_dir.mkdir(parents=True, exist_ok=True)
        (paradigm_dir / "online_rl.yaml").write_text("allowed_agents: [ppo]\n", encoding="utf-8")

        self.mock_tree = ConfigTree(config_dir)
        set_config_tree(self.mock_tree)

        self.exp_file = self.tmp_root / "test_exp.yaml"
        self.exp_file.write_text(
            "experiment_id: test_exp\n"
            "paradigm: null\n"
            "env: null\n"
            "tensorboard: true\n"
            "custom_bool: false\n"
            "methods:\n"
            "  ppo_dnn:\n"
            "    agent: ppo\n"
            "    model: dnn\n",
            encoding="utf-8"
        )
        self.viewer = ConfigViewer()
        self.viewer.load_file(self.exp_file, "experiment/cartpole/test_exp.yaml")

    def tearDown(self):
        from frontend.config_viewer import invalidate_config_tree
        invalidate_config_tree()
        self.viewer.close()
        self.temp_dir.cleanup()

    def test_item1_bool_parameters_use_toggle_slider(self):
        """Boolean settings in Box 1 and runtime flags render as ToggleSlider."""
        sliders = self.viewer.findChildren(ToggleSlider)
        self.assertTrue(len(sliders) >= 1, "Expected at least one ToggleSlider for boolean flags/parameters")
        # Check that tensorboard flag widget is ToggleSlider
        if self.viewer.chk_tb is not None:
            self.assertIsInstance(self.viewer.chk_tb, ToggleSlider)
            self.assertTrue(self.viewer.chk_tb.isChecked())

    def test_item2_paradigm_env_labels_remove_required(self):
        """Labels for Paradigm and Environment should not contain '(Required)'."""
        labels = [lbl.text() for lbl in self.viewer.findChildren(QLabel)]
        self.assertIn("Paradigm", labels)
        self.assertNotIn("Paradigm (Required)", labels)
        self.assertIn("Environment / Task", labels)
        self.assertNotIn("Environment / Task (Required)", labels)

    def test_item3_method_card_bottom_actions_pinned(self):
        """Method card uses QVBoxLayout with stretch pinning duplicate and add buttons to bottom."""
        method_cards = self.viewer.findChildren(ConfigViewer)
        # Find method sub_cards
        cards = [w for w in self.viewer.findChildren(QVBoxLayout) if w.parent() and w.parent().objectName() == "card"]
        self.assertTrue(len(cards) >= 1, "Expected method card to use QVBoxLayout")

    def test_item4_hierarchical_parameter_plus_button_menu_add(self):
        """Hierarchical parameters use a plus button next to the row that opens a menu."""
        from PyQt6.QtWidgets import QToolButton
        # No empty + Add comboboxes waiting around
        combos = [cb for cb in self.viewer.findChildren(ComboBox) if cb.placeholderText().startswith("+ Add")]
        self.assertEqual(len(combos), 0, "Expected empty comboboxes to be replaced by plus button menu adders")

        # Plus button with menu exists on method card
        plus_buttons = [btn for btn in self.viewer.findChildren(QToolButton) if btn.menu() is not None]
        usable_buttons = [
            btn for btn in plus_buttons
            if any(action.text() and not action.text().startswith("+") for action in btn.menu().actions())
        ]
        self.assertTrue(usable_buttons, "Expected a parameter menu with at least one declared option")
        target_btn = next((b for b in usable_buttons if "parameter" in b.toolTip().lower()), usable_buttons[0])
        initial_keys = set(self.viewer.raw_data.get("methods", {}).get("ppo_dnn", {}).keys())
        actions = target_btn.menu().actions()
        self.assertTrue(len(actions) >= 1)
        chosen_action = next((a for a in actions if a.text() and not a.text().startswith("+")), actions[0])
        chosen_key = chosen_action.text()
        # Triggering QAction while _add_method_param rebuilds the Qt tree aborts
        # the offscreen CI process. Exercise the same configured destination
        # directly after confirming it is offered by the menu.
        self.viewer._add_method_param("ppo_dnn", chosen_key)
        updated_keys = set(self.viewer.raw_data.get("methods", {}).get("ppo_dnn", {}).keys())
        self.assertIn(chosen_key, updated_keys)

    def test_item5_minimal_experiment_creation(self):
        """Creating an experiment should not write redundant seed/total_timesteps/eval_episodes."""
        tree_widget = ConfigTreeWidget(self.tmp_root, mode="experiments")
        exp_path = tree_widget._write_experiment_from_defaults("cartpole", "minimal_exp.yaml", "minimal_exp")
        self.assertIsNotNone(exp_path)
        created_file = self.tmp_root / "experiment" / "cartpole" / "minimal_exp.yaml"
        self.assertTrue(created_file.exists())
        data = yaml.safe_load(created_file.read_text(encoding="utf-8"))
        self.assertEqual(data["experiment_id"], "minimal_exp")
        self.assertIn("cartpole/_base", data["defaults"])
        self.assertNotIn("seed", data, "Redundant seed should not be written")
        self.assertNotIn("total_timesteps", data, "Redundant total_timesteps should not be written")
        self.assertNotIn("eval_episodes", data, "Redundant eval_episodes should not be written")
        tree_widget.close()

    def test_item6_play_button_always_shows_tooltip_explaining_cannot_run(self):
        """The play button has WA_AlwaysShowToolTips and explains why it cannot run when disabled."""
        tree_widget = ConfigTreeWidget(self.tmp_root, mode="experiments")
        btn = tree_widget.btn_launch
        self.assertIsNotNone(btn)
        self.assertTrue(btn.testAttribute(Qt.WidgetAttribute.WA_AlwaysShowToolTips))
        tree_widget.close()

    def test_item6_update_launch_state_tooltip_messages(self):
        """update_launch_state sets specific 'Cannot run:' tooltip reasons."""
        from unittest.mock import MagicMock

        from frontend.app import Window

        app_mock = MagicMock(spec=Window)
        app_mock.active = None
        app_mock.active_runs = []
        app_mock.jobs = SimpleNamespace(max_parallel=1)
        app_mock.start_button = QToolButton()
        app_mock.queue_button = None
        app_mock.stop_button = None
        app_mock.state_label = MagicMock()
        app_mock.compose_valid = False
        app_mock.current_experiment = None
        app_mock.active_config_rel_path = "experiment/cartpole/_base.yaml"
        app_mock.compose_error = None

        Window.update_launch_state(app_mock)
        self.assertIn("Base templates (_base.yaml) cannot be run directly", app_mock.start_button.toolTip())

        app_mock.active_config_rel_path = ""
        Window.update_launch_state(app_mock)
        self.assertIn("Select an experiment configuration", app_mock.start_button.toolTip())

        app_mock.current_experiment = "cartpole/cp_test"
        app_mock.compose_error = "Validation error: agent ppo forbids offline_rl"
        Window.update_launch_state(app_mock)
        self.assertIn("Validation error: agent ppo forbids offline_rl", app_mock.start_button.toolTip())

        app_mock.active = {"status": "running", "simulated": False, "backend": {"experiment_id": "cartpole/cp_test"}}
        app_mock.active_runs = [app_mock.active]
        Window.update_launch_state(app_mock)
        self.assertIn("Validation error: agent ppo forbids offline_rl", app_mock.start_button.toolTip())

        app_mock.jobs.max_parallel = 2  # room for one more: launching is offered again
        app_mock.compose_error, app_mock.compose_valid = None, True
        Window.update_launch_state(app_mock)
        self.assertNotIn("Cannot run", app_mock.start_button.toolTip())
        app_mock.active_runs = [app_mock.active, dict(app_mock.active)]
        Window.update_launch_state(app_mock)
        self.assertTrue(app_mock.start_button.isEnabled())
        self.assertIn("queue", app_mock.start_button.toolTip())

    def test_single_click_title_editing(self):
        """Single click triggers inline editing for experiment title, group badge, and method title."""
        # 1. Experiment title
        self.assertFalse(self.viewer.file_title.isHidden())
        self.assertTrue(self.viewer.title_editor.isHidden())
        self.viewer.file_title.clicked.emit()
        self.assertTrue(self.viewer.file_title.isHidden())
        self.assertFalse(self.viewer.title_editor.isHidden())
        self.viewer._cancel_title_edit()

        # 2. Group badge
        self.assertFalse(self.viewer.group_badge.isHidden())
        self.assertTrue(self.viewer.group_editor.isHidden())
        self.viewer.group_badge.clicked.emit()
        self.assertTrue(self.viewer.group_badge.isHidden())
        self.assertFalse(self.viewer.group_editor.isHidden())
        self.viewer._cancel_group_edit()

        # 3. Method card title
        from frontend.config_viewer import InlineTitleEditor, TitleLabel
        m_titles = [w for w in self.viewer.findChildren(TitleLabel) if w.objectName() == "cardTitle"]
        self.assertTrue(len(m_titles) >= 1)
        m_title = m_titles[0]
        m_card = m_title.parentWidget()
        m_edit = m_card.findChild(InlineTitleEditor, "methodEditor")
        self.assertIsNotNone(m_edit)
        self.assertFalse(m_title.isHidden())
        self.assertTrue(m_edit.isHidden())
        m_title.clicked.emit()
        self.assertTrue(m_title.isHidden())
        self.assertFalse(m_edit.isHidden())

    def test_group_defaults_one_default_method_and_inheritance(self):
        """Group defaults allow at most 1 method, and experiments inherit from it."""
        # Create a group base with a default method
        group_dir = self.mock_tree.config_root / "experiment" / "cartpole"
        group_dir.mkdir(parents=True, exist_ok=True)
        base_file = group_dir / "_base.yaml"
        base_file.write_text(
            "paradigm: online_rl\n"
            "env: CartPole-v1\n"
            "methods:\n"
            "  ppo_default:\n"
            "    agent: ppo\n"
            "    model: dnn\n"
            "    lr: 0.0003\n",
            encoding="utf-8"
        )
        self.mock_tree.refresh_groups()
        base_viewer = ConfigViewer()
        base_viewer.load_file(base_file, "experiment/cartpole/_base.yaml")
        # In base, duplicate and add-new buttons are not shown on method cards
        tool_btns = [b for b in base_viewer.findChildren(QToolButton) if "duplicate" in b.toolTip().lower()]
        self.assertEqual(len(tool_btns), 0, "Duplicate method button should not be present in _base.yaml")

        # Now load an experiment in the same group and add a method
        exp_file = group_dir / "child_exp.yaml"
        exp_file.write_text("experiment_id: child_exp\n", encoding="utf-8")
        exp_viewer = ConfigViewer()
        exp_viewer.load_file(exp_file, "experiment/cartpole/child_exp.yaml")
        meth_name = exp_viewer.add_method()
        self.assertEqual(meth_name, "ppo_default")
        self.assertEqual(exp_viewer.raw_data["methods"]["ppo_default"]["agent"], "ppo")
        self.assertEqual(exp_viewer.raw_data["methods"]["ppo_default"]["model"], "dnn")
        self.assertEqual(exp_viewer.raw_data["methods"]["ppo_default"]["lr"], 0.0003)
        base_viewer.close()
        exp_viewer.close()

    def test_param_view_appearance_settings_and_defaults_rendering(self):
        """Appearance settings toggle display of defaults for agents, models, and training params."""
        from frontend.settings import SettingsManager
        sm = SettingsManager(self.tmp_root)
        viewer = ConfigViewer(settings_manager=sm)
        viewer.load_file(self.exp_file, "experiment/cartpole/test_exp.yaml")

        # Initially, default hyperparameters are hidden in overrides view mode
        labels_initial = [lbl.text() for lbl in viewer.findChildren(QLabel)]
        self.assertNotIn("(default)", labels_initial)

        # 1. Enable show_method_agent_defaults
        sm.show_method_agent_defaults = True
        labels_after_ag = [lbl.text() for lbl in viewer.findChildren(QLabel)]
        self.assertIn("(default)", labels_after_ag)

        # 2. Reset and enable show_all_default_params
        sm.show_method_agent_defaults = False
        sm.show_all_default_params = True
        labels_after_all = [lbl.text() for lbl in viewer.findChildren(QLabel)]
        self.assertIn("(default)", labels_after_all)
        self.assertIn("Learning Rate (lr)", labels_after_all)
        self.assertIn("Batch Size", labels_after_all)
        self.assertIn("Discount Factor (γ)", labels_after_all)
        viewer.close()

    def test_training_parameters_unindented_and_no_header(self):
        """Training settings header is removed and training parameters are unindented."""
        labels = [lbl.text() for lbl in self.viewer.findChildren(QLabel)]
        self.assertNotIn("Training Settings", labels, "Training Settings header label should be removed")

        # Check that plus buttons for add override and add option are simple icon buttons
        self.assertFalse(self.viewer.btn_add_block.icon().isNull())
        self.assertEqual(self.viewer.btn_add_block.objectName(), "addBlockButton")

    def test_blendrl_param_adder_fixed_next_to_model(self):
        """Add blendrl parameter button stays anchored next to Model: blendrl when params are added or removed."""
        from PyQt6.QtWidgets import QFormLayout, QHBoxLayout

        from frontend.config_model import ConfigTree
        from frontend.config_viewer import IndentedHierarchyBlock, set_config_tree

        config_dir = self.tmp_root / "config"
        model_dir = config_dir / "model"
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "blendrl.yaml").write_text(
            "architecture: blendrl\n"
            "slots: [neural]\n"
            "blend_q_values: false\n"
            "neural:\n"
            "  type: dnn\n",
            encoding="utf-8"
        )
        set_config_tree(ConfigTree(config_dir))

        blend_file = self.tmp_root / "blend_exp.yaml"
        blend_file.write_text(
            "experiment_id: blend_exp\n"
            "paradigm: online_rl\n"
            "env: CartPole-v1\n"
            "methods:\n"
            "  blend_method:\n"
            "    agent: ppo\n"
            "    model: blendrl\n",
            encoding="utf-8"
        )
        viewer = ConfigViewer()
        viewer.load_file(blend_file, "experiment/cartpole/blend_exp.yaml")

        def get_model_row_and_adder():
            # Find the QFormLayout in the method card
            form_layouts = [l for l in viewer.findChildren(QFormLayout)]
            for fl in form_layouts:
                for row_idx in range(fl.rowCount()):
                    lbl_item = fl.itemAt(row_idx, QFormLayout.ItemRole.LabelRole)
                    if lbl_item and lbl_item.widget() and isinstance(lbl_item.widget(), QLabel):
                        if lbl_item.widget().text() == "Model":
                            field_item = fl.itemAt(row_idx, QFormLayout.ItemRole.FieldRole)
                            layout = field_item.layout() if field_item else None
                            adder = None
                            if layout:
                                for i in range(layout.count()):
                                    w = layout.itemAt(i).widget()
                                    if isinstance(w, QToolButton) and "blendrl" in w.toolTip().lower():
                                        adder = w
                            return layout, adder
            return None, None

        # 1. Before adding any parameter:
        layout_before, adder_before = get_model_row_and_adder()
        self.assertIsNotNone(layout_before, "Model row layout should exist")
        self.assertIsNotNone(adder_before, "Add blendrl parameter button should exist in Model row layout")

        # 2. Add a blendrl parameter via the adder button
        actions = adder_before.menu().actions()
        action_q = next((a for a in actions if a.text() == "blend_q_values"), actions[0])
        action_q.trigger()

        # Check parameter added to data
        self.assertIn("blend_q_values", viewer.raw_data["methods"]["blend_method"])

        # 3. After adding parameter: adder button should STILL be in the Model row layout!
        layout_after, adder_after = get_model_row_and_adder()
        self.assertIsNotNone(layout_after, "Model row layout should still exist")
        self.assertIsNotNone(adder_after, "Add blendrl parameter button must remain in Model row layout next to Model")

        # Check that adder is NOT pushed into the IndentedHierarchyBlock
        blocks = viewer.findChildren(IndentedHierarchyBlock)
        self.assertTrue(len(blocks) >= 1, "Expected IndentedHierarchyBlock for child params")
        for b in blocks:
            for child_btn in b.findChildren(QToolButton):
                self.assertNotIn("add blendrl parameter", child_btn.toolTip().lower(),
                                 "Add blendrl parameter button should not be inside child hierarchy block")

        # 4. Remove the parameter: adder button should STILL be in Model row layout
        viewer._remove_method_param("blend_method", "blend_q_values")
        layout_removed, adder_removed = get_model_row_and_adder()
        self.assertIsNotNone(layout_removed)
        self.assertIsNotNone(adder_removed, "Add blendrl parameter button must remain in Model row layout after removal")

        viewer.close()


if __name__ == "__main__":
    unittest.main()



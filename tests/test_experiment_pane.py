"""Unit tests for the overhauled Experiment Pane in Theta-IDE:
- ConfigTreeWidget mirroring in/config/
- Boxed ConfigViewer with real-time updates and saving
- On-demand Hydra YAML preview (hidden by default)
- Experiment creation and duplication
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

os.environ["QT_QPA_PLATFORM"] = "offscreen"

try:
    from PyQt6 import QtWebEngineWidgets
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QApplication, QLabel, QTreeWidget, QWidget
    app = QApplication.instance() or QApplication(sys.argv[:1])
    from frontend.app import Window
    from frontend.components_panel import ComponentsPanel
    from frontend.config_tree import ConfigTreeWidget
    from frontend.config_viewer import ConfigBox, ConfigViewer
    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False


@unittest.skipIf(not HAS_PYQT6, "PyQt6 not installed in current environment")
class TestConfigTree(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        # Create mock in/config structure
        (self.root / "experiment" / "cartpole").mkdir(parents=True)
        (self.root / "agent").mkdir(parents=True)
        (self.root / "env").mkdir(parents=True)

        (self.root / "experiment" / "cartpole" / "exp1.yaml").write_text(
            "experiment_id: exp1\nseed: 42\ntotal_timesteps: 5000\n", encoding="utf-8"
        )
        (self.root / "agent" / "ppo.yaml").write_text(
            "lr: 0.0003\nbatch_size: 64\n", encoding="utf-8"
        )
        (self.root / "config.yaml").write_text("paradigm: online_rl\n", encoding="utf-8")

        self.tree_widget = ConfigTreeWidget(root_dir=self.root)

    def tearDown(self):
        self.tree_widget.close()
        self.temp_dir.cleanup()

    def test_selected_group_scopes_expansion_and_delete_key(self):
        from PyQt6.QtTest import QTest

        (self.root / "experiment" / "cartpole" / "nested" / "deep").mkdir(parents=True)
        (self.root / "experiment" / "other" / "nested").mkdir(parents=True)
        widget = ConfigTreeWidget(root_dir=self.root, mode="experiments")
        self.addCleanup(widget.close)
        tree = widget.tree
        groups = {tree.topLevelItem(i).text(0): tree.topLevelItem(i)
                  for i in range(tree.topLevelItemCount())}
        group, other = groups["cartpole"], groups["other"]
        tree.setCurrentItem(group)
        widget.expand_all()
        self.assertTrue(group.isExpanded())
        self.assertTrue(group.child(0).isExpanded())
        self.assertFalse(other.child(0).isExpanded())
        other.setExpanded(True)
        widget.collapse_all()
        self.assertFalse(group.isExpanded())
        self.assertTrue(other.isExpanded())
        with patch.object(widget, "prompt_delete_dir") as delete:
            QTest.keyClick(tree, Qt.Key.Key_Delete)
            delete.assert_called_once_with("experiment/cartpole")
        tree.clearSelection()
        widget.expand_all()
        self.assertTrue(other.child(0).isExpanded())

    def test_no_footer_text_at_bottom_of_tree(self):
        tree_exp = ConfigTreeWidget(root_dir=self.root, mode="experiments")
        tree_comp = ConfigTreeWidget(root_dir=self.root, mode="components")
        self.assertIsNone(tree_exp.footer)
        self.assertIsNone(tree_comp.footer)
        from PyQt6.QtWidgets import QLabel
        labels_exp = tree_exp.findChildren(QLabel)
        labels_comp = tree_comp.findChildren(QLabel)
        self.assertFalse(any("in/config" in lbl.text() for lbl in labels_exp))
        self.assertFalse(any("in/config" in lbl.text() for lbl in labels_comp))
        tree_exp.close()
        tree_comp.close()

    def test_tree_populates_directories_and_files(self):
        # Top-level should have experiment, agent, env, and config.yaml
        top_count = self.tree_widget.tree.topLevelItemCount()
        self.assertGreaterEqual(top_count, 4)

        top_texts = [self.tree_widget.tree.topLevelItem(i).text(0) for i in range(top_count)]
        self.assertTrue(any("experiment" in t for t in top_texts))
        self.assertTrue(any("agent" in t for t in top_texts))
        self.assertTrue(any("config" in t for t in top_texts))

    def test_experiment_expanded_by_default(self):
        for i in range(self.tree_widget.tree.topLevelItemCount()):
            item = self.tree_widget.tree.topLevelItem(i)
            if "experiment" in item.text(0):
                self.assertTrue(item.isExpanded())

    def test_select_file_emits_signal(self):
        selected = []
        self.tree_widget.file_selected.connect(lambda path, rel: selected.append((path, rel)))

        ok = self.tree_widget.select_file("experiment/cartpole/exp1.yaml")
        self.assertTrue(ok)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0][1], "experiment/cartpole/exp1.yaml")

    def test_click_selects_and_double_click_opens_a_new_tab(self):
        selected, opened = [], []
        self.tree_widget.file_selected.connect(lambda path, rel: selected.append(rel))
        self.tree_widget.file_opened.connect(lambda path, rel: opened.append(rel))
        self.tree_widget.select_file("experiment/cartpole/exp1.yaml", emit_signal=False)
        item = self.tree_widget.tree.currentItem()
        self.tree_widget._on_item_clicked(item, 0)
        self.assertEqual(selected, ["experiment/cartpole/exp1.yaml"])
        self.tree_widget._on_item_double_clicked(item, 0)
        self.tree_widget._on_item_double_clicked(item, 0)
        self.assertEqual(opened, ["experiment/cartpole/exp1.yaml"] * 2)

    def test_filter_tree(self):
        self.tree_widget.filter_tree("exp1")
        # exp1 item should be visible
        for i in range(self.tree_widget.tree.topLevelItemCount()):
            item = self.tree_widget.tree.topLevelItem(i)
            if "agent" in item.text(0):
                self.assertTrue(item.isHidden())

        self.tree_widget.filter_tree("")
        for i in range(self.tree_widget.tree.topLevelItemCount()):
            item = self.tree_widget.tree.topLevelItem(i)
            self.assertFalse(item.isHidden())

    def test_experiments_mode_shows_groups_directly_without_experiment_folder(self):
        exp_tree = ConfigTreeWidget(root_dir=self.root, mode="experiments")
        try:
            top_count = exp_tree.tree.topLevelItemCount()
            self.assertEqual(top_count, 1)
            item = exp_tree.tree.topLevelItem(0)
            self.assertEqual(item.text(0), "cartpole")
            top_texts = [exp_tree.tree.topLevelItem(i).text(0) for i in range(top_count)]
            self.assertFalse(any(t == "experiment" for t in top_texts))
            self.assertFalse(any("agent" in t for t in top_texts))
            self.assertFalse(any("env" in t for t in top_texts))
            self.assertFalse(any(t == "config" for t in top_texts))
        finally:
            exp_tree.close()

    def test_base_yaml_displayed_as_group_defaults_with_gear_icon(self):
        (self.root / "experiment" / "cartpole" / "_base.yaml").write_text("paradigm: online_rl\n", encoding="utf-8")
        exp_tree = ConfigTreeWidget(root_dir=self.root, mode="experiments")
        try:
            cartpole_item = exp_tree.tree.topLevelItem(0)
            self.assertEqual(cartpole_item.text(0), "cartpole")
            child_texts = [cartpole_item.child(i).text(0) for i in range(cartpole_item.childCount())]
            self.assertIn("group defaults", child_texts)
            self.assertFalse(any("_base" in t for t in child_texts))

            defaults_item = next(
                cartpole_item.child(i)
                for i in range(cartpole_item.childCount())
                if cartpole_item.child(i).text(0) == "group defaults"
            )
            self.assertFalse(defaults_item.icon(0).isNull(), "group defaults must have a gear icon")
        finally:
            exp_tree.close()

    def test_components_mode_excludes_experiment_directory(self):
        comp_tree = ConfigTreeWidget(root_dir=self.root, mode="components")
        try:
            top_count = comp_tree.tree.topLevelItemCount()
            top_texts = [comp_tree.tree.topLevelItem(i).text(0) for i in range(top_count)]
            self.assertTrue(any(t in ("Algorithms", "agent") for t in top_texts))
            self.assertTrue(any(t in ("Environments", "env") for t in top_texts))
            self.assertTrue(any("System" in t or "config" in t.lower() for t in top_texts))
            self.assertFalse(any("experiment" in t for t in top_texts))
        finally:
            comp_tree.close()

    def test_yaml_files_have_no_icon_and_no_yaml_extension_in_tree(self):
        """YAML files must not display the .yaml extension and must not have a file icon."""
        def find_file_items(parent):
            items = []
            count = parent.topLevelItemCount() if isinstance(parent, QTreeWidget) else parent.childCount()
            for i in range(count):
                child = parent.topLevelItem(i) if isinstance(parent, QTreeWidget) else parent.child(i)
                data = child.data(0, Qt.ItemDataRole.UserRole)
                if data and data.get("type") == "file":
                    items.append(child)
                items.extend(find_file_items(child))
            return items

        file_items = find_file_items(self.tree_widget.tree)
        self.assertGreater(len(file_items), 0)
        for item in file_items:
            # No .yaml or .yml extension in displayed text
            self.assertFalse(item.text(0).endswith(".yaml"))
            self.assertFalse(item.text(0).endswith(".yml"))
            # No icon / emoji next to yaml files
            self.assertTrue(item.icon(0).isNull())

    def test_directory_nodes_retain_folder_icons(self):
        """Directory nodes in the tree should still display folder icons."""
        for i in range(self.tree_widget.tree.topLevelItemCount()):
            item = self.tree_widget.tree.topLevelItem(i)
            data = item.data(0, Qt.ItemDataRole.UserRole)
            if data and data.get("type") == "dir":
                self.assertFalse(item.icon(0).isNull())

    def test_collapse_and_expand_all(self):
        exp_tree = ConfigTreeWidget(root_dir=self.root, mode="experiments")
        try:
            # Initially, experiment is expanded
            self.assertTrue(exp_tree.tree.topLevelItem(0).isExpanded())

            # Click collapse all
            exp_tree.btn_collapse.click()
            self.assertFalse(exp_tree.tree.topLevelItem(0).isExpanded())

            # Click expand all
            exp_tree.btn_expand.click()
            self.assertTrue(exp_tree.tree.topLevelItem(0).isExpanded())
        finally:
            exp_tree.close()

    def test_populate_preserves_expanded_folders_on_install_and_uninstall(self):
        """Folders in the tree do not collapse when new components/configs are installed or uninstalled."""
        comp_tree = ConfigTreeWidget(root_dir=self.root, mode="components")
        try:
            # Locate agent folder item and expand it
            agent_item = None
            for i in range(comp_tree.tree.topLevelItemCount()):
                item = comp_tree.tree.topLevelItem(i)
                if item.text(0) in ("Algorithms", "agent"):
                    agent_item = item
                    item.setExpanded(True)
                    break
            self.assertIsNotNone(agent_item)
            self.assertTrue(agent_item.isExpanded())

            # Simulate install into agent folder
            (self.root / "agent" / "new_algo.yaml").write_text("algo: new\n", encoding="utf-8")
            comp_tree.populate(ensure_expanded="agent")

            # Check that agent folder did NOT collapse
            found_agent = None
            for i in range(comp_tree.tree.topLevelItemCount()):
                item = comp_tree.tree.topLevelItem(i)
                if item.text(0) in ("Algorithms", "agent"):
                    found_agent = item
                    break
            self.assertIsNotNone(found_agent)
            self.assertTrue(found_agent.isExpanded(), "Folder must not collapse when something is installed into it")

            # Simulate uninstall from agent folder
            (self.root / "agent" / "new_algo.yaml").unlink()
            comp_tree.populate(ensure_expanded="agent")

            # Check that agent folder did NOT collapse
            found_agent_after = None
            for i in range(comp_tree.tree.topLevelItemCount()):
                item = comp_tree.tree.topLevelItem(i)
                if item.text(0) in ("Algorithms", "agent"):
                    found_agent_after = item
                    break
            self.assertIsNotNone(found_agent_after)
            self.assertTrue(found_agent_after.isExpanded(), "Folder must not collapse when something is uninstalled from it")
        finally:
            comp_tree.close()

    def test_experiments_mode_header_buttons(self):
        from PyQt6.QtWidgets import QToolButton
        exp_tree = ConfigTreeWidget(root_dir=self.root, mode="experiments")
        try:
            self.assertIsInstance(exp_tree.btn_launch, QToolButton)
            self.assertIsNone(exp_tree.btn_save)
            self.assertIsInstance(exp_tree.btn_export, QToolButton)
            self.assertIsInstance(exp_tree.btn_toggle_yaml, QToolButton)
            self.assertEqual(exp_tree.btn_launch.property("svg_icon"), "launch")
            self.assertEqual(exp_tree.btn_export.property("svg_icon"), "export")
            self.assertEqual(exp_tree.btn_toggle_yaml.property("svg_icon"), "raw_yaml")
            self.assertEqual(exp_tree.btn_launch.size().width(), 30)
            self.assertEqual(exp_tree.btn_launch.size().height(), 30)
            self.assertEqual(exp_tree.btn_export.size().width(), 30)
            self.assertEqual(exp_tree.btn_export.size().height(), 30)
            self.assertEqual(exp_tree.btn_toggle_yaml.size().width(), 30)
            self.assertEqual(exp_tree.btn_toggle_yaml.size().height(), 30)
            self.assertFalse(exp_tree.btn_launch.isEnabled())
            self.assertTrue(exp_tree.btn_toggle_yaml.isCheckable())
            self.assertIsNone(exp_tree.btn_hub)
            self.assertIsNone(exp_tree.btn_toggle_raw)

            header_widgets = [
                exp_tree.header_layout.itemAt(i).widget()
                for i in range(exp_tree.header_layout.count())
                if exp_tree.header_layout.itemAt(i).widget() is not None
            ]
            self.assertIn(exp_tree.btn_new, header_widgets)
            self.assertIn(exp_tree.btn_duplicate, header_widgets)
            self.assertIn(exp_tree.btn_launch, header_widgets)
            self.assertIn(exp_tree.btn_export, header_widgets)
            self.assertIn(exp_tree.btn_toggle_yaml, header_widgets)
            self.assertNotIn(exp_tree.btn_save, header_widgets)
        finally:
            exp_tree.close()

    def test_hyperparameter_sweepers_folder_renamed(self):
        from PyQt6.QtWidgets import QTreeWidgetItem
        (self.root / "hydra" / "sweeper").mkdir(parents=True)
        (self.root / "hydra" / "sweeper" / "optuna.yaml").write_text("n_trials: 10\n", encoding="utf-8")
        tree = ConfigTreeWidget(root_dir=self.root, mode="components")
        try:
            sweeper_items = []
            def find_sweepers(parent):
                count = parent.childCount() if isinstance(parent, QTreeWidgetItem) else parent.topLevelItemCount()
                for i in range(count):
                    child = parent.child(i) if isinstance(parent, QTreeWidgetItem) else parent.topLevelItem(i)
                    if child.text(0) == "Hyperparameter Sweepers":
                        sweeper_items.append(child)
                    find_sweepers(child)
            find_sweepers(tree.tree)
            self.assertEqual(len(sweeper_items), 1)
        finally:
            tree.close()



@unittest.skipIf(not HAS_PYQT6, "PyQt6 not installed in current environment")
class TestConfigViewer(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.file_path = Path(self.temp_dir.name) / "test_exp.yaml"
        self.file_path.write_text(
            "experiment_id: test_exp\n"
            "paradigm: online_rl\n"
            "seed: 123\n"
            "total_timesteps: 25000\n"
            "methods:\n"
            "  ppo:\n"
            "    agent: ppo\n"
            "    model: dnn\n"
            "    lr: 0.0005\n"
            "    batch_size: 128\n"
            "    gamma: 0.98\n",
            encoding="utf-8"
        )
        self.viewer = ConfigViewer()
        self.viewer.auto_save = True
        self.viewer.load_file(self.file_path, "experiment/test_exp.yaml")

    def tearDown(self):
        self.viewer.close()
        self.temp_dir.cleanup()

    def test_loads_experiment_values_into_boxes(self):
        self.assertEqual(self.viewer.txt_exp_id.text(), "test_exp")
        self.assertEqual(self.viewer.spin_seed.value(), 123)
        self.assertEqual(self.viewer.spin_timesteps.value(), 25000)

        # Boxes should be created as styled cards
        cards = self.viewer.findChildren(ConfigBox)
        self.assertGreaterEqual(len(cards), 2)

    def test_edits_mark_dirty_and_remember_values(self):
        events = []
        self.viewer.config_changed.connect(lambda: events.append(True))

        self.assertFalse(self.viewer.is_dirty)
        self.assertEqual(self.viewer.dirty_status.text(), "")
        self.viewer.spin_seed.setValue(999)

        # Edits immediately auto-save to disk without showing dirty status
        self.assertFalse(self.viewer.is_dirty)
        self.assertEqual(self.viewer.dirty_status.text(), "")
        self.assertEqual(len(events), 1)
        self.assertEqual(self.viewer.raw_data["seed"], 999)

        # Verify disk contents are updated immediately
        disk_data = yaml.safe_load(self.file_path.read_text(encoding="utf-8"))
        self.assertEqual(disk_data["seed"], 999)

    def test_save_to_disk(self):
        self.viewer.spin_timesteps.setValue(50000)
        saved = self.viewer.save_to_disk()
        self.assertTrue(saved)
        self.assertFalse(self.viewer.is_dirty)
        self.assertEqual(self.viewer.dirty_status.text(), "")

        # Verify disk contents
        data = yaml.safe_load(self.file_path.read_text(encoding="utf-8"))
        self.assertEqual(data["total_timesteps"], 50000)

    def test_get_overrides(self):
        overrides = self.viewer.get_overrides()
        self.assertIn("++experiment_id='test_exp'", overrides)
        self.assertIn("seed=123", overrides)
        self.assertIn("total_timesteps=25000", overrides)

    def test_boxes_do_not_have_titles_or_subtitles(self):
        cards = self.viewer.findChildren(ConfigBox)
        self.assertGreaterEqual(len(cards), 2)
        for card in cards:
            self.assertIsNone(card.title_label)
            self.assertIsNone(card.subtitle_label)

    def test_config_viewer_header_has_no_save_button(self):
        from PyQt6.QtWidgets import QPushButton
        buttons = self.viewer.header_bar.findChildren(QPushButton)
        self.assertEqual(len(buttons), 0)

    def test_title_displayed_without_yaml(self):
        self.assertEqual(self.viewer.file_title.text(), "test_exp")

    def test_redundant_experiment_id_not_in_box_1_form(self):
        cards = self.viewer.findChildren(ConfigBox)
        self.assertGreaterEqual(len(cards), 1)
        box_1 = cards[0]
        # Inspect rows in form layouts inside box 1
        labels = [lbl.text() for lbl in box_1.findChildren(QLabel)]
        self.assertNotIn("Experiment ID", labels)

    def test_paradigm_badge_hidden_in_experiment_pane(self):
        self.assertTrue(self.viewer.paradigm_badge.isHidden())
        header_widgets = [
            self.viewer.header_bar.layout().itemAt(i).widget()
            for i in range(self.viewer.header_bar.layout().count())
            if self.viewer.header_bar.layout().itemAt(i).widget() is not None
        ]
        self.assertNotIn(self.viewer.paradigm_badge, header_widgets)

    def test_experiment_identity_box_removed_and_first_box_is_budget(self):
        cards = self.viewer.findChildren(ConfigBox)
        self.assertGreaterEqual(len(cards), 2)  # Budget, Methods, and any configured overrides

        # Toggling to Show Resolved Defaults resolves and renders inherited blocks
        self.viewer.chk_view_mode.setChecked(True)
        resolved_cards = self.viewer.findChildren(ConfigBox)
        self.assertGreaterEqual(len(resolved_cards), 2)
        self.assertTrue(self.viewer.chk_view_mode.isChecked())

        all_labels = [lbl.text() for lbl in self.viewer.findChildren(QLabel)]
        self.assertNotIn("Group", all_labels)
        self.assertNotIn("Inherits Defaults", all_labels)

        budget_card = next(
            card for card in cards
            if "Total Timesteps" in [lbl.text() for lbl in card.findChildren(QLabel)]
        )
        budget_labels = [lbl.text() for lbl in budget_card.findChildren(QLabel)]
        self.assertIn("Total Timesteps", budget_labels)
        self.assertIn("Random Seed", budget_labels)

    def test_redundant_params_stripped_from_experiment_yamls(self):
        exp_file = Path(self.temp_dir.name) / "custom_exp.yaml"
        exp_file.write_text(
            "experiment_id: custom_exp\n"
            "group: cartpole\n"
            "seed: 42\n"
            "total_timesteps: 10000\n",
            encoding="utf-8"
        )
        viewer = ConfigViewer()
        viewer.auto_save = True
        viewer.load_file(exp_file, "experiment/cartpole/custom_exp.yaml")

        self.assertNotIn("experiment_id", viewer.raw_data)
        self.assertNotIn("group", viewer.raw_data)

        viewer.save_to_disk()
        saved_text = exp_file.read_text(encoding="utf-8")
        saved_data = yaml.safe_load(saved_text)
        self.assertNotIn("experiment_id", saved_data)
        self.assertNotIn("group", saved_data)
        self.assertEqual(saved_data.get("seed"), 42)
        self.assertEqual(saved_data.get("total_timesteps"), 10000)

    def test_redundant_params_filtered_in_generic_boxes(self):
        agent_file = Path(self.temp_dir.name) / "agent_test.yaml"
        agent_file.write_text("algorithm: test_algo\nlr: 0.001\nbatch_size: 64\n", encoding="utf-8")
        viewer2 = ConfigViewer()
        viewer2.load_file(agent_file, "agent/agent_test.yaml")
        self.assertEqual(viewer2.file_title.text(), "agent_test")
        labels = [lbl.text() for lbl in viewer2.findChildren(QLabel)]
        self.assertNotIn("algorithm", labels)
        self.assertIn("lr", labels)
        self.assertIn("batch_size", labels)
        viewer2.close()

    def test_title_double_click_and_cancel(self):
        self.viewer._start_title_edit()
        self.assertFalse(self.viewer.title_editor.isHidden())
        self.assertTrue(self.viewer.file_title.isHidden())
        self.assertEqual(self.viewer.title_editor.text(), "test_exp")

        self.viewer._cancel_title_edit()
        self.assertTrue(self.viewer.title_editor.isHidden())
        self.assertFalse(self.viewer.file_title.isHidden())
        self.assertEqual(self.viewer.file_title.text(), "test_exp")

    def test_title_edit_commit_renames_file_and_rewrites_params(self):
        renamed_signals = []
        self.viewer.file_renamed.connect(lambda old, new: renamed_signals.append((old, new)))

        self.viewer._start_title_edit()
        self.viewer.title_editor.setText("renamed_exp")
        self.viewer._commit_title_edit()

        # Title display updated
        self.assertEqual(self.viewer.file_title.text(), "renamed_exp")
        self.assertFalse(self.viewer.title_editor.isVisible())

        # Old file should not exist, new file should exist
        old_file = Path(self.temp_dir.name) / "test_exp.yaml"
        new_file = Path(self.temp_dir.name) / "renamed_exp.yaml"
        self.assertFalse(old_file.exists())
        self.assertTrue(new_file.exists())

        # Redundant experiment_id is stripped from experiment YAMLs
        self.assertNotIn("experiment_id", self.viewer.raw_data)
        disk_data = yaml.safe_load(new_file.read_text(encoding="utf-8"))
        self.assertNotIn("experiment_id", disk_data)
        self.assertEqual(self.viewer.txt_exp_id.text(), "renamed_exp")

        # Signal emitted
        self.assertEqual(len(renamed_signals), 1)
        self.assertEqual(renamed_signals[0], ("experiment/test_exp.yaml", "experiment/renamed_exp.yaml"))

    def test_title_edit_commit_capitalization_case_change(self):
        # Renaming with only capitalization change (e.g. test_exp -> Test_Exp)
        # must succeed on case-insensitive filesystems without false "already exists" error
        self.viewer._start_title_edit()
        self.viewer.title_editor.setText("Test_Exp")
        self.viewer._commit_title_edit()

        self.assertEqual(self.viewer.file_title.text(), "Test_Exp")
        self.assertEqual(self.viewer.txt_exp_id.text(), "Test_Exp")
        self.assertNotIn("experiment_id", self.viewer.raw_data)
        self.assertEqual(self.viewer.current_path.name, "Test_Exp.yaml")
        self.assertTrue(self.viewer.current_path.exists())

    def test_breadcrumb_and_group_defaults_header(self):
        exp_path = Path(self.temp_dir.name) / "exp1.yaml"
        exp_path.write_text("total_timesteps: 5000\n", encoding="utf-8")
        self.viewer.load_file(exp_path, "experiment/cartpole/exp1.yaml")
        self.assertEqual(self.viewer.file_title.text(), "exp1")
        self.assertFalse(self.viewer.group_badge.isHidden())
        self.assertFalse(self.viewer.breadcrumb_sep.isHidden())

        base_path = Path(self.temp_dir.name) / "_base.yaml"
        base_path.write_text("paradigm: online_rl\n", encoding="utf-8")
        self.viewer.load_file(base_path, "experiment/cartpole/_base.yaml")
        self.assertEqual(self.viewer.file_title.text(), "Group Defaults")
        self.assertEqual(self.viewer.group_badge.text(), "cartpole")
        self.assertFalse(self.viewer.group_badge.isHidden())
        self.assertTrue(self.viewer.breadcrumb_sep.isHidden())

    def test_description_field_persistence_and_overrides(self):
        self.assertIsNotNone(self.viewer.txt_description)
        self.viewer.auto_save = False
        self.viewer.txt_description.setText("My test hypothesis")
        self.assertEqual(self.viewer.raw_data.get("description"), "My test hypothesis")
        self.assertTrue(self.viewer.is_dirty)

        overrides = self.viewer.get_overrides()
        self.assertIn("++description='My test hypothesis'", overrides)

        self.viewer.save_to_disk()
        saved = yaml.safe_load(self.viewer.current_path.read_text(encoding="utf-8"))
        self.assertEqual(saved.get("description"), "My test hypothesis")

        # Empty description is cleaned
        self.viewer.txt_description.setText("")
        self.assertNotIn("description", self.viewer.raw_data)
        self.viewer.save_to_disk()
        saved_empty = yaml.safe_load(self.viewer.current_path.read_text(encoding="utf-8"))
        self.assertNotIn("description", saved_empty)

    def test_universal_params_crud(self):
        # Adding universal parameter
        self.viewer._add_universal_param("cql_alpha", 5.0)
        self.assertEqual(self.viewer.raw_data["methods"]["params"]["cql_alpha"], 5.0)
        self.assertIn("++methods.params.cql_alpha=5.0", self.viewer.get_overrides())

        # Modifying universal parameter
        self.viewer._on_universal_param_edited("cql_alpha", "10.0")
        self.assertEqual(self.viewer.raw_data["methods"]["params"]["cql_alpha"], 10.0)

        # Removing universal parameter
        self.viewer._remove_universal_param("cql_alpha")
        self.assertNotIn("cql_alpha", self.viewer.raw_data.get("methods", {}).get("params", {}))

    def test_shared_params_block_starts_empty(self):
        self.viewer.add_block("params")
        self.assertEqual(self.viewer.raw_data.get("methods", {}).get("params", {}), {})
        labels = [lbl.text() for lbl in self.viewer.findChildren(QLabel)]
        self.assertIn("Shared Parameters", labels)
        self.assertNotIn("No shared parameters configured.", labels)
        self.viewer._add_universal_param("lr", 0.001)
        self.assertEqual(self.viewer.raw_data["methods"]["params"]["lr"], 0.001)

    def test_shared_params_picker_uses_component_groups(self):
        from PyQt6.QtWidgets import QToolButton

        self.viewer.add_block("params")
        button = next(b for b in self.viewer.findChildren(QToolButton) if b.toolTip() == "Add shared parameter")
        actions = {a.text(): a for a in button.menu().actions()}
        self.assertIn("Algorithm", actions)
        self.assertIn("Model", actions)
        self.assertFalse(any(name.startswith(("agent.", "model.")) for name in actions))
        actions["Algorithm"].trigger()
        self.assertEqual(self.viewer.raw_data["methods"]["params"]["agent"], {})
        self.assertTrue(any(lbl.text() == "Algorithm" for lbl in self.viewer.findChildren(QLabel)))
        self.viewer._select_shared_component("agent", "", "test_agent")
        self.assertEqual(self.viewer.raw_data["methods"]["params"]["agent"], {"test_agent": {}})
        self.assertTrue(any(b.toolTip() == "Add test_agent parameter" for b in self.viewer.findChildren(QToolButton)))

    def test_shared_algorithm_choices_follow_paradigm(self):
        from frontend.config_viewer import config_tree
        from frontend.widgets import ComboBox

        tree = config_tree()
        self.viewer.add_block("params")
        self.viewer._add_universal_param("agent", {})
        selector = self.viewer.findChild(ComboBox, "shared_agent_selector")
        self.assertIsNotNone(selector)
        choices = {selector.itemData(i) for i in range(1, selector.count())}
        self.assertEqual(choices, {agent.name for agent in tree.agents_for("online_rl")})

    def test_removing_last_shared_parameter_keeps_box(self):
        self.viewer._add_universal_param("lr", 0.001)
        self.viewer._remove_universal_param("lr")
        self.assertIn("Shared Parameters", [lbl.text() for lbl in self.viewer.findChildren(QLabel)])
        self.assertNotIn("lr", self.viewer.raw_data.get("methods", {}).get("params", {}))

    def test_methods_and_shared_params_title_sizes(self):
        """Verify method box title matches experiment title size (23px heading) and shared params matches method titles (16px cardTitle)."""
        self.viewer._add_universal_param("lr", 0.001)

        # Method box title "Methods"
        methods_labels = [
            lbl for lbl in self.viewer.findChildren(QLabel)
            if lbl.text() == "Methods"
        ]
        self.assertEqual(len(methods_labels), 1)
        methods_lbl = methods_labels[0]
        self.assertEqual(methods_lbl.objectName(), "heading")
        self.assertIn("23px", methods_lbl.styleSheet())
        self.assertEqual(self.viewer.file_title.objectName(), "heading")

        # Shared params title "Shared Parameters"
        shared_labels = [
            lbl for lbl in self.viewer.findChildren(QLabel)
            if lbl.text() == "Shared Parameters"
        ]
        self.assertEqual(len(shared_labels), 1)
        shared_lbl = shared_labels[0]
        self.assertEqual(shared_lbl.objectName(), "cardTitle")
        self.assertIn("16px", shared_lbl.styleSheet())

    def test_method_hyperparameter_crud(self):
        # Adding hyperparameter to method
        self.viewer._add_method_param("ppo", "ent_coef", 0.05)
        self.assertEqual(self.viewer.raw_data["methods"]["ppo"]["ent_coef"], 0.05)
        self.assertIn("++methods.ppo.ent_coef=0.05", self.viewer.get_overrides())

        # Removing hyperparameter from method
        self.viewer._remove_method_param("ppo", "ent_coef")
        self.assertNotIn("ent_coef", self.viewer.raw_data["methods"]["ppo"])

    def test_method_card_fields_have_consistent_widths(self):
        from PyQt6.QtWidgets import QAbstractSpinBox, QComboBox
        from frontend.config_viewer import MethodCardsContainer, TEXT_FIELD_WIDTH

        container = self.viewer.findChild(MethodCardsContainer)
        card = next(card for name, card in container._cards if name == "ppo")
        fields = card.findChildren(QAbstractSpinBox) + card.findChildren(QComboBox)
        self.assertGreaterEqual(len(fields), 2)
        expected_width = round(TEXT_FIELD_WIDTH * 2 / 3)
        self.assertTrue(all(field.minimumWidth() == field.maximumWidth() == expected_width for field in fields))

    def test_experiment_sections_can_be_reordered_and_survive_rerender(self):
        from frontend.config_viewer import ConfigBox

        def titles():
            boxes = [self.viewer.boxes_layout.itemAt(i).widget() for i in range(self.viewer.boxes_layout.count())]
            return [next(label.text() for label in box.findChildren(QLabel)
                         if label.objectName() in ("heading", "cardTitle"))
                    for box in boxes if isinstance(box, ConfigBox)]

        original = titles()
        self.assertGreaterEqual(len(original), 2)
        self.viewer._move_experiment_box(original[1], -1)
        self.assertEqual(titles()[:2], [original[1], original[0]])
        self.viewer._render_boxes()
        self.assertEqual(titles()[:2], [original[1], original[0]])

    def test_modular_block_crud(self):
        # Runtime defaults live in the top card; component overrides are
        # available even when no matching plugin is installed.
        self.assertNotIn("resources", self.viewer.raw_data)
        menu_labels = [action.text() for action in self.viewer.btn_add_block.menu().actions()]
        self.assertIn("Data Loaders", menu_labels)
        self.assertIn("Evaluators", menu_labels)
        self.assertIn("Figures and Tables", menu_labels)
        self.assertIn("Checkpoint Settings", menu_labels)
        self.assertIn("Recorded Metrics", menu_labels)

        # Unsupported direct calls do not create hidden configuration blocks.
        self.viewer.add_block("resources")
        self.assertNotIn("resources", self.viewer.raw_data)

    def test_sweep_override_menu_add_and_remove(self):
        from PyQt6.QtWidgets import QLineEdit

        actions = self.viewer.btn_add_block.menu().actions()
        action = next(a for a in actions if a.text() == "Hyperparameter Sweep (Optuna)")
        action.trigger()
        self.assertEqual(self.viewer.raw_data["tuning"]["n_trials"], 40)
        self.assertIsNotNone(self.viewer.findChild(QLineEdit, "sweepMetric"))
        from frontend.widgets import ToggleSlider
        enabled = self.viewer.findChild(ToggleSlider, "sweepEnabled")
        self.assertIsNone(enabled)
        self.assertNotIn("enabled", self.viewer.raw_data["tuning"])
        self.assertNotIn("n_jobs", self.viewer.raw_data["tuning"])
        titles = [widget.text() for widget in self.viewer.findChildren(QLabel)]
        self.assertEqual(titles.count("Hyperparameter Sweep"), 1)
        self.assertNotIn("tuning", titles)
        self.viewer.remove_block("tuning")
        self.assertNotIn("tuning", self.viewer.raw_data)

    def test_optuna_component_settings_are_saved_in_its_options(self):
        from PyQt6.QtWidgets import QComboBox
        from frontend.widgets import SpinBox

        self.viewer.add_block("tuning")
        selector = self.viewer.findChild(QComboBox, "sweepComponent")
        self.assertEqual(selector.currentText(), "optuna")
        sampler = self.viewer.findChild(QComboBox, "sweep_sampler")
        sampler.setCurrentText("random")
        self.viewer._add_tuner_option("optuna", "seed", 42)
        seed = self.viewer.findChild(SpinBox, "sweep_seed")
        seed.setValue(17)
        options = self.viewer.raw_data["tuning"]["options"]["optuna"]
        self.assertEqual(options["sampler"], "random")
        self.assertEqual(options["seed"], 17)
        self.assertIsNone(self.viewer.findChild(SpinBox, "sweep_tpe_candidates"))
        self.viewer.findChild(QComboBox, "sweep_sampler").setCurrentText("tpe")
        self.assertIsNone(self.viewer.findChild(SpinBox, "sweep_tpe_candidates"))
        self.viewer._add_tuner_option("optuna", "tpe_candidates", 24)
        self.assertIsNotNone(self.viewer.findChild(SpinBox, "sweep_tpe_candidates"))
        self.assertIsNone(self.viewer.findChild(SpinBox, "sweep_pruner_startup_trials"))
        self.viewer._add_tuner_option("optuna", "pruner", "median")
        self.viewer._add_tuner_option("optuna", "pruner_startup_trials", 5)
        self.assertIsNotNone(self.viewer.findChild(SpinBox, "sweep_pruner_startup_trials"))

    def test_tuner_component_supplies_its_own_editor_fields(self):
        from PyQt6.QtWidgets import QComboBox, QLineEdit

        class OtherTuner:
            api_version = 1
            def settings(self):
                return [{"key": "profile", "label": "Profile", "type": "text", "default": "fast"}]
            def validate_config(self, cfg): pass
            def validate_dimension(self, target, dimension): pass
            def compile(self, overrides, **context): return overrides
            def job_entrypoint(self, cfg, method_cfg, script, python): return script, python
            def study_name(self, cfg, method): return "study"
            def prepare(self, cfg, storage, study): pass
            def finalize(self, group, experiment, method, storage, study): pass
            def storage(self, cfg): return None

        with patch("src.app.core.plugin_discovery.discover_entry_points", return_value={"other": OtherTuner}):
            self.viewer.add_block("tuning")
            self.viewer.findChild(QComboBox, "sweepComponent").setCurrentText("other")
            profile = self.viewer.findChild(QLineEdit, "sweep_profile")
            self.assertEqual(profile.text(), "fast")
            profile.setText("thorough")
            self.assertEqual(self.viewer.raw_data["tuning"]["options"]["other"]["profile"], "thorough")
            self.viewer.findChild(QComboBox, "sweepComponent").setCurrentText("optuna")
            self.assertEqual(self.viewer.raw_data["tuning"]["options"]["other"]["profile"], "thorough")

    def test_invalid_tuner_values_remain_visible_for_correction(self):
        from PyQt6.QtWidgets import QComboBox, QLineEdit

        self.viewer.add_block("tuning")
        self.viewer.raw_data["tuning"]["options"] = {"optuna": {"sampler": "unknown", "seed": "bad"}}
        self.viewer._render_boxes()
        self.assertEqual(self.viewer.findChild(QComboBox, "sweep_sampler").currentText(), "unknown")
        self.assertEqual(self.viewer.findChild(QLineEdit, "sweep_seed").text(), "bad")
        self.assertEqual(self.viewer.raw_data["tuning"]["options"]["optuna"]["seed"], "bad")

    def test_parameter_sweep_range_edit_and_remove(self):
        from PyQt6.QtWidgets import QComboBox, QLineEdit, QToolButton

        self.viewer.add_block("tuning")
        button = self.viewer.findChild(QToolButton, "sweepToggle_ppo_agent.lr")
        self.assertIsNotNone(button)
        button.click()
        picker = self.viewer.findChild(QComboBox, "sweepType_ppo_agent.lr")
        picker.setCurrentIndex(picker.findData("float"))
        picker.activated.emit(picker.currentIndex())
        self.viewer.show()
        QApplication.processEvents()
        picker = self.viewer.findChild(QComboBox, "sweepType_ppo_agent.lr")
        remove_sweep = self.viewer.findChild(QToolButton, "sweepRemove_ppo_agent.lr")
        from frontend.config_viewer import _RowHoverReveal, IndentedHierarchyBlock
        hover = remove_sweep.findChild(_RowHoverReveal)
        self.assertIn(picker, [ref() for ref in hover._targets])
        self.assertTrue(self.viewer.findChildren(IndentedHierarchyBlock))
        lower = self.viewer.findChild(QLineEdit, "sweep_low_ppo_agent.lr")
        sweep_block = lower.parentWidget()
        self.assertIsInstance(sweep_block, IndentedHierarchyBlock)
        self.assertLess(sweep_block.mapTo(self.viewer, sweep_block.rect().topLeft()).x(),
                        picker.mapTo(self.viewer, picker.rect().topLeft()).x())
        with patch("frontend.config_viewer.QCursor.pos", return_value=picker.mapToGlobal(picker.rect().center())):
            hover._update_visibility()
        self.assertTrue(remove_sweep.isVisible())
        low = self.viewer.findChild(QLineEdit, "sweep_low_ppo_agent.lr")
        high = self.viewer.findChild(QLineEdit, "sweep_high_ppo_agent.lr")
        low.setText("0.0001")
        high.setText("0.01")
        spec = self.viewer.raw_data["methods"]["ppo"]
        self.assertEqual(spec["tune"]["agent.lr"], {"type": "float", "low": 0.0001, "high": 0.01})
        self.assertEqual(spec["lr"], 0.0005)
        self.assertEqual(self.viewer.findChild(QLineEdit, "sweep_step_ppo_agent.lr").placeholderText(), "Leave blank for no step")
        self.viewer.findChild(QToolButton, "sweepRemove_ppo_agent.lr").click()
        self.assertNotIn("tune", spec)
        self.assertEqual(spec["lr"], 0.0005)
        delete = next(button for button in self.viewer.findChildren(QToolButton) if button.toolTip() == "Remove lr parameter")
        delete.click()
        self.assertNotIn("lr", spec)

    def test_shared_and_experiment_sweeps_apply_to_each_method(self):
        from PyQt6.QtWidgets import QToolButton, QComboBox
        from src.app.pipeline.config import parse_methods_dict
        from src.app.pipeline.commands import build_method_overrides

        self.viewer.raw_data["batch_size"] = 64
        self.viewer.raw_data["methods"]["params"] = {"lr": 0.01}
        self.viewer.raw_data["methods"]["other"] = {"agent": "ppo", "model": "dnn"}
        self.viewer.add_block("tuning")
        self.viewer.findChild(QToolButton, "sweepToggle_params_agent.lr").click()
        picker = self.viewer.findChild(QComboBox, "sweepType_params_agent.lr")
        picker.setCurrentIndex(picker.findData("float"))
        picker.activated.emit(picker.currentIndex())
        self.viewer.findChild(QToolButton, "sweepToggle_params_config.batch_size").click()
        picker = self.viewer.findChild(QComboBox, "sweepType_params_config.batch_size")
        picker.setCurrentIndex(picker.findData("int"))
        picker.activated.emit(picker.currentIndex())
        methods = parse_methods_dict(self.viewer.raw_data)
        for name, method in methods.items():
            if name == "ppo":
                self.assertNotIn("agent.lr", method["tune"])
            else:
                self.assertIn("agent.lr", method["tune"])
            self.assertIn("config.batch_size", method["tune"])
            overrides = build_method_overrides(name, method, cfg=self.viewer.raw_data, is_sweep=True)
            self.assertIn("batch_size=range(64, 64, 1)", overrides)

    def test_sweep_setup_cancel_and_list_entries(self):
        from PyQt6.QtWidgets import QComboBox, QLineEdit, QToolButton

        self.viewer.add_block("tuning")
        self.viewer.findChild(QToolButton, "sweepToggle_ppo_agent.lr").click()
        self.assertNotIn("tune", self.viewer.raw_data["methods"]["ppo"])
        self.viewer._cancel_sweep_setup("ppo", "agent.lr")
        self.assertEqual(self.viewer.raw_data["methods"]["ppo"]["lr"], 0.0005)
        self.viewer.findChild(QToolButton, "sweepToggle_ppo_agent.lr").click()
        picker = self.viewer.findChild(QComboBox, "sweepType_ppo_agent.lr")
        picker.setCurrentIndex(picker.findData("choice"))
        picker.activated.emit(picker.currentIndex())
        self.viewer.findChild(QLineEdit, "sweep_choice_ppo_agent.lr_1").setText("0.01")
        self.viewer.findChild(QToolButton, "sweepAddChoice_ppo_agent.lr").click()
        self.assertEqual(self.viewer.raw_data["methods"]["ppo"]["tune"]["agent.lr"]["values"], [0.0005, 0.01, ""])
        self.viewer._toggle_method_sweep("ppo", "agent.lr", 0.0005)
        self.assertEqual(self.viewer.raw_data["methods"]["ppo"]["lr"], 0.0005)
        self.assertNotIn("tune", self.viewer.raw_data["methods"]["ppo"])

    def test_saved_sweep_can_open_and_close_in_event_loop(self):
        import subprocess

        data = yaml.safe_load(self.file_path.read_text())
        data["tuning"] = {"metric": "eval/reward", "direction": "maximize"}
        data["methods"]["ppo"]["tune"] = {"agent.lr": {"type": "float", "low": 0.0001, "high": 0.01}}
        self.file_path.write_text(yaml.safe_dump(data))
        code = '''
import sys
from pathlib import Path
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer
from frontend.config_viewer import ConfigViewer
app = QApplication([])
viewer = ConfigViewer()
viewer.load_file(Path(sys.argv[1]), "experiment/test_exp.yaml")
viewer.show()
QTimer.singleShot(200, app.quit)
app.exec()
'''
        result = subprocess.run([sys.executable, "-X", "faulthandler", "-c", code, str(self.file_path)],
                                capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_dataset_card_specification(self):
        # An online RL experiment cannot opt into a static dataset source.
        self.viewer.add_block("dataset")
        self.assertNotIn("dataset_path", self.viewer.raw_data)

    def test_method_inline_rename(self):
        self.viewer.auto_save = False
        self.viewer.raw_data["methods"] = {"ppo": {"agent": "ppo"}, "cql": {"agent": "cql"}}
        self.viewer.rename_method("ppo", "ppo_renamed")
        self.assertIn("ppo_renamed", self.viewer.raw_data["methods"])
        self.assertNotIn("ppo", self.viewer.raw_data["methods"])
        self.assertEqual(list(self.viewer.raw_data["methods"].keys()), ["ppo_renamed", "cql"])
        self.assertTrue(self.viewer.is_dirty)

    def test_method_reordering(self):
        self.viewer.auto_save = False
        self.viewer.raw_data["methods"] = {
            "params": {"lr": 0.001},
            "m1": {"agent": "ppo"},
            "m2": {"agent": "cql"},
            "m3": {"agent": "cew"},
        }
        self.viewer.reorder_method("m3", 0)
        self.assertEqual(list(self.viewer.raw_data["methods"].keys()), ["params", "m3", "m1", "m2"])
        self.viewer.reorder_method("m1", 2)
        self.assertEqual(list(self.viewer.raw_data["methods"].keys()), ["params", "m3", "m2", "m1"])
        self.assertTrue(self.viewer.is_dirty)

    def test_method_param_auto_populates_default(self):
        self.viewer.raw_data["methods"] = {"ppo": {"agent": "ppo"}}
        self.viewer._add_method_param("ppo", "ent_coef")
        self.assertEqual(self.viewer.raw_data["methods"]["ppo"]["ent_coef"], 0.01)

        self.viewer._add_universal_param("lr")
        self.assertEqual(self.viewer.raw_data["methods"]["params"]["lr"], 0.0003)

    def test_add_overrides_button_and_labels(self):
        self.assertFalse(self.viewer.btn_add_block.icon().isNull())
        self.assertEqual(self.viewer.btn_add_block.toolTip(), "Add overrides")

    def test_group_double_click_rename(self):
        root = Path(self.temp_dir.name)
        group_dir = root / "experiment" / "cartpole"
        group_dir.mkdir(parents=True, exist_ok=True)
        exp_file = group_dir / "exp1.yaml"
        exp_file.write_text("paradigm: online_rl\n", encoding="utf-8")

        self.viewer.load_file(exp_file, "experiment/cartpole/exp1.yaml")
        self.viewer._start_group_edit()
        self.viewer.group_editor.setText("cartpole_renamed")
        renamed_signals = []
        self.viewer.group_renamed.connect(lambda o, n: renamed_signals.append((o, n)))
        self.viewer._commit_group_edit()

        self.assertEqual(self.viewer.current_rel_path, "experiment/cartpole_renamed/exp1.yaml")
        self.assertTrue((root / "experiment" / "cartpole_renamed" / "exp1.yaml").exists())
        self.assertEqual(renamed_signals, [("cartpole", "cartpole_renamed")])


@unittest.skipIf(not HAS_PYQT6, "PyQt6 not installed in current environment")
class TestComponentsPanel(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        # The real layout, <workspace>/in/config: the panel takes the workspace (and the Hub's
        # .thetaide) from two levels up, which for a bare temp dir is "/" on Linux and unwritable
        self.root = Path(self.temp_dir.name) / "in" / "config"
        (self.root / "experiment" / "cartpole").mkdir(parents=True)
        (self.root / "agent").mkdir(parents=True)
        (self.root / "env").mkdir(parents=True)

        (self.root / "agent" / "ppo.yaml").write_text("lr: 0.0003\nbatch_size: 64\n", encoding="utf-8")
        (self.root / "env" / "cartpole.yaml").write_text("name: cartpole\nreward_shaping: false\n", encoding="utf-8")
        (self.root / "config.yaml").write_text("paradigm: online_rl\n", encoding="utf-8")

        self.panel = ComponentsPanel()
        self.panel.components_tree.root_dir = self.root
        self.panel.components_tree.populate()
        self.panel.components_tree.select_file("agent/ppo.yaml")

    def tearDown(self):
        self.panel.close()
        self.temp_dir.cleanup()

    def test_components_panel_tree_only_has_components(self):
        top_count = self.panel.components_tree.tree.topLevelItemCount()
        top_texts = [self.panel.components_tree.tree.topLevelItem(i).text(0) for i in range(top_count)]
        self.assertTrue(any(t in ("Algorithms", "agent") for t in top_texts))
        self.assertFalse(any("experiment" in t for t in top_texts))

    def test_select_component_loads_into_viewer_and_raw_editor(self):
        ok = self.panel.components_tree.select_file("agent/ppo.yaml")
        self.assertTrue(ok)
        self.assertEqual(self.panel.viewer.file_title.text(), "ppo")
        self.assertIn("lr", self.panel.viewer.raw_data)
        cards = self.panel.viewer.findChildren(ConfigBox)
        self.assertGreaterEqual(len(cards), 1)
        for card in cards:
            self.assertIsNone(card.title_label)
            self.assertIsNone(card.subtitle_label)

    def test_toggle_raw_yaml_view(self):
        self.assertTrue(self.panel.raw_panel.isHidden())
        self.panel.toggle_raw_preview(True)
        self.assertFalse(self.panel.raw_panel.isHidden())
        self.panel.toggle_raw_preview(False)
        self.assertTrue(self.panel.raw_panel.isHidden())

    def test_component_hub_replaces_parameters_panels_and_restores_on_selection(self):
        # Initially on parameters view
        self.assertFalse(self.panel.is_hub_active)
        self.assertFalse(self.panel.btn_hub.isChecked())
        self.assertFalse(self.panel.btn_toggle_raw.isHidden())
        self.assertEqual(self.panel.content_stack.currentWidget(), self.panel.content_splitter)

        # Click Component Hub button in top right
        self.panel.btn_hub.click()
        self.assertTrue(self.panel.is_hub_active)
        self.assertTrue(self.panel.btn_hub.isChecked())
        self.assertTrue(self.panel.btn_toggle_raw.isHidden())
        self.assertEqual(self.panel.content_stack.currentWidget(), self.panel.hub_view)

        # Click on a component again in the tree
        self.panel.components_tree.select_file("env/cartpole.yaml")
        # Brought back to parameters view
        self.assertFalse(self.panel.is_hub_active)
        self.assertFalse(self.panel.btn_hub.isChecked())
        self.assertFalse(self.panel.btn_toggle_raw.isHidden())
        self.assertEqual(self.panel.content_stack.currentWidget(), self.panel.content_splitter)
        self.assertEqual(self.panel.viewer.file_title.text(), "cartpole")

    def test_component_hub_toggle_and_close_button(self):
        # Open hub
        self.panel.btn_hub.click()
        self.assertTrue(self.panel.is_hub_active)

        # Click btn_hub again to toggle back
        self.panel.btn_hub.click()
        self.assertFalse(self.panel.is_hub_active)
        self.assertEqual(self.panel.content_stack.currentWidget(), self.panel.content_splitter)

        # Open hub again
        self.panel.btn_hub.click()
        self.assertTrue(self.panel.is_hub_active)

        # Click Back to Parameters button in Hub footer
        self.panel.hub_view.btn_close.click()
        self.assertFalse(self.panel.is_hub_active)
        self.assertEqual(self.panel.content_stack.currentWidget(), self.panel.content_splitter)

    def test_reload_components_preserves_hub_view_on_install_and_uninstall(self):
        # Open hub
        self.panel.btn_hub.click()
        self.assertTrue(self.panel.is_hub_active)
        self.assertEqual(self.panel.content_stack.currentWidget(), self.panel.hub_view)

        # Simulate install: new component created and reload_components called
        (self.root / "agent" / "new_dqn.yaml").write_text("gamma: 0.99\n", encoding="utf-8")
        self.panel.reload_components(target_rel_path="agent/new_dqn.yaml", ensure_expanded="agent")
        # Must still be in the Hub!
        self.assertTrue(self.panel.is_hub_active)
        self.assertEqual(self.panel.content_stack.currentWidget(), self.panel.hub_view)

        # Simulate uninstall: component deleted and reload_components called
        (self.root / "agent" / "new_dqn.yaml").unlink()
        self.panel.reload_components(target_rel_path=None, ensure_expanded="agent")
        # Must still be in the Hub!
        self.assertTrue(self.panel.is_hub_active)
        self.assertEqual(self.panel.content_stack.currentWidget(), self.panel.hub_view)

        # Only clicking a component in the tree brings you back to params
        self.panel.components_tree.select_file("agent/ppo.yaml")
        self.assertFalse(self.panel.is_hub_active)
        self.assertEqual(self.panel.content_stack.currentWidget(), self.panel.content_splitter)

    def test_hub_and_raw_yaml_buttons_in_tree_header(self):
        from PyQt6.QtWidgets import QToolButton
        tree = self.panel.components_tree
        tree = self.panel.components_tree
        self.assertIsNone(self.panel.btn_save)
        self.assertIsInstance(self.panel.btn_hub, QToolButton)
        self.assertIsInstance(self.panel.btn_toggle_raw, QToolButton)
        self.assertEqual(self.panel.btn_hub.property("svg_icon"), "hub")
        self.assertEqual(self.panel.btn_toggle_raw.property("svg_icon"), "raw_yaml")
        self.assertEqual(self.panel.btn_hub.size().width(), 30)
        self.assertEqual(self.panel.btn_hub.size().height(), 30)
        self.assertEqual(self.panel.btn_toggle_raw.size().width(), 30)
        self.assertEqual(self.panel.btn_toggle_raw.size().height(), 30)
        # Verify they reside in the tree header layout next to btn_new and btn_duplicate
        self.assertIs(tree.btn_hub, self.panel.btn_hub)
        self.assertIs(tree.btn_toggle_raw, self.panel.btn_toggle_raw)
        header_widgets = [
            tree.header_layout.itemAt(i).widget()
            for i in range(tree.header_layout.count())
            if tree.header_layout.itemAt(i).widget() is not None
        ]
        self.assertIn(tree.btn_new, header_widgets)
        self.assertIn(tree.btn_duplicate, header_widgets)
        self.assertIn(tree.btn_hub, header_widgets)
        self.assertIn(tree.btn_toggle_raw, header_widgets)
        self.assertNotIn(tree.btn_save, header_widgets)

    def test_filetree_unsynced_indicator(self):
        # Initial state: clean
        self.panel.components_tree.select_file("agent/ppo.yaml")
        item = self.panel.components_tree.find_file_item("agent/ppo.yaml")
        self.assertIsNotNone(item)
        self.assertEqual(item.text(0), "ppo")
        self.assertEqual(self.panel.viewer.dirty_status.text(), "")
        self.assertIsNone(self.panel.btn_save)

        # Modify value in viewer
        from frontend.config_viewer import CompactDoubleSpinBox
        from frontend.widgets import DoubleSpinBox, SpinBox
        spins = self.panel.viewer.findChildren((SpinBox, DoubleSpinBox, CompactDoubleSpinBox))
        self.assertGreater(len(spins), 0)
        orig_val = spins[0].value()
        spins[0].setValue(orig_val + 1)

        # Value immediately auto-saves to disk without needing a save button or dirty dot
        self.assertFalse(self.panel.viewer.is_dirty)
        self.assertEqual(self.panel.viewer.dirty_status.text(), "")
        self.assertEqual(item.text(0), "ppo")
        self.assertIsNone(item.data(0, Qt.ItemDataRole.ForegroundRole))

        # Disk is updated live
        ppo_disk = (self.root / "agent" / "ppo.yaml").read_text(encoding="utf-8")
        self.assertIn(str(orig_val + 1), ppo_disk)

    def test_switching_clean_files_does_not_mark_dirty_or_corrupt_colors(self):
        # Click item 1
        self.panel.components_tree.select_file("agent/ppo.yaml")
        item1 = self.panel.components_tree.find_file_item("agent/ppo.yaml")
        self.assertEqual(item1.text(0), "ppo")
        self.assertIsNone(item1.data(0, Qt.ItemDataRole.ForegroundRole))

        # Immediately click item 2 without making changes
        self.panel.components_tree.select_file("env/cartpole.yaml")
        item2 = self.panel.components_tree.find_file_item("env/cartpole.yaml")
        self.assertEqual(item2.text(0), "cartpole")
        self.assertIsNone(item2.data(0, Qt.ItemDataRole.ForegroundRole))

        # Item 1 must remain clean and readable (no dot, no black brush override)
        self.assertEqual(item1.text(0), "ppo")
        self.assertIsNone(item1.data(0, Qt.ItemDataRole.ForegroundRole))
        self.assertFalse(self.panel.viewer.is_dirty)

    def test_component_rename_rewrites_params_and_updates_tree(self):
        self.panel.components_tree.select_file("agent/ppo.yaml")
        self.assertEqual(self.panel.viewer.file_title.text(), "ppo")

        # Edit title in place
        self.panel.viewer._start_title_edit()
        self.panel.viewer.title_editor.setText("ppo_custom")
        self.panel.viewer._commit_title_edit()

        # Display updated
        self.assertEqual(self.panel.viewer.file_title.text(), "ppo_custom")
        self.assertEqual(self.panel.active_rel_path, "agent/ppo_custom.yaml")

        # Check disk
        old_file = self.root / "agent" / "ppo.yaml"
        new_file = self.root / "agent" / "ppo_custom.yaml"
        self.assertFalse(old_file.exists())
        self.assertTrue(new_file.exists())

        # Parameters rewritten in YAML
        data = yaml.safe_load(new_file.read_text(encoding="utf-8"))
        self.assertEqual(data.get("algorithm"), "ppo_custom")

        # Raw editor preview updated
        self.assertIn("ppo_custom", self.panel.raw_edit.toPlainText())

        # Tree updated with new file
        self.assertEqual(self.panel.components_tree.current_rel_path, "agent/ppo_custom.yaml")
        item = self.panel.components_tree.find_file_item("agent/ppo_custom.yaml")
        self.assertIsNotNone(item)
        self.assertEqual(item.text(0), "ppo_custom")



@unittest.skipIf(not HAS_PYQT6, "PyQt6 not installed in current environment")
class TestExperimentPaneWindowIntegration(unittest.TestCase):
    class HeadlessTerminalPanel(QWidget):
        """Keep window integration tests from launching a real interactive shell."""

        def __init__(self, *, parent=None, **_kwargs):
            super().__init__(parent)
            self.terminals = []

        def configure(self, **_kwargs):
            pass

        def set_font_size(self, _size):
            pass

        def apply_theme(self, _theme):
            pass

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp_dir.name)
        with (
            patch("frontend.app.TerminalPanel", self.HeadlessTerminalPanel),
            patch.object(Window, "init_default_experiment", lambda _self: None),
        ):
            self.window = Window(self.data_dir)
        self.window.show()

    def tearDown(self):
        self.window.close()
        self.temp_dir.cleanup()

    def test_hydra_yaml_preview_hidden_by_default(self):
        self.assertTrue(self.window.preview_panel.isHidden())
        self.assertFalse(self.window.btn_toggle_yaml.isChecked())

    def test_toggle_hydra_yaml_preview(self):
        # Click toggle to open
        self.window.toggle_yaml_preview(True)
        self.assertFalse(self.window.preview_panel.isHidden())
        self.assertTrue(self.window.btn_toggle_yaml.isChecked())

        # Click toggle to close
        self.window.toggle_yaml_preview(False)
        self.assertTrue(self.window.preview_panel.isHidden())
        self.assertFalse(self.window.btn_toggle_yaml.isChecked())

    def test_selecting_tree_item_loads_into_viewer(self):
        # Select quick_test.yaml
        ok = self.window.config_tree.select_file("experiment/cartpole/quick_test.yaml")
        if ok:
            self.assertEqual(self.window.current_experiment, "cartpole/quick_test")
            self.assertEqual(self.window.config_viewer.file_title.text(), "quick_test")

    def test_experiment_pane_only_has_experiments(self):
        self.assertEqual(self.window.config_tree.mode, "experiments")
        top_texts = [self.window.config_tree.tree.topLevelItem(i).text(0) for i in range(self.window.config_tree.tree.topLevelItemCount())]
        self.assertFalse(any(t == "experiment" for t in top_texts))
        self.assertFalse(any(t in ("agent", "env", "model", "paradigms", "site") for t in top_texts))
        self.assertTrue(any(t in ("thetaide", "tabular_classification", "cartpole", "mimic") for t in top_texts))

    def test_components_pane_integrated_in_window(self):
        self.assertIsNotNone(self.window.components_panel)
        self.assertEqual(self.window.components_panel.components_tree.mode, "components")
        self.assertIn("components", self.window.tabs.tabs)
        top_count = self.window.components_panel.components_tree.tree.topLevelItemCount()
        top_texts = [self.window.components_panel.components_tree.tree.topLevelItem(i).text(0) for i in range(top_count)]
        self.assertFalse(any("experiment" in t for t in top_texts))

    def test_experiment_pane_square_header_buttons_and_actions_bar_removed(self):
        from PyQt6.QtWidgets import QToolButton
        tree = self.window.config_tree
        self.assertIsInstance(self.window.start_button, QToolButton)
        self.assertIsNone(self.window.btn_save)
        self.assertIsInstance(self.window.btn_export, QToolButton)
        self.assertIsInstance(self.window.btn_toggle_yaml, QToolButton)
        self.assertIs(self.window.start_button, tree.btn_launch)
        self.assertIs(self.window.btn_export, tree.btn_export)
        self.assertIs(self.window.btn_toggle_yaml, tree.btn_toggle_yaml)
        self.assertEqual(self.window.start_button.property("svg_icon"), "launch")
        self.assertEqual(self.window.btn_export.property("svg_icon"), "export")
        self.assertEqual(self.window.btn_toggle_yaml.property("svg_icon"), "raw_yaml")
        # Add to queue and Stop are not in the experiment pane: they are the Queue tab's run controls
        self.assertIs(self.window.queue_button, self.window.queue_panel.add_button)
        self.assertIs(self.window.stop_button, self.window.queue_panel.stop_button)
        self.assertFalse(tree.isAncestorOf(self.window.queue_button) or tree.isAncestorOf(self.window.stop_button))

        # Header layout widgets check
        header_widgets = [
            tree.header_layout.itemAt(i).widget()
            for i in range(tree.header_layout.count())
            if tree.header_layout.itemAt(i).widget() is not None
        ]
        self.assertIn(tree.btn_new, header_widgets)
        self.assertIn(tree.btn_duplicate, header_widgets)
        self.assertIn(tree.btn_launch, header_widgets)
        self.assertIn(tree.btn_export, header_widgets)
        self.assertIn(tree.btn_toggle_yaml, header_widgets)
        self.assertNotIn(tree.btn_save, header_widgets)

        # Ensure right panel layout has no actions_bar
        right_panel = self.window.preview_splitter.parentWidget()
        right_layout = right_panel.layout()
        self.assertEqual(right_layout.count(), 1)
        self.assertIs(right_layout.itemAt(0).widget(), self.window.preview_splitter)


if __name__ == "__main__":
    unittest.main()

"""Unit tests for nested subfolder CRUD in Theta-IDE."""
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QLabel

app = QApplication.instance() or QApplication(sys.argv[:1])

from frontend.config_model import ConfigTree, Experiment
from frontend.config_tree import TOOL_ICON_COLORS, ConfigTreeWidget
from frontend.sidetabs import svg_icon


class TestFolderCRUD(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

        # Setup standard directory structure
        self.exp_root = self.root / "experiment"
        self.group = self.exp_root / "cartpole"
        self.group.mkdir(parents=True)

        self.base = self.group / "_base.yaml"
        self.base.write_text("paradigm: online_rl\nenv: cartpole\n", encoding="utf-8")

        self.exp1 = self.group / "quick_test.yaml"
        self.exp1.write_text("experiment_id: quick_test\nseed: 42\nbudget: {total_timesteps: 1000}\n", encoding="utf-8")

        self.widget = ConfigTreeWidget(root_dir=self.root, mode="experiments")

    def test_create_subfolder(self):
        rel, err = self.widget.create_dir("experiment/cartpole", "ablations")
        self.assertIsNone(err)
        self.assertEqual(rel, "experiment/cartpole/ablations")
        self.assertTrue((self.group / "ablations").is_dir())
        self.assertFalse((self.group / "ablations" / "_base.yaml").exists())

    def test_nested_bases_are_templates_not_group_defaults(self):
        nested = self.group / "experiment" / "ablations"
        nested.mkdir(parents=True)
        (nested / "_base.yaml").write_text("seed: 99\n", encoding="utf-8")

        for root in (self.root, self.exp_root):
            with self.subTest(root=root):
                widget = ConfigTreeWidget(root_dir=root, mode="experiments")
                group = widget.tree.topLevelItem(0)
                self.assertEqual(group.child(0).text(0), "group defaults")
                folder = group.child(1)
                nested_folder = folder.child(0)
                self.assertFalse(folder.data(0, Qt.ItemDataRole.UserRole)["is_group"])
                self.assertFalse(nested_folder.data(0, Qt.ItemDataRole.UserRole)["is_group"])
                template = nested_folder.child(0)
                self.assertEqual(template.text(0), "_base")
                self.assertFalse(template.data(0, Qt.ItemDataRole.UserRole)["is_base"])
                widget.close()

    def test_nested_experiments_use_top_level_group_default_method(self):
        from frontend.config_viewer import ConfigViewer

        self.base.write_text("paradigm: online_rl\nmethods:\n  group_method:\n    model: dnn\n", encoding="utf-8")
        nested = self.group / "ablations" / "deep"
        nested.mkdir(parents=True)
        local_base = nested / "_base.yaml"
        local_base.write_text("methods:\n  folder_method:\n    model: cnn\n", encoding="utf-8")
        exp = nested / "trial.yaml"
        exp.write_text("experiment_id: trial\n", encoding="utf-8")

        viewer = ConfigViewer()
        with patch("frontend.config_viewer.config_tree", return_value=ConfigTree(self.root)):
            viewer.load_file(exp, "experiment/cartpole/ablations/deep/trial.yaml")
            self.assertEqual(viewer._get_group_default_method(), ("group_method", {"model": "dnn"}))
            name = viewer.add_method(name="trial_method")
            self.assertEqual(name, "trial_method")
            viewer.save_to_disk()
            self.assertEqual(yaml.safe_load(exp.read_text())["methods"][name], {"model": "dnn"})
            viewer.load_file(local_base, "experiment/cartpole/ablations/deep/_base.yaml")
            self.assertFalse(viewer._is_base_config())
            self.assertEqual(viewer.file_title.text(), "_base")
        viewer.close()

    def test_create_subfolder_rejects_empty_or_invalid(self):
        rel, err = self.widget.create_dir("experiment/cartpole", "")
        self.assertIsNone(rel)
        self.assertIn("empty", err)

        rel, err = self.widget.create_dir("experiment/cartpole", "invalid/name")
        self.assertIsNone(rel)
        self.assertIn("slashes", err)

    def test_create_subfolder_rejects_existing(self):
        (self.group / "existing").mkdir()
        rel, err = self.widget.create_dir("experiment/cartpole", "existing")
        self.assertIsNone(rel)
        self.assertIn("already exists", err)

    def test_rename_subfolder(self):
        (self.group / "old_folder").mkdir()
        renamed_signals = []
        self.widget.dir_renamed.connect(lambda o, n: renamed_signals.append((o, n)))

        new_rel, err = self.widget.rename_dir("experiment/cartpole/old_folder", "new_folder")
        self.assertIsNone(err)
        self.assertEqual(new_rel, "experiment/cartpole/new_folder")
        self.assertFalse((self.group / "old_folder").exists())
        self.assertTrue((self.group / "new_folder").exists())
        self.assertEqual(renamed_signals, [("experiment/cartpole/old_folder", "experiment/cartpole/new_folder")])

    def test_delete_subfolder(self):
        sub = self.group / "to_delete"
        sub.mkdir()
        (sub / "nested.yaml").write_text("experiment_id: nested\n", encoding="utf-8")

        deleted_signals = []
        self.widget.dir_deleted.connect(lambda rel: deleted_signals.append(rel))

        removed, err = self.widget.delete_dir("experiment/cartpole/to_delete")
        self.assertTrue(removed)
        self.assertIsNone(err)
        self.assertFalse(sub.exists())
        self.assertEqual(deleted_signals, ["experiment/cartpole/to_delete"])

    def test_write_experiment_into_subfolder(self):
        created_rel = self.widget._write_experiment(
            "cartpole", "sweep1.yaml", "sweep1", subpath="sweeps"
        )
        self.assertEqual(created_rel, "experiment/cartpole/sweeps/sweep1.yaml")
        self.assertTrue((self.group / "sweeps" / "sweep1.yaml").exists())

    def test_config_model_discovers_nested_experiments(self):
        sub = self.group / "category" / "subcategory"
        sub.mkdir(parents=True)
        nested_exp = sub / "deep_test.yaml"
        nested_exp.write_text("experiment_id: deep_test\nseed: 123\n", encoding="utf-8")

        tree = ConfigTree(self.root)
        self.assertIn("cartpole/category/subcategory/deep_test", tree.experiments)
        exp = tree.experiments["cartpole/category/subcategory/deep_test"]
        self.assertEqual(exp.group, "cartpole")
        self.assertEqual(exp.name, "cartpole/category/subcategory/deep_test")

    def test_group_defaults_pinned_above_folders(self):
        """Group defaults (_base.yaml) should be pinned to the top of the group above any subfolders and files."""
        # Create a subfolder inside cartpole
        (self.group / "ablations").mkdir()
        (self.group / "ablations" / "ablation_exp.yaml").write_text("experiment_id: ablation_exp\n", encoding="utf-8")

        self.widget.populate()
        cartpole_item = self.widget.tree.topLevelItem(0)
        self.assertEqual(cartpole_item.text(0), "cartpole")
        self.assertGreaterEqual(cartpole_item.childCount(), 3)

        # First child must be group defaults (_base.yaml)
        self.assertEqual(cartpole_item.child(0).text(0), "group defaults")
        # Second child must be the subfolder ablations
        self.assertEqual(cartpole_item.child(1).text(0), "ablations")
        # Third child must be the experiment file quick_test
        self.assertEqual(cartpole_item.child(2).text(0), "quick_test")

    def test_subfolder_uses_folder_icon(self):
        """Subfolders must display a folder icon, not the storage/database icon."""
        (self.group / "ablations").mkdir()
        self.widget.populate()

        cartpole_item = self.widget.tree.topLevelItem(0)
        subfolder_item = cartpole_item.child(1)
        self.assertEqual(subfolder_item.text(0), "ablations")

        expected_folder_icon = svg_icon("folder", TOOL_ICON_COLORS)
        database_icon = svg_icon("storage", TOOL_ICON_COLORS)

        item_pixmap = subfolder_item.icon(0).pixmap(16, 16).toImage()
        folder_pixmap = expected_folder_icon.pixmap(16, 16).toImage()
        storage_pixmap = database_icon.pixmap(16, 16).toImage()

        self.assertEqual(item_pixmap, folder_pixmap)
        self.assertNotEqual(item_pixmap, storage_pixmap)

        # After refresh_icons(), it should still be folder icon
        self.widget.refresh_icons()
        refreshed_pixmap = subfolder_item.icon(0).pixmap(16, 16).toImage()
        self.assertEqual(refreshed_pixmap, folder_pixmap)


    def test_header_buttons_split(self):
        """Header buttons must be split into new group, new subfolder, and new experiment with correct icons."""
        self.assertIsNotNone(self.widget.btn_new_group)
        self.assertIsNotNone(self.widget.btn_new_subfolder)
        self.assertIsNotNone(self.widget.btn_new_experiment)
        self.assertEqual(self.widget.btn_new_group.property("svg_icon"), "new_folder")
        self.assertEqual(self.widget.btn_new_subfolder.property("svg_icon"), "new_subfolder")
        self.assertEqual(self.widget.btn_new_experiment.property("svg_icon"), "new_file")

    def test_inline_group_creation_cancel_on_empty(self):
        """Canceling or blurring with empty name removes placeholder without creating a group folder."""
        self.widget.create_new_group_inline()
        self.assertIsNotNone(self.widget._active_inline_item)
        editor = self.widget.tree.itemWidget(self.widget._active_inline_item, 0)
        self.assertIsNotNone(editor)
        self.assertEqual(editor.text(), "")

        # Escape cancels inline group
        editor.escape_pressed.emit()
        self.assertIsNone(self.widget._active_inline_item)
        self.assertFalse((self.root / "experiment" / "new_group").exists())

    def test_inline_group_creation_success_creates_empty_defaults(self):
        """Committing valid group name creates empty plain _base.yaml and selects it."""
        selected_signals = []
        self.widget.file_selected.connect(lambda p, r: selected_signals.append((p, r)))

        self.widget.create_new_group_inline()
        item = self.widget._active_inline_item
        editor = self.widget.tree.itemWidget(item, 0)
        editor.setText("robotics")
        editor.returnPressed.emit()

        base_file = self.root / "experiment" / "robotics" / "_base.yaml"
        self.assertTrue(base_file.is_file())
        content = base_file.read_text(encoding="utf-8")
        self.assertIn("paradigm: null", content)
        self.assertIn("override /env: null", content)
        self.assertIn("seed: 1", content)
        self.assertIn("save_dataset: false", content)

        self.assertEqual(self.widget.last_focused_group, "robotics")
        self.assertEqual(self.widget.current_rel_path, "experiment/robotics/_base.yaml")
        self.assertEqual(len(selected_signals), 1)
        self.assertEqual(selected_signals[0][1], "experiment/robotics/_base.yaml")

    def test_inline_subfolder_and_experiment_creation(self):
        self.widget.prompt_new_folder("experiment/cartpole")
        item = self.widget._active_inline_item
        self.assertEqual(item.parent().text(0), "cartpole")
        editor = self.widget.tree.itemWidget(item, 0)
        self.assertEqual(editor.text(), "")
        editor.setText("sweeps")
        editor.returnPressed.emit()
        self.assertTrue((self.group / "sweeps").is_dir())

        self.widget.prompt_new_in_folder("experiment/cartpole/sweeps")
        item = self.widget._active_inline_item
        self.assertEqual(item.parent().text(0), "sweeps")
        editor = self.widget.tree.itemWidget(item, 0)
        self.assertEqual(editor.text(), "")
        editor.setText("trial")
        editor.returnPressed.emit()
        self.assertTrue((self.group / "sweeps" / "trial.yaml").is_file())
        self.assertEqual(self.widget.current_rel_path, "experiment/cartpole/sweeps/trial.yaml")

    def test_inline_child_escape_cancels(self):
        self.widget.prompt_new_folder("experiment/cartpole")
        item = self.widget._active_inline_item
        editor = self.widget.tree.itemWidget(item, 0)
        editor.escape_pressed.emit()
        self.assertIsNone(self.widget._active_inline_item)
        self.assertFalse((self.group / "new_folder").exists())

    def test_new_experiment_generated_from_defaults(self):
        """Experiments inherit defined params and prompt null for missing ones."""
        # 1. Cartpole defines paradigm: online_rl and env: cartpole
        exp_rel = self.widget._write_experiment_from_defaults("cartpole", "test_cartpole.yaml", "test_cartpole")
        self.assertEqual(exp_rel, "experiment/cartpole/test_cartpole.yaml")
        exp_text = (self.root / "experiment" / "cartpole" / "test_cartpole.yaml").read_text(encoding="utf-8")
        self.assertIn("defaults:\n  - cartpole/_base", exp_text)
        self.assertNotIn("paradigm:", exp_text)  # inherited
        self.assertNotIn("env:", exp_text)       # inherited

        # 2. Group with null defaults prompts paradigm and env with null
        empty_group_dir = self.root / "experiment" / "empty_group"
        empty_group_dir.mkdir(parents=True, exist_ok=True)
        (empty_group_dir / "_base.yaml").write_text("paradigm: null\ndefaults:\n  - override /env: null\n", encoding="utf-8")

        exp_rel2 = self.widget._write_experiment_from_defaults("empty_group", "test_empty.yaml", "test_empty")
        exp_text2 = (self.root / "experiment" / "empty_group" / "test_empty.yaml").read_text(encoding="utf-8")
        self.assertIn("defaults:\n  - empty_group/_base", exp_text2)
        self.assertIn("paradigm: null", exp_text2)  # prompted
        self.assertIn("env: null", exp_text2)       # prompted

    def test_config_viewer_group_defaults_boxes_and_no_strip(self):
        """ConfigViewer displays Group Paradigm & Environment card on _base.yaml and preserves paradigm."""
        from frontend.config_viewer import ConfigBox, ConfigViewer
        base_file = self.root / "experiment" / "cartpole" / "_base.yaml"
        base_file.write_text(
            "# @package _global_\ndefaults:\n  - override /env: cartpole\nparadigm: online_rl\nseed: 1\n",
            encoding="utf-8"
        )
        viewer = ConfigViewer()
        env_dir = self.root / "env"
        env_dir.mkdir()
        (env_dir / "cartpole.yaml").write_text(
            "compatible_paradigms: [online_rl]\nenv_id: CartPole-v1\n", encoding="utf-8"
        )
        tree = ConfigTree(self.root)
        with patch("frontend.config_viewer.config_tree", return_value=tree):
            viewer.load_file(base_file, "experiment/cartpole/_base.yaml")

        self.assertEqual(viewer.file_title.text(), "Group Defaults")
        self.assertEqual(viewer.group_badge.text(), "cartpole")

        cards = viewer.findChildren(ConfigBox)
        self.assertGreaterEqual(len(cards), 2)
        # First card is Group Paradigm & Environment containing Paradigm and Environment labels
        labels = [lbl.text() for lbl in cards[0].findChildren(QLabel)]
        self.assertIn("Paradigm", labels)
        self.assertIn("Environment", labels)
        self.assertEqual(viewer.combo_paradigm.currentText(), "online_rl")
        self.assertEqual(viewer.combo_env.currentText(), "cartpole")

        # Saving does not strip paradigm or defaults
        with patch("frontend.config_viewer.config_tree", return_value=tree):
            viewer.save_to_disk()
        saved = (self.root / "experiment" / "cartpole" / "_base.yaml").read_text(encoding="utf-8")
        self.assertIn("paradigm: online_rl", saved)
        self.assertIn("override /env: cartpole", saved)

    def test_config_viewer_prompts_required_in_experiment(self):
        """Experiments with null paradigm and env show prompt controls in budget card."""
        from frontend.config_viewer import ConfigBox, ConfigViewer
        exp_file = self.root / "experiment" / "cartpole" / "prompt_exp.yaml"
        exp_file.write_text(
            "experiment_id: prompt_exp\nparadigm: null\nenv: null\ntotal_timesteps: 5000\n",
            encoding="utf-8"
        )
        viewer = ConfigViewer()
        viewer.load_file(exp_file, "experiment/cartpole/prompt_exp.yaml")

        cards = viewer.findChildren(ConfigBox)
        self.assertEqual(len(cards), 2)  # Budget & Methods
        budget_card = cards[0]
        labels = [lbl.text() for lbl in budget_card.findChildren(QLabel)]
        self.assertIn("Paradigm", labels)
        self.assertIn("Environment / Task", labels)


if __name__ == "__main__":
    unittest.main()

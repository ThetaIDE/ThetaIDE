"""Unit tests for the ConfigTreeWidget header toolbar:
- 2-row layout with left and right groups per row
- 4 creation buttons positioned on row 2 left
- Draggable buttons that can be reordered within groups and moved between groups
- Persistence and resetting to default layout
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PyQt6.QtCore import QPoint, QSettings, Qt
from PyQt6.QtWidgets import QApplication

app = QApplication.instance() or QApplication(sys.argv[:1])

from frontend.config_tree import ConfigTreeWidget, DraggableToolButton, TreeHeaderWidget


class TestFileTreeToolbar(unittest.TestCase):
    def setUp(self):
        QSettings("ThetaIDE", "ThetaIDE").clear()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name) / "src/usr/config"
        (self.root / "experiment" / "cartpole").mkdir(parents=True)
        (self.root / "experiment" / "cartpole" / "exp1.yaml").write_text("experiment_id: exp1\n", encoding="utf-8")
        self.tree_exp = ConfigTreeWidget(root_dir=self.root, mode="experiments")

    def tearDown(self):
        self.tree_exp.close()
        self.temp_dir.cleanup()
        QSettings("ThetaIDE", "ThetaIDE").clear()


    def test_buttons_are_draggable(self):
        """Verify toolbar buttons are DraggableToolButton instances."""
        for b in [
            self.tree_exp.btn_new_group,
            self.tree_exp.btn_new_subfolder,
            self.tree_exp.btn_new_experiment,
            self.tree_exp.btn_duplicate,
            self.tree_exp.btn_refresh,
            self.tree_exp.btn_launch,
            self.tree_exp.btn_delete,
        ]:
            self.assertIsInstance(b, DraggableToolButton)

    def test_reorder_buttons_within_group(self):
        """Verify moving a button within the same group reorders it."""
        hw = self.tree_exp.header_widget
        # Move 'copy' (index 3) to the beginning (index 0) of row2_left
        hw.move_button("copy", "row2_left", 0)

        r2_left_ids = [b.button_id for b in hw.header_layout.groups["row2_left"]]
        self.assertEqual(r2_left_ids, ["copy", "new_folder", "new_subfolder", "new_file"])

    def test_move_button_between_groups(self):
        """Verify moving a button between different groups."""
        hw = self.tree_exp.header_widget
        # Move 'copy' from row2_left to row1_right at index 0
        hw.move_button("copy", "row1_right", 0)

        r2_left_ids = [b.button_id for b in hw.header_layout.groups["row2_left"]]
        self.assertNotIn("copy", r2_left_ids)
        self.assertEqual(r2_left_ids, ["new_folder", "new_subfolder", "new_file"])

        r1_right_ids = [b.button_id for b in hw.header_layout.groups["row1_right"]]
        self.assertIn("copy", r1_right_ids)
        self.assertEqual(r1_right_ids, ["copy", "launch", "export", "raw_yaml"])


    def test_results_toolbar_reorders_persists_and_resets(self):
        from frontend.results.explorer import RunExplorer

        explorer = RunExplorer()
        try:
            self.assertIsInstance(explorer.btn_archive, DraggableToolButton)
            self.assertIsInstance(explorer.btn_delete, DraggableToolButton)
            explorer.header_widget.move_button("archive", "row1_left", 0)
            restored = RunExplorer()
            try:
                self.assertEqual(restored.header_layout.groups["row1_left"][0].button_id, "archive")
                restored.reset_toolbar_layout()
                self.assertEqual([b.button_id for b in restored.header_layout.groups["row2_left"]],
                                 ["archive", "trash"])
            finally:
                restored.close()
        finally:
            explorer.close()


    def test_components_mode_default_layout(self):
        """Verify components mode has creation buttons on row 2 left."""
        tree_comp = ConfigTreeWidget(root_dir=self.root, mode="components")
        try:
            hw = tree_comp.header_widget
            groups = hw.header_layout.groups
            self.assertEqual([b.button_id for b in groups["row1_left"]], ["reload", "collapse_all", "expand_all", "search"])
            self.assertEqual([b.button_id for b in groups["row1_right"]], ["plus", "hub", "raw_yaml"])
            self.assertEqual([b.button_id for b in groups["row2_left"]], ["copy", "deep_copy"])
            self.assertEqual([b.button_id for b in groups["row2_right"]], ["trash"])
        finally:
            tree_comp.close()

    def test_config_toolbar_and_context_menu_use_the_same_delete_target(self):
        (self.root / "agent").mkdir()
        (self.root / "agent" / "ppo.yaml").touch()
        components = ConfigTreeWidget(root_dir=self.root, mode="components")
        self.addCleanup(components.close)
        for widget, path in ((self.tree_exp, "experiment/cartpole/exp1.yaml"), (components, "agent/ppo.yaml")):
            with self.subTest(mode=widget.mode), patch.object(widget, "prompt_delete") as delete:
                self.assertFalse(widget.btn_delete.isEnabled())
                widget.select_file(path, emit_signal=False)
                self.assertTrue(widget.btn_delete.isEnabled())
                widget.btn_delete.click()
                action = widget.build_context_menu(widget.tree.currentItem()).actions()[-1]
                self.assertEqual(action.text(), "Delete…")
                action.trigger()
                self.assertEqual([call.args for call in delete.call_args_list], [(path,), (path,)])


if __name__ == "__main__":
    unittest.main()

"""Tests for deleting and renaming experiment configs.

These files are tracked in git, and a group's _base.yaml is inherited by every
experiment beside it, so the destructive paths need guarding rather than just
wiring up.
"""

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from frontend.config_model import deletion_blocked_reason, rename_experiment_text  # noqa: E402

try:
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(sys.argv[:1])
    from frontend.config_tree import ConfigTreeWidget

    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False


BASE = "# @package _global_\ndefaults:\n  - override /env: cartpole\n\nparadigm: online_rl\n"
EXPERIMENT = "# @package _global_\ndefaults:\n  - demo/_base\n\nexperiment_id: first\nseed: 42\n"


class TestDeletionGuard(unittest.TestCase):
    """deletion_blocked_reason is the rule; it needs no Qt."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.group = Path(self.tmp.name) / "experiment" / "demo"
        self.group.mkdir(parents=True)
        self.base = self.group / "_base.yaml"
        self.base.write_text(BASE)
        self.experiment = self.group / "first.yaml"
        self.experiment.write_text(EXPERIMENT)

    def test_a_plain_experiment_is_safe_to_delete(self):
        self.assertIsNone(deletion_blocked_reason(self.experiment))

    def test_base_is_protected_while_experiments_inherit_it(self):
        reason = deletion_blocked_reason(self.base)
        self.assertIsNotNone(reason)
        self.assertIn("_base.yaml", reason)

    def test_the_reason_names_what_would_break(self):
        self.assertIn("first", deletion_blocked_reason(self.base))

    def test_base_may_go_once_the_group_is_empty(self):
        self.experiment.unlink()
        self.assertIsNone(deletion_blocked_reason(self.base))

    def test_several_dependants_are_counted(self):
        (self.group / "second.yaml").write_text(EXPERIMENT)
        self.assertIn("2 experiment", deletion_blocked_reason(self.base))

    def test_a_missing_file_is_reported_rather_than_deleted(self):
        self.assertIsNotNone(deletion_blocked_reason(self.group / "absent.yaml"))


class TestRenameKeepsExperimentIdInStep(unittest.TestCase):
    def test_experiment_id_follows_the_new_name(self):
        self.assertIn("experiment_id: second", rename_experiment_text(EXPERIMENT, "second"))

    def test_the_old_id_is_gone(self):
        self.assertNotIn("experiment_id: first", rename_experiment_text(EXPERIMENT, "second"))

    def test_the_rest_of_the_file_is_untouched(self):
        renamed = rename_experiment_text(EXPERIMENT, "second")
        self.assertIn("# @package _global_", renamed)
        self.assertIn("- demo/_base", renamed)
        self.assertIn("seed: 42", renamed)

    def test_comments_and_key_order_survive(self):
        text = "# a note\nexperiment_id: first\n# another note\nseed: 7\n"
        renamed = rename_experiment_text(text, "second")
        self.assertEqual(renamed, "# a note\nexperiment_id: second\n# another note\nseed: 7\n")

    def test_indented_experiment_id_is_handled(self):
        self.assertIn("  experiment_id: second", rename_experiment_text("  experiment_id: first\n", "second"))

    def test_a_file_without_experiment_id_is_left_alone(self):
        text = "seed: 1\n"
        self.assertEqual(rename_experiment_text(text, "second"), text)

    def test_only_the_first_occurrence_is_rewritten(self):
        text = "experiment_id: first\nnested:\n  experiment_id: first\n"
        renamed = rename_experiment_text(text, "second")
        self.assertEqual(renamed.count("experiment_id: second"), 1)

    def test_result_is_still_valid_yaml(self):
        self.assertEqual(yaml.safe_load(rename_experiment_text(EXPERIMENT, "second"))["experiment_id"], "second")


@unittest.skipIf(not HAS_PYQT6, "PyQt6 not installed in current environment")
class TestTreeDeleteAndRename(unittest.TestCase):
    """The widget paths, driven against a throwaway config tree."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name) / "src/usr/config"
        group = root / "experiment" / "demo"
        group.mkdir(parents=True)
        (group / "_base.yaml").write_text(BASE)
        (group / "first.yaml").write_text(EXPERIMENT)
        self.root = root
        self.widget = ConfigTreeWidget(root_dir=root, mode="experiments")
        self.addCleanup(self._destroy_widget)

    def _destroy_widget(self):
        """close() only hides a widget; without deleteLater the instances pile
        up in the application and later tests slow to a crawl."""
        self.widget.close()
        self.widget.setParent(None)
        self.widget.deleteLater()
        QApplication.processEvents()

    def test_delete_removes_the_file(self):
        removed, error = self.widget.delete_config("experiment/demo/first.yaml")
        self.assertTrue(removed, error)
        self.assertFalse((self.root / "experiment" / "demo" / "first.yaml").exists())

    def test_delete_refuses_a_base_other_experiments_need(self):
        removed, error = self.widget.delete_config("experiment/demo/_base.yaml")
        self.assertFalse(removed)
        self.assertIn("_base.yaml", error)
        self.assertTrue((self.root / "experiment" / "demo" / "_base.yaml").exists())

    def test_delete_clears_the_selection_when_it_was_the_open_file(self):
        self.widget.current_rel_path = "experiment/demo/first.yaml"
        self.widget.delete_config("experiment/demo/first.yaml")
        self.assertIsNone(self.widget.current_rel_path)

    def test_rename_moves_the_file(self):
        self.widget.rename_config("experiment/demo/first.yaml", "renamed")
        group = self.root / "experiment" / "demo"
        self.assertFalse((group / "first.yaml").exists())
        self.assertTrue((group / "renamed.yaml").exists())

    def test_rename_updates_experiment_id_on_disk(self):
        self.widget.rename_config("experiment/demo/first.yaml", "renamed")
        data = yaml.safe_load((self.root / "experiment" / "demo" / "renamed.yaml").read_text())
        self.assertEqual(data["experiment_id"], "renamed")

    def test_rename_accepts_a_name_that_already_ends_in_yaml(self):
        self.widget.rename_config("experiment/demo/first.yaml", "renamed.yaml")
        self.assertTrue((self.root / "experiment" / "demo" / "renamed.yaml").exists())

    def test_rename_refuses_to_overwrite(self):
        shutil.copy(self.root / "experiment" / "demo" / "first.yaml",
                    self.root / "experiment" / "demo" / "taken.yaml")
        result, error = self.widget.rename_config("experiment/demo/first.yaml", "taken")
        self.assertIsNone(result)
        self.assertIn("already exists", error)
        self.assertTrue((self.root / "experiment" / "demo" / "first.yaml").exists())

    def test_rename_to_the_same_name_is_a_no_op(self):
        result, error = self.widget.rename_config("experiment/demo/first.yaml", "first")
        self.assertIsNone(error)
        self.assertEqual(result, "experiment/demo/first.yaml")
        self.assertTrue((self.root / "experiment" / "demo" / "first.yaml").exists())

    def test_rename_ignores_an_empty_name(self):
        result, error = self.widget.rename_config("experiment/demo/first.yaml", "   ")
        self.assertIsNone(result)
        self.assertTrue(error)
        self.assertTrue((self.root / "experiment" / "demo" / "first.yaml").exists())

    def test_base_may_be_deleted_once_its_group_is_empty(self):
        self.widget.delete_config("experiment/demo/first.yaml")
        removed, error = self.widget.delete_config("experiment/demo/_base.yaml")
        self.assertTrue(removed, error)


if __name__ == "__main__":
    unittest.main()

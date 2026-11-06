"""Unit and integration tests for Theta Hub component lifecycle and file tree synchronization.

Verifies:
1. File Tree Deletion Interception (Option 1): Component discovery and metadata-driven uninstallation.
2. Component configs are hidden on uninstall and restored on reinstall.
3. Dialog instantiation and choice contracts.
"""
import os
import json
import shutil
import tempfile
import unittest
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "offscreen"
from PyQt6.QtWidgets import QApplication, QLabel
app = QApplication.instance() or QApplication([])

from frontend.hub.installer import HubInstaller
from frontend.hub.models import HubComponent
from frontend.hub.dialog import UninstallComponentDialog
from frontend.config_tree import ComponentDeleteDialog


class TestHubLifecycleSync(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="theta_test_sync_"))
        self.workspace_dir = self.temp_dir / "workspace"
        self.data_dir = self.temp_dir / "data"
        self.workspace_dir.mkdir(parents=True)
        self.data_dir.mkdir(parents=True)

        self.installer = HubInstaller(
            workspace_dir=self.workspace_dir,
            data_dir=self.data_dir,
        )

        # Create a mock installed component
        self.comp_dir = self.workspace_dir / "in" / "envs" / "mock_game"
        self.comp_dir.mkdir(parents=True)
        self.config_dir = self.workspace_dir / "in" / "config" / "env" / "mock_game"
        self.config_dir.mkdir(parents=True)

        self.cfg1 = self.config_dir / "level1.yaml"
        self.cfg1.write_text("name: mock_game/level1\n", encoding="utf-8")
        self.cfg2 = self.config_dir / "level2.yaml"
        self.cfg2.write_text("name: mock_game/level2\n", encoding="utf-8")

        self.meta_file = self.comp_dir / ".theta_component.json"
        self.meta_file.write_text(json.dumps({
            "id": "mock-game",
            "name": "Mock Game Suite",
            "kind": "env",
            "version": "1.0.0",
            "config_paths": [
                "in/config/env/mock_game/level1.yaml",
                "in/config/env/mock_game/level2.yaml",
            ]
        }), encoding="utf-8")

        self.component = HubComponent(
            id="mock-game",
            name="Mock Game Suite",
            kind="env",
            version="1.0.0",
            target_path="in/envs/mock_game",
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_has_configs_detection(self):
        """has_configs correctly detects presence of config files on disk."""
        self.assertTrue(self.installer.has_configs(self.component))

        # Remove config files
        self.cfg1.unlink()
        self.cfg2.unlink()
        self.assertFalse(self.installer.has_configs(self.component))

    def test_find_component_for_source_path(self):
        """find_component_for_path detects component by source directory."""
        meta = self.installer.find_component_for_path(self.comp_dir)
        self.assertIsNotNone(meta)
        self.assertEqual(meta["id"], "mock-game")
        self.assertEqual(meta["name"], "Mock Game Suite")

    def test_find_component_for_config_dir_path(self):
        """find_component_for_path detects component by config directory (Option 1)."""
        meta = self.installer.find_component_for_path(self.config_dir)
        self.assertIsNotNone(meta)
        self.assertEqual(meta["id"], "mock-game")

    def test_find_component_for_config_file_path(self):
        """find_component_for_path detects component by individual config file."""
        meta = self.installer.find_component_for_path(self.cfg1)
        self.assertIsNotNone(meta)
        self.assertEqual(meta["id"], "mock-game")

    def test_find_component_returns_none_for_unrelated_path(self):
        """find_component_for_path returns None for regular user folders."""
        other_dir = self.workspace_dir / "in" / "config" / "experiment" / "my_exp"
        other_dir.mkdir(parents=True)
        meta = self.installer.find_component_for_path(other_dir)
        self.assertIsNone(meta)

    def test_installed_files_are_distinct_from_user_copies(self):
        meta = json.loads(self.meta_file.read_text())
        meta["installed_from"] = "https://example.test/component.zip"
        meta["user_config_paths"] = ["in/config/env/mock_game/custom.yaml"]
        self.meta_file.write_text(json.dumps(meta))
        source = self.comp_dir / "runner.py"
        source.write_text("pass\n")
        user_copy = self.config_dir / "custom.yaml"
        user_copy.write_text("name: custom\n")
        self.assertTrue(self.installer.is_installed_file(source))
        self.assertTrue(self.installer.is_installed_file(self.cfg1))
        self.assertTrue(self.installer.is_installed_file(self.config_dir))
        self.assertFalse(self.installer.is_installed_file(user_copy))
        self.installer.protect_installed_components()
        self.assertEqual(source.stat().st_mode & 0o222, 0)
        self.assertEqual(self.cfg1.stat().st_mode & 0o222, 0)
        self.assertNotEqual(user_copy.stat().st_mode & 0o200, 0)
        meta.pop("installed_from")
        meta["origin"] = "local"
        self.meta_file.chmod(self.meta_file.stat().st_mode | 0o200)
        self.meta_file.write_text(json.dumps(meta))
        self.assertFalse(self.installer.is_installed_file(source))
        self.assertFalse(self.installer.is_installed_file(self.cfg1))

    def test_uninstall_hides_configs_and_reinstall_restores_them(self):
        """Uninstall hides configs outside the active tree and reinstall restores them."""
        ok = self.installer.uninstall(self.component)
        self.assertTrue(ok)

        # Source code is removed
        self.assertFalse(self.comp_dir.exists())

        # Active configs are hidden, while user data remains in IDE-local storage.
        self.assertFalse(self.cfg1.exists())
        self.assertFalse(self.cfg2.exists())
        self.assertFalse(self.config_dir.exists())
        retained1 = self.data_dir / "retained_configs" / "mock-game" / "in/config/env/mock_game/level1.yaml"
        retained2 = self.data_dir / "retained_configs" / "mock-game" / "in/config/env/mock_game/level2.yaml"
        self.assertTrue(retained1.is_file())
        self.assertTrue(retained2.is_file())

        restored_paths = []
        self.installer._restore_retained_configs("mock-game", restored_paths, {})
        self.assertEqual(set(restored_paths), {
            "in/config/env/mock_game/level1.yaml",
            "in/config/env/mock_game/level2.yaml",
        })
        self.assertTrue(self.cfg1.is_file())
        self.assertTrue(self.cfg2.is_file())

    def test_uninstall_removing_configs(self):
        """Uninstall removes configs from the active tree."""
        ok = self.installer.uninstall(self.component)
        self.assertTrue(ok)

        # Source code is removed
        self.assertFalse(self.comp_dir.exists())

        # Configs and empty parent folder are cleaned up!
        self.assertFalse(self.cfg1.exists())
        self.assertFalse(self.cfg2.exists())
        self.assertFalse(self.config_dir.exists())

    def test_uninstall_by_metadata(self):
        """uninstall_by_metadata cleanly uninstalls when triggered from file tree (Option 1)."""
        meta = self.installer.find_component_for_path(self.config_dir)
        self.assertIsNotNone(meta)

        ok = self.installer.uninstall_by_metadata(meta)
        self.assertTrue(ok)
        self.assertFalse(self.comp_dir.exists())
        self.assertFalse(self.config_dir.exists())

    def test_dialog_instantiation(self):
        """Delete and uninstall dialogs initialize with the new lifecycle text."""
        del_dlg = ComponentDeleteDialog("in/config/env/mock_game", "Mock Game Suite")
        self.assertEqual(del_dlg.choice, "cancel")

        uninst_dlg = UninstallComponentDialog("Mock Game Suite")
        self.assertIn("restored", " ".join(label.text() for label in uninst_dlg.findChildren(QLabel)))


if __name__ == "__main__":
    unittest.main()


def test_reinstall_merges_new_defaults_without_losing_user_values_or_comments(tmp_path):
    from frontend.hub.installer import _add_missing_yaml_defaults
    import yaml
    retained = "# user settings\nmodel:\n  width: 64 # chosen\n  nested:\n    keep: true\n"
    packaged = "model:\n  width: 32\n  nested:\n    keep: false\n    new: 4\n  logic_lr: null\nnew_top: 3\n"
    result = _add_missing_yaml_defaults(retained, packaged)
    assert "# user settings" in result and "# chosen" in result
    assert yaml.safe_load(result) == {"model": {"width": 64, "nested": {"keep": True, "new": 4}, "logic_lr": None}, "new_top": 3}


def test_update_and_reinstall_preserve_user_config_and_add_defaults(tmp_path):
    from zipfile import ZipFile
    import yaml
    from frontend.hub.models import ReleaseInfo

    workspace, data = tmp_path / "workspace", tmp_path / "data"
    workspace.mkdir()
    data.mkdir()
    installer = HubInstaller(workspace, data)
    def release(version, config):
        archive = tmp_path / f"{version}.zip"
        with ZipFile(archive, "w") as z:
            z.writestr("main.py", "# component\n")
            z.writestr("config.yaml", config)
        return ReleaseInfo(tag=version, url=archive.as_uri(), sha256="")
    component = HubComponent(id="example", name="Example", kind="model", version="1.0.0", releases={
        "1.0.0": release("1.0.0", "width: 32\n"),
        "1.1.0": release("1.1.0", "width: 32\nlogic_lr: null\n"),
        "1.2.0": release("1.2.0", "width: 32\nlogic_lr: null\nblender_lr: null\n"),
    })
    path = workspace / "in/config/model/example.yaml"
    assert installer.install(component, "1.0.0")
    # Simulate an older installation edited before read-only protection.
    path.chmod(path.stat().st_mode | 0o200)
    path.write_text("# personal\nwidth: 64\n")
    assert installer.install(component, "1.1.0")
    assert yaml.safe_load(path.read_text()) == {"width": 64, "logic_lr": None}
    assert "# personal" in path.read_text()
    assert installer.uninstall(component)
    assert installer.install(component, "1.2.0")
    assert yaml.safe_load(path.read_text()) == {"width": 64, "logic_lr": None, "blender_lr": None}
    assert installer.uninstall(component)
    assert installer.install(component, "1.2.0")
    assert yaml.safe_load(path.read_text()) == {"width": 64, "logic_lr": None, "blender_lr": None}


def test_root_relative_reinstall_restores_retained_yaml_after_source_copy(tmp_path):
    from zipfile import ZipFile
    import yaml
    from frontend.hub.models import ReleaseInfo
    workspace, data = tmp_path / "workspace", tmp_path / "data"
    workspace.mkdir()
    data.mkdir()
    installer = HubInstaller(workspace, data)
    def release(version, text):
        archive = tmp_path / f"root-{version}.zip"
        manifest = {"id": "example", "name": "Example", "kind": "model", "version": version,
                    "config_path": "in/config/model/example.yaml"}
        with ZipFile(archive, "w") as z:
            z.writestr("in/config/model/example.yaml", text)
            z.writestr("src/usr/models/example/.theta_component.json", json.dumps(manifest))
        return ReleaseInfo(tag=version, url=archive.as_uri(), sha256="")
    component = HubComponent(id="example", name="Example", kind="model", version="1.0.0", releases={
        "1.0.0": release("1.0.0", "width: 32\n"),
        "1.1.0": release("1.1.0", "width: 32\nlogic_lr: null\n"),
    })
    path = workspace / "in/config/model/example.yaml"
    assert installer.install(component, "1.0.0")
    path.chmod(path.stat().st_mode | 0o200)  # Simulate a legacy edit.
    path.write_text("width: 64\n")
    assert installer.uninstall(component)
    assert installer.install(component, "1.1.0")
    assert yaml.safe_load(path.read_text()) == {"width": 64, "logic_lr": None}

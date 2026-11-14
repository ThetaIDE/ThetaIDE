"""Unit and integration tests for ThetaIDE Plugin System."""
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "offscreen"

try:
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QApplication
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True)
    app = QApplication.instance() or QApplication(sys.argv[:1])
    from frontend.app import Window
    from frontend.plugins import Plugin, PluginContext, PluginManager, PluginManifest
    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False


SAMPLE_PLUGIN_JSON = {
    "id": "sample_plugin",
    "name": "Sample Plugin",
    "version": "0.1.0",
    "description": "Dynamic fixture plugin for testing ThetaIDE plugin system.",
    "author": "ThetaIDE Test",
    "default_enabled": False,
    "entry_point": "SamplePlugin",
}

SAMPLE_PLUGIN_SOURCE = """
from PyQt6.QtWidgets import QLabel
from src.app.api.desktop import Plugin

class SamplePlugin(Plugin):
    def activate(self, context):
        self.context = context
        self.label = QLabel("Sample Content")
        context.add_sidebar_tab(
            tab_id="sample_plugin",
            widget=self.label,
            title="Sample Tab",
            short_label="Sample",
        )

    def deactivate(self):
        if self.context:
            self.context.remove_sidebar_tab("sample_plugin")
            self.context = None
"""


@unittest.skipIf(not HAS_PYQT6, "PyQt6 not installed in current environment")
class TestPluginSystem(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="theta_test_plugins_")

        # Install a dynamic fixture plugin as a native package, as the Hub would.
        self.plugin_id = "sample_plugin"
        self.workspace = Path(self.temp_dir) / "workspace"
        self.workspace.mkdir(parents=True, exist_ok=True)
        self._install_sample()

        self.window = Window(data_dir=self.temp_dir, workspace=self.workspace)

    def _install_sample(self):
        from zipfile import ZipFile
        from src.app.components.installer import HubInstaller
        from src.app.components.models import HubComponent, ReleaseInfo
        from src.app.workspace import WorkspacePaths
        meta = {"id": self.plugin_id, "name": "Sample Plugin", "kind": "plugin", "version": "0.1.0",
                "storage_version": 2, "api_version": 1,
                "payload": {"implementations": ["__init__.py"], "resources": [], "configs": {}},
                "capabilities": [{"id": self.plugin_id, "kind": "plugin", "entry_point": "__init__.py:SamplePlugin"}]}
        archive = Path(self.temp_dir) / f"{self.plugin_id}-0.1.0.zip"
        with ZipFile(archive, "w") as bundle:
            bundle.writestr(".theta_component.json", json.dumps(meta))
            bundle.writestr("plugin.json", json.dumps(SAMPLE_PLUGIN_JSON))
            bundle.writestr("__init__.py", SAMPLE_PLUGIN_SOURCE)
        component = HubComponent.from_dict(meta)
        component.releases = {"0.1.0": ReleaseInfo("0.1.0", archive.as_uri(), "")}
        installer = HubInstaller(self.workspace, WorkspacePaths(self.workspace).data_dir)
        installer.install(component, install_dependencies=False)
        self.plugin_dir = installer.resolve_target_dir(component)

    def tearDown(self):
        if hasattr(self, "window"):
            self.window.close()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_zero_bundled_plugins_shipped_with_theta(self):
        """ThetaIDE source must not ship with any bundled plugins; all plugins come from the Hub."""
        builtin_plugins_dir = Path(__file__).parent.parent / "frontend" / "plugins"
        bundled_plugins = [
            p.parent.name for p in builtin_plugins_dir.glob("*/plugin.json")
        ]
        self.assertEqual(bundled_plugins, [], "ThetaIDE source repository should ship with zero bundled plugins!")

    def test_plugin_discovered(self):
        """Plugin manifest is discovered and parsed with correct metadata."""
        manager = self.window.plugin_manager
        self.assertIn("sample_plugin", manager.manifests)
        manifest = manager.manifests["sample_plugin"]
        self.assertEqual(manifest.id, "sample_plugin")
        self.assertEqual(manifest.name, "Sample Plugin")
        self.assertFalse(manifest.default_enabled)

    def test_plugin_disabled_on_initialization(self):
        """Plugin must be disabled by default on clean initialization."""
        manager = self.window.plugin_manager
        self.assertFalse(manager.is_plugin_enabled("sample_plugin"))
        self.assertNotIn("sample_plugin", manager.instances)

        # Check UI sidebar tabs: plugin must NOT be in tabs
        self.assertNotIn("sample_plugin", self.window.tabs.tabs)
        self.assertNotIn("sample_plugin", self.window.tabs.tab_order)

        # Check toggle slider in settings
        if "sample_plugin" in self.window.plugin_sliders:
            self.assertFalse(self.window.plugin_sliders["sample_plugin"].isChecked())

    def test_enable_and_disable_plugin_lifecycle(self):
        """Enabling dynamically mounts the sidebar tab; disabling unmounts it cleanly."""
        manager = self.window.plugin_manager

        # 1. Enable plugin
        success = manager.enable_plugin("sample_plugin")
        self.assertTrue(success)
        self.assertTrue(manager.is_plugin_enabled("sample_plugin"))
        self.assertIn("sample_plugin", manager.instances)

        # Tab should now be present in SideTabs
        self.assertIn("sample_plugin", self.window.tabs.tabs)
        self.assertIn("sample_plugin", self.window.tabs.tab_order)

        # 2. Disable plugin
        success = manager.disable_plugin("sample_plugin")
        self.assertTrue(success)
        self.assertFalse(manager.is_plugin_enabled("sample_plugin"))
        self.assertNotIn("sample_plugin", manager.instances)

        # Tab should be cleanly removed from SideTabs
        self.assertNotIn("sample_plugin", self.window.tabs.tabs)
        self.assertNotIn("sample_plugin", self.window.tabs.tab_order)

    def test_persistence_of_plugin_state(self):
        """State is persisted to settings.toml and restored on reload."""
        manager = self.window.plugin_manager

        # Initially saved as disabled
        self.assertFalse(manager.is_plugin_enabled("sample_plugin"))

        # Enable and verify disk write to settings.toml
        manager.enable_plugin("sample_plugin")
        settings_file = self.window.settings_manager.workspace_settings_path
        self.assertTrue(settings_file.exists())
        self.assertIn("sample_plugin", self.window.settings_manager.plugins_enabled)

        # Simulate new Window instance loading same data directory
        new_window = Window(data_dir=self.temp_dir, workspace=self.workspace)
        self.assertTrue(new_window.plugin_manager.is_plugin_enabled("sample_plugin"))
        self.assertIn("sample_plugin", new_window.tabs.tabs)

        # Cleanup new_window
        new_window.plugin_manager.disable_plugin("sample_plugin")

    def test_settings_toggle_slider_interaction(self):
        """Toggling the slider in the Settings card enables/disables the plugin."""
        self.window.settings_view.build_pages()  # settings pages are built on first open
        self.assertIn("sample_plugin", self.window.plugin_sliders)
        slider = self.window.plugin_sliders["sample_plugin"]
        self.assertFalse(slider.isChecked())

        # Toggle to True
        slider.click()
        self.assertTrue(slider.isChecked())
        self.assertTrue(self.window.plugin_manager.is_plugin_enabled("sample_plugin"))
        self.assertIn("sample_plugin", self.window.tabs.tabs)

        # Toggle back to False
        slider.click()
        self.assertFalse(slider.isChecked())
        self.assertFalse(self.window.plugin_manager.is_plugin_enabled("sample_plugin"))
        self.assertNotIn("sample_plugin", self.window.tabs.tabs)

    def test_uninstalled_plugin_lifecycle_and_disappearance(self):
        """Uninstalling a plugin removes its directory from disk and from Settings UI."""
        self.window.settings_view.build_pages()  # settings pages are built on first open
        manager = self.window.plugin_manager
        self.assertIn("sample_plugin", manager.manifests)
        self.assertIn("sample_plugin", self.window.plugin_sliders)

        # Uninstall removes files and unregisters manifest
        manager.uninstall_plugin("sample_plugin")
        self.assertNotIn("sample_plugin", manager.manifests)
        self.assertFalse(self.plugin_dir.exists())

        # UI refresh removes it from settings menu
        self.window.refresh_plugins_ui()
        self.assertNotIn("sample_plugin", self.window.plugin_sliders)

        # Rescanning discovery does not resurrect it
        manager.discover()
        self.assertNotIn("sample_plugin", manager.manifests)

        # Simulate re-installing the plugin archive (e.g. from Hub)
        self._install_sample()

        # Re-discovery naturally restores it without needing blacklist manipulation
        manager.discover()
        self.assertIn("sample_plugin", manager.manifests)
        self.window.refresh_plugins_ui()
        self.assertIn("sample_plugin", self.window.plugin_sliders)

    def test_plugins_menu_only_shows_plugins_not_other_components(self):
        """Settings plugin menu strictly ignores components whose kind != 'plugin'."""
        data_plugins = Path(self.temp_dir) / "plugins"
        data_plugins.mkdir(parents=True, exist_ok=True)

        # Create a non-plugin component in plugin search path
        method_dir = data_plugins / "cql_algo"
        method_dir.mkdir(parents=True, exist_ok=True)
        (method_dir / "plugin.json").write_text(json.dumps({
            "id": "cql_algo",
            "name": "CQL Algorithm",
            "kind": "method",
            "version": "1.0.0"
        }), encoding="utf-8")

        self.window.plugin_manager.discover()
        self.assertNotIn("cql_algo", self.window.plugin_manager.manifests)

        self.window.refresh_plugins_ui()
        self.assertNotIn("cql_algo", self.window.plugin_sliders)

    def test_hub_dialog_sort_uninstalled_first(self):
        """HubDialog can sort components such that uninstalled items appear first."""
        from frontend.hub.models import HubComponent
        from frontend.hub.dialog import HubDialog

        client = self.window.hub_client
        c1 = HubComponent(id="comp_inst", name="Alpha Installed", kind="plugin", version="1.0.0", is_installed=True)
        c2 = HubComponent(id="comp_uninst", name="Beta Uninstalled", kind="plugin", version="1.0.0", is_installed=False)
        client.components = [c1, c2]
        client.registry = {"comp_inst": c1, "comp_uninst": c2}
        client.fetch_index_async = lambda *a, **k: None

        dialog = HubDialog(client, parent=self.window)
        try:
            dialog.client.indexLoaded.disconnect(dialog._on_index_loaded)
        except Exception:
            pass
        dialog._populate_cards([c1, c2])

        # Sort by uninstalled first
        idx = dialog.sort_combo.findData("uninstalled_first")
        self.assertGreaterEqual(idx, 0)
        dialog.sort_combo.setCurrentIndex(idx)

        # Search / filter run
        dialog._apply_filters()
        # Card for comp_uninst should be positioned before comp_inst in card layout
        idx_uninst = dialog.card_layout.indexOf(dialog.cards["comp_uninst"])
        idx_inst = dialog.card_layout.indexOf(dialog.cards["comp_inst"])
        self.assertLess(idx_uninst, idx_inst)
        dialog.close()


if __name__ == "__main__":
    unittest.main()

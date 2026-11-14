"""An independent app plugin works through the SDK host facade alone.

The fixture plugin imports only ``src.app.api``; it never touches the host
window. Its module records every import in a marker file so tests can prove
that refused activation and headless discovery execute no plugin code.
"""
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import pytest

pytest.importorskip("PyQt6")
from PyQt6.QtWidgets import QApplication

from src.app.components.installer import HubInstaller
from src.app.components.models import HubComponent, ReleaseInfo
from test_sdk_boundary import private_references

ROOT = Path(__file__).resolve().parents[1]

PLUGIN_SOURCE = '''\
from pathlib import Path

from src.app.api.desktop import Plugin, label
from src.app.api.workspace import WorkspacePaths

# Every import is recorded so tests can prove when plugin code ran.
_MARKER = Path(__file__).with_name("imported.log")
with _MARKER.open("a") as _stream:
    _stream.write("imported\\n")


class LatticeNotes(Plugin):
    def activate(self, context):
        self.context = context
        self.paths = context.workspace_paths()
        assert isinstance(self.paths, WorkspacePaths)
        self.restyled = 0
        self.opened = []
        self.widget = label("Lattice notes")
        context.add_sidebar_tab("lattice_notes", self.widget, "Lattice notes")
        context.register_command("lattice_notes.activations", "Lattice: activations", self.activations)
        context.add_appearance_listener(self.restyle)
        context.register_editor("Lattice notes", self.open_file, make_default=context.activated_explicitly)
        context.set_setting("activations", context.get_setting("activations", 0) + 1)
        context.set_editor_preference("wrap_lines", True)

    def activations(self):
        return self.context.get_setting("activations")

    def restyle(self):
        self.restyled += 1

    def open_file(self, path, line=None, column=None):
        self.opened.append((Path(path), line))
        return True

    def deactivate(self):
        self.widget.deleteLater()
'''


class Tabs:
    """Minimal sidebar honoring the calls PluginContext makes."""

    def __init__(self):
        self.tabs = {}
        self.current = None

    def addTab(self, widget, text, icon=None, short=None, tab_id=None):
        self.tabs[tab_id] = {"widget": widget, "text": text}

    def removeTab(self, tab_id):
        self.tabs.pop(tab_id, None)

    def setCurrentWidget(self, widget):
        self.current = widget


@pytest.fixture
def qapp():
    return QApplication.instance() or QApplication([])


def install_fixture(tmp_path, *, sdk=">=1.0,<2", plugin_id="lattice_notes"):
    archive = tmp_path / f"{plugin_id}.zip"
    meta = {"id": plugin_id, "name": "Lattice notes", "kind": "plugin", "version": "0.3.0",
            "storage_version": 2, "api_version": 1,
            "payload": {"implementations": ["__init__.py"], "resources": [], "configs": {}},
            "capabilities": [{"id": plugin_id, "kind": "plugin", "entry_point": "__init__.py:LatticeNotes"}]}
    with ZipFile(archive, "w") as bundle:
        bundle.writestr(".theta_component.json", json.dumps(meta))
        bundle.writestr("plugin.json", json.dumps({"id": plugin_id, "name": "Lattice notes",
                                                   "entry_point": "LatticeNotes", "sdk": sdk}))
        bundle.writestr("__init__.py", PLUGIN_SOURCE)
    component = HubComponent.from_dict(meta)
    component.releases = {"0.3.0": ReleaseInfo("0.3.0", archive.as_uri(), "")}
    installer = HubInstaller(tmp_path / "workspace", tmp_path / "state")
    installer.install(component, install_dependencies=False)
    return installer.paths, installer.resolve_target_dir(component)


def host(paths, tmp_path):
    from frontend.editor_service import EditorService
    from frontend.plugins.commands import CommandRegistry
    from frontend.plugins.manager import PluginManager
    from frontend.plugins.slots import ExtensionSlotRegistry
    from frontend.settings import SettingsManager
    from frontend.theme import ThemeManager

    settings = SettingsManager(paths.data_dir)
    logs = []
    window = SimpleNamespace(
        log=logs.append, store=SimpleNamespace(root=paths.data_dir), workspace_paths=paths,
        command_registry=CommandRegistry(), extension_slots=ExtensionSlotRegistry(), tabs=Tabs(),
        settings_manager=settings, theme_manager=ThemeManager(tmp_path / ".appearance.json"),
        editor_service=EditorService(settings))
    manager = PluginManager(window, paths=paths, settings_manager=settings)
    manager.discover()
    return window, manager, logs


def imports(package_root):
    marker = package_root / "imported.log"
    return marker.read_text().count("imported") if marker.is_file() else 0


def test_fixture_source_uses_only_the_sdk(tmp_path):
    source = tmp_path / "plugin.py"
    source.write_text(PLUGIN_SOURCE)
    assert private_references(source) == []
    assert "window" not in PLUGIN_SOURCE


def test_independent_plugin_lifecycle_through_the_facade(qapp, tmp_path, monkeypatch):
    paths, package_root = install_fixture(tmp_path)
    monkeypatch.chdir(tmp_path)  # an unrelated CWD; the plugin must use the selected workspace
    window, manager, logs = host(paths, tmp_path)

    assert manager.enable_plugin("lattice_notes"), logs
    plugin = manager.instances["lattice_notes"]
    assert plugin.paths.root == paths.root != Path.cwd()
    assert "lattice_notes" in window.tabs.tabs
    assert window.command_registry.get_command("lattice_notes.activations").callback() == 1
    # Explicit enabling made the plugin the default editor; host file routing reaches it.
    assert window.settings_manager.default_text_editor == "lattice_notes"
    assert window.editor_service.open_file(paths.root / "notes.md", line=3)
    assert plugin.opened == [((paths.root / "notes.md").resolve(), 3)]
    assert window.settings_manager.get("editor", "wrap_lines") is True
    before = plugin.restyled
    window.theme_manager.changed.emit()
    window.settings_manager.changed.emit()
    assert plugin.restyled == before + 2

    assert manager.disable_plugin("lattice_notes")
    assert "lattice_notes" not in window.tabs.tabs
    assert window.command_registry.get_command("lattice_notes.activations") is None
    assert "lattice_notes" not in dict(window.editor_service.providers())
    released = plugin.restyled
    window.theme_manager.changed.emit()
    window.settings_manager.changed.emit()
    assert plugin.restyled == released  # listeners were released with the context

    # Restoring saved state is not an explicit choice: the user's default editor stays.
    window.settings_manager.default_text_editor = "system"
    manager.enabled_states["lattice_notes"] = True
    manager.initialize_plugins()
    restored = manager.instances["lattice_notes"]
    assert not manager.contexts["lattice_notes"].activated_explicitly
    assert window.settings_manager.default_text_editor == "system"
    assert restored.activations() == 2  # plugin-scoped settings persisted
    assert manager.disable_plugin("lattice_notes")


@pytest.mark.parametrize("sdk", [">=2", "definitely not a range"])
def test_incompatible_sdk_requirement_is_refused_before_import(qapp, tmp_path, sdk):
    paths, package_root = install_fixture(tmp_path, sdk=sdk)
    window, manager, logs = host(paths, tmp_path)
    assert "lattice_notes" in manager.manifests  # discovered from metadata
    assert not manager.enable_plugin("lattice_notes")
    assert any("sdk" in line.lower() for line in logs), logs
    assert imports(package_root) == 0
    assert "lattice_notes" not in manager.instances and "lattice_notes" not in window.tabs.tabs


def test_headless_catalog_lists_the_plugin_without_qt_or_its_code(tmp_path):
    paths, package_root = install_fixture(tmp_path)
    code = (
        "import sys\nfrom pathlib import Path\nfrom src.app.workspace import WorkspacePaths\n"
        "from src.app.core.provider_catalog import get_catalog\n"
        f"paths = WorkspacePaths(Path({str(paths.root)!r}), {str(paths.data_dir)!r})\n"
        "caps = [c for p in get_catalog(paths).packages for c in p.capabilities if c.kind == 'plugin']\n"
        "plugin = next(c for c in caps if c.id == 'lattice_notes')\n"
        "assert plugin.metadata['sdk'] == '>=1.0,<2', plugin.metadata\n"
        "assert not [k for k in sys.modules if k.startswith('PyQt6')]\n")
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert imports(package_root) == 0


def test_bundled_plugins_declare_the_sdk_range_they_use():
    manifests = sorted((ROOT / "src/plugins/app/bundled").glob("*/plugin.json"))
    assert len(manifests) == 5
    from src.app.api import sdk_requirement_error
    for manifest in manifests:
        spec = json.loads(manifest.read_text()).get("sdk")
        assert spec and sdk_requirement_error(spec) is None, manifest

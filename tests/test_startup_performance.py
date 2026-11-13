"""Startup must build the active editor without importing executable providers."""
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest
from PyQt6.QtWidgets import QApplication

from frontend.config_tab_manager import ConfigTabManager
from frontend.config_viewer import ConfigViewer
from src.app.pipeline.exceptions import ConfigurationError
from src.app.pipeline.validation import find_missing_components, find_unavailable_components
from src.app.workspace import WorkspacePaths


@pytest.fixture
def declared_workspace(tmp_path, monkeypatch):
    paths = WorkspacePaths(tmp_path)
    root = paths.user("model") / "declared"
    root.mkdir(parents=True)
    (root / "provider.py").write_text("import missing_startup_dependency_123\nclass Provider: pass\n")
    manifest = {
        "id": "declared", "name": "Declared", "version": "1.0.0", "kind": "model",
        "storage_version": 2, "api_version": 1,
        "capabilities": [
            {"id": name, "kind": kind, "aliases": [alias], "entry_point": "provider.py:Provider"}
            for kind, name, alias in [("algorithm", "declared/agent", "AgentAlias"),
                                      ("model", "declared/model", "ModelAlias"),
                                      ("env", "declared/env", "EnvAlias")]
        ],
        "payload": {"implementations": ["provider.py"], "resources": [], "configs": {}},
    }
    (root / ".theta_component.json").write_text(json.dumps(manifest))
    monkeypatch.setenv("THETA_WORKSPACE", str(paths.root))
    monkeypatch.delenv("THETA_DATA_DIR", raising=False)
    return paths, root


def test_availability_uses_aliases_variants_and_one_discovery_pass(declared_workspace, monkeypatch):
    paths, root = declared_workspace
    cfg = {"paradigm": "online_rl", "env": "EnvAlias", "methods": {
        "one": {"agent": "AgentAlias_variant", "model": "ModelAlias"},
        "two": {"agent": "declared/agent", "model": "declared/model"},
        "disabled": {"enabled": False, "agent": "absent", "model": "absent"},
    }}
    # Count actual discovery without suppressing freshness checks.
    original = WorkspacePaths.manifests
    calls = []
    def manifests(self, *args, **kwargs):
        calls.append(self)
        return original(self, *args, **kwargs)
    monkeypatch.setattr(WorkspacePaths, "manifests", manifests)
    with patch("src.app.core.provider_loader.load_capability", side_effect=AssertionError("executed")):
        assert find_unavailable_components(cfg, paths=paths) == []
    assert calls.count(paths) == 1
    # Runtime preflight still proves imports work and reports missing dependencies.
    with pytest.raises(ConfigurationError, match="missing_startup_dependency_123"):
        find_missing_components(cfg)
    # External removals and additions are visible on the next operation.
    manifest = root / ".theta_component.json"
    content = manifest.read_text()
    manifest.unlink()
    assert "model: ModelAlias" in find_unavailable_components(cfg, paths=paths)
    manifest.write_text(content)
    assert find_unavailable_components(cfg, paths=paths) == []


def test_restore_loads_only_active_and_reads_latest_inactive_file(tmp_path):
    app = QApplication.instance() or QApplication([])
    config = tmp_path / "in/config"
    folder = config / "experiment/g"
    folder.mkdir(parents=True)
    for name in ("one", "two", "three"):
        (folder / f"{name}.yaml").write_text(f"experiment_id: {name}\nseed: 1\n")
    manager = ConfigTabManager()
    events = []
    manager.current_tab_changed.connect(lambda rel, viewer: events.append(rel))
    rels = [f"experiment/g/{name}.yaml" for name in ("one", "two", "three")]
    loaded = []
    original = ConfigViewer.load_file
    def load(self, path, rel):
        loaded.append(rel)
        return original(self, path, rel)
    with patch.object(ConfigViewer, "load_file", load):
        manager.restore_files(config, rels + [rels[0]], rels[1])
        assert loaded == [rels[1]]
        assert events == [rels[1]]
        assert [manager.widget(i).current_rel_path for i in range(manager.count())] == rels
        assert all(manager.tabBar().tabData(i) == "pinned" for i in range(manager.count()))
        inactive = manager._tabs[rels[0]]
        assert inactive.raw_data == {}
        with patch.object(inactive, "_build_boxes") as build:
            inactive._on_plugin_state_changed("tensorboard", True)
            build.assert_not_called()
        replacement = folder / "one.tmp"
        replacement.write_text("experiment_id: one\nseed: 123\n")
        replacement.replace(folder / "one.yaml")
        manager._reload_external_changes()
        manager.reload_open_files()
        manager.refresh_component_choices()
        assert inactive.raw_data == {}  # Refreshes do not materialize hidden forms.
        manager.setCurrentWidget(inactive)
        assert inactive.raw_data["seed"] == 123
        assert loaded.count(rels[0]) == 1
        inactive.raw_data["seed"] = 456
        inactive.is_dirty = True
        manager.setCurrentIndex(1)
        manager.setCurrentWidget(inactive)
        assert inactive.raw_data["seed"] == 456 and inactive.is_dirty
        # Unseen tabs can be renamed and closed without loading or writing YAML.
        (folder / "three.yaml").rename(folder / "renamed.yaml")
        manager._on_viewer_file_renamed(rels[2], "experiment/g/renamed.yaml")
        manager.close_tabs_matching(lambda rel: rel.endswith("renamed.yaml"))
        assert rels[2] not in loaded and "experiment/g/renamed.yaml" not in manager._tabs
    manager.close()


def test_missing_saved_active_falls_back_to_first_tab(tmp_path):
    app = QApplication.instance() or QApplication([])
    (tmp_path / "one.yaml").write_text("seed: 7\n")
    manager = ConfigTabManager()
    manager.restore_files(tmp_path, ["one.yaml"], "gone.yaml")
    assert manager.current_viewer().raw_data["seed"] == 7
    manager.close()


@pytest.mark.parametrize("saved_session", [True, False])
def test_fresh_window_restores_without_training_imports_or_initial_composition(tmp_path, saved_session):
    # Use a fresh interpreter so earlier tests cannot conceal eager imports.
    script = r'''
import importlib.abc, json, sys
from pathlib import Path
attempted = []
class RejectTraining(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'lightning', 'pytorch_lightning', 'matplotlib', 'stable_baselines3', 'optuna'}:
            attempted.append(fullname)
            raise AssertionError('startup imported ' + fullname)
sys.meta_path.insert(0, RejectTraining())
from PyQt6.QtCore import QObject, Qt
from PyQt6.QtWidgets import QApplication, QWidget
import frontend.app as module
from frontend.job_manager import JobManager
QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True)
app = QApplication([])
class Terminal(QWidget):
    def __init__(self, *, parent=None, **kwargs):
        super().__init__(parent)
        self.terminals = []
    def apply_theme(self, *args): pass
    def set_font_size(self, *args): pass
class Jobs(JobManager):
    def __init__(self, **kwargs):
        QObject.__init__(self, kwargs.get('parent'))
        self.jobs = {}
        self._queue = {'running': False, 'active': [], 'queued': [], 'finished': []}
    def shutdown(self): pass
class Backend:
    def __init__(self, *args, **kwargs): self.compositions = []; self.schemas = []
    def compose(self, *args, **kwargs): self.compositions.append((args, kwargs))
    def config_schema(self, *args): self.schemas.append(args)
    def list_runs(self, *args, **kwargs): pass
module.TerminalPanel = Terminal
module.JobManager = Jobs
module.LocalBackend = Backend
root = Path(sys.argv[1])
folder = root / 'in/config/experiment/g'
folder.mkdir(parents=True)
for name in ('one', 'two'):
    (folder / (name + '.yaml')).write_text('experiment_id: ' + name + '\nmethods: {m: {agent: startup/agent, model: startup/model}}\n')
provider = root / 'src/usr/models/startup'
provider.mkdir(parents=True)
(provider / 'provider.py').write_text('import torch\nraise AssertionError("provider executed")\n')
(provider / '.theta_component.json').write_text(json.dumps({
    'id': 'startup', 'name': 'Startup', 'version': '1.0.0', 'kind': 'model',
    'storage_version': 2, 'api_version': 1,
    'capabilities': [{'kind': k, 'id': 'startup/' + n, 'entry_point': 'provider.py:Provider'}
                     for k, n in [('algorithm', 'agent'), ('model', 'model'), ('env', 'env')]],
    'payload': {'implementations': ['provider.py'], 'resources': [], 'configs': {}}}))
data = root / '.thetaide'
data.mkdir()
(data / 'settings.toml').write_text('[plugins]\nenabled = []\ndisabled = ["documentation", "text_editor", "arxiv_papers", "tensorboard", "model_builder"]\n'
    '[session]\nactive_pane = "config"\nopen_configs = ["experiment/g/one.yaml", "experiment/g/two.yaml"]\nactive_config = "experiment/g/two.yaml"\n')
saved_session = sys.argv[2] == 'True'
if not saved_session:
    settings = (data / 'settings.toml').read_text().replace(
        'open_configs = ["experiment/g/one.yaml", "experiment/g/two.yaml"]',
        'open_configs = ["experiment/g/missing.yaml"]')
    (data / 'settings.toml').write_text(settings)
expected = 'experiment/g/two.yaml' if saved_session else 'experiment/g/one.yaml'
loads = []
fallbacks = []
load = module.ConfigViewer.load_file
fallback = module.Window.init_default_experiment
module.ConfigViewer.load_file = lambda self, path, rel: (loads.append(rel), load(self, path, rel))[-1]
module.Window.init_default_experiment = lambda self: (fallbacks.append(True), fallback(self))[-1]
window = module.Window(data, workspace=root)
assert not attempted, attempted
assert not window.settings_view.page_widgets
assert window.components_panel.active_path is None
assert window.backend.compositions == [] and window.backend.schemas == []
assert loads.count(expected) == 1, loads
assert bool(fallbacks) is not saved_session
assert window.config_tab_manager.count() == (2 if saved_session else 1)
if saved_session:
    assert 'experiment/g/one.yaml' not in loads
assert not window.start_button.isEnabled()
window.show()
app.processEvents()
# First pane selection must not reload the already loaded form.
assert loads.count(expected) == 1, loads
window.request_compose()
assert len(window.backend.compositions) == 1
assert window.backend.compositions[0][0][0] == expected.removeprefix('experiment/').removesuffix('.yaml')
assert not window.start_button.isEnabled()
assert not window.backend.schemas
window.save_session()
if saved_session:
    assert 'experiment/g/one.yaml' not in loads
# A successful result enables Run without searching unrelated recipes.
callback = window.backend.compositions[0][0][2]
report = {'valid': True, 'config_yaml': 'seed: 1',
          'config': {'env': 'startup/env', 'methods': {'m': {'agent': 'startup/agent', 'model': 'startup/model'}}}}
callback(report, None)
assert window.start_button.isEnabled(), window.start_button.toolTip()
assert not window.backend.schemas
if saved_session:
    window.config_tab_manager.setCurrentIndex(0)
    assert loads.count('experiment/g/one.yaml') == 1
    assert not window.start_button.isEnabled()
    callback(report, None)  # The preceding tab's validation must stay obsolete.
    assert not window.start_button.isEnabled()
    latest = window.backend.compositions[-1][0][2]
    latest(None, 'provider dependency unavailable')
    assert not window.start_button.isEnabled()
    assert window.compose_error == 'provider dependency unavailable'
# Lazily built settings remain usable in any navigation order.
for category in ('hotkeys', 'appearance', 'terminal', 'storage', 'backend', 'core_plugins', 'community_plugins', 'editor'):
    window.settings_view.show_tab(category)
    assert category in window.settings_view.page_widgets
window.close()
assert not attempted, attempted
'''
    result = subprocess.run([sys.executable, "-c", script, str(tmp_path), str(saved_session)],
                            cwd=Path(__file__).resolve().parents[1],
                            env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr

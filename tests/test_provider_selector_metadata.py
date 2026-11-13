"""Authoring dropdowns use declarations, never executable provider classes."""
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest

from src.app.core.data_loaders import loader_metadata, loader_selection
from src.app.core.provider_discovery import provider_discovery_snapshot
from src.app.core.provider_loader import invalidate
from src.app.core.provider_metadata import selector_metadata
from src.app.workspace import WorkspacePaths


@pytest.fixture
def selector_workspace(tmp_path, monkeypatch):
    paths = WorkspacePaths(tmp_path)
    paths.config.mkdir(parents=True)
    root = paths.user("package") / "selectors"
    root.mkdir(parents=True)
    (root / "provider.py").write_text("raise AssertionError('selector executed provider code')\n")
    profile = root / "configs/data/selector_train.yaml"
    profile.parent.mkdir(parents=True)
    profile.write_text("data_module: selector/train\nbatch_size: 17\n")
    state_profile = root / "configs/data/selector_state.yaml"
    state_profile.write_text("name: selector/state\nfeature_count: 4\n")
    meta = {
        "id": "selector-package", "name": "Selectors", "kind": "suite", "version": "1.0.0",
        "storage_version": 2, "api_version": 1, "paradigms": ["supervised"],
        "capabilities": [
            {"id": "selector/train", "kind": "datamodule", "aliases": ["TrainAlias"],
             "entry_point": "provider.py:Train", "role": "training",
             "output_contracts": ["batches/v1"],
             "parameter_schema": {"type": "object", "properties": {"batch_size": {"type": "integer"}}}},
            {"id": "selector/state", "kind": "datamodule", "aliases": ["StateAlias"],
             "entry_point": "provider.py:State", "role": "symbolic_state",
             "paradigms": ["online_rl"]},
            {"id": "selector/eval", "kind": "evaluator", "aliases": ["EvalAlias"],
             "entry_point": "provider.py:Evaluator"},
            {"id": "selector/plot", "kind": "plotter", "aliases": ["PlotAlias"],
             "entry_point": "provider.py:Plotter", "paradigms": ["online_rl"]},
        ],
        "payload": {"implementations": ["provider.py"], "resources": [], "configs": {
            "in/config/data/selector_train.yaml": "configs/data/selector_train.yaml",
            "in/config/data/selector_state.yaml": "configs/data/selector_state.yaml",
        }},
    }
    (root / ".theta_component.json").write_text(json.dumps(meta))
    monkeypatch.setenv("THETA_WORKSPACE", str(paths.root))
    return paths, root


@pytest.fixture
def selector_viewer(selector_workspace, monkeypatch):
    from PyQt6.QtWidgets import QApplication
    from frontend.config_model import ConfigTree
    from frontend.config_viewer import ConfigViewer

    app = QApplication.instance() or QApplication([])
    paths, _ = selector_workspace
    tree = ConfigTree(paths.config)
    monkeypatch.setattr("frontend.config_viewer.config_tree", lambda *args, **kwargs: tree)
    widget = ConfigViewer()
    widget.raw_data = {"paradigm": "supervised"}
    yield widget
    widget.close()


def test_choices_use_capability_metadata_profiles_aliases_and_builtins(selector_viewer):
    viewer = selector_viewer
    with patch("src.app.core.provider_loader.load_capability", side_effect=AssertionError("executed")) as load:
        with provider_discovery_snapshot():
            train = viewer._component_choices("data_module")
            assert {"selector/train", "TrainAlias", "SupervisedDataModule"} <= train.keys()
            assert train["selector/train"]["batch_size"] == 17
            assert "selector/state" not in train and "StateAlias" not in train
            assert {"selector/eval", "EvalAlias", "ClassificationEvalProtocol"} <= viewer._component_choices("evaluator").keys()
            assert "selector/plot" not in viewer._component_choices("plots")
            assert "StateAlias" not in viewer._component_choices("symbolic_state_loader")
            viewer.raw_data["paradigm"] = "online_rl"
            assert "TrainAlias" not in viewer._component_choices("data_module")
            assert "EvalAlias" not in viewer._component_choices("evaluator")
            assert {"selector/plot", "PlotAlias"} <= viewer._component_choices("plots").keys()
            assert {"selector/state", "StateAlias", "selector_state"} <= viewer._component_choices("symbolic_state_loader").keys()
        load.assert_not_called()


def test_build_reuses_choices_profiles_and_loader_metadata(selector_viewer, selector_workspace):
    from src.app.core.provider_catalog import ProviderCatalog

    paths, root = selector_workspace
    original_read = Path.read_text
    reads = []

    def read(path, *args, **kwargs):
        if path == root / "configs/data/selector_train.yaml":
            reads.append(path)
        return original_read(path, *args, **kwargs)

    original_resolve = ProviderCatalog.resolve
    resolves = []

    def resolve(catalog, kind, name):
        resolves.append((kind, name))
        return original_resolve(catalog, kind, name)

    # Warm catalog validation (which separately reads declared YAML payloads).
    selector_metadata("datamodule", paths)
    with patch.object(Path, "read_text", read), patch.object(ProviderCatalog, "resolve", resolve):
        with provider_discovery_snapshot():
            for _ in range(5):
                choices = selector_viewer._component_choices("data_module")
                assert choices["selector/train"]["batch_size"] == 17
                choices["selector/train"]["batch_size"] = 99
                loader_metadata({"name": "selector/train"}, paths=paths)["output_contracts"].clear()
            assert loader_metadata({"name": "selector/train"}, paths=paths)["output_contracts"] == ["batches/v1"]
            assert reads == [root / "configs/data/selector_train.yaml"]
            assert resolves.count(("datamodule", "selector/train")) == 1
    # Profile edits and lifecycle invalidation reset the per-build memo too.
    with provider_discovery_snapshot():
        assert selector_viewer._component_choices("data_module")["selector/train"]["batch_size"] == 17
        (root / "configs/data/selector_train.yaml").write_text("data_module: selector/train\nbatch_size: 29\n")
        invalidate(paths, root)
        assert selector_viewer._component_choices("data_module")["selector/train"]["batch_size"] == 29
    (root / "configs/data/selector_train.yaml").write_text("data_module: selector/train\nbatch_size: 31\n")
    assert selector_viewer._component_choices("data_module")["selector/train"]["batch_size"] == 31


def test_loader_profiles_are_read_once_and_copies_are_isolated(selector_workspace):
    import src.app.core.data_loaders as loaders

    paths, _ = selector_workspace
    with patch.object(loaders.yaml, "safe_load", wraps=loaders.yaml.safe_load) as parse:
        # Warm catalog validation before counting profile parses.
        selector_metadata("datamodule", paths)
        parse.reset_mock()
        with provider_discovery_snapshot():
            for _ in range(4):
                selected, _ = loader_selection({"data_module": "selector_train"}, "data_module", paths=paths)
                assert selected == {"name": "selector/train", "params": {"batch_size": 17}}
                selected["params"]["batch_size"] = 99
            assert parse.call_count == 1


def test_duplicate_selectors_respect_explicit_overrides(selector_workspace):
    paths, root = selector_workspace
    other = paths.user("package") / "duplicate"
    other.mkdir()
    (other / "provider.py").write_text("raise AssertionError('executed')")
    meta = json.loads((root / ".theta_component.json").read_text())
    meta.update(id="duplicate", paradigms=["online_rl"])
    meta["payload"] = {"implementations": ["provider.py"], "resources": [], "configs": {}}
    (other / ".theta_component.json").write_text(json.dumps(meta))
    with provider_discovery_snapshot():
        assert "TrainAlias" not in selector_metadata("datamodule", paths)
    paths.data_dir.mkdir(exist_ok=True)
    (paths.data_dir / "provider_overrides.json").write_text('{"datamodule:trainalias": "duplicate"}')
    with provider_discovery_snapshot():
        assert selector_metadata("datamodule", paths)["TrainAlias"]["paradigms"] == ["online_rl"]


def test_cold_themes_form_never_attempts_training_or_plot_imports(tmp_path):
    script = r'''
import importlib.abc
import sys
attempted = []
class RejectTrainingImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'lightning', 'pytorch_lightning', 'matplotlib', 'stable_baselines3', 'optuna'}:
            attempted.append(fullname)
            raise AssertionError('form attempted executable import: ' + fullname)
sys.meta_path.insert(0, RejectTrainingImports())
from pathlib import Path
from PyQt6.QtWidgets import QApplication
from frontend.config_viewer import ConfigViewer
app = QApplication([])
widget = ConfigViewer()
widget.load_file(Path('src/usr/packages/themes-dev/configs/experiment/AL/codex/themes_synthetic.yaml'), 'experiment/AL/codex/themes_synthetic.yaml')
assert 'ticc_em_edm' in widget.raw_data.get('methods', {}), widget.raw_data
assert widget._inherited_paradigm() == 'apprenticeship'
assert widget.field_widgets, 'form failed to render'
for kind in ('data_module', 'symbolic_state_loader', 'evaluator', 'plots'):
    widget._component_choices(kind)
assert not attempted, attempted
assert not any(name.split('.')[0] in {'torch', 'lightning', 'matplotlib'} for name in sys.modules)
'''
    result = subprocess.run([sys.executable, "-c", script], cwd=Path(__file__).resolve().parents[1],
                            env={**os.environ, "QT_QPA_PLATFORM": "offscreen", "MPLCONFIGDIR": str(tmp_path)},
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr

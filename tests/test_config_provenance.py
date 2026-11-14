"""Provider-neutral profile ownership across UI, composition and worker processes."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from src.app.core.provider_catalog import DuplicateProvider
from src.app.pipeline.compose import compose_experiment
from src.app.workspace import WorkspacePaths


def provider(paths, package_id, value, *, origin="user", selector="fixture/net"):
    root = (paths.user("package") if origin == "user" else paths.installed_components) / package_id
    (root / "configs/model/fixture").mkdir(parents=True)
    (root / "provider.py").write_text("raise RuntimeError('catalog must not import provider code')\n")
    (root / "configs/model/fixture/net.yaml").write_text(f"architecture: net\nwidth: {value}\n")
    destination = f"in/config/model/{selector}.yaml"
    (root / ".theta_component.json").write_text(json.dumps({
        "id": package_id, "name": package_id, "version": "1.0.0", "kind": "model",
        "storage_version": 2, "api_version": 1,
        "capabilities": [{"id": selector, "kind": "model", "entry_point": "provider.py:Network"}],
        "payload": {"implementations": ["provider.py"], "resources": [],
                    "configs": {destination: "configs/model/fixture/net.yaml"}},
        "parameter_schema": {"type": "object", "x-theta-validate": True,
                             "properties": {"width": {"type": "integer", "maximum": value}}},
    }))
    return root


def recipe(paths):
    directory = paths.config / "experiment"
    directory.mkdir(parents=True)
    (directory / "probe.yaml").write_text(
        "# @package _global_\ndefaults: [override /model: fixture/net]\nno_plot: true\n")


@pytest.mark.parametrize("origin", ["user", "installed"])
def test_one_profile_owner_in_gui_cli_and_worker(tmp_path, monkeypatch, origin):
    from frontend.config_model import ConfigTree
    from src.app.pipeline.component_validation import schemas_for_method
    paths = WorkspacePaths(tmp_path / "workspace")
    provider(paths, "independent", 17, origin=origin)
    recipe(paths)
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    monkeypatch.chdir(unrelated)
    resolved = paths.config_resolver().resolve("model/fixture/net.yaml", required=True)
    assert resolved.package.id == "independent"
    assert not resolved.overridden
    assert ConfigTree(paths.config, workspace=paths).models["fixture/net"].parameters["width"] == 17
    assert schemas_for_method(paths, {"model": "fixture/net"})["model"] == resolved.schema
    assert compose_experiment("probe", workspace=paths).cfg.model.width == 17
    env = {**os.environ, **paths.process_environment(), "PYTHONPATH": str(paths.application)}
    env.pop("THETA_PROVIDER_SNAPSHOT", None)
    result = subprocess.run([sys.executable, str(paths.application / "src/app/train.py"),
                             "--cfg", "job", "+experiment=probe"], cwd=unrelated,
                            env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "width: 17" in result.stdout


def test_override_changes_value_keeps_provider_schema(tmp_path):
    paths = WorkspacePaths(tmp_path)
    root = provider(paths, "independent", 21)
    user = paths.config / "model/fixture/net.yaml"
    user.parent.mkdir(parents=True)
    user.write_text("architecture: net\nwidth: 9\n")
    recipe(paths)
    resolved = paths.config_resolver().resolve("model/fixture/net.yaml")
    assert resolved.file == user
    assert resolved.package.id == "independent" and resolved.overridden
    assert resolved.schema["properties"]["width"]["maximum"] == 21
    assert compose_experiment("probe", workspace=paths).cfg.model.width == 9
    assert "width: 21" in (root / "configs/model/fixture/net.yaml").read_text()


def test_ambiguity_and_explicit_selection_share_profile_and_schema(tmp_path):
    from frontend.config_model import ConfigTree
    from src.app.pipeline.component_validation import schemas_for_method
    paths = WorkspacePaths(tmp_path)
    provider(paths, "first", 11)
    provider(paths, "second", 22)
    recipe(paths)
    with pytest.raises(DuplicateProvider, match="Ambiguous"):
        compose_experiment("probe", workspace=paths)
    paths.data_dir.mkdir()
    (paths.data_dir / "provider_overrides.json").write_text(json.dumps({"model:fixture/net": "second"}))
    resolved = paths.config_resolver().resolve("model/fixture/net.yaml")
    assert resolved.package.id == "second"
    assert ConfigTree(paths.config).models["fixture/net"].parameters["width"] == 22
    assert ConfigTree(paths.config).models["fixture/net"].parameter_schema == resolved.schema
    assert schemas_for_method(paths, {"model": "fixture/net"})["model"] == resolved.schema
    assert compose_experiment("probe", workspace=paths).cfg.model.width == 22


def test_two_workspaces_do_not_capture_import_time_selection(tmp_path, monkeypatch):
    workspaces = [WorkspacePaths(tmp_path / name) for name in ("one", "two")]
    for i, paths in enumerate(workspaces):
        provider(paths, "fixture", 10 + i)
        recipe(paths)
    for i, paths in enumerate(workspaces):
        monkeypatch.setenv("THETA_WORKSPACE", str(paths.root))
        assert compose_experiment("probe").cfg.model.width == 10 + i
        assert compose_experiment("probe", workspace=paths).workspace == paths


def test_missing_profile_is_actionable_and_selector_cannot_escape(tmp_path):
    resolver = WorkspacePaths(tmp_path).config_resolver()
    with pytest.raises(FileNotFoundError, match="install/select its provider"):
        resolver.resolve("model/missing", required=True)
    for value in ("../private", "/private/config"):
        with pytest.raises(ValueError, match="relative"):
            resolver.resolve(value)


def test_explicit_editable_copy_is_under_usr_and_survives_uninstall(tmp_path):
    from frontend.config_model import ConfigTree
    from src.app.components.configuration import ConfigurationFiles
    from src.app.components.installer import HubInstaller
    paths = WorkspacePaths(tmp_path)
    root = provider(paths, "independent", 17, origin="installed")
    recipe(paths)
    source = root / "configs/model/fixture/net.yaml"
    original = source.read_bytes()
    files = ConfigurationFiles(paths)
    override = files.create_override("model/fixture/net")
    assert override == paths.user_config / "model/fixture/net.yaml"
    assert override.read_bytes() == original == source.read_bytes()
    files.configure(override, ["width=12"])
    assert files.create_override("model/fixture/net") == override
    assert "width: 12" in override.read_text()
    assert source.read_bytes() == original
    assert compose_experiment("probe", workspace=paths).cfg.model.width == 12
    assert ConfigTree(paths.config).models["fixture/net"].parameters["width"] == 12
    assert json.loads((paths.data_dir / "config_overrides.json").read_text())["model/fixture/net.yaml"]["package"] == "independent"
    from src.app.components.lifecycle import uninstall
    from src.app.core.provider_catalog import get_catalog
    metadata = {**get_catalog(paths).package("independent").manifest, "target_dir": str(root)}
    assert uninstall(HubInstaller(paths.root, paths.data_dir), metadata)
    assert override.is_file() and "width: 12" in override.read_text()
    assert compose_experiment("probe", workspace=paths).cfg.model.width == 12


def test_override_transaction_rolls_back_on_record_failure(tmp_path, monkeypatch):
    from src.app.components.configuration import ConfigurationFiles
    from src.app.components import transactions
    paths = WorkspacePaths(tmp_path)
    provider(paths, "independent", 17)
    original = transactions.write_json
    def fail_record(path, value):
        if Path(path).name == "config_overrides.json":
            raise OSError("fixture disk failure")
        return original(path, value)
    monkeypatch.setattr(transactions, "write_json", fail_record)
    with pytest.raises(OSError, match="disk failure"):
        ConfigurationFiles(paths).create_override("model/fixture/net")
    assert not (paths.user_config / "model/fixture/net.yaml").exists()
    assert not (paths.data_dir / "config_overrides.json").exists()


def test_headless_cli_can_create_provider_override(tmp_path):
    paths = WorkspacePaths(tmp_path)
    provider(paths, "independent", 17, origin="installed")
    env = {**os.environ, **paths.process_environment(), "PYTHONPATH": str(paths.application)}
    result = subprocess.run([sys.executable, str(paths.application / "run_pipeline.py"), "component", "--root", str(tmp_path),
                             "config", "override", "model/fixture/net"], cwd=tmp_path,
                            env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert Path(json.loads(result.stdout)["config"]) == paths.user_config / "model/fixture/net.yaml"


def test_report_cache_tracks_selected_workspace_and_usr_edits(tmp_path):
    from src.app.services.config_service import compose_report
    from src.app.components.configuration import ConfigurationFiles
    workspaces = [WorkspacePaths(tmp_path / name) for name in ("one", "two")]
    for width, paths in enumerate(workspaces, 17):
        provider(paths, "independent", width)
        recipe(paths)
        assert compose_report("probe", workspace=paths)["config"]["model"]["width"] == width
    paths = workspaces[0]
    override = ConfigurationFiles(paths).create_override("model/fixture/net")
    override.write_text("architecture: net\nwidth: 19\n")
    assert compose_report("probe", workspace=paths)["config"]["model"]["width"] == 19
    override.write_text("architecture: net\nwidth: 20\n")
    assert compose_report("probe", workspace=paths)["config"]["model"]["width"] == 20


def test_usr_only_workspace_and_root_override_use_framework_defaults(tmp_path):
    from src.app.components.configuration import ConfigurationFiles
    paths = WorkspacePaths(tmp_path)
    provider(paths, "independent", 17)
    (paths.user_config / "experiment").mkdir(parents=True)
    (paths.user_config / "experiment/probe.yaml").write_text(
        "# @package _global_\ndefaults: [override /model: fixture/net]\n")
    assert not paths.config.exists()
    assert compose_experiment("probe", workspace=paths).cfg.model.width == 17
    files = ConfigurationFiles(paths)
    root_override = files.create_override("config")
    files.configure(root_override, ["seed=123"])
    composed = compose_experiment("probe", workspace=paths)
    assert composed.cfg.seed == 123
    assert composed.cfg.model.width == 17
    assert not paths.config.exists()


def test_override_rejects_symlinked_user_destination(tmp_path):
    from src.app.components.configuration import ConfigurationFiles
    paths = WorkspacePaths(tmp_path / "workspace")
    provider(paths, "independent", 17)
    outside = tmp_path / "outside"
    outside.mkdir()
    paths.user_config.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic links"):
        ConfigurationFiles(paths).create_override("model/fixture/net")
    assert list(outside.iterdir()) == []

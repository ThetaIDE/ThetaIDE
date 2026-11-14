"""Package format v3 and the src/usr workspace layout, end to end."""
import json
import zipfile

import pytest
import yaml

from src.app.components.installer import HubInstaller
from src.app.components.models import HubComponent, ReleaseInfo
from src.app.pipeline.compose import compose_experiment
from src.app.pipeline.datasets import resolve_dataset_path
from src.app.workspace import WorkspacePaths


def authored_model(workspace):
    root = workspace / "src/usr/models/roundtrip"
    (root / "configs/model/roundtrip").mkdir(parents=True)
    (root / "model.py").write_text("class Model:\n    value = 5\n")
    (root / "configs/model/roundtrip/default.yaml").write_text("architecture: roundtrip\nwidth: 8\n")
    (root / ".theta_component.json").write_text(json.dumps({
        "id": "roundtrip", "name": "Round trip", "kind": "model", "version": "1.0.0",
        "storage_version": 3, "api_version": 1,
        "capabilities": [{"id": "roundtrip", "kind": "model", "entry_point": "model.py:Model"}],
        "payload": {"implementations": ["model.py"], "resources": [],
                    "configs": {"model/roundtrip/default.yaml": "configs/model/roundtrip/default.yaml"}},
        "config_path": "model/roundtrip/default.yaml"}))
    return root / "configs/model/roundtrip/default.yaml"


def test_v3_export_install_resolve_override_and_uninstall(tmp_path):
    from src.app.services.component_exporter import ComponentExporter, ComponentExportOptions
    source = tmp_path / "author"
    profile = authored_model(source)
    archive = tmp_path / "roundtrip.zip"
    result = ComponentExporter(workspace_root=source).export(
        profile, options=ComponentExportOptions(export_format="zip", output_path=archive, version="1.0.0"))
    assert result.success, result.error
    with zipfile.ZipFile(archive) as bundle:
        meta = json.loads(bundle.read(".theta_component.json"))
    assert meta["storage_version"] == 3
    assert set(meta["payload"]["configs"]) == {"model/roundtrip/default.yaml"}

    installer = HubInstaller(tmp_path / "workspace", tmp_path / "state")
    component = HubComponent.from_dict(meta)
    component.releases = {"1.0.0": ReleaseInfo("1.0.0", archive.as_uri(), "")}
    assert installer.install(component, install_dependencies=False)
    paths = installer.paths
    resolved = paths.config_resolver().resolve("model/roundtrip/default.yaml", required=True)
    assert resolved.package.id == "roundtrip" and resolved.origin != "workspace"
    assert resolved.file.is_relative_to(installer.resolve_target_dir(component))
    assert not paths.config.exists(), "installing never writes the user's config root"

    from src.app.components.configuration import ConfigurationFiles
    override = ConfigurationFiles(paths).create_override("model/roundtrip/default.yaml")
    override.write_text("architecture: roundtrip\nwidth: 99\n")
    assert paths.resolve_config("model/roundtrip/default.yaml") == override
    assert installer.uninstall(component)
    assert paths.config_resolver().resolve("model/roundtrip/default.yaml").file == override
    assert override.read_text() == "architecture: roundtrip\nwidth: 99\n"
    assert not (paths.root / "in").exists()


def test_storage_2_archives_are_rejected(tmp_path):
    meta = {"id": "old", "name": "Old", "kind": "model", "version": "1.0.0", "storage_version": 2, "api_version": 1,
            "capabilities": [{"id": "old", "kind": "model", "entry_point": "model.py:Model"}],
            "payload": {"implementations": ["model.py"], "resources": [],
                        "configs": {"model/old.yaml": "configs/model/old.yaml"}}}
    archive = tmp_path / "old.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr(".theta_component.json", json.dumps(meta))
        bundle.writestr("model.py", "class Model: pass\n")
        bundle.writestr("configs/model/old.yaml", "architecture: old\n")
    installer = HubInstaller(tmp_path / "workspace", tmp_path / "state")
    component = HubComponent.from_dict(meta)
    component.releases = {"1.0.0": ReleaseInfo("1.0.0", archive.as_uri(), "")}
    with pytest.raises(ValueError, match="storage_version: 3"):
        installer.install(component, install_dependencies=False)
    assert not installer.paths.installed_components.exists()


def test_composition_from_src_usr_config(tmp_path):
    """Root config.yaml, absolute defaults, interpolation and nested group references."""
    paths = WorkspacePaths(tmp_path)
    config = paths.config
    root = yaml.safe_load((paths.framework_config / "config.yaml").read_text())
    root["seed"] = 321
    (config / "model/nested").mkdir(parents=True)
    (config / "config.yaml").write_text(yaml.safe_dump(root, sort_keys=False))
    (config / "model/nested/wide.yaml").write_text("architecture: wide\nwidth: ${seed}\n")
    (config / "experiment/group").mkdir(parents=True)
    (config / "experiment/group/_base.yaml").write_text(
        "# @package _global_\ndefaults:\n  - override /model: nested/wide\n  - _self_\nno_plot: true\n")
    (config / "experiment/group/run.yaml").write_text(
        "# @package _global_\ndefaults:\n  - /experiment/group/_base\n  - _self_\nexperiment_id: run\n")
    composed = compose_experiment("group/run", workspace=paths)
    assert composed.cfg.seed == 321
    assert composed.cfg.model.architecture == "wide"
    assert composed.cfg.model.width == 321
    assert composed.cfg.no_plot is True
    assert not (tmp_path / "in").exists()


def test_dataset_discovery_uses_src_usr_datasets(tmp_path):
    paths = WorkspacePaths(tmp_path)
    grouped = paths.datasets / "group/run/states"
    grouped.mkdir(parents=True)
    (grouped / "part.pkl").write_bytes(b"x")
    shared = paths.datasets / "shared.npz"
    shared.write_bytes(b"x")
    assert resolve_dataset_path("states", "group", "run", paths=paths) == grouped
    assert resolve_dataset_path("shared", paths=paths) == shared
    assert paths.datasets == tmp_path / "src/usr/datasets"

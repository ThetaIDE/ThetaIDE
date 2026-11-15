"""Tests for ExperimentExporter service and HubInstaller roundtrip."""
import json
from pathlib import Path
import zipfile

from src.app.components.installer import HubInstaller
from src.app.components.models import HubComponent, ReleaseInfo
from src.app.services.experiment_exporter import ExperimentExporter, ExportOptions


def _provider_package(workspace: Path, kind: str, package_id: str, capabilities) -> None:
    """A declared native package whose code must never run during export."""
    from src.app.workspace import WorkspacePaths
    root = WorkspacePaths(workspace).user(kind) / package_id
    root.mkdir(parents=True)
    (root / "provider.py").write_text("raise AssertionError('export executed provider code')\n", encoding="utf-8")
    (root / ".theta_component.json").write_text(json.dumps({
        "id": package_id, "name": package_id, "kind": kind, "version": "1.0.0",
        "storage_version": 3, "api_version": 1,
        "capabilities": [{"id": name, "kind": kind, "entry_point": "provider.py:Provider"} for name in capabilities],
        "payload": {"implementations": ["provider.py"], "resources": [], "configs": {}},
    }), encoding="utf-8")


def _setup_test_experiment(workspace: Path) -> str:
    """A minimal experiment whose environment, algorithm and model come from declared packages."""
    _provider_package(workspace, "env", "demo-env", ["demo_env"])
    _provider_package(workspace, "algorithm", "demo-agent", ["demo_agent"])
    _provider_package(workspace, "model", "demo-model", ["demo_model"])
    exp_dir = workspace / "src/usr/config" / "experiment" / "demo"
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "_base.yaml").write_text("paradigm: online_rl\n", encoding="utf-8")
    (exp_dir / "quick_test.yaml").write_text(
        "defaults:\n  - demo/_base\nexperiment_id: quick_test\nenv: demo_env\n"
        "methods:\n  test_method:\n    agent: demo_agent\n    model: demo_model\n",
        encoding="utf-8",
    )
    return "demo/quick_test"


def test_export_experiment_success(tmp_path):
    workspace = tmp_path / "src_workspace"
    workspace.mkdir()
    exp_name = _setup_test_experiment(workspace)

    exporter = ExperimentExporter(workspace_root=workspace)

    out_zip = tmp_path / "test_export.zip"
    opts = ExportOptions(
        include_checkpoints=False,
        include_environment=True,
        include_models=True,
        include_rules=True,
        output_path=out_zip,
        author="Tester",
        version="1.2.3",
        description="Test export of experiment",
    )

    result = exporter.export(exp_name, options=opts)
    assert result.success is True, result.error
    assert result.archive_path == out_zip
    assert out_zip.is_file()

    with zipfile.ZipFile(out_zip, "r") as zf:
        namelist = zf.namelist()
        assert ".theta_component.json" in namelist
        assert "README.md" in namelist
        assert "configs/experiment/demo/quick_test.yaml" in namelist
        assert "configs/experiment/demo/_base.yaml" in namelist
        # Providers stay in their own packages and become pinned dependencies.
        assert not any(name.endswith("provider.py") for name in namelist)

        manifest = json.loads(zf.read(".theta_component.json").decode("utf-8"))
        assert manifest["kind"] == "experiment"
        assert manifest["version"] == "1.2.3"
        assert manifest["author"]["name"] == "Tester"
        assert manifest["environment"] == "demo_env"
        assert manifest["agents"] == ["demo_agent"]
        assert {d["id"]: d["version"] for d in manifest["dependencies"]["components"]} == {
            "demo-env": "==1.0.0", "demo-agent": "==1.0.0", "demo-model": "==1.0.0"}


def test_export_nonexistent_experiment(tmp_path):
    exporter = ExperimentExporter(workspace_root=tmp_path)
    result = exporter.export("nonexistent_group/nonexistent_exp", options=ExportOptions(output_path=tmp_path / "dummy.zip"))
    assert result.success is False
    assert "not found" in result.error.lower()


def test_hub_installer_roundtrip(tmp_path):
    import shutil
    from src.app.workspace import WorkspacePaths

    workspace = tmp_path / "src_workspace"
    workspace.mkdir()
    exp_name = _setup_test_experiment(workspace)

    exporter = ExperimentExporter(workspace_root=workspace)

    pkg_zip = tmp_path / "demo_pkg.zip"
    res = exporter.export(
        exp_name,
        options=ExportOptions(output_path=pkg_zip, version="2.0.0"),
    )
    assert res.success is True, res.error

    # The target workspace already has the experiment's provider packages.
    mock_workspace = tmp_path / "target_workspace"
    mock_data = tmp_path / "target_data"
    mock_data.mkdir(parents=True)
    shutil.copytree(workspace / "src/usr", mock_workspace / "src/usr",
                    ignore=shutil.ignore_patterns("config"))

    installer = HubInstaller(workspace_dir=mock_workspace, data_dir=mock_data)

    comp = HubComponent(
        id="demo_quick_test",
        name=exp_name,
        kind="experiment",
        version="2.0.0",
        description="Roundtrip test",
        releases={
            "2.0.0": ReleaseInfo(
                tag="v2.0.0",
                url=f"file://{pkg_zip.resolve()}",
                sha256="",
            )
        },
    )

    installed = installer.install(comp)
    assert installed is True

    # The installed package supplies the recipe; installation writes no workspace config.
    resolved = WorkspacePaths(mock_workspace).config_resolver().resolve("experiment/demo/quick_test.yaml")
    assert resolved is not None and resolved.package.id == "demo_quick_test"
    assert not (mock_workspace / "src/usr/config").exists()

    is_inst, ver = installer.check_installed(comp)
    assert is_inst is True
    assert ver == "2.0.0"


def test_export_composite_model_and_recursive_configs(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _provider_package(workspace, "env", "demo-env", ["arcade/seaquest"])
    _provider_package(workspace, "algorithm", "demo-agent", ["offline/agent"])
    _provider_package(workspace, "model", "demo-models", ["demo_blend", "demo_neural", "demo_symbolic"])

    exp_dir = workspace / "src/usr/config" / "experiment" / "suite"
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "_base.yaml").write_text("paradigm: offline_rl\n", encoding="utf-8")
    (exp_dir / "hybrid.yaml").write_text(
        """defaults:
  - suite/_base
experiment_id: hybrid
env: arcade/seaquest
methods:
  blend:
    agent: offline/agent
    model:
      demo_blend:
        neural: demo_neural
        symbolic:
          demo_symbolic:
            threshold: 0.03
""",
        encoding="utf-8",
    )

    exporter = ExperimentExporter(workspace_root=workspace)
    out_zip = tmp_path / "hybrid_export.zip"
    opts = ExportOptions(
        include_environment=True,
        include_models=True,
        output_path=out_zip,
    )

    result = exporter.export("suite/hybrid", options=opts)
    assert result.success is True, result.error
    assert out_zip.is_file()

    with zipfile.ZipFile(out_zip, "r") as zf:
        namelist = zf.namelist()
        assert "configs/experiment/suite/hybrid.yaml" in namelist
        assert "configs/experiment/suite/_base.yaml" in namelist

        manifest = json.loads(zf.read(".theta_component.json").decode("utf-8"))
        assert manifest["agents"] == ["offline/agent"]
        assert {"demo_blend", "demo_neural", "demo_symbolic"} <= set(manifest["models"])
        assert sorted(d["id"] for d in manifest["dependencies"]["components"]) == [
            "demo-agent", "demo-env", "demo-models"]


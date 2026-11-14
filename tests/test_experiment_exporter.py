"""Tests for ExperimentExporter service and HubInstaller roundtrip."""
import json
from pathlib import Path
import tempfile
import zipfile

import pytest

from frontend.hub.installer import HubInstaller
from frontend.hub.models import AuthorInfo, HubComponent, ReleaseInfo
from src.app.services.experiment_exporter import ExperimentExporter, ExportOptions, ExportResult


def _setup_test_experiment(workspace: Path) -> str:
    """Helper to create a minimal experiment in a test workspace."""
    exp_dir = workspace / "src/usr/config" / "experiment" / "demo"
    exp_dir.mkdir(parents=True, exist_ok=True)
    env_dir = workspace / "src/usr/config" / "env"
    env_dir.mkdir(parents=True, exist_ok=True)

    (exp_dir / "_base.yaml").write_text("paradigm: online_rl\n", encoding="utf-8")
    (exp_dir / "quick_test.yaml").write_text(
        "defaults:\n  - demo/_base\nexperiment_id: quick_test\nenv: demo_env\nmethods:\n  test_method:\n    agent: ppo\n",
        encoding="utf-8",
    )
    (env_dir / "demo_env.yaml").write_text("name: demo_env\n", encoding="utf-8")
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
    assert result.success is True
    assert result.archive_path == out_zip
    assert out_zip.is_file()

    # Verify zip archive contents
    with zipfile.ZipFile(out_zip, "r") as zf:
        namelist = zf.namelist()
        assert ".theta_component.json" in namelist
        assert "README.md" in namelist
        assert "src/usr/config/experiment/demo/quick_test.yaml" in namelist
        assert "src/usr/config/experiment/demo/_base.yaml" in namelist
        assert "src/usr/config/env/demo_env.yaml" in namelist

        # Inspect manifest
        manifest_raw = zf.read(".theta_component.json").decode("utf-8")
        manifest = json.loads(manifest_raw)
        assert manifest["kind"] == "experiment"
        assert manifest["version"] == "1.2.3"
        assert manifest["author"]["name"] == "Tester"
        assert manifest["environment"] == "demo_env"
        assert "ppo" in manifest["agents"]


def test_export_nonexistent_experiment(tmp_path):
    workspace_root = Path(__file__).resolve().parents[1]
    exporter = ExperimentExporter(workspace_root=workspace_root)
    result = exporter.export("nonexistent_group/nonexistent_exp", options=ExportOptions(output_path=tmp_path / "dummy.zip"))
    assert result.success is False
    assert "not found" in result.error.lower()


def test_hub_installer_roundtrip(tmp_path):
    workspace = tmp_path / "src_workspace"
    workspace.mkdir()
    exp_name = _setup_test_experiment(workspace)

    exporter = ExperimentExporter(workspace_root=workspace)

    pkg_zip = tmp_path / "demo_pkg.zip"
    res = exporter.export(
        exp_name,
        options=ExportOptions(output_path=pkg_zip, version="2.0.0"),
    )
    assert res.success is True

    # Setup isolated mock workspace
    mock_workspace = tmp_path / "target_workspace"
    mock_data = tmp_path / "target_data"
    mock_workspace.mkdir(parents=True)
    mock_data.mkdir(parents=True)

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

    # Check that experiment yaml was unpacked to target workspace
    dest_exp_yaml = mock_workspace / "src/usr/config" / "experiment" / "demo" / "quick_test.yaml"
    assert dest_exp_yaml.is_file()

    # Check that env config was unpacked to target workspace
    dest_env_yaml = mock_workspace / "src/usr/config" / "env" / "demo_env.yaml"
    assert dest_env_yaml.is_file()

    # Check installer detection
    is_inst, ver = installer.check_installed(comp)
    assert is_inst is True
    assert ver == "2.0.0"


def test_export_composite_model_and_recursive_configs(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    # Create recursive agent config
    agent_dir = workspace / "src/usr/config" / "agent" / "deep"
    agent_dir.mkdir(parents=True, exist_ok=True)
    (agent_dir / "cql.yaml").write_text("name: cql\nalgorithm: cql\n", encoding="utf-8")

    # Create recursive and flat model configs
    model_dir = workspace / "src/usr/config" / "model"
    (model_dir / "symbolic").mkdir(parents=True, exist_ok=True)
    (model_dir / "dueling_resnet.yaml").write_text("name: dueling_resnet\n", encoding="utf-8")
    (model_dir / "symbolic" / "cew.yaml").write_text("name: cew\n", encoding="utf-8")

    # Create recursive env config
    env_dir = workspace / "src/usr/config" / "env" / "atari"
    env_dir.mkdir(parents=True, exist_ok=True)
    (env_dir / "seaquest.yaml").write_text("name: seaquest\n", encoding="utf-8")

    # Create experiment with composite BlendRL submodels
    exp_dir = workspace / "src/usr/config" / "experiment" / "suite"
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "_base.yaml").write_text("paradigm: offline_rl\n", encoding="utf-8")
    (exp_dir / "hybrid.yaml").write_text(
        """defaults:
  - suite/_base
experiment_id: hybrid
env: atari/seaquest
methods:
  cql_blend:
    agent: cql
    model:
      blendrl:
        neural: dueling_resnet
        symbolic:
          cew:
            ecm_dthr: 0.03
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
    assert result.success is True
    assert out_zip.is_file()

    with zipfile.ZipFile(out_zip, "r") as zf:
        namelist = zf.namelist()
        assert "src/usr/config/experiment/suite/hybrid.yaml" in namelist
        assert "src/usr/config/experiment/suite/_base.yaml" in namelist
        assert "src/usr/config/agent/deep/cql.yaml" in namelist
        assert "src/usr/config/model/dueling_resnet.yaml" in namelist
        assert "src/usr/config/model/symbolic/cew.yaml" in namelist
        assert "src/usr/config/env/atari/seaquest.yaml" in namelist

        manifest = json.loads(zf.read(".theta_component.json").decode("utf-8"))
        assert "cql" in manifest["agents"]
        assert "dueling_resnet" in manifest["models"]
        assert "cew" in manifest["models"]
        assert "blendrl" in manifest["models"]


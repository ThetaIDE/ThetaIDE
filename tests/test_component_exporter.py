"""Tests for ComponentExporter service and Theta Hub package compatibility."""
import json
from pathlib import Path
import tempfile
import zipfile

import pytest

from frontend.hub.installer import HubInstaller
from src.app.core.manifest_schema import validate_manifest_file
from src.app.services.component_exporter import (
    ComponentExporter,
    ComponentExportOptions,
    ComponentExportResult,
)


@pytest.fixture
def sample_component_workspace(tmp_path):
    ws = tmp_path / "workspace"
    cfg_dir = ws / "in" / "config" / "model"
    cfg_dir.mkdir(parents=True)
    cfg_file = cfg_dir / "dueling_resnet.yaml"
    cfg_file.write_text("name: dueling_resnet\ntype: dueling_resnet\n", encoding="utf-8")

    code_dir = ws / "src" / "usr" / "models" / "dueling_resnet"
    code_dir.mkdir(parents=True)
    (code_dir / "model.py").write_text("class Model: pass\n", encoding="utf-8")
    (code_dir / ".theta_component.json").write_text(
        json.dumps({
            "id": "dueling_resnet",
            "name": "Dueling ResNet",
            "kind": "model",
            "version": "1.0.0",
            "storage_version":2,"api_version":1,
            "capabilities":[{"id":"dueling_resnet","kind":"model","entry_point":"model.py:Model"}],
            "payload":{"implementations":["model.py"],"resources":[],"configs":{"in/config/model/dueling_resnet.yaml":"profiles/default.yaml"}},
            "config_paths":["in/config/model/dueling_resnet.yaml"],
            "config_path": "in/config/model/dueling_resnet.yaml",
        }),
        encoding="utf-8",
    )
    (code_dir/"profiles").mkdir()
    (code_dir/"profiles/default.yaml").write_bytes(cfg_file.read_bytes())
    return ws, cfg_file


def test_export_model_component_zip(tmp_path, sample_component_workspace):
    ws, cfg_file = sample_component_workspace
    exporter = ComponentExporter(workspace_root=ws)

    out_zip = tmp_path / "test_dueling_resnet_export.zip"
    opts = ComponentExportOptions(
        export_format="zip",
        output_path=out_zip,
        author="Tester",
        version="1.2.3",
        description="Test export of Dueling ResNet model component",
    )

    res = exporter.export(cfg_file, options=opts)
    assert res.success is True
    assert res.export_path == out_zip
    assert out_zip.is_file()

    # Inspect zip contents
    with zipfile.ZipFile(out_zip, "r") as zf:
        namelist = zf.namelist()
        assert ".theta_component.json" in namelist
        assert "README.md" in namelist
        assert any(n.startswith("configs/") and n.endswith(".yaml") for n in namelist)

        # Inspect manifest
        manifest_raw = zf.read(".theta_component.json").decode("utf-8")
        manifest = json.loads(manifest_raw)
        assert manifest["id"] == "dueling_resnet"
        assert manifest["kind"] == "model"
        assert manifest["version"] == "1.2.3"
        assert manifest["author"]["name"] == "Tester"


def test_export_component_folder(tmp_path, sample_component_workspace):
    ws, cfg_file = sample_component_workspace
    exporter = ComponentExporter(workspace_root=ws)

    out_dir = tmp_path / "my_exported_component"
    opts = ComponentExportOptions(
        export_format="folder",
        output_path=out_dir,
        author="Custom Author",
        version="2.0.0",
        description="Standalone export",
        generate_git_scaffold=True,
    )

    res = exporter.export(cfg_file, options=opts)
    assert res.success is True
    assert res.export_path == out_dir
    assert out_dir.is_dir()

    assert (out_dir / ".theta_component.json").is_file()
    assert (out_dir / "README.md").is_file()
    assert (out_dir / ".gitignore").is_file()
    assert (out_dir / "pyproject.toml").is_file()
    assert (out_dir / "configs/model/dueling_resnet.yaml").is_file()
    assert not {"legacy_paths","module_aliases","target_path"} & res.manifest.keys()

    # Validate manifest against schema
    is_valid, err = validate_manifest_file(out_dir / ".theta_component.json")
    assert is_valid is True, f"Manifest validation error: {err}"


def test_export_nonexistent_component(tmp_path, sample_component_workspace):
    ws, _ = sample_component_workspace
    exporter = ComponentExporter(workspace_root=ws)
    res = exporter.export("in/config/model/nonexistent_xyz.yaml", options=ComponentExportOptions(output_path=tmp_path / "dummy.zip"))
    assert res.success is False
    assert "not found" in res.error.lower()


def test_hub_installer_can_install_exported_component(tmp_path, sample_component_workspace):
    ws, cfg_file = sample_component_workspace
    exporter = ComponentExporter(workspace_root=ws)

    out_zip = tmp_path / "dueling_resnet_pkg.zip"
    opts = ComponentExportOptions(
        export_format="zip",
        output_path=out_zip,
        author="Theta Team",
        version="1.0.0",
        description="Reusable Dueling ResNet architecture",
    )
    res = exporter.export(cfg_file, options=opts)
    assert res.success is True

    # Setup isolated target workspace
    target_ws = tmp_path / "client_workspace"
    (target_ws / "in" / "config").mkdir(parents=True)
    installer = HubInstaller(workspace_dir=target_ws, data_dir=tmp_path / ".thetaide")

    # Install the exported archive
    from frontend.hub.models import HubComponent, ReleaseInfo, AuthorInfo
    comp = HubComponent(
        id="dueling_resnet",
        name="Dueling ResNet",
        kind="model",
        version="1.0.0",
        description="Model component",
        author=AuthorInfo(name="Theta Team"),
        releases={"1.0.0": ReleaseInfo(tag="1.0.0", url=f"file://{out_zip}", sha256="")},
    )
    installed = installer.install(comp)
    assert installed is True

    # Check installed assets
    assert any((target_ws / "in" / "config").rglob("dueling_resnet.yaml"))
    assert (target_ws / "src/plugins/components/installed/dueling_resnet").is_dir()
    assert (target_ws / "src/plugins/components/installed/dueling_resnet" / ".theta_component.json").is_file()


def test_export_plotter_component_zip(tmp_path):
    ws = tmp_path / "workspace"
    plot_dir = ws / "src/usr/plotters/pareto"
    plot_dir.mkdir(parents=True)
    cfg_dir = ws / "in" / "config" / "plot"
    cfg_dir.mkdir(parents=True)
    cfg_file = cfg_dir / "pareto.yaml"
    cfg_file.write_text("metrics: ['eval/reward']\n", encoding="utf-8")
    (plot_dir / "pareto_plotter.py").write_text("class Plotter: pass\n", encoding="utf-8")
    (plot_dir / ".theta_component.json").write_text(
        json.dumps({
            "id": "pareto",
            "name": "ParetoPlotter",
            "kind": "plotter",
            "version": "1.0.0",
            "storage_version":2,"api_version":1,
            "capabilities":[{"id":"pareto","kind":"plotter","entry_point":"pareto_plotter.py:Plotter"}],
            "payload":{"implementations":["pareto_plotter.py"],"resources":[],"configs":{"in/config/plot/pareto.yaml":"profiles/default.yaml"}},
            "config_paths":["in/config/plot/pareto.yaml"],
            "config_path": "in/config/plot/pareto.yaml",
        }),
        encoding="utf-8",
    )

    (plot_dir/"profiles").mkdir()
    (plot_dir/"profiles/default.yaml").write_bytes(cfg_file.read_bytes())
    exporter = ComponentExporter(workspace_root=ws)
    out_zip = tmp_path / "pareto_plotter.zip"
    opts = ComponentExportOptions(
        export_format="zip",
        output_path=out_zip,
        author="Visualizer",
        version="1.0.0",
        description="Pareto plotter component",
    )
    res = exporter.export(cfg_file, options=opts)
    assert res.success is True
    assert res.export_path == out_zip
    assert out_zip.is_file()

    with zipfile.ZipFile(out_zip, "r") as zf:
        namelist = zf.namelist()
        assert ".theta_component.json" in namelist
        assert any(n.startswith("configs/") and n.endswith(".yaml") for n in namelist)
        manifest = json.loads(zf.read(".theta_component.json").decode("utf-8"))
        assert manifest["id"] == "pareto"
        assert manifest["kind"] == "plotter"


def test_export_namespaced_component(tmp_path):
    ws = tmp_path / "workspace"
    cfg_dir = ws / "in" / "config" / "model" / "basics"
    cfg_dir.mkdir(parents=True)
    cfg_file = cfg_dir / "mlp.yaml"
    cfg_file.write_text("name: mlp\ntype: mlp\n", encoding="utf-8")

    code_dir = ws / "src" / "usr" / "models" / "basics" / "mlp"
    code_dir.mkdir(parents=True)
    (code_dir / "mlp_model.py").write_text("class MLP: pass\n", encoding="utf-8")
    (code_dir/'profiles').mkdir()
    (code_dir/'profiles/default.yaml').write_bytes(cfg_file.read_bytes())
    (code_dir/'.theta_component.json').write_text(json.dumps({'id':'mlp','name':'MLP','kind':'model','version':'1.0.0','storage_version':2,'api_version':1,
        'capabilities':[{'id':'mlp','kind':'model','entry_point':'mlp_model.py:MLP'}],
        'payload':{'implementations':['mlp_model.py'],'resources':[],'configs':{'in/config/model/basics/mlp.yaml':'profiles/default.yaml'}},
        'config_paths':['in/config/model/basics/mlp.yaml']}))

    exporter = ComponentExporter(workspace_root=ws)
    out_zip = tmp_path / "mlp_export.zip"
    opts = ComponentExportOptions(
        export_format="zip",
        output_path=out_zip,
        author="Tester",
        version="1.0.0",
    )

    res = exporter.export(cfg_file, options=opts)
    assert res.success is True
    assert out_zip.is_file()

    with zipfile.ZipFile(out_zip, "r") as zf:
        namelist = zf.namelist()
        assert ".theta_component.json" in namelist
        assert any(n.startswith("configs/") and n.endswith(".yaml") for n in namelist)
        assert json.loads(zf.read(".theta_component.json"))["payload"]["configs"]["in/config/model/basics/mlp.yaml"] == "configs/model/basics/mlp.yaml"
        assert any("mlp_model.py" in n for n in namelist)



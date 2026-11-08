"""Tests for Experiment Plugin abstraction: grouping, export, Hub display, and installation."""
from pathlib import Path
import json
import shutil
import tempfile
import pytest

from src.app.services.experiment_exporter import ExperimentExporter, ExportOptions
from frontend.hub.models import HubComponent
from frontend.hub.installer import HubInstaller
from frontend.hub.dialog import HubView
from frontend.hub.client import HubClient
from PyQt6.QtWidgets import QApplication

# A small registry index, so the Hub tests don't depend on a developer's .thetaide cache
HUB_INDEX = Path(__file__).resolve().parent / "fixtures" / "hub_index.json"


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app



@pytest.fixture
def mock_isolated_workspace():
    """Create a temporary isolated workspace for testing experiment plugin install/uninstall."""
    temp_dir = tempfile.mkdtemp(prefix="theta_exp_ws_")
    ws = Path(temp_dir)
    (ws / "in" / "config" / "experiment").mkdir(parents=True)
    (ws / "in" / "config" / "env").mkdir(parents=True)
    (ws / "in" / "config" / "data").mkdir(parents=True)
    (ws / "in" / "config" / "eval").mkdir(parents=True)
    (ws / "in" / "config" / "plot").mkdir(parents=True)
    (ws / "in" / "envs").mkdir(parents=True)
    (ws / "plot").mkdir(parents=True)
    (ws / "src" / "usr" / "data").mkdir(parents=True)
    (ws / "src" / "usr" / "eval").mkdir(parents=True)
    (ws / ".thetaide").mkdir(parents=True)
    yield ws
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def native_export_workspace(fixture_workspace):
    from src.app.workspace import WorkspacePaths
    paths=WorkspacePaths(fixture_workspace)
    for kind,name,aliases in [('model','mlp_classifier',[]),('model','dnn',[]),('algorithm','ppo',[]),('env','cartpole',['gym_classic_control/cartpole'])]:
        root=paths.user(kind)/name;root.mkdir(parents=True,exist_ok=True)
        (root/'provider.py').write_text('class Provider: pass\n')
        (root/'.theta_component.json').write_text(json.dumps({'id':name,'name':name,'kind':kind,'version':'1.0.0','storage_version':2,'api_version':1,
            'capabilities':[{'id':name,'aliases':aliases,'kind':kind,'entry_point':'provider.py:Provider','resource_root':'.'}],
            'payload':{'implementations':['provider.py'],'resources':[],'configs':{}}}))
    return fixture_workspace


def test_experiment_exporter_export_group(native_export_workspace):
    """Test exporting an entire experiment group as an installable Experiment Plugin."""
    ws = native_export_workspace
    exporter = ExperimentExporter(workspace_root=ws)

    # Export tabular_classification group
    opts = ExportOptions(
        include_checkpoints=False,
        include_environment=True,
        include_models=True,
        include_evaluators=True,
        include_plotters=True,
        include_data_modules=True,
        include_all_group_experiments=True,
        author="Theta Team",
        version="1.0.0",
    )
    res = exporter.export_group("tabular_classification", options=opts)
    assert res.success, f"Failed to export tabular_classification group: {res.error}"
    assert res.archive_path is not None and res.archive_path.is_file()
    assert res.manifest is not None
    assert res.manifest["id"] == "tabular_classification"
    assert res.manifest["kind"] == "experiment"
    assert "in/config/experiment/tabular_classification/_base.yaml" in res.manifest["config_paths"]
    assert "in/config/experiment/tabular_classification/quick_test.yaml" in res.manifest["config_paths"]

    # Clean up exported test zip
    if res.archive_path.exists():
        res.archive_path.unlink()


def test_experiment_exporter_export_thetaide_group(native_export_workspace):
    """Test exporting thetaide experiment group."""
    ws = native_export_workspace
    exporter = ExperimentExporter(workspace_root=ws)

    opts = ExportOptions(
        include_checkpoints=False,
        include_environment=True,
        include_models=True,
        include_evaluators=True,
        include_plotters=True,
        include_all_group_experiments=True,
        version="1.0.0",
    )
    res = exporter.export_group("thetaide", options=opts)
    assert res.success, f"Failed to export thetaide group: {res.error}"
    assert res.manifest["id"] == "thetaide"
    assert res.manifest["kind"] == "experiment"
    assert "in/config/experiment/thetaide/_base.yaml" in res.manifest["config_paths"]
    assert "in/config/experiment/thetaide/cartpole_ppo_reference.yaml" in res.manifest["config_paths"]

    if res.archive_path.exists():
        res.archive_path.unlink()


def test_hub_experiment_filtering_and_hidden_subcomponents(qapp, tmp_path):
    """Verify that experiment plugins are shown under 'experiment' tab, while subcomponents are hidden."""
    ws = Path(__file__).resolve().parent.parent
    client = HubClient(workspace_dir=ws, data_dir=tmp_path / ".thetaide")
    data = json.loads(HUB_INDEX.read_text(encoding="utf-8"))
    client.components = [HubComponent.from_dict(c) for c in data["components"]]

    # 1. Component Hub view
    view_comp = HubView(client=client, mode="components")
    view_comp._populate_cards(client.components)

    comp_ids = set(view_comp.cards.keys())
    assert "mimic" in comp_ids
    assert "pyrenees" in comp_ids

    # Subcomponents must NOT be scattered in the Hub
    assert "mimic-env" not in comp_ids
    assert "mimic-clinical-suite" not in comp_ids
    assert "mimic-early-prediction" not in comp_ids
    assert "reward-shaping" not in comp_ids
    assert "clinical-timeseries-data" not in comp_ids
    assert "pyrenees-env" not in comp_ids
    assert "pyrenees-benchmark-suite" not in comp_ids
    assert "pyrenees-evaluator" not in comp_ids

    # Ensure Experiments pill exists and filters properly
    assert "experiment" in view_comp.pills
    view_comp._select_kind("experiment")
    exp_cards = {cid: card for cid, card in view_comp.cards.items() if not card.isHidden()}
    assert "mimic" in exp_cards
    assert "pyrenees" in exp_cards
    for card in exp_cards.values():
        assert card.component.kind == "experiment"

    view_comp.deleteLater()


def test_experiment_plugin_mock_install_and_clean_uninstall(mock_isolated_workspace):
    """Test full lifecycle install and clean uninstall of an experiment plugin."""
    ws = mock_isolated_workspace
    installer = HubInstaller(workspace_dir=ws, data_dir=ws / ".thetaide")

    dist_pkg = Path(__file__).resolve().parents[2] / "theta-plugins" / "dist" / "pyrenees-1.0.0.zip"
    try:
        if not dist_pkg.is_file():
            pytest.skip("pyrenees-1.0.0.zip dist package not built")
        with dist_pkg.open("rb"):
            pass
    except (OSError, PermissionError):
        pytest.skip("pyrenees-1.0.0.zip dist package not accessible")

    # Load component definition from index
    data = json.loads(HUB_INDEX.read_text(encoding="utf-8"))
    pyrenees_data = next(c for c in data["components"] if c["id"] == "pyrenees")
    comp = HubComponent.from_dict(pyrenees_data)

    comp.releases["1.0.0"].url = f"file://{dist_pkg.resolve()}"

    # This historical archive contains in/rules, which the rules contract explicitly rejects.
    with pytest.raises(ValueError, match="Unsupported"):
        installer.install(comp)
    assert not installer.resolve_target_dir(comp).exists()
    assert not (ws / 'src/usr/eval/pyrenees_evaluator.py').exists()


def test_canonical_experiment_bundle_manages_one_owned_payload(tmp_path):
    from zipfile import ZipFile
    from src.app.components.models import ReleaseInfo
    workspace = tmp_path / 'workspace'
    installer = HubInstaller(workspace, tmp_path / 'state')
    configs = ['in/config/experiment/demo/_base.yaml', 'in/config/experiment/demo/final.yaml', 'in/config/env/demo.yaml']
    meta = {'id':'demo-suite','name':'Demo','kind':'experiment','version':'1.0.0','storage_version':2,'api_version':1,
            'payload':{'configs':{path:path for path in configs}, 'implementations':['env/valuation.py','evaluation.py','plotter.py'], 'resources':['rules/clauses.txt']},
            'capabilities':[{'id':'demo_eval','kind':'evaluator','entry_point':'evaluation.py:Evaluator'}]}
    archive = tmp_path / 'demo.zip'
    with ZipFile(archive,'w') as z:
        z.writestr('.theta_component.json',json.dumps(meta))
        for path in configs: z.writestr(path,'name: demo\n')
        z.writestr('env/valuation.py','# predicates\n')
        z.writestr('evaluation.py','class Evaluator: pass\n')
        z.writestr('plotter.py','# plotting provider\n')
        z.writestr('rules/clauses.txt','example(X).\n')
    component=HubComponent.from_dict(meta)
    component.releases={'1.0.0':ReleaseInfo('1.0.0',archive.as_uri(),'')}
    installer.install(component,install_dependencies=False)
    target=installer.resolve_target_dir(component)
    assert (target/'env/valuation.py').is_file() and (target/'evaluation.py').is_file()
    assert not (workspace/'src/usr').exists() and not (workspace/'in/envs').exists()
    owner=installer.find_component_for_path(workspace/configs[1]);assert owner['id']=='demo-suite'
    installer.uninstall_by_metadata(owner)
    assert not target.exists() and not (workspace/configs[1]).exists()

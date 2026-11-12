"""Public AL recipe readiness and the real GUI Run guard; no training in AL root."""
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from pathlib import Path
from types import SimpleNamespace
import yaml
from PyQt6.QtWidgets import QApplication,QPushButton
from frontend.app import Window
from frontend.config_model import ConfigTree
from src.app.pipeline.compose import compose_experiment,effective_config
from src.app.pipeline.validation import find_missing_components
from src.app.services import config_service
from test_themes_integration import workspace
from test_themes_csv_loader import csv_fixture


def test_public_recipe_has_resolved_dataset_and_enabled_run_button(workspace,monkeypatch):
    folder=workspace.root/'in/datasets/AL/codex';folder.mkdir(parents=True)
    csv_fixture(folder/'MIMIC_md_expert_Q10_120.csv')
    monkeypatch.setenv('THEMES_DATASETS_DIR',str(folder))
    monkeypatch.setattr(config_service,'CONFIG_DIR',workspace.config)
    from src.app.pipeline import compose
    monkeypatch.setattr(compose,'CONFIG_DIR',workspace.config)
    recipe='AL/themes_mimic_q10_120'
    path=workspace.config/'experiment'/f'{recipe}.yaml'
    raw=path.read_text().replace('../../THEMES/datasets/MIMIC_md_expert_Q10_120.csv',
        '${oc.env:THEMES_DATASETS_DIR,../../THEMES/datasets}/MIMIC_md_expert_Q10_120.csv')
    path.write_text(raw)
    # A supported interpolated override must pass the same Run guard as the relative default.
    report=config_service.compose_report(recipe)
    assert report['valid'],report['errors']
    assert '${' not in report['config']['dataset_path']
    assert find_missing_components(report['config'])==[]
    app=QApplication.instance() or QApplication([])
    button=QPushButton()
    viewer=SimpleNamespace(raw_data=yaml.safe_load(raw),_get_explicit_env=lambda:None,_get_base_env=lambda:None)
    host=SimpleNamespace(current_experiment=recipe,compose_valid=report['valid'],compose_error=None,
        composed_config=report['config'],config_viewer=viewer,
        config_tree=SimpleNamespace(tree_model=ConfigTree(workspace.config)),
        active_runs=[],start_button=button,builder_status=None,state_label=None)
    assert Window.check_missing_components(host)==[]
    Window.update_launch_state(host)
    assert button.isEnabled(),button.toolTip()
    assert button.text()=='Queue and run'
    # Dirty editor previews must retain the same readiness as saved YAML.
    draft=config_service.compose_report(recipe,None,raw)
    assert draft['valid'],draft['errors']
    assert find_missing_components(draft['config'])==[]
    button.deleteLater();app.processEvents()


def test_hydra_sweep_placeholders_do_not_poison_effective_config(workspace):
    composed=compose_experiment('AL/codex/themes_paper_synthetic',workspace=workspace)
    data=effective_config(composed.cfg)
    assert 'hydra' not in data
    assert data['experiment_name']=='AL/codex/themes_paper_synthetic'
    assert data['experiment_id']=='codex/themes_paper_synthetic'


def test_only_ready_recipe_is_exposed_at_al_root(workspace):
    names={p.name for p in (workspace.config/'experiment/AL').glob('*.yaml') if not p.name.startswith('_')}
    assert names=={'themes_mimic_q10_120.yaml'}
    internal=workspace.config/'experiment/AL/codex'
    assert (internal/'themes_paper_synthetic.yaml').is_file()
    assert (internal/'themes_mimic_q10_120_port.yaml').is_file()

"""External file errors must not be presented as missing Hub components."""
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from src.app.pipeline.datasets import resolve_dataset_for_method
from src.app.pipeline.compose import compose_experiment, validate_composed
from src.app.pipeline.exceptions import ConfigurationError
from src.app.pipeline.validation import find_missing_components
from src.app.workspace import WorkspacePaths


def test_explicit_missing_path_does_not_fall_back_to_other_data(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'in/datasets/AL/demo').mkdir(parents=True)
    explicit = tmp_path / 'downloaded/missing.csv'
    with pytest.raises(FileNotFoundError, match='Check dataset_path') as error:
        resolve_dataset_for_method('ticc_em_edm', {}, {'dataset_path': str(explicit), 'group': 'AL', 'experiment_id': 'demo'})
    assert str(explicit) in str(error.value)


def test_missing_leading_slash_is_reported_without_guessing(tmp_path):
    source = tmp_path / 'data.csv'
    source.write_text('fixture')
    bad_path = str(source).lstrip('/')
    with pytest.raises(FileNotFoundError, match='missing its leading'):
        resolve_dataset_for_method('ticc_em_edm', {}, {'dataset_path': bad_path})


def test_home_expansion_and_absolute_paths(tmp_path, monkeypatch):
    source = tmp_path / 'data.csv'
    source.write_text('fixture')
    # Do not repurpose HOME; mock pathlib's expansion dependency for this check.
    from unittest.mock import patch
    with patch('os.path.expanduser', return_value=str(tmp_path)):
        assert resolve_dataset_for_method('method', {}, {'dataset_path': '~/data.csv'}) == source
    assert resolve_dataset_for_method('method', {}, {'dataset_path': str(source)}) == source


def test_inventory_names_external_path_not_method():
    source = '/no-such-dataset-folder/visits.csv'
    missing = find_missing_components({'paradigm': 'apprenticeship', 'dataset_path': source,
                                      'methods': {'ticc_em_edm': {'agent': 'themes-dev/em_edm', 'model': 'themes-dev/ticc'}}})
    assert f'dataset: {source}' in missing
    assert 'dataset: ticc_em_edm' not in missing


def test_al_preflight_reports_dataset_file_without_hub_guidance(tmp_path):
    paths = WorkspacePaths(Path(__file__).resolve().parents[1])
    source = tmp_path / 'missing.csv'
    cfg = compose_experiment('AL/themes_mimic_q10_120', [f'++dataset_path={source}'], workspace=paths)
    with pytest.raises(ConfigurationError) as error:
        validate_composed(cfg)
    message = str(error.value)
    assert str(source) in message
    assert 'dataset_path' in message
    assert 'Component Hub' not in message
    assert 'dataset: ticc_em_edm' not in message


def test_relative_dataset_is_resolved_from_workspace_not_cwd(tmp_path, monkeypatch):
    project = tmp_path / 'Theta/Theta-IDE'
    project.mkdir(parents=True)
    external = tmp_path / 'THEMES/datasets'
    external.mkdir(parents=True)
    source = external / 'MIMIC_md_expert_Q10_120.csv'
    source.write_text('fixture')
    elsewhere = tmp_path / 'elsewhere'
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    monkeypatch.setenv('THETA_WORKSPACE', str(project))
    relative = '../../THEMES/datasets/MIMIC_md_expert_Q10_120.csv'
    assert resolve_dataset_for_method('ticc_em_edm', {}, {'dataset_path': relative}) == source.resolve()


def test_explicit_workspace_argument_controls_dataset_resolution(tmp_path, monkeypatch):
    project = tmp_path / 'project'
    project.mkdir()
    source = project / 'data.csv'
    source.write_text('fixture')
    monkeypatch.chdir(tmp_path)
    assert resolve_dataset_for_method('method', {}, {'dataset_path': 'data.csv'}, paths=WorkspacePaths(project)) == source.resolve()


def test_draft_preview_keeps_dataset_relative_to_project_not_temporary_config(tmp_path):
    import os
    import shutil
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/usr/packages/themes-dev'))
    from themes_al.data import PHYSIOLOGY
    project = Path(__file__).resolve().parents[1]
    draft_config = tmp_path / 'preview/config'
    shutil.copytree(project / 'in/config', draft_config)
    source = tmp_path / 'external.csv'
    source.write_text(','.join(['VisitIdentifier', 'MinutesFromArrival', *PHYSIOLOGY, 'a_2']) + '\n')
    relative = os.path.relpath(source, project)
    cfg = compose_experiment('AL/themes_mimic_q10_120', [f'++dataset_path={relative}'],
                             config_dir=draft_config, workspace=WorkspacePaths(project))
    validate_composed(cfg)

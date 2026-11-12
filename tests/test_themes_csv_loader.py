"""Artificial CSV fixtures exercise the custom data provider; no clinical data in CI."""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / 'src/usr/packages/themes-dev'
sys.path.insert(0, str(PACKAGE))
from themes_al.csv_data import load_csv
from themes_al.data import PHYSIOLOGY
from themes_al.artifacts import verify
from themes_al.demo import run
from src.app.core.provider_catalog import get_catalog
from src.app.core.provider_loader import load_capability
from src.app.workspace import WorkspacePaths


def csv_fixture(path, visits=24):
    rng = np.random.default_rng(19)
    rows = []
    for visit in range(visits):
        for time in [120, 0, 60, 180, 240, 300]:
            row = dict(zip(PHYSIOLOGY, rng.normal(size=len(PHYSIOLOGY))))
            row.update(VisitIdentifier=f'{visit:04d}', MinutesFromArrival=time,
                       a_0=0, a_1=0, a_2=int(time % 120 == 0), a_3=0,
                       Action=2, Shock=np.nan)
            rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)
    return frame


def loader(**params):
    paths = WorkspacePaths(ROOT)
    selected = get_catalog(paths).resolve('datamodule', 'themes-dev/csv_trajectories')
    return load_capability(*selected, paths=paths)(**params)


def test_provider_owns_ordering_action_mapping_lineage_and_preprocessing(tmp_path):
    path = tmp_path / 'visits.csv'
    frame = csv_fixture(path)
    source = loader()
    data = source.load(path, 0, 42)
    prepared = source.prepare(data, 42)
    assert data['vids'][0] == '0000'
    np.testing.assert_array_equal(data['timestamps'][0], [0, 60, 120, 180, 240, 300])
    np.testing.assert_array_equal(data['slots'][0], [1, 2, 0, 3, 4, 5])
    np.testing.assert_array_equal(data['actions'][0], frame.a_2.iloc[[1, 2, 0, 3, 4, 5]])
    assert data['source_schema']['action_column'] == 'a_2'
    assert set(data['observation_feature_names']) == set(PHYSIOLOGY)
    assert not data['outcome_available']
    assert (data['outcomes'] == -1).all()
    assert prepared['policy_states'].shape == (len(frame), 42)
    train = prepared['row_train']
    np.testing.assert_allclose(prepared['mean'], prepared['raw'][train].mean(axis=0))
    assert not prepared['valid'][prepared['offsets'][:-1]].any()


@pytest.mark.parametrize('change,match', [
    ('fractional_action', 'binary'), ('missing_feature', 'missing columns'),
    ('bad_time', 'Non-finite'), ('duplicate_time', 'Duplicate timestamps'),
    ('bad_state', 'Non-finite')])
def test_bad_sources_are_rejected(tmp_path, change, match):
    path = tmp_path / 'bad.csv'
    frame = csv_fixture(path)
    if change == 'fractional_action': frame.loc[0, 'a_2'] = 0.5
    if change == 'missing_feature': frame = frame.drop(columns=[PHYSIOLOGY[0]])
    if change == 'bad_time': frame.loc[0, 'MinutesFromArrival'] = np.nan
    if change == 'duplicate_time': frame.loc[0, 'MinutesFromArrival'] = 0
    if change == 'bad_state': frame.loc[0, PHYSIOLOGY[0]] = np.inf
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match=match):
        loader().load(path, 0, 42)


def test_loader_parameters_and_actual_person_split_are_applied(tmp_path):
    path = tmp_path / 'custom.csv'
    frame = csv_fixture(path)
    frame = frame.rename(columns={'VisitIdentifier': 'visit', 'MinutesFromArrival': 'minutes', 'a_2': 'binary_action'})
    frame['person'] = np.repeat([f'p{i // 2}' for i in range(24)], 6)
    frame.to_csv(path, index=False)
    source = loader(visit_column='visit', time_column='minutes', action_column='binary_action', person_column='person', test_fraction=0.25)
    data = source.load(path, 0, 42)
    prepared = source.prepare(data, 42)
    train = set(data['person_ids'][prepared['splits'] == 'train'])
    test = set(data['person_ids'][prepared['splits'] == 'test'])
    assert not train & test
    assert (prepared['splits'] == 'test').sum() == 6
    assert data['action_feature'] == 'binary_action'


def test_loader_rejects_action_or_outcome_as_a_feature(tmp_path):
    path = tmp_path / 'unsafe.csv'
    csv_fixture(path)
    for name in ['a_2', 'Shock', 'Action', 'MinutesFromArrival', 'VisitIdentifier']:
        with pytest.raises(ValueError, match='cannot include'):
            loader(feature_columns=[name]).load(path, 0, 42)


def test_missing_shock_and_multiclass_action_do_not_change_policy_inputs(tmp_path):
    path = tmp_path / 'visits.csv'
    frame = csv_fixture(path)
    first = loader().load(path, 0, 42)
    frame['Shock'] = 1
    frame['Action'] = 123
    frame.to_csv(path, index=False)
    second = loader().load(path, 0, 42)
    for a, b in zip(first['observations'], second['observations']):
        np.testing.assert_array_equal(a, b)
    for a, b in zip(first['actions'], second['actions']):
        np.testing.assert_array_equal(a, b)


def test_csv_model_reload_preserves_timestamps_rows_and_custom_schema(tmp_path):
    import torch
    path = tmp_path / 'visits.csv'
    frame = csv_fixture(path).rename(columns={'a_2': 'binary_action'})
    frame.to_csv(path, index=False)
    source = loader(action_column='binary_action')
    args = SimpleNamespace(dataset=path, output=tmp_path / 'bundle', patients=0, seed=42,
        stages=2, sparsity=0.01, switch_penalty=4., ticc_iterations=1,
        groups=1, min_group_segments=1, em_iterations=1, updates=2, sgld_steps=1, density_weight=0.1)
    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        run(args, data_loader=source.load, prepare_fn=source.prepare, plots=False)
    finally:
        torch.set_num_threads(threads)
    manifest = json.loads((args.output / 'manifest.json').read_text())
    assert manifest['action_feature'] == 'binary_action'
    assert manifest['data_loader'] == 'themes-dev/csv_trajectories'
    assert manifest['source_format'] == 'csv' and manifest['timestamps_available']
    with np.load(args.output / 'grouped_segments.npz', allow_pickle=False) as artifact:
        data = source.load(path, 0, 42)
        np.testing.assert_array_equal(artifact['minutes_from_arrival'], np.concatenate(data['timestamps']))
        np.testing.assert_array_equal(artifact['source_slots'], np.concatenate(data['slots']))
    assert verify(args.output)['source_lineage_checked']
    result = subprocess.run([sys.executable, '-m', 'themes_al', 'verify', '--output', str(args.output)],
        cwd=tmp_path, env={**os.environ, 'PYTHONPATH': str(PACKAGE)}, text=True, capture_output=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr


def test_external_csv_recipes_preflight_and_real_worker_use_custom_loader(tmp_path, monkeypatch):
    import shutil
    from src.app.pipeline.compose import compose_experiment, validate_composed
    workspace = tmp_path / 'workspace'
    shutil.copytree(ROOT / 'src/app/config_defaults', workspace / 'in/config')
    package = workspace / 'src/usr/packages/themes-dev'
    shutil.copytree(PACKAGE, package, ignore=shutil.ignore_patterns('__pycache__', '*.egg-info', 'build'))
    manifest = json.loads((package / '.theta_component.json').read_text())
    for dest, source in manifest['payload']['configs'].items():
        target = workspace / dest
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(package / source, target)
    from test_themes_integration import historical_test_workspace
    historical_test_workspace(workspace)
    external = tmp_path / 'downloaded'
    external.mkdir()
    for name in ['MIMIC_md_expert_Q10_120.csv', 'MIMIC_cleantrain_md_expert_Q10_116.csv']:
        csv_fixture(external / name)
    monkeypatch.setenv('THETA_WORKSPACE', str(workspace))
    monkeypatch.setenv('THEMES_DATASETS_DIR', os.path.relpath(external, workspace))
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv('THETA_DATA_DIR', raising=False)
    monkeypatch.delenv('THETA_PROVIDER_SNAPSHOT', raising=False)
    paths = WorkspacePaths(workspace)
    for name in ['themes_mimic_q10_120', 'themes_mimic_cleantrain_q10_116']:
        composed = compose_experiment('AL/codex/' + name, workspace=paths)
        validate_composed(composed)
        assert composed.cfg.data_module == 'themes-dev/csv_trajectories'
        from src.app.pipeline.datasets import resolve_dataset_for_method
        assert resolve_dataset_for_method('ticc_em_edm', {}, composed.cfg, paths=paths).parent == external.resolve()
    overrides = ['++methods.ticc_em_edm.agent.' + key + '=' + str(value) for key, value in
                 dict(groups=1, min_group_segments=1, em_iterations=1, updates=2, sgld_steps=1).items()]
    overrides += ['++methods.ticc_em_edm.model.stage.stages=2', '++methods.ticc_em_edm.model.stage.ticc_iterations=1']
    result = subprocess.run([sys.executable, str(ROOT / 'run_pipeline.py'), 'AL/codex/themes_mimic_q10_120', *overrides],
        cwd=workspace, env={**os.environ, 'PYTHONPATH': str(ROOT), 'MPLCONFIGDIR': os.environ.get('MPLCONFIGDIR', str(tmp_path / 'mpl'))},
        text=True, capture_output=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    record = json.loads(next((workspace / 'results').rglob('execution.json')).read_text())
    assert record['metadata']['data_provider'] == 'themes-dev/csv_trajectories'
    assert record['status'] == 'completed'
    bundle = Path(record['result']['artifacts'][0]['path'])
    assert json.loads((bundle / 'manifest.json').read_text())['action_feature'] == 'a_2'

"""Normalized study exports remain readable without a tuner or desktop runtime."""
import json
import os
from pathlib import Path
import subprocess
import sys
import shutil

import pytest

from src.app.pipeline.tuning_monitor import local_study_summaries, study_summaries, valid_study_summary


def summary(**extra):
    return {'schema_version': 1, 'component': 'unfamiliar', 'study': 'search',
            'trials': [{'id': 0, 'state': 'running', 'log_version': 'version_trial', 'parameters': {'rate': .1}}], **extra}


def save(root, data, relative='m/tuning_summary.json'):
    path = root / 'results/checkpoints/g/e' / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return path


def test_summaries_are_headless_and_provider_independent(tmp_path):
    path = save(tmp_path, summary())
    before = path.read_bytes()
    script = '''import builtins, json, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
 if name.split('.')[0] in ('PyQt6', 'optuna', 'sqlalchemy', 'frontend'):
  raise AssertionError('summary reading imported ' + name)
 return original(name, *args, **kwargs)
builtins.__import__ = guarded
from src.app.pipeline.tuning_monitor import study_summaries
print(json.dumps(study_summaries(sys.argv[1], {'backend': {'group': 'g', 'experiment_id': 'e'}})))
'''
    result = subprocess.run([sys.executable, '-c', script, str(tmp_path)], text=True, capture_output=True,
                            cwd=Path(__file__).resolve().parents[1])
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)[0][1]['component'] == 'unfamiliar'
    assert path.read_bytes() == before


def test_current_export_wins_same_study_and_archives_remain_distinct(tmp_path):
    save(tmp_path, summary(objective_metric='archived'), 'm/sweeps/old/tuning_summary.json')
    save(tmp_path, summary(study='another'), 'm/sweeps/another/tuning_summary.json')
    current = save(tmp_path, summary(objective_metric='current'))
    studies = local_study_summaries(tmp_path, 'g', 'e')
    assert len(studies) == 2
    assert next(s for _, s in studies if s['study'] == 'search')['objective_metric'] == 'current'
    for path in tmp_path.rglob('tuning_summary.json'):
        os.utime(path, (10, 10))
    os.utime(current, (30, 30))
    assert local_study_summaries(tmp_path, 'g', 'e', started=20) == [('m', summary(objective_metric='current'))]


@pytest.mark.parametrize('data', [[], {}, summary(schema_version=2), summary(schema_version=True),
    summary(component={}), summary(trials=[{'id': 0, 'state': []}]),
    summary(trials=[{'id': True, 'state': 'complete'}]),
    summary(trials=[{'id': 0, 'state': 'complete', 'value': float('nan')}]),
    summary(trials=[{'id': 0, 'state': 'running', 'parameters': []}]),
    summary(trials=[{'id': 0, 'state': 'running'}] * 2),
    summary(requested_trials=-1), summary(parameter_importance={'rate': 'bad'}), summary(direction=None)])
def test_partial_invalid_and_future_exports_are_skipped(tmp_path, data):
    assert not valid_study_summary(data)
    save(tmp_path, data)
    assert local_study_summaries(tmp_path, 'g', 'e') == []


def test_partial_json_and_markerless_legacy_exports(tmp_path):
    path = save(tmp_path, summary())
    path.write_text('{')
    assert local_study_summaries(tmp_path, 'g', 'e') == []
    data = summary()
    data.pop('schema_version')
    path.write_text(json.dumps(data))
    assert study_summaries(tmp_path, {'backend': {'group': 'g', 'experiment_id': 'e'}})[0][1] == data


def test_unknown_provider_states_remain_visible_without_interpretation(tmp_path):
    data = summary(trials=[{'id': 0, 'state': 'awaiting-provider-input'}])
    save(tmp_path, data)
    assert local_study_summaries(tmp_path, 'g', 'e') == [('m', data)]


def test_remote_decoder_requires_readiness_before_import(tmp_path):
    from src.app.pipeline.exceptions import ConfigurationError
    from src.app.pipeline.tuner_execution import TunerExecutionPlan, read_remote_snapshots
    from src.app.workspace import WorkspacePaths
    root = tmp_path / 'src/usr/packages/grid'
    shutil.copytree(Path(__file__).parent / 'fixtures/portable_tuner', root)
    manifest = root / '.theta_component.json'
    data = json.loads(manifest.read_text())
    data['capabilities'][0].update(contribution_api=1, readiness={'config': ['tuning.options.token']})
    manifest.write_text(json.dumps(data))
    (root / 'component.py').write_text('raise AssertionError("unconfigured decoder imported")')
    plan = TunerExecutionPlan('fixture-grid', 'a' * 32)
    record = {'execution_id': 'a' * 32, 'tuner_execution': plan.record(),
              'config': {'tuning': {'component': 'fixture-grid'}}}
    with pytest.raises(ConfigurationError, match='configure tuning.options.token'):
        read_remote_snapshots(tmp_path, record, paths=WorkspacePaths(tmp_path, application=tmp_path / 'empty'))

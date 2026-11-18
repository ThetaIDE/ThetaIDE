"""Portable publication/continuation uses independently named, non-SQLite state."""
import json
from pathlib import Path
import shutil

import pytest
from omegaconf import OmegaConf

from src.app.pipeline.tuner_execution import (TunerArtifact, TunerExecutionPlan, TunerLaunchBindings,
    prepare_plan, record_plan)
from src.app.pipeline.exceptions import ConfigurationError
from src.app.sites.continuation import compatibility
from src.app.sites.publication import Publisher
from src.app.sites.policy import DEFAULTS
from src.app.workspace import WorkspacePaths

FIXTURE = Path(__file__).parent / 'fixtures/portable_tuner'


def install(root):
    target = root / 'src/usr/tuners/fixture-grid'
    shutil.copytree(FIXTURE, target)
    return WorkspacePaths(root)


def test_remote_contract_is_explicit(tmp_path, monkeypatch):
    paths = install(tmp_path)
    cfg = OmegaConf.create({'tuning': {'component': 'fixture-grid', 'options': {'fixture-grid': {'workers': 2}}}})
    class LocalOnly: pass
    monkeypatch.setattr('src.app.pipeline.tuners.get_tuner', lambda *a, **kw: LocalOnly())
    with pytest.raises(ConfigurationError, match='run locally'):
        prepare_plan(cfg, 'a'*32, paths=paths)


def test_compatibility_keeps_objective_and_search_space_exact(tmp_path):
    paths = install(tmp_path)
    cfg = {'tuning': {'component': 'fixture-grid', 'options': {'fixture-grid': {'attempts': 2}}},
           'objective': {'metric': 'score'}, 'methods': {'x': {'tune': {'model.width': [1, 2]}}}}
    other = json.loads(json.dumps(cfg)); other['tuning']['options']['fixture-grid']['attempts'] = 8
    assert compatibility(cfg, paths=paths) == compatibility(other, paths=paths)
    other['objective']['metric'] = 'different'
    assert compatibility(cfg, paths=paths) != compatibility(other, paths=paths)


@pytest.mark.parametrize('path', ['/escape', '../escape', 'grid/state.json', 'grid/other/state.json'])
def test_artifacts_are_scoped_to_the_execution(path):
    with pytest.raises(ValueError):
        TunerExecutionPlan('grid', 'a'*32, artifacts=(TunerArtifact('state', path),))


@pytest.mark.parametrize('bindings', [
    {'environment': {'THETA_WORKSPACE': '/escape'}}, {'environment': {'OMP_NUM_THREADS': '999'}},
    {'overrides': ('++trainer.devices=8',)}, {'overrides': ('not-an-assignment',)},
])
def test_bindings_cannot_override_host_resource_or_workspace_policy(bindings):
    with pytest.raises(ValueError): TunerLaunchBindings(**bindings)


def test_receipts_must_record_their_tuner_plan():
    with pytest.raises(ValueError, match='records no tuner_execution'):
        record_plan({'execution_id': 'a'*32})
    assert record_plan({'execution_id': 'a'*32, 'tuner_execution': None}) is None


def test_final_publication_drains_multiple_artifacts_with_one_queue_slot(tmp_path):
    runtime, work = tmp_path / 'runtime', tmp_path / 'work'
    runtime.mkdir(); work.mkdir(); install(runtime)
    identity = 'a' * 32
    artifacts = (TunerArtifact('one', f'grid/{identity}/first.json'),
                 TunerArtifact('two', f'grid/{identity}/second.csv'))
    plan = TunerExecutionPlan('fixture-grid', identity, artifacts=artifacts)
    record = {'execution_id': identity, 'project_id': 'fixture', 'group': 'g',
              'result_experiment_id': 'e', 'tuner_execution': plan.record()}
    (runtime / 'src/usr/results/sites' / identity).mkdir(parents=True)
    for artifact in artifacts:
        path = WorkspacePaths(runtime).results / artifact.path; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"trials": []}')  # Complete snapshot bytes need no trailing newline.
    publisher = Publisher(runtime, work, record, {**DEFAULTS, 'max_queue_files': 1})
    publisher.finish(seconds=10)
    assert {item['path'] for item in publisher.manifest['files']} == {WorkspacePaths(runtime).results_relative + '/' + item.path for item in artifacts}
    for item in publisher.manifest['files']:
        data = b''.join((work / chunk['source']).read_bytes() for chunk in item['chunks'])
        assert json.loads(data) == {'trials': []}


def test_snapshot_hook_must_publish_a_closed_file(tmp_path, monkeypatch):
    from types import SimpleNamespace
    runtime = tmp_path / 'runtime'; runtime.mkdir(); install(runtime)
    plan = prepare_plan({'tuning': {'component': 'fixture-grid'}}, 'a'*32, paths=WorkspacePaths(runtime))
    record = {'execution_id': 'a'*32, 'project_id': 'fixture', 'group': 'g',
              'result_experiment_id': 'e', 'tuner_execution': plan.record()}
    (runtime / 'src/usr/results/sites' / ('a'*32)).mkdir(parents=True)
    state = WorkspacePaths(runtime).results / plan.artifact('search').path; state.parent.mkdir(parents=True)
    state.write_text('{}')
    publisher = Publisher(runtime, tmp_path / 'work', record, DEFAULTS)
    monkeypatch.setattr('src.app.pipeline.tuner_execution.remote_tuner',
                        lambda *a, **kw: SimpleNamespace(snapshot_artifact=lambda *a, **kw: None))
    with pytest.raises(ValueError, match='regular snapshot'):
        publisher.finish(seconds=1)


@pytest.mark.parametrize('change', [{'schema': 'future'}, {'distributions': 'optuna'}, {'artifacts': 'state'}])
def test_serialized_plan_versions_and_collections_are_validated(change):
    value = TunerExecutionPlan('grid', 'a'*32).record()
    value.update(change)
    with pytest.raises(ValueError): TunerExecutionPlan.from_record(value)



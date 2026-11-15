"""Portable publication/continuation uses independently named, non-SQLite state."""
import builtins
import json
from pathlib import Path
import shutil

import pytest
from omegaconf import OmegaConf

from src.app.pipeline.tuner_execution import (TunerArtifact, TunerExecutionPlan, TunerLaunchBindings,
    prepare_plan, record_plan, read_remote_snapshots)
from src.app.pipeline.exceptions import ConfigurationError
from src.app.sites import execution
from src.app.sites.continuation import compatibility, restore
from src.app.sites.publication import Publisher
from src.app.sites.policy import DEFAULTS
from src.app.sites.snapshot import archive
from src.app.sites.storage import digest, read_json, write_json
from src.app.workspace import WorkspacePaths
from test_ncshare_hardening import managed, fake_submit

FIXTURE = Path(__file__).parent / 'fixtures/portable_tuner'


def install(root):
    from component_fixtures import install_optional
    install_optional(root, 'slurm', root.parent / 'archives')
    target = root / 'src/usr/tuners/fixture-grid'
    shutil.copytree(FIXTURE, target)
    return WorkspacePaths(root)


def refresh_source(root):
    bundle = root / 'fresh.zip'
    source = archive(root, bundle)
    target = root / '.thetaide/sources' / (source['sha256'] + '.zip')
    bundle.replace(target)
    write_json(root / '.thetaide/source.json', {**source, 'archive_sha256': digest(target)})


def test_independent_worker_publication_monitor_and_continuation(tmp_path, monkeypatch):
    script = '''import json, os
from pathlib import Path
p = Path(os.environ['GRID_STATE_PATH'])
p.parent.mkdir(parents=True)
p.write_text(json.dumps({'trials': [{'id': 0, 'state': 'complete', 'value': 7}, {'id': 1, 'state': 'running'}]}))
'''
    root, scratch, work, cfg = managed(tmp_path, script)
    paths = install(root)
    refresh_source(root)
    cfg.tuning = {'component': 'fixture-grid', 'options': {'fixture-grid': {'attempts': 2}}}
    fake_submit(monkeypatch)
    original_import = builtins.__import__
    def block(name, *args, **kwargs):
        if name.split('.')[0] in ('optuna', 'sqlalchemy') or 'bundled.optuna' in name:
            raise AssertionError('Independent tuner imported Optuna')
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', block)
    parent = execution.submit(root, 'g/e', [], cfg)
    assert parent['tuner_execution']['component'] == 'fixture-grid'
    assert 'optuna' not in json.dumps(parent['overrides'])
    assert execution.run(root, parent['execution_id']) == 0
    parent = read_json(execution.receipt_path(root, parent['execution_id']))
    manifest = read_json(work / 'results/sites' / parent['execution_id'] / 'publication.json')
    export = next(item for item in manifest['files'] if item['category'] == 'database')
    assert export['path'].endswith('/search.json')
    assert 'optuna' not in manifest['dependencies']
    # Materialize the verified closed export exactly as the generic downloader does.
    target = root / export['path']; target.parent.mkdir(parents=True)
    target.write_bytes(b''.join((work / chunk['source']).read_bytes() for chunk in export['chunks']))
    assert read_remote_snapshots(root, parent)[0][1]['trials'][0]['value'] == 7
    parent.update(state='timeout', scheduler_state='timeout')
    write_json(execution.receipt_path(root, parent['execution_id']), parent)
    cfg.tuning.options['fixture-grid'].attempts = 5
    cfg.study_continuation = {'execution_id': parent['execution_id'], 'export_sha256': export['sha256'], 'study_name': 'grid'}
    child = execution.submit(root, 'g/e', [], cfg)
    fresh = scratch / 'continued'; fresh.mkdir()
    install(fresh)
    restore(fresh, child)
    plan = record_plan(child)
    continued = read_json(WorkspacePaths(fresh).results / plan.artifact('search').path)
    assert [item['state'] for item in continued['trials']] == ['complete', 'failed']
    assert continued['parent'] == parent['execution_id']
    assert not list(root.rglob('*.sqlite'))


def test_ordinary_remote_execution_has_no_tuner_or_storage_defaults(tmp_path, monkeypatch):
    script = 'import os\nassert "THETA_SQLITE_ROOT" not in os.environ\n'
    root, scratch, work, cfg = managed(tmp_path, script)
    fake_submit(monkeypatch)
    monkeypatch.setattr('src.app.pipeline.tuners.get_tuner', lambda *a, **kw: pytest.fail('Ordinary execution resolved a tuner'))
    record = execution.submit(root, 'g/e', [], cfg)
    assert record['tuner_execution'] is None
    assert record_plan(record) is None
    assert execution.run(root, record['execution_id']) == 0


def test_remote_contract_and_concurrency_are_explicit(tmp_path, monkeypatch):
    paths = install(tmp_path)
    cfg = OmegaConf.create({'tuning': {'component': 'fixture-grid', 'options': {'fixture-grid': {'workers': 2}}}})
    cfg.site = {'scheduler': 'slurm'}
    with pytest.raises(ConfigurationError, match='concurrency=1'):
        execution.validate_allocation(cfg, paths=paths)
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
    (runtime / 'results/sites' / identity).mkdir(parents=True)
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
    (runtime / 'results/sites' / ('a'*32)).mkdir(parents=True)
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


def test_shared_workspace_allocation_applies_portable_tuner_environment(tmp_path, monkeypatch):
    script = '''import os
from pathlib import Path
p=Path(os.environ['GRID_STATE_PATH']); p.parent.mkdir(parents=True)
p.write_text('{"trials": []}')
'''
    root, scratch, work, cfg = managed(tmp_path, script)
    install(root)
    cfg.site.execution = None
    cfg.tuning = {'component': 'fixture-grid'}
    fake_submit(monkeypatch)
    record = execution.submit(root, 'g/e', [], cfg)
    assert execution.run(root, record['execution_id']) == 0
    assert (WorkspacePaths(root).results / record_plan(record).artifact('search').path).is_file()

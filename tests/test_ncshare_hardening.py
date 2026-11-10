"""Filesystem/subprocess acceptance of the managed CPU path; no live cluster calls."""
import json
import os
import shutil
import sqlite3
import stat
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from src.app.sites import execution, transfer
from src.app.sites.policy import DEFAULTS, resolve, stage_inputs
from src.app.sites.publication import Publisher, sqlite_export
from src.app.sites.snapshot import archive, extract
from src.app.sites.storage import digest, read_json, workspace_id, write_json
from src.app.pipeline.exceptions import ConfigurationError
from src.app.pipeline.slurm import generate_sbatch_header


def cfg(**changes):
    value = {'group': 'g', 'experiment_id': 'e', 'site': {'name': 'cluster', 'scheduler': 'slurm',
             'python_executable': sys.executable, 'resources': {'partition': 'common', 'gpus': 0}}}
    value.update(changes)
    return OmegaConf.create(value)


def managed(tmp_path, script=None):
    root, scratch, work = [tmp_path / name for name in ('control', 'scratch', 'work')]
    for path in (root, scratch, work): path.mkdir(parents=True)
    (root / 'run_pipeline.py').write_text(script or '# worker\n')
    workspace_id(root)
    bundle = root / 'source.zip'
    source = archive(root, bundle)
    target = root / '.thetaide/sources' / (source['sha256'] + '.zip')
    target.parent.mkdir(parents=True)
    write_json(target.parent / 'ownership.json', {'project_id': source['project_id']})
    bundle.replace(target)
    write_json(root / '.thetaide/source.json', {**source, 'archive_sha256': digest(target)})
    config = cfg()
    config.site.execution = {**DEFAULTS, 'work_root': str(work), 'scratch_root': str(scratch)}
    return root, scratch, work, config


def fake_submit(monkeypatch):
    monkeypatch.setenv('SLURM_JOB_ID', '123')
    calls = []
    def run(args, **kwargs):
        if args[0] == 'squeue': return SimpleNamespace(returncode=0, stdout='PENDING\n', stderr='')
        assert args[0] == 'sbatch'
        calls.append((args, kwargs))
        return SimpleNamespace(returncode=0, stdout='123\n', stderr='')
    monkeypatch.setattr(subprocess, 'run', run)
    return calls


@pytest.mark.parametrize('resources,flags', [({'gpus': 1, 'gres': 'gpu:h200:8', 'partition': 'gpu'}, []),
    ({'ntasks': 2}, []), ({}, ['--cpus-per-task=8']), ({}, ['-N8']), ({}, ['--requeue']),
    ({}, ['--output=/tmp/leak']), ({}, ['--constraint=node --nodes=8'])])
def test_reject_conflicting_actual_requests(resources, flags):
    config = cfg(resources=resources)
    config.site.extra_sbatch_flags = flags
    with pytest.raises(ConfigurationError): execution.validate_allocation(config)


def test_normalized_request_and_environment(monkeypatch, tmp_path):
    config = cfg(resources={'cores': 3, 'memory': '6G'})
    resources = execution.validate_allocation(config)
    header = generate_sbatch_header('cpu', 'logs', config)
    assert f'--cpus-per-task={resources["cores"]}' in header
    assert '--ntasks=1' in header and '--gres' not in header
    monkeypatch.setenv('SBATCH_GRES', 'gpu:8')
    with pytest.raises(ConfigurationError, match='SBATCH_GRES'):
        execution.submit(tmp_path, 'g/e', [], config)
    assert not (tmp_path / 'results').exists()


@pytest.mark.parametrize('change', [{'tuning': {'options': {'optuna': {'n_jobs': 2}}}}, {'plot_only': True},
                                  {'variability': {'parallel_runs': 2}}])
def test_unsupported_slurm_modes_are_explicit(change):
    with pytest.raises(ConfigurationError): execution.validate_allocation(cfg(**change))


def test_archive_refuses_links_and_traversal_before_writing(tmp_path):
    for name, mode in [('../escape', stat.S_IFREG), ('/escape', stat.S_IFREG), ('link', stat.S_IFLNK)]:
        path = tmp_path / 'bad.zip'
        with zipfile.ZipFile(path, 'w') as bundle:
            entry = zipfile.ZipInfo(name)
            entry.external_attr = (mode | 0o644) << 16
            bundle.writestr(entry, 'bad')
        destination = tmp_path / 'out'
        with pytest.raises(ValueError): extract(path, destination)
        assert not destination.exists()


def test_input_limits_and_fresh_mtimes(tmp_path):
    root, destination = tmp_path / 'root', tmp_path / 'staged'
    root.mkdir(); destination.mkdir()
    (root / 'data').write_bytes(b'payload')
    os.utime(root / 'data', (1, 1))
    policy = {**DEFAULTS, 'inputs': [{'source': 'data', 'target': 'in/datasets/data'}]}
    stage_inputs(root, destination, policy)
    assert (destination / 'in/datasets/data').stat().st_mtime > time.time() - 10
    assert read_json(destination / '.theta-inputs.json')['files'][0]['original_mtime_ns'] == 1000000000
    policy['max_input_bytes'] = 1
    with pytest.raises(ValueError, match='max_input_bytes'): stage_inputs(root, tmp_path / 'other', policy)


def test_scratch_worker_publishes_without_desktop_and_separates_allocations(tmp_path, monkeypatch):
    script = '''import json, os, pathlib, sqlite3, sys
root = pathlib.Path.cwd()
assert root == pathlib.Path(os.environ['THETA_WORKSPACE'])
assert root == pathlib.Path(os.environ['THETA_OUTPUT_ROOT'])
assert 'THETA_RESOURCE_POOL' not in os.environ
name = json.loads(next(a.split('=',1)[1] for a in sys.argv if a.startswith('++experiment_id=')))
p = root / 'results/logs/g' / name / 'm/version_0'
p.mkdir(parents=True)
(p / 'metrics.csv').write_text('step,reward\\n1,7\\n')
db = pathlib.Path(os.environ['THETA_SQLITE_ROOT']) / 'study.sqlite'
db.parent.mkdir(parents=True)
connection = sqlite3.connect(db)
connection.execute('PRAGMA journal_mode=WAL')
connection.execute('CREATE TABLE trials(score)')
connection.execute('INSERT INTO trials VALUES (7)')
connection.commit()
print(root)
'''
    root, scratch, work, config = managed(tmp_path, script)
    fake_submit(monkeypatch)
    first = execution.submit(root, 'g/e', [], config)
    second = execution.submit(root, 'g/e', [], config)
    assert first['source_sha256'] == second['source_sha256']
    assert len(list((root / '.thetaide/sources').glob('*.zip'))) == 1
    assert not list((root / 'results/slurm').glob('*/workspace'))
    for record in (first, second):
        assert execution.run(root, record['execution_id']) == 0
        final = read_json(execution.receipt_path(root, record['execution_id']))
        assert final['publication_state'] == 'sealed' and not final['artifacts_finalized']
        assert final['last_durable']['database']['sha256']
        publication = read_json(work / 'results/slurm' / record['execution_id'] / 'publication.json')
        exported = next(item for item in publication['files'] if item['category'] == 'database')
        copy = tmp_path / (record['execution_id'] + '.sqlite')
        copy.write_bytes(b''.join((work / chunk['source']).read_bytes() for chunk in exported['chunks']))
        with sqlite3.connect(copy) as database:
            assert database.execute('select score from trials').fetchone() == (7,)
            assert database.execute('pragma journal_mode').fetchone() == ('delete',)
    assert not list(scratch.iterdir())
    assert not list(root.rglob('*-wal')) and not list(work.rglob('*-wal'))


def publisher_fixture(tmp_path):
    runtime, work = tmp_path / 'runtime', tmp_path / 'work'
    runtime.mkdir(); work.mkdir()
    record = {'execution_id': 'a' * 32, 'project_id': workspace_id(runtime), 'group': 'g', 'result_experiment_id': 'e__a'}
    (runtime / 'results/slurm' / record['execution_id']).mkdir(parents=True)
    logs = runtime / 'results/logs/g/e__a/m/version_0'
    logs.mkdir(parents=True)
    return runtime, work, record, logs


def test_atomic_publication_chunks_priority_coalescing_and_no_rehash(tmp_path, monkeypatch):
    runtime, work, record, logs = publisher_fixture(tmp_path)
    clock = [0]
    policy = {**DEFAULTS, 'chunk_bytes': 16, 'tick_bytes': 64, 'metrics_seconds': 60}
    publisher = Publisher(runtime, work, record, policy, clock=lambda: clock[0])
    (logs / 'events.jsonl').write_bytes(b'{"step":1}\n{"step":')
    publisher.tick()
    entry = publisher.entries[next(iter(publisher.entries))]
    assert b''.join((work / c['source']).read_bytes() for c in entry['chunks']) == b'{"step":1}\n'
    db = runtime / 'results/optuna' / record['execution_id'] / 'study.sqlite'
    db.parent.mkdir(parents=True)
    with sqlite3.connect(db) as connection: connection.execute('CREATE TABLE trials(score)')
    exports = []
    original = sqlite_export
    def backup(*args, **kwargs): exports.append(clock[0]); return original(*args, **kwargs)
    monkeypatch.setattr('src.app.sites.publication.sqlite_export', backup)
    publisher.tick()
    assert exports == [0]
    # Drain the DB with bounded ticks; no second backup until the spacing expires.
    for _ in range(150): publisher.tick()
    for value in (1, 2, 29):
        clock[0] = value
        write_json(runtime / '.thetaide/db-export-request.json', {'requested_ns': value})
        publisher.tick()
    assert exports == [0]
    clock[0] = 30
    publisher.tick()
    assert exports == [0, 30]
    for _ in range(150): publisher.tick()
    before = dict(publisher.counters)
    publisher.tick()
    assert publisher.counters == before
    # A failed copy never appears as a committed new path.
    (logs / 'metrics.csv').write_text('a,b\n1,2\n')
    clock[0] = 90
    def fail(_): raise OSError('disk full')
    monkeypatch.setattr(publisher, '_object', fail)
    with pytest.raises(OSError): publisher.tick()
    assert not any(item['path'].endswith('metrics.csv') for item in read_json(publisher.manifest_path)['files'])
    publisher.close()


def test_index_never_scans_payload_or_opens_database(tmp_path, monkeypatch):
    root, scratch, work, config = managed(tmp_path)
    fake_submit(monkeypatch)
    record = execution.submit(root, 'g/e', [], config)
    record.update(state='completed', scheduler_state='completed')
    write_json(execution.receipt_path(root, record['execution_id']), record)
    files = [{'path': 'results/checkpoints/g/e/model.ckpt', 'category': 'checkpoints', 'size': 0,
              'sha256': '0' * 64, 'chunks': []}]
    write_json(work / 'results/slurm' / record['execution_id'] / 'publication.json',
               {'schema': 'theta.publication/v1', 'sealed': True, 'files': files})
    monkeypatch.setattr(transfer, 'owned_files', lambda *_: pytest.fail('login scanned runtime'))
    monkeypatch.setattr(transfer, 'digest', lambda *_: pytest.fail('login hashed payload'))
    monkeypatch.setattr(sqlite3, 'connect', lambda *_a, **_k: pytest.fail('login opened SQLite'))
    for _ in range(2):
        listing = transfer.index(root, 'results', record['execution_id'])
        assert listing['files'][0]['sha256'] == '0' * 64
        transfer.release(root, listing['generation'])
    assert not list((root / 'results/slurm/.sync').iterdir())


def test_diagnostic_budget_is_shared_and_optional(tmp_path, monkeypatch):
    from src.app.monitoring.budget import admit, BudgetExceeded
    monkeypatch.setenv('THETA_OUTPUT_ROOT', str(tmp_path))
    monkeypatch.setenv('THETA_DIAGNOSTIC_BUDGET', json.dumps({'diagnostic_bytes': 100, 'diagnostic_files': 10, 'snapshot_bytes': 80}))
    payload = tmp_path / 'payload'
    payload.write_bytes(b'x' * 40)
    admit([(payload, tmp_path / 'trial1.npz')], position=1)
    with pytest.raises(BudgetExceeded, match='diagnostic_bytes'):
        admit([(payload, tmp_path / 'trial2.npz')], position=2)
    admit([(payload, tmp_path / 'trial2.npz')], position=3, final=True)
    state = read_json(tmp_path / '.thetaide/diagnostic-budget.json')
    assert sum(state['files'].values()) == 80 and state['skipped'] == 1


def test_small_diagnostics_bundle_once_and_reconstruct_incrementally(tmp_path, monkeypatch):
    runtime, work, record, logs = publisher_fixture(tmp_path)
    (logs / 'monitors/m').mkdir(parents=True)
    for index in range(5): (logs / f'monitors/m/snapshot_{index}.npz').write_bytes(bytes([index]) * 10)
    publisher = Publisher(runtime, work, record, DEFAULTS)
    publisher.finish(seconds=10)
    diagnostics = [item for item in publisher.manifest['files'] if item['category'] == 'diagnostics']
    assert len(diagnostics) == 5
    assert len({item['chunks'][0]['source'] for item in diagnostics}) == 1
    local = tmp_path / 'local'
    local.mkdir()
    listing = {'project_id': record['project_id'], 'executions': [], 'files': [dict(item, remote_root=str(work)) for item in diagnostics]}
    calls = []
    def pull(args, **kwargs):
        calls.append(args)
        names = Path(next(arg.split('=', 1)[1] for arg in args if arg.startswith('--files-from=')))
        for name in names.read_bytes().decode().strip('\0').split('\0'):
            target = Path(args[-1]) / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(work / name, target)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(subprocess, 'run', pull)
    config = {'host': 'fake', 'remote_root': str(work)}
    messages = []
    monkeypatch.setattr(transfer.transfer_output, 'emit', messages.append)
    first = transfer.sync_objects(local, 'cluster', config, listing)
    assert first['bytes'] == 50 and len(calls) == 1
    assert (local / logs.relative_to(runtime) / 'monitors/m/snapshot_3.npz').read_bytes() == bytes([3]) * 10
    assert any(message.startswith('Saved ') for message in messages)
    messages.clear()
    second = transfer.sync_objects(local, 'cluster', config, listing)
    assert not any(message.startswith('Saved ') for message in messages)
    assert second['transferred'] == 0 and second['bytes'] == 0 and len(calls) == 1


def test_continuation_history_reconciliation_ownership_and_compatibility(tmp_path, monkeypatch):
    optuna = pytest.importorskip('optuna')
    from src.app.sites.continuation import pin, restore
    root, scratch, work, config = managed(tmp_path)
    config.tuning = {'component': 'optuna', 'options': {'optuna': {'n_jobs': 1, 'n_trials': 2}}}
    fake_submit(monkeypatch)
    parent = execution.submit(root, 'g/e', [], config)
    runtime = scratch / 'prior'
    runtime.mkdir()
    (runtime / 'results/slurm' / parent['execution_id']).mkdir(parents=True)
    db = runtime / 'results/optuna' / parent['execution_id'] / 'study.sqlite'
    db.parent.mkdir(parents=True)
    storage = optuna.storages.RDBStorage('sqlite:///' + str(db))
    study = optuna.create_study(storage=storage, study_name='cpu', direction='maximize')
    trial = study.ask(); study.tell(trial, 7)
    running = study.ask()
    publisher = Publisher(runtime, work, parent, parent['execution_policy'])
    publisher.finish(seconds=10)
    export = next(item for item in publisher.manifest['files'] if item['category'] == 'database')
    parent.update(state='timeout', scheduler_state='timeout')
    write_json(execution.receipt_path(root, parent['execution_id']), parent)
    request = {'execution_id': parent['execution_id'], 'export_sha256': export['sha256'], 'study_name': 'cpu'}
    config.study_continuation = request
    child = execution.submit(root, 'g/e', [], config)
    assert child['execution_id'] != parent['execution_id'] and child['result_experiment_id'] != parent['result_experiment_id']
    assert child['continuation']['training_resume'] is False
    fresh = scratch / 'new'; fresh.mkdir()
    restore(fresh, child)
    url = 'sqlite:///' + str(fresh / 'results/optuna' / child['execution_id'] / 'study.sqlite')
    continued = optuna.load_study(storage=url, study_name='cpu')
    assert [trial.state.name for trial in continued.trials] == ['COMPLETE', 'FAIL']
    assert continued.trials[0].value == 7 and continued.ask().number == 2
    with pytest.raises(ValueError, match='owner'): execution.submit(root, 'g/e', [], config)
    modified = dict(child); modified['config'] = {**child['config'], 'objective': {'metric': 'different'}}
    with pytest.raises(ValueError, match='incompatible'): pin(root, modified, request)
    broken = {**request, 'export_sha256': 'f' * 64}
    with pytest.raises(ValueError, match='missing'): pin(root, child, broken)
    # Checkpoint/study dependencies remain pinned even after the new owner terminates.
    child.update(state='completed', scheduler_state='completed')
    write_json(execution.receipt_path(root, child['execution_id']), child)
    parent = execution.finalize(root, parent['execution_id'])
    proof = {'project_id': parent['project_id'], 'files': read_json(execution.receipt_path(root, parent['execution_id']).parent / 'artifacts.json')['files']}
    with pytest.raises(ValueError, match='pins'): transfer.clean_remote(root, parent['execution_id'], proof)
    storage.engine.dispose()


def test_gc_preserves_active_unknown_and_legacy_and_releases_verified_payload(tmp_path, monkeypatch):
    from src.app.sites.maintenance import collect
    root, scratch, work, config = managed(tmp_path)
    fake_submit(monkeypatch)
    record = execution.submit(root, 'g/e', [], config)
    lease = root / 'results/slurm/.sync' / ('b' * 32)
    write_json(lease / '.theta-managed.json', {'project_id': record['project_id'], 'expires_at': time.time() + 7200})
    old = root / 'results/slurm/.sync/legacy'
    old.mkdir()
    bundle = root / '.thetaide/sources' / (record['source_sha256'] + '.zip')
    assert collect(root, apply=True)['candidates'] == []
    assert bundle.exists() and old.exists() and lease.exists()
    orphan = root / '.thetaide/sources' / ('c' * 64 + '.zip')
    orphan.write_bytes(b'garbage')
    assert len(collect(root)['candidates']) == 1 and orphan.exists()
    collect(root, apply=True)
    assert not orphan.exists()
    write_json(lease / '.theta-managed.json', {'project_id': record['project_id'], 'expires_at': 0})
    collect(root, apply=True)
    assert not lease.exists() and old.exists()


def test_scheduler_queries_are_batched_cached_and_preserve_training_outcome(tmp_path, monkeypatch):
    records = []
    for index in range(3):
        record = {'execution_id': f'{index:032x}', 'project_id': workspace_id(tmp_path), 'job_id': str(index + 1),
                  'state': 'failed' if index == 1 else 'running'}
        write_json(execution.receipt_path(tmp_path, record['execution_id']), record)
        records.append(record)
    calls = []
    def scheduler(args, **kwargs):
        calls.append(args[0])
        return SimpleNamespace(stdout='1|RUNNING|\n' if args[0] == 'squeue' else '2|COMPLETED|1:0\n3|TIMEOUT|0:15\n')
    monkeypatch.setattr(subprocess, 'run', scheduler)
    execution.refresh_many(tmp_path, records)
    assert calls == ['squeue', 'sacct']
    assert records[1]['training_state'] == 'failed' and records[1]['scheduler_state'] == 'completed'
    assert records[1]['state'] == 'failed'  # scheduler success cannot overwrite training failure
    execution.refresh_many(tmp_path, records)
    assert len(calls) == 2


@pytest.mark.parametrize('recipe', ['cpu_smoke', 'cpu_study'])
def test_real_cpu_pipeline_in_fresh_scratch_with_frozen_providers(tmp_path, monkeypatch, recipe):
    pytest.importorskip('stable_baselines3')
    from src.app.sites.snapshot import freeze
    from src.app.pipeline.compose import compose_experiment, validate_composed
    from src.app.workspace import WorkspacePaths
    project = Path(__file__).resolve().parents[1]
    root, scratch, work = [tmp_path / name for name in ('control', 'scratch', 'work')]
    for path in (root, scratch, work): path.mkdir()
    freeze(project, root)
    target = root / 'in/config/experiment/ncshare'
    target.mkdir(parents=True, exist_ok=True)
    for name in ('cpu_smoke', 'cpu_study'):
        shutil.copyfile(project / f'docs/development/examples/ncshare_{name}.yaml', target / f'{name}.yaml')
    bundle = root / 'source.zip'
    source = archive(root, bundle)
    destination = root / '.thetaide/sources' / (source['sha256'] + '.zip')
    destination.parent.mkdir(parents=True)
    bundle.replace(destination)
    write_json(root / '.thetaide/source.json', {**source, 'archive_sha256': digest(destination)})
    monkeypatch.setenv('THETA_WORKSPACE', str(root))
    monkeypatch.setenv('SLURM_CPUS_PER_TASK', '2')
    overrides = [f'++site.execution.work_root={work}', f'++site.execution.scratch_root={scratch}',
                 f'++site.python_executable={sys.executable}']
    composed = compose_experiment(f'ncshare/{recipe}', overrides, workspace=WorkspacePaths(root, application=root))
    assert validate_composed(composed) == []
    fake_submit(monkeypatch)
    record = execution.submit(root, composed.experiment, overrides, composed.cfg)
    code = execution.run(root, record['execution_id'])
    publication = read_json(work / 'results/slurm' / record['execution_id'] / 'publication.json')
    log = next(item for item in publication['files'] if item['path'].endswith('pipeline.log'))
    text = b''.join((work / chunk['source']).read_bytes() for chunk in log['chunks']).decode(errors='replace')
    assert code == 0, text
    assert publication['sealed']
    assert any(item['path'].endswith('metrics.csv') for item in publication['files'])
    assert any(item['category'] == 'checkpoints' for item in publication['files'])
    if recipe == 'cpu_study':
        assert publication['durable']['database']['sha256']
        assert any(item['path'].endswith('tuning_summary.json') for item in publication['files'])
        parent = read_json(execution.receipt_path(root, record['execution_id']))
        parent['scheduler_state'] = 'completed'
        write_json(execution.receipt_path(root, record['execution_id']), parent)
        export = next(item for item in publication['files'] if item['category'] == 'database')
        with __import__('omegaconf').open_dict(composed.cfg):
            composed.cfg.study_continuation = {'execution_id': record['execution_id'],
                'export_sha256': export['sha256'], 'study_name': 'ncshare_cpu_pilot'}
            composed.cfg.tuning.options.optuna.n_trials = 1
        continued = execution.submit(root, composed.experiment,
            [*overrides, '++tuning.options.optuna.n_trials=1'], composed.cfg)
        code = execution.run(root, continued['execution_id'])
        manifest = read_json(work / 'results/slurm' / continued['execution_id'] / 'publication.json')
        log = next(item for item in manifest['files'] if item['path'].endswith('pipeline.log'))
        output = b''.join((work / chunk['source']).read_bytes() for chunk in log['chunks']).decode(errors='replace')
        assert code == 0, output
        database = next(item for item in manifest['files'] if item['category'] == 'database')
        copy = tmp_path / 'continued.sqlite'
        copy.write_bytes(b''.join((work / chunk['source']).read_bytes() for chunk in database['chunks']))
        import optuna
        study = optuna.load_study(study_name='ncshare_cpu_pilot', storage='sqlite:///' + str(copy))
        assert len(study.trials) == 4 and all(trial.state.name == 'COMPLETE' for trial in study.trials)


def test_signal_stageout_and_abrupt_unsealed_boundary(tmp_path, monkeypatch):
    import signal
    script = '''import json,os,pathlib,sys,time
root=pathlib.Path.cwd()
name=json.loads(next(a.split('=',1)[1] for a in sys.argv if a.startswith('++experiment_id=')))
p=root/'results/logs/g'/name/'m/version_0'
p.mkdir(parents=True)
(p/'metrics.csv').write_text('step,reward\\n1,7\\n')
print('ready',flush=True)
time.sleep(30)
'''
    root, scratch, work, config = managed(tmp_path, script)
    config.site.execution.metrics_seconds = 1
    fake_submit(monkeypatch)
    record = execution.submit(root, 'g/e', [], config)
    # Use a real separate allocation owner process, not a mocked training Popen.
    owner = subprocess.Popen([sys.executable, '-m', 'src.app.sites', '_run', '--root', str(root), record['execution_id']],
                             env={**os.environ, 'PYTHONPATH': str(Path(__file__).resolve().parents[1])},
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    manifest = work / 'results/slurm' / record['execution_id'] / 'publication.json'
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if any(item['path'].endswith('metrics.csv') for item in read_json(manifest).get('files', [])): break
        time.sleep(.05)
    else:
        owner.kill(); pytest.fail('Allocation did not publish before signal')
    assert not read_json(manifest)['sealed']
    owner.send_signal(signal.SIGTERM)
    out, error = owner.communicate(timeout=10)
    final = read_json(execution.receipt_path(root, record['execution_id']))
    assert final['state'] == 'cancelled', out + error
    assert final['publication_state'] == 'sealed'
    assert final['artifacts_finalized'] is False  # never invent scheduler termination
    # A payload stopped without finish remains unsealed and cannot be finalized.
    final['scheduler_state'] = 'cancelled'
    write_json(execution.receipt_path(root, record['execution_id']), final)
    value = read_json(manifest); value['sealed'] = False; write_json(manifest, value)
    with pytest.raises(ValueError, match='unsealed'): execution.finalize(root, record['execution_id'])


def test_live_wal_backup_is_standalone_and_allocation_storage_is_exclusive(tmp_path, monkeypatch):
    source = tmp_path / 'scratch/study.sqlite'
    source.parent.mkdir()
    writer = sqlite3.connect(source)
    writer.execute('PRAGMA journal_mode=WAL')
    writer.execute('CREATE TABLE trials(score)')
    writer.execute('INSERT INTO trials VALUES(9)')
    writer.commit()
    assert source.with_name(source.name + '-wal').exists()
    target = tmp_path / 'export.sqlite'
    sqlite_export(source, target)
    with sqlite3.connect(target) as exported:
        assert exported.execute('SELECT score FROM trials').fetchone() == (9,)
        assert exported.execute('PRAGMA journal_mode').fetchone() == ('delete',)
    writer.close()
    from src.plugins.components.bundled.optuna.utils import get_optuna_storage
    monkeypatch.setenv('THETA_SQLITE_ROOT', str(source.parent))
    with pytest.raises(ValueError, match='exclusively owned'):
        get_optuna_storage('sqlite:///' + str(tmp_path / 'shared/study.sqlite'))


def test_runtime_budget_failure_blocks_new_trials_and_preserves_manifest(tmp_path, monkeypatch):
    from src.app.sites.publication import admission
    runtime, work, record, logs = publisher_fixture(tmp_path)
    publisher = Publisher(runtime, work, record, {**DEFAULTS, 'max_runtime_bytes': 1})
    (logs / 'metrics.csv').write_text('step,reward\n1,7\n')
    with pytest.raises(RuntimeError, match='storage budget'): publisher.tick()
    monkeypatch.setenv('THETA_PUBLICATION_STOP', str(publisher.stop_path))
    with pytest.raises(RuntimeError, match='no new trials'): admission()
    assert not publisher.manifest.get('sealed')


def test_cancel_transfer_stops_only_transfer_process(tmp_path):
    import threading
    cancelled = threading.Event()
    timer = threading.Timer(.2, cancelled.set)
    timer.start()
    start = time.monotonic()
    try:
        with pytest.raises(RuntimeError, match='cancelled'):
            transfer.run_transfer([sys.executable, '-c', 'import time; time.sleep(30)'], cancel_event=cancelled)
    finally:
        timer.cancel()
    assert time.monotonic() - start < 5


def test_managed_cleanup_requires_full_copy_and_transfer_release(tmp_path, monkeypatch):
    script = '''import json,pathlib,sys
name=json.loads(next(a.split('=',1)[1] for a in sys.argv if a.startswith('++experiment_id=')))
root=pathlib.Path('results/checkpoints/g')/name
root.mkdir(parents=True)
(root/'model.ckpt').write_bytes(b'committed checkpoint')
'''
    root, scratch, work, config = managed(tmp_path, script)
    fake_submit(monkeypatch)
    record = execution.submit(root, 'g/e', [], config)
    assert execution.run(root, record['execution_id']) == 0
    path = execution.receipt_path(root, record['execution_id'])
    value = read_json(path); value['scheduler_state'] = 'completed'; write_json(path, value)
    local = tmp_path / 'local'; local.mkdir()
    write_json(local / '.thetaide/workspace.json', {'id': record['project_id']})
    transfer.configure(local, 'cluster', 'fake', str(root))
    def remote(_root, site, command, *args, **kwargs):
        if command == '_index': return transfer.index(root, args[1])
        if command == '_release': return transfer.release(root, args[0])
        if command == '_renew': return transfer.renew(root, args[0])
        if command == '_clean': return transfer.clean_remote(root, args[0], json.loads(kwargs['input']), preview='--preview' in args)
        pytest.fail(command)
    monkeypatch.setattr(transfer, 'remote', remote)
    def pull(args, **kwargs):
        names = Path(next(arg.split('=', 1)[1] for arg in args if arg.startswith('--files-from=')))
        origin = Path(args[-2].split(':', 1)[1])
        for name in names.read_bytes().decode().strip('\0').split('\0'):
            target = Path(args[-1]) / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(origin / name, target)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(subprocess, 'run', pull)
    transfer.sync(local, 'cluster', 'metrics')
    with pytest.raises(FileNotFoundError): transfer.clean(local, 'cluster', record['execution_id'])
    transfer.sync(local, 'cluster', 'results')
    preview = transfer.clean(local, 'cluster', record['execution_id'], preview=True)
    assert preview['payload_bytes'] > 0 and (work / 'results/slurm' / record['execution_id']).exists()
    lease = transfer.index(root, 'results')
    with pytest.raises(ValueError, match='lease'): transfer.clean(local, 'cluster', record['execution_id'])
    transfer.release(root, lease['generation'])
    assert transfer.clean(local, 'cluster', record['execution_id'])['removed'] > 0
    assert not (work / 'results/slurm' / record['execution_id']).exists()
    assert not (work / 'inputs' / record['execution_id']).exists()
    assert path.exists() and (path.parent / 'artifacts.json').exists()
    assert (local / 'results/checkpoints/g' / record['result_experiment_id'] / 'model.ckpt').read_bytes() == b'committed checkpoint'


def test_collection_serializes_with_submission_and_preserves_foreign_leases(tmp_path):
    from src.app.sites.maintenance import collect
    from src.app.sites.storage import transfer_lock
    workspace_id(tmp_path)
    foreign = tmp_path / 'results/slurm/.sync' / ('c' * 32)
    write_json(foreign / '.theta-managed.json', {'project_id': 'different-project', 'expires_at': 0})
    with transfer_lock(tmp_path):
        with pytest.raises(RuntimeError, match='in progress'): collect(tmp_path, apply=True)
    collect(tmp_path, apply=True)
    assert foreign.exists()


def test_bounded_queue_finalization_drains_deferred_closed_files(tmp_path):
    runtime, work, record, logs = publisher_fixture(tmp_path)
    for index in range(20): (logs / f'closed_{index}.json').write_text(json.dumps({'index': index}))
    publisher = Publisher(runtime, work, record, {**DEFAULTS, 'max_queue_files': 3})
    publisher.tick(final=True)
    assert len(publisher.pending) <= 3 and publisher.deferred > 0
    publisher.finish(seconds=10)
    assert len(publisher.manifest['files']) == 20 and publisher.manifest['sealed']


def test_append_segments_reuse_prefix_but_detect_csv_header_rewrite(tmp_path):
    runtime, work, record, logs = publisher_fixture(tmp_path)
    clock = [0]
    publisher = Publisher(runtime, work, record, {**DEFAULTS, 'chunk_bytes': 16, 'tick_bytes': 4096, 'metrics_seconds': 1}, clock=lambda: clock[0])
    path = logs / 'metrics.csv'
    path.write_bytes(b'a,b\n' + b'1,2\n' * 100)
    publisher.tick()
    initial = publisher.counters['source_bytes_read']
    with path.open('ab') as output: output.write(b'3,4\n')
    clock[0] = 2; publisher.tick()
    assert publisher.counters['source_bytes_read'] - initial <= 32
    entry = publisher.entries[path.relative_to(runtime).as_posix()]
    assert b''.join((work / chunk['source']).read_bytes() for chunk in entry['chunks']) == path.read_bytes()
    path.write_bytes(b'a,b,c\n' + b'1,2,3\n' * 100)
    clock[0] = 3; publisher.tick()
    entry = publisher.entries[path.relative_to(runtime).as_posix()]
    assert b''.join((work / chunk['source']).read_bytes() for chunk in entry['chunks']) == path.read_bytes()


def test_final_backup_supersedes_queued_precompletion_database(tmp_path):
    runtime, work, record, logs = publisher_fixture(tmp_path)
    db = runtime / 'results/optuna' / record['execution_id'] / 'study.sqlite'
    db.parent.mkdir(parents=True)
    with sqlite3.connect(db) as writer:
        writer.execute('CREATE TABLE trials(score)')
        writer.execute('INSERT INTO trials VALUES(1)')
    publisher = Publisher(runtime, work, record, {**DEFAULTS, 'chunk_bytes': 64, 'tick_bytes': 128})
    publisher.tick()
    assert publisher.pending
    with sqlite3.connect(db) as writer: writer.execute('INSERT INTO trials VALUES(2)')
    publisher.finish(seconds=10)
    exported = next(item for item in publisher.manifest['files'] if item['category'] == 'database')
    copy = tmp_path / 'final.sqlite'
    copy.write_bytes(b''.join((work / chunk['source']).read_bytes() for chunk in exported['chunks']))
    with sqlite3.connect(copy) as reader:
        assert reader.execute('SELECT score FROM trials ORDER BY score').fetchall() == [(1,), (2,)]


def test_transient_wal_disappearance_does_not_cancel_publication(tmp_path, monkeypatch):
    runtime, work, record, logs = publisher_fixture(tmp_path)
    (logs / 'metrics.csv').write_text('a,b\n1,2\n')
    transient = logs / 'study.sqlite-wal'
    transient.write_bytes(b'wal')
    original = Path.stat
    calls = [0]
    def disappearing(path, *args, **kwargs):
        if path == transient:
            calls[0] += 1
            if calls[0] >= 2: raise FileNotFoundError(str(path))
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'stat', disappearing)
    publisher = Publisher(runtime, work, record, DEFAULTS)
    publisher.finish(seconds=10)
    assert publisher.manifest['sealed']
    assert any(item['path'].endswith('metrics.csv') for item in publisher.manifest['files'])
    assert not any(item['path'].endswith('-wal') for item in publisher.manifest['files'])


def test_site_gres_template_cannot_bypass_normalized_count():
    config = cfg(resources={'partition': 'gpu', 'gpus': 1, 'gpu_type': 'h200'})
    config.site.gres_format = 'gpu:{gpu_type}:8'
    with pytest.raises(ConfigurationError, match='match'): execution.validate_allocation(config)


def test_oversized_monitor_metadata_leaves_no_partial_snapshot(tmp_path, monkeypatch):
    from src.app.monitoring.api import Result, write_snapshot
    from src.app.monitoring.budget import BudgetExceeded
    monkeypatch.setenv('THETA_OUTPUT_ROOT', str(tmp_path))
    monkeypatch.setenv('THETA_DIAGNOSTIC_BUDGET', json.dumps({'diagnostic_bytes': 4096, 'diagnostic_files': 10, 'snapshot_bytes': 1024}))
    context = SimpleNamespace(snapshot_index=0, identity='m', spec={'target': ''}, transitions=1,
        updates=1, probes=SimpleNamespace(identity='p'), access=SimpleNamespace(score_kind='q_values'), lineage='l')
    with pytest.raises(BudgetExceeded, match='metadata'):
        write_snapshot(tmp_path / 'monitors/m', context, Result(details={'huge': 'x' * 4096}),
                       model_hash='m', provider={}, elapsed=.1)
    assert list((tmp_path / 'monitors/m').iterdir()) == []

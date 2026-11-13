"""Cluster contracts tested with local files and fake scheduler/transfer commands only."""
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from src.app.sites import profiles, scheduler
from src.app.sites import execution, transfer
from src.app.sites.snapshot import freeze
from src.app.sites.storage import digest, read_json, safe_path, workspace_id, write_json
from src.app.workspace import WorkspacePaths


@pytest.fixture(autouse=True)
def no_ssh(monkeypatch):
    original = subprocess.run
    def guarded(args, *a, **kw):
        assert args[0] not in ('ssh', 'rsync', 'sbatch', 'squeue', 'sacct'), 'Test attempted a real cluster command'
        return original(args, *a, **kw)
    monkeypatch.setattr(subprocess, 'run', guarded)


def project(root):
    (root / 'src').mkdir(parents=True)
    (root / 'src/worker.py').write_text('print("original")\n')
    (root / 'in/config').mkdir(parents=True)
    (root / 'in/config/config.yaml').write_text('defaults: [{site: local}]\n')
    (root / 'run_pipeline.py').write_text('# runner\n')
    workspace_id(root)
    return root


def record(root, state='completed'):
    project(root)
    execution_id = 'a' * 32
    value = {'schema_version': 1, 'execution_id': execution_id, 'project_id': workspace_id(root),
             'site': 'cluster', 'group': 'demo', 'result_experiment_id': 'nested/test__' + execution_id,
             'state': state, 'scheduler_state': state, 'job_id': '123', 'artifacts_finalized': False}
    path = execution.receipt_path(root, execution_id)
    write_json(path, value)
    logs = root / 'results/logs/demo' / value['result_experiment_id'] / 'agent/version_0'
    logs.mkdir(parents=True)
    (logs / 'metrics.csv').write_text('step,reward\n1,2\n')
    (logs / 'events.jsonl').write_text('{"type":"metrics"}\n')
    ckpt = root / 'results/checkpoints/demo' / value['result_experiment_id'] / 'model.ckpt'
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b'weights')
    if state in execution.TERMINAL: execution.finalize(root, execution_id)
    return value, logs, ckpt


def fake_pull(monkeypatch, remote_root, local_root):
    transfer.configure(local_root, 'cluster', 'fake-cluster', '/unused')
    monkeypatch.setattr(transfer, 'remote', lambda root, site, cmd, *a, **kw: transfer.release(remote_root, a[0]) if cmd == '_release' else transfer.index(remote_root, a[1] if a else 'metrics'))
    def rsync(args, **kw):
        assert args[0] == 'rsync'
        listing = Path(next(a.split('=', 1)[1] for a in args if a.startswith('--files-from=')))
        stage = Path(args[-1])
        for relative in listing.read_bytes().decode().strip('\0').split('\0'):
            if not relative: continue
            target = stage / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(remote_root / relative, target)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(subprocess, 'run', rsync)


def test_explicit_scheduler_and_packaged_profiles(tmp_path):
    assert scheduler({'site': {'name': 'custom', 'scheduler': 'local'}}) == 'local'
    assert scheduler({'site': {'name': 'legacy_cluster'}}) == 'slurm'
    with pytest.raises(ValueError): scheduler({'site': {'scheduler': 'unsupported'}})
    paths = WorkspacePaths(tmp_path)
    assert 'local/default' in profiles(paths)


def test_frozen_source_and_portable_identity(tmp_path):
    root = project(tmp_path / 'project')
    (root / '.thetaide/site_connections.json').write_text('{"private": true}')
    source = freeze(root, tmp_path / 'snapshot')
    assert '.thetaide/site_connections.json' not in source['files']
    (root / 'src/worker.py').write_text('changed')
    assert (tmp_path / 'snapshot/src/worker.py').read_text() == 'print("original")\n'
    assert source['project_id'] == workspace_id(tmp_path / 'snapshot')


@pytest.mark.parametrize('relative', ['../x', '/x', 'a/../x', 'a\nb', 'a\\b', 'a//b'])
def test_path_traversal_rejected(tmp_path, relative):
    with pytest.raises(ValueError): safe_path(tmp_path, relative)


def test_destination_symlink_rejected(tmp_path):
    (tmp_path / 'results').symlink_to(tmp_path.parent)
    with pytest.raises(ValueError): safe_path(tmp_path, 'results/file')


def test_sync_metrics_then_results_is_idempotent(tmp_path, monkeypatch):
    remote_root, local_root = tmp_path / 'cluster', tmp_path / 'local'
    value, logs, ckpt = record(remote_root)
    local_root.mkdir()
    write_json(local_root / '.thetaide/workspace.json', {'id': value['project_id']})
    fake_pull(monkeypatch, remote_root, local_root)
    messages = []
    monkeypatch.setattr(transfer.transfer_output, 'emit', messages.append)
    first = transfer.sync(local_root, 'cluster', 'metrics')
    assert first['transferred'] > 0
    assert any(message.startswith('Saved ') and 'metrics.csv' in message for message in messages)
    messages.clear()
    assert not (local_root / ckpt.relative_to(remote_root)).exists()
    again = transfer.sync(local_root, 'cluster', 'metrics')
    assert again['transferred'] == 0
    assert not any(message.startswith('Saved ') for message in messages)
    full = transfer.sync(local_root, 'cluster', 'results')
    assert full['transferred'] == 1
    assert (local_root / ckpt.relative_to(remote_root)).read_bytes() == b'weights'
    assert ckpt.exists()
    assert transfer.sync(local_root, 'cluster', 'results')['transferred'] == 0


def test_changed_during_transfer_never_published(tmp_path, monkeypatch):
    remote_root, local_root = tmp_path / 'cluster', tmp_path / 'local'
    value, logs, _ = record(remote_root, 'running')
    local_root.mkdir()
    write_json(local_root / '.thetaide/workspace.json', {'id': value['project_id']})
    monkeypatch.setattr(transfer, 'refresh_status', lambda root, value: value)
    fake_pull(monkeypatch, remote_root, local_root)
    listing = transfer.index(remote_root, 'metrics')
    item = next(item for item in listing['files'] if item['path'].endswith('events.jsonl'))
    (remote_root / item['source']).write_text('changed during sync')
    monkeypatch.setattr(transfer, 'remote', lambda *a, **kw: listing)
    with pytest.raises(ValueError, match='changed during transfer'):
        transfer.sync(local_root, 'cluster')
    assert not (local_root / (logs / 'events.jsonl').relative_to(remote_root)).exists()
    assert read_json(local_root / '.thetaide/site_sync.json')['cluster']['error']


def test_remote_identity_and_local_collision_protection(tmp_path, monkeypatch):
    remote_root, local_root = tmp_path / 'cluster', tmp_path / 'local'
    value, logs, _ = record(remote_root)
    local_root.mkdir()
    workspace_id(local_root)
    fake_pull(monkeypatch, remote_root, local_root)
    with pytest.raises(ValueError, match='identity'): transfer.sync(local_root, 'cluster')
    write_json(local_root / '.thetaide/workspace.json', {'id': value['project_id']})
    collision = local_root / (logs / 'metrics.csv').relative_to(remote_root)
    collision.parent.mkdir(parents=True)
    collision.write_text('my local output')
    with pytest.raises(ValueError, match='collision'): transfer.sync(local_root, 'cluster')
    assert collision.read_text() == 'my local output'


def test_unknown_submission_is_not_retried(tmp_path, monkeypatch):
    root = project(tmp_path)
    calls = []
    def submit(args, **kw):
        calls.append(args)
        raise subprocess.TimeoutExpired(args, 60)
    monkeypatch.setattr(subprocess, 'run', submit)
    cfg = OmegaConf.create({'group': 'demo', 'experiment_id': 'test', 'site': {'name': 'cluster', 'python_executable': sys.executable}, 'resources': {}})
    with pytest.raises(RuntimeError): execution.submit(root, 'demo/test', [], cfg)
    assert len(calls) == 1
    receipt = read_json(next((root / 'results/slurm').glob('*/submission.json')))
    assert receipt['state'] == 'submission_unknown'
    assert receipt['result_experiment_id'] == 'test/' + receipt['execution_id']
    script = next((root / 'results/slurm').glob('*/job.sbatch')).read_text()
    assert 'unset THETA_RESOURCE_POOL' in script
    assert '_run' in script


def test_cleanup_requires_all_verified_artifacts(tmp_path, monkeypatch):
    remote_root, local_root = tmp_path / 'cluster', tmp_path / 'local'
    value, _, checkpoint = record(remote_root)
    local_root.mkdir()
    write_json(local_root / '.thetaide/workspace.json', {'id': value['project_id']})
    fake_pull(monkeypatch, remote_root, local_root)
    transfer.sync(local_root, 'cluster', 'metrics')
    with pytest.raises(FileNotFoundError): transfer.clean(local_root, 'cluster', value['execution_id'])
    transfer.sync(local_root, 'cluster', 'results')
    monkeypatch.setattr(transfer, 'remote', lambda root, site, cmd, eid, input: transfer.clean_remote(remote_root, eid, json.loads(input)))
    result = transfer.clean(local_root, 'cluster', value['execution_id'])
    assert result['removed'] > 0
    assert not checkpoint.exists()
    assert execution.receipt_path(remote_root, value['execution_id']).exists()
    assert transfer.index(remote_root, 'results')['executions'][0]['remote_cleaned_at']


def test_source_activation_removes_only_previously_managed_files(tmp_path):
    source = project(tmp_path / 'source')
    remote_root = tmp_path / 'remote'
    remote_root.mkdir()
    snapshot = tmp_path / 'upload1'
    freeze(source, snapshot)
    transfer.activate(remote_root, snapshot)
    unrelated = remote_root / 'src/unmanaged.txt'
    unrelated.write_text('keep')
    (source / 'src/worker.py').unlink()
    freeze(source, tmp_path / 'upload2')
    transfer.activate(remote_root, tmp_path / 'upload2')
    assert not (remote_root / 'src/worker.py').exists()
    assert unrelated.read_text() == 'keep'


def test_active_metrics_snapshot_has_complete_records(tmp_path, monkeypatch):
    value, logs, _ = record(tmp_path, 'running')
    monkeypatch.setattr(transfer, 'refresh_status', lambda root, value: value)
    (logs / 'events.jsonl').write_bytes(b'{"step":1}\n{"step":')
    listing = transfer.index(tmp_path, 'metrics')
    item = next(item for item in listing['files'] if item['path'].endswith('events.jsonl'))
    (logs / 'events.jsonl').write_bytes(b'{"step":2}\n')
    assert (tmp_path / item['source']).read_bytes() == b'{"step":1}\n'
    transfer.release(tmp_path, listing['generation'])
    assert not (tmp_path / item['source']).exists()


def test_site_component_scaffold_is_installable(tmp_path):
    from src.app.services.plugin_scaffolder import PluginScaffolder, ScaffoldOptions
    from src.app.components.archive import validate_native_package
    result = PluginScaffolder(tmp_path).scaffold(ScaffoldOptions(name='cluster', kind='site'))
    assert result.success, result.error
    manifest = read_json(result.target_dir / '.theta_component.json')
    validate_native_package(manifest, result.target_dir)
    assert 'cluster/default' in profiles(WorkspacePaths(tmp_path))


def test_allocation_worker_runs_frozen_pipeline_and_records_outputs(tmp_path, monkeypatch):
    monkeypatch.setenv('SLURM_JOB_ID', '321')
    root = project(tmp_path)
    script = root / 'run_pipeline.py'
    script.write_text('''import json, os, pathlib, sys
assert os.environ['THETA_IN_ALLOCATION'] == '1'
assert 'THETA_RESOURCE_POOL' not in os.environ
value = next(item.split('=', 1)[1] for item in sys.argv if item.startswith('++experiment_id='))
name = json.loads(value)
path = pathlib.Path('results/logs/demo') / name / 'agent/version_0'
path.mkdir(parents=True)
(path / 'metrics.csv').write_text('step,reward\\n1,5\\n')
print('frozen source executed')
''')
    def sbatch(args, **kw):
        assert args[0] == 'sbatch'
        return SimpleNamespace(returncode=0, stdout='321;cluster\n', stderr='')
    monkeypatch.setattr(subprocess, 'run', sbatch)
    cfg = OmegaConf.create({'group': 'demo', 'experiment_id': 'test', 'site': {'name': 'cluster', 'python_executable': sys.executable}, 'resources': {}})
    result = execution.submit(root, 'demo/test', [], cfg)
    script.write_text('raise RuntimeError("mutable project must not run")')
    monkeypatch.setenv('THETA_RESOURCE_POOL', '/unrelated/workstation-pool')
    code = execution.run(root, result['execution_id'])
    assert code == 0
    receipt = read_json(execution.receipt_path(root, result['execution_id']))
    assert receipt['state'] == 'completed'
    assert not receipt['artifacts_finalized']  # wait for scheduler termination before sealing stdout/stderr
    assert list((root / 'results/logs').rglob('metrics.csv'))
    assert 'frozen source executed' in (execution.receipt_path(root, result['execution_id']).parent / 'pipeline.log').read_text()


def test_scheduler_finishing_is_distinct_from_submission_exit(tmp_path, monkeypatch):
    value, _, _ = record(tmp_path, 'running')
    replies = iter(['', '123|TIMEOUT|0:15\n'])
    monkeypatch.setattr(subprocess, 'run', lambda *a, **kw: SimpleNamespace(stdout=next(replies)))
    updated = execution.refresh_status(tmp_path, value)
    assert updated['state'] == 'timeout'
    assert updated['exit_code'] == '0:15'


def test_legacy_live_sqlite_sync_is_rejected(tmp_path, monkeypatch):
    import sqlite3
    value, logs, _ = record(tmp_path, 'running')
    with sqlite3.connect(logs / 'study.sqlite') as db:
        db.execute('create table trials (score integer)')
        db.execute('insert into trials values (7)')
    monkeypatch.setattr(transfer, 'refresh_status', lambda root, value: value)
    monkeypatch.setattr(sqlite3, 'connect', lambda *a, **k: pytest.fail('login helper opened live SQLite'))
    with pytest.raises(ValueError, match='owning compute node'):
        transfer.index(tmp_path, 'results')
    assert not list((tmp_path / 'results/slurm/.sync').iterdir())


def test_default_batch_interpreter_preserves_virtualenv_path(tmp_path, monkeypatch):
    real_run = subprocess.run
    root = project(tmp_path)
    executable = root / 'venv/bin/python3'
    executable.parent.mkdir(parents=True)
    executable.symlink_to(sys.executable)
    monkeypatch.setattr(subprocess, 'run', lambda *a, **kw: SimpleNamespace(returncode=0, stdout='123\n', stderr=''))
    cfg = OmegaConf.create({'group': 'demo', 'experiment_id': 'test', 'site': {'name': 'cluster'}, 'resources': {}})
    value = execution.submit(root, 'demo/test', [], cfg)
    script = (execution.receipt_path(root, value['execution_id']).parent / 'job.sbatch').read_text()
    assert str(executable) in script
    assert real_run(['/bin/bash', '-n', str(execution.receipt_path(root, value['execution_id']).parent / 'job.sbatch')]).returncode == 0


def test_hydra_inherits_and_overrides_site_group(tmp_path, monkeypatch):
    monkeypatch.setenv("THETA_WORKSPACE", str(tmp_path))
    from src.app.pipeline.compose import compose_experiment
    config = tmp_path / 'in/config'
    (config / 'experiment/g').mkdir(parents=True)
    (config / 'site').mkdir()
    (config / 'config.yaml').write_text('defaults:\n  - site: local\n  - _self_\ngroup: ungrouped\nexperiment_id: default_exp\nexperiment_name: default_exp\n')
    (config / 'site/local.yaml').write_text('scheduler: local\n')
    (config / 'site/cluster.yaml').write_text('scheduler: slurm\n')
    (config / 'experiment/g/_base.yaml').write_text('# @package _global_\ndefaults:\n  - override /site: cluster\n  - _self_\n')
    (config / 'experiment/g/test.yaml').write_text('# @package _global_\ndefaults:\n  - /experiment/g/_base\n  - _self_\n')
    composed = compose_experiment('g/test', workspace=WorkspacePaths(tmp_path))
    assert composed.cfg.site.scheduler == 'slurm'
    assert compose_experiment('g/test', ['site=local'], workspace=WorkspacePaths(tmp_path)).cfg.site.scheduler == 'local'


def test_repeat_results_sync_discovers_new_post_run_plot(tmp_path, monkeypatch):
    remote_root, local_root = tmp_path / 'cluster', tmp_path / 'local'
    value, logs, _ = record(remote_root)
    local_root.mkdir()
    write_json(local_root / '.thetaide/workspace.json', {'id': value['project_id']})
    fake_pull(monkeypatch, remote_root, local_root)
    transfer.sync(local_root, 'cluster', 'results')
    plot = remote_root / 'results/plots' / value['group'] / value['result_experiment_id'] / 'late.png'
    plot.parent.mkdir(parents=True)
    plot.write_bytes(b'new plot after completion')
    result = transfer.sync(local_root, 'cluster', 'results')
    assert result['transferred'] == 1
    assert (local_root / plot.relative_to(remote_root)).read_bytes() == plot.read_bytes()

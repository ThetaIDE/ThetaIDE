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
from src.app.sites import bundle
from bundle_fixtures import fake_selection
from src.app.sites.storage import digest, read_json, safe_path, workspace_id, write_json
from src.app.workspace import WorkspacePaths


@pytest.fixture(autouse=True)
def no_ssh(monkeypatch):
    original = subprocess.run
    def guarded(args, *a, **kw):
        assert args[0] not in ('ssh', 'rsync'), 'Test attempted a real remote command'
        return original(args, *a, **kw)
    monkeypatch.setattr(subprocess, 'run', guarded)


def project(root):
    (root / 'src/app').mkdir(parents=True)
    (root / 'src/app/worker.py').write_text('print("original")\n')
    (root / 'src/usr/config').mkdir(parents=True)
    (root / 'src/usr/config/config.yaml').write_text('defaults: [{site: local}]\n')
    (root / 'run_pipeline.py').write_text('# runner\n')
    workspace_id(root)
    from component_fixtures import install_fixture
    # Cluster contracts run through an installed scheduler; the test one runs no commands.
    install_fixture(root, 'batch-scheduler', root.parent / (root.name + '-archives'))
    return root


def record(root, state='completed'):
    project(root)
    execution_id = 'a' * 32
    value = {'schema_version': 1, 'scheduler_component': 'batch', 'tuner_execution': None,
             'execution_id': execution_id, 'project_id': workspace_id(root),
             'site': 'cluster', 'group': 'demo', 'result_experiment_id': 'nested/test__' + execution_id,
             'state': state, 'scheduler_state': state, 'job_id': '123', 'artifacts_finalized': False}
    path = execution.receipt_path(root, execution_id)
    write_json(path, value)
    logs = root / 'src/usr/results/logs/demo' / value['result_experiment_id'] / 'agent/version_0'
    logs.mkdir(parents=True)
    (logs / 'metrics.csv').write_text('step,reward\n1,2\n')
    (logs / 'events.jsonl').write_text('{"type":"metrics"}\n')
    ckpt = root / 'src/usr/results/checkpoints/demo' / value['result_experiment_id'] / 'model.ckpt'
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
    assert scheduler({'site': {'name': 'local'}}) == 'local'
    with pytest.raises(ValueError, match="does not declare site.scheduler"):
        scheduler({'site': {'name': 'cluster'}})
    # Any nonempty selector is a capability name; installation is checked when it is resolved.
    assert scheduler({'site': {'scheduler': 'lattice-batch'}}) == 'lattice-batch'
    with pytest.raises(ValueError): scheduler({'site': {'scheduler': ' '}})
    paths = WorkspacePaths(tmp_path)
    assert 'local/default' in profiles(paths)


CLUSTER = OmegaConf.create({'group': 'demo', 'experiment_id': 'test', 'site': {'name': 'cluster', 'scheduler': 'batch',
                            'python_executable': sys.executable}, 'resources': {}})


def test_bundle_is_frozen_and_keeps_portable_identity(tmp_path):
    root = project(tmp_path / 'project')
    (root / '.thetaide/site_connections.json').write_text('{"private": true}')
    (root / 'src/usr/results/logs').mkdir(parents=True)
    (root / 'src/usr/results/logs/old.csv').write_text('step\n')
    manifest = bundle.build(fake_selection(root, 'demo/test', CLUSTER), tmp_path / 'snapshot')
    assert '.thetaide/site_connections.json' not in manifest['files']
    assert not any(name.startswith('src/usr/results') for name in manifest['files'])
    (root / 'src/app/worker.py').write_text('changed')
    assert (tmp_path / 'snapshot/src/app/worker.py').read_text() == 'print("original")\n'
    assert manifest['project_id'] == workspace_id(tmp_path / 'snapshot')
    assert bundle.verify(tmp_path / 'snapshot')['bundle_sha256'] == manifest['bundle_sha256']


@pytest.mark.parametrize('relative', ['../x', '/x', 'a/../x', 'a\nb', 'a\\b', 'a//b'])
def test_path_traversal_rejected(tmp_path, relative):
    with pytest.raises(ValueError): safe_path(tmp_path, relative)


def test_destination_symlink_rejected(tmp_path):
    (tmp_path / 'src/usr').mkdir(parents=True)
    (tmp_path / 'src/usr/results').symlink_to(tmp_path.parent)
    with pytest.raises(ValueError): safe_path(tmp_path, 'src/usr/results/file')


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
    bundle.build(fake_selection(source, 'demo/test', CLUSTER), snapshot)
    transfer.activate(remote_root, snapshot)
    unrelated = remote_root / 'src/unmanaged.txt'
    unrelated.write_text('keep')
    (source / 'src/app/worker.py').unlink()
    bundle.build(fake_selection(source, 'demo/test', CLUSTER), tmp_path / 'upload2')
    transfer.activate(remote_root, tmp_path / 'upload2')
    assert not (remote_root / 'src/app/worker.py').exists()
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


def test_status_and_cancellation_dispatch_to_the_receipts_scheduler(tmp_path):
    value, _, _ = record(tmp_path, 'running')
    (execution.receipt_path(tmp_path, value['execution_id']).parent / 'batch-state').write_text('timeout')
    assert execution.refresh_status(tmp_path, value)['state'] == 'timeout'
    value, _, _ = record(tmp_path / 'other', 'running')
    assert execution.cancel(tmp_path / 'other', value)['scheduler_state'] == 'cancelled'
    assert read_json(execution.receipt_path(tmp_path / 'other', value['execution_id']))['state'] == 'cancelled'


def test_hydra_inherits_and_overrides_site_group(tmp_path, monkeypatch):
    monkeypatch.setenv("THETA_WORKSPACE", str(tmp_path))
    from src.app.pipeline.compose import compose_experiment
    config = tmp_path / 'src/usr/config'
    (config / 'experiment/g').mkdir(parents=True)
    (config / 'site').mkdir()
    (config / 'config.yaml').write_text('defaults:\n  - site: local\n  - _self_\ngroup: ungrouped\nexperiment_id: default_exp\nexperiment_name: default_exp\n')
    (config / 'site/local.yaml').write_text('scheduler: local\n')
    (config / 'site/cluster.yaml').write_text('scheduler: batch\n')
    (config / 'experiment/g/_base.yaml').write_text('# @package _global_\ndefaults:\n  - override /site: cluster\n  - _self_\n')
    (config / 'experiment/g/test.yaml').write_text('# @package _global_\ndefaults:\n  - /experiment/g/_base\n  - _self_\n')
    composed = compose_experiment('g/test', workspace=WorkspacePaths(tmp_path))
    assert composed.cfg.site.scheduler == 'batch'
    assert compose_experiment('g/test', ['site=local'], workspace=WorkspacePaths(tmp_path)).cfg.site.scheduler == 'local'


def test_repeat_results_sync_discovers_new_post_run_plot(tmp_path, monkeypatch):
    remote_root, local_root = tmp_path / 'cluster', tmp_path / 'local'
    value, logs, _ = record(remote_root)
    local_root.mkdir()
    write_json(local_root / '.thetaide/workspace.json', {'id': value['project_id']})
    fake_pull(monkeypatch, remote_root, local_root)
    transfer.sync(local_root, 'cluster', 'results')
    plot = remote_root / 'src/usr/results/plots' / value['group'] / value['result_experiment_id'] / 'late.png'
    plot.parent.mkdir(parents=True)
    plot.write_bytes(b'new plot after completion')
    result = transfer.sync(local_root, 'cluster', 'results')
    assert result['transferred'] == 1
    assert (local_root / plot.relative_to(remote_root)).read_bytes() == plot.read_bytes()

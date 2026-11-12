"""Managed downloads batch work without weakening verification or retry safety."""
import hashlib
import shutil
import sys
import threading
from pathlib import Path

import pytest

from src.app.sites import transfer, transfer_output
from src.app.sites.slurm import ssh_arguments
from src.app.sites.storage import workspace_id


def fixture(tmp_path, *, roots=1, generation=None):
    local = tmp_path / 'local'
    local.mkdir()
    files = []
    for index in range(6):
        origin = tmp_path / f'remote-{index % roots}'
        data = f'payload-{index}'.encode()
        sha = hashlib.sha256(data).hexdigest()
        source = f'results/slurm/job/objects/{sha}'
        path = origin / source
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        files.append({'path': f'results/output-{index}.bin', 'size': len(data), 'sha256': sha,
                      'remote_root': str(origin), 'chunks': [{'source': source, 'size': len(data), 'sha256': sha}]})
    return local, {'host': 'fake', 'remote_root': '/control'}, {
        'files': files, 'executions': [], 'generation': generation}


def requested(args):
    names = Path(next(arg.split('=', 1)[1] for arg in args if arg.startswith('--files-from=')))
    return names.read_bytes().decode().strip('\0').split('\0')


def copy_objects(args, names=None):
    origin = Path(args[-2].split(':', 1)[1])
    for name in requested(args) if names is None else names:
        target = Path(args[-1]) / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(origin / name, target)


@pytest.mark.parametrize('roots', [1, 2])
def test_one_batch_per_root_and_no_transfer_when_unchanged(tmp_path, monkeypatch, roots):
    local, config, listing = fixture(tmp_path, roots=roots)
    calls = []

    def pull(args, **kwargs):
        calls.append(requested(args))
        copy_objects(args)

    monkeypatch.setattr(transfer, 'run_transfer', pull)
    result = transfer.sync_objects(local, 'cluster', config, listing)
    assert len(calls) == roots
    assert sum(map(len, calls)) == 6
    assert result['transferred'] == 6
    for index in range(6):
        assert (local / f'results/output-{index}.bin').read_bytes() == f'payload-{index}'.encode()
    result = transfer.sync_objects(local, 'cluster', config, listing)
    assert result['transferred'] == result['bytes'] == 0
    assert len(calls) == roots


def test_failed_batch_retains_verified_objects_and_retry_requests_only_missing(tmp_path, monkeypatch):
    local, config, listing = fixture(tmp_path)
    calls = []

    def pull(args, **kwargs):
        names = requested(args)
        calls.append(names)
        if len(calls) == 1:
            copy_objects(args, names[:2])
            # Model a partial rsync file as well as two completed objects.
            partial = Path(args[-1]) / '.theta-partial/unfinished'
            partial.parent.mkdir()
            partial.write_bytes(b'partial')
            raise RuntimeError('connection lost')
        copy_objects(args)

    monkeypatch.setattr(transfer, 'run_transfer', pull)
    with pytest.raises(RuntimeError, match='connection lost'):
        transfer.sync_objects(local, 'cluster', config, listing)
    assert not (local / 'results/output-0.bin').exists()
    assert list((local / '.thetaide').rglob('unfinished'))
    result = transfer.sync_objects(local, 'cluster', config, listing)
    assert len(calls[1]) == 4 and not set(calls[0][:2]) & set(calls[1])
    assert result['transferred'] == 6
    assert not list((local / '.thetaide').rglob('unfinished'))


def test_corrupt_batch_is_not_materialized(tmp_path, monkeypatch):
    local, config, listing = fixture(tmp_path)

    def pull(args, **kwargs):
        copy_objects(args)
        (Path(args[-1]) / requested(args)[0]).write_bytes(b'corrupt')

    monkeypatch.setattr(transfer, 'run_transfer', pull)
    with pytest.raises(ValueError, match='checksum mismatch'):
        transfer.sync_objects(local, 'cluster', config, listing)
    assert not (local / 'results/output-0.bin').exists()


def test_lease_renewal_is_time_based_within_a_single_batch(tmp_path, monkeypatch):
    local, config, listing = fixture(tmp_path, generation='lease')
    clock = [0]
    renewals = []
    monkeypatch.setattr(transfer.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(transfer, 'remote', lambda *args: renewals.append((clock[0], args[2:])))

    def pull(args, *, on_poll, **kwargs):
        for now in (1, 299, 300, 301, 599, 600):
            clock[0] = now
            on_poll()
        copy_objects(args)

    monkeypatch.setattr(transfer, 'run_transfer', pull)
    transfer.sync_objects(local, 'cluster', config, listing)
    assert renewals == [(300, ('_renew', 'lease')), (600, ('_renew', 'lease'))]


def test_failed_sync_releases_lease_in_same_ssh_session(tmp_path, monkeypatch):
    config = {'host': 'fake', 'remote_root': '/control'}
    listing = {'project_id': workspace_id(tmp_path), 'generation': 'lease', 'files': [],
               'executions': [{'execution_id': 'job', 'execution_policy': {'work_root': '/work'}}]}
    commands = []

    def remote(root, site, command, *args):
        commands.append((command, ssh_arguments(root, config)))
        return listing if command == '_index' else {}

    def fail(*args, **kwargs):
        raise RuntimeError('download failed')

    monkeypatch.setattr(transfer, 'connection', lambda *args: config)
    monkeypatch.setattr(transfer, 'remote', remote)
    monkeypatch.setattr(transfer, 'sync_objects', fail)
    with pytest.raises(RuntimeError, match='download failed'):
        transfer.sync(tmp_path, 'cluster', 'results')
    assert [command for command, _ in commands] == ['_index', '_release']
    assert commands[0][1] == commands[1][1]
    assert not any('ControlPath' in arg for arg in ssh_arguments(tmp_path, config))


@pytest.mark.parametrize('streaming', [False, True])
def test_transfer_polls_lease_and_cancels_child(streaming):
    cancel = threading.Event()
    calls = []

    def poll():
        calls.append(True)
        cancel.set()

    with transfer_output.output_to((lambda line: None) if streaming else None):
        with pytest.raises(RuntimeError, match='cancelled'):
            transfer.run_transfer([sys.executable, '-c', 'import time; time.sleep(30)'],
                                  cancel_event=cancel, on_poll=poll)
    assert calls


@pytest.mark.parametrize('streaming', [False, True])
def test_renewal_failure_stops_running_transfer(streaming):
    def fail():
        raise RuntimeError('lease renewal failed')

    with transfer_output.output_to((lambda line: None) if streaming else None):
        with pytest.raises(RuntimeError, match='lease renewal failed'):
            transfer.run_transfer([sys.executable, '-c', 'import time; time.sleep(30)'], on_poll=fail)

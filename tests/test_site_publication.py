"""Allocation-side publication, staging and transfer primitives; no scheduler or cluster calls."""
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

from src.app.sites import transfer
from src.app.sites.policy import DEFAULTS, stage_inputs
from src.app.sites.publication import Publisher, sqlite_export
from src.app.sites.bundle import extract
from src.app.sites.storage import read_json, workspace_id, write_json


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
    policy = {**DEFAULTS, 'inputs': [{'source': 'data', 'target': 'src/usr/datasets/data'}]}
    stage_inputs(root, destination, policy)
    assert (destination / 'src/usr/datasets/data').stat().st_mtime > time.time() - 10
    assert read_json(destination / '.theta-inputs.json')['files'][0]['original_mtime_ns'] == 1000000000
    policy['max_input_bytes'] = 1
    with pytest.raises(ValueError, match='max_input_bytes'): stage_inputs(root, tmp_path / 'other', policy)


def publisher_fixture(tmp_path):
    runtime, work = tmp_path / 'runtime', tmp_path / 'work'
    runtime.mkdir(); work.mkdir()
    from component_fixtures import install_fixture, search_plan
    # The receipt names the test tuner's plan; its study snapshot runs through the installed provider.
    install_fixture(runtime, 'search-tuner', tmp_path / 'archives')
    execution_id = 'a' * 32
    record = {'execution_id': execution_id, 'project_id': workspace_id(runtime), 'group': 'g', 'result_experiment_id': 'e__a',
              'tuner_execution': search_plan(execution_id)}
    (runtime / 'src/usr/results/sites' / record['execution_id']).mkdir(parents=True)
    logs = runtime / 'src/usr/results/logs/g/e__a/m/version_0'
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
    db = runtime / 'src/usr/results/search' / record['execution_id'] / 'study.sqlite'
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


def test_live_wal_backup_is_standalone(tmp_path):
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


def test_collection_serializes_with_submission_and_preserves_foreign_leases(tmp_path):
    from src.app.sites.maintenance import collect
    from src.app.sites.storage import transfer_lock
    workspace_id(tmp_path)
    foreign = tmp_path / 'src/usr/results/sites/.sync' / ('c' * 32)
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
    db = runtime / 'src/usr/results/search' / record['execution_id'] / 'study.sqlite'
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

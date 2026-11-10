import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from PyQt6.QtWidgets import QApplication, QWidget

from frontend.site_sync import SiteSyncControl
from src.app.sites import transfer, transfer_output
from src.app.workspace import WorkspacePaths

app = QApplication.instance() or QApplication([])


def test_transfer_streams_stdout_and_stderr_before_completion(tmp_path):
    marker = tmp_path / 'received'
    lines = []
    def receive(line):
        lines.append(line)
        if line == 'ready': marker.touch()
    script = '''import pathlib,sys,time
print('ready', flush=True)
marker = pathlib.Path(sys.argv[1])
deadline = time.monotonic() + 5
while not marker.exists() and time.monotonic() < deadline: time.sleep(.01)
assert marker.exists(), 'output was buffered until process exit'
print('stderr detail', file=sys.stderr, flush=True)
print('finished', flush=True)
'''
    with transfer_output.output_to(receive):
        transfer.run_transfer([sys.executable, '-u', '-c', script, str(marker)])
    assert lines[0].startswith('$ ')
    assert 'ready' in lines and 'stderr detail' in lines and 'finished' in lines
    assert lines[-1] == 'Process exited with code 0'


def test_failed_command_logs_actual_error_and_exit_code():
    lines = []
    with transfer_output.output_to(lines.append):
        with pytest.raises(RuntimeError, match='permission denied'):
            transfer.run_transfer([sys.executable, '-c',
                'import sys; print("permission denied",file=sys.stderr); sys.exit(7)'])
    assert 'permission denied' in lines
    assert lines[-1] == 'Process exited with code 7'


def test_logged_transfer_can_still_be_cancelled():
    cancelled = threading.Event()
    def receive(line):
        if line == 'ready': cancelled.set()
    with transfer_output.output_to(receive):
        with pytest.raises(RuntimeError, match='cancelled'):
            transfer.run_transfer([sys.executable, '-u', '-c',
                'import time; print("ready",flush=True); time.sleep(30)'], cancel_event=cancelled)


def test_output_is_scoped_to_each_worker_and_restored():
    barrier = threading.Barrier(2)
    def worker(name):
        lines = []
        with transfer_output.output_to(lines.append):
            barrier.wait(timeout=5)
            transfer_output.emit(name)
        transfer_output.emit('outside scope')
        return lines
    with ThreadPoolExecutor(max_workers=2) as pool:
        a, b = pool.submit(worker, 'one'), pool.submit(worker, 'two')
        assert a.result() == ['one']
        assert b.result() == ['two']


def test_ssh_logs_command_and_both_streams_without_changing_response(monkeypatch, tmp_path):
    lines = []
    monkeypatch.setattr(transfer.subprocess, 'run', lambda *a, **k:
        SimpleNamespace(returncode=0, stdout='{"ok": true}\n', stderr='remote warning\n'))
    config = {'host': 'cluster', 'remote_root': '/remote/project', '_root': str(tmp_path)}
    with transfer_output.output_to(lines.append):
        assert transfer.ssh(config, ['python', '--version']) == '{"ok": true}\n'
    assert lines[0].startswith('$ ssh ') and 'python --version' in lines[0]
    assert '{"ok": true}' in lines and 'remote warning' in lines


def test_sync_button_delivers_worker_output_to_console_on_gui_thread(monkeypatch, tmp_path):
    monkeypatch.setattr(WorkspacePaths, 'for_process', classmethod(lambda cls: cls(tmp_path)))
    received = []
    gui_thread = threading.get_ident()
    window = QWidget()
    window.log = lambda message: received.append((threading.get_ident(), message))
    control = SiteSyncControl('results', window)
    completed = []
    control.synced.connect(completed.append)
    def sync(*args, **kwargs):
        assert threading.get_ident() != gui_thread
        transfer_output.command(['rsync', 'remote/path', 'local/path'])
        transfer_output.emit('transferred file')
        return {'last_sync': '2026-11-10T12:00:00Z', 'transferred': 1, 'executions': [{'private_config': 'do not print this'}]}
    monkeypatch.setattr(transfer, 'sync', sync)
    control.start('cluster', 'sync')
    deadline = time.monotonic() + 5
    while not completed and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(.01)
    assert completed == ['cluster']
    assert all(thread == gui_thread for thread, _ in received)
    messages = [message for _, message in received]
    assert any('$ rsync remote/path local/path' in message for message in messages)
    assert any('transferred file' in message for message in messages)
    assert any('sync complete: 1 files saved' in message for message in messages)
    assert not any('private_config' in message for message in messages)
    window.close()


def test_internal_ssh_response_is_parsed_without_dumping_manifest(monkeypatch, tmp_path):
    import json
    payload = {'executions': [{'config': 'large captured config'}], 'files': []}
    monkeypatch.setattr(transfer, 'connection', lambda *_: {
        'host': 'cluster', 'remote_root': '/remote/project', '_root': str(tmp_path), 'python': 'python'})
    monkeypatch.setattr(transfer.subprocess, 'run', lambda *a, **k:
        SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr='warning from remote'))
    lines = []
    with transfer_output.output_to(lines.append):
        assert transfer.remote(tmp_path, 'cluster', '_index', '--mode', 'results') == payload
    assert lines[0].startswith('$ ssh ')
    assert 'warning from remote' in lines
    assert not any('captured config' in line or 'executions' in line for line in lines)

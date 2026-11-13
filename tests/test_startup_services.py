"""Startup workers preserve data and never wait on the GUI thread for readiness."""
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from PyQt6.QtCore import QThreadPool
from PyQt6.QtWidgets import QApplication

from frontend.job_manager import JobManager
from frontend.local_backend import LocalBackend
from frontend.model import Config, Store, latest, load_run_metrics, new_run, result_row


@pytest.fixture
def app():
    instance = QApplication.instance() or QApplication([])
    return instance


def pump(app, condition, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if condition():
            return
        time.sleep(.01)
    assert condition()


def test_store_summaries_skip_history_and_preserve_it_on_note_save(tmp_path, monkeypatch):
    store = Store(tmp_path)
    run = new_run(Config())
    run['status'] = 'completed'
    run['metrics'] = [{'step': i, 'reward': i % 17, 'loss': 1 / (i + 1)} for i in range(1000)]
    store.save(run)
    original = Path.read_text
    reads = []
    def read(path, *args, **kwargs):
        reads.append(path.name)
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'read_text', read)
    summaries, errors = store.load_summaries()
    assert not errors and len(summaries) == 1
    summary = summaries[0]
    assert 'record.json' not in reads
    assert summary['metrics'] == [] and latest(summary, 'reward') == 999 % 17
    assert result_row(summary) == result_row(run)
    summary['notes'] = 'keep all metric points'
    store.save(summary)
    persisted = json.loads((tmp_path / run['id'] / 'record.json').read_text())
    assert persisted['metrics'] == run['metrics'] and persisted['notes'] == summary['notes']
    assert not any(key.startswith('_metric') for key in persisted)


def test_legacy_migration_corruption_and_external_replacement(tmp_path):
    store = Store(tmp_path)
    run = new_run(Config())
    run['metrics'] = [{'step': 42, 'reward': 7}]
    folder = tmp_path / run['id']
    folder.mkdir()
    record = folder / 'record.json'
    record.write_text(json.dumps(run))  # An existing installation with no summary.
    summaries, errors = store.load_summaries()
    assert not errors and (folder / 'summary.json').exists()
    assert load_run_metrics(summaries[0]) == run['metrics']
    (folder / 'summary.json').write_text('[]')
    assert store.load_summaries()[0][0]['_metric_summary']['step'] == 42
    replacement = folder / 'replacement.json'
    run['metrics'].append({'step': 123, 'reward': 20})
    replacement.write_text(json.dumps(run))
    replacement.replace(record)
    summary = store.load_summaries()[0][0]
    assert summary['_metric_summary']['step'] == 123
    assert latest(summary, 'reward') == 20
    assert load_run_metrics(summary) == run['metrics']


def test_missing_history_cannot_be_overwritten_by_empty_summary(tmp_path):
    store = Store(tmp_path)
    run = new_run(Config())
    run['metrics'] = [{'step': 1, 'reward': 12}]
    store.save(run)
    summary = store.load_summaries()[0][0]
    (tmp_path / run['id'] / 'record.json').unlink()
    with pytest.raises(OSError):
        store.save(summary)
    assert not (tmp_path / run['id'] / 'record.json').exists()


def test_obsolete_queued_compositions_never_execute_and_late_results_are_dropped(app, monkeypatch):
    backend = LocalBackend()
    blocked, release = threading.Event(), threading.Event()
    backend._submit(lambda: (blocked.set(), release.wait(10)), callback=lambda *_: None)
    assert blocked.wait(2)
    executions, replies = [], []
    def compose(experiment, overrides, draft):
        executions.append((experiment, draft))
        return {'valid': True}
    monkeypatch.setattr('frontend.local_backend.config_service.compose_report', compose)
    backend.compose('g/old', [], lambda *_: replies.append('old'))
    backend.compose('g/new', [], lambda *_: replies.append('new'), draft_yaml='seed: 2')
    release.set()
    pump(app, lambda: replies == ['new'])
    assert executions == [('g/new', 'seed: 2')]
    assert not backend._relays
    # Cancel as soon as the user edits, before submitting the next debounce.
    blocked.clear(); release.clear()
    backend._submit(lambda: (blocked.set(), release.wait(10)), callback=lambda *_: None)
    assert blocked.wait(2)
    backend.compose('g/cancelled', [], lambda *_: replies.append('cancelled'))
    backend.cancel_compose()
    release.set()
    assert backend.wait()
    app.processEvents()
    assert executions == [('g/new', 'seed: 2')]


def snapshot(jobs=None):
    return {'jobs': jobs or [], 'queue': {'hierarchy_version': 2, 'running': False,
            'active': [], 'queued': [], 'finished': []}, 'max_parallel': None, 'updated': time.time()}


def test_runner_readiness_timeout_recovery_and_shutdown_never_sleep(app, tmp_path, monkeypatch):
    monkeypatch.setattr('frontend.job_manager.subprocess.Popen', Mock())
    with patch('frontend.job_manager.time.sleep', side_effect=AssertionError('GUI slept')):
        manager = JobManager(store_path=tmp_path / 'jobs.db', project_root=tmp_path)
        manager.timer.stop()
        assert not manager.ready and not manager.jobs
        assert manager.queue_state()['running'] is False
        with pytest.raises(Exception, match='Connecting'):
            manager.start_queue()
        changes = []
        manager.readiness_changed.connect(lambda ready, error: changes.append((ready, error)))
        manager._readiness_deadline = 0
        manager._poll()
        manager.timer.stop()
        assert not manager.ready and 'did not respond' in manager.readiness_error
        job = {'job_id': 'live', 'status': 'running'}
        reply = manager._readiness_request.with_suffix('.reply')
        reply.write_text(json.dumps({'result': snapshot([job])}))
        manager._poll()
        manager.timer.stop()
        assert manager.ready and manager.job('live')['status'] == 'running'
        # A stale/missing snapshot loses readiness without changing live status.
        manager._poll()
        manager.timer.stop()
        assert not manager.ready and manager.job('live')['status'] == 'running'
        manager._readiness_request.with_suffix('.reply').write_text(json.dumps({'result': snapshot([job])}))
        manager._poll()
        manager.timer.stop()
        assert manager.ready
        heartbeat, request = manager._heartbeat, manager._readiness_request
        manager.shutdown()
        assert not heartbeat.exists() and not request.exists()
        assert [ready for ready, _ in changes] == [False, True, False, True]


def test_failed_runner_spawn_does_not_prevent_opening_desktop(app, tmp_path, monkeypatch):
    monkeypatch.setattr('frontend.job_manager.subprocess.Popen', Mock(side_effect=OSError('unavailable')))
    manager = JobManager(store_path=tmp_path / 'jobs.db')
    assert not manager.ready and 'unavailable' in manager.readiness_error
    manager.shutdown()


def test_real_runner_handshake_is_asynchronous(app, tmp_path):
    manager = JobManager(store_path=tmp_path / 'jobs.db', project_root=tmp_path)
    try:
        assert not manager.ready
        pump(app, lambda: manager.ready)
        assert manager.queue_state()['running'] is False
    finally:
        manager.shutdown()


def test_not_ready_runner_does_not_interrupt_saved_jobs():
    from frontend.app import Window
    run = {'status': 'running', 'simulated': False, 'backend': {'job_id': 'a'}}
    host = SimpleNamespace(jobs=SimpleNamespace(ready=False), runs=[run])
    Window.reconnect_live_run(host)
    assert run['status'] == 'running'


def test_replaying_late_loaded_records_resets_only_their_feeds(app, tmp_path, monkeypatch):
    monkeypatch.setattr('frontend.job_manager.subprocess.Popen', Mock())
    manager = JobManager(store_path=tmp_path / 'jobs.db')
    manager._feeds = {('restore', 'agent'): object(), ('new', 'agent'): object()}
    manager._finished_scanned = {'restore', 'new'}
    manager._logs = {'restore': ['old'], 'new': ['keep']}
    manager.replay_history(['restore'])
    assert ('restore', 'agent') not in manager._feeds and ('new', 'agent') in manager._feeds
    assert manager._finished_scanned == {'new'} and manager._logs == {'new': ['keep']}
    manager.shutdown()


def test_lazy_terminal_appearance_does_not_create_a_web_view(app, tmp_path, monkeypatch):
    from frontend.terminal import TerminalPanel
    from frontend.theme import current_theme
    create = Mock(side_effect=AssertionError('hidden terminal created a renderer'))
    monkeypatch.setattr('frontend.terminal.terminal_widget.TerminalWidget', create)
    panel = TerminalPanel(cwd=str(tmp_path), lazy=True)
    panel.apply_theme(current_theme())
    panel.set_font_size(14)
    panel.configure(command=['custom-shell'], cwd=str(tmp_path), env={'TEST': '1'})
    assert panel.terminals == []
    create.assert_not_called()
    # Explicit upload/SSH/new-shell actions must still create their requested
    # shell immediately even before the pane has ever been selected.
    with pytest.raises(AssertionError, match='renderer'):
        panel.new_shell(command=['ssh', 'example'])
    assert create.call_args.kwargs['command'] == ['ssh', 'example']
    panel.close()


def test_lazy_editor_can_theme_and_close_without_a_renderer(app, tmp_path, monkeypatch):
    from frontend.settings import SettingsManager
    from frontend.theme import ThemeManager
    from src.plugins.app.bundled.text_editor.panel import TextEditorPanel
    create = Mock(side_effect=AssertionError('hidden editor created a renderer'))
    monkeypatch.setattr('src.plugins.app.bundled.text_editor.panel.QWebEngineView', create)
    context = SimpleNamespace(window=SimpleNamespace(
        settings_manager=SettingsManager(tmp_path), theme_manager=ThemeManager(tmp_path / '.appearance.json')))
    editor = TextEditorPanel(context, lazy=True)
    editor.apply_theme()
    assert editor.web_view is None and editor.can_close()
    create.assert_not_called()
    editor.cleanup()
    editor.close()


def test_corrupt_snapshot_does_not_replace_live_runner_state(app, tmp_path, monkeypatch):
    monkeypatch.setattr('frontend.job_manager.subprocess.Popen', Mock())
    manager = JobManager(store_path=tmp_path / 'jobs.db')
    manager.timer.stop()
    manager._apply_snapshot(snapshot([{'job_id': 'live', 'status': 'running'}]))
    manager._readiness_request.with_suffix('.reply').write_text(json.dumps({'result': {
        'jobs': [], 'queue': {}, 'max_parallel': None}}))
    manager._poll()
    assert not manager.ready and manager.job('live')['status'] == 'running'
    assert manager.queue_state()['running'] is False
    manager.shutdown()


def test_history_merge_preserves_new_runs_and_replaces_only_duplicate_transients():
    from frontend.app import Window
    persisted = {'id': 'saved', 'created': '2025', 'simulated': False,
                 'backend': {'group': 'g', 'experiment_id': 'e', 'job_id': 'old'}}
    transient = {'id': 'disk:g/e', 'created': '2025', 'transient': True, 'simulated': False,
                 'backend': {'group': 'g', 'experiment_id': 'e'}}
    fresh = {'id': 'fresh', 'created': '2026', 'simulated': True, 'notes': 'keep these edits'}
    host = SimpleNamespace(_shutting_down=False, runs=[fresh, transient], selected=transient,
        jobs=SimpleNamespace(replay_history=Mock()), _mark_services_ready=Mock(),
        log=Mock(), refresh_runs=Mock(), reconnect_live_run=Mock(),
        monitor=SimpleNamespace(isVisible=lambda: False), plot_viewer=SimpleNamespace(isVisible=lambda: False))
    Window._history_loaded(host, [persisted, dict(fresh, notes='old notes')], [])
    assert host.runs == [fresh, persisted]
    assert host.selected is fresh and fresh['notes'] == 'keep these edits'


def test_isolated_validation_reuses_worker_and_keeps_training_imports_out_of_gui(fixture_workspace, tmp_path):
    import os
    import subprocess
    import sys
    script = r'''
import importlib.abc, sys, time
from pathlib import Path
attempted = []
class RejectTraining(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'lightning', 'matplotlib', 'stable_baselines3', 'optuna'}:
            attempted.append(fullname)
            raise AssertionError('GUI imported ' + fullname)
sys.meta_path.insert(0, RejectTraining())
from PyQt6.QtWidgets import QApplication
from frontend.local_backend import LocalBackend
app = QApplication([])
backend = LocalBackend(isolate_validation=True)
assert backend._validator._process is None
replies = []
def pump():
    deadline = time.monotonic() + 40
    while not replies and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(.01)
    assert replies, 'validation never replied'
backend.compose('theta_validation/missing_model', [], lambda data, error: replies.append((data, error)))
pump()
data, error = replies.pop()
assert error is None, error
assert data['valid'] is False and data['errors']
process = backend._validator._process
worker = process.pid
backend.compose('theta_validation/missing_model', ['seed=2'], lambda data, error: replies.append((data, error)))
pump()
assert backend._validator._process.pid == worker
assert not attempted, attempted
assert not any(name in sys.modules for name in ('torch','lightning','matplotlib'))
directory = backend._validator.directory
backend.shutdown()
assert backend.wait()
assert process.poll() is not None and not directory.exists()
'''
    result = subprocess.run([sys.executable, '-c', script], cwd=Path(__file__).resolve().parents[1],
                            env={**os.environ, 'QT_QPA_PLATFORM': 'offscreen',
                                 'THETA_WORKSPACE': str(fixture_workspace), 'THETA_DATA_DIR': str(tmp_path / 'state')},
                            capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr


def test_fresh_metric_reply_replaces_a_deferred_history_before_persistence(tmp_path):
    from frontend.app import Window
    store = Store(tmp_path)
    run = new_run(Config())
    run['metrics'] = [{'step': 1, 'reward': 3}]
    store.save(run)
    summary = store.load_summaries()[0][0]
    host = SimpleNamespace(persist=store.save, selected=None, refresh_runs=Mock())
    Window.disk_metrics_received(host, summary, {'metrics': [{'transitions': '99', 'eval/reward': '42'}]}, None)
    assert '_metrics_deferred' not in summary
    assert store.load()[0][0]['metrics'][0]['step'] == 99


def test_results_work_is_not_queued_behind_isolated_validation(app, monkeypatch):
    backend = LocalBackend(isolate_validation=True)
    blocked, release = threading.Event(), threading.Event()
    def validate(*args):
        blocked.set()
        release.wait(5)
        return {'valid': True}
    monkeypatch.setattr(backend._validator, 'call', validate)
    backend.compose('g/e', [], lambda *_: None)
    assert blocked.wait(2)
    replies = []
    backend._submit(lambda: 'catalog ready', callback=lambda data, error: replies.append(data))
    try:
        pump(app, lambda: bool(replies), timeout=2)
        assert replies == ['catalog ready'] and not release.is_set()
    finally:
        release.set()
        backend.shutdown()
        assert backend.wait()


def test_terminal_initial_shell_is_not_created_twice_during_reentrant_show(app, monkeypatch):
    from PyQt6.QtCore import pyqtSignal
    from PyQt6.QtWidgets import QWidget
    from frontend.terminal import TerminalPanel
    creations = []
    class Terminal(QWidget):
        session_started = pyqtSignal()
        session_exited = pyqtSignal(int)
        def __init__(self, *, parent=None, **kwargs):
            super().__init__(parent)
            creations.append(self)
            self.pty = SimpleNamespace(command=['test'], is_tmux_active=lambda: False)
            parent._ensure_initial_shell()  # WebEngine may process Qt events while constructing.
        def fit_terminal(self): pass
        def focus_terminal(self): pass
        def session_active(self): return True
    monkeypatch.setattr('frontend.terminal.terminal_widget.TerminalWidget', Terminal)
    panel = TerminalPanel(lazy=True)
    panel.show()
    terminal = panel.terminal
    app.processEvents()
    assert terminal is panel.terminal and len(creations) == panel.shell_count() == 1
    panel.close()


def test_closing_a_lazy_terminal_cancels_its_scheduled_initial_shell(app, monkeypatch):
    from frontend.terminal import TerminalPanel
    create = Mock(side_effect=AssertionError('closed panel created a renderer'))
    monkeypatch.setattr('frontend.terminal.terminal_widget.TerminalWidget', create)
    panel = TerminalPanel(lazy=True)
    panel.show()
    panel.close()
    app.processEvents()
    create.assert_not_called()


def test_telemetry_error_does_not_disconnect_a_healthy_runner(app, tmp_path, monkeypatch):
    monkeypatch.setattr('frontend.job_manager.subprocess.Popen', Mock())
    manager = JobManager(store_path=tmp_path / 'jobs.db')
    manager.timer.stop()
    job = {'job_id': 'live', 'status': 'running', 'started': time.time()}
    data = snapshot([job])
    manager._readiness_request.with_suffix('.reply').write_text(json.dumps({'result': data}))
    manager._poll()
    manager.timer.stop()
    (manager.ipc_dir / 'state.json').write_text(json.dumps(data))
    monkeypatch.setattr(manager, '_scan_metrics', Mock(side_effect=ValueError('bad telemetry')))
    manager._poll()
    assert manager.ready and manager.job('live')['status'] == 'running'
    manager.shutdown()


def test_startup_never_replays_finished_job_archives(app, tmp_path, monkeypatch):
    monkeypatch.setattr('frontend.job_manager.subprocess.Popen', Mock())
    manager = JobManager(store_path=tmp_path / 'jobs.db', defer_telemetry=True)
    manager.timer.stop()
    jobs = [{'job_id': str(i), 'status': 'completed', 'started': 1} for i in range(128)]
    data = snapshot(jobs)
    manager._readiness_request.with_suffix('.reply').write_text(json.dumps({'result': data}))
    manager._poll()
    manager.timer.stop()
    (manager.ipc_dir / 'state.json').write_text(json.dumps(data))
    scan = Mock(side_effect=AssertionError('archive replay on GUI thread'))
    monkeypatch.setattr(manager, '_scan_metrics', scan)
    manager._poll()
    manager.start_streaming()
    manager.replay_history([str(i) for i in range(128)])
    manager._poll()
    scan.assert_not_called()
    assert manager.ready
    manager.shutdown()


def test_final_live_tail_drains_across_bounded_batches(app, tmp_path, monkeypatch):
    monkeypatch.setattr('frontend.job_manager.subprocess.Popen', Mock())
    manager = JobManager(store_path=tmp_path / 'jobs.db')
    manager.timer.stop()
    manager._apply_snapshot(snapshot([{'job_id': 'live', 'status': 'running', 'started': 1}]))
    manager.ready = True
    data = snapshot([{'job_id': 'live', 'status': 'completed', 'started': 1}])
    (manager.ipc_dir / 'state.json').write_text(json.dumps(data))
    calls = []
    def scan(job_id):
        calls.append(job_id)
        manager._telemetry_pending[job_id] = len(calls) == 1
    monkeypatch.setattr(manager, '_scan_metrics', scan)
    monkeypatch.setattr(manager, '_sample_hardware', lambda *_: None)
    manager._poll()
    assert 'live' not in manager._finished_scanned
    manager._poll()
    assert 'live' in manager._finished_scanned
    manager._poll()
    assert calls == ['live', 'live']
    manager.shutdown()


def test_bounded_stream_and_csv_reads_keep_partial_records_and_all_final_points(tmp_path):
    from src.app.services.metrics import StreamTail, MetricsTail
    from src.app.telemetry.records import encode, record
    path = tmp_path / 'events.jsonl'
    lines = [encode(record('metrics', step=i, data={'loss': i})) + '\n' for i in range(10)]
    path.write_text(''.join(lines[:9]) + lines[9][:10])
    tail = StreamTail(path)
    seen = []
    while True:
        batch, _ = tail.read(max_records=3, max_bytes=100000)
        assert len(batch) <= 3
        seen.extend(r['step'] for r in batch)
        if not tail.pending:
            break
    assert seen == list(range(9))
    with path.open('a') as stream:
        stream.write(lines[9][10:])
    assert [r['step'] for r in tail.read(max_records=1)[0]] == [9]
    assert not tail.pending
    csv = tmp_path / 'metrics.csv'
    csv.write_text('step,loss\n' + ''.join(f'{i},{i}\n' for i in range(10)))
    tail = MetricsTail(csv)
    seen = []
    while True:
        batch, _ = tail.read(max_records=3)
        assert len(batch) <= 3
        seen.extend(r['step'] for r in batch)
        if not tail.pending:
            break
    assert seen == list(range(10))
